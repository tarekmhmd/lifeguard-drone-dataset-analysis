"""Deterministic synthetic FrameResult scenarios with ground truth.

These scenarios test logic (risk engine, evaluator, replay), never model accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lifeguard.contracts.enums import DetClass, TrackState
from lifeguard.contracts.interfaces import PoolGeometry
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import (
    FrameResult,
    GroundTruthEvent,
    GroundTruthType,
    Submersion,
    TrackObs,
    make_track_uid,
)
from lifeguard.system.geometry.stub import StubPoolGeometry

SCENARIO_KINDS: tuple[str, ...] = (
    "normal_swim",
    "idle",
    "dive",
    "edge_exit",
    "submersion",
    "distress_like",
)

_LOST_EMIT_S = 2.0
_BOX_W_PX = 40.0
_BOX_H_PX = 60.0
_DET_CONF_LO = 0.55
_DET_CONF_HI = 0.9
_BASE_T_CAPTURE = 1000.0
_TIMESTAMP_ORIGIN = "2026-01-01T00:00:00.000Z"
_INVISIBLE_PROB = 0.3
_MAX_ATTEMPTS = 100

_DURATIONS: dict[str, float] = {
    "normal_swim": 60.0,
    "idle": 60.0,
    "dive": 40.0,
    "edge_exit": 40.0,
    "submersion": 60.0,
    "distress_like": 90.0,
}


@dataclass(frozen=True)
class Scenario:
    """One synthetic scenario: frames plus ground-truth events."""

    kind: str
    seed: int
    fps: float
    session_id: str
    frames: tuple[FrameResult, ...]
    events: tuple[GroundTruthEvent, ...]


@dataclass(frozen=True)
class _SubjectFrame:
    """Deterministic subject state for a single frame (before det_conf)."""

    track_id: int
    x: float
    y: float
    state: TrackState
    merged_from: tuple[int, ...] = ()
    bob_y_px: float = 0.0


def _swim_state(
    t: float,
    v: float,
    direction: int,
    y0: float,
    length_m: float,
    start_x: float = 2.0,
) -> tuple[float, float, int]:
    """Return the bouncing-swim (x, y, direction) at time t."""
    span = length_m - 3.0
    x = start_x + direction * v * t
    period = 2.0 * span
    r = (x - 1.5) % period
    if r <= span:
        xr = 1.5 + r
        dir_out = direction
    else:
        xr = 1.5 + (period - r)
        dir_out = -direction
    y = y0 + 0.15 * math.sin(2.0 * math.pi * 0.5 * t)
    return xr, y, dir_out


def _draw_subject_params(
    rng: np.random.Generator,
    kind: str,
    length_m: float,
    width_m: float,
) -> dict[str, float | int]:
    """Draw the subject parameters for a kind, in the fixed RNG order."""
    if kind == "edge_exit":
        return {}
    if kind == "idle":
        return {
            "x0": float(rng.uniform(3.0, length_m / 2.0 - 2.0)),
            "y0": float(rng.uniform(3.0, width_m - 3.0)),
        }
    params: dict[str, float | int] = {
        "y0": float(rng.uniform(2.0, width_m - 2.0)),
        "v": float(rng.uniform(0.5, 1.2)),
        "dir": int(rng.choice([1, -1])),
    }
    if kind == "dive":
        params["d"] = float(rng.uniform(3.0, 6.0))
        params["r"] = float(rng.uniform(0.5, 1.5))
    elif kind == "distress_like":
        params["f"] = float(rng.uniform(0.5, 1.0))
        params["phi"] = float(rng.uniform(0.0, 2.0 * math.pi))
        params["s_d"] = float(rng.uniform(0.02, 0.08))
    return params


def _draw_subject_params_constrained(
    rng: np.random.Generator,
    kind: str,
    length_m: float,
    width_m: float,
    geometry: PoolGeometry,
) -> dict[str, float | int]:
    """Draw subject parameters, re-drawing the whole list until the interior constraint holds."""
    check_time = {"dive": 15.0, "submersion": 20.0, "distress_like": 20.0}.get(kind)
    for _ in range(_MAX_ATTEMPTS):
        params = _draw_subject_params(rng, kind, length_m, width_m)
        if check_time is None:
            return params
        x, y, _ = _swim_state(check_time, params["v"], params["dir"], params["y0"], length_m)
        if geometry.is_interior(x, y, 2.0):
            return params
    raise ValueError("cannot place subject in pool interior")


def _draw_background_params(
    rng: np.random.Generator,
    n_background: int,
    width_m: float,
) -> list[dict[str, float | int]]:
    """Draw background swimmer parameters in increasing track-id order."""
    backgrounds: list[dict[str, float | int]] = []
    for j in range(n_background):
        backgrounds.append(
            {
                "v": float(rng.uniform(0.5, 1.2)),
                "dir": int(rng.choice([1, -1])),
                "y": width_m * (j + 1) / (n_background + 2),
            }
        )
    return backgrounds


def _make_box(
    x: float,
    y: float,
    geometry: PoolGeometry,
    bob_y_px: float = 0.0,
) -> tuple[float, float, float, float]:
    """Build a 40x60 px box centred on the pool point, clipped to the frame."""
    u, v = geometry.to_image(x, y)
    v += bob_y_px
    x1 = max(0.0, u - _BOX_W_PX / 2.0)
    x2 = min(float(geometry.image_width), u + _BOX_W_PX / 2.0)
    y1 = max(0.0, v - _BOX_H_PX / 2.0)
    y2 = min(float(geometry.image_height), v + _BOX_H_PX / 2.0)
    return (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))


def _make_track(
    track_id: int,
    cam_id: str,
    session_id: str,
    x: float,
    y: float,
    det_conf: float,
    state: TrackState,
    geometry: PoolGeometry,
    merged_from: tuple[int, ...] = (),
    bob_y_px: float = 0.0,
) -> TrackObs:
    """Build a TrackObs for a person at a pool position."""
    return TrackObs(
        track_id=track_id,
        track_uid=make_track_uid(cam_id, session_id, track_id),
        cls=DetClass.PERSON,
        box_xyxy=_make_box(x, y, geometry, bob_y_px),
        det_conf=round(det_conf, 3),
        state=state,
        pool_xy=(round(x, 3), round(y, 3)),
        in_water=True,
        zone=geometry.zone(x, y),
        merged_from=merged_from,
    )


def _make_submersion(
    track_uid: str,
    x: float,
    y: float,
    seconds: float,
    since: float,
) -> Submersion:
    """Build a Submersion timer entry."""
    return Submersion(
        track_uid=track_uid,
        last_pool_xy=(round(x, 3), round(y, 3)),
        seconds_submerged=round(seconds, 3),
        since_t_capture=round(since, 3),
    )


def make_scenario(
    kind: str,
    seed: int = 0,
    fps: float = 10.0,
    geometry: PoolGeometry | None = None,
    n_background: int = 2,
    cam_id: str = "cam_sim",
) -> Scenario:
    """Build a deterministic synthetic scenario.

    Args:
        kind: One of :data:`SCENARIO_KINDS`.
        seed: RNG seed.
        fps: Frames per second.
        geometry: Pool geometry; defaults to :class:`StubPoolGeometry`.
        n_background: Number of background swimmers.
        cam_id: Camera identifier used in every track uid.

    Returns:
        The generated scenario.

    Raises:
        ValueError: For an unknown kind, ``fps <= 0`` or ``n_background < 0``.
    """
    if kind not in SCENARIO_KINDS:
        raise ValueError(f"unknown scenario kind: {kind!r}")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if n_background < 0:
        raise ValueError("n_background must be >= 0")
    geometry = geometry or StubPoolGeometry()
    length_m = geometry.length_m
    width_m = geometry.width_m

    rng = np.random.default_rng(seed)
    params = _draw_subject_params_constrained(rng, kind, length_m, width_m, geometry)
    backgrounds = _draw_background_params(rng, n_background, width_m)

    n_frames = round(_DURATIONS[kind] * fps)
    session_id = f"{kind}_seed{seed}"
    subject_uid = make_track_uid(cam_id, session_id, 1)

    # Distress_like drift state (the last valid drift position).
    drift_x = drift_y = 0.0
    if kind == "distress_like":
        drift_x, drift_y, _ = _swim_state(
            20.0, params["v"], params["dir"], params["y0"], length_m
        )

    max_gap_frames = round(1.0 * fps)
    gap_run = 0
    last_visible_t = -1.0
    last_visible_x = last_visible_y = 0.0

    frames: list[FrameResult] = []
    for i in range(n_frames):
        t = i / fps
        t_capture = round(_BASE_T_CAPTURE + t, 3)
        t_media = round(t, 3)
        t_utc = add_seconds_utc(_TIMESTAMP_ORIGIN, t)

        subject: _SubjectFrame | None = None
        submersion: Submersion | None = None

        if kind == "normal_swim":
            x, y, _ = _swim_state(t, params["v"], params["dir"], params["y0"], length_m)
            subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
        elif kind == "idle":
            jitter = np.clip(rng.normal(0.0, 0.05, 2), -0.1, 0.1)
            x = float(params["x0"] + jitter[0])
            y = float(params["y0"] + jitter[1])
            subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
        elif kind == "dive":
            d = float(params["d"])
            t_d = 15.0
            t_last = t_d - 1.0 / fps
            if t < t_d:
                x, y, _ = _swim_state(t, params["v"], params["dir"], params["y0"], length_m)
                subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
            else:
                x_last, y_last, _ = _swim_state(
                    t_last, params["v"], params["dir"], params["y0"], length_m
                )
                if t < t_d + d:
                    submersion = _make_submersion(
                        subject_uid, x_last, y_last, t - t_last, _BASE_T_CAPTURE + t_last
                    )
                if t < t_d + _LOST_EMIT_S:
                    subject = _SubjectFrame(1, x_last, y_last, TrackState.LOST)
                elif t >= t_d + d:
                    x15, _, dir15 = _swim_state(
                        t_last, params["v"], params["dir"], params["y0"], length_m
                    )
                    x_start = x15 + dir15 * float(params["r"])
                    x, y, _ = _swim_state(
                        t - (t_d + d), params["v"], dir15, params["y0"], length_m, x_start
                    )
                    subject = _SubjectFrame(
                        2 + n_background, x, y, TrackState.TRACKED, merged_from=(1,)
                    )
        elif kind == "edge_exit":
            if t <= 20.0:
                frac = t / 20.0
                x = length_m / 2.0 * (1.0 - frac)
                y = width_m / 2.0 * (1.0 - frac) + 2.0 * frac
                subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
        elif kind == "submersion":
            t_d = 20.0
            t_last = t_d - 1.0 / fps
            if t < t_d:
                x, y, _ = _swim_state(t, params["v"], params["dir"], params["y0"], length_m)
                subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
            else:
                x_last, y_last, _ = _swim_state(
                    t_last, params["v"], params["dir"], params["y0"], length_m
                )
                submersion = _make_submersion(
                    subject_uid, x_last, y_last, t - t_last, _BASE_T_CAPTURE + t_last
                )
                if t < t_d + _LOST_EMIT_S:
                    subject = _SubjectFrame(1, x_last, y_last, TrackState.LOST)
        elif kind == "distress_like":
            if t < 20.0:
                x, y, _ = _swim_state(t, params["v"], params["dir"], params["y0"], length_m)
                subject = _SubjectFrame(1, x, y, TrackState.TRACKED)
                last_visible_t = t
                last_visible_x, last_visible_y = x, y
            elif t < 60.0:
                s_d = float(params["s_d"])
                phi = float(params["phi"])
                candidate_x = drift_x + s_d * (t - 20.0) * math.cos(phi)
                candidate_y = drift_y + s_d * (t - 20.0) * math.sin(phi)
                if geometry.is_interior(candidate_x, candidate_y, 1.0):
                    drift_x, drift_y = candidate_x, candidate_y

                invisible = bool(rng.random() < _INVISIBLE_PROB)
                if invisible and gap_run + 1 > max_gap_frames:
                    invisible = False
                if invisible:
                    gap_run += 1
                    submersion = _make_submersion(
                        subject_uid,
                        last_visible_x,
                        last_visible_y,
                        t - last_visible_t,
                        _BASE_T_CAPTURE + last_visible_t,
                    )
                    subject = _SubjectFrame(1, last_visible_x, last_visible_y, TrackState.LOST)
                else:
                    gap_run = 0
                    bob = 0.25 * 60.0 * math.sin(2.0 * math.pi * float(params["f"]) * t)
                    subject = _SubjectFrame(
                        1, drift_x, drift_y, TrackState.TRACKED, bob_y_px=bob
                    )
                    last_visible_t = t
                    last_visible_x, last_visible_y = drift_x, drift_y
            else:
                t_last = 60.0 - 1.0 / fps
                submersion = _make_submersion(
                    subject_uid, drift_x, drift_y, t - t_last, _BASE_T_CAPTURE + t_last
                )
                if t < 60.0 + _LOST_EMIT_S:
                    subject = _SubjectFrame(1, drift_x, drift_y, TrackState.LOST)

        tracks: list[TrackObs] = []
        if subject is not None:
            tracks.append(
                _make_track(
                    subject.track_id,
                    cam_id,
                    session_id,
                    subject.x,
                    subject.y,
                    float(rng.uniform(_DET_CONF_LO, _DET_CONF_HI)),
                    subject.state,
                    geometry,
                    merged_from=subject.merged_from,
                    bob_y_px=subject.bob_y_px,
                )
            )
        for j, bg in enumerate(backgrounds):
            x, y, _ = _swim_state(t, bg["v"], bg["dir"], bg["y"], length_m)
            tracks.append(
                _make_track(
                    2 + j,
                    cam_id,
                    session_id,
                    x,
                    y,
                    float(rng.uniform(_DET_CONF_LO, _DET_CONF_HI)),
                    TrackState.TRACKED,
                    geometry,
                )
            )

        frames.append(
            FrameResult(
                cam_id=cam_id,
                session_id=session_id,
                frame_idx=i,
                t_capture=t_capture,
                t_utc=t_utc,
                t_media=t_media,
                frame_width=geometry.image_width,
                frame_height=geometry.image_height,
                latency_ms=0.0,
                tracks=tuple(tracks),
                submersions=(submersion,) if submersion is not None else (),
                model_versions={"generator": "synthetic_v1"},
            )
        )

    events: tuple[GroundTruthEvent, ...] = ()
    if kind == "submersion":
        x_last, y_last, _ = _swim_state(
            20.0 - 1.0 / fps, params["v"], params["dir"], params["y0"], length_m
        )
        events = (
            GroundTruthEvent(
                event_id=f"{session_id}_evt1",
                session_id=session_id,
                video_file="",
                type=GroundTruthType.SUBMERSION,
                t_onset=20.0,
                t_end=None,
                track_uid=subject_uid,
                pool_x=round(x_last, 3),
                pool_y=round(y_last, 3),
            ),
        )
    elif kind == "distress_like":
        p20x, p20y, _ = _swim_state(
            20.0, params["v"], params["dir"], params["y0"], length_m
        )
        events = (
            GroundTruthEvent(
                event_id=f"{session_id}_evt1",
                session_id=session_id,
                video_file="",
                type=GroundTruthType.DISTRESS,
                t_onset=20.0,
                t_end=90.0,
                track_uid=subject_uid,
                pool_x=round(p20x, 3),
                pool_y=round(p20y, 3),
            ),
        )

    return Scenario(
        kind=kind,
        seed=seed,
        fps=fps,
        session_id=session_id,
        frames=tuple(frames),
        events=events,
    )
