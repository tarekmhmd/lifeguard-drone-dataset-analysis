"""Labelled training samples for the behaviour classifier (roadmap P5-1).

A *sample* is the feature window of one in-water track at one moment (``TrackObs.features`` from the real
engine), sub-sampled every ``stride_s`` seconds per track (Blueprint M6: a feature vector every 0.5 s).

Labels come from ground-truth events (contract ``GroundTruthEvent``):
- the person an event refers to is resolved at its onset: the track with the event's ``track_uid``, or
  else the in-water track nearest to the labelled position within ``radius_m``; that identity is then
  followed through ``merged_from`` re-associations;
- while the event is active (``t_onset <= t_media <= t_end``; open-ended if ``t_end`` is None) that
  person's samples get the event label: DISTRESS -> ``distress``, DROWNING -> ``drowning``; SUBMERSION
  events are excluded (the person is not visible, and the hard rule in the risk engine handles them);
- if an active event cannot be resolved to a track, every sample in its interval is dropped rather than
  risk a wrong negative label;
- all other in-water samples get ``default_label`` (a catch-all "not in distress" class, because the
  event labels do not distinguish swim/idle/play/dive; a per-behaviour annotation format is pending M1-04).

Training on synthetic scenarios exercises the code path only; it is never evidence of real accuracy.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from lifeguard.ai.features.spec import load_feature_spec
from lifeguard.contracts.enums import GroundTruthType, Status
from lifeguard.contracts.types import FrameResult, GroundTruthEvent, parse_track_uid

BEHAVIOR_CLASSES: tuple[str, ...] = ("normal_swim", "idle", "play", "dive", "distress", "drowning")
# Feature names and order come from the versioned feature spec (M1-03), the single source of truth.
FEATURE_SPEC_VERSION: str = load_feature_spec().spec_version
FEATURE_NAMES: tuple[str, ...] = load_feature_spec().names
VECTOR_SIZE = 2 * len(FEATURE_NAMES)

_EVENT_LABELS: dict[GroundTruthType, str | None] = {
    GroundTruthType.DISTRESS: "distress",
    GroundTruthType.DROWNING: "drowning",
    GroundTruthType.SUBMERSION: None,
}
_EPS = 1e-9


def vectorize(features: Mapping[str, float]) -> np.ndarray:
    """Fixed-order vector: the value of each feature (0 if absent) followed by a presence flag."""
    values = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
    present = [1.0 if name in features else 0.0 for name in FEATURE_NAMES]
    return np.array(values + present, dtype=float)


@dataclass(frozen=True)
class Sample:
    """One labelled feature window."""

    session_id: str
    track_uid: str
    t_media: float
    x: np.ndarray
    label: str


@dataclass
class _ActiveEvent:
    truth: GroundTruthEvent
    label: str | None
    target_uid: str | None = None
    resolved: bool = False

    def active(self, t: float) -> bool:
        end = math.inf if self.truth.t_end is None else self.truth.t_end
        return self.truth.t_onset <= t <= end


def _resolve(event: _ActiveEvent, frame: FrameResult, radius_m: float) -> None:
    """Pick the track the event refers to, at the first frame at or after its onset."""
    event.resolved = True
    truth = event.truth
    uids = {t.track_uid for t in frame.tracks}
    if truth.track_uid is not None and truth.track_uid in uids:
        event.target_uid = truth.track_uid
        return
    if truth.pool_x is None or truth.pool_y is None:
        return
    best: tuple[float, str] | None = None
    for track in frame.tracks:
        if not track.in_water or track.pool_xy is None:
            continue
        d = math.dist(track.pool_xy, (truth.pool_x, truth.pool_y))
        if d <= radius_m and (best is None or d < best[0]):
            best = (d, track.track_uid)
    if best is not None:
        event.target_uid = best[1]


def build_samples(
    frames: Iterable[FrameResult],
    truths: Iterable[GroundTruthEvent],
    *,
    default_label: str = "normal_swim",
    stride_s: float = 0.5,
    radius_m: float = 2.0,
) -> list[Sample]:
    """Labelled samples from one session's frames and its ground-truth events."""
    if default_label not in BEHAVIOR_CLASSES:
        raise ValueError(f"unknown label: {default_label!r}")
    ordered = sorted(
        (f for f in frames if f.status != Status.ERROR and f.t_media is not None),
        key=lambda f: f.t_media,  # type: ignore[arg-type, return-value]
    )
    events = [_ActiveEvent(t, _EVENT_LABELS[t.type]) for t in truths]
    last_emit: dict[str, float] = {}
    samples: list[Sample] = []

    for frame in ordered:
        t = float(frame.t_media)  # type: ignore[arg-type]
        for event in events:
            if not event.resolved and t >= event.truth.t_onset:
                _resolve(event, frame, radius_m)
            if event.target_uid is not None:
                old_id = parse_track_uid(event.target_uid)[2]
                for track in frame.tracks:
                    if old_id in track.merged_from and event.target_uid.startswith(
                        f"{frame.cam_id}:{frame.session_id}:"
                    ):
                        event.target_uid = track.track_uid
        active = [e for e in events if e.active(t)]
        if any(e.target_uid is None and e.label is not None for e in active):
            continue  # an unresolved distress/drowning event is in progress: do not risk wrong labels

        for track in frame.tracks:
            if not track.in_water or not track.features:
                continue
            if t - last_emit.get(track.track_uid, -math.inf) < stride_s - _EPS:
                continue
            owner = next((e for e in active if e.target_uid == track.track_uid), None)
            if owner is not None and owner.label is None:
                continue  # submersion: excluded from behaviour training
            label = owner.label if owner is not None else default_label
            last_emit[track.track_uid] = t
            samples.append(Sample(frame.session_id, track.track_uid, t, vectorize(track.features), label))
    return samples
