"""Operator dashboard back end (roadmap P3-5): runs the ``LifeguardRuntime`` on a worker thread and turns
snapshots into display rows. No Streamlit import here, so it is testable headless.

Every operator control maps to a real runtime call:
- Confirm / Dismiss -> ``LifeguardRuntime.acknowledge`` -> risk engine + alert service (+ response platform);
- Confirm release   -> ``LifeguardRuntime.confirm_release`` -> simulated drone executor;
- Hardware stop / e-stop / test commands -> ``HardwareService`` -> command manager (all checks apply).
"""
from __future__ import annotations

import logging
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lifeguard.ai.config import load_ai_config
from lifeguard.app.hardware_hub import HardwareHub
from lifeguard.app.hardware_view import (  # noqa: F401 - hardware_badges/rows re-exported for existing callers
    HardwareConsole,
    hardware_badges,
    hardware_rows,
)
from lifeguard.app.main import open_source
from lifeguard.app.runtime import LifeguardRuntime, Snapshot
from lifeguard.common.models import ModelIntegrityError, integrity_fault
from lifeguard.contracts.enums import AckDecision, RiskLevel, TrackState
from lifeguard.contracts.errors import LifeguardError
from lifeguard.contracts.types import FrameResult
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.video import VideoFrame

logger = logging.getLogger(__name__)

# BGR colours per risk level for the overlay.
LEVEL_COLOURS: dict[RiskLevel, tuple[int, int, int]] = {
    RiskLevel.NORMAL: (80, 200, 80),
    RiskLevel.WATCH: (0, 220, 255),
    RiskLevel.PRE_ALERT: (0, 140, 255),
    RiskLevel.ALARM: (0, 0, 255),
    RiskLevel.DISPATCH: (255, 0, 255),
    RiskLevel.RESOLVED: (200, 200, 200),
}


# Pose overlay (COCO-17). Joints below POSE_DRAW_CONF are not drawn, and an edge is drawn only when both of its
# joints pass, so a missing or uncertain joint never appears as a fabricated limb. Colours (BGR) are per body
# part and kept apart from the risk colours, which stay on the boxes and labels.
POSE_DRAW_CONF = 0.5
POSE_COLOURS: dict[str, tuple[int, int, int]] = {
    "head": (255, 255, 255),   # white: nose, eyes, ears
    "torso": (200, 200, 0),    # teal: shoulders-hips frame
    "left": (255, 120, 0),     # blue: swimmer's left arm and leg
    "right": (0, 200, 255),    # amber: swimmer's right arm and leg
}
_KPT_PART = ["head"] * 5 + ["left", "right"] * 6
POSE_EDGES: tuple[tuple[int, int, str], ...] = (
    (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"),
    (5, 6, "torso"), (5, 11, "torso"), (6, 12, "torso"), (11, 12, "torso"),
    (5, 7, "left"), (7, 9, "left"), (11, 13, "left"), (13, 15, "left"),
    (6, 8, "right"), (8, 10, "right"), (12, 14, "right"), (14, 16, "right"),
)


def draw_pose(image: np.ndarray, keypoints: tuple[tuple[float, float, float], ...], sx: float, sy: float,
              min_conf: float = POSE_DRAW_CONF) -> int:
    """Draw one skeleton (frame-pixel keypoints scaled by sx, sy); returns the number of joints drawn."""
    import cv2

    thick = max(1, round(min(image.shape[:2]) / 360))
    pts = [(round(x * sx), round(y * sy)) if c >= min_conf else None for x, y, c in keypoints]
    for a, b, part in POSE_EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(image, pts[a], pts[b], POSE_COLOURS[part], thick, cv2.LINE_AA)
    drawn = 0
    for i, p in enumerate(pts):
        if p is None:
            continue
        conf = keypoints[i][2]
        radius = thick + 1 + int(2 * (conf - min_conf) / max(1e-6, 1.0 - min_conf))  # larger = more confident
        cv2.circle(image, p, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)               # dark rim: readable on water
        cv2.circle(image, p, radius, POSE_COLOURS[_KPT_PART[i]], -1, cv2.LINE_AA)
        drawn += 1
    return drawn


def pose_summary(keypoints: tuple[tuple[float, float, float], ...] | None, min_conf: float = POSE_DRAW_CONF) -> str:
    """'k/17' joints at or above ``min_conf``; '-' when the track has no pose."""
    if keypoints is None:
        return "-"
    return f"{sum(1 for _, _, c in keypoints if c >= min_conf)}/17"


@dataclass(frozen=True)
class SourceChoice:
    """What the operator selected in the sidebar."""

    kind: str                     # "synthetic" | "video" | "camera"
    value: str = ""               # video path or camera index / URL
    realtime: bool = True


def clean_input(text: str) -> str:
    """Normalise a typed or pasted path/source: drop invisible format characters and surrounding quotes.

    Text copied from a right-to-left document (e.g. an Arabic PDF) often carries bidi marks such as
    U+200E/U+200F/U+202A-U+202E, which make an existing file look missing.
    """
    visible = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return visible.strip().strip('"').strip("'").strip()


def annotate(frame: np.ndarray | None, result: FrameResult | None,
             levels: dict[str, tuple[RiskLevel, float]], show_pose: bool = True,
             pose_min_conf: float = POSE_DRAW_CONF) -> np.ndarray | None:
    """Draw track boxes coloured by risk level, plus pose skeletons when tracks carry keypoints.

    Returns an RGB image (None without a frame).
    """
    if frame is None:
        return None
    image = frame.copy()
    try:
        import cv2
    except ImportError:
        return image[:, :, ::-1]
    if result is not None and result.tracks:
        sy = image.shape[0] / result.frame_height
        sx = image.shape[1] / result.frame_width
        for track in result.tracks:
            level, risk = levels.get(track.track_uid, (RiskLevel.NORMAL, 0.0))
            colour = LEVEL_COLOURS[level]
            x1, y1, x2, y2 = track.box_xyxy
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            thickness = 1 if track.state == TrackState.LOST else 2
            cv2.rectangle(image, p1, p2, colour, thickness)
            if show_pose and track.keypoints is not None:
                draw_pose(image, track.keypoints, sx, sy, pose_min_conf)
            # Label last so the skeleton never hides the track id.
            cv2.putText(image, f"{track.track_id} {level.value} {risk:.2f}", (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    return image[:, :, ::-1]


def track_rows(snap: Snapshot) -> list[dict[str, Any]]:
    """One row per tracked person, highest risk first."""
    rows = []
    tracks = {t.track_uid: t for t in (snap.result.tracks if snap.result else ())}
    for uid, (level, risk) in snap.levels.items():
        track = tracks.get(uid)
        rows.append({
            "track": uid.rsplit(":", 1)[-1],
            "level": level.value,
            "risk": round(risk, 3),
            "state": track.state.value if track else "not visible",
            "pool_xy": None if track is None or track.pool_xy is None else tuple(round(v, 1) for v in track.pool_xy),
            "zone": track.zone.value if track and track.zone else "",
            "pose": pose_summary(track.keypoints) if track else "-",
        })
    return sorted(rows, key=lambda r: -r["risk"])


def health_rows(snap: Snapshot) -> list[dict[str, str]]:
    return [{"component": h.component.value, "status": h.status.value, "detail": h.detail} for h in snap.health]


class DashboardController:
    """Owns one runtime and its worker thread."""

    def __init__(self, hub: HardwareHub | None = None) -> None:
        """``hub``: the process-wide shared hardware owner (two-page app). Without it the runtime owns its own
        hardware service, exactly as before (standalone dashboard)."""
        self.hub = hub
        self._hub_attached = False
        self.runtime: LifeguardRuntime | None = None
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._source: object | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, ai_config: Path, system_config: Path, choice: SourceChoice) -> bool:
        """Build the runtime and start processing; False (with ``error`` set) if it cannot start."""
        self.stop()
        self.error = None
        try:
            ai_cfg = load_ai_config(ai_config)
            sys_cfg = load_system_config(system_config)
            args = _Args(choice)
            source, size, fps = open_source(args, ai_cfg)
            shared = None
            if self.hub is not None:     # one hardware service per process, shared with the Hardware page
                shared = self.hub.acquire_for_ai(sys_cfg, str(system_config))
                self._hub_attached = shared is not None
            self.runtime = LifeguardRuntime(ai_cfg, sys_cfg, size, hardware=shared)
        except (LifeguardError, OSError, ValueError, ImportError) as err:  # shown to the operator, no crash
            self._release_hub()
            self.error = f"{type(err).__name__}: {err}"
            if isinstance(err, ModelIntegrityError):
                self.error = f"SYSTEM FAULT: {integrity_fault(err).detail}"
            logger.error("dashboard could not start: %s", self.error)
            return False
        self._source = source
        self._stop = threading.Event()
        realtime = fps if (choice.realtime and fps) else None
        self._thread = threading.Thread(
            target=self._run, args=(source, realtime), name="lifeguard-runtime", daemon=True
        )
        self._thread.start()
        return True

    def _run(self, source: Iterable[VideoFrame | None], realtime: float | None) -> None:
        assert self.runtime is not None
        try:
            self.runtime.run(source, stop=self._stop, realtime_fps=realtime)
        except Exception as err:
            self.error = f"runtime stopped: {type(err).__name__}: {err}"
            logger.exception("runtime stopped")

    def stop(self) -> None:
        self._stop.set()
        stop_source = getattr(self._source, "stop", None)
        if callable(stop_source):
            stop_source()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        try:
            if self.runtime is not None:
                self.runtime.close()     # a shared hardware service is not stopped here (the hub owns it)
        finally:
            self.runtime = None
            self._release_hub()

    def _release_hub(self) -> None:
        if self.hub is not None and self._hub_attached:
            self.hub.release_ai()
        self._hub_attached = False

    # ------------------------------------------------------- operator actions

    def acknowledge(self, incident_id: str, decision: AckDecision, by: str) -> str:
        """Confirm or dismiss an incident; returns a message for the operator."""
        if self.runtime is None:
            return "system not running"
        if not by.strip():
            return "enter the lifeguard's name first"
        event_id = self.runtime.alerts.latest_event_id(incident_id)
        if event_id is None:
            return f"unknown incident {incident_id}"
        result, outcome = self.runtime.acknowledge(event_id, decision, by.strip())
        if not result.events:
            return "no change (incident already handled)"
        parts = [f"{decision.value}: {result.events[-1].from_level.value} -> {result.events[-1].to_level.value}"]
        if outcome is not None:
            parts.append(outcome.detail)
        return "; ".join(parts)

    @property
    def hardware_console(self) -> HardwareConsole:
        """Operator hardware controls on the running runtime's hardware service (same as the Hardware Demo)."""
        return HardwareConsole(self.runtime.hardware if self.runtime is not None else None)

    def hardware_command(self, action: str, by: str, target: str | None = None,
                         params: dict[str, Any] | None = None, pin: str | None = None,
                         confirmed: bool = True) -> str:
        """Operator hardware action (estop, safe_stop, clear_estop, ping, request_status, set_motor, set_actuator,
        connect, disconnect)."""
        console = self.hardware_console
        if action == "connect":
            return console.connect(by)
        if action == "disconnect":
            return console.disconnect(by)
        return console.command(action, by, target, params, pin=pin, confirmed=confirmed)

    def confirm_release(self, by: str) -> str:
        if self.runtime is None:
            return "system not running"
        if not by.strip():
            return "enter the lifeguard's name first"
        outcome = self.runtime.confirm_release(by.strip())
        return "no release pending" if outcome is None else outcome.detail


class _Args:
    """Adapter from a SourceChoice to the CLI argument object ``open_source`` expects."""

    def __init__(self, choice: SourceChoice) -> None:
        self.video = Path(choice.value) if choice.kind == "video" else None
        self.camera = choice.value if choice.kind == "camera" else None
        self.synthetic = choice.kind == "synthetic"
        self.max_frames = None
