"""Operator dashboard back end (roadmap P3-5): runs the ``LifeguardRuntime`` on a worker thread and turns
snapshots into display rows. No Streamlit import here, so it is testable headless.

Every operator control maps to a real runtime call:
- Confirm / Dismiss -> ``LifeguardRuntime.acknowledge`` -> risk engine + alert service (+ response platform);
- Confirm release   -> ``LifeguardRuntime.confirm_release`` -> simulated drone executor.
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
             levels: dict[str, tuple[RiskLevel, float]]) -> np.ndarray | None:
    """Draw track boxes coloured by risk level; returns an RGB image (None without a frame)."""
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
        })
    return sorted(rows, key=lambda r: -r["risk"])


def health_rows(snap: Snapshot) -> list[dict[str, str]]:
    return [{"component": h.component.value, "status": h.status.value, "detail": h.detail} for h in snap.health]


class DashboardController:
    """Owns one runtime and its worker thread."""

    def __init__(self) -> None:
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
            self.runtime = LifeguardRuntime(ai_cfg, sys_cfg, size)
        except (LifeguardError, OSError, ValueError, ImportError) as err:  # shown to the operator, no crash
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
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

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
