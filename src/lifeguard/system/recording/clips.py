"""Alarm clip recorder (Blueprint M9: "10 s before + 5 s after an ALARM", roadmap P3-3).

REAL implementation. Keeps a ring buffer of recent frames (down-scaled to ``max_width`` to bound memory)
and, when an incident is triggered, writes ``pre_s`` seconds before plus ``post_s`` seconds after to an
MP4 under ``out_dir``. Failures never stop the lifeguard pipeline:
- OpenCV missing or the writer cannot open -> the clip is skipped with an ERROR log (``failed`` list);
- no ``out_dir`` configured -> recording is disabled (``enabled`` is False);
- frames of a different size than the clip's first frame are resized to it.
Memory: about ``(pre_s + post_s) * fps * max_width^2 * 0.56 * 3`` bytes (~92 MB for 15 s, 10 fps, 640 px).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

from lifeguard.system.recording.config import RecordingConfig

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\-]")


def _cv2() -> ModuleType | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


@dataclass
class _Pending:
    incident_id: str
    path: Path
    end_t: float
    frames: list[tuple[float, np.ndarray]] = field(default_factory=list)


class ClipRecorder:
    """Ring buffer + per-incident clip writer."""

    def __init__(self, config: RecordingConfig | None = None, fps_hint: float = 10.0) -> None:
        self._config = config or RecordingConfig()
        self._fps_hint = fps_hint
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._pending: list[_Pending] = []
        self.completed: list[Path] = []
        self.failed: list[str] = []
        self._cv2 = _cv2()
        if self.enabled and self._cv2 is None:
            logger.error("clip recording disabled: OpenCV is not installed")

    @property
    def enabled(self) -> bool:
        return self._config.out_dir is not None

    def on_frame(self, t: float, image: np.ndarray) -> list[Path]:
        """Buffer a frame; return clips finished by this frame."""
        if not self.enabled or self._cv2 is None:
            return []
        small = self._shrink(image)
        self._buffer.append((t, small))
        while self._buffer and self._buffer[0][0] < t - self._config.pre_s:
            self._buffer.popleft()
        done: list[Path] = []
        for pending in list(self._pending):
            pending.frames.append((t, small))
            if t >= pending.end_t:
                self._pending.remove(pending)
                path = self._write(pending)
                if path is not None:
                    done.append(path)
        return done

    def trigger(self, incident_id: str, t: float) -> Path | None:
        """Start a clip for an incident; returns its planned path (None if recording is unavailable)."""
        if not self.enabled or self._cv2 is None:
            return None
        if any(p.incident_id == incident_id for p in self._pending):
            return next(p.path for p in self._pending if p.incident_id == incident_id)
        name = _SAFE_NAME.sub("_", incident_id)
        path = Path(self._config.out_dir) / f"{name}.mp4"  # type: ignore[arg-type]
        self._pending.append(_Pending(incident_id, path, t + self._config.post_s, list(self._buffer)))
        logger.info("recording clip for %s -> %s", incident_id, path)
        return path

    def close(self) -> list[Path]:
        """Write every unfinished clip with the frames collected so far (shutdown)."""
        done = [p for p in (self._write(pending) for pending in self._pending) if p is not None]
        self._pending.clear()
        return done

    # ---------------------------------------------------------------- helpers

    def _shrink(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if width <= self._config.max_width:
            return image.copy()
        scale = self._config.max_width / width
        size = (self._config.max_width, max(2, round(height * scale)))
        return self._cv2.resize(image, size, interpolation=self._cv2.INTER_AREA)  # type: ignore[union-attr]

    def _write(self, pending: _Pending) -> Path | None:
        cv2 = self._cv2
        if cv2 is None or not pending.frames:
            self.failed.append(pending.incident_id)
            return None
        times = [t for t, _ in pending.frames]
        span = times[-1] - times[0]
        fps = (len(times) - 1) / span if span > 0 else self._fps_hint
        height, width = pending.frames[0][1].shape[:2]
        try:
            pending.path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(pending.path), cv2.VideoWriter_fourcc(*self._config.codec),
                                     max(1.0, fps), (width, height))
        except (OSError, cv2.error) as err:
            logger.error("cannot write clip %s: %s", pending.path, err)
            self.failed.append(pending.incident_id)
            return None
        if not writer.isOpened():
            logger.error("cannot open video writer for %s (codec %s)", pending.path, self._config.codec)
            self.failed.append(pending.incident_id)
            return None
        try:
            for _, frame in pending.frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        self.completed.append(pending.path)
        logger.info("clip written: %s (%d frames)", pending.path, len(pending.frames))
        return pending.path
