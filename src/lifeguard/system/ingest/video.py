"""Video file ingest in deterministic replay mode (roadmap P4-1, Blueprint M1 / MEMBER_2 M2-03).

REAL implementation for recorded video. Timing in replay mode:
- ``frame_idx`` is the index of the decoded frame in the file (identical on repeated runs);
- ``t_media = frame_idx / fps`` (seconds into the file, aligns with ground-truth labels, C-04);
- ``t_capture = start_t_capture + t_media`` (monotonic-like, evenly spaced, so time-based logic sees
  the recording's own clock rather than the speed of this machine);
- ``t_utc = start_utc + t_media``.

Live cameras / RTSP with a latest-frame policy and reconnect are a separate component (not built yet).
OpenCV is imported lazily; install ``opencv-python`` (``vision`` extra) to use this module.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING

import numpy as np

from lifeguard.contracts.errors import LifeguardError
from lifeguard.contracts.timeutil import add_seconds_utc, utc_now

if TYPE_CHECKING:
    # typing.Self needs Python 3.11; the project supports 3.10, so it is imported for type checkers only
    # (annotations are not evaluated at runtime because of ``from __future__ import annotations``).
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class VideoSourceError(LifeguardError):
    """The video cannot be opened or has no usable frame rate."""


@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame with its timestamps."""

    image: np.ndarray
    frame_idx: int
    t_capture: float
    t_utc: str
    t_media: float | None  # None for live cameras (no media clock)


def _import_cv2() -> ModuleType:
    """Import OpenCV lazily (optional dependency)."""
    try:
        import cv2
    except ImportError as err:
        raise VideoSourceError(
            'OpenCV is not installed; install it with: pip install -e ".[vision]"'
        ) from err
    return cv2


class VideoFileSource:
    """Iterate the frames of a video file (BGR uint8), optionally every ``stride``-th frame."""

    def __init__(
        self,
        path: Path | str,
        *,
        fps: float | None = None,
        stride: int = 1,
        max_frames: int | None = None,
        start_t_capture: float | None = None,
        start_utc: str | None = None,
    ) -> None:
        """Open the file.

        Args:
            path: Video file.
            fps: Override the container frame rate (required if the file reports none).
            stride: Yield every ``stride``-th decoded frame (frame_idx keeps the decoded index).
            max_frames: Stop after this many yielded frames.
            start_t_capture: ``t_capture`` of frame 0 (default: ``time.monotonic()`` at open).
            start_utc: ``t_utc`` of frame 0 (default: now).

        Raises:
            VideoSourceError: Missing file, unreadable video, no frame rate, or bad arguments.
        """
        if stride < 1:
            raise VideoSourceError("stride must be >= 1")
        if max_frames is not None and max_frames < 1:
            raise VideoSourceError("max_frames must be >= 1")
        self._path = Path(path)
        if not self._path.is_file():
            raise VideoSourceError(f"video not found: {self._path}")
        cv2 = _import_cv2()
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise VideoSourceError(f"cannot open video: {self._path}")
        container_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        chosen = fps if fps is not None else container_fps
        if not (math.isfinite(chosen) and chosen > 0):
            self._cap.release()
            raise VideoSourceError(f"video has no usable frame rate, pass fps explicitly: {self._path}")
        self.fps = float(chosen)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._stride = stride
        self._max_frames = max_frames
        self._t0 = time.monotonic() if start_t_capture is None else start_t_capture
        self._utc0 = utc_now() if start_utc is None else start_utc
        logger.info("opened %s: %dx%d @ %.3f fps", self._path.name, self.width, self.height, self.fps)

    def __iter__(self) -> Iterator[VideoFrame]:
        decoded = 0
        yielded = 0
        while self._max_frames is None or yielded < self._max_frames:
            ok, image = self._cap.read()
            if not ok or image is None:
                break
            idx = decoded
            decoded += 1
            if idx % self._stride:
                continue
            if image.ndim == 2:
                image = self._cv2.cvtColor(image, self._cv2.COLOR_GRAY2BGR)
            t_media = round(idx / self.fps, 6)
            yield VideoFrame(
                image=np.ascontiguousarray(image, dtype=np.uint8),
                frame_idx=idx,
                t_capture=self._t0 + t_media,
                t_utc=add_seconds_utc(self._utc0, t_media),
                t_media=t_media,
            )
            yielded += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
