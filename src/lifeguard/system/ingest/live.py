"""Live camera / stream ingest with a latest-frame policy and reconnect (Blueprint M1, MEMBER_2 M2-03).

REAL implementation; needs a camera or stream to be useful (not available in CI). A background thread
reads continuously and keeps only the newest frame: when processing is slower than the camera, stale frames
are dropped (counted in ``dropped``) instead of queueing, so latency never builds up. On a read failure the
capture is released and reopened every ``reconnect_s`` seconds.

Iterating yields ``VideoFrame`` when a new frame is ready and ``None`` after ``poll_s`` without one, so
the caller can run health checks while the camera is silent.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from lifeguard.contracts.timeutil import utc_now
from lifeguard.system.ingest.video import VideoFrame, VideoSourceError

logger = logging.getLogger(__name__)


def _cv2_factory(source: int | str) -> Any:
    try:
        import cv2
    except ImportError as err:
        raise VideoSourceError('OpenCV is not installed: pip install -e ".[vision]"') from err
    return cv2.VideoCapture(source)


def parse_source(text: str) -> int | str:
    """``"0"`` -> camera index 0; anything else (``rtsp://...``, a device path) stays a string."""
    return int(text) if text.isdigit() else text


class LiveCameraSource:
    """Threaded latest-frame reader."""

    frozen_check = True  # real cameras: frozen-frame detection is meaningful

    def __init__(
        self,
        source: int | str,
        *,
        reconnect_s: float = 2.0,
        poll_s: float = 0.5,
        capture_factory: Callable[[int | str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._reconnect_s = reconnect_s
        self._poll_s = poll_s
        self._factory = capture_factory or _cv2_factory
        self._clock = clock
        self._cond = threading.Condition()
        self._slot: tuple[float, str, np.ndarray] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.reconnects = 0
        self.connected = False

    def start(self) -> LiveCameraSource:
        if self._thread is None:
            self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _reader(self) -> None:
        cap = None
        while not self._stop.is_set():
            if cap is None:
                try:
                    cap = self._factory(self._source)
                except VideoSourceError as err:
                    logger.error("%s", err)
                    self._stop.wait(self._reconnect_s)
                    continue
                if not cap.isOpened():
                    logger.warning("camera %r not available; retrying in %.1f s", self._source, self._reconnect_s)
                    cap.release()
                    cap = None
                    self._stop.wait(self._reconnect_s)
                    continue
                self.connected = True
                logger.info("camera %r connected", self._source)
            ok, image = cap.read()
            if not ok or image is None:
                logger.warning("camera %r read failed; reconnecting", self._source)
                cap.release()
                cap = None
                self.connected = False
                self.reconnects += 1
                self._stop.wait(self._reconnect_s)
                continue
            with self._cond:
                if self._slot is not None:
                    self.dropped += 1
                self._slot = (self._clock(), utc_now(), image)
                self._cond.notify_all()
        if cap is not None:
            cap.release()
        self.connected = False

    def __iter__(self) -> Iterator[VideoFrame | None]:
        self.start()
        idx = 0
        while not self._stop.is_set():
            with self._cond:
                if self._slot is None:
                    self._cond.wait(self._poll_s)
                item, self._slot = self._slot, None
            if item is None:
                yield None
                continue
            t, t_utc, image = item
            yield VideoFrame(image=image, frame_idx=idx, t_capture=t, t_utc=t_utc, t_media=None)
            idx += 1


class SyntheticFrameSource:
    """Black frames at a fixed rate (for the mock engine and demos without a camera or weights).

    SIMULATION: images carry no information; detections come from the mock engine's fixture.
    """

    frozen_check = False  # identical black frames are expected

    def __init__(self, width: int, height: int, n_frames: int, fps: float = 10.0,
                 start_t_capture: float | None = None, start_utc: str | None = None) -> None:
        self._image = np.zeros((height, width, 3), np.uint8)
        self._n = n_frames
        self._fps = fps
        self._t0 = time.monotonic() if start_t_capture is None else start_t_capture
        self._utc0 = utc_now() if start_utc is None else start_utc
        self.fps = fps

    def __iter__(self) -> Iterator[VideoFrame]:
        from lifeguard.contracts.timeutil import add_seconds_utc

        for i in range(self._n):
            t_media = round(i / self._fps, 6)
            yield VideoFrame(self._image, i, self._t0 + t_media, add_seconds_utc(self._utc0, t_media), t_media)
