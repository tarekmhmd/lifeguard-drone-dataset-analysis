"""System health watchdog (Blueprint M13, roadmap P3-4): make failures loud.

REAL implementation. Inputs are pushed by the runtime (frames, perception results); ``check()`` returns one
contract ``SystemHealth`` per monitored component. A lifeguard system that silently stops is worse than
none, so the alert service raises a fault with the same priority as a drowning alarm.

| Component | ERROR when | DEGRADED when |
| --- | --- | --- |
| INGEST | no frame for ``camera_timeout_s`` (camera unplugged, stream lost, video ended unexpectedly), or the same image for ``frozen_after_s`` | - |
| PERCEPTION | ``max_consecutive_engine_errors`` (N, system.yaml) error results in a row | 1..N-1 errors in a row, FPS below ``fps_floor``, or a degraded result |
| STORAGE | free space below ``min_free_disk_mb`` or the path is unusable | no storage path configured |

Time is the caller's ``t`` (monotonic seconds), so behaviour is deterministic in tests and replays.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lifeguard.contracts.enums import HealthComponent, Status
from lifeguard.contracts.types import FrameResult, SystemHealth
from lifeguard.system.health.config import HealthConfig

logger = logging.getLogger(__name__)

_SAMPLE_STEP = 8          # frame digest uses every 8th pixel in both directions
_BYTES_PER_MB = 1024 * 1024


def frame_digest(image: np.ndarray) -> str:
    """Cheap fingerprint of a frame (sub-sampled pixels), used for frozen-frame detection."""
    sample = np.ascontiguousarray(image[::_SAMPLE_STEP, ::_SAMPLE_STEP])
    return hashlib.blake2b(sample.tobytes(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class _State:
    status: Status
    detail: str


class Watchdog:
    """Tracks ingest, perception and storage health."""

    def __init__(
        self,
        config: HealthConfig | None = None,
        storage_path: Path | None = None,
        max_consecutive_errors: int = 1,
    ) -> None:
        """``max_consecutive_errors`` is ``max_consecutive_engine_errors`` from ``system.yaml`` (integration
        checklist: SYSTEM_FAULT after N consecutive errors); fewer errors report DEGRADED."""
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        self._config = config or HealthConfig()
        self._storage_path = storage_path
        self._max_consecutive_errors = max_consecutive_errors
        self._consecutive_errors = 0
        self._started_t: float | None = None
        self._last_frame_t: float | None = None
        self._last_digest: str | None = None
        self._same_since_t: float | None = None
        self._frame_times: deque[float] = deque()
        self._last_result_status: Status | None = None
        self._last_error: str | None = None
        self._previous: dict[HealthComponent, Status] = {}

    def start(self, t: float) -> None:
        """Mark the moment monitoring starts (the camera timeout counts from here)."""
        self._started_t = t

    def on_frame(self, t: float, image: np.ndarray | None) -> None:
        """Record that a frame arrived at ``t`` (pass ``image=None`` to skip the frozen check)."""
        if self._started_t is None:
            self._started_t = t
        self._last_frame_t = t
        self._frame_times.append(t)
        while self._frame_times and self._frame_times[0] < t - self._config.fps_window_s:
            self._frame_times.popleft()
        if image is None:
            return
        digest = frame_digest(image)
        if digest != self._last_digest:
            self._last_digest = digest
            self._same_since_t = t
        elif self._same_since_t is None:
            self._same_since_t = t

    def on_result(self, result: FrameResult) -> None:
        """Record the latest perception outcome and count consecutive error frames."""
        self._last_result_status = result.status
        if result.status == Status.ERROR:
            self._consecutive_errors += 1
            self._last_error = result.error
        else:
            self._consecutive_errors = 0
            if result.status == Status.DEGRADED:
                self._last_error = result.error

    def check(self, t: float, t_utc: str) -> list[SystemHealth]:
        """Current health of every monitored component."""
        states = {
            HealthComponent.INGEST: self._ingest(t),
            HealthComponent.PERCEPTION: self._perception(t),
            HealthComponent.STORAGE: self._storage(),
        }
        out: list[SystemHealth] = []
        for component, state in states.items():
            if self._previous.get(component) != state.status:
                log = logger.error if state.status == Status.ERROR else logger.info
                log("health %s -> %s: %s", component.value, state.status.value, state.detail)
                self._previous[component] = state.status
            out.append(SystemHealth(component=component, status=state.status, detail=state.detail,
                                    t_utc=t_utc, t_capture=t))
        return out

    # -------------------------------------------------------------- checks

    def _ingest(self, t: float) -> _State:
        cfg = self._config
        reference = self._last_frame_t if self._last_frame_t is not None else self._started_t
        if reference is None:
            return _State(Status.ERROR, "no video source started")
        if t - reference > cfg.camera_timeout_s:
            return _State(Status.ERROR, f"no frame for {t - reference:.1f} s (camera or stream unavailable)")
        if self._same_since_t is not None and self._last_frame_t is not None:
            frozen_for = self._last_frame_t - self._same_since_t
            if frozen_for >= cfg.frozen_after_s:
                return _State(Status.ERROR, f"frozen video: identical frames for {frozen_for:.1f} s")
        return _State(Status.OK, "frames arriving")

    def _perception(self, t: float) -> _State:
        cfg = self._config
        n = self._consecutive_errors
        if n >= self._max_consecutive_errors:
            return _State(Status.ERROR, f"{n} consecutive perception errors (limit {self._max_consecutive_errors}): "
                                        f"{self._last_error}")
        if n > 0:
            return _State(Status.DEGRADED, f"{n} consecutive perception error(s) "
                                           f"(fault at {self._max_consecutive_errors}): {self._last_error}")
        started = self._started_t if self._started_t is not None else t
        if t - started >= cfg.fps_window_s and self._frame_times:
            fps = len(self._frame_times) / cfg.fps_window_s
            if fps < cfg.fps_floor:
                return _State(Status.DEGRADED, f"processing at {fps:.1f} fps, below the {cfg.fps_floor:.1f} floor")
        if self._last_result_status == Status.DEGRADED:
            return _State(Status.DEGRADED, f"perception degraded: {self._last_error}")
        return _State(Status.OK, "perception running")

    def _storage(self) -> _State:
        if self._storage_path is None:
            return _State(Status.DEGRADED, "no storage path configured (clips and incident file disabled)")
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            free_mb = shutil.disk_usage(self._storage_path).free / _BYTES_PER_MB
        except OSError as err:
            return _State(Status.ERROR, f"storage unusable: {err}")
        if free_mb < self._config.min_free_disk_mb:
            return _State(Status.ERROR, f"only {free_mb:.0f} MB free (< {self._config.min_free_disk_mb:.0f} MB)")
        return _State(Status.OK, f"{free_mb:.0f} MB free")
