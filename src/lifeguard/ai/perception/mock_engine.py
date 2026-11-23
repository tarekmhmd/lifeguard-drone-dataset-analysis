"""Mock PerceptionEngine that replays recorded FrameResult fixtures."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Literal

import numpy as np

from lifeguard.contracts.enums import Status
from lifeguard.contracts.errors import ContractValidationError, FixtureError
from lifeguard.contracts.interfaces import PerceptionEngine
from lifeguard.contracts.io import read_jsonl
from lifeguard.contracts.timeutil import UTC_PATTERN, utc_now
from lifeguard.contracts.types import (
    EngineHeartbeat,
    FrameResult,
    make_track_uid,
    parse_track_uid,
)

logger = logging.getLogger(__name__)

_IDENT_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def _is_valid_utc(value: object) -> bool:
    """Return True if value matches the UTC timestamp pattern."""
    return isinstance(value, str) and re.fullmatch(UTC_PATTERN, value) is not None


def _valid_t_media(value: object) -> bool:
    """Return True if value is a non-negative number or None."""
    if value is None:
        return True
    return isinstance(value, (int, float)) and value >= 0


class MockPerceptionEngine(PerceptionEngine):
    """Replay a JSONL fixture of FrameResult objects through the PerceptionEngine interface."""

    def __init__(
        self,
        fixture_path: Path | str,
        cam_id: str,
        session_id: str,
        on_exhausted: Literal["empty", "loop", "error"] = "empty",
    ) -> None:
        """Load and validate the fixture.

        Args:
            fixture_path: JSONL file of FrameResult lines.
            cam_id: Camera identifier stamped onto every replayed frame.
            session_id: Session identifier stamped onto every replayed frame.
            on_exhausted: Behaviour once the fixture is fully consumed.

        Raises:
            ValueError: If cam_id or session_id is not a valid identifier.
            FixtureError: If the fixture is missing, empty, invalid, or mixed-size.
        """
        if not _IDENT_PATTERN.fullmatch(cam_id):
            raise ValueError(f"invalid cam_id: {cam_id!r}")
        if not _IDENT_PATTERN.fullmatch(session_id):
            raise ValueError(f"invalid session_id: {session_id!r}")

        path = Path(fixture_path)
        try:
            frames = read_jsonl(path, FrameResult)
        except FileNotFoundError as err:
            raise FixtureError(f"fixture not found: {path}") from err
        except ContractValidationError as err:
            raise FixtureError(f"invalid fixture {path}: {err}") from err

        if not frames:
            raise FixtureError(f"fixture is empty: {path}")

        width, height = frames[0].frame_width, frames[0].frame_height
        if any(f.frame_width != width or f.frame_height != height for f in frames):
            raise FixtureError(f"fixture has mixed frame sizes: {path}")

        self._path = path
        self._frames = frames
        self._cam_id = cam_id
        self._session_id = session_id
        self._on_exhausted = on_exhausted
        self._k = 0
        self._exhausted_warned = False
        self._last_frame_idx: int | None = None
        self._last_latency_ms: float | None = None

    @property
    def cam_id(self) -> str:
        """Camera identifier."""
        return self._cam_id

    @property
    def session_id(self) -> str:
        """Session identifier."""
        return self._session_id

    @property
    def expected_frame_size(self) -> tuple[int, int]:
        """(width, height) of the fixture frames."""
        return (self._frames[0].frame_width, self._frames[0].frame_height)

    @property
    def fixture_length(self) -> int:
        """Number of frames in the fixture."""
        return len(self._frames)

    @property
    def exhausted(self) -> bool:
        """True once the fixture has been consumed (never for loop mode)."""
        return self._on_exhausted != "loop" and self._k >= self.fixture_length

    def reset(self) -> None:
        """Restart replay from the first fixture line."""
        self._k = 0
        self._exhausted_warned = False

    def heartbeat(self) -> EngineHeartbeat:
        """Report engine health and counters."""
        detail = f"mock: {self._path.name}, {self.fixture_length} frames, exhausted={self.exhausted}"
        return EngineHeartbeat(
            alive=True,
            frames_processed=self._k,
            last_frame_idx=self._last_frame_idx,
            last_latency_ms=self._last_latency_ms,
            fps=None,
            gpu_mem_mb=None,
            detail=detail,
        )

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None = None,
    ) -> FrameResult:
        """Process one frame; never raises."""
        start = time.perf_counter()
        try:
            error = self._input_error(frame_bgr, frame_idx, t_capture, t_utc)
            if error is not None:
                return self._error_result(error, start, frame_idx, t_capture, t_utc, t_media)
            result = self._next_result(frame_idx, t_capture, t_utc, t_media, start)
            self._k += 1
            self._last_frame_idx = frame_idx
            self._last_latency_ms = result.latency_ms
            return result
        except Exception as err:
            logger.exception("MockPerceptionEngine.process failed")
            return self._error_result(
                f"{type(err).__name__}: {err}",
                start,
                frame_idx,
                t_capture,
                t_utc,
                t_media,
            )

    def _input_error(
        self,
        frame_bgr: object,
        frame_idx: object,
        t_capture: object,
        t_utc: object,
    ) -> str | None:
        """Return an error message if any input is invalid, otherwise None."""
        if not isinstance(frame_bgr, np.ndarray):
            return "frame_bgr must be a numpy array"
        if frame_bgr.ndim != 3:
            return "frame_bgr must be a 3-dimensional array"
        if frame_bgr.shape[2] != 3:
            return "frame_bgr must have 3 channels"
        if frame_bgr.dtype != np.uint8:
            return "frame_bgr dtype must be uint8"
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.expected_frame_size:
            return "frame_bgr size does not match the expected frame size"
        if not isinstance(frame_idx, int) or isinstance(frame_idx, bool) or frame_idx < 0:
            return "frame_idx must be a non-negative integer"
        if not isinstance(t_capture, (int, float)) or t_capture < 0:
            return "t_capture must be >= 0"
        if not _is_valid_utc(t_utc):
            return "t_utc must match UTC_PATTERN"
        return None

    def _next_result(
        self,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None,
        start: float,
    ) -> FrameResult:
        """Select and build the next result based on the replay position."""
        if self._k < self.fixture_length:
            return self._restamp(self._frames[self._k], frame_idx, t_capture, t_utc, t_media, start)
        if self._on_exhausted == "loop":
            template = self._frames[self._k % self.fixture_length]
            return self._restamp(template, frame_idx, t_capture, t_utc, t_media, start)
        if self._on_exhausted == "error":
            return self._error_result("fixture exhausted", start, frame_idx, t_capture, t_utc, t_media)
        if not self._exhausted_warned:
            logger.warning("mock fixture exhausted; returning empty frames")
            self._exhausted_warned = True
        return self._build_empty(frame_idx, t_capture, t_utc, t_media, start)

    def _restamp(
        self,
        template: FrameResult,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None,
        start: float,
    ) -> FrameResult:
        """Re-stamp a fixture line with the engine identifiers and frame metadata."""
        tracks = tuple(
            t.model_copy(
                update={"track_uid": make_track_uid(self._cam_id, self._session_id, t.track_id)}
            )
            for t in template.tracks
        )
        submersions = tuple(
            s.model_copy(
                update={
                    "track_uid": make_track_uid(
                        self._cam_id, self._session_id, parse_track_uid(s.track_uid)[2]
                    ),
                    "since_t_capture": t_capture - s.seconds_submerged,
                }
            )
            for s in template.submersions
        )
        return FrameResult(
            cam_id=self._cam_id,
            session_id=self._session_id,
            frame_idx=frame_idx,
            t_capture=t_capture,
            t_utc=t_utc,
            t_media=t_media if _valid_t_media(t_media) else None,
            frame_width=self.expected_frame_size[0],
            frame_height=self.expected_frame_size[1],
            status=template.status,
            error=template.error,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tracks=tracks,
            submersions=submersions,
            model_versions={"engine": "mock", "fixture": self._path.name},
        )

    def _build_empty(
        self,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None,
        start: float,
    ) -> FrameResult:
        """Build an ok result with no tracks once the fixture is exhausted."""
        return FrameResult(
            cam_id=self._cam_id,
            session_id=self._session_id,
            frame_idx=frame_idx,
            t_capture=t_capture,
            t_utc=t_utc,
            t_media=t_media if _valid_t_media(t_media) else None,
            frame_width=self.expected_frame_size[0],
            frame_height=self.expected_frame_size[1],
            status=Status.OK,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tracks=(),
            submersions=(),
            model_versions={"engine": "mock", "fixture": self._path.name},
        )

    def _error_result(
        self,
        message: str,
        start: float,
        frame_idx: object,
        t_capture: object,
        t_utc: object,
        t_media: object,
    ) -> FrameResult:
        """Build an error result with sanitised inputs."""
        if not isinstance(frame_idx, int) or isinstance(frame_idx, bool) or frame_idx < 0:
            frame_idx = 0
        if not isinstance(t_capture, (int, float)) or t_capture < 0:
            t_capture = 0.0
        if not _is_valid_utc(t_utc):
            t_utc = utc_now()
        return FrameResult(
            cam_id=self._cam_id,
            session_id=self._session_id,
            frame_idx=frame_idx,
            t_capture=t_capture,
            t_utc=t_utc,
            t_media=t_media if _valid_t_media(t_media) else None,
            frame_width=self.expected_frame_size[0],
            frame_height=self.expected_frame_size[1],
            status=Status.ERROR,
            error=message,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tracks=(),
            submersions=(),
            model_versions={"engine": "mock", "fixture": self._path.name},
        )
