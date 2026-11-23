"""Frame loop shared by the integration commands: frames -> PerceptionEngine -> RiskEngine (roadmap P7).

No decision logic lives here: perception produces ``FrameResult``, the risk engine produces ``RiskEvent``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from lifeguard.contracts.enums import RiskLevel, Status
from lifeguard.contracts.errors import LifeguardError
from lifeguard.contracts.interfaces import PerceptionEngine, RiskEngine
from lifeguard.contracts.types import FrameResult, RiskEvent
from lifeguard.system.ingest.video import VideoFrame

logger = logging.getLogger(__name__)


class TooManyEngineErrors(LifeguardError):
    """The perception engine returned ``max_consecutive_engine_errors`` error frames in a row."""


@dataclass
class RunSummary:
    """Counters for one run."""

    frames: int = 0
    ok: int = 0
    degraded: int = 0
    error: int = 0
    max_submersion_s: float = 0.0
    risk_events: int = 0
    alarms: int = 0
    track_uids: set[str] = field(default_factory=set)

    def line(self) -> str:
        """The single summary line printed by the CLI."""
        return (
            f"frames={self.frames} ok={self.ok} degraded={self.degraded} error={self.error} "
            f"unique_tracks={len(self.track_uids)} max_submersion_s={self.max_submersion_s:.3f} "
            f"risk_events={self.risk_events} alarms={self.alarms}"
        )


def run_pipeline(
    frames: Iterable[VideoFrame],
    engine: PerceptionEngine,
    risk: RiskEngine,
    max_consecutive_errors: int,
    on_result: Callable[[FrameResult], None] | None = None,
    on_event: Callable[[RiskEvent], None] | None = None,
) -> RunSummary:
    """Process every frame; raise ``TooManyEngineErrors`` when the error limit is reached."""
    summary = RunSummary()
    consecutive = 0
    for frame in frames:
        result = engine.process(frame.image, frame.frame_idx, frame.t_capture, frame.t_utc, frame.t_media)
        summary.frames += 1
        if result.status == Status.ERROR:
            summary.error += 1
            consecutive += 1
            if consecutive >= max_consecutive_errors:
                raise TooManyEngineErrors(
                    f"{consecutive} consecutive engine errors at frame {frame.frame_idx}: {result.error}"
                )
        else:
            consecutive = 0
            if result.status == Status.DEGRADED:
                summary.degraded += 1
            else:
                summary.ok += 1
        summary.track_uids.update(t.track_uid for t in result.tracks)
        for sub in result.submersions:
            summary.max_submersion_s = max(summary.max_submersion_s, sub.seconds_submerged)
        if on_result is not None:
            on_result(result)

        for event in risk.update(result):
            summary.risk_events += 1
            if event.to_level == RiskLevel.ALARM:
                summary.alarms += 1
            if on_event is not None:
                on_event(event)
    return summary
