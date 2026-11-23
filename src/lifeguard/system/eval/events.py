"""Event-level evaluation of risk alarms against ground-truth events (roadmap P2-3, MEMBER_2 M2-06).

Evaluate per *event* and per *hour of monitoring*, not per frame (Blueprint 11). One call evaluates one
monitored session: the caller passes that session's alarms and labels.

Matching rule (documented, deterministic):
- an *alarm* is a ``RiskEvent`` whose ``to_level`` is in ``EvalConfig.alarm_levels`` (default: ALARM);
- an alarm *matches* a ground-truth event when its ``t_media`` lies in ``[t_onset, t_onset + l_max_s]``
  AND it concerns the right person: the same ``track_uid``, or (when both sides have positions) a
  ``pool_xy`` within ``match_radius_m`` of the labelled ``(pool_x, pool_y)``; an event with neither a
  track nor a position matches any alarm in its window;
- an event is *detected* if at least one alarm matches it; its latency is the first match minus onset;
- every alarm that matches no event is a *false alarm*; FA/h = false alarms / monitored hours.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from lifeguard.contracts.enums import RiskLevel
from lifeguard.contracts.types import GroundTruthEvent, RiskEvent

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class EvalConfig:
    """Matching parameters."""

    l_max_s: float = 30.0
    match_radius_m: float = 2.0
    alarm_levels: frozenset[RiskLevel] = frozenset({RiskLevel.ALARM})

    def __post_init__(self) -> None:
        if not (self.l_max_s > 0 and self.match_radius_m > 0 and self.alarm_levels):
            raise ValueError("l_max_s and match_radius_m must be > 0 and alarm_levels non-empty")


@dataclass(frozen=True)
class EventReport:
    """Result of one evaluation."""

    n_events: int
    detected: tuple[str, ...]
    missed: tuple[str, ...]
    n_alarms: int
    false_alarms: int
    monitored_s: float
    latencies_s: tuple[float, ...] = field(default=())

    @property
    def fa_per_hour(self) -> float:
        return self.false_alarms / (self.monitored_s / _SECONDS_PER_HOUR)

    @property
    def recall(self) -> float | None:
        return None if self.n_events == 0 else len(self.detected) / self.n_events

    @property
    def median_latency_s(self) -> float | None:
        return statistics.median(self.latencies_s) if self.latencies_s else None

    @property
    def max_latency_s(self) -> float | None:
        return max(self.latencies_s) if self.latencies_s else None

    def line(self) -> str:
        """One-line summary."""
        def fmt(value: float | None) -> str:
            return "na" if value is None else f"{value:.3f}"

        return (
            f"events={self.n_events} detected={len(self.detected)} missed={len(self.missed)} "
            f"alarms={self.n_alarms} false_alarms={self.false_alarms} "
            f"fa_per_hour={self.fa_per_hour:.3f} recall={fmt(self.recall)} "
            f"median_latency_s={fmt(self.median_latency_s)} max_latency_s={fmt(self.max_latency_s)}"
        )


def _same_person(alarm: RiskEvent, truth: GroundTruthEvent, radius_m: float) -> bool:
    if truth.track_uid is not None and alarm.track_uid == truth.track_uid:
        return True
    if truth.pool_x is not None and truth.pool_y is not None and alarm.pool_xy is not None:
        return math.dist(alarm.pool_xy, (truth.pool_x, truth.pool_y)) <= radius_m
    return truth.track_uid is None and truth.pool_x is None


def evaluate_events(
    risk_events: Iterable[RiskEvent],
    truths: Iterable[GroundTruthEvent],
    monitored_s: float,
    config: EvalConfig | None = None,
) -> EventReport:
    """Match alarms to ground-truth events for one monitored session.

    Raises:
        ValueError: If ``monitored_s <= 0`` or an alarm has no ``t_media`` (labels use media time).
    """
    cfg = config or EvalConfig()
    if not monitored_s > 0:
        raise ValueError("monitored_s must be > 0")
    alarms = sorted(
        (e for e in risk_events if e.to_level in cfg.alarm_levels),
        key=lambda e: (e.t_media if e.t_media is not None else -1.0, e.event_id),
    )
    for alarm in alarms:
        if alarm.t_media is None:
            raise ValueError(f"alarm {alarm.event_id} has no t_media; cannot align with labels")
    truth_list = sorted(truths, key=lambda g: (g.t_onset, g.event_id))

    matched_alarms: set[str] = set()
    detected: list[str] = []
    missed: list[str] = []
    latencies: list[float] = []
    for truth in truth_list:
        hits = [
            a for a in alarms
            if truth.t_onset <= a.t_media <= truth.t_onset + cfg.l_max_s  # type: ignore[operator]
            and _same_person(a, truth, cfg.match_radius_m)
        ]
        if hits:
            detected.append(truth.event_id)
            latencies.append(hits[0].t_media - truth.t_onset)  # type: ignore[operator]
            matched_alarms.update(a.event_id for a in hits)
        else:
            missed.append(truth.event_id)

    return EventReport(
        n_events=len(truth_list),
        detected=tuple(detected),
        missed=tuple(missed),
        n_alarms=len(alarms),
        false_alarms=sum(1 for a in alarms if a.event_id not in matched_alarms),
        monitored_s=monitored_s,
        latencies_s=tuple(latencies),
    )
