"""Event-level evaluator with hand-computed answers (MEMBER_2 M2-06)."""
from __future__ import annotations

import pytest

from lifeguard.contracts.enums import GroundTruthType, RiskLevel, RiskReason
from lifeguard.contracts.types import GroundTruthEvent, RiskEvent
from lifeguard.system.eval.events import EvalConfig, evaluate_events

T0 = "2026-01-01T00:00:00.000Z"


def _alarm(
    event_id: str,
    t: float | None,
    track: int = 1,
    xy: tuple[float, float] | None = None,
    to_level: RiskLevel = RiskLevel.ALARM,
) -> RiskEvent:
    return RiskEvent(
        event_id=event_id,
        cam_id="c",
        session_id="s",
        track_uid=f"c:s:{track}",
        from_level=RiskLevel.PRE_ALERT if to_level == RiskLevel.ALARM else RiskLevel.NORMAL,
        to_level=to_level,
        risk=1.0,
        reason=RiskReason.RISK_ABOVE_THRESHOLD,
        pool_xy=xy,
        t_capture=0.0 if t is None else t,
        t_utc=T0,
        t_media=t,
    )


def _truth(
    event_id: str,
    onset: float,
    track: int | None = None,
    xy: tuple[float, float] | None = None,
) -> GroundTruthEvent:
    return GroundTruthEvent(
        event_id=event_id,
        session_id="s",
        type=GroundTruthType.SUBMERSION,
        t_onset=onset,
        track_uid=None if track is None else f"c:s:{track}",
        pool_x=None if xy is None else xy[0],
        pool_y=None if xy is None else xy[1],
    )


def test_hand_computed_example_from_m2_06() -> None:
    """3 events, 2 matched, 1 false alarm in 0.5 h -> FA/h = 2.0."""
    truths = [_truth("e1", 100.0, track=1), _truth("e2", 500.0, track=2), _truth("e3", 900.0, track=3)]
    alarms = [
        _alarm("a1", 110.0, track=1),   # e1, latency 10
        _alarm("a2", 520.0, track=2),   # e2, latency 20
        _alarm("a3", 1500.0, track=4),  # nothing -> false alarm
    ]
    report = evaluate_events(alarms, truths, monitored_s=1800.0)
    assert report.n_events == 3
    assert report.detected == ("e1", "e2")
    assert report.missed == ("e3",)
    assert report.false_alarms == 1
    assert report.fa_per_hour == pytest.approx(2.0)
    assert report.recall == pytest.approx(2 / 3)
    assert report.latencies_s == (10.0, 20.0)
    assert report.median_latency_s == pytest.approx(15.0)
    assert report.max_latency_s == pytest.approx(20.0)
    assert report.line().startswith("events=3 detected=2 missed=1 alarms=3 false_alarms=1 fa_per_hour=2.000")


def test_alarm_outside_window_is_missed_and_false() -> None:
    report = evaluate_events([_alarm("a1", 150.0)], [_truth("e1", 100.0, track=1)], 3600.0, EvalConfig(l_max_s=30))
    assert report.missed == ("e1",) and report.false_alarms == 1


def test_alarm_before_onset_does_not_count() -> None:
    report = evaluate_events([_alarm("a1", 99.0)], [_truth("e1", 100.0, track=1)], 3600.0)
    assert report.missed == ("e1",) and report.false_alarms == 1


def test_wrong_person_is_not_a_detection() -> None:
    report = evaluate_events([_alarm("a1", 105.0, track=2)], [_truth("e1", 100.0, track=1)], 3600.0)
    assert report.missed == ("e1",) and report.false_alarms == 1


def test_location_match_when_track_ids_differ() -> None:
    truth = _truth("e1", 100.0, track=1, xy=(10.0, 5.0))
    near = evaluate_events([_alarm("a1", 105.0, track=9, xy=(11.0, 5.5))], [truth], 3600.0)
    far = evaluate_events([_alarm("a1", 105.0, track=9, xy=(15.0, 5.0))], [truth], 3600.0)
    assert near.detected == ("e1",) and near.false_alarms == 0
    assert far.missed == ("e1",) and far.false_alarms == 1


def test_event_without_person_matches_any_alarm_in_window() -> None:
    report = evaluate_events([_alarm("a1", 105.0, track=7)], [_truth("e1", 100.0)], 3600.0)
    assert report.detected == ("e1",)


def test_duplicate_alarms_for_one_event_are_not_false_alarms() -> None:
    alarms = [_alarm("a1", 105.0), _alarm("a2", 115.0)]
    report = evaluate_events(alarms, [_truth("e1", 100.0, track=1)], 3600.0)
    assert report.false_alarms == 0 and report.latencies_s == (5.0,)


def test_only_configured_levels_count_as_alarms() -> None:
    pre = _alarm("p1", 105.0, to_level=RiskLevel.PRE_ALERT)
    truths = [_truth("e1", 100.0, track=1)]
    assert evaluate_events([pre], truths, 3600.0).missed == ("e1",)
    lenient = EvalConfig(alarm_levels=frozenset({RiskLevel.PRE_ALERT, RiskLevel.ALARM}))
    assert evaluate_events([pre], truths, 3600.0, lenient).detected == ("e1",)


def test_no_events_no_alarms() -> None:
    report = evaluate_events([], [], 600.0)
    assert report.recall is None and report.fa_per_hour == 0.0
    assert "recall=na" in report.line()


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        evaluate_events([], [], 0.0)
    with pytest.raises(ValueError):
        evaluate_events([_alarm("a1", None)], [], 60.0)
    with pytest.raises(ValueError):
        EvalConfig(l_max_s=0.0)
