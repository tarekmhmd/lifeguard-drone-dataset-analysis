"""RuleBasedRiskEngine: synthetic fixture acceptance (M2-05) and incident lifecycle unit tests."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from lifeguard.contracts.enums import AckDecision, DetClass, RiskLevel, RiskReason, Status, TrackState, Zone
from lifeguard.contracts.interfaces import RiskEngine
from lifeguard.contracts.io import read_events_csv, read_jsonl
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import (
    Acknowledgement,
    FrameResult,
    RiskEvent,
    Submersion,
    TrackObs,
    make_track_uid,
)
from lifeguard.system.config import load_system_config
from lifeguard.system.risk.config import RiskConfig, RiskWeights
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"
CONFIG = load_system_config(ROOT / "configs" / "system.yaml").risk

CAM, SESS = "cam_t", "s1"
T0 = "2026-01-01T00:00:00.000Z"
DT = 0.1


# --------------------------------------------------------------------------- helpers


def _uid(track_id: int) -> str:
    return make_track_uid(CAM, SESS, track_id)


def _track(
    track_id: int,
    xy: tuple[float, float] | None = (5.0, 5.0),
    state: TrackState = TrackState.TRACKED,
    in_water: bool = True,
    zone: Zone | None = Zone.SHALLOW,
    merged_from: tuple[int, ...] = (),
) -> TrackObs:
    return TrackObs(
        track_id=track_id,
        track_uid=_uid(track_id),
        merged_from=merged_from,
        cls=DetClass.PERSON,
        box_xyxy=(10.0, 10.0, 50.0, 70.0),
        det_conf=0.8,
        state=state,
        pool_xy=xy,
        in_water=in_water,
        zone=zone,
    )


def _frame(
    t: float,
    tracks: Iterable[TrackObs] = (),
    subs: Iterable[Submersion] = (),
    status: Status = Status.OK,
) -> FrameResult:
    return FrameResult(
        cam_id=CAM,
        session_id=SESS,
        frame_idx=round(t / DT),
        t_capture=t,
        t_utc=add_seconds_utc(T0, t),
        t_media=t,
        frame_width=1920,
        frame_height=1080,
        status=status,
        error="camera unavailable" if status == Status.ERROR else None,
        latency_ms=0.0,
        tracks=tuple(tracks),
        submersions=tuple(subs),
    )


def _sub(track_id: int, t: float, start: float) -> Submersion:
    return Submersion(
        track_uid=_uid(track_id),
        last_pool_xy=(5.0, 5.0),
        seconds_submerged=round(t - start, 3),
        since_t_capture=start,
    )


def _times(start: float, stop: float) -> list[float]:
    """Frame times from start to stop inclusive at 10 fps."""
    return [round(start + i * DT, 3) for i in range(round((stop - start) / DT) + 1)]


def _submerge_until_alarm(engine: RuleBasedRiskEngine, track_id: int = 1) -> tuple[RiskEvent, float]:
    """Track ``track_id`` is under water from t = 0; return its ALARM event and the last frame time."""
    events: list[RiskEvent] = []
    t_last = 0.0
    for t in _times(0.0, CONFIG.t_sub_s + 1.0):
        events += engine.update(_frame(t, subs=[_sub(track_id, t, 0.0)]))
        t_last = t
    alarms = [e for e in events if e.to_level == RiskLevel.ALARM]
    assert len(alarms) == 1
    return alarms[0], t_last


def _ack(event: RiskEvent, decision: AckDecision, t: float) -> Acknowledgement:
    return Acknowledgement(
        event_id=event.event_id,
        track_uid=event.track_uid,
        decision=decision,
        by="lifeguard_1",
        t_utc=add_seconds_utc(T0, t),
    )


def _replay_fixture(kind: str) -> list[RiskEvent]:
    engine = RuleBasedRiskEngine(CONFIG)
    events: list[RiskEvent] = []
    for frame in read_jsonl(FIXTURES / f"{kind}_seed0.jsonl", FrameResult):
        events += engine.update(frame)
    return events


# ------------------------------------------------------- synthetic fixture acceptance


def test_engine_implements_contract() -> None:
    assert isinstance(RuleBasedRiskEngine(), RiskEngine)


def test_submersion_fixture_reaches_alarm_by_hard_rule() -> None:
    """The synthetic submersion event reaches ALARM within t_sub (+0.5 s) of its onset."""
    (truth,) = read_events_csv(FIXTURES / "submersion_seed0_events.csv")
    alarms = [e for e in _replay_fixture("submersion") if e.to_level == RiskLevel.ALARM]
    subject = [e for e in alarms if e.track_uid == truth.track_uid]
    assert subject, "submersion subject never reached ALARM"
    first = subject[0]
    assert first.reason == RiskReason.SUBMERSION_TIMEOUT
    assert first.risk == 1.0
    assert truth.t_onset + CONFIG.t_sub_s <= first.t_media <= truth.t_onset + CONFIG.t_sub_s + 0.5


def test_distress_like_fixture_reaches_alarm() -> None:
    """The synthetic distress event reaches ALARM inside its labelled interval."""
    (truth,) = read_events_csv(FIXTURES / "distress_like_seed0_events.csv")
    alarms = [
        e for e in _replay_fixture("distress_like")
        if e.to_level == RiskLevel.ALARM and e.track_uid == truth.track_uid
    ]
    assert alarms
    assert truth.t_onset <= alarms[0].t_media <= truth.t_end


@pytest.mark.parametrize("kind", ["normal_swim", "idle", "dive", "edge_exit"])
def test_negative_fixtures_never_alarm(kind: str) -> None:
    """Hard negatives (including a 3-6 s dive) produce no ALARM for any track."""
    events = _replay_fixture(kind)
    assert [e for e in events if e.to_level in (RiskLevel.ALARM, RiskLevel.DISPATCH)] == []


@pytest.mark.parametrize("kind", ["submersion", "distress_like", "dive"])
def test_replay_is_deterministic_and_events_round_trip(kind: str) -> None:
    """Two replays give identical events, and every event survives JSON round-tripping."""
    first = _replay_fixture(kind)
    assert first == _replay_fixture(kind)
    for event in first:
        assert RiskEvent.model_validate_json(event.model_dump_json()) == event


# ----------------------------------------------------------- lifecycle unit tests


def test_error_frames_are_ignored() -> None:
    engine = RuleBasedRiskEngine(CONFIG)
    assert engine.update(_frame(0.0, status=Status.ERROR)) == []
    assert engine.levels() == {}


def test_confirm_on_alarm_dispatches_once() -> None:
    """Only a human CONFIRM moves ALARM to DISPATCH; a second confirm has no effect."""
    engine = RuleBasedRiskEngine(CONFIG)
    alarm, t = _submerge_until_alarm(engine)
    assert alarm.reason == RiskReason.SUBMERSION_TIMEOUT

    (dispatch,) = engine.acknowledge(_ack(alarm, AckDecision.CONFIRM, t))
    assert (dispatch.from_level, dispatch.to_level) == (RiskLevel.ALARM, RiskLevel.DISPATCH)
    assert dispatch.reason == RiskReason.LIFEGUARD_CONFIRMED
    assert engine.acknowledge(_ack(alarm, AckDecision.CONFIRM, t)) == []


def test_close_incident_resolves_dispatch() -> None:
    engine = RuleBasedRiskEngine(CONFIG)
    alarm, t = _submerge_until_alarm(engine)
    engine.acknowledge(_ack(alarm, AckDecision.CONFIRM, t))
    (closed,) = engine.close_incident(_uid(1), add_seconds_utc(T0, t))
    assert (closed.to_level, closed.reason) == (RiskLevel.RESOLVED, RiskReason.INCIDENT_CLOSED)
    assert engine.close_incident(_uid(1), add_seconds_utc(T0, t)) == []


def test_dismiss_then_cooldown_returns_to_normal() -> None:
    """DISMISS -> RESOLVED; a resurfaced swimmer returns to NORMAL after the cooldown."""
    engine = RuleBasedRiskEngine(CONFIG)
    alarm, t_ack = _submerge_until_alarm(engine)
    (resolved,) = engine.acknowledge(_ack(alarm, AckDecision.DISMISS, t_ack))
    assert (resolved.to_level, resolved.reason) == (RiskLevel.RESOLVED, RiskReason.LIFEGUARD_DISMISSED)

    events: list[RiskEvent] = []
    for t in _times(t_ack + DT, t_ack + CONFIG.cooldown_s + 1.0):
        x = 5.0 + 0.5 * (t - t_ack)  # swimming away at 0.5 m/s
        events += engine.update(_frame(t, tracks=[_track(1, xy=(x, 5.0))]))
    assert [(e.to_level, e.reason) for e in events] == [(RiskLevel.NORMAL, RiskReason.COOLDOWN_ELAPSED)]
    assert events[0].t_capture >= t_ack + CONFIG.cooldown_s - 1e-9


def test_dismissed_but_still_submerged_realarms() -> None:
    """The hard rule overrides the cooldown: a dismissed swimmer still under water re-alarms."""
    engine = RuleBasedRiskEngine(CONFIG)
    alarm, t = _submerge_until_alarm(engine)
    engine.acknowledge(_ack(alarm, AckDecision.DISMISS, t))
    t_next = round(t + DT, 3)
    (event,) = engine.update(_frame(t_next, subs=[_sub(1, t_next, 0.0)]))
    assert (event.from_level, event.to_level) == (RiskLevel.RESOLVED, RiskLevel.ALARM)
    assert event.reason == RiskReason.SUBMERSION_TIMEOUT


def test_alarm_is_never_cleared_when_the_track_vanishes() -> None:
    """An ALARM whose track disappears stays ALARM until a human decides."""
    engine = RuleBasedRiskEngine(CONFIG)
    _, t = _submerge_until_alarm(engine)
    events: list[RiskEvent] = []
    for t_empty in _times(t + DT, t + 60.0):
        events += engine.update(_frame(t_empty))
    assert events == []
    assert engine.levels()[_uid(1)][0] == RiskLevel.ALARM


def test_watch_track_that_exits_returns_to_normal() -> None:
    """A WATCH track that is no longer observed is closed with TRACK_EXITED after the timeout."""
    cfg = RiskConfig(
        weights=RiskWeights(submersion=0.0, static=1.0, visibility=0.0, deep_zone=0.0, behavior=0.0)
    )
    engine = RuleBasedRiskEngine(cfg)
    t_last = 0.0
    for t in _times(0.0, 10.0):  # a motionless person in the water
        engine.update(_frame(t, tracks=[_track(1)]))
        t_last = t
        if engine.levels()[_uid(1)][0] == RiskLevel.WATCH:
            break
    assert engine.levels()[_uid(1)][0] == RiskLevel.WATCH

    events: list[RiskEvent] = []
    for t in _times(t_last + DT, t_last + cfg.track_timeout_s + 1.0):
        events += engine.update(_frame(t))
    assert [(e.to_level, e.reason) for e in events] == [(RiskLevel.NORMAL, RiskReason.TRACK_EXITED)]
    assert _uid(1) not in engine.levels()


def test_merged_track_inherits_incident() -> None:
    """Risk history and the open incident follow merged_from; old event ids still acknowledge."""
    engine = RuleBasedRiskEngine(CONFIG)
    alarm, t = _submerge_until_alarm(engine, track_id=1)
    t_next = round(t + DT, 3)
    engine.update(_frame(t_next, tracks=[_track(5, merged_from=(1,))]))

    levels = engine.levels()
    assert _uid(1) not in levels
    assert levels[_uid(5)][0] == RiskLevel.ALARM

    (dispatch,) = engine.acknowledge(_ack(alarm, AckDecision.CONFIRM, t_next))
    assert dispatch.track_uid == _uid(5)
    assert dispatch.to_level == RiskLevel.DISPATCH


def test_unknown_acknowledgement_is_ignored() -> None:
    engine = RuleBasedRiskEngine(CONFIG)
    ack = Acknowledgement(
        event_id="risk-unknown-000001",
        track_uid=_uid(1),
        decision=AckDecision.CONFIRM,
        by="lifeguard_1",
        t_utc=T0,
    )
    assert engine.acknowledge(ack) == []


def test_person_on_deck_never_raises_risk() -> None:
    """Static evidence and zones count only in the water."""
    engine = RuleBasedRiskEngine(CONFIG)
    events: list[RiskEvent] = []
    for t in _times(0.0, 60.0):
        events += engine.update(_frame(t, tracks=[_track(1, xy=None, in_water=False, zone=None)]))
    assert events == []


def test_reset_clears_state() -> None:
    engine = RuleBasedRiskEngine(CONFIG)
    _submerge_until_alarm(engine)
    engine.reset()
    assert engine.levels() == {}
