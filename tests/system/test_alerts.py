"""Alert service: incidents, notifications, unacknowledged-ALARM repeats, acknowledgements and dispatch."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from lifeguard.contracts.enums import AckDecision, DetClass, RiskLevel, RiskReason, TrackState, Zone
from lifeguard.contracts.interfaces import RiskEngine
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import (
    Acknowledgement,
    FrameResult,
    RiskEvent,
    Submersion,
    TrackObs,
    make_track_uid,
)
from lifeguard.system.alerts.config import AlertConfig
from lifeguard.system.alerts.service import AlertService, Notification, Priority
from lifeguard.system.alerts.store import IncidentStore
from lifeguard.system.config import load_system_config
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
CAM, SESS = "cam_t", "s1"
T0 = "2026-01-01T00:00:00.000Z"
DT = 0.1


def _uid(track_id: int) -> str:
    return make_track_uid(CAM, SESS, track_id)


def _utc(t: float) -> str:
    return add_seconds_utc(T0, t)


def _times(start: float, stop: float) -> list[float]:
    return [round(start + i * DT, 3) for i in range(round((stop - start) / DT) + 1)]


def _frame(t: float, tracks: Iterable[TrackObs] = (), subs: Iterable[Submersion] = ()) -> FrameResult:
    return FrameResult(
        cam_id=CAM, session_id=SESS, frame_idx=round(t / DT), t_capture=t, t_utc=_utc(t), t_media=t,
        frame_width=1920, frame_height=1080, latency_ms=0.0, tracks=tuple(tracks), submersions=tuple(subs),
    )


def _sub(track_id: int, t: float) -> Submersion:
    return Submersion(track_uid=_uid(track_id), last_pool_xy=(5.0, 5.0), seconds_submerged=t, since_t_capture=0.0)


def _track(track_id: int, merged_from: tuple[int, ...] = ()) -> TrackObs:
    return TrackObs(
        track_id=track_id, track_uid=_uid(track_id), merged_from=merged_from, cls=DetClass.PERSON,
        box_xyxy=(10.0, 10.0, 50.0, 70.0), det_conf=0.8, state=TrackState.TRACKED, pool_xy=(5.2, 5.0),
        in_water=True, zone=Zone.SHALLOW,
    )


def _step(alerts: AlertService, risk: RiskEngine, frame: FrameResult) -> list[Notification]:
    """The runtime order: follow merges, update risk, handle its events, then repeat-check."""
    alerts.on_frame(frame)
    notes = alerts.handle(risk.update(frame))
    return notes + alerts.tick(frame.t_capture, frame.t_utc)


def _until_alarm(store: IncidentStore | None = None) -> tuple[RuleBasedRiskEngine, AlertService, list[Notification], float]:
    """Track 1 under water from t = 0; the hard rule raises ALARM at 10.1 s; stop at 11.0 s."""
    risk = RuleBasedRiskEngine()
    alerts = AlertService(risk, AlertConfig(realert_interval_s=15.0), store)
    notes: list[Notification] = []
    for t in _times(0.0, 11.0):
        notes += _step(alerts, risk, _frame(t, subs=[_sub(1, t)]))
    return risk, alerts, notes, 11.0


def _alarm_event(alerts: AlertService) -> RiskEvent:
    (incident,) = alerts.store.incidents()
    return next(e for e in alerts.store.events(incident["incident_id"]) if e.to_level == RiskLevel.ALARM)


def _ack(alerts: AlertService, decision: AckDecision, t: float) -> Acknowledgement:
    alarm = _alarm_event(alerts)
    return Acknowledgement(event_id=alarm.event_id, track_uid=alarm.track_uid, decision=decision,
                           by="lifeguard_1", t_utc=_utc(t))


def test_alarm_opens_incident_and_notifies_once() -> None:
    _, alerts, notes, _ = _until_alarm()
    alarms = [n for n in notes if n.priority == Priority.ALARM]
    assert len(alarms) == 1 and alarms[0].repeat == 0
    assert "unconfirmed" in alarms[0].message
    (incident,) = alerts.store.incidents()
    assert incident["max_level"] == "ALARM" and incident["closed_t_utc"] is None
    assert alerts.open_incidents()[_uid(1)][1] == RiskLevel.ALARM


def test_unacknowledged_alarm_repeats_and_never_dispatches() -> None:
    risk, alerts, _, t_end = _until_alarm()
    notes: list[Notification] = []
    for t in _times(t_end + DT, t_end + 30.0):
        notes += _step(alerts, risk, _frame(t, subs=[_sub(1, t)]))
    repeats = [n for n in notes if n.priority == Priority.ALARM]
    assert [n.repeat for n in repeats] == [1, 2]
    assert "NOT ACKNOWLEDGED" in repeats[0].message
    assert all(isinstance(n, Notification) for n in notes)
    assert alerts.store.dispatches() == []
    assert risk.levels()[_uid(1)][0] == RiskLevel.ALARM


def test_confirm_creates_exactly_one_dispatch_command() -> None:
    risk, alerts, _, t = _until_alarm()
    result = alerts.acknowledge(_ack(alerts, AckDecision.CONFIRM, t))
    assert [e.to_level for e in result.events] == [RiskLevel.DISPATCH]
    assert result.dispatch is not None
    assert result.dispatch.confirmed_by == "lifeguard_1"
    assert result.dispatch.target_pool_xy == (5.0, 5.0)
    assert result.dispatch.event_id == result.events[0].event_id
    assert any(n.priority == Priority.INFO for n in result.notifications)
    assert alerts.store.dispatches() == [result.dispatch]

    again = alerts.acknowledge(_ack(alerts, AckDecision.CONFIRM, t))
    assert again.dispatch is None and again.events == ()
    later = [n for tt in _times(t + DT, t + 40.0) for n in _step(alerts, risk, _frame(tt))]
    assert [n for n in later if n.priority == Priority.ALARM] == []


def test_dismiss_records_a_hard_negative_and_stops_repeats() -> None:
    _, alerts, _, t = _until_alarm()
    result = alerts.acknowledge(_ack(alerts, AckDecision.DISMISS, t))
    assert result.dispatch is None
    assert [e.to_level for e in result.events] == [RiskLevel.RESOLVED]
    assert any(n.priority == Priority.CLEAR for n in result.notifications)
    (negative,) = alerts.store.hard_negatives()
    assert negative["ack_by"] == "lifeguard_1" and negative["level_before"] == "ALARM"
    later = [n for tt in _times(t + DT, t + 20.0) for n in alerts.tick(tt, _utc(tt))]
    assert later == []


def test_unknown_acknowledgement_changes_nothing() -> None:
    _, alerts, _, t = _until_alarm()
    ack = Acknowledgement(event_id="risk-nope-000001", track_uid=_uid(1), decision=AckDecision.CONFIRM,
                          by="lifeguard_1", t_utc=_utc(t))
    result = alerts.acknowledge(ack)
    assert result.events == () and result.dispatch is None and result.notifications == ()


def test_incident_follows_a_merged_track() -> None:
    risk, alerts, _, t = _until_alarm()
    t_next = round(t + DT, 3)
    _step(alerts, risk, _frame(t_next, tracks=[_track(5, merged_from=(1,))]))
    assert set(alerts.open_incidents()) == {_uid(5)}
    result = alerts.acknowledge(_ack(alerts, AckDecision.CONFIRM, t_next))
    assert result.dispatch is not None and result.dispatch.track_uid == _uid(5)
    assert alerts.store.incidents()[0]["track_uid"] == _uid(5)


class _DispatchWithoutPosition(RiskEngine):
    """Stub risk engine: every acknowledgement yields DISPATCH with no position."""

    def update(self, frame_result: FrameResult) -> list[RiskEvent]:
        return []

    def acknowledge(self, ack: Acknowledgement) -> list[RiskEvent]:
        return [RiskEvent(
            event_id="risk-s1-000009", cam_id=CAM, session_id=SESS, track_uid=_uid(1),
            from_level=RiskLevel.ALARM, to_level=RiskLevel.DISPATCH, risk=1.0,
            reason=RiskReason.LIFEGUARD_CONFIRMED, pool_xy=None, t_capture=1.0, t_utc=ack.t_utc,
        )]

    def reset(self) -> None:
        pass


def test_confirmed_dispatch_without_position_creates_no_mission() -> None:
    alerts = AlertService(_DispatchWithoutPosition())
    ack = Acknowledgement(event_id="risk-s1-000001", track_uid=_uid(1), decision=AckDecision.CONFIRM,
                          by="lifeguard_1", t_utc=T0)
    result = alerts.acknowledge(ack)
    assert result.dispatch is None
    assert any(n.priority == Priority.ERROR for n in result.notifications)
    assert alerts.store.dispatches() == []


def test_watch_is_logged_without_notification() -> None:
    alerts = AlertService(_DispatchWithoutPosition())

    def event(eid: str, a: RiskLevel, b: RiskLevel) -> RiskEvent:
        return RiskEvent(event_id=eid, cam_id=CAM, session_id=SESS, track_uid=_uid(2), from_level=a, to_level=b,
                         risk=0.5, reason=RiskReason.RISK_ABOVE_THRESHOLD, t_capture=1.0, t_utc=T0)

    assert alerts.handle([event("e1", RiskLevel.NORMAL, RiskLevel.WATCH)]) == []
    assert alerts.handle([event("e2", RiskLevel.WATCH, RiskLevel.NORMAL)]) == []
    (incident,) = alerts.store.incidents()
    assert incident["max_level"] == "WATCH" and incident["closed_t_utc"] == T0
    assert alerts.open_incidents() == {}


def test_messages_never_claim_a_detected_drowning() -> None:
    risk, alerts, notes, t = _until_alarm()
    for tt in _times(t + DT, t + 20.0):
        notes += _step(alerts, risk, _frame(tt, subs=[_sub(1, tt)]))
    notes += alerts.acknowledge(_ack(alerts, AckDecision.CONFIRM, t + 20.0)).notifications
    for note in notes:
        text = note.message.lower()
        assert "drowning detected" not in text and "confirmed drowning" not in text


def test_incident_log_persists_to_a_file(tmp_path: Path) -> None:
    db = tmp_path / "events" / "incidents.sqlite3"
    store = IncidentStore(db)
    _, alerts, _, t = _until_alarm(store)
    alerts.acknowledge(_ack(alerts, AckDecision.DISMISS, t))
    store.close()
    with IncidentStore(db) as reopened:
        assert len(reopened.incidents()) == 1
        assert len(reopened.hard_negatives()) == 1


def test_alert_config_in_system_yaml(tmp_path: Path) -> None:
    assert load_system_config(ROOT / "configs" / "system.yaml").alerts == AlertConfig()
    path = tmp_path / "system.yaml"
    path.write_text("alerts:\n  incident_db: db/incidents.sqlite3\n", encoding="utf-8")
    assert load_system_config(path).alerts.incident_db == (tmp_path / "db" / "incidents.sqlite3").resolve()
