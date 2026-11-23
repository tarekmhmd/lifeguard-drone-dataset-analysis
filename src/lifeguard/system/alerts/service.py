"""Alert service (Blueprint M9, roadmap P3-1 / P3-2): RiskEvent -> incidents and operator notifications.

REAL implementation of the alerting logic. Delivery channels (dashboard banner, siren relay, phone push)
are not built yet; callers receive ``Notification`` objects and decide how to present them.

Rules:
- WATCH is logged only; PRE_ALERT -> visual notification; ALARM -> visual + sound;
- an ALARM nobody acknowledges is repeated every ``realert_interval_s`` (louder / other channels);
  the repeat NEVER dispatches (decision D-03) and is not a RiskEvent (the contract requires a level change);
- a ``DispatchCommand`` is created in exactly one place: ``acknowledge()`` when a lifeguard's CONFIRM moved
  an incident to DISPATCH; ``confirmed_by`` is the lifeguard's name;
- every message says the alert is machine-generated and unconfirmed; nothing claims a drowning was detected.
"""
from __future__ import annotations

import enum
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from lifeguard.contracts.enums import AckDecision, HealthComponent, RiskLevel, RiskReason, Status
from lifeguard.contracts.interfaces import RiskEngine
from lifeguard.contracts.types import (
    Acknowledgement,
    DispatchCommand,
    FrameResult,
    RiskEvent,
    SystemHealth,
    make_track_uid,
)
from lifeguard.system.alerts.config import AlertConfig
from lifeguard.system.alerts.store import IncidentStore
from lifeguard.system.risk.fsm import SEVERITY

logger = logging.getLogger(__name__)


class Priority(str, enum.Enum):
    """How a notification should be presented."""

    VISUAL = "visual"   # dashboard highlight
    ALARM = "alarm"     # visual + sound (+ push when available)
    INFO = "info"
    CLEAR = "clear"     # remove the highlight
    ERROR = "error"     # the system could not do what was asked


@dataclass(frozen=True)
class Notification:
    """One message for the operator."""

    incident_id: str
    track_uid: str          # "" for system-health notifications
    priority: Priority
    level: RiskLevel | None  # None for system-health notifications
    message: str
    t_utc: str
    repeat: int = 0


@dataclass(frozen=True)
class AckResult:
    """Outcome of a lifeguard decision."""

    events: tuple[RiskEvent, ...]
    notifications: tuple[Notification, ...]
    dispatch: DispatchCommand | None


@dataclass
class _Incident:
    incident_id: str
    track_uid: str
    level: RiskLevel
    max_level: RiskLevel
    pool_xy: tuple[float, float] | None
    first_alarm_t: float | None = None
    last_alert_t: float | None = None
    repeat: int = 0


def _where(pool_xy: tuple[float, float] | None) -> str:
    return "position unknown" if pool_xy is None else f"near pool ({pool_xy[0]:.1f} m, {pool_xy[1]:.1f} m)"


class AlertService:
    """Turns risk transitions into incidents and notifications; routes lifeguard decisions."""

    def __init__(
        self,
        risk: RiskEngine,
        config: AlertConfig | None = None,
        store: IncidentStore | None = None,
    ) -> None:
        self._risk = risk
        self._config = config or AlertConfig()
        self._store = store or IncidentStore()
        self._open: dict[str, _Incident] = {}        # by track_uid
        self._event_incident: dict[str, str] = {}    # event_id -> incident_id
        self._by_id: dict[str, _Incident] = {}       # incident_id -> incident
        self._health: dict[HealthComponent, Status] = {}

    @property
    def store(self) -> IncidentStore:
        return self._store

    def incident_for_event(self, event_id: str) -> tuple[str, str] | None:
        """``(incident_id, current track_uid)`` of the incident a RiskEvent belongs to, or None."""
        incident_id = self._event_incident.get(event_id)
        inc = self._by_id.get(incident_id) if incident_id is not None else None
        return None if inc is None else (inc.incident_id, inc.track_uid)

    def latest_event_id(self, incident_id: str) -> str | None:
        """The most recent RiskEvent id of an incident (what an operator acknowledges)."""
        latest = None
        for event_id, owner in self._event_incident.items():
            if owner == incident_id:
                latest = event_id
        return latest

    def open_incidents(self) -> dict[str, tuple[str, RiskLevel]]:
        """``track_uid -> (incident_id, level)`` for every open incident."""
        return {uid: (inc.incident_id, inc.level) for uid, inc in self._open.items()}

    # ------------------------------------------------------------- inputs

    def on_frame(self, frame: FrameResult) -> None:
        """Follow re-associated tracks (``merged_from``) so incidents stay on the same person."""
        for track in frame.tracks:
            for old_id in track.merged_from:
                old_uid = make_track_uid(frame.cam_id, frame.session_id, old_id)
                inc = self._open.get(old_uid)
                if inc is None or track.track_uid in self._open:
                    continue
                del self._open[old_uid]
                inc.track_uid = track.track_uid
                self._open[track.track_uid] = inc
                self._persist(inc, closed_t_utc=None)

    def handle(self, events: Iterable[RiskEvent]) -> list[Notification]:
        """Record risk transitions and return the notifications they cause."""
        notes: list[Notification] = []
        for event in events:
            note = self._handle_one(event)
            if note is not None:
                notes.append(note)
        return notes

    def on_health(self, states: Iterable[SystemHealth]) -> list[Notification]:
        """SYSTEM FAULT notifications (Blueprint M13: same priority as a drowning alarm) and recoveries."""
        notes: list[Notification] = []
        for state in states:
            before = self._health.get(state.component)
            self._health[state.component] = state.status
            if before == state.status:
                continue
            key = f"health-{state.component.value}"
            if state.status == Status.ERROR:
                notes.append(Notification(
                    key, "", Priority.ALARM, None,
                    f"SYSTEM FAULT ({state.component.value}): {state.detail}. Automatic monitoring may be "
                    "impaired: watch the pool directly.", state.t_utc,
                ))
            elif before == Status.ERROR:
                notes.append(Notification(key, "", Priority.CLEAR, None,
                                          f"{state.component.value} recovered: {state.detail}.", state.t_utc))
            elif state.status == Status.DEGRADED:
                notes.append(Notification(key, "", Priority.INFO, None,
                                          f"{state.component.value} degraded: {state.detail}.", state.t_utc))
        return notes

    def health(self) -> dict[HealthComponent, Status]:
        """Latest status per component."""
        return dict(self._health)

    def attach_clip(self, incident_id: str, path: str, t_utc: str) -> None:
        """Record an alarm clip as evidence for an incident."""
        self._store.add_clip(incident_id, path, t_utc)

    def tick(self, t_capture: float, t_utc: str) -> list[Notification]:
        """Repeat every unacknowledged ALARM whose repeat interval has elapsed. Never dispatches."""
        notes: list[Notification] = []
        for inc in self._open.values():
            if inc.level != RiskLevel.ALARM or inc.last_alert_t is None or inc.first_alarm_t is None:
                continue
            if t_capture - inc.last_alert_t < self._config.realert_interval_s:
                continue
            inc.repeat += 1
            inc.last_alert_t = t_capture
            waited = t_capture - inc.first_alarm_t
            notes.append(Notification(
                inc.incident_id, inc.track_uid, Priority.ALARM, RiskLevel.ALARM,
                f"ALARM NOT ACKNOWLEDGED for {waited:.0f} s: check swimmer {inc.track_uid} now, "
                f"{_where(inc.pool_xy)}. Machine-generated alert, unconfirmed.",
                t_utc, repeat=inc.repeat,
            ))
            logger.warning("incident %s: ALARM unacknowledged for %.0f s (repeat %d)",
                           inc.incident_id, waited, inc.repeat)
        return notes

    def acknowledge(self, ack: Acknowledgement) -> AckResult:
        """Forward a lifeguard decision to the risk engine; create a DispatchCommand only on CONFIRM."""
        incident_id = self._event_incident.get(ack.event_id)
        inc = self._by_id.get(incident_id) if incident_id is not None else None
        level_before = inc.level.value if inc is not None else "UNKNOWN"

        events = tuple(self._risk.acknowledge(ack))
        notes = self.handle(events)
        if inc is not None:
            self._store.add_ack(inc.incident_id, ack, level_before)

        dispatch: DispatchCommand | None = None
        for event in events:
            if not (
                ack.decision == AckDecision.CONFIRM
                and event.to_level == RiskLevel.DISPATCH
                and event.reason == RiskReason.LIFEGUARD_CONFIRMED
            ):
                continue
            owner = self._event_incident[event.event_id]
            if event.pool_xy is None:
                notes.append(Notification(
                    owner, event.track_uid, Priority.ERROR, RiskLevel.DISPATCH,
                    "Dispatch confirmed but the swimmer position is unknown: no drone mission was created. "
                    "Respond manually.",
                    ack.t_utc,
                ))
                logger.error("incident %s: confirmed dispatch without a position", owner)
                continue
            dispatch = DispatchCommand(
                event_id=event.event_id,
                track_uid=event.track_uid,
                target_pool_xy=event.pool_xy,
                confirmed_by=ack.by,
                t_utc=ack.t_utc,
            )
            self._store.add_dispatch(owner, dispatch)
            logger.info("incident %s: dispatch confirmed by %s", owner, ack.by)
        return AckResult(events, tuple(notes), dispatch)

    # ----------------------------------------------------------- internals

    def _handle_one(self, event: RiskEvent) -> Notification | None:
        inc = self._open.get(event.track_uid)
        if inc is None:
            if event.to_level == RiskLevel.NORMAL:
                return None
            inc = _Incident(event.event_id, event.track_uid, event.to_level, event.to_level, event.pool_xy)
            self._open[event.track_uid] = inc
            self._by_id[inc.incident_id] = inc
            self._store.open_incident(inc.incident_id, event)

        self._event_incident[event.event_id] = inc.incident_id
        self._store.add_event(inc.incident_id, event)
        inc.level = event.to_level
        if SEVERITY[event.to_level] > SEVERITY[inc.max_level]:
            inc.max_level = event.to_level
        if event.pool_xy is not None:
            inc.pool_xy = event.pool_xy

        note = self._notification_for(inc, event)
        if event.to_level == RiskLevel.NORMAL:
            del self._open[inc.track_uid]
            self._persist(inc, closed_t_utc=event.t_utc)
        else:
            self._persist(inc, closed_t_utc=None)
        return note

    def _notification_for(self, inc: _Incident, event: RiskEvent) -> Notification | None:
        level = event.to_level
        where = _where(inc.pool_xy)
        if level == RiskLevel.PRE_ALERT:
            return Notification(inc.incident_id, inc.track_uid, Priority.VISUAL, level,
                                f"Possible distress: please check swimmer {inc.track_uid}, {where}. "
                                "Machine-generated alert, unconfirmed.", event.t_utc)
        if level == RiskLevel.ALARM:
            if inc.first_alarm_t is None:
                inc.first_alarm_t = event.t_capture
            inc.last_alert_t = event.t_capture
            inc.repeat = 0
            return Notification(inc.incident_id, inc.track_uid, Priority.ALARM, level,
                                f"ALARM: check swimmer {inc.track_uid} now, {where} "
                                f"(reason: {event.reason.value}). Machine-generated alert, unconfirmed.",
                                event.t_utc)
        if level == RiskLevel.DISPATCH:
            return Notification(inc.incident_id, inc.track_uid, Priority.INFO, level,
                                "Lifeguard confirmed the emergency; simulated drone dispatch requested.",
                                event.t_utc)
        if level in (RiskLevel.RESOLVED, RiskLevel.NORMAL) and SEVERITY[inc.max_level] >= SEVERITY[
            RiskLevel.PRE_ALERT
        ]:
            return Notification(inc.incident_id, inc.track_uid, Priority.CLEAR, level,
                                f"Incident {level.value.lower()} ({event.reason.value}).", event.t_utc)
        return None

    def _persist(self, inc: _Incident, closed_t_utc: str | None) -> None:
        self._store.update_incident(
            inc.incident_id,
            track_uid=inc.track_uid,
            level=inc.level.value,
            max_level=inc.max_level.value,
            closed_t_utc=closed_t_utc,
        )
