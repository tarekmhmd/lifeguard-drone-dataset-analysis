"""Rule-based implementation of the ``RiskEngine`` contract (Blueprint M8, roadmap P2-1).

REAL implementation of the decision logic. Its thresholds and weights are Blueprint starting values and
must be tuned on real validation sessions (roadmap P8) before the system is relied upon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lifeguard.contracts.enums import AckDecision, RiskLevel, RiskReason, Status, TrackState, Zone
from lifeguard.contracts.interfaces import RiskEngine
from lifeguard.contracts.types import (
    DISTRESS_FEATURE,
    Acknowledgement,
    FrameResult,
    RiskEvent,
    TrackObs,
    make_track_uid,
)
from lifeguard.system.risk.config import RiskConfig
from lifeguard.system.risk.evidence import Observation, TrackEvidence
from lifeguard.system.risk.fsm import HELD_LEVELS, SEVERITY, RiskFSM, Transition

logger = logging.getLogger(__name__)

_SESSION_PREFIX_CHARS = 40

# Acknowledgement decision tables (D-03: only CONFIRM on ALARM reaches DISPATCH).
_CONFIRM_TARGET: dict[RiskLevel, RiskLevel] = {
    RiskLevel.WATCH: RiskLevel.ALARM,
    RiskLevel.PRE_ALERT: RiskLevel.ALARM,
    RiskLevel.ALARM: RiskLevel.DISPATCH,
}
_DISMISSABLE = frozenset({RiskLevel.WATCH, RiskLevel.PRE_ALERT, RiskLevel.ALARM, RiskLevel.DISPATCH})


@dataclass
class _TrackRisk:
    """Mutable per-track state owned by the engine."""

    uid: str
    cam_id: str
    session_id: str
    evidence: TrackEvidence
    fsm: RiskFSM
    last_seen_t: float
    last_t_media: float | None = None
    last_pool_xy: tuple[float, float] | None = None
    last_zone: Zone | None = None
    incident_events: list[str] = field(default_factory=list)


def _behavior_score(track: TrackObs) -> float:
    """P(distress) + P(drowning) from the temporal classifier, or 0 when absent."""
    if track.behavior is None:
        return 0.0
    return min(1.0, track.behavior.distress + track.behavior.drowning)


class RuleBasedRiskEngine(RiskEngine):
    """Hybrid evidence scoring + per-track FSM. Never dispatches without a human CONFIRM."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        """Create an engine with the given configuration (defaults to ``RiskConfig()``)."""
        self._config = config or RiskConfig()
        self._tracks: dict[str, _TrackRisk] = {}
        self._event_owner: dict[str, str] = {}
        self._event_counter = 0

    @property
    def config(self) -> RiskConfig:
        """The active configuration."""
        return self._config

    def levels(self) -> dict[str, tuple[RiskLevel, float]]:
        """Snapshot of every tracked person: ``track_uid -> (level, smoothed risk)``."""
        return {uid: (tr.fsm.level, tr.evidence.risk) for uid, tr in self._tracks.items()}

    def reset(self) -> None:
        """Clear all track state, open incidents and the event counter."""
        self._tracks.clear()
        self._event_owner.clear()
        self._event_counter = 0

    # ------------------------------------------------------------------ frames

    def update(self, frame_result: FrameResult) -> list[RiskEvent]:
        """Consume one FrameResult and return the level transitions it caused."""
        fr = frame_result
        if fr.status == Status.ERROR:
            logger.debug("skipping error frame %d: %s", fr.frame_idx, fr.error)
            return []

        t = fr.t_capture
        self._apply_merges(fr)
        observations = self._observations(fr)

        events: list[RiskEvent] = []
        for uid, obs in observations.items():
            tr = self._tracks.get(uid)
            if tr is None:
                tr = _TrackRisk(
                    uid=uid,
                    cam_id=fr.cam_id,
                    session_id=fr.session_id,
                    evidence=TrackEvidence(self._config),
                    fsm=RiskFSM(self._config, t),
                    last_seen_t=t,
                )
                self._tracks[uid] = tr
            tr.last_seen_t = t
            tr.last_t_media = fr.t_media
            if obs.pool_xy is not None:
                tr.last_pool_xy = obs.pool_xy
            if obs.zone is not None:
                tr.last_zone = obs.zone

            risk = tr.evidence.observe(obs)
            transition = tr.fsm.step(t, risk, obs.seconds_submerged, obs.detector_distress)
            if transition is not None:
                reported = 1.0 if transition.reason == RiskReason.SUBMERSION_TIMEOUT else risk
                events.append(self._emit(tr, transition, reported, t, fr.t_utc, fr.t_media))

        events.extend(self._handle_absent(fr, set(observations)))
        return events

    def _observations(self, fr: FrameResult) -> dict[str, Observation]:
        """Build one Observation per track or submersion timer in the frame (deterministic order)."""
        t = fr.t_capture
        submersions = {s.track_uid: s for s in fr.submersions}
        observations: dict[str, Observation] = {}
        for track in fr.tracks:
            sub = submersions.get(track.track_uid)
            observations[track.track_uid] = Observation(
                t=t,
                visible=track.state == TrackState.TRACKED,
                pool_xy=track.pool_xy,
                in_water=track.in_water,
                zone=track.zone,
                seconds_submerged=sub.seconds_submerged if sub is not None else 0.0,
                behavior_score=_behavior_score(track),
                detector_distress=track.features.get(DISTRESS_FEATURE, 0.0),
            )
        for uid, sub in submersions.items():
            if uid in observations:
                continue
            known = self._tracks.get(uid)
            observations[uid] = Observation(
                t=t,
                visible=False,
                pool_xy=sub.last_pool_xy,
                in_water=True,
                zone=known.last_zone if known is not None else None,
                seconds_submerged=sub.seconds_submerged,
            )
        return observations

    def _apply_merges(self, fr: FrameResult) -> None:
        """Move risk history from re-associated track ids to the new id (decision C-08)."""
        for track in fr.tracks:
            for old_id in track.merged_from:
                old_uid = make_track_uid(fr.cam_id, fr.session_id, old_id)
                old = self._tracks.pop(old_uid, None)
                if old is None:
                    continue
                current = self._tracks.get(track.track_uid)
                if current is None or SEVERITY[old.fsm.level] > SEVERITY[current.fsm.level]:
                    keep, drop = old, current
                else:
                    keep, drop = current, old
                if drop is not None:
                    self._forget_incident(drop)
                keep.uid = track.track_uid
                for event_id in keep.incident_events:
                    self._event_owner[event_id] = track.track_uid
                self._tracks[track.track_uid] = keep
                logger.info("risk history of %s moved to %s (level %s)", old_uid, track.track_uid,
                            keep.fsm.level.value)

    def _handle_absent(self, fr: FrameResult, observed: set[str]) -> list[RiskEvent]:
        """Time out tracks of this camera/session that were not observed in the frame."""
        t = fr.t_capture
        prefix = f"{fr.cam_id}:{fr.session_id}:"
        events: list[RiskEvent] = []
        for uid, tr in list(self._tracks.items()):
            if uid in observed or not uid.startswith(prefix):
                continue
            level = tr.fsm.level
            if level in HELD_LEVELS:
                continue  # never cleared automatically; waits for a human decision
            if level == RiskLevel.RESOLVED:
                if t - tr.fsm.entered_at >= self._config.cooldown_s:
                    transition = tr.fsm.go(RiskLevel.NORMAL, RiskReason.COOLDOWN_ELAPSED, t)
                    events.append(self._emit(tr, transition, tr.evidence.risk, t, fr.t_utc, fr.t_media))
                    self._drop(uid)
                continue
            if t - tr.last_seen_t > self._config.track_timeout_s:
                if level != RiskLevel.NORMAL:
                    transition = tr.fsm.go(RiskLevel.NORMAL, RiskReason.TRACK_EXITED, t)
                    events.append(self._emit(tr, transition, tr.evidence.risk, t, fr.t_utc, fr.t_media))
                self._drop(uid)
        return events

    # ------------------------------------------------------- human decisions

    def acknowledge(self, ack: Acknowledgement) -> list[RiskEvent]:
        """Apply a lifeguard CONFIRM or DISMISS to the incident that ``ack.event_id`` belongs to."""
        uid = self._event_owner.get(ack.event_id)
        if uid is None:
            logger.warning("acknowledgement for unknown or closed event %s ignored", ack.event_id)
            return []
        tr = self._tracks[uid]
        if ack.track_uid != uid:
            logger.info("acknowledgement names %s; incident now belongs to %s", ack.track_uid, uid)

        level = tr.fsm.level
        if ack.decision == AckDecision.CONFIRM:
            target = _CONFIRM_TARGET.get(level)
            reason = RiskReason.LIFEGUARD_CONFIRMED
        else:
            target = RiskLevel.RESOLVED if level in _DISMISSABLE else None
            reason = RiskReason.LIFEGUARD_DISMISSED
        if target is None:
            logger.info("acknowledgement %s on %s at level %s has no effect",
                        ack.decision.value, uid, level.value)
            return []

        if reason == RiskReason.LIFEGUARD_DISMISSED:
            logger.info("lifeguard %s dismissed %s at %s (hard negative)", ack.by, uid, level.value)
        transition = tr.fsm.go(target, reason, tr.last_seen_t)
        return [self._emit(tr, transition, tr.evidence.risk, tr.last_seen_t, ack.t_utc, tr.last_t_media)]

    def close_incident(self, track_uid: str, t_utc: str) -> list[RiskEvent]:
        """Close a DISPATCH incident (mission finished or handed over); DISPATCH -> RESOLVED."""
        tr = self._tracks.get(track_uid)
        if tr is None or tr.fsm.level != RiskLevel.DISPATCH:
            logger.info("close_incident(%s) ignored: no open dispatch", track_uid)
            return []
        transition = tr.fsm.go(RiskLevel.RESOLVED, RiskReason.INCIDENT_CLOSED, tr.last_seen_t)
        return [self._emit(tr, transition, tr.evidence.risk, tr.last_seen_t, t_utc, tr.last_t_media)]

    # ---------------------------------------------------------------- helpers

    def _emit(
        self,
        tr: _TrackRisk,
        transition: Transition,
        risk: float,
        t_capture: float,
        t_utc: str,
        t_media: float | None,
    ) -> RiskEvent:
        """Build the RiskEvent for a transition and maintain incident bookkeeping."""
        self._event_counter += 1
        event_id = f"risk-{tr.session_id[:_SESSION_PREFIX_CHARS]}-{self._event_counter:06d}"
        event = RiskEvent(
            event_id=event_id,
            cam_id=tr.cam_id,
            session_id=tr.session_id,
            track_uid=tr.uid,
            from_level=transition.from_level,
            to_level=transition.to_level,
            risk=min(1.0, max(0.0, risk)),
            reason=transition.reason,
            pool_xy=tr.last_pool_xy,
            t_capture=t_capture,
            t_utc=t_utc,
            t_media=t_media,
        )
        if transition.to_level == RiskLevel.NORMAL:
            self._forget_incident(tr)
        else:
            tr.incident_events.append(event_id)
            self._event_owner[event_id] = tr.uid
        logger.info("%s %s -> %s (%s, risk=%.2f) %s", tr.uid, transition.from_level.value,
                    transition.to_level.value, transition.reason.value, event.risk, event_id)
        return event

    def _forget_incident(self, tr: _TrackRisk) -> None:
        """Remove the track's incident events from the acknowledgement index."""
        for event_id in tr.incident_events:
            self._event_owner.pop(event_id, None)
        tr.incident_events.clear()

    def _drop(self, uid: str) -> None:
        """Forget a track completely."""
        tr = self._tracks.pop(uid, None)
        if tr is not None:
            self._forget_incident(tr)
