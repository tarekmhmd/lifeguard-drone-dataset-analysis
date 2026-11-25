"""Per-track risk state machine with hysteresis, dwell times, hard submersion rule and cooldown.

Levels follow decision D-02 (Blueprint): NORMAL, WATCH, PRE_ALERT, ALARM, DISPATCH, RESOLVED.
Safety rules (decision D-03):
- the FSM never enters DISPATCH by itself; only a human CONFIRM acknowledgement does;
- ALARM and DISPATCH are never left automatically (only by acknowledgement or incident close);
- the hard submersion rule overrides everything below ALARM, including the RESOLVED cooldown.
"""
from __future__ import annotations

from dataclasses import dataclass

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.system.risk.config import RiskConfig

# Levels that are only left through a human decision.
HELD_LEVELS = frozenset({RiskLevel.ALARM, RiskLevel.DISPATCH})

# Severity order used when two histories must be merged.
SEVERITY: dict[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.RESOLVED: 1,
    RiskLevel.WATCH: 2,
    RiskLevel.PRE_ALERT: 3,
    RiskLevel.ALARM: 4,
    RiskLevel.DISPATCH: 5,
}


@dataclass(frozen=True)
class Transition:
    """One level change."""

    from_level: RiskLevel
    to_level: RiskLevel
    reason: RiskReason


class RiskFSM:
    """State machine for one track. Time is ``t_capture`` seconds (monotonic)."""

    def __init__(self, config: RiskConfig, t: float) -> None:
        """Start in NORMAL at time t."""
        self._config = config
        self.level = RiskLevel.NORMAL
        self.entered_at = t
        self._since: dict[str, float] = {}

    def go(self, to_level: RiskLevel, reason: RiskReason, t: float) -> Transition:
        """Force a transition (used by the engine for acknowledgements and track exit)."""
        transition = Transition(self.level, to_level, reason)
        self.level = to_level
        self.entered_at = t
        self._since.clear()
        return transition

    def step(
        self, t: float, risk: float, seconds_submerged: float, detector_distress: float = 0.0
    ) -> Transition | None:
        """Advance the FSM with the smoothed risk, the submersion time and the detector distress score."""
        cfg = self._config
        level = self.level
        distress = (
            cfg.detector_distress_threshold is not None
            and detector_distress >= cfg.detector_distress_threshold
        )

        if seconds_submerged > cfg.t_sub_s and level not in HELD_LEVELS:
            return self.go(RiskLevel.ALARM, RiskReason.SUBMERSION_TIMEOUT, t)

        if level in HELD_LEVELS:
            return None

        if level == RiskLevel.RESOLVED:
            if t - self.entered_at >= cfg.cooldown_s:
                return self.go(RiskLevel.NORMAL, RiskReason.COOLDOWN_ELAPSED, t)
            return None

        if (
            level == RiskLevel.PRE_ALERT
            and cfg.pre_alert_ack_timeout_s is not None
            and t - self.entered_at >= cfg.pre_alert_ack_timeout_s
        ):
            return self.go(RiskLevel.ALARM, RiskReason.ACK_TIMEOUT_ESCALATION, t)

        if distress and level in (RiskLevel.NORMAL, RiskLevel.WATCH):
            return self.go(RiskLevel.PRE_ALERT, RiskReason.DETECTOR_DISTRESS, t)

        if level == RiskLevel.NORMAL:
            if self._held("up", risk > cfg.t_watch, t, cfg.hold_watch_s):
                return self.go(RiskLevel.WATCH, RiskReason.RISK_ABOVE_THRESHOLD, t)
        elif level == RiskLevel.WATCH:
            if self._held("up", risk > cfg.t_pre_alert, t, cfg.hold_pre_alert_s):
                return self.go(RiskLevel.PRE_ALERT, RiskReason.RISK_ABOVE_THRESHOLD, t)
            if self._held("down", risk < cfg.t_watch - cfg.hysteresis, t, cfg.hold_down_s):
                return self.go(RiskLevel.NORMAL, RiskReason.RISK_BELOW_THRESHOLD, t)
        elif level == RiskLevel.PRE_ALERT:
            if self._held("up", risk > cfg.t_alarm, t, cfg.hold_alarm_s):
                return self.go(RiskLevel.ALARM, RiskReason.RISK_ABOVE_THRESHOLD, t)
            # A PRE_ALERT is held while the detector distress persists (no flapping with the risk score).
            low = risk < cfg.t_pre_alert - cfg.hysteresis and not distress
            if self._held("down", low, t, cfg.hold_down_s):
                return self.go(RiskLevel.WATCH, RiskReason.RISK_BELOW_THRESHOLD, t)
        return None

    def _held(self, name: str, condition: bool, t: float, hold_s: float) -> bool:
        """True once ``condition`` has been continuously true for at least ``hold_s`` seconds."""
        if not condition:
            self._since.pop(name, None)
            return False
        since = self._since.setdefault(name, t)
        return t - since >= hold_s
