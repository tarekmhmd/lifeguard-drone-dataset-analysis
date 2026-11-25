"""Detector distress rule: sustained detector "drowning" evidence raises PRE_ALERT, never ALARM by itself."""
from __future__ import annotations

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.system.risk.config import RiskConfig
from lifeguard.system.risk.fsm import RiskFSM

ON = RiskConfig(detector_distress_threshold=0.5)


def _steps(fsm: RiskFSM, distress: float, until: float, risk: float = 0.0, t0: float = 0.0) -> list:
    out = []
    for i in range(round(until / 0.1) + 1):
        t = round(t0 + i * 0.1, 6)
        tr = fsm.step(t, risk, 0.0, distress)
        if tr is not None:
            out.append(tr)
    return out


def test_disabled_by_default() -> None:
    fsm = RiskFSM(RiskConfig(), 0.0)
    assert _steps(fsm, distress=1.0, until=10.0) == []
    assert fsm.level == RiskLevel.NORMAL


def test_distress_at_threshold_raises_pre_alert_immediately() -> None:
    fsm = RiskFSM(ON, 0.0)
    (tr,) = _steps(fsm, distress=0.5, until=1.0)
    assert tr.to_level == RiskLevel.PRE_ALERT and tr.reason == RiskReason.DETECTOR_DISTRESS


def test_below_threshold_does_nothing() -> None:
    fsm = RiskFSM(ON, 0.0)
    assert _steps(fsm, distress=0.49, until=10.0) == []


def test_distress_never_raises_alarm_by_itself() -> None:
    fsm = RiskFSM(ON, 0.0)
    _steps(fsm, distress=1.0, until=60.0)
    assert fsm.level == RiskLevel.PRE_ALERT


def test_pre_alert_held_while_distress_persists_then_released() -> None:
    fsm = RiskFSM(ON, 0.0)
    _steps(fsm, distress=0.9, until=20.0)                 # low risk score, but distress keeps PRE_ALERT
    assert fsm.level == RiskLevel.PRE_ALERT
    out = _steps(fsm, distress=0.0, until=10.0, t0=20.1)    # distress gone: normal hysteresis applies
    assert [(tr.to_level, tr.reason) for tr in out] == [
        (RiskLevel.WATCH, RiskReason.RISK_BELOW_THRESHOLD),
        (RiskLevel.NORMAL, RiskReason.RISK_BELOW_THRESHOLD),
    ]


def test_submersion_rule_still_raises_alarm() -> None:
    fsm = RiskFSM(ON, 0.0)
    fsm.step(0.0, 0.0, 0.0, 0.9)
    tr = fsm.step(0.1, 0.0, ON.t_sub_s + 0.1, 0.9)
    assert tr is not None and tr.to_level == RiskLevel.ALARM and tr.reason == RiskReason.SUBMERSION_TIMEOUT
