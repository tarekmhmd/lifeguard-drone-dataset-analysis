"""Unit tests for every risk FSM transition, the hard rule, cooldown and hysteresis (anti-flicker)."""
from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.system.risk.config import RiskConfig
from lifeguard.system.risk.fsm import RiskFSM, Transition

CFG = RiskConfig()
DT = 0.1


def _fsm_at(level: RiskLevel, cfg: RiskConfig = CFG) -> RiskFSM:
    """An FSM placed at ``level`` at t = 0."""
    fsm = RiskFSM(cfg, 0.0)
    if level != RiskLevel.NORMAL:
        fsm.go(level, RiskReason.RISK_ABOVE_THRESHOLD, 0.0)
    return fsm


def _run(
    fsm: RiskFSM,
    risk: Callable[[float], float],
    until: float,
    dt: float = DT,
    submerged: Callable[[float], float] = lambda t: 0.0,
) -> list[tuple[float, Transition]]:
    """Step the FSM from t = 0 to ``until`` and collect (t, transition) pairs."""
    out: list[tuple[float, Transition]] = []
    for i in range(round(until / dt) + 1):
        t = round(i * dt, 6)
        transition = fsm.step(t, risk(t), submerged(t))
        if transition is not None:
            out.append((t, transition))
    return out


@pytest.mark.parametrize(
    ("start", "risk", "hold", "target", "reason"),
    [
        (RiskLevel.NORMAL, 0.5, CFG.hold_watch_s, RiskLevel.WATCH, RiskReason.RISK_ABOVE_THRESHOLD),
        (RiskLevel.WATCH, 0.7, CFG.hold_pre_alert_s, RiskLevel.PRE_ALERT, RiskReason.RISK_ABOVE_THRESHOLD),
        (RiskLevel.WATCH, 0.2, CFG.hold_down_s, RiskLevel.NORMAL, RiskReason.RISK_BELOW_THRESHOLD),
        (RiskLevel.PRE_ALERT, 0.9, CFG.hold_alarm_s, RiskLevel.ALARM, RiskReason.RISK_ABOVE_THRESHOLD),
        (RiskLevel.PRE_ALERT, 0.4, CFG.hold_down_s, RiskLevel.WATCH, RiskReason.RISK_BELOW_THRESHOLD),
    ],
)
def test_soft_transition_after_dwell(
    start: RiskLevel, risk: float, hold: float, target: RiskLevel, reason: RiskReason
) -> None:
    """Each soft transition fires exactly when its condition has held for the dwell time."""
    transitions = _run(_fsm_at(start), lambda t: risk, until=hold + 1.0)
    assert len(transitions) >= 1
    t_first, first = transitions[0]
    assert first == Transition(start, target, reason)
    assert t_first == pytest.approx(hold)


def test_interrupted_condition_restarts_dwell() -> None:
    """A single sample below the threshold restarts the dwell timer."""
    fsm = _fsm_at(RiskLevel.NORMAL)

    def risk(t: float) -> float:
        return 0.3 if 0.85 <= t < 0.95 else 0.5

    transitions = _run(fsm, risk, until=3.0)
    # Only the t = 0.9 sample is below; the dwell restarts at t = 1.0.
    assert transitions[0][0] == pytest.approx(1.0 + CFG.hold_watch_s)


@pytest.mark.parametrize(
    "start",
    [RiskLevel.NORMAL, RiskLevel.WATCH, RiskLevel.PRE_ALERT, RiskLevel.RESOLVED],
)
def test_hard_submersion_rule_from_every_lower_level(start: RiskLevel) -> None:
    """Submersion beyond t_sub forces ALARM immediately, even at zero risk and during cooldown."""
    fsm = _fsm_at(start)
    transition = fsm.step(1.0, 0.0, CFG.t_sub_s + 0.1)
    assert transition == Transition(start, RiskLevel.ALARM, RiskReason.SUBMERSION_TIMEOUT)


def test_submersion_at_limit_is_not_alarm() -> None:
    """The hard rule uses a strict inequality."""
    assert _fsm_at(RiskLevel.NORMAL).step(1.0, 0.0, CFG.t_sub_s) is None


@pytest.mark.parametrize("held", [RiskLevel.ALARM, RiskLevel.DISPATCH])
def test_alarm_and_dispatch_never_leave_automatically(held: RiskLevel) -> None:
    """Only a human decision leaves ALARM or DISPATCH: zero risk and long submersion change nothing."""
    assert _run(_fsm_at(held), lambda t: 0.0, until=120.0) == []
    assert _run(_fsm_at(held), lambda t: 1.0, until=10.0, submerged=lambda t: 60.0) == []


def test_resolved_cooldown_then_normal() -> None:
    """RESOLVED returns to NORMAL after the cooldown and ignores high soft risk meanwhile."""
    transitions = _run(_fsm_at(RiskLevel.RESOLVED), lambda t: 0.95, until=CFG.cooldown_s + 1.0)
    t_first, first = transitions[0]
    assert first == Transition(RiskLevel.RESOLVED, RiskLevel.NORMAL, RiskReason.COOLDOWN_ELAPSED)
    assert t_first == pytest.approx(CFG.cooldown_s)


def test_pre_alert_ack_timeout_disabled_by_default() -> None:
    """With the default (OD-1 open) a PRE_ALERT does not escalate on time alone."""
    assert _run(_fsm_at(RiskLevel.PRE_ALERT), lambda t: 0.65, until=120.0) == []


def test_pre_alert_ack_timeout_when_configured() -> None:
    """When configured, an un-acknowledged PRE_ALERT escalates to ALARM (never to DISPATCH)."""
    cfg = RiskConfig(pre_alert_ack_timeout_s=5.0)
    transitions = _run(_fsm_at(RiskLevel.PRE_ALERT, cfg), lambda t: 0.65, until=10.0)
    t_first, first = transitions[0]
    assert first == Transition(RiskLevel.PRE_ALERT, RiskLevel.ALARM, RiskReason.ACK_TIMEOUT_ESCALATION)
    assert t_first == pytest.approx(5.0)
    assert all(tr.to_level != RiskLevel.DISPATCH for _, tr in transitions)


def test_fsm_never_dispatches_by_itself() -> None:
    """No combination of risk and submersion moves the FSM to DISPATCH."""
    for start in (RiskLevel.NORMAL, RiskLevel.WATCH, RiskLevel.PRE_ALERT, RiskLevel.ALARM):
        transitions = _run(_fsm_at(start), lambda t: 1.0, until=60.0, submerged=lambda t: t)
        assert all(tr.to_level != RiskLevel.DISPATCH for _, tr in transitions)


@pytest.mark.parametrize(
    ("start", "centre"),
    [
        pytest.param(RiskLevel.NORMAL, CFG.t_watch, id="normal-around-t_watch"),
        pytest.param(RiskLevel.WATCH, CFG.t_pre_alert, id="watch-around-t_pre_alert"),
        pytest.param(RiskLevel.WATCH, CFG.t_watch - CFG.hysteresis, id="watch-around-exit"),
        pytest.param(RiskLevel.PRE_ALERT, CFG.t_alarm, id="pre_alert-around-t_alarm"),
        pytest.param(RiskLevel.PRE_ALERT, CFG.t_pre_alert - CFG.hysteresis, id="pre_alert-around-exit"),
    ],
)
def test_no_flicker_on_sine_crossing_every_half_second(start: RiskLevel, centre: float) -> None:
    """A risk signal crossing a threshold every 0.5 s causes 0 state changes (M2-05 criterion)."""
    transitions = _run(
        _fsm_at(start),
        lambda t: centre + 0.05 * math.sin(2.0 * math.pi * t),
        until=60.0,
        dt=0.01,
    )
    assert transitions == []
