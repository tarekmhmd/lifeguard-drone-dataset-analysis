"""Mission executors (SIMULATION ONLY) and the response platforms.

- SimulatedExecutor: pure-Python kinematic model.
- MavlinkExecutor: driven against a scripted FAKE MAVLink connection (no ArduPilot SITL here); this checks
  the command sequence and failsafe logic, not real flight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.contracts.types import DispatchCommand, RiskEvent
from lifeguard.robotics.executors.base import ExecState, ExecutorError
from lifeguard.robotics.executors.config import DroneConfig, ResponseConfig
from lifeguard.robotics.executors.mavlink import (
    COPTER_GUIDED,
    COPTER_RTL,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_DO_SET_SERVO,
    MAV_CMD_NAV_TAKEOFF,
    MavlinkExecutor,
    check_endpoint,
)
from lifeguard.robotics.executors.simulated import SimulatedExecutor
from lifeguard.robotics.mission.planner import MissionPlan, MissionPlanner
from lifeguard.robotics.response import DroneResponse, build_executor
from lifeguard.system.config import load_system_config
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.response import ManualResponse, OutcomeKind

ROOT = Path(__file__).resolve().parents[2]
GEO = StubPoolGeometry()
T0 = "2026-01-01T00:00:00.000Z"
UID = "cam_t:s1:1"


def _cmd(target: tuple[float, float] = (10.0, 6.0), event_id: str = "risk-s1-000003") -> DispatchCommand:
    return DispatchCommand(event_id=event_id, track_uid=UID, target_pool_xy=target,
                           confirmed_by="lifeguard_1", t_utc=T0)


def _plan() -> MissionPlan:
    from lifeguard.robotics.mission.planner import VehicleStatus

    plan = MissionPlanner(GEO).plan(_cmd(), VehicleStatus(90.0, True), "mission-0001")
    assert isinstance(plan, MissionPlan)
    return plan


def _run(ex: SimulatedExecutor, t0: float, seconds: float, on_tick=None) -> tuple[list, float]:
    events, t = [], t0
    for i in range(round(seconds / 0.1)):
        t = round(t0 + (i + 1) * 0.1, 3)
        if on_tick is not None:
            on_tick(ex, t)
        events += ex.step(t)
        if ex.state == ExecState.LANDED:
            break
    return events, t


# --------------------------------------------------------------- simulated executor


def test_simulated_mission_waits_for_human_release_then_lands() -> None:
    ex = SimulatedExecutor(DroneConfig())
    ex.start(_plan(), 0.0)
    events, t = _run(ex, 0.0, 60.0)
    kinds = [e.kind for e in events]
    assert "awaiting_release" in kinds and "released" not in kinds and not ex.released
    assert ex.state == ExecState.AWAITING_RELEASE
    assert ex.confirm_release("lifeguard_1", t)
    events, _ = _run(ex, t, 60.0)
    kinds = [e.kind for e in events]
    assert kinds[0] == "released" and "lifeguard_1" in events[0].detail and kinds[-1] == "completed"
    assert ex.released and ex.state == ExecState.LANDED
    assert ex.position_ned == pytest.approx((-1.0, -1.0, 0.0))


def test_release_confirmation_is_rejected_when_not_hovering() -> None:
    ex = SimulatedExecutor()
    ex.start(_plan(), 0.0)
    assert not ex.confirm_release("lifeguard_1", 0.0)


def test_link_loss_triggers_rtl_failsafe() -> None:
    ex = SimulatedExecutor(DroneConfig(link_timeout_s=3.0))
    ex.start(_plan(), 0.0)
    _run(ex, 0.0, 1.0)
    ex.set_link(False, 1.0)
    events, _ = _run(ex, 1.0, 60.0)
    failsafe = [e for e in events if e.kind == "failsafe"]
    assert failsafe and "link" in failsafe[0].detail and failsafe[0].t >= 4.0 - 1e-9
    assert events[-1].kind == "completed" and not ex.released


def test_low_battery_triggers_rtl_failsafe() -> None:
    ex = SimulatedExecutor(DroneConfig(rtl_battery_pct=25.0, sim_drain_pct_per_s=5.0), battery_pct=40.0)
    ex.start(_plan(), 0.0)
    events, _ = _run(ex, 0.0, 60.0)
    assert any(e.kind == "failsafe" and "battery" in e.detail for e in events)
    assert events[-1].kind == "completed" and not ex.released


def test_start_refused_when_busy_or_link_down() -> None:
    ex = SimulatedExecutor()
    ex.start(_plan(), 0.0)
    with pytest.raises(ExecutorError):
        ex.start(_plan(), 0.1)
    down = SimulatedExecutor()
    down.set_link(False, 0.0)
    with pytest.raises(ExecutorError):
        down.start(_plan(), 0.0)


# ------------------------------------------------------------------ MAVLink executor


@dataclass
class _Msg:
    kind: str
    fields: dict[str, float] = field(default_factory=dict)

    def get_type(self) -> str:
        return self.kind

    def __getattr__(self, name: str) -> float:
        try:
            return self.fields[name]
        except KeyError as err:
            raise AttributeError(name) from err


class _FakeMav:
    def __init__(self) -> None:
        self.commands: list[tuple[int, tuple[float, ...]]] = []
        self.targets: list[tuple[float, float, float]] = []

    def command_long_send(self, sysid: int, compid: int, command: int, confirmation: int, *p: float) -> None:
        self.commands.append((command, p))

    def set_position_target_local_ned_send(self, *args: Any) -> None:
        self.targets.append((args[5], args[6], args[7]))


class _FakeConn:
    """Scripted stand-in for a pymavlink connection (no network, no SITL)."""

    target_system = 1
    target_component = 1

    def __init__(self, heartbeat: bool = True) -> None:
        self.mav = _FakeMav()
        self.inbox: list[_Msg] = []
        self._heartbeat = heartbeat
        self.closed = False

    def wait_heartbeat(self, timeout: float) -> object | None:
        return object() if self._heartbeat else None

    def recv_match(self, blocking: bool = False) -> _Msg | None:
        return self.inbox.pop(0) if self.inbox else None

    def at(self, ned: tuple[float, float, float]) -> None:
        self.inbox += [_Msg("HEARTBEAT"), _Msg("SYS_STATUS", {"battery_remaining": 90.0}),
                       _Msg("LOCAL_POSITION_NED", {"x": ned[0], "y": ned[1], "z": ned[2]})]

    def close(self) -> None:
        self.closed = True


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.parametrize("endpoint", ["COM3", "/dev/ttyUSB0", "udp:192.168.1.5:14550", "serial:/dev/ttyACM0"])
def test_physical_endpoints_are_refused(endpoint: str) -> None:
    with pytest.raises(ExecutorError, match="physical deployment is disabled"):
        check_endpoint(endpoint)
    check_endpoint("udp:127.0.0.1:14550")
    check_endpoint("tcp:localhost:5760")


def test_no_heartbeat_is_an_executor_error() -> None:
    with pytest.raises(ExecutorError, match="no heartbeat"):
        MavlinkExecutor(DroneConfig(), connection_factory=lambda e: _FakeConn(heartbeat=False))


def test_mavlink_command_sequence_with_human_release() -> None:
    conn, clock = _FakeConn(), _Clock()
    ex = MavlinkExecutor(DroneConfig(), connection_factory=lambda e: conn, clock=clock)
    plan = _plan()
    conn.at((-1.0, -1.0, 0.0))
    assert ex.vehicle_status().battery_pct == 90.0 and ex.vehicle_status().link_ok
    ex.start(plan, 0.0)
    cmds = conn.mav.commands
    assert cmds[0] == (MAV_CMD_DO_SET_MODE, (1, COPTER_GUIDED, 0, 0, 0, 0, 0))
    assert cmds[1][0] == MAV_CMD_COMPONENT_ARM_DISARM and cmds[1][1][0] == 1
    assert cmds[2][0] == MAV_CMD_NAV_TAKEOFF and cmds[2][1][6] == pytest.approx(6.0)

    kinds: list[str] = []
    for wp in plan.waypoints[:4]:  # TAKEOFF, TRANSIT, HOVER, RELEASE positions reached in turn
        conn.at(wp.ned)
        kinds += [e.kind for e in ex.step(1.0)]
        kinds += [e.kind for e in ex.step(1.1)]
    assert kinds.count("progress") == 3 and "awaiting_release" in kinds
    assert conn.mav.targets[0] == pytest.approx(plan.waypoints[1].ned)
    assert not any(c[0] == MAV_CMD_DO_SET_SERVO for c in conn.mav.commands)

    assert ex.confirm_release("lifeguard_1", 2.0)
    released = ex.step(2.0)
    assert released[0].kind == "released"
    assert (MAV_CMD_DO_SET_SERVO, (9, 1900, 0, 0, 0, 0, 0)) in conn.mav.commands
    ex.step(2.1)
    assert conn.mav.commands[-1] == (MAV_CMD_DO_SET_MODE, (1, COPTER_RTL, 0, 0, 0, 0, 0))
    conn.at((-1.0, -1.0, -0.1))
    assert [e.kind for e in ex.step(3.0)] == ["completed"]
    assert ex.state == ExecState.LANDED
    ex.close()
    assert conn.closed


def test_mavlink_link_loss_failsafe() -> None:
    conn, clock = _FakeConn(), _Clock()
    ex = MavlinkExecutor(DroneConfig(link_timeout_s=3.0), connection_factory=lambda e: conn, clock=clock)
    ex.start(_plan(), 0.0)
    clock.now = 5.0  # no heartbeat for 5 s
    (event,) = ex.step(5.0)
    assert event.kind == "failsafe" and "link" in event.detail
    assert conn.mav.commands[-1] == (MAV_CMD_DO_SET_MODE, (1, COPTER_RTL, 0, 0, 0, 0, 0))
    assert ex.state == ExecState.RETURNING


def test_mavlink_abort_sends_rtl() -> None:
    conn = _FakeConn()
    ex = MavlinkExecutor(DroneConfig(), connection_factory=lambda e: conn, clock=_Clock())
    ex.start(_plan(), 0.0)
    ex.abort("lifeguard dismissed", 1.0)
    assert conn.mav.commands[-1] == (MAV_CMD_DO_SET_MODE, (1, COPTER_RTL, 0, 0, 0, 0, 0))


def test_mavlink_executor_without_sitl_fails_safely() -> None:
    """No pymavlink installed or no SITL listening: a clear ExecutorError, never a hang or a crash."""
    with pytest.raises(ExecutorError):
        build_executor(DroneConfig(executor="mavlink_sitl", heartbeat_timeout_s=0.5))


# -------------------------------------------------------------------- response layer


def _dismissal() -> RiskEvent:
    return RiskEvent(event_id="risk-s1-000010", cam_id="cam_t", session_id="s1", track_uid=UID,
                     from_level=RiskLevel.DISPATCH, to_level=RiskLevel.RESOLVED, risk=0.5,
                     reason=RiskReason.LIFEGUARD_DISMISSED, t_capture=5.0, t_utc=T0)


def test_drone_response_full_simulated_cycle() -> None:
    drone = DroneResponse(GEO, executor=SimulatedExecutor())
    accepted = drone.dispatch(_cmd(), 0.0)
    assert accepted.kind == OutcomeKind.ACCEPTED and "SIMULATION" in accepted.detail
    assert drone.dispatch(_cmd(event_id="risk-s1-000004"), 0.0).kind == OutcomeKind.REJECTED

    outcomes, t = [], 0.0
    for i in range(1, 600):
        t = round(i * 0.1, 3)
        outcomes += drone.step(t)
        if any(o.kind == OutcomeKind.AWAITING_CONFIRMATION for o in outcomes):
            break
    assert drone.confirm_release("lifeguard_1") is not None
    for i in range(1, 600):
        outcomes += drone.step(round(t + i * 0.1, 3))
        if outcomes[-1:] and outcomes[-1].kind == OutcomeKind.COMPLETED:
            break
    assert outcomes[-1].kind == OutcomeKind.COMPLETED
    assert drone.status()["mission"] == "-"


def test_drone_response_aborts_on_dismissal() -> None:
    ex = SimulatedExecutor()
    drone = DroneResponse(GEO, executor=ex)
    drone.dispatch(_cmd(), 0.0)
    drone.step(1.0)
    (aborted,) = drone.on_risk_event(_dismissal())
    assert aborted.kind == OutcomeKind.ABORTED and "RTL" in aborted.detail
    assert ex.state == ExecState.RETURNING


def test_drone_not_dispatched_when_link_down() -> None:
    ex = SimulatedExecutor()
    ex.set_link(False, 0.0)
    result = DroneResponse(GEO, executor=ex).dispatch(_cmd(), 0.0)
    assert result.kind == OutcomeKind.REJECTED and "respond in person" in result.detail


def test_manual_response_for_fixed_camera_deployments() -> None:
    out = ManualResponse().dispatch(_cmd(), 0.0)
    assert out.kind == OutcomeKind.MANUAL and "lifeguards respond in person" in out.detail
    assert "lifeguard_1" in out.detail


def test_response_config_defaults_to_no_platform() -> None:
    cfg = load_system_config(ROOT / "configs" / "system.yaml")
    assert cfg.response == ResponseConfig()
    assert cfg.response.platform == "none"
    assert cfg.response.drone.require_release_confirmation is True
