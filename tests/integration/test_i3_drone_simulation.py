"""Milestone I3 — drone dispatch in simulation (SYSTEM_INTEGRATION_CHECKLIST.md section 2), run against the
pure-Python SimulatedExecutor. The official milestone requires ArduPilot SITL + Gazebo; that part remains
open and is NOT claimed here.

Exit criteria checked in pure-Python simulation:
- the hover point is never over the swimmer zone (>= stand-off from the swimmer and from everyone else);
- every failure scenario (low battery, link loss, alarm cancelled mid-flight) ends landed at launch; a target
  outside the fence/pool is refused before take-off;
- the position error at the hover point is measured and reported.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.contracts.types import DispatchCommand, RiskEvent
from lifeguard.robotics.executors.base import ExecState
from lifeguard.robotics.executors.config import DroneConfig
from lifeguard.robotics.executors.simulated import SimulatedExecutor
from lifeguard.robotics.mission.config import MissionConfig
from lifeguard.robotics.mission.planner import MissionPlan, MissionPlanner, VehicleStatus
from lifeguard.robotics.response import DroneResponse
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.response import OutcomeKind, ResponseOutcome

GEO = StubPoolGeometry()
T0 = "2026-01-01T00:00:00.000Z"
UID = "cam_sim:i3:1"
STANDOFF = MissionConfig().standoff_m


def _cmd(target: tuple[float, float]) -> DispatchCommand:
    return DispatchCommand(event_id="risk-i3-000001", track_uid=UID, target_pool_xy=target,
                           confirmed_by="lifeguard_1", t_utc=T0)


def _fly(
    drone: DroneResponse,
    until: float = 120.0,
    on_tick: Callable[[DroneResponse, float], None] | None = None,
) -> list[ResponseOutcome]:
    outcomes = []
    for i in range(1, round(until / 0.1)):
        t = round(i * 0.1, 3)
        if on_tick is not None:
            on_tick(drone, t)
        outcomes += drone.step(t)
        if drone.executor.state == ExecState.AWAITING_RELEASE:
            drone.confirm_release("lifeguard_1")
        if drone.executor.state == ExecState.LANDED:
            break
    return outcomes


@pytest.mark.parametrize(("x", "y"), list(itertools.product([2.0, 8.0, 12.5, 18.0, 23.0], [2.0, 6.25, 10.5])))
def test_hover_point_is_never_over_the_swimmer(x: float, y: float) -> None:
    others = [(x + 1.0, y), (x, y - 1.0)]
    plan = MissionPlanner(GEO).plan(_cmd((x, y)), VehicleStatus(90.0, True), "m", people=others)
    assert isinstance(plan, MissionPlan)
    assert math.dist(plan.hover_pool_xy, (x, y)) >= STANDOFF - 1e-3
    assert all(math.dist(plan.hover_pool_xy, o) >= STANDOFF - 1e-3 for o in others)


def test_nominal_mission_hover_position_error_and_landing() -> None:
    ex = SimulatedExecutor(DroneConfig())
    drone = DroneResponse(GEO, executor=ex)
    assert drone.dispatch(_cmd((12.0, 6.0)), 0.0).kind == OutcomeKind.ACCEPTED
    plan = drone.active_mission
    assert plan is not None
    hover_ned = plan.waypoints[2].ned
    error_m = None
    for i in range(1, 2000):
        t = round(i * 0.1, 3)
        drone.step(t)
        if ex.state == ExecState.AWAITING_RELEASE and error_m is None:
            error_m = math.dist(ex.position_ned, hover_ned)
            drone.confirm_release("lifeguard_1")
        if ex.state == ExecState.LANDED:
            break
    assert error_m is not None and error_m <= DroneConfig().arrive_tolerance_m
    print(f"[SIMULATED] hover position error {error_m:.3f} m")
    assert ex.state == ExecState.LANDED and ex.released


def test_low_battery_ends_landed() -> None:
    ex = SimulatedExecutor(DroneConfig(sim_drain_pct_per_s=5.0), battery_pct=45.0)
    drone = DroneResponse(GEO, executor=ex)
    drone.dispatch(_cmd((20.0, 8.0)), 0.0)
    outcomes = _fly(drone)
    assert any(o.kind == OutcomeKind.ABORTED and "battery" in o.detail for o in outcomes)
    assert ex.state == ExecState.LANDED and not ex.released


def test_link_loss_ends_landed() -> None:
    ex = SimulatedExecutor(DroneConfig(link_timeout_s=3.0))
    drone = DroneResponse(GEO, executor=ex)
    drone.dispatch(_cmd((20.0, 8.0)), 0.0)

    def lose_link(d: DroneResponse, t: float) -> None:
        if math.isclose(t, 1.0):
            ex.set_link(False, t)

    outcomes = _fly(drone, on_tick=lose_link)
    assert any("link" in o.detail for o in outcomes)
    assert ex.state == ExecState.LANDED and not ex.released


def test_alarm_cancelled_mid_flight_ends_landed() -> None:
    ex = SimulatedExecutor()
    drone = DroneResponse(GEO, executor=ex)
    drone.dispatch(_cmd((20.0, 8.0)), 0.0)
    drone.step(1.0)
    cancel = RiskEvent(event_id="risk-i3-000002", cam_id="cam_sim", session_id="i3", track_uid=UID,
                       from_level=RiskLevel.DISPATCH, to_level=RiskLevel.RESOLVED, risk=0.2,
                       reason=RiskReason.LIFEGUARD_DISMISSED, t_capture=1.0, t_utc=T0)
    assert drone.on_risk_event(cancel)[0].kind == OutcomeKind.ABORTED
    _fly(drone)
    assert ex.state == ExecState.LANDED and not ex.released


def test_target_outside_the_fence_is_refused_before_take_off() -> None:
    ex = SimulatedExecutor()
    result = DroneResponse(GEO, executor=ex).dispatch(_cmd((40.0, 6.0)), 0.0)
    assert result.kind == OutcomeKind.REJECTED and ex.state == ExecState.IDLE
