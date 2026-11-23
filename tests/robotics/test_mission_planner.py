"""Mission planner and manager (simulation only): stand-off hover point, geofence, pre-flight checks, abort."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from lifeguard.contracts.enums import AckDecision, DetClass, RiskLevel, RiskReason, TrackState, Zone
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import (
    Acknowledgement,
    DispatchCommand,
    FrameResult,
    RiskEvent,
    Submersion,
    TrackObs,
    make_track_uid,
)
from lifeguard.robotics.mission.config import MissionConfig
from lifeguard.robotics.mission.planner import (
    AbortCommand,
    MissionManager,
    MissionPlan,
    MissionPlanner,
    MissionRejected,
    VehicleStatus,
)
from lifeguard.system.alerts.service import AlertService
from lifeguard.system.config import load_system_config
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
CAM, SESS = "cam_t", "s1"
T0 = "2026-01-01T00:00:00.000Z"
GEO = StubPoolGeometry()  # 25 m x 12.5 m, NED frame = pool frame (no yaw, no offset)
READY = VehicleStatus(battery_pct=90.0, link_ok=True)


def _uid(track_id: int) -> str:
    return make_track_uid(CAM, SESS, track_id)


def _cmd(target: tuple[float, float], track_id: int = 1, event_id: str = "risk-s1-000003") -> DispatchCommand:
    return DispatchCommand(event_id=event_id, track_uid=_uid(track_id), target_pool_xy=target,
                           confirmed_by="lifeguard_1", t_utc=T0)


def _event(to_level: RiskLevel, reason: RiskReason, track_id: int = 1) -> RiskEvent:
    return RiskEvent(event_id="risk-s1-000009", cam_id=CAM, session_id=SESS, track_uid=_uid(track_id),
                     from_level=RiskLevel.DISPATCH, to_level=to_level, risk=0.5, reason=reason,
                     t_capture=1.0, t_utc=T0)


# --------------------------------------------------------------------------- planner


def test_nominal_plan_hovers_at_standoff_towards_home() -> None:
    plan = MissionPlanner(GEO).plan(_cmd((10.0, 6.0)), READY, "m1")
    assert isinstance(plan, MissionPlan)
    cfg = MissionConfig()
    assert math.dist(plan.hover_pool_xy, (10.0, 6.0)) == pytest.approx(cfg.standoff_m, abs=1e-3)
    assert math.dist(plan.hover_pool_xy, cfg.home_pool_xy) < math.dist((10.0, 6.0), cfg.home_pool_xy)
    assert [w.name for w in plan.waypoints] == ["TAKEOFF", "TRANSIT", "HOVER", "RELEASE", "RTL"]
    hover = plan.waypoints[2]
    assert hover.altitude_m == cfg.hover_altitude_m
    assert hover.ned == pytest.approx((plan.hover_pool_xy[0], plan.hover_pool_xy[1], -cfg.hover_altitude_m))
    assert all(w.altitude_m <= cfg.max_altitude_m for w in plan.waypoints)
    assert plan.waypoints[0].pool_xy == plan.waypoints[-1].pool_xy == cfg.home_pool_xy
    assert plan.confirmed_by == "lifeguard_1"


def test_hover_point_keeps_clear_of_other_people() -> None:
    planner = MissionPlanner(GEO)
    default = planner.hover_point((10.0, 6.0))
    assert default is not None
    plan = planner.plan(_cmd((10.0, 6.0)), READY, "m1", people=[default, (10.0, 6.0)])
    assert isinstance(plan, MissionPlan)
    assert math.dist(plan.hover_pool_xy, default) >= MissionConfig().standoff_m
    assert math.dist(plan.hover_pool_xy, (10.0, 6.0)) == pytest.approx(MissionConfig().standoff_m, abs=1e-3)


def test_hover_point_stays_inside_the_fence_in_a_corner() -> None:
    planner = MissionPlanner(GEO, MissionConfig(home_pool_xy=(0.0, 0.0), fence_margin_m=0.0))
    plan = planner.plan(_cmd((0.2, 0.2)), READY, "m1")
    assert isinstance(plan, MissionPlan)
    assert planner.in_fence(plan.hover_pool_xy)


@pytest.mark.parametrize(
    ("target", "vehicle", "reason"),
    [
        pytest.param((30.0, 6.0), READY, "outside the pool", id="target-outside-pool"),
        pytest.param((10.0, 6.0), VehicleStatus(20.0, True), "battery", id="low-battery"),
        pytest.param((10.0, 6.0), VehicleStatus(90.0, False), "link", id="link-down"),
    ],
)
def test_plan_rejections(target: tuple[float, float], vehicle: VehicleStatus, reason: str) -> None:
    result = MissionPlanner(GEO).plan(_cmd(target), vehicle, "m1")
    assert isinstance(result, MissionRejected) and reason in result.reason


def test_no_safe_hover_point_is_rejected() -> None:
    tiny = StubPoolGeometry(length_m=1.0, width_m=1.0)
    planner = MissionPlanner(tiny, MissionConfig(home_pool_xy=(0.0, 0.0), fence_margin_m=0.0))
    result = planner.plan(_cmd((0.5, 0.5)), READY, "m1")
    assert isinstance(result, MissionRejected) and "hover" in result.reason


def test_config_validation() -> None:
    with pytest.raises(ValidationError):
        MissionConfig(hover_altitude_m=8.0, cruise_altitude_m=6.0)
    with pytest.raises(ValueError):
        MissionPlanner(GEO, MissionConfig(home_pool_xy=(-10.0, 0.0)))
    assert load_system_config(ROOT / "configs" / "system.yaml").mission == MissionConfig()


# --------------------------------------------------------------------------- manager


def test_one_drone_one_mission() -> None:
    manager = MissionManager(MissionPlanner(GEO))
    first = manager.dispatch(_cmd((10.0, 6.0)), READY)
    second = manager.dispatch(_cmd((15.0, 6.0), track_id=2, event_id="risk-s1-000004"), READY)
    assert isinstance(first, MissionPlan) and first.mission_id == "mission-0001"
    assert isinstance(second, MissionRejected) and "busy" in second.reason
    assert manager.complete(first.mission_id) is True
    assert manager.complete(first.mission_id) is False
    assert isinstance(manager.dispatch(_cmd((15.0, 6.0), track_id=2), READY), MissionPlan)


@pytest.mark.parametrize("reason", [RiskReason.LIFEGUARD_DISMISSED, RiskReason.INCIDENT_CLOSED])
def test_dismissal_or_close_aborts_with_rtl(reason: RiskReason) -> None:
    manager = MissionManager(MissionPlanner(GEO))
    plan = manager.dispatch(_cmd((10.0, 6.0)), READY)
    assert isinstance(plan, MissionPlan)
    abort = manager.on_risk_event(_event(RiskLevel.RESOLVED, reason))
    assert abort == AbortCommand(plan.mission_id, abort.reason, "RTL")
    assert manager.active is None


def test_unrelated_events_do_not_abort() -> None:
    manager = MissionManager(MissionPlanner(GEO))
    manager.dispatch(_cmd((10.0, 6.0)), READY)
    assert manager.on_risk_event(_event(RiskLevel.RESOLVED, RiskReason.LIFEGUARD_DISMISSED, track_id=2)) is None
    assert manager.on_risk_event(_event(RiskLevel.RESOLVED, RiskReason.COOLDOWN_ELAPSED)) is None
    assert manager.active is not None


def test_abort_follows_a_merged_track() -> None:
    manager = MissionManager(MissionPlanner(GEO))
    manager.dispatch(_cmd((10.0, 6.0)), READY)
    track = TrackObs(track_id=5, track_uid=_uid(5), merged_from=(1,), cls=DetClass.PERSON,
                     box_xyxy=(10.0, 10.0, 50.0, 70.0), det_conf=0.8, state=TrackState.TRACKED,
                     pool_xy=(10.3, 6.0), in_water=True, zone=Zone.SHALLOW)
    frame = FrameResult(cam_id=CAM, session_id=SESS, frame_idx=0, t_capture=1.0, t_utc=T0,
                        frame_width=1920, frame_height=1080, latency_ms=0.0, tracks=(track,))
    manager.on_frame(frame)
    assert isinstance(manager.on_risk_event(_event(RiskLevel.RESOLVED, RiskReason.LIFEGUARD_DISMISSED, 5)),
                      AbortCommand)


# ----------------------------------------------------------- end to end (no hardware)


def test_alarm_confirm_dispatch_plan_chain() -> None:
    """Risk engine ALARM -> lifeguard CONFIRM -> DispatchCommand -> mission plan."""
    risk = RuleBasedRiskEngine()
    alerts = AlertService(risk)
    manager = MissionManager(MissionPlanner(GEO))
    alarm: RiskEvent | None = None
    for i in range(111):  # under water for 11 s at 10 fps
        t = round(i * 0.1, 3)
        frame = FrameResult(
            cam_id=CAM, session_id=SESS, frame_idx=i, t_capture=t, t_utc=add_seconds_utc(T0, t), t_media=t,
            frame_width=1920, frame_height=1080, latency_ms=0.0,
            submersions=(Submersion(track_uid=_uid(1), last_pool_xy=(10.0, 6.0), seconds_submerged=t,
                                    since_t_capture=0.0),),
        )
        for event in risk.update(frame):
            alerts.handle([event])
            if event.to_level == RiskLevel.ALARM:
                alarm = event
    assert alarm is not None

    ack = Acknowledgement(event_id=alarm.event_id, track_uid=alarm.track_uid, decision=AckDecision.CONFIRM,
                          by="lifeguard_1", t_utc=add_seconds_utc(T0, 11.0))
    result = alerts.acknowledge(ack)
    assert result.dispatch is not None
    plan = manager.dispatch(result.dispatch, READY)
    assert isinstance(plan, MissionPlan)
    assert plan.target_pool_xy == (10.0, 6.0) and plan.confirmed_by == "lifeguard_1"
