"""MAVLink executor for ArduPilot SITL (roadmap P6-2/P6-3). SIMULATION ONLY.

Physical deployment is disabled by construction: only loopback UDP/TCP endpoints (a local SITL instance)
are accepted; serial ports, radios and remote hosts are refused.

Sequence (ArduCopter): GUIDED -> arm -> NAV_TAKEOFF -> SET_POSITION_TARGET_LOCAL_NED per waypoint ->
(human-confirmed) DO_SET_SERVO on the release channel -> RTL. Failsafes: no heartbeat for
``link_timeout_s`` or battery below ``rtl_battery_pct`` -> RTL (ArduPilot's own GCS/battery failsafes
should be configured too, Blueprint M11). The plan's NED frame must equal the vehicle's EKF origin frame
(measured pool -> NED transform in the calibration, roadmap P4-2).

Requires ``pymavlink`` (``pip install -e ".[drone]"``) unless a connection factory is injected (tests).
VERIFICATION STATUS: exercised against a scripted fake connection only; NOT yet run against ArduPilot SITL.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from typing import Any

from lifeguard.robotics.executors.base import ExecState, ExecutorError, ExecutorEvent, MissionExecutor
from lifeguard.robotics.executors.config import DroneConfig
from lifeguard.robotics.mission.planner import MissionPlan, VehicleStatus

logger = logging.getLogger(__name__)

# MAVLink common message-set constants (stable protocol values).
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_SET_SERVO = 183
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_FRAME_LOCAL_NED = 1
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
POSITION_ONLY_MASK = 0b0000_1111_1111_1000  # use x, y, z; ignore velocity, acceleration, yaw
# ArduCopter custom modes.
COPTER_GUIDED = 4
COPTER_RTL = 6

_LOCAL_ENDPOINT = re.compile(r"^(udp|udpin|udpout|tcp):(127\.0\.0\.1|localhost):\d{2,5}$")
_LANDED_ALT_M = 0.3


def check_endpoint(endpoint: str) -> None:
    """Refuse anything but a local SITL endpoint."""
    if not _LOCAL_ENDPOINT.match(endpoint):
        raise ExecutorError(
            f"refused endpoint {endpoint!r}: physical deployment is disabled; only local SITL endpoints "
            "such as udp:127.0.0.1:14550 are allowed"
        )


def _default_factory(endpoint: str) -> Any:
    try:
        from pymavlink import mavutil
    except ImportError as err:
        raise ExecutorError('pymavlink is not installed: pip install -e ".[drone]"') from err
    return mavutil.mavlink_connection(endpoint)


class MavlinkExecutor(MissionExecutor):
    """Drives an ArduCopter SITL vehicle through a MissionPlan."""

    def __init__(
        self,
        config: DroneConfig | None = None,
        connection_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config or DroneConfig()
        check_endpoint(self._cfg.endpoint)
        self._clock = clock
        factory = connection_factory or _default_factory
        self._conn = factory(self._cfg.endpoint)
        if self._conn.wait_heartbeat(timeout=self._cfg.heartbeat_timeout_s) is None:
            raise ExecutorError(f"no heartbeat from {self._cfg.endpoint} within {self._cfg.heartbeat_timeout_s} s")
        self._last_heartbeat = self._clock()
        self._battery_pct: float | None = None
        self._pos: tuple[float, float, float] | None = None
        self._state = ExecState.IDLE
        self._plan: MissionPlan | None = None
        self._targets: list[tuple[str, tuple[float, float, float]]] = []
        self._sent_target: str | None = None
        self._release_confirmed_by: str | None = None
        self._awaiting_reported = False
        self._failsafe = False
        logger.info("[SITL] connected to %s", self._cfg.endpoint)

    # ------------------------------------------------------------ messages

    def _pump(self) -> None:
        while True:
            msg = self._conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                self._last_heartbeat = self._clock()
            elif kind == "SYS_STATUS" and getattr(msg, "battery_remaining", -1) >= 0:
                self._battery_pct = float(msg.battery_remaining)
            elif kind == "LOCAL_POSITION_NED":
                self._pos = (float(msg.x), float(msg.y), float(msg.z))

    def _command(self, command: int, *params: float) -> None:
        p = [*params, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:7]
        self._conn.mav.command_long_send(self._conn.target_system, self._conn.target_component,
                                         command, 0, *p)

    def _set_mode(self, mode: int) -> None:
        self._command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)

    def _goto(self, ned: tuple[float, float, float]) -> None:
        self._conn.mav.set_position_target_local_ned_send(
            0, self._conn.target_system, self._conn.target_component, MAV_FRAME_LOCAL_NED,
            POSITION_ONLY_MASK, ned[0], ned[1], ned[2], 0, 0, 0, 0, 0, 0, 0, 0,
        )

    def _link_ok(self) -> bool:
        return self._clock() - self._last_heartbeat < self._cfg.link_timeout_s

    # ------------------------------------------------------- MissionExecutor

    @property
    def state(self) -> ExecState:
        return self._state

    def vehicle_status(self) -> VehicleStatus:
        self._pump()
        battery = self._battery_pct if self._battery_pct is not None else 0.0
        return VehicleStatus(battery_pct=battery, link_ok=self._link_ok())

    def start(self, plan: MissionPlan, t: float) -> None:
        if self._state not in (ExecState.IDLE, ExecState.LANDED):
            raise ExecutorError(f"vehicle busy ({self._state.value})")
        self._plan = plan
        self._targets = [(w.name, w.ned) for w in plan.waypoints[1:-1]]  # TRANSIT, HOVER, RELEASE
        self._sent_target = None
        self._release_confirmed_by = None
        self._awaiting_reported = False
        self._failsafe = False
        takeoff_alt = -plan.waypoints[0].ned[2]
        self._set_mode(COPTER_GUIDED)
        self._command(MAV_CMD_COMPONENT_ARM_DISARM, 1)
        self._command(MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, takeoff_alt)
        self._targets.insert(0, ("TAKEOFF", plan.waypoints[0].ned))
        self._sent_target = "TAKEOFF"
        self._state = ExecState.FLYING
        logger.info("[SITL] %s: GUIDED, armed, takeoff to %.1f m", plan.mission_id, takeoff_alt)

    def abort(self, reason: str, t: float) -> None:
        if self._plan is None or self._state in (ExecState.IDLE, ExecState.LANDED):
            return
        self._set_mode(COPTER_RTL)
        self._state = ExecState.RETURNING
        logger.warning("[SITL] %s aborted: %s", self._plan.mission_id, reason)

    def confirm_release(self, by: str, t: float) -> bool:
        if self._state != ExecState.AWAITING_RELEASE:
            return False
        self._release_confirmed_by = by
        return True

    def step(self, t: float) -> list[ExecutorEvent]:
        self._pump()
        if self._plan is None or self._state in (ExecState.IDLE, ExecState.LANDED):
            return []
        mid = self._plan.mission_id
        events: list[ExecutorEvent] = []

        if self._state == ExecState.RETURNING:
            if self._pos is not None and -self._pos[2] < _LANDED_ALT_M:
                self._state = ExecState.LANDED
                events.append(ExecutorEvent("completed", "[SITL] landed after RTL", mid, t))
            return events

        if not self._failsafe:
            reason = None
            if not self._link_ok():
                reason = "link lost: return to launch"
            elif self._battery_pct is not None and self._battery_pct < self._cfg.rtl_battery_pct:
                reason = f"battery {self._battery_pct:.0f}%: return to launch"
            if reason is not None:
                self._failsafe = True
                self._set_mode(COPTER_RTL)
                self._state = ExecState.RETURNING
                return [ExecutorEvent("failsafe", reason, mid, t)]

        if not self._targets:
            self._set_mode(COPTER_RTL)
            self._state = ExecState.RETURNING
            return [ExecutorEvent("progress", "mission done: RTL", mid, t)]

        name, target = self._targets[0]
        if self._state == ExecState.AWAITING_RELEASE:
            if self._release_confirmed_by is None:
                return events
            self._command(MAV_CMD_DO_SET_SERVO, self._cfg.release_servo, self._cfg.release_pwm)
            events.append(ExecutorEvent("released", f"[SITL] DO_SET_SERVO {self._cfg.release_servo}="
                                        f"{self._cfg.release_pwm} confirmed by {self._release_confirmed_by}",
                                        mid, t))
            self._targets.pop(0)
            self._state = ExecState.FLYING
            return events

        if self._sent_target != name:
            self._goto(target)
            self._sent_target = name
        if self._pos is None or math.dist(self._pos, target) > self._cfg.arrive_tolerance_m:
            return events
        if name == "RELEASE":
            if self._cfg.require_release_confirmation:
                self._state = ExecState.AWAITING_RELEASE
                if not self._awaiting_reported:
                    self._awaiting_reported = True
                    events.append(ExecutorEvent("awaiting_release", "hovering; release needs a human "
                                                "confirmation", mid, t))
                return events
            self._command(MAV_CMD_DO_SET_SERVO, self._cfg.release_servo, self._cfg.release_pwm)
            events.append(ExecutorEvent("released", "[SITL] DO_SET_SERVO sent", mid, t))
        else:
            events.append(ExecutorEvent("progress", f"reached {name}", mid, t))
        self._targets.pop(0)
        return events

    def close(self) -> None:
        close = getattr(self._conn, "close", None)
        if callable(close):
            close()
