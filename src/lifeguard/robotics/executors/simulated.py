"""Pure-Python kinematic drone simulator (SIMULATION ONLY, no ArduPilot, no physics).

Moves a point at constant horizontal/vertical speed through the plan's NED waypoints, drains a simulated
battery and applies the same failsafes the real system needs: link loss -> RTL, low battery -> RTL,
abort -> RTL. The release is a logged event; nothing is released anywhere. Used for tests, demos and for
the runtime when no SITL is available. It says nothing about real flight performance.
"""
from __future__ import annotations

import logging
import math

from lifeguard.robotics.executors.base import ExecState, ExecutorError, ExecutorEvent, MissionExecutor
from lifeguard.robotics.executors.config import DroneConfig
from lifeguard.robotics.mission.planner import MissionPlan, VehicleStatus

logger = logging.getLogger(__name__)

Vec = tuple[float, float, float]


def _move(pos: Vec, target: Vec, dt: float, h_speed: float, v_speed: float) -> Vec:
    dn, de, dd = (target[0] - pos[0], target[1] - pos[1], target[2] - pos[2])
    horizontal = math.hypot(dn, de)
    h_step = min(horizontal, h_speed * dt)
    n = pos[0] + (dn / horizontal * h_step if horizontal > 0 else 0.0)
    e = pos[1] + (de / horizontal * h_step if horizontal > 0 else 0.0)
    d = pos[2] + math.copysign(min(abs(dd), v_speed * dt), dd)
    return (n, e, d)


class SimulatedExecutor(MissionExecutor):
    """Deterministic simulated vehicle driven by the caller's clock."""

    def __init__(self, config: DroneConfig | None = None, battery_pct: float = 100.0) -> None:
        self._cfg = config or DroneConfig()
        self.battery_pct = battery_pct
        self._link_ok = True
        self._link_lost_since: float | None = None
        self._state = ExecState.IDLE
        self._plan: MissionPlan | None = None
        self._targets: list[tuple[str, Vec]] = []
        self._pos: Vec = (0.0, 0.0, 0.0)
        self._home: Vec = (0.0, 0.0, 0.0)
        self._last_t: float | None = None
        self._pending: list[ExecutorEvent] = []
        self._release_confirmed_by: str | None = None
        self._awaiting_reported = False
        self.released = False

    # ----------------------------------------------------- simulation controls

    def set_link(self, ok: bool, t: float) -> None:
        """Inject a link loss (False) or recovery (True)."""
        self._link_ok = ok
        self._link_lost_since = None if ok else t

    @property
    def position_ned(self) -> Vec:
        return self._pos

    # ------------------------------------------------------- MissionExecutor

    @property
    def state(self) -> ExecState:
        return self._state

    def vehicle_status(self) -> VehicleStatus:
        return VehicleStatus(battery_pct=self.battery_pct, link_ok=self._link_ok)

    def start(self, plan: MissionPlan, t: float) -> None:
        if self._state not in (ExecState.IDLE, ExecState.LANDED):
            raise ExecutorError(f"vehicle busy ({self._state.value})")
        if not self._link_ok:
            raise ExecutorError("link down")
        self._plan = plan
        home = plan.waypoints[0].ned
        self._home = (home[0], home[1], 0.0)
        self._pos = self._home
        self._targets = [(w.name, w.ned) for w in plan.waypoints]
        self._targets.append(("LAND", self._home))
        self._state = ExecState.FLYING
        self._last_t = t
        self._release_confirmed_by = None
        self._awaiting_reported = False
        self.released = False
        logger.info("[SIMULATED] %s started", plan.mission_id)

    def abort(self, reason: str, t: float) -> None:
        if self._plan is None or self._state in (ExecState.IDLE, ExecState.LANDED):
            return
        self._rtl("aborted", reason, t)

    def confirm_release(self, by: str, t: float) -> bool:
        if self._state != ExecState.AWAITING_RELEASE:
            return False
        self._release_confirmed_by = by
        return True

    def step(self, t: float) -> list[ExecutorEvent]:
        events, self._pending = self._pending, []
        if self._plan is None or self._state in (ExecState.IDLE, ExecState.LANDED):
            return events
        dt = max(0.0, t - (self._last_t if self._last_t is not None else t))
        self._last_t = t
        self.battery_pct = max(0.0, self.battery_pct - self._cfg.sim_drain_pct_per_s * dt)
        mid = self._plan.mission_id

        if self._state != ExecState.RETURNING:
            if (not self._link_ok and self._link_lost_since is not None
                    and t - self._link_lost_since >= self._cfg.link_timeout_s):
                self._rtl("failsafe", "link lost: return to launch", t)
            elif self.battery_pct < self._cfg.rtl_battery_pct:
                self._rtl("failsafe", f"battery {self.battery_pct:.0f}%: return to launch", t)
            events += self._pending
            self._pending = []

        if not self._targets:
            return events
        name, target = self._targets[0]
        if self._state == ExecState.AWAITING_RELEASE:
            if self._release_confirmed_by is None:
                return events
            self.released = True
            events.append(ExecutorEvent("released", f"[SIMULATED] flotation release confirmed by "
                                        f"{self._release_confirmed_by}", mid, t))
            self._targets.pop(0)
            self._state = ExecState.FLYING
            return events

        self._pos = _move(self._pos, target, dt, self._cfg.sim_speed_mps, self._cfg.sim_vertical_mps)
        if math.dist(self._pos, target) > self._cfg.arrive_tolerance_m:
            return events
        self._pos = target
        if name == "RELEASE":
            if self._cfg.require_release_confirmation:
                self._state = ExecState.AWAITING_RELEASE
                if not self._awaiting_reported:
                    self._awaiting_reported = True
                    events.append(ExecutorEvent("awaiting_release", "hovering; release needs a human "
                                                "confirmation", mid, t))
                return events
            self.released = True
            events.append(ExecutorEvent("released", "[SIMULATED] flotation release", mid, t))
        elif name == "LAND":
            self._state = ExecState.LANDED
            events.append(ExecutorEvent("completed", "[SIMULATED] landed at launch point", mid, t))
        else:
            events.append(ExecutorEvent("progress", f"reached {name}", mid, t))
        self._targets.pop(0)
        return events

    def _rtl(self, kind: str, reason: str, t: float) -> None:
        assert self._plan is not None
        cruise = self._plan.waypoints[0].ned[2]
        self._targets = [("RTL", (self._home[0], self._home[1], cruise)), ("LAND", self._home)]
        self._state = ExecState.RETURNING
        self._pending.append(ExecutorEvent(kind, reason, self._plan.mission_id, t))
        logger.warning("[SIMULATED] %s %s: %s", self._plan.mission_id, kind, reason)
