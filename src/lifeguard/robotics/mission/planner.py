"""Mission planner and manager: human-confirmed DispatchCommand -> safe waypoint plan (roadmap P6-1).

REAL planning logic, SIMULATION ONLY. It produces an abstract plan (waypoints in pool metres and in the
drone's local NED frame); a separate executor sends it to ArduPilot SITL over MAVLink (roadmap P6-2, not
built). Nothing here talks to hardware.

Safety rules (Blueprint M10, MEMBER_2 M2-10, decision D-03):
- input is a ``DispatchCommand``, which by contract always carries a human ``confirmed_by``;
- the target must be inside the pool; the vehicle must report battery >= minimum and a healthy link;
- the drone hovers ``standoff_m`` from the swimmer, never over them, and never within ``standoff_m`` of any
  other known person; the hover point and the launch point must lie inside the geofence (pool rectangle
  grown by ``fence_margin_m``); no waypoint exceeds ``max_altitude_m``;
- one drone, one mission: a second dispatch while a mission is active is rejected (respond manually);
- a dismissal or incident close for the mission's swimmer aborts the mission with return-to-launch;
- the payload release is a planned step for the executor; it is simulated, never triggered by the AI alone.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass

from lifeguard.contracts.enums import RiskLevel, RiskReason
from lifeguard.contracts.interfaces import PoolGeometry
from lifeguard.contracts.types import DispatchCommand, FrameResult, RiskEvent, make_track_uid
from lifeguard.robotics.mission.config import MissionConfig

logger = logging.getLogger(__name__)

_N_DIRECTIONS = 16
_ABORT_REASONS = frozenset({RiskReason.LIFEGUARD_DISMISSED, RiskReason.INCIDENT_CLOSED})


@dataclass(frozen=True)
class VehicleStatus:
    """Pre-flight state reported by the executor (simulated vehicle)."""

    battery_pct: float
    link_ok: bool


@dataclass(frozen=True)
class Waypoint:
    """One mission step. ``ned`` is (north, east, down) metres from the drone's home frame origin."""

    name: str
    pool_xy: tuple[float, float]
    altitude_m: float
    ned: tuple[float, float, float]


@dataclass(frozen=True)
class MissionPlan:
    """An accepted mission. Steps: TAKEOFF, TRANSIT, HOVER, RELEASE (simulated), RTL."""

    mission_id: str
    event_id: str
    track_uid: str
    confirmed_by: str
    target_pool_xy: tuple[float, float]
    hover_pool_xy: tuple[float, float]
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class MissionRejected:
    """A dispatch the planner refused; the lifeguards must respond without the drone."""

    event_id: str
    reason: str


@dataclass(frozen=True)
class AbortCommand:
    """Stop the active mission and return to launch."""

    mission_id: str
    reason: str
    action: str = "RTL"


class MissionPlanner:
    """Pure geometry: plan a mission for one dispatch."""

    def __init__(self, geometry: PoolGeometry, config: MissionConfig | None = None) -> None:
        self._geometry = geometry
        self._config = config or MissionConfig()
        if not self.in_fence(self._config.home_pool_xy):
            raise ValueError(f"home point {self._config.home_pool_xy} is outside the geofence")

    def in_fence(self, xy: tuple[float, float]) -> bool:
        """Inside the pool rectangle grown by ``fence_margin_m`` (boundary included)."""
        m = self._config.fence_margin_m
        return (-m <= xy[0] <= self._geometry.length_m + m) and (-m <= xy[1] <= self._geometry.width_m + m)

    def hover_point(
        self, target: tuple[float, float], people: Iterable[tuple[float, float]] = ()
    ) -> tuple[float, float] | None:
        """A point ``standoff_m`` from the target, inside the fence and clear of other people.

        Candidates start with the direction towards the launch point (shortest approach) and then rotate
        in 22.5 degree steps alternating left/right; the first valid candidate wins. None if none fits.
        """
        cfg = self._config
        others = [p for p in people if math.dist(p, target) > 1e-6]
        hx, hy = cfg.home_pool_xy
        base = math.atan2(hy - target[1], hx - target[0])
        step = 2.0 * math.pi / _N_DIRECTIONS
        order = [0] + [s * k for k in range(1, _N_DIRECTIONS // 2 + 1) for s in (1, -1)]
        for k in order[:_N_DIRECTIONS]:
            angle = base + k * step
            p = (target[0] + cfg.standoff_m * math.cos(angle), target[1] + cfg.standoff_m * math.sin(angle))
            if not self.in_fence(p):
                continue
            if any(math.dist(p, o) < cfg.standoff_m for o in others):
                continue
            return (round(p[0], 3), round(p[1], 3))
        return None

    def plan(
        self,
        command: DispatchCommand,
        vehicle: VehicleStatus,
        mission_id: str,
        people: Iterable[tuple[float, float]] = (),
    ) -> MissionPlan | MissionRejected:
        """Plan the mission, or explain why it cannot be flown."""
        cfg = self._config
        target = command.target_pool_xy
        if not self._geometry.contains(*target):
            return MissionRejected(command.event_id, f"target {target} is outside the pool")
        if not vehicle.link_ok:
            return MissionRejected(command.event_id, "vehicle link is down")
        if vehicle.battery_pct < cfg.min_battery_pct:
            return MissionRejected(
                command.event_id, f"battery {vehicle.battery_pct:.0f}% below {cfg.min_battery_pct:.0f}%"
            )
        hover = self.hover_point(target, people)
        if hover is None:
            return MissionRejected(command.event_id, "no safe hover point inside the geofence")

        home = cfg.home_pool_xy

        def wp(name: str, xy: tuple[float, float], alt: float) -> Waypoint:
            return Waypoint(name, xy, alt, self._geometry.to_local_ned(xy[0], xy[1], alt))

        waypoints = (
            wp("TAKEOFF", home, cfg.cruise_altitude_m),
            wp("TRANSIT", hover, cfg.cruise_altitude_m),
            wp("HOVER", hover, cfg.hover_altitude_m),
            wp("RELEASE", hover, cfg.hover_altitude_m),
            wp("RTL", home, cfg.cruise_altitude_m),
        )
        return MissionPlan(
            mission_id=mission_id,
            event_id=command.event_id,
            track_uid=command.track_uid,
            confirmed_by=command.confirmed_by,
            target_pool_xy=target,
            hover_pool_xy=hover,
            waypoints=waypoints,
        )


class MissionManager:
    """One drone: accepts dispatches, tracks the active mission, aborts it on dismissal or close."""

    def __init__(self, planner: MissionPlanner) -> None:
        self._planner = planner
        self._active: MissionPlan | None = None
        self._active_uid: str | None = None
        self._counter = 0

    @property
    def active(self) -> MissionPlan | None:
        return self._active

    def dispatch(
        self,
        command: DispatchCommand,
        vehicle: VehicleStatus,
        people: Iterable[tuple[float, float]] = (),
    ) -> MissionPlan | MissionRejected:
        if self._active is not None:
            return MissionRejected(
                command.event_id, f"drone busy with mission {self._active.mission_id}; respond manually"
            )
        self._counter += 1
        result = self._planner.plan(command, vehicle, f"mission-{self._counter:04d}", people)
        if isinstance(result, MissionPlan):
            self._active = result
            self._active_uid = result.track_uid
            logger.info("%s planned for %s (confirmed by %s), hover at %s", result.mission_id,
                        result.track_uid, result.confirmed_by, result.hover_pool_xy)
        else:
            logger.warning("dispatch %s rejected: %s", command.event_id, result.reason)
        return result

    def on_frame(self, frame: FrameResult) -> None:
        """Follow the swimmer across re-associated track IDs (``merged_from``)."""
        if self._active_uid is None:
            return
        for track in frame.tracks:
            for old_id in track.merged_from:
                if make_track_uid(frame.cam_id, frame.session_id, old_id) == self._active_uid:
                    self._active_uid = track.track_uid

    def on_risk_event(self, event: RiskEvent) -> AbortCommand | None:
        """Abort when the mission's incident is dismissed or closed."""
        if self._active is None or event.track_uid != self._active_uid:
            return None
        if event.to_level in (RiskLevel.RESOLVED, RiskLevel.NORMAL) and event.reason in _ABORT_REASONS:
            abort = AbortCommand(self._active.mission_id, f"{event.reason.value} ({event.event_id})")
            logger.warning("%s aborted: %s", abort.mission_id, abort.reason)
            self._active = None
            self._active_uid = None
            return abort
        return None

    def complete(self, mission_id: str) -> bool:
        """Mark the active mission finished (executor reports landing). False if it was not active."""
        if self._active is None or self._active.mission_id != mission_id:
            return False
        self._active = None
        self._active_uid = None
        return True
