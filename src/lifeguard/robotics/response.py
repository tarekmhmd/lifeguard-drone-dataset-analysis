"""Drone response platform (SIMULATION ONLY): planner + manager + executor behind ``ResponsePlatform``.

Wiring: human-confirmed ``DispatchCommand`` -> ``MissionManager`` (one drone, safety checks) ->
``MissionExecutor`` (``SimulatedExecutor`` or ``MavlinkExecutor`` against local SITL). Dismissal or closure
of the incident aborts with RTL; the release waits for a human confirmation when configured.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from lifeguard.contracts.interfaces import PoolGeometry
from lifeguard.contracts.types import DispatchCommand, FrameResult, RiskEvent
from lifeguard.robotics.executors.base import ExecutorError, MissionExecutor
from lifeguard.robotics.executors.config import DroneConfig
from lifeguard.robotics.mission.config import MissionConfig
from lifeguard.robotics.mission.planner import MissionManager, MissionPlan, MissionPlanner
from lifeguard.system.response import OutcomeKind, ResponseOutcome, ResponsePlatform

logger = logging.getLogger(__name__)

_EVENT_KINDS = {
    "progress": OutcomeKind.PROGRESS,
    "awaiting_release": OutcomeKind.AWAITING_CONFIRMATION,
    "released": OutcomeKind.PROGRESS,
    "failsafe": OutcomeKind.ABORTED,
    "aborted": OutcomeKind.ABORTED,
    "completed": OutcomeKind.COMPLETED,
}


def build_executor(config: DroneConfig) -> MissionExecutor:
    """Create the configured executor. Raises ExecutorError if SITL is unreachable or pymavlink missing."""
    if config.executor == "simulated":
        from lifeguard.robotics.executors.simulated import SimulatedExecutor

        return SimulatedExecutor(config)
    from lifeguard.robotics.executors.mavlink import MavlinkExecutor

    return MavlinkExecutor(config)


class DroneResponse(ResponsePlatform):
    """Simulated drone response."""

    def __init__(
        self,
        geometry: PoolGeometry,
        mission: MissionConfig | None = None,
        executor: MissionExecutor | None = None,
        drone: DroneConfig | None = None,
    ) -> None:
        self._manager = MissionManager(MissionPlanner(geometry, mission))
        self._executor = executor or build_executor(drone or DroneConfig())
        self._kind = type(self._executor).__name__

    @property
    def name(self) -> str:
        return f"drone[{self._kind}]"

    @property
    def executor(self) -> MissionExecutor:
        return self._executor

    @property
    def active_mission(self) -> MissionPlan | None:
        """The accepted plan currently being flown, if any."""
        return self._manager.active

    def dispatch(
        self, command: DispatchCommand, t: float, people: Sequence[tuple[float, float]] = ()
    ) -> ResponseOutcome:
        result = self._manager.dispatch(command, self._executor.vehicle_status(), people)
        if not isinstance(result, MissionPlan):
            return ResponseOutcome(OutcomeKind.REJECTED, self.name,
                                   f"drone not dispatched ({result.reason}): lifeguards respond in person",
                                   event_id=command.event_id)
        try:
            self._executor.start(result, t)
        except ExecutorError as err:
            self._manager.complete(result.mission_id)
            return ResponseOutcome(OutcomeKind.FAILED, self.name,
                                   f"drone could not start ({err}): lifeguards respond in person",
                                   event_id=command.event_id, mission_id=result.mission_id)
        return ResponseOutcome(OutcomeKind.ACCEPTED, self.name,
                               f"[SIMULATION] {result.mission_id} started, hover at {result.hover_pool_xy}",
                               event_id=command.event_id, mission_id=result.mission_id)

    def on_frame(self, frame: FrameResult) -> None:
        self._manager.on_frame(frame)

    def on_risk_event(self, event: RiskEvent) -> list[ResponseOutcome]:
        abort = self._manager.on_risk_event(event)
        if abort is None:
            return []
        self._executor.abort(abort.reason, event.t_capture)
        return [ResponseOutcome(OutcomeKind.ABORTED, self.name, f"{abort.mission_id}: {abort.action} "
                                f"({abort.reason})", event_id=event.event_id, mission_id=abort.mission_id)]

    def step(self, t: float) -> list[ResponseOutcome]:
        outcomes: list[ResponseOutcome] = []
        for ev in self._executor.step(t):
            outcomes.append(ResponseOutcome(_EVENT_KINDS.get(ev.kind, OutcomeKind.PROGRESS), self.name,
                                            f"{ev.kind}: {ev.detail}", mission_id=ev.mission_id))
            if ev.kind == "completed":
                self._manager.complete(ev.mission_id)
        return outcomes

    def confirm_release(self, by: str) -> ResponseOutcome | None:
        active = self._manager.active
        if active is None or not self._executor.confirm_release(by, 0.0):
            return None
        return ResponseOutcome(OutcomeKind.PROGRESS, self.name, f"release confirmed by {by}",
                               mission_id=active.mission_id)

    def status(self) -> dict[str, str]:
        active = self._manager.active
        return {
            "platform": self.name,
            "state": self._executor.state.value,
            "mission": active.mission_id if active else "-",
        }

    def close(self) -> None:
        self._executor.close()
