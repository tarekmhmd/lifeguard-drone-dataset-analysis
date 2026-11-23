"""Mission executor interface: flies an accepted ``MissionPlan`` (SIMULATION ONLY).

Implementations: ``SimulatedExecutor`` (pure-Python kinematic model, no external software) and
``MavlinkExecutor`` (ArduPilot SITL over MAVLink). There is no executor for physical hardware.

Lifecycle: ``start(plan)`` -> repeated ``step(t)`` -> COMPLETED / ABORTED / FAILSAFE.
The flotation release step waits for ``confirm_release(by)`` when ``require_release_confirmation`` is set.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from lifeguard.robotics.mission.planner import MissionPlan, VehicleStatus


class ExecutorError(Exception):
    """The executor could not start or talk to the (simulated) vehicle."""


class ExecState(str, enum.Enum):
    IDLE = "idle"
    FLYING = "flying"
    AWAITING_RELEASE = "awaiting_release"
    RETURNING = "returning"
    LANDED = "landed"


@dataclass(frozen=True)
class ExecutorEvent:
    """Progress report. ``kind``: progress, awaiting_release, released, failsafe, aborted, completed."""

    kind: str
    detail: str
    mission_id: str
    t: float


class MissionExecutor(ABC):
    """Runs one mission at a time."""

    @abstractmethod
    def vehicle_status(self) -> VehicleStatus:
        """Pre-flight battery and link state."""

    @abstractmethod
    def start(self, plan: MissionPlan, t: float) -> None:
        """Begin flying ``plan``. Raises ExecutorError if the vehicle cannot start."""

    @abstractmethod
    def step(self, t: float) -> list[ExecutorEvent]:
        """Advance the mission to time ``t`` and report what happened."""

    @abstractmethod
    def abort(self, reason: str, t: float) -> None:
        """Return to launch now (dismissal, failsafe, operator)."""

    @abstractmethod
    def confirm_release(self, by: str, t: float) -> bool:
        """Human confirmation of the (simulated) release; False if no release is pending."""

    @property
    @abstractmethod
    def state(self) -> ExecState:
        """Current execution state."""

    def close(self) -> None:
        """Release resources."""
