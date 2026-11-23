"""Response-platform interface: what happens after a lifeguard CONFIRMs an emergency.

The AI core (perception, tracking, behaviour, risk engine) and the alert service never depend on how the
response is carried out. A platform receives the human-confirmed ``DispatchCommand`` and reports outcomes.
The final platform is an open supervisor decision, so the runtime selects one by configuration:

| Platform | Module | Kind |
| --- | --- | --- |
| ``ManualResponse`` (``platform: none``) | this module | REAL: fixed-camera / alert-only deployment; lifeguards respond in person |
| ``DroneResponse`` (``platform: drone``) | ``lifeguard.robotics.response`` | SIMULATION ONLY: mission planner + simulated or SITL executor |

Extension point: a future fixed-camera actuator (e.g. a poolside siren zone, a rescue-tube dispenser or a
pager to the nearest lifeguard) implements ``ResponsePlatform``; nothing upstream changes.
"""
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from lifeguard.contracts.types import DispatchCommand, FrameResult, RiskEvent


class OutcomeKind(str, enum.Enum):
    """What a platform did with a dispatch."""

    MANUAL = "manual"          # no automated platform: humans respond
    ACCEPTED = "accepted"      # automated response started
    REJECTED = "rejected"      # refused (busy, unsafe, preconditions not met): humans respond
    PROGRESS = "progress"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(frozen=True)
class ResponseOutcome:
    """One report from a response platform."""

    kind: OutcomeKind
    platform: str
    detail: str
    event_id: str | None = None
    mission_id: str | None = None


class ResponsePlatform(ABC):
    """Carries out a human-confirmed dispatch."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short platform name for logs and the dashboard."""

    @abstractmethod
    def dispatch(
        self, command: DispatchCommand, t: float, people: Sequence[tuple[float, float]] = ()
    ) -> ResponseOutcome:
        """Act on a human-confirmed DispatchCommand; ``people`` are known pool positions of others."""

    def on_frame(self, frame: FrameResult) -> None:
        """Optional: follow track re-associations."""

    def on_risk_event(self, event: RiskEvent) -> list[ResponseOutcome]:
        """Optional: react to dismissals or closures (e.g. abort a mission)."""
        return []

    def step(self, t: float) -> list[ResponseOutcome]:
        """Optional: advance asynchronous work and report progress."""
        return []

    def confirm_release(self, by: str) -> ResponseOutcome | None:
        """Optional: human confirmation of a flotation release (platforms that have one)."""
        return None

    def status(self) -> dict[str, str]:
        """Optional: key/value status for the dashboard."""
        return {"platform": self.name}

    def close(self) -> None:
        """Optional: release resources."""


class ManualResponse(ResponsePlatform):
    """No automated platform (fixed-camera / alert-only deployment): every dispatch is a human response."""

    @property
    def name(self) -> str:
        return "manual"

    def dispatch(
        self, command: DispatchCommand, t: float, people: Sequence[tuple[float, float]] = ()
    ) -> ResponseOutcome:
        x, y = command.target_pool_xy
        return ResponseOutcome(
            OutcomeKind.MANUAL, self.name,
            f"No automated response platform configured: lifeguards respond in person to "
            f"({x:.1f} m, {y:.1f} m), confirmed by {command.confirmed_by}.",
            event_id=command.event_id,
        )
