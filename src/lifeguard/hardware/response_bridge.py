"""Response bridge: the only route from an AI incident to hardware.

``HardwareLinkedResponse`` wraps the configured ``ResponsePlatform`` (manual or simulated drone). It forwards
everything unchanged and, only when a lifeguard has CONFIRMED a dispatch (``DispatchCommand.confirmed_by``),
asks the hardware service to send the commands listed in ``hardware.on_dispatch``. Those commands pass the
same checks as any other (permission ``response``, ranges, physical-actuation gate, link, staleness, latch).

Nothing in perception, tracking or the risk engine can command hardware: there is no other path.
``on_dispatch`` is empty by default, so the runtime is unchanged unless it is configured.
"""
from __future__ import annotations

from collections.abc import Sequence

from lifeguard.contracts.types import DispatchCommand, FrameResult, RiskEvent
from lifeguard.hardware.config import CommandSpec
from lifeguard.hardware.models import CommandStatus
from lifeguard.hardware.service import HardwareService
from lifeguard.system.response import ResponseOutcome, ResponsePlatform


class HardwareLinkedResponse(ResponsePlatform):
    """A response platform plus configured hardware commands on a human-confirmed dispatch."""

    def __init__(self, inner: ResponsePlatform, service: HardwareService, commands: Sequence[CommandSpec]) -> None:
        self._inner = inner
        self._service = service
        self._commands = tuple(commands)
        self.last_results: list[str] = []

    @property
    def name(self) -> str:
        return f"{self._inner.name}+hardware"

    def dispatch(self, command: DispatchCommand, t: float,
                 people: Sequence[tuple[float, float]] = ()) -> ResponseOutcome:
        outcome = self._inner.dispatch(command, t, people)
        results = []
        for spec in self._commands:
            record = self._service.request(spec.name, spec.target, dict(spec.params), source="response",
                                           by=command.confirmed_by)
            results.append(f"{spec.name}{'/' + spec.target if spec.target else ''}: {record.status.value}"
                           + (f" ({record.detail})" if record.status != CommandStatus.PENDING else ""))
        self.last_results = results
        if not results:
            return outcome
        return ResponseOutcome(outcome.kind, outcome.platform, f"{outcome.detail}; hardware: {'; '.join(results)}",
                               outcome.event_id, outcome.mission_id)

    def on_frame(self, frame: FrameResult) -> None:
        self._inner.on_frame(frame)

    def on_risk_event(self, event: RiskEvent) -> list[ResponseOutcome]:
        return self._inner.on_risk_event(event)

    def step(self, t: float) -> list[ResponseOutcome]:
        return self._inner.step(t)

    def confirm_release(self, by: str) -> ResponseOutcome | None:
        return self._inner.confirm_release(by)

    def status(self) -> dict[str, str]:
        status = dict(self._inner.status())
        status["hardware_on_dispatch"] = ", ".join(s.name for s in self._commands) or "none"
        if self.last_results:
            status["hardware_last_dispatch"] = "; ".join(self.last_results)
        return status

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
