"""Command manager: the only path from the host to the hardware.

Every command is checked, in this order: known name -> source permission -> caller identity -> parameters and
ranges (from the component config) -> physical-actuation gate -> link connected -> fresh telemetry ->
emergency-stop latch -> device faults -> conflicts with commands still in flight. Then it is sent and tracked
until the device acknowledges it, it times out or the link is lost. Every status change is recorded (history
and optional JSON-lines log).

Acknowledgement state machine (a queued or written message is NOT an executed command; only the device's ack
for that ``cmd_id`` moves it on):

    PENDING  --ack accepted-->  ACCEPTED  --ack completed-->  COMPLETED
    PENDING | ACCEPTED  --ack completed | rejected | failed-->  COMPLETED | REJECTED | FAILED
    PENDING  --no ack within command_timeout_s-->  TIMEOUT        (e-stop: re-sent first, see below)
    ACCEPTED --no completion within the commanded duration (or command_timeout_s if untimed)-->  TIMEOUT
    PENDING | ACCEPTED  --link lost-->  FAILED;   --safe/emergency stop-->  SUPERSEDED

Acks for unknown or already-final commands are ignored (logged). Command ids carry a per-process prefix, so ids
never repeat across host restarts.

Safety behaviour:
- ``emergency_stop`` is never blocked by staleness, faults or conflicts; it latches immediately on the host,
  supersedes commands in flight and is re-sent every ``emergency_stop_retry_s`` up to ``emergency_stop_retries``
  times until acknowledged (also after a failed send); on every (re)connect the service re-asserts it while the
  host latch is set;
- ``safe_stop`` also supersedes commands in flight;
- moving commands always carry a bounded duration (no continuous runs) and are refused while the latch is set,
  and for components the connected device did not announce in its ``hello``;
- only ``clear_emergency_stop`` (an authorised operator, fresh telemetry) releases the host latch, and only
  after the device acknowledges it ``completed``.
Software checks do not make a machine safe: see docs/HARDWARE_INTEGRATION.md §11 for the required hardware
protections (physical e-stop, driver enable, interlocks).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import secrets
from collections import deque
from typing import Any

from lifeguard.contracts.timeutil import utc_now
from lifeguard.hardware.config import ACTUATING, COMMAND_NAMES, HardwareConfig
from lifeguard.hardware.interface import HardwareInterface
from lifeguard.hardware.models import Ack, CommandRecord, CommandStatus, HardwareCommand, LinkState
from lifeguard.hardware.telemetry import TelemetryManager

logger = logging.getLogger(__name__)
_STOPS = frozenset({"safe_stop", "emergency_stop"})
_ACK_STATUS = {"accepted": CommandStatus.ACCEPTED, "completed": CommandStatus.COMPLETED,
               "rejected": CommandStatus.REJECTED, "failed": CommandStatus.FAILED}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class CommandManager:
    """Validates, sends and tracks hardware commands."""

    def __init__(self, config: HardwareConfig, adapter: HardwareInterface, telemetry: TelemetryManager) -> None:
        self._cfg = config
        self._adapter = adapter
        self._telemetry = telemetry
        self._history: deque[CommandRecord] = deque(maxlen=config.history_size)
        self._active: dict[str, CommandRecord] = {}
        self._counter = 0
        self._prefix = f"c-{secrets.token_hex(2)}"     # per process: ids never repeat across host restarts
        self.estop_latched = False

    # ----------------------------------------------------------------- queries
    @property
    def history(self) -> tuple[CommandRecord, ...]:
        """Copies, newest last."""
        return tuple(dataclasses.replace(r) for r in self._history)

    @property
    def active(self) -> tuple[CommandRecord, ...]:
        return tuple(dataclasses.replace(r) for r in self._active.values())

    # ----------------------------------------------------------------- requests
    def request(self, name: str, target: str | None = None, params: dict[str, Any] | None = None, *,
                source: str, by: str, now: float) -> CommandRecord:
        """Check and send one command; returns its record (REJECTED/FAILED records are returned too)."""
        params = dict(params or {})
        self._counter += 1
        command = HardwareCommand(f"{self._prefix}-{self._counter:06d}", str(name), target, {}, source, by.strip())
        record = CommandRecord(command, CommandStatus.PENDING, "", t_created=now)
        problem, wire = self._validate(command, params)
        if problem is None:
            problem = self._gate(command, now)
        if problem is not None:
            return self._finish(record, CommandStatus.REJECTED, problem, now, add=True)
        record.command = dataclasses.replace(command, params=wire)
        if command.name == "emergency_stop":
            self.estop_latched = True                     # latch first: nothing may move from now on
        if command.name in _STOPS:
            self._supersede(f"superseded by {command.name}", now)
        if self._adapter.link_state != LinkState.CONNECTED:
            detail = "device not connected"
            if command.name == "emergency_stop":
                detail += "; the firmware's link-loss failsafe and the PHYSICAL e-stop must stop the hardware"
                logger.error("EMERGENCY STOP could not be sent: device not connected")
            return self._finish(record, CommandStatus.FAILED, detail, now, add=True)
        self._history.append(record)
        return self._send(record, now)

    def emergency_stop(self, *, source: str, by: str, now: float) -> CommandRecord:
        return self.request("emergency_stop", source=source, by=by, now=now)

    # ----------------------------------------------------------------- device input
    def on_ack(self, ack: Ack, now: float) -> None:
        record = self._active.get(ack.cmd_id)
        if record is None:
            logger.info("hardware ack for unknown or finished command %s (%s)", ack.cmd_id, ack.status)
            return
        status = _ACK_STATUS[ack.status]
        if status == CommandStatus.ACCEPTED:
            if record.status == CommandStatus.ACCEPTED:
                logger.info("hardware: duplicate 'accepted' for %s ignored", ack.cmd_id)
                return
            record.status, record.detail = status, ack.detail or "accepted by the device"
            duration_ms = record.command.params.get("duration_ms")
            # Completion is due after the commanded run; an untimed command must complete right away.
            record.deadline_t = now + (duration_ms / 1000.0 if isinstance(duration_ms, (int, float)) else 0.0)
            self._log(record, now)
            return
        if status == CommandStatus.COMPLETED and record.command.name == "clear_emergency_stop":
            self.estop_latched = False
            logger.warning("emergency stop CLEARED by %s", record.command.requested_by)
        self._finish(record, status, ack.detail, now)

    def on_telemetry_estop(self, engaged: bool | None) -> None:
        """A device-side e-stop (e.g. the physical button) latches the host too."""
        if engaged and not self.estop_latched:
            self.estop_latched = True
            logger.warning("device reports its emergency stop engaged: host latch set")

    def pending(self, cmd_id: str) -> CommandRecord | None:
        """The live record of a command still waiting to go out; None once finished or superseded."""
        record = self._active.get(cmd_id)
        return record if record is not None and record.status == CommandStatus.PENDING else None

    def on_send_failed(self, cmd_id: str, now: float, detail: str = "send failed") -> None:
        """The link could not carry a queued command (the service sends outside its state lock).

        An emergency stop stays PENDING and is re-sent by ``tick`` (while retries remain and the link is up).
        """
        record = self._active.get(cmd_id)
        if record is None:
            return
        if record.command.name == "emergency_stop" and record.attempts <= self._cfg.emergency_stop_retries:
            record.detail = f"{detail}; will re-send"
            logger.error("EMERGENCY STOP send failed (attempt %d); will re-send", record.attempts)
            self._log(record, now)
            return
        self._finish(record, CommandStatus.FAILED, detail, now)

    def has_active_actuation(self) -> bool:
        return any(r.command.name in ACTUATING for r in self._active.values())

    def on_link_lost(self, now: float) -> None:
        for record in list(self._active.values()):
            self._finish(record, CommandStatus.FAILED, "link lost before completion (firmware failsafe applies)", now)

    def tick(self, now: float) -> None:
        """Timeouts and emergency-stop re-sends."""
        timeout = self._cfg.command_timeout_s
        for record in list(self._active.values()):
            estop = record.command.name == "emergency_stop"
            wait = self._cfg.emergency_stop_retry_s if estop else timeout
            if record.status == CommandStatus.PENDING and record.t_sent is not None and now - record.t_sent > wait:
                if (estop and record.attempts <= self._cfg.emergency_stop_retries
                        and self._adapter.link_state == LinkState.CONNECTED):
                    logger.error("emergency stop not acknowledged; re-sending (attempt %d)", record.attempts + 1)
                    self._send(record, now)
                    continue
                if estop:
                    detail = (f"no acknowledgement after {record.attempts} attempt(s); the host latch stays set and "
                              "the e-stop is re-sent on reconnect - use the PHYSICAL e-stop")
                    logger.error("EMERGENCY STOP not acknowledged: %s", detail)
                else:
                    detail = f"no acknowledgement within {timeout:.1f} s"
                self._finish(record, CommandStatus.TIMEOUT, detail, now)
            elif (record.status == CommandStatus.ACCEPTED and record.deadline_t is not None
                  and now > record.deadline_t + timeout):
                self._finish(record, CommandStatus.TIMEOUT, "no completion reported after the commanded duration", now)

    # ----------------------------------------------------------------- internals
    def _validate(self, command: HardwareCommand, params: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        name, target = command.name, command.target
        if name not in COMMAND_NAMES:
            return f"unknown command {name!r}", {}
        if name not in self._cfg.permissions.get(command.source, ()):  # type: ignore[call-overload]
            return f"source {command.source!r} is not permitted to send {name!r}", {}
        if command.source != "system" and not command.requested_by:
            return "a named operator is required", {}
        comps = self._cfg.components
        if name == "set_motor":
            spec = comps.motors.get(target or "")
            if spec is None:
                return f"unknown motor {target!r}", {}
            if set(params) - {"speed", "duration_s"}:
                return f"unexpected parameters {sorted(set(params) - {'speed', 'duration_s'})}", {}
            speed, duration = _number(params.get("speed")), _number(params.get("duration_s"))
            if speed is None or abs(speed) > spec.max_speed:
                return f"speed must be a number within ±{spec.max_speed}", {}
            if duration is None or not 0 < duration <= spec.max_duration_s:
                return f"duration_s must be within (0, {spec.max_duration_s}] (no continuous runs)", {}
            return None, {"speed": round(speed, 4), "duration_ms": round(duration * 1000)}
        if name == "set_actuator":
            spec_a = comps.actuators.get(target or "")
            if spec_a is None:
                return f"unknown actuator {target!r}", {}
            if set(params) - {"state", "duration_s"}:
                return f"unexpected parameters {sorted(set(params) - {'state', 'duration_s'})}", {}
            state = params.get("state")
            if state not in spec_a.states:
                return f"state must be one of {list(spec_a.states)}", {}
            if state == spec_a.safe_state:
                return None, {"state": state}
            duration = _number(params.get("duration_s"))
            if duration is None or not 0 < duration <= spec_a.max_duration_s:
                return f"duration_s must be within (0, {spec_a.max_duration_s}] for a non-safe state", {}
            return None, {"state": state, "duration_ms": round(duration * 1000)}
        if target is not None or params:
            return f"{name} takes no target or parameters", {}
        return None, {}

    def _gate(self, command: HardwareCommand, now: float) -> str | None:
        name = command.name
        if name in ACTUATING:
            if self._adapter.kind == "esp32" and not self._cfg.allow_physical_actuation:
                return "physical actuation is disabled (hardware.allow_physical_actuation: false)"
            if self._adapter.link_state != LinkState.CONNECTED:
                return "device not connected"
            if self._telemetry.is_stale(now):
                return "telemetry is stale or missing; refusing to move anything"
            if self.estop_latched:
                return "emergency stop is latched; an operator must clear it first"
            if self._telemetry.faults:
                return f"device reports faults: {', '.join(self._telemetry.faults)}"
            announced = getattr(self._adapter, "announced", None)
            kind = "motors" if name == "set_motor" else "actuators"
            if announced is not None and command.target not in announced.get(kind, ()):
                return f"the connected device did not announce {kind[:-1]} {command.target!r} in its hello"
            for other in self._active.values():
                if other.command.target == command.target:
                    return f"conflict: {other.command.cmd_id} ({other.command.name}) is still in flight on {command.target}"
        elif name not in _STOPS:
            if self._adapter.link_state != LinkState.CONNECTED:
                return "device not connected"
            if name == "clear_emergency_stop":
                if not self.estop_latched:
                    return "the emergency stop is not latched; nothing to clear"
                if self._telemetry.is_stale(now):
                    return "telemetry is stale or missing; the device state must be visible before clearing"
            for other in self._active.values():
                if other.command.name == name:
                    return f"duplicate: {other.command.cmd_id} ({name}) is still in flight"
        return None

    def _supersede(self, detail: str, now: float) -> None:
        for record in list(self._active.values()):
            if record.command.name not in _STOPS:
                self._finish(record, CommandStatus.SUPERSEDED, detail, now)

    def _send(self, record: CommandRecord, now: float) -> CommandRecord:
        record.attempts += 1
        record.t_sent = now
        record.status = CommandStatus.PENDING
        self._active[record.command.cmd_id] = record
        if not self._adapter.send_command(record.command, now):
            return self._finish(record, CommandStatus.FAILED, "send failed", now)
        level = logging.WARNING if record.command.name in (*ACTUATING, *_STOPS) else logging.INFO
        logger.log(level, "hardware command %s %s %s %s sent (source %s, by %s)", record.command.cmd_id,
                   record.command.name, record.command.target or "", record.command.params, record.command.source,
                   record.command.requested_by or "-")
        self._log(record, now)
        return record

    def _finish(self, record: CommandRecord, status: CommandStatus, detail: str, now: float,
                add: bool = False) -> CommandRecord:
        record.status, record.detail, record.t_done = status, detail, now
        self._active.pop(record.command.cmd_id, None)
        if add:
            self._history.append(record)
        if status in (CommandStatus.REJECTED, CommandStatus.FAILED, CommandStatus.TIMEOUT):
            logger.warning("hardware command %s %s %s: %s", record.command.cmd_id, record.command.name,
                           status.value, detail)
        self._log(record, now)
        return record

    def _log(self, record: CommandRecord, now: float) -> None:
        path = self._cfg.command_log
        if path is None:
            return
        line = {"t_utc": utc_now(), "t": round(now, 3), "cmd_id": record.command.cmd_id,
                "name": record.command.name, "target": record.command.target, "params": record.command.params,
                "source": record.command.source, "by": record.command.requested_by,
                "status": record.status.value, "detail": record.detail, "attempts": record.attempts}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError as err:
            logger.warning("hardware command log not written (%s): %s", path, err)
