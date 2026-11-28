"""Data types shared by the hardware adapters, managers, service and dashboard.

Unknown or unavailable values are ``None`` with an explicit status; nothing is filled in.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class LinkState(str, enum.Enum):
    """Connection state of the hardware link."""

    DISCONNECTED = "disconnected"   # no device connected (or the link timed out)
    CONNECTING = "connecting"       # transport open, waiting for a valid "hello"
    CONNECTED = "connected"         # identified device, messages flowing
    FAULT = "fault"                 # the adapter itself failed (e.g. port in use); the AI pipeline is unaffected


class CommandStatus(str, enum.Enum):
    """Life cycle of one command."""

    PENDING = "pending"         # sent, waiting for the device's acknowledgement
    ACCEPTED = "accepted"       # device accepted it and is executing (e.g. a timed motor run)
    COMPLETED = "completed"     # device reports it done
    REJECTED = "rejected"       # refused by the host-side checks or by the device
    FAILED = "failed"           # could not be sent, or the device reports failure, or the link was lost
    TIMEOUT = "timeout"         # no acknowledgement / completion in time
    SUPERSEDED = "superseded"   # replaced by a safe-stop or emergency-stop before it finished

    @property
    def final(self) -> bool:
        return self not in (CommandStatus.PENDING, CommandStatus.ACCEPTED)


@dataclass(frozen=True)
class SensorReading:
    """``status``: ok | unavailable | out_of_range | fault | unconfigured."""

    value: float | None
    unit: str | None
    status: str


@dataclass(frozen=True)
class MotorState:
    """``state``: stopped | running | fault | unknown. ``speed`` normalised to [-1, 1], None if not reported."""

    state: str
    speed: float | None


@dataclass(frozen=True)
class ActuatorState:
    """``state`` is one of the configured states, or None if not reported."""

    state: str | None


@dataclass(frozen=True)
class PowerStatus:
    """Only what the device reports; both None if it has no power sensing."""

    voltage_v: float | None
    battery_pct: float | None


@dataclass(frozen=True)
class HardwareTelemetry:
    """One telemetry report, normalised. ``t_host`` is the host's monotonic receive time."""

    device_id: str
    seq: int
    t_host: float
    t_device_ms: int | None
    source: str                                   # "simulation" | "esp32"
    sensors: dict[str, SensorReading]
    motors: dict[str, MotorState]
    actuators: dict[str, ActuatorState]
    faults: tuple[str, ...]
    estop_engaged: bool | None
    power: PowerStatus | None


@dataclass(frozen=True)
class DeviceInfo:
    """What the device said about itself in "hello"."""

    device_id: str
    firmware: str | None
    proto: int
    components: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Ack:
    """Device acknowledgement. ``status``: accepted | completed | rejected | failed."""

    cmd_id: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class HardwareCommand:
    """A command as sent on the wire."""

    cmd_id: str
    name: str
    target: str | None
    params: dict[str, Any]
    source: str
    requested_by: str


@dataclass
class CommandRecord:
    """A command plus its outcome (history entry)."""

    command: HardwareCommand
    status: CommandStatus
    detail: str
    t_created: float
    t_sent: float | None = None
    t_done: float | None = None
    attempts: int = 0
    deadline_t: float | None = None      # end of the commanded run (timed motor / actuator commands)


@dataclass(frozen=True)
class HardwareEvent:
    """What an adapter reports from ``poll``. ``kind``: connected | disconnected | telemetry | heartbeat | ack | error."""

    kind: str
    t: float
    detail: str = ""
    telemetry: HardwareTelemetry | None = None
    ack: Ack | None = None
    device: DeviceInfo | None = None


@dataclass
class LinkStats:
    """Communication counters for the dashboard."""

    messages_in: int = 0
    messages_out: int = 0
    invalid_messages: int = 0
    duplicate_or_old_seq: int = 0
    connects: int = 0
    disconnects: int = 0
    last_heard_t: float | None = None
    last_error: str = ""


@dataclass(frozen=True)
class TransportInfo:
    """Where the link is and what the transport is doing (for the dashboard; never invented)."""

    role: str                          # "server" (host listens) | "client" (host connects) | "simulation" | "none"
    endpoint: str                      # listening URL, device URL, or a simulation label
    active: bool                       # listening / connector running
    peer: str | None = None            # connected device address, if known
    attempt: int = 0                   # client role: failed connection attempts in a row
    next_retry_in_s: float | None = None   # client role: seconds until the next attempt
    last_error: str = ""
    gave_up: bool = False              # client role: max_attempts reached (Connect starts again)


@dataclass(frozen=True)
class CommEntry:
    """One line of the communication log (raw telemetry/heartbeats are counted, not listed; tokens redacted)."""

    t_utc: str
    direction: str                     # "in" | "out" | "link" | "host"
    kind: str
    text: str


@dataclass(frozen=True)
class HardwareStatus:
    """Thread-safe snapshot for the dashboard and the runtime."""

    enabled: bool
    mode: str
    device_id: str
    physical_actuation_allowed: bool
    link: LinkState
    device: DeviceInfo | None
    telemetry: HardwareTelemetry | None
    telemetry_age_s: float | None
    telemetry_stale: bool
    estop_latched: bool
    faults: tuple[str, ...]
    stats: LinkStats
    commands: tuple[CommandRecord, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = ()
    running: bool = False                          # link opened (start / Connect) and not stopped
    transport: TransportInfo | None = None
    last_heard_age_s: float | None = None          # seconds since the last valid device message
    comm_log: tuple[CommEntry, ...] = ()
    config_error: str | None = None                # hardware configuration missing/invalid
    estop_clear_needs_pin: bool = False
    estop_clear_operators: tuple[str, ...] = ()

    @property
    def simulated(self) -> bool:
        return self.mode == "simulation"
