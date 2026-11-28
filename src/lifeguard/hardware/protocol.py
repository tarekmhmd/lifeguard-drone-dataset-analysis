"""Host <-> ESP32 message contract, version 1 (JSON text messages; full spec: docs/HARDWARE_INTEGRATION.md §4).

Device -> host: ``hello``, ``telemetry``, ``heartbeat``, ``ack``.  Host -> device: ``welcome``, ``error``,
``command``, ``heartbeat``. Every message carries ``type``, ``proto`` (= 1) and a per-sender ``seq``.

Parsing is strict about what the host relies on (types, ranges, sizes, no NaN/Infinity) and tolerant of extra
fields, so firmware can add diagnostics without breaking the host. Any violation raises ``ProtocolError``;
the adapter counts and logs it and never crashes.
"""
from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROTOCOL_VERSION = 1
_ID = r"^[A-Za-z0-9_.\-]{1,64}$"
_FAULT = r"^[A-Za-z0-9_.:\-]{1,64}$"          # fault codes may be qualified, e.g. "sensor_fault:sensor_1"
Ident = Annotated[str, Field(pattern=_ID)]
Seq = Annotated[int, Field(ge=0, le=2**53)]


class ProtocolError(ValueError):
    """A message that does not follow the contract."""


class _Msg(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    proto: Literal[1]
    device_id: Ident
    seq: Seq


class Hello(_Msg):
    type: Literal["hello"]
    firmware: Annotated[str, Field(max_length=64)] | None = None
    token: Annotated[str, Field(max_length=256)] | None = None
    # The host uses "motors", "actuators" and "sensors"; other kinds are accepted and ignored (forward compatible).
    components: dict[Ident, list[Ident]] = Field(default_factory=dict)


class SensorValue(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    value: float | None = None            # null = unavailable; never replaced by a guess
    unit: Annotated[str, Field(max_length=16)] | None = None
    fault: bool = False


class MotorValue(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    state: Literal["stopped", "running", "fault", "unknown"] = "unknown"
    speed: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None

    @field_validator("state", mode="before")
    @classmethod
    def _unknown_state(cls, value: Any) -> Any:
        """A state the host does not know is shown as "unknown" rather than dropping the whole report."""
        return value if value in ("stopped", "running", "fault", "unknown") else "unknown"


class ActuatorValue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Annotated[str, Field(max_length=32)] | None = None


class PowerValue(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    voltage_v: Annotated[float, Field(ge=0.0, le=1000.0)] | None = None
    battery_pct: Annotated[float, Field(ge=0.0, le=100.0)] | None = None


class Telemetry(_Msg):
    type: Literal["telemetry"]
    t_device_ms: Annotated[int, Field(ge=0)] | None = None
    sensors: dict[Ident, SensorValue] = Field(default_factory=dict)
    motors: dict[Ident, MotorValue] = Field(default_factory=dict)
    actuators: dict[Ident, ActuatorValue] = Field(default_factory=dict)
    faults: list[Annotated[str, Field(pattern=_FAULT)]] = Field(default_factory=list, max_length=64)
    estop_engaged: bool | None = None
    power: PowerValue | None = None


class Heartbeat(_Msg):
    type: Literal["heartbeat"]
    t_device_ms: Annotated[int, Field(ge=0)] | None = None


class AckMsg(_Msg):
    type: Literal["ack"]
    cmd_id: Ident
    status: Literal["accepted", "completed", "rejected", "failed"]
    detail: Annotated[str, Field(max_length=256)] = ""


DeviceMessage = Hello | Telemetry | Heartbeat | AckMsg
_BY_TYPE: dict[str, type[BaseModel]] = {"hello": Hello, "telemetry": Telemetry, "heartbeat": Heartbeat, "ack": AckMsg}


def _reject_constant(value: str) -> Any:
    raise ProtocolError(f"non-finite number {value} is not allowed")


def parse_device_message(text: str | bytes, max_bytes: int = 16384) -> DeviceMessage:
    """Parse one device message. Raises ProtocolError on any violation."""
    if isinstance(text, bytes):
        raise ProtocolError("binary messages are not part of the protocol")
    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        raise ProtocolError(f"message larger than {max_bytes} bytes")
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as err:
        raise ProtocolError(f"invalid JSON: {err.msg}") from err
    if not isinstance(data, dict):
        raise ProtocolError("a message must be a JSON object")
    kind = data.get("type")
    model = _BY_TYPE.get(kind) if isinstance(kind, str) else None
    if model is None:
        raise ProtocolError(f"unknown message type {kind!r}")
    if data.get("proto") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {data.get('proto')!r} (expected {PROTOCOL_VERSION})")
    try:
        return model.model_validate(data)  # type: ignore[return-value]
    except ValidationError as err:
        first = err.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise ProtocolError(f"invalid {kind} message at {where or 'root'}: {first.get('msg')}") from err


def _dump(obj: dict[str, Any]) -> str:
    for value in obj.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolError("refusing to send a non-finite number")
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, sort_keys=True)


def encode_welcome(device_id: str, seq: int, heartbeat_ms: int, telemetry_ms: int, link_timeout_ms: int) -> str:
    return _dump({"type": "welcome", "proto": PROTOCOL_VERSION, "device_id": device_id, "seq": seq,
                  "heartbeat_interval_ms": heartbeat_ms, "telemetry_interval_ms": telemetry_ms,
                  "link_timeout_ms": link_timeout_ms})


def encode_error(device_id: str, seq: int, detail: str) -> str:
    return _dump({"type": "error", "proto": PROTOCOL_VERSION, "device_id": device_id, "seq": seq,
                  "detail": detail[:256]})


def encode_heartbeat(device_id: str, seq: int, t_host_ms: int) -> str:
    return _dump({"type": "heartbeat", "proto": PROTOCOL_VERSION, "device_id": device_id, "seq": seq,
                  "t_host_ms": t_host_ms})


def encode_command(device_id: str, seq: int, cmd_id: str, name: str, target: str | None,
                   params: dict[str, Any]) -> str:
    for key, value in params.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ProtocolError(f"refusing to send non-finite parameter {key}")
    return _dump({"type": "command", "proto": PROTOCOL_VERSION, "device_id": device_id, "seq": seq,
                  "cmd_id": cmd_id, "name": name, "target": target, "params": params})
