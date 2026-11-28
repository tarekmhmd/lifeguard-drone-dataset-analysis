"""Hardware integration configuration (ESP32 pool prototype).

Safe defaults: the subsystem is disabled, the mode is ``simulation``, and physical actuation is refused even in
``esp32`` mode until ``allow_physical_actuation: true`` is set explicitly. Component names below are placeholders:
the real motors, actuators and sensors are not decided yet (docs/HARDWARE_INTEGRATION.md, section 14).
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

# Every command the host may send. ``set_motor`` and ``set_actuator`` move things (ACTUATING); the others never do.
COMMAND_NAMES: tuple[str, ...] = (
    "ping", "request_status", "safe_stop", "emergency_stop", "clear_emergency_stop", "set_motor", "set_actuator",
)
ACTUATING: frozenset[str] = frozenset({"set_motor", "set_actuator"})
CommandName = Literal[
    "ping", "request_status", "safe_stop", "emergency_stop", "clear_emergency_stop", "set_motor", "set_actuator",
]
Source = Literal["operator", "response", "system"]


def _check_id(value: str, what: str) -> str:
    if not _ID.fullmatch(value):
        raise ValueError(f"invalid {what} {value!r}: use 1-64 letters, digits, '_', '.', '-'")
    return value


class MotorSpec(BaseModel):
    """One motor channel. Speeds are normalised to [-1, 1]; the firmware maps them to the real driver."""

    model_config = ConfigDict(extra="forbid")

    max_speed: float = Field(0.5, gt=0.0, le=1.0)          # |speed| allowed by the host
    max_duration_s: float = Field(3.0, gt=0.0, le=60.0)    # every run must end by itself (no continuous runs)
    description: str = ""


class ActuatorSpec(BaseModel):
    """One discrete actuator (relay, valve, indicator...). ``safe_state`` is what safe-stop returns it to."""

    model_config = ConfigDict(extra="forbid")

    states: tuple[str, ...] = ("off", "on")
    safe_state: str = "off"
    max_duration_s: float = Field(5.0, gt=0.0, le=60.0)    # any non-safe state must time out
    description: str = ""

    @model_validator(mode="after")
    def _safe_state_known(self) -> ActuatorSpec:
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("actuator states must be non-empty and distinct")
        if self.safe_state not in self.states:
            raise ValueError(f"safe_state {self.safe_state!r} is not one of {self.states}")
        return self


class SensorSpec(BaseModel):
    """One sensor reading. ``valid_min``/``valid_max`` flag implausible values; ``sim_value`` feeds the simulator."""

    model_config = ConfigDict(extra="forbid")

    unit: str | None = None
    valid_min: float | None = None
    valid_max: float | None = None
    # Simulation only. None means the simulator reports the sensor as unavailable (never an invented value).
    sim_value: float | None = None
    description: str = ""


class ComponentsConfig(BaseModel):
    """The prototype's components by id (placeholders until the hardware is specified)."""

    model_config = ConfigDict(extra="forbid")

    motors: dict[str, MotorSpec] = Field(default_factory=dict)
    actuators: dict[str, ActuatorSpec] = Field(default_factory=dict)
    sensors: dict[str, SensorSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ids_valid_and_unique(self) -> ComponentsConfig:
        ids = [*self.motors, *self.actuators, *self.sensors]
        for value in ids:
            _check_id(value, "component id")
        if len(ids) != len(set(ids)):
            raise ValueError("component ids must be unique across motors, actuators and sensors")
        return self


_HOSTNAME = re.compile(r"^(?=.{1,253}$)[A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
                       r"(\.[A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*$")
_ENV_NAME = r"^[A-Z_][A-Z0-9_]{0,63}$"


def _check_host(value: str, what: str) -> str:
    """An IP address or a DNS host name (no scheme, port or path)."""
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    if not _HOSTNAME.fullmatch(value):
        raise ValueError(f"invalid {what} {value!r}: use an IP address or a host name (no scheme, port or path)")
    return value


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ReconnectConfig(BaseModel):
    """Host-side reconnection (``role: client`` only): exponential back-off between connection attempts."""

    model_config = ConfigDict(extra="forbid")

    initial_backoff_s: float = Field(1.0, gt=0.0, le=60.0)
    max_backoff_s: float = Field(30.0, gt=0.0, le=600.0)
    multiplier: float = Field(2.0, ge=1.0, le=10.0)
    # 0 = keep trying (at most every max_backoff_s). N > 0 = give up after N failed attempts (Connect retries).
    max_attempts: int = Field(0, ge=0, le=100_000)

    @model_validator(mode="after")
    def _ordered(self) -> ReconnectConfig:
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("reconnect.max_backoff_s must be >= reconnect.initial_backoff_s")
        return self


class Esp32LinkConfig(BaseModel):
    """WebSocket link to the ESP32 (docs section 3).

    ``role: server`` (default): this computer listens on ``bind_host:port``; the ESP32 connects to it.
    ``role: client``: this computer connects to the ESP32 at ``device_host:device_port`` (the firmware runs a
    WebSocket server) and reconnects with back-off. The messages are identical in both roles.
    """

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["websocket"] = "websocket"
    role: Literal["server", "client"] = "server"
    # Server role. Loopback by default: a real ESP32 needs the computer's LAN address (or 0.0.0.0) set explicitly.
    bind_host: str = "127.0.0.1"
    port: int = Field(8765, ge=1024, le=65535)
    # Server role: if non-empty, only these device IP addresses may take the (single) device slot.
    allowed_peers: tuple[str, ...] = ()
    # Client role: the ESP32's address (from the team's network setup; never hard-coded in Python).
    device_host: str | None = None
    device_port: int = Field(8765, ge=1, le=65535)
    path: str = Field("/lifeguard-hw", pattern=r"^/[A-Za-z0-9_\-/]{0,63}$")
    # Name of an environment variable holding a shared token the device must send in "hello". Never the token itself.
    auth_token_env: str | None = Field(None, pattern=_ENV_NAME)
    max_message_bytes: int = Field(16384, ge=512, le=1_048_576)
    open_timeout_s: float = Field(3.0, gt=0.0, le=30.0)     # WebSocket opening handshake
    close_timeout_s: float = Field(1.0, gt=0.0, le=10.0)    # closing handshake, then the socket is aborted
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig)

    @field_validator("bind_host")
    @classmethod
    def _bind(cls, value: str) -> str:
        return _check_host(value, "bind_host")

    @field_validator("device_host")
    @classmethod
    def _device(cls, value: str | None) -> str | None:
        return None if value is None else _check_host(value, "device_host")

    @field_validator("allowed_peers")
    @classmethod
    def _peers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for peer in value:
            try:
                ipaddress.ip_address(peer)
            except ValueError as err:
                raise ValueError(f"allowed_peers entry {peer!r} is not an IP address") from err
        return value

    @property
    def endpoint(self) -> str:
        """Where the link is: the listening URL (server) or the device URL (client)."""
        if self.role == "client":
            return f"ws://{self.device_host}:{self.device_port}{self.path}"
        return f"ws://{self.bind_host}:{self.port}{self.path}"

    @property
    def off_loopback(self) -> bool:
        host = self.device_host if self.role == "client" else self.bind_host
        return host is None or not is_loopback(host)


class EstopClearConfig(BaseModel):
    """Who may clear a latched emergency stop (always an operator; never the AI response or the system)."""

    model_config = ConfigDict(extra="forbid")

    # If non-empty, only these operator names may clear (case-insensitive).
    operators: tuple[str, ...] = ()
    # If set, clearing also needs the PIN held in this environment variable (the PIN is never in a file).
    pin_env: str | None = Field(None, pattern=_ENV_NAME)


class SimulationConfig(BaseModel):
    """Software-only device used when ``mode: simulation``."""

    model_config = ConfigDict(extra="forbid")

    connect_delay_s: float = Field(0.0, ge=0.0, le=30.0)


class CommandSpec(BaseModel):
    """A command the response bridge sends after a lifeguard-confirmed dispatch."""

    model_config = ConfigDict(extra="forbid")

    name: CommandName
    target: str | None = None
    params: dict[str, float | str] = Field(default_factory=dict)


def _default_permissions() -> dict[str, tuple[str, ...]]:
    return {
        "operator": COMMAND_NAMES,                                   # dashboard (a named lifeguard/operator)
        "response": (),                                              # nothing unless configured explicitly
        "system": ("ping", "request_status", "safe_stop", "emergency_stop"),   # safety housekeeping only
    }


class HardwareConfig(BaseModel):
    """The ``hardware:`` block of the system configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["simulation", "esp32"] = "simulation"
    device_id: str = "pool-prototype-01"
    # Physical safety gate: in esp32 mode, set_motor / set_actuator are refused unless this is true.
    allow_physical_actuation: bool = False

    telemetry_interval_s: float = Field(0.5, gt=0.0, le=10.0)
    heartbeat_interval_s: float = Field(1.0, gt=0.0, le=10.0)
    link_timeout_s: float = Field(3.0, gt=0.0, le=60.0)      # nothing heard for this long -> link lost
    stale_telemetry_s: float = Field(2.0, gt=0.0, le=60.0)   # telemetry older than this -> actuation refused
    command_timeout_s: float = Field(2.0, gt=0.0, le=30.0)   # no acknowledgement within this -> TIMEOUT
    hello_timeout_s: float = Field(2.0, gt=0.0, le=30.0)     # connection without a valid hello -> closed
    emergency_stop_retries: int = Field(3, ge=0, le=10)
    emergency_stop_retry_s: float = Field(0.5, gt=0.0, le=10.0)   # re-send interval of an unacknowledged e-stop
    auto_connect: bool = True                 # open the link when the application starts (else: Connect button)
    safe_stop_on_connect: bool = True         # (re)connect -> safe_stop (emergency_stop if the host latch is set)
    safe_stop_on_shutdown: bool = True        # stop / disconnect -> safe_stop first
    safe_stop_on_stale_telemetry: bool = True  # telemetry goes stale while something moves -> safe_stop
    estop_clear: EstopClearConfig = Field(default_factory=EstopClearConfig)

    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    permissions: dict[Source, tuple[CommandName, ...]] = Field(default_factory=_default_permissions)
    on_dispatch: tuple[CommandSpec, ...] = ()

    esp32: Esp32LinkConfig = Field(default_factory=Esp32LinkConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    history_size: int = Field(200, ge=10, le=10_000)
    command_log: Path | None = None          # JSON lines: every command and status change
    log_messages: bool = False               # DEBUG-log every raw message on the link

    @field_validator("device_id")
    @classmethod
    def _device_id(cls, value: str) -> str:
        return _check_id(value, "device_id")

    @model_validator(mode="after")
    def _consistent(self) -> HardwareConfig:
        if self.link_timeout_s <= self.heartbeat_interval_s:
            raise ValueError("link_timeout_s must be longer than heartbeat_interval_s")
        if self.stale_telemetry_s <= self.telemetry_interval_s:
            raise ValueError("stale_telemetry_s must be longer than telemetry_interval_s")
        for source in ("operator", "response", "system"):
            self.permissions.setdefault(source, _default_permissions()[source])
        for source in ("response", "system"):
            if "clear_emergency_stop" in self.permissions[source]:
                raise ValueError(f"permissions.{source} may not include clear_emergency_stop (operators only)")
        link = self.esp32
        if self.mode == "esp32" and link.role == "client" and link.device_host is None:
            raise ValueError("esp32.role client needs esp32.device_host (the ESP32's IP address or host name)")
        if (self.mode == "esp32" and self.allow_physical_actuation and link.off_loopback
                and link.auth_token_env is None):
            raise ValueError("physical actuation over a network link needs esp32.auth_token_env (shared token)")
        allowed = set(self.permissions["response"])
        for spec in self.on_dispatch:
            if spec.name not in allowed:
                raise ValueError(f"on_dispatch command {spec.name!r} is not in permissions.response")
            if spec.name == "set_motor" and spec.target not in self.components.motors:
                raise ValueError(f"on_dispatch set_motor target {spec.target!r} is not a configured motor")
            if spec.name == "set_actuator" and spec.target not in self.components.actuators:
                raise ValueError(f"on_dispatch set_actuator target {spec.target!r} is not a configured actuator")
        return self


# ------------------------------------------------------------------ dedicated hardware file (configs/hardware.yaml)
class HardwareConfigError(ValueError):
    """The hardware configuration file is missing or invalid (the AI application keeps running without it)."""


def load_hardware_config(path: Path | str) -> HardwareConfig:
    """Load a hardware file: a YAML mapping with exactly one key, ``hardware:`` (the same block as inline).

    Relative ``command_log`` paths are resolved against the file's directory. Raises ``HardwareConfigError``.
    """
    file = Path(path)
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise HardwareConfigError(f"hardware configuration file not found: {file}") from err
    except (OSError, UnicodeError) as err:
        raise HardwareConfigError(f"cannot read hardware configuration {file}: {err}") from err
    except yaml.YAMLError as err:
        raise HardwareConfigError(f"invalid YAML in hardware configuration {file}: {err}") from err
    if not isinstance(data, dict) or set(data) != {"hardware"} or not isinstance(data["hardware"], dict):
        raise HardwareConfigError(f"{file} must contain exactly one top-level mapping key: 'hardware'")
    try:
        config = HardwareConfig.model_validate(data["hardware"])
    except ValidationError as err:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'hardware'}: {e['msg']}" for e in err.errors())
        raise HardwareConfigError(f"invalid hardware configuration in {file}: {problems}") from err
    if config.command_log is not None and not config.command_log.is_absolute():
        config.command_log = (file.parent / config.command_log).resolve()
    return config


def load_hardware_config_safe(path: Path | str) -> tuple[HardwareConfig | None, str | None]:
    """``(config, None)`` or ``(None, reason)``: never raises, so a bad file cannot stop the AI demo."""
    try:
        return load_hardware_config(path), None
    except HardwareConfigError as err:
        return None, str(err)
