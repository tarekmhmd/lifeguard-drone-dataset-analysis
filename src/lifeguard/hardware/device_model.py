"""Reference device behaviour (what the ESP32 firmware must do), as a pure-Python state machine.

Used by the in-process simulator (``simulator.py``) and by the fake ESP32 client (``fake_device.py``), and meant
as an executable specification for the firmware (docs/HARDWARE_INTEGRATION.md §10). It works on wire-format
dicts, so it behaves exactly like a device on the link.

Firmware rules it implements:
- validate every command itself (never trust the host): known target, range, duration;
- every motor run / non-safe actuator state ends by itself after ``duration_ms`` (deadman timer);
- ``emergency_stop`` stops everything immediately and latches; only ``clear_emergency_stop`` releases the latch
  (a physical e-stop must additionally cut power in hardware — software cannot guarantee that);
- no host heartbeat for ``link_timeout_ms`` -> safe stop (failsafe on link loss);
- an active fault blocks actuation; motors are stopped.

This is SIMULATION. It moves nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lifeguard.hardware.config import ComponentsConfig


@dataclass
class _Run:
    value: Any
    until: float | None          # host-independent end time (device clock), None = indefinitely (safe state only)
    cmd_id: str | None


@dataclass
class DeviceModel:
    """A device with motors, actuators and sensors as configured; times are device-clock seconds."""

    device_id: str
    components: ComponentsConfig
    link_timeout_s: float = 3.0
    firmware: str = "sim-0.1"
    motors: dict[str, _Run] = field(default_factory=dict)
    actuators: dict[str, _Run] = field(default_factory=dict)
    sensor_values: dict[str, float | None] = field(default_factory=dict)
    faults: set[str] = field(default_factory=set)
    estop: bool = False              # software latch (emergency_stop command)
    physical_estop: bool = False     # the physical e-stop switch is pressed: outputs off, clearing refused
    ignore_commands: bool = False    # simulated firmware hang: commands are neither executed nor acknowledged
    last_host_t: float | None = None
    seq: int = 0

    def __post_init__(self) -> None:
        self.motors = {m: _Run(0.0, None, None) for m in self.components.motors}
        self.actuators = {a: _Run(spec.safe_state, None, None) for a, spec in self.components.actuators.items()}
        self.sensor_values = {s: spec.sim_value for s, spec in self.components.sensors.items()}

    # ------------------------------------------------------------------ messages out
    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def hello(self, token: str | None = None) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "type": "hello", "proto": 1, "device_id": self.device_id, "seq": self._next(), "firmware": self.firmware,
            "components": {"motors": list(self.motors), "actuators": list(self.actuators),
                           "sensors": list(self.sensor_values)},
        }
        if token is not None:
            msg["token"] = token
        return msg

    def heartbeat(self, now: float) -> dict[str, Any]:
        return {"type": "heartbeat", "proto": 1, "device_id": self.device_id, "seq": self._next(),
                "t_device_ms": int(now * 1000)}

    def telemetry(self, now: float) -> dict[str, Any]:
        sensors = {}
        for sid, value in self.sensor_values.items():
            spec = self.components.sensors[sid]
            sensors[sid] = {"value": value, "unit": spec.unit, "fault": f"sensor_fault:{sid}" in self.faults}
        motors = {mid: {"state": "fault" if "motor_driver_fault" in self.faults else
                        ("running" if run.value != 0.0 else "stopped"), "speed": float(run.value)}
                  for mid, run in self.motors.items()}
        return {"type": "telemetry", "proto": 1, "device_id": self.device_id, "seq": self._next(),
                "t_device_ms": int(now * 1000), "sensors": sensors, "motors": motors,
                "actuators": {aid: {"state": run.value} for aid, run in self.actuators.items()},
                "faults": sorted(self.faults), "estop_engaged": self.estop or self.physical_estop,
                "power": {"voltage_v": None, "battery_pct": None}}

    def _ack(self, cmd_id: str, status: str, detail: str = "") -> dict[str, Any]:
        return {"type": "ack", "proto": 1, "device_id": self.device_id, "seq": self._next(), "cmd_id": cmd_id,
                "status": status, "detail": detail}

    # ------------------------------------------------------------------ state changes
    def _stop_all(self) -> None:
        """Stop motors and return actuators to their safe state (interrupted runs are superseded host-side)."""
        for run in self.motors.values():
            run.value, run.until, run.cmd_id = 0.0, None, None
        for aid, run in self.actuators.items():
            run.value, run.until, run.cmd_id = self.components.actuators[aid].safe_state, None, None

    def host_seen(self, now: float) -> None:
        self.last_host_t = now

    def tick(self, now: float) -> list[dict[str, Any]]:
        """Advance timers; returns 'completed' acks of runs that ended and enforces the link failsafe."""
        out: list[dict[str, Any]] = []
        if self.last_host_t is not None and now - self.last_host_t > self.link_timeout_s:
            self._stop_all()                       # failsafe: host silent
        for run in self.motors.values():
            if run.until is not None and now >= run.until:
                if run.cmd_id:
                    out.append(self._ack(run.cmd_id, "completed", "run duration elapsed"))
                run.value, run.until, run.cmd_id = 0.0, None, None
        for aid, run in self.actuators.items():
            if run.until is not None and now >= run.until:
                if run.cmd_id:
                    out.append(self._ack(run.cmd_id, "completed", "duration elapsed"))
                run.value, run.until, run.cmd_id = self.components.actuators[aid].safe_state, None, None
        if self.physical_estop:
            self._stop_all()                       # the switch cuts the outputs
        if "motor_driver_fault" in self.faults:
            for run in self.motors.values():
                run.value, run.until, run.cmd_id = 0.0, None, None
        return out

    def handle(self, msg: dict[str, Any], now: float) -> list[dict[str, Any]]:
        """React to one host message; returns the device's replies."""
        kind = msg.get("type")
        if kind in ("heartbeat", "welcome"):
            self.host_seen(now)
            return []
        if kind != "command":
            return []
        self.host_seen(now)
        if self.ignore_commands:
            return []
        cmd_id = str(msg.get("cmd_id", ""))
        name = msg.get("name")
        target = msg.get("target")
        params = msg.get("params") or {}
        if name == "emergency_stop":
            self.estop = True
            self._stop_all()
            return [self._ack(cmd_id, "completed", "emergency stop latched")]
        if name == "clear_emergency_stop":
            if self.physical_estop:
                return [self._ack(cmd_id, "rejected", "physical emergency stop is still engaged")]
            self.estop = False
            return [self._ack(cmd_id, "completed", "emergency stop cleared")]
        if name == "safe_stop":
            self._stop_all()
            return [self._ack(cmd_id, "completed", "all outputs safe")]
        if name in ("ping", "request_status"):
            replies = [self._ack(cmd_id, "completed", "pong" if name == "ping" else "status sent")]
            if name == "request_status":
                replies.append(self.telemetry(now))
            return replies
        if name not in ("set_motor", "set_actuator"):
            return [self._ack(cmd_id, "rejected", f"unknown command {name!r}")]
        if self.estop or self.physical_estop:
            return [self._ack(cmd_id, "rejected", "emergency stop latched")]
        if self.faults:
            return [self._ack(cmd_id, "rejected", f"device fault: {', '.join(sorted(self.faults))}")]
        duration_ms = params.get("duration_ms")
        if name == "set_actuator":
            spec_safe = self.components.actuators.get(str(target))
            if spec_safe is not None and params.get("state") == spec_safe.safe_state:
                run = self.actuators[str(target)]
                run.value, run.until, run.cmd_id = spec_safe.safe_state, None, None
                return [self._ack(cmd_id, "completed", "safe state")]
        if not isinstance(duration_ms, (int, float)) or isinstance(duration_ms, bool) or duration_ms <= 0:
            return [self._ack(cmd_id, "rejected", "duration_ms required")]
        if name == "set_motor":
            spec = self.components.motors.get(str(target))
            speed = params.get("speed")
            if spec is None:
                return [self._ack(cmd_id, "rejected", f"unknown motor {target!r}")]
            if not isinstance(speed, (int, float)) or abs(speed) > spec.max_speed:
                return [self._ack(cmd_id, "rejected", "speed out of range")]
            if duration_ms > spec.max_duration_s * 1000:
                return [self._ack(cmd_id, "rejected", "duration too long")]
            run = self.motors[str(target)]
            if run.cmd_id:            # a newer command replaces a running one
                replaced = [self._ack(run.cmd_id, "completed", "replaced by a newer command")]
            else:
                replaced = []
            run.value, run.until, run.cmd_id = float(speed), now + duration_ms / 1000.0, cmd_id
            return [*replaced, self._ack(cmd_id, "accepted", "running")]
        spec_a = self.components.actuators.get(str(target))
        state = params.get("state")
        if spec_a is None:
            return [self._ack(cmd_id, "rejected", f"unknown actuator {target!r}")]
        if state not in spec_a.states:
            return [self._ack(cmd_id, "rejected", "unknown state")]
        if duration_ms > spec_a.max_duration_s * 1000:
            return [self._ack(cmd_id, "rejected", "duration too long")]
        run = self.actuators[str(target)]
        run.value, run.until, run.cmd_id = state, now + duration_ms / 1000.0, cmd_id
        return [self._ack(cmd_id, "accepted", f"{state} for {duration_ms / 1000.0:.1f} s")]
