"""Software-only hardware: the reference device model behind an in-memory link (``mode: simulation``).

Not the AI video simulation and not ArduPilot SITL: this stands in for the ESP32 pool prototype. Messages
go through exactly the same JSON protocol and ``DeviceSession`` code as the real link, so simulation exercises
the real parsing, sequencing, heartbeat, timeout and acknowledgement paths.

Test / demo hooks (``SimulatedHardware.device_sim``): inject or clear faults, set a sensor value or make it
unavailable, take the device offline (and back), freeze telemetry (stale), send a corrupt message, press the
physical e-stop, make the firmware ignore commands. From other threads use ``HardwareService.simulate`` (locked).
"""
from __future__ import annotations

import json

from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.device_model import DeviceModel
from lifeguard.hardware.models import TransportInfo
from lifeguard.hardware.session import DeviceSession
from lifeguard.hardware.transport import Transport

_RECONNECT_S = 1.0


class SimulatedDeviceTransport(Transport):
    """In-memory link to a ``DeviceModel`` that behaves like a device: hello, periodic telemetry, heartbeats."""

    def __init__(self, config: HardwareConfig) -> None:
        self._cfg = config
        self.model = DeviceModel(config.device_id, config.components, link_timeout_s=config.link_timeout_s)
        self._started_at: float | None = None
        self._connected = False
        self._conn_id = 0
        self._outbox: list[str] = []
        self._now = 0.0
        self._next_telemetry = 0.0
        self._next_heartbeat = 0.0
        self._reconnect_at: float | None = None
        self.offline = False             # hook: device unreachable (power off, Wi-Fi lost)
        self.telemetry_frozen = False    # hook: link alive but no telemetry (stale)
        self._corrupt_next = False

    # --------------------------------------------------------------- hooks
    def inject_fault(self, code: str) -> None:
        self.model.faults.add(code)

    def clear_faults(self) -> None:
        self.model.faults.clear()

    def set_sensor(self, sensor_id: str, value: float | None) -> None:
        if sensor_id not in self.model.sensor_values:
            raise KeyError(sensor_id)
        self.model.sensor_values[sensor_id] = value

    def go_offline(self) -> None:
        self.offline = True
        self._connected = False

    def come_online(self) -> None:
        self.offline = False
        self._reconnect_at = self._now

    def send_corrupt_message(self) -> None:
        self._corrupt_next = True

    def press_physical_estop(self, pressed: bool = True) -> None:
        """The (simulated) physical e-stop switch: outputs off, reported as engaged, clearing refused."""
        self.model.physical_estop = pressed

    def ignore_commands(self, ignore: bool = True) -> None:
        """Simulated firmware hang: commands get no acknowledgement (exercises timeouts and e-stop re-sends)."""
        self.model.ignore_commands = ignore

    # --------------------------------------------------------------- transport
    def start(self) -> None:
        self._started_at = None          # set on the first receive (uses the service clock)
        self._active = True

    def stop(self) -> None:
        self._connected = False
        self._active = False
        self._outbox.clear()

    def info(self) -> TransportInfo:
        return TransportInfo(role="simulation", endpoint="in-process simulated ESP32 (no network, nothing moves)",
                             active=getattr(self, "_active", False),
                             peer="simulator" if self._connected else None,
                             last_error="device offline (simulated)" if self.offline else "")

    def _open(self, now: float) -> None:
        self._connected = True
        self._conn_id += 1
        self.model.seq = 0
        self.model.host_seen(now)
        self._outbox.append(json.dumps(self.model.hello()))
        self._next_telemetry = now
        self._next_heartbeat = now + self._cfg.heartbeat_interval_s

    def receive(self, now: float) -> list[str | bytes]:
        self._now = now
        if self._started_at is None:
            self._started_at = now
            self._reconnect_at = now + self._cfg.simulation.connect_delay_s
        if not self._connected and not self.offline and self._reconnect_at is not None and now >= self._reconnect_at:
            self._open(now)
        if self._connected:
            self._outbox.extend(json.dumps(m) for m in self.model.tick(now))
            if not self.telemetry_frozen and now >= self._next_telemetry:
                self._outbox.append(json.dumps(self.model.telemetry(now)))
                self._next_telemetry = now + self._cfg.telemetry_interval_s
            if now >= self._next_heartbeat:     # heartbeats continue while telemetry is frozen (stale case)
                self._outbox.append(json.dumps(self.model.heartbeat(now)))
                self._next_heartbeat = now + self._cfg.heartbeat_interval_s
        if self._corrupt_next and self._connected:
            self._outbox.append('{"type": "telemetry", "proto": 1, "seq": ')     # truncated JSON
            self._corrupt_next = False
        out: list[str | bytes] = list(self._outbox)
        self._outbox.clear()
        return out

    def send(self, text: str) -> bool:
        if not self._connected:
            return False
        self._outbox.extend(json.dumps(m) for m in self.model.handle(json.loads(text), self._now))
        return True

    def drop(self, reason: str) -> None:
        self._connected = False
        if not self.offline:
            self._reconnect_at = self._now + _RECONNECT_S      # a real ESP32 reconnects by itself

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_id(self) -> int:
        return self._conn_id


class SimulatedHardware(DeviceSession):
    """The simulator as a ``HardwareInterface``."""

    def __init__(self, config: HardwareConfig) -> None:
        self.device_sim = SimulatedDeviceTransport(config)
        super().__init__(config, self.device_sim, kind="simulation")
