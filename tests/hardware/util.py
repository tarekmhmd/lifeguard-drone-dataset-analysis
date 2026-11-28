"""Helpers for the hardware tests: a deterministic clock, a standard config and a scripted transport."""
from __future__ import annotations

import json
from typing import Any

from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.service import HardwareService
from lifeguard.hardware.transport import Transport


def config(**overrides: Any) -> HardwareConfig:
    base: dict[str, Any] = {
        "enabled": True, "telemetry_interval_s": 0.2, "heartbeat_interval_s": 0.5, "link_timeout_s": 2.0,
        "stale_telemetry_s": 1.0, "command_timeout_s": 1.0,
        "components": {
            "motors": {"motor_1": {"max_speed": 0.5, "max_duration_s": 3.0}},
            "actuators": {"indicator_1": {"states": ["off", "on"], "safe_state": "off", "max_duration_s": 5.0}},
            "sensors": {"sensor_1": {"unit": "cm", "sim_value": 12.5, "valid_min": 0, "valid_max": 100},
                        "sensor_2": {"unit": "C"}},
        },
    }
    base.update(overrides)
    return HardwareConfig.model_validate(base)


class Clock:
    """Manually advanced clock."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def run(service: HardwareService, clock: Clock, seconds: float, dt: float = 0.05) -> None:
    """Advance the clock and step the service (no background thread)."""
    end = clock.t + seconds
    while clock.t < end - 1e-9:
        clock.t = round(clock.t + dt, 6)
        service.step()


def started(cfg: HardwareConfig | None = None, connect_s: float = 0.3) -> tuple[HardwareService, Clock]:
    clock = Clock()
    service = HardwareService(cfg or config(), clock=clock)
    service.start(background=False)
    run(service, clock, connect_s)
    return service, clock


class ScriptedTransport(Transport):
    """In-memory transport: tests push device messages and read what the host sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[str | bytes] = []
        self._connected = False
        self._conn_id = 0
        self.dropped: list[str] = []
        self.started = False

    def device_connects(self) -> None:
        self._connected = True
        self._conn_id += 1

    def device_disconnects(self) -> None:
        self._connected = False

    def push(self, msg: dict[str, Any] | str | bytes) -> None:
        self.inbox.append(msg if isinstance(msg, (str, bytes)) else json.dumps(msg))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self._connected = False

    def send(self, text: str) -> bool:
        if not self._connected:
            return False
        self.sent.append(json.loads(text))
        return True

    def receive(self, now: float) -> list[str | bytes]:
        out, self.inbox = self.inbox, []
        return out

    def drop(self, reason: str) -> None:
        self.dropped.append(reason)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_id(self) -> int:
        return self._conn_id


def msg(kind: str, seq: int, device_id: str = "pool-prototype-01", **fields: Any) -> dict[str, Any]:
    return {"type": kind, "proto": 1, "device_id": device_id, "seq": seq, **fields}
