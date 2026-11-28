"""Telemetry normalisation and tracking: latest report, staleness and active faults.

Telemetry is hardware state, not evidence about swimmers: nothing here feeds the risk engine.
"""
from __future__ import annotations

from lifeguard.hardware.config import ComponentsConfig
from lifeguard.hardware.models import (
    ActuatorState,
    HardwareTelemetry,
    MotorState,
    PowerStatus,
    SensorReading,
)
from lifeguard.hardware.protocol import Telemetry


def normalise(msg: Telemetry, components: ComponentsConfig, t_host: float, source: str) -> HardwareTelemetry:
    """Wire telemetry -> ``HardwareTelemetry``. Missing configured components are reported as unavailable."""
    sensors: dict[str, SensorReading] = {}
    for sid, spec in components.sensors.items():
        raw = msg.sensors.get(sid)
        if raw is None or raw.value is None:
            sensors[sid] = SensorReading(None, spec.unit, "fault" if raw is not None and raw.fault else "unavailable")
            continue
        if raw.fault:
            status = "fault"
        elif (spec.valid_min is not None and raw.value < spec.valid_min) or (
                spec.valid_max is not None and raw.value > spec.valid_max):
            status = "out_of_range"
        else:
            status = "ok"
        sensors[sid] = SensorReading(raw.value, raw.unit or spec.unit, status)
    for sid, raw in msg.sensors.items():          # reported but not configured: shown, flagged
        if sid not in sensors:
            sensors[sid] = SensorReading(raw.value, raw.unit, "unconfigured")
    motors = {mid: MotorState("unknown", None) for mid in components.motors}
    motors.update({mid: MotorState(m.state, m.speed) for mid, m in msg.motors.items()})
    actuators = {aid: ActuatorState(None) for aid in components.actuators}
    actuators.update({aid: ActuatorState(a.state) for aid, a in msg.actuators.items()})
    power = None if msg.power is None else PowerStatus(msg.power.voltage_v, msg.power.battery_pct)
    return HardwareTelemetry(
        device_id=msg.device_id, seq=msg.seq, t_host=t_host, t_device_ms=msg.t_device_ms, source=source,
        sensors=sensors, motors=motors, actuators=actuators, faults=tuple(msg.faults),
        estop_engaged=msg.estop_engaged, power=power,
    )


class TelemetryManager:
    """Keeps the latest telemetry and answers "is it fresh?"."""

    def __init__(self, stale_after_s: float) -> None:
        self._stale_after_s = stale_after_s
        self.latest: HardwareTelemetry | None = None
        self.received = 0

    def update(self, telemetry: HardwareTelemetry) -> None:
        self.latest = telemetry
        self.received += 1

    def clear(self) -> None:
        """Forget the device state (link lost): old values must not look current."""
        self.latest = None

    def age_s(self, now: float) -> float | None:
        return None if self.latest is None else max(0.0, now - self.latest.t_host)

    def is_stale(self, now: float) -> bool:
        """True when there is no telemetry or it is older than ``stale_telemetry_s``."""
        age = self.age_s(now)
        return age is None or age > self._stale_after_s

    @property
    def faults(self) -> tuple[str, ...]:
        if self.latest is None:
            return ()
        sensor_faults = tuple(f"sensor_fault:{sid}" for sid, r in self.latest.sensors.items() if r.status == "fault")
        motor_faults = tuple(f"motor_fault:{mid}" for mid, m in self.latest.motors.items() if m.state == "fault")
        return tuple(dict.fromkeys((*self.latest.faults, *sensor_faults, *motor_faults)))
