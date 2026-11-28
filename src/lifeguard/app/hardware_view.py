"""Hardware display and operator controls, shared by the AI dashboard and the standalone Hardware Demo window.

No Streamlit import here (testable headless). Everything shown comes from ``HardwareStatus``; nothing is invented:
missing telemetry is shown as missing, simulated data is labelled SIMULATION, and every control maps to a real
``HardwareService`` call where all safety checks apply (the UI only explains why a control is unavailable).
"""
from __future__ import annotations

import logging
import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from lifeguard.hardware.config import load_hardware_config_safe
from lifeguard.hardware.models import CommandRecord, HardwareStatus, LinkState
from lifeguard.hardware.service import HardwareService

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HARDWARE_CONFIG = ROOT / "configs" / "hardware.yaml"


# ------------------------------------------------------------------ state shown to the operator
@dataclass(frozen=True)
class StateView:
    """``label``: CONNECTED | CONNECTING | RECONNECTING | DISCONNECTED | FAULT | CONFIGURATION ERROR."""

    label: str
    level: str        # success | warning | error | neutral
    detail: str


def hardware_state(hw: HardwareStatus) -> StateView:
    """The one connection state to display, derived only from what the service reports."""
    t = hw.transport
    endpoint = t.endpoint if t is not None else ""
    if hw.config_error:
        return StateView("CONFIGURATION ERROR", "error", hw.config_error)
    if hw.link == LinkState.FAULT:
        return StateView("FAULT", "error", hw.errors[-1] if hw.errors else "hardware adapter unavailable")
    if not hw.running:
        return StateView("DISCONNECTED", "neutral", "link closed: press Connect to open it")
    if hw.link == LinkState.CONNECTED:
        fw = hw.device.firmware if hw.device and hw.device.firmware else "unknown firmware"
        peer = f" at {t.peer}" if t is not None and t.peer else ""
        return StateView("CONNECTED", "success", f"{hw.device_id} ({fw}){peer}: identity verified")
    if hw.link == LinkState.CONNECTING:
        return StateView("CONNECTING", "warning", "link open, waiting for a valid hello (identity and token check)")
    if t is not None and t.role == "client":
        if t.gave_up:
            return StateView("DISCONNECTED", "error",
                             f"gave up after {t.attempt} attempts ({t.last_error}); press Connect to retry")
        if t.attempt > 0 or hw.stats.connects > 0:
            nxt = f"; next attempt in {t.next_retry_in_s:.1f} s" if t.next_retry_in_s is not None else ""
            why = f" - {t.last_error}" if t.last_error else ""
            return StateView("RECONNECTING", "warning", f"connecting to {endpoint}, attempt {t.attempt + 1}{nxt}{why}")
        return StateView("CONNECTING", "warning", f"connecting to {endpoint}")
    if hw.stats.connects > 0:
        why = f" ({t.last_error})" if t is not None and t.last_error else ""
        return StateView("RECONNECTING", "warning",
                         f"link to {hw.device_id} lost; waiting for it to reconnect{why}")
    where = f"listening on {endpoint}; " if endpoint and (t is None or t.role == "server") else ""
    return StateView("DISCONNECTED", "warning", f"{where}waiting for {hw.device_id} to connect (device offline)")


def hardware_badges(hw: HardwareStatus) -> list[tuple[str, str]]:
    """``(level, text)`` banners for the hardware panel; level is info | success | warning | error."""
    out: list[tuple[str, str]] = []
    if hw.config_error:
        out.append(("error", f"HARDWARE CONFIGURATION ERROR: {hw.config_error} (the AI pipeline keeps running)"))
        return out
    if hw.mode == "simulation":
        out.append(("info", "SIMULATION mode: software device only, nothing physical moves"))
    else:
        gate = "ENABLED" if hw.physical_actuation_allowed else "DISABLED (stop commands only)"
        out.append(("warning", f"PHYSICAL ESP32 mode: physical actuation {gate}"))
    if hw.link == LinkState.FAULT:
        reason = hw.errors[-1] if hw.errors else "adapter unavailable"
        out.append(("error", f"HARDWARE FAULT: {reason} (the AI pipeline keeps running)"))
    elif hw.link == LinkState.CONNECTED:
        stale = " - telemetry STALE" if hw.telemetry_stale else ""
        out.append(("warning" if stale else "success", f"Connected: {hw.device_id}{stale}"))
    elif hw.link == LinkState.CONNECTING:
        out.append(("warning", "Device connected, waiting for a valid hello"))
    else:
        out.append(("error", f"Disconnected: waiting for {hw.device_id}"))
    if hw.estop_latched:
        out.append(("error", "EMERGENCY STOP LATCHED: movement refused until an operator clears it"))
    if hw.faults:
        out.append(("error", f"Device faults: {', '.join(hw.faults)}"))
    return out


def _age(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s ago"


def hardware_rows(hw: HardwareStatus) -> dict[str, list[dict[str, Any]]]:
    """Tables for the hardware panel (missing values shown as unavailable, never invented)."""
    tel = hw.telemetry
    age = "-" if hw.telemetry_age_s is None else f"{hw.telemetry_age_s:.1f} s"
    st = hw.stats
    t = hw.transport
    link = [
        {"item": "mode", "value": hw.mode}, {"item": "device id", "value": hw.device_id},
        {"item": "link", "value": hw.link.value},
        {"item": "firmware", "value": (hw.device.firmware or "unknown") if hw.device else "-"},
        {"item": "telemetry age", "value": age},
        {"item": "messages in / out", "value": f"{st.messages_in} / {st.messages_out}"},
        {"item": "invalid / duplicate", "value": f"{st.invalid_messages} / {st.duplicate_or_old_seq}"},
        {"item": "connects / disconnects", "value": f"{st.connects} / {st.disconnects}"},
        {"item": "last error", "value": st.last_error or "-"},
    ]
    if t is not None:
        link += [
            {"item": "link role", "value": {"server": "host listens, ESP32 connects",
                                            "client": "host connects to the ESP32"}.get(t.role, t.role)},
            {"item": "address", "value": t.endpoint},
            {"item": "device address", "value": t.peer or "-"},
            {"item": "last communication", "value": _age(hw.last_heard_age_s)},
        ]
        if t.role == "client":
            link.append({"item": "reconnect", "value": "gave up (press Connect)" if t.gave_up else
                         (f"attempt {t.attempt + 1}" + (f", next in {t.next_retry_in_s:.1f} s"
                                                         if t.next_retry_in_s is not None else ""))})
    sensors = [] if tel is None else [
        {"sensor": sid, "value": "unavailable" if r.value is None else r.value, "unit": r.unit or "",
         "status": r.status} for sid, r in tel.sensors.items()]
    motors = [] if tel is None else [
        {"motor": mid, "state": m.state, "speed": "-" if m.speed is None else m.speed} for mid, m in tel.motors.items()]
    actuators = [] if tel is None else [
        {"actuator": aid, "state": a.state or "unavailable"} for aid, a in tel.actuators.items()]
    commands = [_command_row(r) for r in reversed(hw.commands)]
    comm = [{"time (UTC)": e.t_utc, "dir": e.direction, "kind": e.kind, "message": e.text}
            for e in reversed(hw.comm_log)]
    return {"link": link, "sensors": sensors, "motors": motors, "actuators": actuators, "commands": commands,
            "comm": comm}


def _command_row(r: CommandRecord) -> dict[str, Any]:
    # Most important first (narrow windows scroll the rest): what, its acknowledged state, why.
    return {"command": r.command.name, "status": r.status.value, "detail": r.detail, "target": r.command.target or "",
            "params": r.command.params, "by": r.command.requested_by, "source": r.command.source,
            "attempts": r.attempts, "id": r.command.cmd_id}


def motion_blockers(hw: HardwareStatus) -> list[str]:
    """Why a motor/actuator test would be refused right now (the command manager enforces these anyway)."""
    out: list[str] = []
    if hw.mode == "esp32" and not hw.physical_actuation_allowed:
        out.append("physical actuation is disabled (hardware.allow_physical_actuation: false)")
    if hw.link != LinkState.CONNECTED:
        out.append("device not connected")
    elif hw.telemetry_stale:
        out.append("telemetry is stale or missing")
    if hw.estop_latched:
        out.append("emergency stop is latched")
    if hw.faults:
        out.append(f"device faults: {', '.join(hw.faults)}")
    return out


def clear_blockers(hw: HardwareStatus) -> list[str]:
    """Why Clear E-STOP is unavailable (the service and the device check again)."""
    out: list[str] = []
    if not hw.estop_latched:
        out.append("emergency stop is not latched")
    if hw.link != LinkState.CONNECTED:
        out.append("device not connected (it must confirm the clear)")
    elif hw.telemetry_stale:
        out.append("telemetry is stale: the device state must be visible")
    return out


# ------------------------------------------------------------------ operator controls
class HardwareConsole:
    """Operator actions on one ``HardwareService``; every method returns a message for the operator."""

    def __init__(self, service: HardwareService | None) -> None:
        self.service = service

    def status(self) -> HardwareStatus | None:
        return self.service.status() if self.service is not None else None

    def _need(self) -> HardwareService | None:
        return self.service

    @staticmethod
    def _record(record: CommandRecord) -> str:
        detail = f" ({record.detail})" if record.detail else ""
        return f"{record.command.name}: {record.status.value}{detail}"

    def connect(self, by: str = "") -> str:
        svc = self._need()
        if svc is None:
            return "hardware not enabled"
        ok = svc.connect(by.strip() or "operator")
        return "link opened" if ok else "link could not be opened (see the hardware errors)"

    def disconnect(self, by: str = "") -> str:
        svc = self._need()
        if svc is None:
            return "hardware not enabled"
        svc.disconnect(by.strip() or "operator")
        return "link closed (safe stop sent first if a device was connected)"

    def command(self, action: str, by: str, target: str | None = None, params: dict[str, Any] | None = None,
                pin: str | None = None, confirmed: bool = False) -> str:
        """estop | safe_stop | clear_estop | ping | request_status | set_motor | set_actuator."""
        svc = self._need()
        if svc is None:
            return "hardware not enabled"
        name = by.strip()
        if action == "estop":
            return self._record(svc.emergency_stop(name or "dashboard"))   # never blocked by a missing name
        if action == "safe_stop":
            return self._record(svc.safe_stop(name or "dashboard"))
        if action == "clear_estop":
            if not name:
                return "enter the operator's name first"
            if not confirmed:
                return "confirm that the pool area is safe and the physical e-stop is released"
            return self._record(svc.clear_emergency_stop(name, pin=pin))
        return self._record(svc.request(action, target, params, source="operator", by=name))

    def simulate(self, action: str, arg: str = "") -> str:
        svc = self._need()
        return "hardware not enabled" if svc is None else svc.simulate(action, arg)


# ------------------------------------------------------------------ standalone window back end
class HardwareDemo:
    """Back end of the Hardware Demo window: one ``HardwareService`` built from a hardware YAML.

    Same service, protocol and safety logic as the AI runtime; only the YAML decides simulation vs real ESP32.
    """

    _instances: ClassVar[weakref.WeakSet[HardwareDemo]] = weakref.WeakSet()

    @classmethod
    def close_all(cls) -> None:
        """Stop every demo service of this process (Streamlit caches them; used by tests and on reload)."""
        for item in list(cls._instances):
            item.close()

    def __init__(self, path: Path | str | None = None) -> None:
        HardwareDemo._instances.add(self)
        self.path = Path(path or os.environ.get("LIFEGUARD_HARDWARE_CONFIG", DEFAULT_HARDWARE_CONFIG))
        config, error = load_hardware_config_safe(self.path)
        self.enabled = config is not None and config.enabled
        self.config = config
        self.error = error
        if error is not None:
            self.service: HardwareService | None = HardwareService.unavailable(error)
        elif config is not None and config.enabled:
            self.service = HardwareService(config)
            if config.auto_connect:
                self.service.start()
        else:
            self.service = None                      # enabled: false -> AI-only, nothing to show but that
        self.console = HardwareConsole(self.service)

    def close(self) -> None:
        if self.service is not None:
            self.service.stop()
