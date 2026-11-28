"""Hardware inside the application: AI pipeline unaffected, failures contained, the confirm-only response path,
module isolation, and the dashboard panel."""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.app.dashboard_controller import hardware_badges, hardware_rows
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.enums import AckDecision, RiskLevel
from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.models import CommandStatus, LinkState
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.live import SyntheticFrameSource

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "configs" / "ai.yaml"
SIZE = (1920, 1080)
T0 = "2026-01-01T00:00:00.000Z"


def _sys(hardware: dict | None = None):
    cfg = load_system_config(ROOT / "configs" / "system.yaml")
    if hardware is not None:
        cfg.hardware = HardwareConfig.model_validate(hardware)
    return cfg


def _events(rt: LifeguardRuntime, n: int = 800) -> list[tuple]:
    out = []
    for frame in SyntheticFrameSource(*SIZE, n_frames=n, start_t_capture=0.0, start_utc=T0):
        out += [(e.track_uid.rsplit(":", 1)[-1], e.from_level, e.to_level, e.reason, round(e.t_capture, 3))
                for e in rt.process(frame)]
    return out


SIM = {"enabled": True, "components": {"motors": {"motor_1": {}}, "actuators": {"indicator_1": {}},
                                       "sensors": {"sensor_1": {"sim_value": 3.0}}}}


def test_hardware_disabled_is_the_previous_behaviour() -> None:
    rt = LifeguardRuntime(load_ai_config(AI), _sys(), SIZE, session_id="hw_off")
    assert rt.hardware is None and "hardware" not in rt.components
    assert rt.snapshot().hardware is None
    rt.close()


def test_risk_decisions_identical_with_simulated_hardware() -> None:
    off = LifeguardRuntime(load_ai_config(AI), _sys(), SIZE, session_id="same")
    on = LifeguardRuntime(load_ai_config(AI), _sys(SIM), SIZE, session_id="same")
    try:
        assert "simulation" in on.components["hardware"]
        events_off, events_on = _events(off), _events(on)
        assert events_off == events_on and any(e[2] == RiskLevel.ALARM for e in events_on)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and on.snapshot().hardware.link != LinkState.CONNECTED:
            time.sleep(0.05)
        assert on.snapshot().hardware.link == LinkState.CONNECTED
    finally:
        off.close()
        on.close()
    assert on.hardware.status().link == LinkState.DISCONNECTED          # closed with the runtime


def test_broken_hardware_never_stops_the_ai(monkeypatch) -> None:
    monkeypatch.delenv("LIFEGUARD_HW_NOT_SET", raising=False)
    rt = LifeguardRuntime(load_ai_config(AI), _sys({"enabled": True, "mode": "esp32",
                                                    "esp32": {"auth_token_env": "LIFEGUARD_HW_NOT_SET"}}),
                          SIZE, session_id="hw_fault")
    try:
        assert rt.components["hardware"].startswith("FAULT")
        assert _events(rt, 300) is not None and rt.snapshot().summary.frames == 300
        assert rt.snapshot().hardware.link == LinkState.FAULT
    finally:
        rt.close()


def test_ai_commands_hardware_only_after_a_lifeguard_confirm() -> None:
    hw = {**SIM, "permissions": {"response": ["set_actuator"]},
          "on_dispatch": [{"name": "set_actuator", "target": "indicator_1",
                           "params": {"state": "on", "duration_s": 2}}]}
    rt = LifeguardRuntime(load_ai_config(AI), _sys(hw), SIZE, session_id="bridge")
    try:
        alarm = None
        for frame in SyntheticFrameSource(*SIZE, n_frames=1200, start_t_capture=0.0, start_utc=T0):
            alarm = next((e for e in rt.process(frame) if e.to_level == RiskLevel.ALARM), None)
            if alarm is not None:
                break
        assert alarm is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and rt.hardware.status().link != LinkState.CONNECTED:
            time.sleep(0.05)
        assert not [r for r in rt.hardware.status().commands if r.command.source == "response"]  # ALARM alone: none
        _, outcome = rt.acknowledge(alarm.event_id, AckDecision.CONFIRM, "lifeguard_1")
        sent = [r for r in rt.hardware.status().commands if r.command.source == "response"]
        assert len(sent) == 1 and sent[0].command.requested_by == "lifeguard_1"
        assert sent[0].status in (CommandStatus.PENDING, CommandStatus.ACCEPTED)
        assert outcome is not None and "hardware: set_actuator/indicator_1" in outcome.detail
    finally:
        rt.close()


def _imports(package_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in package_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
    return names


def test_ai_and_risk_code_never_import_hardware() -> None:
    src = ROOT / "src" / "lifeguard"
    for package in (src / "ai", src / "system" / "risk", src / "contracts"):
        assert not any(m.startswith("lifeguard.hardware") for m in _imports(package)), package


def test_dashboard_helpers() -> None:
    from lifeguard.hardware.service import HardwareService

    service = HardwareService(HardwareConfig.model_validate(SIM))
    status = service.status()                                            # not started yet
    badges = hardware_badges(status)
    assert badges[0] == ("info", "SIMULATION mode: software device only, nothing physical moves")
    assert any(level == "error" and text.startswith("Disconnected") for level, text in badges)
    rows = hardware_rows(status)
    assert rows["sensors"] == [] and rows["link"][0] == {"item": "mode", "value": "simulation"}
    service.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and service.status().telemetry is None:
            time.sleep(0.05)
        rows = hardware_rows(service.status())
        assert {"sensor": "sensor_1", "value": 3.0, "unit": "", "status": "ok"} in rows["sensors"]
        service.emergency_stop("op")
        assert ("error", "EMERGENCY STOP LATCHED: movement refused until an operator clears it") in \
            hardware_badges(service.status())
    finally:
        service.stop()


def test_streamlit_page_shows_hardware_and_estop(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LIFEGUARD_AI_CONFIG", str(AI))
    monkeypatch.setenv("LIFEGUARD_SYSTEM_CONFIG", str(ROOT / "configs" / "system_hardware_sim.yaml"))
    at = AppTest.from_file(str(ROOT / "src" / "lifeguard" / "app" / "dashboard.py"), default_timeout=60)
    at.run()
    next(b for b in at.sidebar.button if b.label == "Start").click()
    at.run()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not any("SIMULATION mode" in i.value for i in at.info):
        time.sleep(0.2)
        at.run()
    assert not at.exception
    assert any("SIMULATION mode" in i.value for i in at.info)
    next(b for b in at.button if b.label == "EMERGENCY STOP").click()
    at.run()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not any("EMERGENCY STOP LATCHED" in e.value for e in at.error):
        time.sleep(0.2)
        at.run()
    assert any("EMERGENCY STOP LATCHED" in e.value for e in at.error)
    next(b for b in at.sidebar.button if b.label == "Stop").click()
    at.run()
