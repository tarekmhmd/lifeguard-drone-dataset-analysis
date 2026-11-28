"""Two-page app: the AI Dashboard and Hardware pages share ONE process-level hardware service."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from lifeguard.ai.config import load_ai_config
from lifeguard.app.dashboard_controller import DashboardController, SourceChoice
from lifeguard.app.hardware_hub import HardwareHub, HardwareShareConflict, shared_hub
from lifeguard.app.hardware_view import HardwareConsole
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.enums import AckDecision, RiskLevel
from lifeguard.hardware.models import LinkState
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.live import SyntheticFrameSource
from tests.hardware.test_hw_reliability import _bounded, _free_port, _port_is_free, _wait_for
from tests.hardware.test_hw_runtime_dashboard import AI, SIZE, T0

ROOT = Path(__file__).resolve().parents[2]
HW_FILE = ROOT / "configs" / "hardware.yaml"
SYS_SHARED = ROOT / "configs" / "system_hardware.yaml"
COMPONENTS = {"motors": {"motor_1": {}}, "actuators": {"indicator_1": {}}, "sensors": {"sensor_1": {}}}


def _service_threads() -> int:
    return sum(t.name == "hardware-service" and t.is_alive() for t in threading.enumerate())


def _hw_yaml(tmp: Path, hardware: dict[str, Any], name: str = "hardware.yaml") -> Path:
    path = tmp / name
    path.write_text(yaml.safe_dump({"hardware": hardware}), encoding="utf-8")
    return path


def _system_yaml(tmp: Path, hardware_file: str | None = None, inline: dict | None = None) -> Path:
    data = yaml.safe_load((ROOT / "configs" / "system.yaml").read_text(encoding="utf-8"))
    if hardware_file is not None:
        data["hardware_file"] = hardware_file
    if inline is not None:
        data["hardware"] = inline
    path = tmp / "system.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def hub():
    h = HardwareHub()
    yield h
    h.shutdown()


# ------------------------------------------------------------------ ownership and lifecycle
def test_hardware_page_and_ai_runtime_get_the_same_instance(hub: HardwareHub) -> None:
    before = _service_threads()
    page_service, reason = hub.for_hardware_page(HW_FILE)
    assert reason is None and page_service is not None
    ai_service = hub.acquire_for_ai(load_system_config(SYS_SHARED), str(SYS_SHARED))
    assert ai_service is page_service and hub.ai_attached
    assert hub.for_hardware_page(HW_FILE)[0] is page_service                 # every later visit: same object
    assert _service_threads() == before + 1                                   # one worker, one connection
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(SYS_SHARED), SIZE, session_id="shared_1",
                          hardware=ai_service)
    try:
        assert rt.hardware is page_service and "hardware" in rt.components
    finally:
        rt.close()
    hub.release_ai()
    assert page_service.running and not hub.ai_attached                       # AI stopped, hardware kept
    assert _wait_for(lambda: page_service.status().link == LinkState.CONNECTED, 5)


def test_concurrent_first_use_creates_one_service_and_no_port_conflict(hub: HardwareHub, tmp_path: Path) -> None:
    pytest.importorskip("websockets")
    port = _free_port()
    path = _hw_yaml(tmp_path, {"enabled": True, "mode": "esp32", "esp32": {"port": port}})
    results: list[Any] = []
    barrier = threading.Barrier(8)

    def first_visit() -> None:
        barrier.wait()
        results.append(hub.for_hardware_page(path)[0])

    threads = [threading.Thread(target=first_visit) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(10)
    assert len(results) == 8 and len({id(s) for s in results}) == 1           # exactly one instance
    status = results[0].status()
    assert status.link != LinkState.FAULT and status.transport.active         # one listener, no "cannot listen"
    hub.shutdown()
    assert _wait_for(lambda: _port_is_free(port), 5) and not results[0].running


def test_a_different_configuration_is_refused_while_running(hub: HardwareHub, tmp_path: Path) -> None:
    service, _ = hub.for_hardware_page(_hw_yaml(tmp_path, {"enabled": True, "components": COMPONENTS}))
    other = load_system_config(_system_yaml(tmp_path, inline={"enabled": True, "device_id": "other-device",
                                                               "components": COMPONENTS}))
    before = _service_threads()
    with pytest.raises(HardwareShareConflict, match="different hardware settings"):
        hub.acquire_for_ai(other, "other.yaml")
    assert hub.service is service and service.running and not hub.ai_attached
    assert _service_threads() == before                                       # nothing new was started
    service.stop()                                                            # operator pressed Disconnect
    replacement = hub.acquire_for_ai(other, "other.yaml")
    assert replacement is not service and replacement.config.device_id == "other-device"
    assert hub.service is replacement and not service.running


def test_ai_only_configuration_leaves_the_shared_service_alone(hub: HardwareHub) -> None:
    service, _ = hub.for_hardware_page(HW_FILE)
    assert hub.acquire_for_ai(load_system_config(ROOT / "configs" / "system.yaml"), "system.yaml") is None
    assert hub.service is service and service.running and not hub.ai_attached


def test_reload_is_refused_while_the_ai_uses_the_service(hub: HardwareHub, tmp_path: Path) -> None:
    service, _ = hub.for_hardware_page(HW_FILE)
    hub.acquire_for_ai(load_system_config(SYS_SHARED), str(SYS_SHARED))
    assert "refused" in hub.reload(HW_FILE) and hub.service is service and service.running
    hub.release_ai()
    assert "reloaded" in hub.reload(_hw_yaml(tmp_path, {"enabled": True, "components": COMPONENTS}))
    assert hub.service is not service and not service.running
    assert "disabled" in hub.reload(_hw_yaml(tmp_path, {"enabled": False}, "off.yaml")) and hub.service is None


def test_configuration_errors_are_shared_not_replaced_by_simulation(hub: HardwareHub, tmp_path: Path) -> None:
    (tmp_path / "hardware.yaml").write_text("hardware:\n  mode: serial\n", encoding="utf-8")
    service, _ = hub.for_hardware_page(tmp_path / "hardware.yaml")
    assert service.status().config_error and service.status().link == LinkState.FAULT
    assert service.simulator is None and service.status().mode == "unconfigured"


# ------------------------------------------------------------------ e-stop and dispatch through the shared service
def test_estop_is_one_latch_for_both_pages(hub: HardwareHub) -> None:
    page_service, _ = hub.for_hardware_page(HW_FILE)
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(SYS_SHARED), SIZE, session_id="shared_estop",
                          hardware=hub.acquire_for_ai(load_system_config(SYS_SHARED), str(SYS_SHARED)))
    try:
        assert _wait_for(lambda: page_service.status().link == LinkState.CONNECTED, 5)
        rt.hardware.emergency_stop("lifeguard_1")                              # pressed on the AI Dashboard page
        assert page_service.status().estop_latched                             # ... latched on the Hardware page
        assert rt.snapshot().hardware.estop_latched
        assert _wait_for(lambda: page_service.status().telemetry.estop_engaged is True, 5)
        msg = HardwareConsole(page_service).command("clear_estop", "lifeguard_1", confirmed=True)   # Hardware page
        assert "clear_emergency_stop" in msg
        assert _wait_for(lambda: not rt.snapshot().hardware.estop_latched, 5)  # ... cleared on the AI page too
    finally:
        rt.close()
        hub.release_ai()
    assert page_service.running


def test_ai_dispatch_commands_go_through_the_shared_service(hub: HardwareHub, tmp_path: Path) -> None:
    hw_path = _hw_yaml(tmp_path, {"enabled": True, "components": COMPONENTS,
                                  "permissions": {"response": ["set_actuator"]},
                                  "on_dispatch": [{"name": "set_actuator", "target": "indicator_1",
                                                   "params": {"state": "on", "duration_s": 2}}]})
    sys_cfg = load_system_config(_system_yaml(tmp_path, hardware_file=hw_path.name))
    page_service, _ = hub.for_hardware_page(hw_path)                          # Hardware page opened first
    shared = hub.acquire_for_ai(sys_cfg, "system.yaml")
    assert shared is page_service
    rt = LifeguardRuntime(load_ai_config(AI), sys_cfg, SIZE, session_id="shared_dispatch", hardware=shared)
    try:
        assert _wait_for(lambda: page_service.status().link == LinkState.CONNECTED, 5)
        alarm = None
        for frame in SyntheticFrameSource(*SIZE, n_frames=1200, start_t_capture=0.0, start_utc=T0):
            alarm = next((e for e in rt.process(frame) if e.to_level == RiskLevel.ALARM), None)
            if alarm is not None:
                break
        assert alarm is not None
        assert not [r for r in page_service.status().commands if r.command.source == "response"]
        _, outcome = rt.acknowledge(alarm.event_id, AckDecision.CONFIRM, "lifeguard_1")
        sent = [r for r in page_service.status().commands if r.command.source == "response"]
        assert len(sent) == 1 and sent[0].command.requested_by == "lifeguard_1"   # visible on the Hardware page
        assert outcome is not None and "hardware: set_actuator/indicator_1" in outcome.detail
    finally:
        rt.close()
        hub.release_ai()
    assert page_service.running and len([r for r in page_service.status().commands
                                         if r.command.source == "response"]) == 1


def test_dashboard_controller_borrows_and_returns_the_shared_service(hub: HardwareHub, tmp_path: Path) -> None:
    page_service, _ = hub.for_hardware_page(HW_FILE)
    ctl = DashboardController(hub=hub)
    choice = SourceChoice("synthetic", "", False)
    try:
        assert ctl.start(AI, SYS_SHARED, choice), ctl.error
        assert ctl.runtime.hardware is page_service and hub.ai_attached
        assert _wait_for(lambda: ctl.runtime.snapshot().summary.frames > 5, 10)
    finally:
        ctl.stop()
    assert not hub.ai_attached and page_service.running and hub.service is page_service
    before = _service_threads()
    conflict = _system_yaml(tmp_path, inline={"enabled": True, "device_id": "other-device"})
    assert ctl.start(AI, conflict, choice) is False and "HardwareShareConflict" in ctl.error
    assert ctl.runtime is None and not hub.ai_attached and _service_threads() == before


def test_standalone_runtime_still_owns_its_hardware() -> None:
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(SYS_SHARED), SIZE, session_id="owned")
    service = rt.hardware
    assert service is not None and service.running
    rt.close()
    assert not service.running                                                # owned -> stopped, as before


# ------------------------------------------------------------------ the two-page Streamlit app
def test_two_page_app_shares_state_across_pages_and_sessions(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LIFEGUARD_HARDWARE_CONFIG", str(HW_FILE))
    app = str(ROOT / "src" / "lifeguard" / "app" / "lifeguard_app.py")
    hub = shared_hub()
    hub.shutdown()
    try:
        first = AppTest.from_file(app, default_timeout=60)
        first.run()
        assert not first.exception and any(b.label == "Start" for b in first.sidebar.button)   # AI Dashboard
        first.switch_page("app_pages/hardware.py")
        first.run()

        def page(at: AppTest) -> str:
            return " ".join(m.value for m in at.markdown)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and "CONNECTED" not in page(first):
            time.sleep(0.3)
            first.run()
        assert "SIMULATION" in page(first) and "CONNECTED" in page(first) and not first.exception
        service = hub.service
        next(b for b in first.button if b.label == "EMERGENCY STOP").click()
        first.run()
        assert service.status().estop_latched and "EMERGENCY STOP ACTIVE" in page(first)
        second = AppTest.from_file(app, default_timeout=60)                    # another tab / browser session
        second.run()
        second.switch_page("app_pages/hardware.py")
        second.run()
        assert hub.service is service and "EMERGENCY STOP ACTIVE" in page(second) and not second.exception
    finally:
        _bounded(hub.shutdown, 20, "hub.shutdown()")
