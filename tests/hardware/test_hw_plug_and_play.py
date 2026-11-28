"""Configuration-driven plug & play: AI-only / simulation / real-ESP32 modes chosen by YAML only.

No physical ESP32: the "real" link is exercised against the fake ESP32 (the reference device model) over real
loopback WebSockets, in both link roles; protocol edge cases use the scripted in-memory transport.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from lifeguard.ai.config import load_ai_config
from lifeguard.app.hardware_view import HardwareConsole, HardwareDemo, clear_blockers, hardware_state
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.errors import ConfigError
from lifeguard.hardware.config import HardwareConfig, HardwareConfigError, load_hardware_config
from lifeguard.hardware.esp32 import Esp32Adapter
from lifeguard.hardware.models import Ack, CommandStatus, LinkState
from lifeguard.hardware.service import HardwareService
from lifeguard.system.config import load_system_config
from lifeguard.system.ingest.live import SyntheticFrameSource
from tests.hardware.test_hw_reliability import _bounded, _free_port, _port_is_free, _wait_for
from tests.hardware.test_hw_runtime_dashboard import AI, SIZE, T0
from tests.hardware.util import ScriptedTransport, config, msg, run

ROOT = Path(__file__).resolve().parents[2]
FAST = {"link_timeout_s": 1.5, "heartbeat_interval_s": 0.3, "telemetry_interval_s": 0.2, "stale_telemetry_s": 1.0,
        "hello_timeout_s": 1.0}
COMPONENTS = {"motors": {"motor_1": {}}, "actuators": {"indicator_1": {}}, "sensors": {"sensor_1": {"unit": "cm"}}}


# ------------------------------------------------------------------ helpers
def _write(tmp: Path, hardware: dict[str, Any], name: str = "hardware.yaml") -> Path:
    path = tmp / name
    path.write_text(yaml.safe_dump({"hardware": hardware}), encoding="utf-8")
    return path


def _system_with(tmp: Path, hardware_file: str) -> Path:
    """A system YAML that is configs/system.yaml plus ``hardware_file`` (like configs/system_hardware.yaml)."""
    data = yaml.safe_load((ROOT / "configs" / "system.yaml").read_text(encoding="utf-8"))
    data["hardware_file"] = hardware_file
    path = tmp / "system.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _client_cfg(port: int, **extra: Any) -> dict[str, Any]:
    esp32 = {"role": "client", "device_host": "127.0.0.1", "device_port": port,
             "reconnect": {"initial_backoff_s": 0.2, "max_backoff_s": 0.5}, **extra.pop("esp32", {})}
    return {"enabled": True, "mode": "esp32", **FAST, "components": COMPONENTS, "esp32": esp32, **extra}


class FakeEsp32Server:
    """The fake ESP32 as a WebSocket server (firmware side of ``role: client``)."""

    def __init__(self, cfg: HardwareConfig, port: int, token: str | None = None) -> None:
        from lifeguard.hardware import fake_device

        self.stop = threading.Event()
        ready = threading.Event()
        self.thread = threading.Thread(target=fake_device.serve,
                                       args=(cfg, "127.0.0.1", port, self.stop),
                                       kwargs={"token": token, "ready": ready}, daemon=True)
        self.thread.start()
        assert ready.wait(5)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)


def _process_for(rt: LifeguardRuntime, seconds: float) -> float:
    """Process frames for ``seconds``; returns the slowest ``process()`` call."""
    worst, end = 0.0, time.monotonic() + seconds
    for frame in SyntheticFrameSource(*SIZE, n_frames=100000, start_t_capture=0.0, start_utc=T0):
        _, took = _bounded(lambda f=frame: rt.process(f), 20, "runtime.process()")
        worst = max(worst, took)
        if time.monotonic() > end:
            return worst
        time.sleep(0.02)
    return worst


# ------------------------------------------------------------------ configuration
def test_shipped_hardware_file_is_simulation_and_safe() -> None:
    cfg = load_hardware_config(ROOT / "configs" / "hardware.yaml")
    assert cfg.enabled and cfg.mode == "simulation" and cfg.allow_physical_actuation is False
    assert cfg.esp32.auth_token_env is None and cfg.estop_clear.pin_env is None    # no secrets in the file
    assert cfg.on_dispatch == () and cfg.auto_connect
    sys_cfg = load_system_config(ROOT / "configs" / "system_hardware.yaml")
    assert sys_cfg.hardware_error is None and sys_cfg.hardware == cfg
    assert load_system_config(ROOT / "configs" / "system.yaml").hardware.enabled is False    # AI-only default


def test_real_hardware_configuration_loading(tmp_path: Path) -> None:
    client = load_hardware_config(_write(tmp_path, _client_cfg(9000, esp32={"auth_token_env": "LIFEGUARD_HW_T"})))
    assert client.esp32.role == "client" and client.esp32.endpoint == "ws://127.0.0.1:9000/lifeguard-hw"
    server = load_hardware_config(_write(tmp_path, {
        "enabled": True, "mode": "esp32", "esp32": {"bind_host": "0.0.0.0", "allowed_peers": ["192.168.4.20"],
                                                    "auth_token_env": "LIFEGUARD_HW_T"},
        "allow_physical_actuation": True}, "server.yaml"))
    assert server.esp32.role == "server" and server.esp32.allowed_peers == ("192.168.4.20",)
    example = load_system_config(ROOT / "configs" / "system_hardware_esp32.example.yaml").hardware
    assert example.mode == "esp32" and example.allow_physical_actuation is False     # inline blocks still work


@pytest.mark.parametrize("bad, why", [
    ({"mode": "esp32", "esp32": {"role": "client"}}, "device_host"),
    ({"esp32": {"device_host": "http://10.0.0.5:80"}}, "device_host"),
    ({"esp32": {"bind_host": "not a host!"}}, "bind_host"),
    ({"esp32": {"allowed_peers": ["esp32.local"]}}, "allowed_peers"),
    ({"permissions": {"response": ["clear_emergency_stop"]}}, "clear_emergency_stop"),
    ({"permissions": {"system": ["clear_emergency_stop"]}}, "clear_emergency_stop"),
    ({"mode": "esp32", "allow_physical_actuation": True, "esp32": {"bind_host": "0.0.0.0"}}, "auth_token_env"),
    ({"esp32": {"reconnect": {"initial_backoff_s": 5, "max_backoff_s": 1}}}, "max_backoff_s"),
    ({"esp32": {"token": "s3cret"}}, "token"),                                # secrets cannot be put in the file
    ({"unknown_key": 1}, "unknown_key"),
])
def test_invalid_hardware_files_are_rejected_with_a_reason(tmp_path: Path, bad: dict, why: str) -> None:
    with pytest.raises(HardwareConfigError, match=why):
        load_hardware_config(_write(tmp_path, {"enabled": True, **bad}))


def test_hardware_file_shape_and_inline_conflict(tmp_path: Path) -> None:
    (tmp_path / "flat.yaml").write_text("enabled: true\nmode: simulation\n", encoding="utf-8")
    with pytest.raises(HardwareConfigError, match="exactly one top-level"):
        load_hardware_config(tmp_path / "flat.yaml")
    (tmp_path / "broken.yaml").write_text("hardware: [unclosed\n", encoding="utf-8")
    with pytest.raises(HardwareConfigError, match="invalid YAML"):
        load_hardware_config(tmp_path / "broken.yaml")
    data = yaml.safe_load((ROOT / "configs" / "system.yaml").read_text(encoding="utf-8"))
    data.update(hardware_file="hardware.yaml", hardware={"enabled": True})
    (tmp_path / "both.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="not both"):
        load_system_config(tmp_path / "both.yaml")


# ------------------------------------------------------------------ the three operating modes
def test_ai_only_mode_without_esp32(tmp_path: Path) -> None:
    sys_path = _system_with(tmp_path, _write(tmp_path, {"enabled": False}).name)
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(sys_path), SIZE, session_id="ai_only")
    try:
        assert rt.hardware is None and rt.snapshot().hardware is None and "hardware" not in rt.components
        assert _process_for(rt, 1.0) < 1.0 and rt.snapshot().summary.frames > 10
    finally:
        rt.close()


def test_simulation_mode_is_labelled_and_never_real() -> None:
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(ROOT / "configs" / "system_hardware.yaml"), SIZE,
                          session_id="sim_mode")
    try:
        assert "simulation" in rt.components["hardware"] and "no physical hardware" in rt.components["hardware"]
        assert _wait_for(lambda: rt.snapshot().hardware.link == LinkState.CONNECTED, 5)
        hw = rt.snapshot().hardware
        assert hw.simulated and hw.transport.role == "simulation" and hw.telemetry.source == "simulation"
        assert hardware_state(hw).label == "CONNECTED"
    finally:
        rt.close()


@pytest.mark.parametrize("content", [None, "hardware:\n  enabled: true\n  mode: serial\n"])
def test_missing_or_invalid_hardware_file_never_stops_the_ai(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "hardware.yaml").write_text(content, encoding="utf-8")
    sys_cfg = load_system_config(_system_with(tmp_path, "hardware.yaml"))     # loading itself never raises
    assert sys_cfg.hardware_error is not None
    rt = LifeguardRuntime(load_ai_config(AI), sys_cfg, SIZE, session_id="bad_hw")
    try:
        hw = rt.snapshot().hardware
        assert hw.link == LinkState.FAULT and hw.config_error and hw.mode == "unconfigured"   # never "simulation"
        assert hardware_state(hw).label == "CONFIGURATION ERROR"
        assert rt.hardware.request("ping", source="operator", by="op").status == CommandStatus.REJECTED
        assert rt.hardware.start() is False and rt.hardware.simulator is None
        assert _process_for(rt, 1.0) < 1.0
    finally:
        rt.close()


def test_failed_real_connection_is_never_replaced_by_simulation(tmp_path: Path) -> None:
    pytest.importorskip("websockets")
    port = _free_port()                                          # nothing listens there
    cfg = load_hardware_config(_write(tmp_path, _client_cfg(port, esp32={"reconnect": {
        "initial_backoff_s": 0.1, "max_backoff_s": 0.2, "max_attempts": 3}})))
    service = HardwareService(cfg)
    service.start()
    try:
        assert _wait_for(lambda: service.status().transport.gave_up, 10)
        hw = service.status()
        assert hw.mode == "esp32" and not hw.simulated and hw.link == LinkState.DISCONNECTED and hw.telemetry is None
        assert hw.transport.attempt == 3 and "failed" in hw.transport.last_error
        view = hardware_state(hw)
        assert view.label == "DISCONNECTED" and "press Connect" in view.detail
        assert service.simulator is None and "not available" in service.simulate("go_offline")
    finally:
        _bounded(service.stop, 15, "stop()")


# ------------------------------------------------------------------ real link (fake ESP32), both roles
def test_client_role_handshake_identity_telemetry_and_reconnect(tmp_path: Path) -> None:
    pytest.importorskip("websockets")
    port = _free_port()
    cfg = load_hardware_config(_write(tmp_path, _client_cfg(port, allow_physical_actuation=True,
                                                            esp32={"auth_token_env": "LIFEGUARD_HW_PNP"})))
    import os
    os.environ["LIFEGUARD_HW_PNP"] = "tok-ä-1"                   # non-ASCII token works end to end
    device = FakeEsp32Server(cfg, port, token="tok-ä-1")
    service = HardwareService(cfg)
    try:
        service.start()
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED
                         and service.status().telemetry is not None, 10)
        hw = service.status()
        assert hw.device.firmware == "fake-esp32-0.1" and hw.transport.role == "client"
        assert hardware_state(hw).label == "CONNECTED" and hw.last_heard_age_s is not None
        assert any(r.command.name == "safe_stop" for r in hw.commands)           # nothing moves on connect
        assert not any(e.text.count("tok-") for e in hw.comm_log)                 # token never logged
        rec = service.request("set_motor", "motor_1", {"speed": 0.2, "duration_s": 0.3}, source="operator", by="op")
        assert _wait_for(lambda: _rec(service, rec).status == CommandStatus.COMPLETED, 5)
        device.close()                                                            # the ESP32 goes away
        assert _wait_for(lambda: hardware_state(service.status()).label == "RECONNECTING", 10)
        device = FakeEsp32Server(cfg, port, token="tok-ä-1")                      # ... and comes back
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 10)
        assert service.status().stats.connects >= 2
    finally:
        os.environ.pop("LIFEGUARD_HW_PNP", None)
        _bounded(service.stop, 15, "stop()")
        device.close()


def _rec(service: HardwareService, record: Any) -> Any:
    return {r.command.cmd_id: r for r in service.status().commands}[record.command.cmd_id]


@pytest.mark.parametrize("problem", ["token", "identity"])
def test_authentication_failure_and_wrong_identity_never_connect(tmp_path: Path, monkeypatch,
                                                                problem: str) -> None:
    pytest.importorskip("websockets")
    monkeypatch.setenv("LIFEGUARD_HW_PNP2", "right-token")
    port = _free_port()
    cfg = load_hardware_config(_write(tmp_path, _client_cfg(port, esp32={"auth_token_env": "LIFEGUARD_HW_PNP2"})))
    device_cfg = cfg if problem == "token" else cfg.model_copy(update={"device_id": "someone-else"})
    device = FakeEsp32Server(device_cfg, port, token="wrong-token" if problem == "token" else "right-token")
    service = HardwareService(cfg)
    try:
        service.start()
        expected = "authentication failed" if problem == "token" else "unexpected device id"
        assert _wait_for(lambda: any(expected in e.text for e in service.status().comm_log), 10)
        time.sleep(0.5)
        assert service.status().link != LinkState.CONNECTED and service.status().device is None
    finally:
        _bounded(service.stop, 15, "stop()")
        device.close()


def test_device_disconnection_during_video_processing(tmp_path: Path) -> None:
    """The ESP32 vanishes and returns while frames are processed: no frame waits, the link recovers by itself."""
    pytest.importorskip("websockets")
    port = _free_port()
    hw_path = _write(tmp_path, _client_cfg(port))
    cfg = load_hardware_config(hw_path)
    device = FakeEsp32Server(cfg, port)
    rt = LifeguardRuntime(load_ai_config(AI), load_system_config(_system_with(tmp_path, hw_path.name)), SIZE,
                          session_id="pnp_video")
    stop = threading.Event()
    try:
        assert _wait_for(lambda: rt.snapshot().hardware.link == LinkState.CONNECTED, 10)

        def dashboard() -> None:
            while not stop.is_set():
                rt.snapshot()
                time.sleep(0.2)

        threading.Thread(target=dashboard, daemon=True).start()
        threading.Timer(0.5, device.close).start()                 # device lost mid-run
        worst = _process_for(rt, 3.0)
        assert worst < 1.0, f"process() stalled {worst:.2f} s during a device disconnection"
        assert rt.snapshot().hardware.link != LinkState.CONNECTED
        device = FakeEsp32Server(cfg, port)
        worst = max(worst, _process_for(rt, 0.5))
        assert _wait_for(lambda: rt.snapshot().hardware.link == LinkState.CONNECTED, 10)   # automatic reconnect
        assert worst < 1.0
    finally:
        stop.set()
        _bounded(rt.close, 20, "runtime.close()")
        device.close()


def test_server_role_allowed_peers_refuses_other_devices() -> None:
    pytest.importorskip("websockets")
    from lifeguard.hardware import fake_device

    port = _free_port()
    cfg = config(mode="esp32", **FAST, esp32={"port": port, "allowed_peers": ["10.9.9.9"]})
    service = HardwareService(cfg)
    service.start()
    stop = threading.Event()
    th = threading.Thread(target=fake_device.run, args=(cfg, f"ws://127.0.0.1:{port}/lifeguard-hw", stop),
                          kwargs={"reconnect_s": 0.2}, daemon=True)
    th.start()
    try:
        assert _wait_for(lambda: "not in allowed_peers" in service.status().transport.last_error, 10)
        assert service.status().link != LinkState.CONNECTED
    finally:
        stop.set()
        _bounded(service.stop, 15, "stop()")
        th.join(timeout=5)


# ------------------------------------------------------------------ protocol / safety (scripted transport)
def _scripted(**overrides: Any) -> tuple[HardwareService, ScriptedTransport, Any]:
    from tests.hardware.util import Clock

    transport, clock = ScriptedTransport(), Clock()
    cfg = config(mode="esp32", **{"allow_physical_actuation": True, **overrides})
    service = HardwareService(cfg, adapter=Esp32Adapter(cfg, transport), clock=clock)
    service.start(background=False)
    transport.device_connects()
    transport.push(msg("hello", 1, components={"motors": ["motor_1"], "actuators": ["indicator_1"]}))
    transport.push(msg("telemetry", 2))
    run(service, clock, 0.1)
    assert service.status().link == LinkState.CONNECTED
    return service, transport, clock


def _ack(service: HardwareService, cmd_id: str, status: str) -> None:
    service.commands.on_ack(Ack(cmd_id, status, ""), service._clock())


def test_command_acknowledgement_lifecycle() -> None:
    service, t, clock = _scripted()
    seq = iter(range(10, 1000))
    rec = service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 0.5}, source="operator", by="op")
    assert rec.status == CommandStatus.PENDING and t.sent[-1]["cmd_id"] == rec.command.cmd_id   # sent != done
    t.push(msg("ack", next(seq), cmd_id=rec.command.cmd_id, status="accepted"))
    t.push(msg("ack", next(seq), cmd_id=rec.command.cmd_id, status="accepted"))      # duplicate: ignored
    run(service, clock, 0.1)
    assert _rec(service, rec).status == CommandStatus.ACCEPTED
    t.push(msg("ack", next(seq), cmd_id=rec.command.cmd_id, status="completed"))
    t.push(msg("telemetry", next(seq)))
    run(service, clock, 0.1)
    assert _rec(service, rec).status == CommandStatus.COMPLETED
    t.push(msg("ack", next(seq), cmd_id=rec.command.cmd_id, status="failed"))        # after final: ignored
    t.push(msg("ack", next(seq), cmd_id="c-dead-000001", status="completed"))       # unknown: ignored
    run(service, clock, 0.1)
    assert _rec(service, rec).status == CommandStatus.COMPLETED
    ping = service.request("ping", source="operator", by="op")                       # accepted, never completed
    t.push(msg("ack", next(seq), cmd_id=ping.command.cmd_id, status="accepted"))
    for _ in range(30):
        t.push(msg("telemetry", next(seq)))
        run(service, clock, 0.1)
    assert _rec(service, ping).status == CommandStatus.TIMEOUT                       # not stuck forever
    assert rec.command.cmd_id.count("-") == 2                                        # c-<boot>-<n>: no reuse


def test_estop_send_failure_is_retried_and_reasserted_on_reconnect() -> None:
    service, t, clock = _scripted(emergency_stop_retries=2)
    original, fails = t.send, {"n": 1}

    def flaky(text: str) -> bool:
        if json.loads(text).get("name") == "emergency_stop" and fails["n"] > 0:
            fails["n"] -= 1
            return False
        return original(text)

    t.send = flaky                                                   # type: ignore[method-assign]
    estop = service.emergency_stop("op")
    assert service.status().estop_latched and estop.status == CommandStatus.PENDING and "re-send" in estop.detail
    for i in range(10):
        t.push(msg("telemetry", 50 + i))
        run(service, clock, 0.1)
    sent = [m for m in t.sent if m.get("name") == "emergency_stop"]
    assert len(sent) == 1 and estop.attempts == 2                   # 1st write failed, the re-send got out
    t.push(msg("ack", 99, cmd_id=estop.command.cmd_id, status="completed"))
    run(service, clock, 0.1)
    assert _rec(service, estop).status == CommandStatus.COMPLETED and service.status().estop_latched
    t.device_disconnects()
    run(service, clock, 0.2)
    t.device_connects()                                              # device rebooted (its latch may be gone)
    t.push(msg("hello", 1, components={"motors": ["motor_1"]}))
    run(service, clock, 0.1)
    assert t.sent[-1]["name"] == "emergency_stop"                    # re-asserted, not just a safe stop


def test_estop_unanswered_times_out_but_the_latch_stays() -> None:
    from tests.hardware.util import Clock

    clock = Clock()
    service = HardwareService(config(emergency_stop_retries=2), clock=clock)
    service.start(background=False)
    run(service, clock, 0.3)
    service.simulate("ignore_commands")
    estop = service.emergency_stop("op")
    run(service, clock, 3.0)
    rec = _rec(service, estop)
    assert rec.status == CommandStatus.TIMEOUT and rec.attempts == 3 and "PHYSICAL" in rec.detail
    assert service.status().estop_latched


def test_duplicate_hello_is_refused() -> None:
    service, t, clock = _scripted()
    t.push(msg("hello", 1))                                          # second hello on the same connection
    run(service, clock, 0.1)
    hw = service.status()
    assert hw.link == LinkState.DISCONNECTED and "duplicate hello" in t.dropped[-1]
    assert any("duplicate hello" in e.text for e in hw.comm_log)


def test_commands_to_components_the_device_did_not_announce_are_refused() -> None:
    service, t, clock = _scripted()
    t.device_disconnects()
    run(service, clock, 0.1)
    t.device_connects()
    t.push(msg("hello", 1, components={"actuators": ["indicator_1"]}))      # no motor announced
    t.push(msg("telemetry", 2))
    run(service, clock, 0.1)
    rec = service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 0.5}, source="operator", by="op")
    assert rec.status == CommandStatus.REJECTED and "did not announce" in rec.detail


def test_stale_telemetry_safe_stops_a_running_output() -> None:
    from tests.hardware.util import Clock

    clock = Clock()
    service = HardwareService(config(), clock=clock)
    service.start(background=False)
    run(service, clock, 0.3)
    motor = service.request("set_motor", "motor_1", {"speed": 0.3, "duration_s": 3}, source="operator", by="op")
    run(service, clock, 0.2)
    service.simulate("freeze_telemetry")
    run(service, clock, 1.5)
    hw = service.status()
    assert any(r.command.requested_by == "stale-telemetry" and r.command.name == "safe_stop" for r in hw.commands)
    assert _rec(service, motor).status == CommandStatus.SUPERSEDED


def test_clear_estop_authorisation_and_physical_estop(monkeypatch) -> None:
    from tests.hardware.util import Clock

    monkeypatch.setenv("LIFEGUARD_HW_PIN_T", "4321")
    clock = Clock()
    service = HardwareService(config(estop_clear={"operators": ["Supervisor"], "pin_env": "LIFEGUARD_HW_PIN_T"}),
                              clock=clock)
    service.start(background=False)
    run(service, clock, 0.3)
    console = HardwareConsole(service)
    assert "not latched" in console.command("clear_estop", "supervisor", pin="4321", confirmed=True)
    service.emergency_stop("op")
    run(service, clock, 0.2)
    assert "confirm" in console.command("clear_estop", "supervisor", pin="4321", confirmed=False)
    assert "not authorised" in console.command("clear_estop", "intruder", pin="4321", confirmed=True)
    assert "wrong PIN" in console.command("clear_estop", "supervisor", pin="0000", confirmed=True)
    service.simulate("press_physical_estop")
    run(service, clock, 0.3)
    console.command("clear_estop", "supervisor", pin="4321", confirmed=True)
    run(service, clock, 0.2)
    assert service.status().estop_latched                                 # the device refused: switch pressed
    assert any("physical emergency stop is still engaged" in r.detail for r in service.status().commands)
    service.simulate("release_physical_estop")
    run(service, clock, 0.3)
    console.command("clear_estop", "Supervisor", pin="4321", confirmed=True)
    run(service, clock, 0.2)
    assert service.status().estop_latched is False


# ------------------------------------------------------------------ lifecycle and display
def test_safe_shutdown_and_repeated_connect_disconnect() -> None:
    pytest.importorskip("websockets")
    from lifeguard.hardware import fake_device

    port = _free_port()
    cfg = config(mode="esp32", **FAST, esp32={"port": port})
    service = HardwareService(cfg)
    console = HardwareConsole(service)
    stop = threading.Event()
    th = threading.Thread(target=fake_device.run, args=(cfg, f"ws://127.0.0.1:{port}/lifeguard-hw", stop),
                          kwargs={"reconnect_s": 0.2}, daemon=True)
    th.start()
    threads_before = sum(t.name == "hardware-service" for t in threading.enumerate())
    try:
        for _ in range(4):
            assert console.connect("op") == "link opened"
            assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 10)
            _, took = _bounded(lambda: console.disconnect("op"), 15, "disconnect")
            assert took < 5.0
            hw = service.status()
            assert not hw.running and hardware_state(hw).label == "DISCONNECTED"
            assert any(r.command.requested_by == "shutdown" and r.status == CommandStatus.COMPLETED
                       for r in hw.commands)                              # safe stop acknowledged first
            assert _wait_for(lambda: _port_is_free(port), 5)
        assert sum(t.name == "hardware-service" for t in threading.enumerate()) == threads_before
    finally:
        stop.set()
        _bounded(service.stop, 15, "stop()")
        th.join(timeout=5)


def test_dashboard_state_follows_the_device() -> None:
    service = HardwareService(config())
    console = HardwareConsole(service)
    assert hardware_state(service.status()).label == "DISCONNECTED"            # not started
    service.start()
    try:
        assert _wait_for(lambda: hardware_state(service.status()).label == "CONNECTED", 5)
        assert "not latched" in clear_blockers(service.status())[0]
        console.simulate("go_offline")
        assert _wait_for(lambda: hardware_state(service.status()).label == "RECONNECTING", 5)
        assert service.status().telemetry is None                              # old values not shown as current
        console.simulate("come_online")
        assert _wait_for(lambda: hardware_state(service.status()).label == "CONNECTED", 5)
        console.command("estop", "op")
        assert service.status().estop_latched
    finally:
        service.stop()


def test_switching_modes_is_configuration_only(tmp_path: Path) -> None:
    sim = HardwareDemo(_write(tmp_path, {"enabled": True, "mode": "simulation", "components": COMPONENTS}))
    try:
        assert _wait_for(lambda: sim.service.status().link == LinkState.CONNECTED, 5)
        assert sim.service.status().transport.role == "simulation"
    finally:
        sim.close()
    port = _free_port()
    real = HardwareDemo(_write(tmp_path, {"enabled": True, "mode": "esp32", "esp32": {"port": port}}, "real.yaml"))
    try:
        hw = real.service.status()
        assert hw.transport.role == "server" and not hw.simulated and hw.link == LinkState.DISCONNECTED
    finally:
        real.close()
    off = HardwareDemo(_write(tmp_path, {"enabled": False}, "off.yaml"))
    assert off.service is None and not off.enabled
    manual = HardwareDemo(_write(tmp_path, {"enabled": True, "auto_connect": False}, "manual.yaml"))
    assert manual.service is not None and not manual.service.running            # waits for Connect
    manual.close()


def test_hardware_demo_window_runs_in_simulation(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LIFEGUARD_HARDWARE_CONFIG", str(ROOT / "configs" / "hardware.yaml"))
    at = AppTest.from_file(str(ROOT / "src" / "lifeguard" / "app" / "hardware_demo.py"), default_timeout=60)
    try:
        at.run()
        assert not at.exception

        def page() -> str:
            return " ".join(m.value for m in at.markdown)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and "CONNECTED" not in page():
            time.sleep(0.3)
            at.run()
        assert "SIMULATION" in page() and "CONNECTED" in page()
        next(b for b in at.button if b.label == "EMERGENCY STOP").click()
        at.run()
        assert "EMERGENCY STOP ACTIVE" in page()
        next(b for b in at.button if b.label == "Disconnect").click()
        at.run()
        assert "DISCONNECTED" in page() and not at.exception
    finally:
        HardwareDemo.close_all()
