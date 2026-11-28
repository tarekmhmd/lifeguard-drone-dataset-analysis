"""ESP32 adapter: handshake, identity, token, sequencing, heartbeats, timeouts, reconnection, invalid input.

Uses a scripted in-memory transport (no ESP32, no network) plus one real loopback WebSocket test with the
fake device (skipped if ``websockets`` is not installed).
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from lifeguard.hardware.esp32 import Esp32Adapter
from lifeguard.hardware.models import CommandStatus, LinkState
from lifeguard.hardware.service import HardwareService
from tests.hardware.util import Clock, ScriptedTransport, config, msg, run


def _adapter(**overrides):
    transport = ScriptedTransport()
    clock = Clock()
    cfg = config(mode="esp32", **overrides)
    service = HardwareService(cfg, adapter=Esp32Adapter(cfg, transport), clock=clock)
    service.start(background=False)
    return service, transport, clock


def _hello(transport, seq=1, **fields):
    transport.device_connects()
    transport.push(msg("hello", seq, firmware="fw-1", components={"motors": ["motor_1"],
                                                                 "actuators": ["indicator_1"]}, **fields))


def test_handshake_welcome_and_identity() -> None:
    service, t, clock = _adapter()
    run(service, clock, 0.1)
    assert service.status().link == LinkState.DISCONNECTED               # no device yet
    t.device_connects()
    run(service, clock, 0.1)
    assert service.status().link == LinkState.CONNECTING                 # connected, no hello yet
    t.device_disconnects()
    _hello(t)
    run(service, clock, 0.1)
    status = service.status()
    assert status.link == LinkState.CONNECTED and status.device.firmware == "fw-1"
    kinds = [m["type"] for m in t.sent]
    assert kinds[0] == "welcome" and "command" in kinds                    # safe_stop on connect
    welcome = t.sent[0]
    assert welcome["heartbeat_interval_ms"] == 500 and welcome["link_timeout_ms"] == 2000


def test_wrong_device_id_is_refused() -> None:
    service, t, clock = _adapter()
    t.device_connects()
    t.push(msg("hello", 1, device_id="someone-else"))
    run(service, clock, 0.1)
    assert service.status().link != LinkState.CONNECTED and t.dropped == ["unexpected device id"]
    assert t.sent[-1]["type"] == "error"


def test_token_required_and_checked(monkeypatch) -> None:
    monkeypatch.setenv("LIFEGUARD_HW_TOKEN_TEST", "s3cret")
    service, t, clock = _adapter(esp32={"auth_token_env": "LIFEGUARD_HW_TOKEN_TEST"})
    _hello(t, token="wrong")
    run(service, clock, 0.1)
    assert t.dropped == ["authentication failed"] and service.status().link != LinkState.CONNECTED
    _hello(t, seq=1, token="s3cret")
    run(service, clock, 0.1)
    assert service.status().link == LinkState.CONNECTED


def test_missing_token_env_is_a_fault_not_a_crash(monkeypatch) -> None:
    monkeypatch.delenv("LIFEGUARD_HW_MISSING", raising=False)
    service = HardwareService(config(mode="esp32", esp32={"auth_token_env": "LIFEGUARD_HW_MISSING"}))
    service.start()
    status = service.status()
    assert status.link == LinkState.FAULT and "LIFEGUARD_HW_MISSING" in status.errors[-1]
    assert service.request("ping", source="operator", by="op").status == CommandStatus.REJECTED
    service.stop()


def test_messages_before_hello_and_invalid_messages_are_counted() -> None:
    service, t, clock = _adapter()
    t.device_connects()
    t.push(msg("telemetry", 1))
    t.push("{broken")
    run(service, clock, 0.1)
    stats = service.status().stats
    assert stats.invalid_messages == 2 and service.status().link == LinkState.CONNECTING


def test_telemetry_sequence_and_unavailable_values() -> None:
    service, t, clock = _adapter()
    _hello(t)
    t.push(msg("telemetry", 2, sensors={"sensor_1": {"value": 150.0}}, motors={"motor_1": {"state": "stopped",
                                                                                         "speed": 0.0}}))
    t.push(msg("telemetry", 2, sensors={"sensor_1": {"value": 1.0}}))   # duplicate seq: dropped
    run(service, clock, 0.1)
    status = service.status()
    assert status.stats.duplicate_or_old_seq == 1
    sensors = status.telemetry.sensors
    assert sensors["sensor_1"].value == 150.0 and sensors["sensor_1"].status == "out_of_range"
    assert sensors["sensor_2"].value is None and sensors["sensor_2"].status == "unavailable"
    assert status.telemetry.actuators["indicator_1"].state is None          # not reported -> unavailable
    assert status.telemetry.power is None


def test_heartbeats_and_link_timeout_then_reconnect() -> None:
    service, t, clock = _adapter()
    _hello(t)
    run(service, clock, 1.2)
    assert sum(1 for m in t.sent if m["type"] == "heartbeat") >= 2
    run(service, clock, 1.5)                                              # device silent > 2 s
    status = service.status()
    assert status.link == LinkState.DISCONNECTED and "link timeout" in t.dropped
    _hello(t, seq=1)                                                      # device reconnects, seq restarts
    run(service, clock, 0.1)
    assert service.status().link == LinkState.CONNECTED and service.status().stats.connects == 2


def test_ack_routing_and_physical_gate() -> None:
    service, t, clock = _adapter()
    _hello(t)
    t.push(msg("telemetry", 2))
    run(service, clock, 0.1)
    blocked = service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1}, source="operator", by="op")
    assert blocked.status == CommandStatus.REJECTED and "physical actuation is disabled" in blocked.detail
    ping = service.request("ping", source="operator", by="op")
    t.push(msg("ack", 3, cmd_id=ping.command.cmd_id, status="completed", detail="pong"))
    run(service, clock, 0.1)
    assert {r.command.cmd_id: r for r in service.status().commands}[ping.command.cmd_id].status == \
        CommandStatus.COMPLETED


def test_physical_actuation_only_when_explicitly_allowed() -> None:
    service, t, clock = _adapter(allow_physical_actuation=True)
    _hello(t)
    t.push(msg("telemetry", 2))
    run(service, clock, 0.1)
    rec = service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1}, source="operator", by="op")
    assert rec.status == CommandStatus.PENDING
    assert t.sent[-1]["name"] == "set_motor" and t.sent[-1]["params"] == {"speed": 0.1, "duration_ms": 1000}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_websocket_loopback_with_fake_device() -> None:
    pytest.importorskip("websockets")
    from lifeguard.hardware import fake_device

    port = _free_port()
    cfg = config(mode="esp32", allow_physical_actuation=True, esp32={"port": port}, telemetry_interval_s=0.1,
                 heartbeat_interval_s=0.2, stale_telemetry_s=1.0)
    service = HardwareService(cfg)
    service.start()
    stop = threading.Event()
    device = threading.Thread(target=fake_device.run,
                              args=(cfg, f"ws://127.0.0.1:{port}/lifeguard-hw", stop), daemon=True)
    device.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (service.status().link != LinkState.CONNECTED
                                               or service.status().telemetry is None):
            time.sleep(0.05)
        assert service.status().link == LinkState.CONNECTED
        rec = service.request("set_motor", "motor_1", {"speed": 0.2, "duration_s": 0.3}, source="operator",
                              by="op")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            done = {r.command.cmd_id: r for r in service.status().commands}[rec.command.cmd_id]
            if done.status == CommandStatus.COMPLETED:
                break
            time.sleep(0.05)
        assert done.status == CommandStatus.COMPLETED                     # accepted, ran, completed over the wire
        stop.set()                                                        # device goes away
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and service.status().link == LinkState.CONNECTED:
            time.sleep(0.05)
        assert service.status().link == LinkState.DISCONNECTED
    finally:
        stop.set()
        service.stop()
        device.join(timeout=5)
