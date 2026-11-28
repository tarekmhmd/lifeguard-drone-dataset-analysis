"""Patch H1 - runtime reliability: a dead, silent or stuck hardware link never stalls the AI pipeline.

Regression for the audit finding: with a silent WebSocket peer, ``transport.drop()`` -> ``conn.close()`` waited for
the websockets default close timeout (10 s) while ``HardwareService.step`` held the service lock, so
``status()``, ``LifeguardRuntime.snapshot()`` and through the runtime lock ``LifeguardRuntime.process()`` stalled.

Thresholds are deliberately loose (the original stall was ~10 s; a healthy call takes milliseconds), and every
potentially hanging call runs under ``_bounded`` so a regression fails instead of hanging the suite.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.hardware.esp32 import Esp32Adapter
from lifeguard.hardware.models import CommandStatus, LinkState
from lifeguard.hardware.service import HardwareService
from lifeguard.system.ingest.live import SyntheticFrameSource
from tests.hardware.test_hw_runtime_dashboard import AI, SIZE, T0, _sys
from tests.hardware.util import ScriptedTransport, config, msg

STALL_LIMIT_S = 1.0          # any single call slower than this counts as a stall (original: ~10 s)
LINK_TIMEOUT_S = 1.5
FAST = {"link_timeout_s": LINK_TIMEOUT_S, "heartbeat_interval_s": 0.3, "telemetry_interval_s": 0.2,
        "stale_telemetry_s": 1.0}


# ------------------------------------------------------------------ helpers
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bounded(fn: Callable[[], Any], limit_s: float, what: str) -> tuple[Any, float]:
    """Run ``fn`` on a thread; fail (instead of hanging) if it does not return within ``limit_s``."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as err:  # noqa: BLE001 - re-raised below
            box["error"] = err

    start = time.monotonic()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(limit_s)
    elapsed = time.monotonic() - start
    assert not th.is_alive(), f"{what} did not return within {limit_s:.0f} s"
    if "error" in box:
        raise box["error"]
    return box.get("value"), elapsed


def _wait_for(cond: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _frame(payload: bytes) -> bytes:
    """One masked client text frame."""
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack("!H", n))
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _silent_client(port: int, hello: bool = True) -> socket.socket:
    """A WebSocket client that completes the handshake, optionally says hello, then never reads or answers.

    This is the half-open / frozen device: it does not answer pings or the closing handshake.
    """
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /lifeguard-hw HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    assert b" 101 " in s.recv(4096)
    if hello:
        s.sendall(_frame(json.dumps(msg("hello", 1, firmware="silent")).encode()))
    return s


def _esp32_service(port: int, **overrides: Any) -> HardwareService:
    return HardwareService(config(mode="esp32", esp32={"port": port}, **{**FAST, **overrides}))


def _port_is_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ------------------------------------------------------------------ 1. dashboard status polling
def test_silent_peer_never_blocks_status_polling() -> None:
    pytest.importorskip("websockets")
    port = _free_port()
    service = _esp32_service(port)
    service.start()
    peer = _silent_client(port)
    try:
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 5)
        worst = 0.0
        end = time.monotonic() + LINK_TIMEOUT_S + 2.5          # across the link timeout and the drop
        while time.monotonic() < end:
            _, took = _bounded(service.status, 15, "status()")
            worst = max(worst, took)
            time.sleep(0.02)
        assert worst < STALL_LIMIT_S, f"status() stalled for {worst:.2f} s"
        assert service.status().link == LinkState.DISCONNECTED     # the silent device was dropped
        assert service.status().stats.disconnects >= 1
    finally:
        peer.close()
        _bounded(service.stop, 15, "stop()")


# ------------------------------------------------------------------ 2. AI runtime (the critical regression)
def test_silent_peer_never_stalls_runtime_process() -> None:
    """A stalled hardware transport must not stall ``LifeguardRuntime.process()`` (dashboard polling too)."""
    pytest.importorskip("websockets")
    port = _free_port()
    hw = {"enabled": True, "mode": "esp32", "esp32": {"port": port}, **FAST,
          "components": {"motors": {"motor_1": {}}, "actuators": {"indicator_1": {}}, "sensors": {"sensor_1": {}}}}
    rt = LifeguardRuntime(load_ai_config(AI), _sys(hw), SIZE, session_id="h1_stall")
    stop = threading.Event()
    peer = None
    try:
        peer = _silent_client(port)
        assert _wait_for(lambda: rt.hardware.status().link == LinkState.CONNECTED, 5)

        def dashboard() -> None:                               # the Streamlit fragment polls snapshots
            while not stop.is_set():
                rt.snapshot()
                time.sleep(0.2)

        threading.Thread(target=dashboard, daemon=True).start()
        frames = SyntheticFrameSource(*SIZE, n_frames=100000, start_t_capture=0.0, start_utc=T0)
        worst, n = 0.0, 0
        end = time.monotonic() + LINK_TIMEOUT_S + 3.0
        for frame in frames:
            _, took = _bounded(lambda f=frame: rt.process(f), 20, "runtime.process()")
            worst, n = max(worst, took), n + 1
            if time.monotonic() > end:
                break
            time.sleep(0.02)
        assert worst < STALL_LIMIT_S, f"runtime.process() stalled for {worst:.2f} s with a silent hardware peer"
        assert n > 50
        assert rt.hardware.status().link == LinkState.DISCONNECTED
    finally:
        stop.set()
        if peer is not None:
            peer.close()
        _bounded(rt.close, 20, "runtime.close()")


# ------------------------------------------------------------------ 3. bounded WebSocket close
def test_websocket_close_is_bounded_and_does_not_block_the_caller() -> None:
    pytest.importorskip("websockets")
    from lifeguard.hardware import transport as transport_module
    from lifeguard.hardware.transport import WebSocketServerTransport

    close_timeout_s = getattr(transport_module, "CLOSE_TIMEOUT_S", 1.0)
    port = _free_port()
    t = WebSocketServerTransport("127.0.0.1", port, "/lifeguard-hw", 16384)
    t.start()
    peer = _silent_client(port, hello=False)
    try:
        assert _wait_for(lambda: t.connected, 5)
        _, took = _bounded(lambda: t.drop("test"), 5, "drop()")
        assert took < 0.5 and not t.connected                      # returns at once, slot released at once
        start, data = time.monotonic(), b""
        peer.settimeout(8)
        while True:                                                 # the server ends the TCP connection
            try:
                chunk = peer.recv(4096)
            except (ConnectionResetError, ConnectionAbortedError):
                break
            if not chunk:
                break
            data += chunk
        closed_after = time.monotonic() - start
        assert closed_after < close_timeout_s + 2.0, f"connection closed only after {closed_after:.1f} s"
        conn_before = t.connection_id
        other = _silent_client(port, hello=False)                   # a new device can connect right away
        assert _wait_for(lambda: t.connected and t.connection_id == conn_before + 1, 5)
        other.close()
    finally:
        peer.close()
        _bounded(t.stop, 15, "transport.stop()")
    assert _port_is_free(port)


# ------------------------------------------------------------------ 4. shutdown with a stuck adapter
class _StuckTransport(ScriptedTransport):
    """Scripted transport whose ``receive`` can be made to block (a misbehaving link)."""

    def __init__(self) -> None:
        super().__init__()
        self.block = threading.Event()
        self.release = threading.Event()
        self.inside = threading.Event()
        self.stopped = False

    def receive(self, now: float) -> list[str | bytes]:
        if self.block.is_set():
            self.inside.set()
            self.release.wait(30)
        return super().receive(now)

    def stop(self) -> None:
        self.stopped = True
        super().stop()


def test_shutdown_while_transport_polling_is_blocked() -> None:
    transport = _StuckTransport()
    cfg = config(mode="esp32", **FAST)
    service = HardwareService(cfg, adapter=Esp32Adapter(cfg, transport))
    service.start()
    try:
        transport.device_connects()
        transport.push(msg("hello", 1, components={"motors": ["motor_1"]}))
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 5)
        transport.block.set()
        assert transport.inside.wait(5)                             # the service thread is now stuck in poll
        status, took = _bounded(service.status, 5, "status()")
        assert took < STALL_LIMIT_S and status.link == LinkState.CONNECTED
        _, took = _bounded(lambda: service.emergency_stop("op"), 5, "emergency_stop()")
        assert took < STALL_LIMIT_S and service.status().estop_latched   # host latch set even while stuck
        _, took = _bounded(service.stop, 15, "stop()")
        assert took < 8.0
        assert transport.stopped                                    # the link was closed anyway
        assert any("did not stop" in e for e in service.status().errors)
        assert not service.running
    finally:
        transport.release.set()


# ------------------------------------------------------------------ 5. runtime.close() with a failing cleanup
def test_runtime_close_stops_hardware_even_if_store_close_raises() -> None:
    pytest.importorskip("websockets")
    port = _free_port()
    hw = {"enabled": True, "mode": "esp32", "esp32": {"port": port}, **FAST, "components": {"motors": {"motor_1": {}}}}
    before = _service_threads()
    rt = LifeguardRuntime(load_ai_config(AI), _sys(hw), SIZE, session_id="h1_close")
    assert rt.hardware is not None and _service_threads() == before + 1 and not _port_is_free(port)

    def boom() -> None:
        raise OSError("disk full while closing the incident store")

    rt.store.close = boom                                            # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):                 # the cleanup error is not swallowed
        _bounded(rt.close, 20, "runtime.close()")
    assert _wait_for(lambda: _service_threads() == before, 5)       # worker thread joined
    assert rt.hardware.status().link == LinkState.DISCONNECTED
    assert _wait_for(lambda: _port_is_free(port), 5)                 # no server left listening


# ------------------------------------------------------------------ 6. idempotent start
def _service_threads() -> int:
    return sum(t.name == "hardware-service" and t.is_alive() for t in threading.enumerate())


def test_start_twice_is_a_no_op() -> None:
    before = _service_threads()
    sim = HardwareService(config())
    sim.start()
    sim.start()
    try:
        assert _service_threads() == before + 1
    finally:
        sim.stop()
    assert _wait_for(lambda: _service_threads() == before, 5)

    pytest.importorskip("websockets")
    from lifeguard.hardware import fake_device

    port = _free_port()
    cfg = config(mode="esp32", esp32={"port": port}, **FAST)
    service = HardwareService(cfg)
    service.start()
    service.start()                                                  # second call: no second server, no FAULT
    stop = threading.Event()
    device = threading.Thread(target=fake_device.run, args=(cfg, f"ws://127.0.0.1:{port}/lifeguard-hw", stop),
                              kwargs={"reconnect_s": 0.2}, daemon=True)
    device.start()
    try:
        assert service.status().link != LinkState.FAULT
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 10)
    finally:
        stop.set()
        _bounded(service.stop, 15, "stop()")
        device.join(timeout=5)


def test_start_after_a_failed_start_and_with_a_bad_configuration(monkeypatch) -> None:
    pytest.importorskip("websockets")
    port = _free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen()
    service = _esp32_service(port)
    try:
        assert service.start() is False and service.status().link == LinkState.FAULT
        assert not service.running
    finally:
        blocker.close()
    assert service.start() is True                                   # the port is free now: retry succeeds
    try:
        assert service.status().link == LinkState.DISCONNECTED and service.running
    finally:
        _bounded(service.stop, 15, "stop()")

    monkeypatch.delenv("LIFEGUARD_HW_H1_MISSING", raising=False)
    broken = HardwareService(config(mode="esp32", esp32={"auth_token_env": "LIFEGUARD_HW_H1_MISSING"}))
    assert broken.start() is False and broken.start() is False       # configuration fault: never starts
    assert broken.status().link == LinkState.FAULT and not broken.running
    broken.stop()


# ------------------------------------------------------------------ 7. stop and restart on the same port
def test_stop_and_restart_on_the_same_port() -> None:
    pytest.importorskip("websockets")
    from lifeguard.hardware import fake_device

    port = _free_port()
    cfg = config(mode="esp32", esp32={"port": port}, **FAST)
    service = HardwareService(cfg)
    stop = threading.Event()
    device = threading.Thread(target=fake_device.run, args=(cfg, f"ws://127.0.0.1:{port}/lifeguard-hw", stop),
                              kwargs={"reconnect_s": 0.2}, daemon=True)
    device.start()
    try:
        service.start()
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 10)
        _bounded(service.stop, 15, "stop()")
        assert service.status().link == LinkState.DISCONNECTED
        assert _wait_for(lambda: _port_is_free(port), 5)
        service.start()                                              # same object, same port
        assert service.status().link != LinkState.FAULT
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 10)   # the device reconnects
        ping = service.request("ping", source="operator", by="op")
        assert _wait_for(lambda: {r.command.cmd_id: r for r in service.status().commands}[ping.command.cmd_id]
                         .status == CommandStatus.COMPLETED, 5)
        assert service.status().stats.connects >= 2
    finally:
        stop.set()
        _bounded(service.stop, 15, "stop()")
        device.join(timeout=5)


# ------------------------------------------------------------------ 8. concurrency
def test_concurrent_status_polling_and_state_changes() -> None:
    service = HardwareService(config())
    service.start()
    errors: list[BaseException] = []
    stop = threading.Event()

    def guard(fn: Callable[[], None]) -> Callable[[], None]:
        def loop() -> None:
            try:
                while not stop.is_set():
                    fn()
            except BaseException as err:  # noqa: BLE001 - collected and asserted
                errors.append(err)
        return loop

    def poll() -> None:
        status = service.status()
        assert status.link in tuple(LinkState)
        ids = [r.command.cmd_id for r in status.commands]
        assert len(ids) == len(set(ids))

    def operate() -> None:
        service.request("ping", source="operator", by="op")
        service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 0.2}, source="operator", by="op")
        service.safe_stop("op")
        service.emergency_stop("op")
        service.clear_emergency_stop("op")
        time.sleep(0.01)

    def link_flaps() -> None:
        sim = service.simulator
        sim.go_offline()
        time.sleep(0.15)
        sim.come_online()
        time.sleep(0.3)

    threads = [threading.Thread(target=guard(f), daemon=True) for f in (poll, poll, poll, operate, operate, link_flaps)]
    for th in threads:
        th.start()
    time.sleep(2.0)
    stop.set()
    for th in threads:
        th.join(timeout=10)
    try:
        assert not errors, errors
        assert not any(th.is_alive() for th in threads)
        assert not [e for e in service.status().errors if e.startswith("internal error")]
        assert _wait_for(lambda: service.status().link == LinkState.CONNECTED, 5)
        service.emergency_stop("op")
        assert service.status().estop_latched
    finally:
        _bounded(service.stop, 15, "stop()")


def test_command_superseded_before_it_could_be_sent_is_never_sent() -> None:
    """Commands are sent outside the state lock; one superseded in between must not reach the device."""
    transport = ScriptedTransport()
    cfg = config(mode="esp32", allow_physical_actuation=True, **FAST)
    service = HardwareService(cfg, adapter=Esp32Adapter(cfg, transport))
    service.start(background=False)
    transport.device_connects()
    transport.push(msg("hello", 1, components={"motors": ["motor_1"]}))
    transport.push(msg("telemetry", 2))
    service.step()
    transport.push(msg("telemetry", 3))
    service.step()
    assert service.status().link == LinkState.CONNECTED
    with service._io_lock:                                          # the link is busy (e.g. mid-poll)
        motor = service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1}, source="operator", by="op")
        assert motor.status == CommandStatus.PENDING
        service.emergency_stop("op")                                # supersedes the queued motor command
    service.step()
    names = [m.get("name") for m in transport.sent if m["type"] == "command"]
    assert "emergency_stop" in names and "set_motor" not in names
    service.stop()
