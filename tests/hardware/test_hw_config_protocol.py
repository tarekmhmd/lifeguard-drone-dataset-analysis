"""Hardware configuration defaults/validation and the ESP32 message contract (parsing, invalid input)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.device_model import DeviceModel
from lifeguard.hardware.protocol import (
    AckMsg,
    Hello,
    ProtocolError,
    Telemetry,
    encode_command,
    parse_device_message,
)
from lifeguard.system.config import load_system_config
from tests.hardware.util import config, msg

ROOT = Path(__file__).resolve().parents[2]


def test_safe_defaults() -> None:
    cfg = HardwareConfig()
    assert cfg.enabled is False and cfg.mode == "simulation" and cfg.allow_physical_actuation is False
    assert cfg.esp32.bind_host == "127.0.0.1" and cfg.esp32.auth_token_env is None
    assert cfg.permissions["response"] == () and "clear_emergency_stop" not in cfg.permissions["system"]
    assert cfg.on_dispatch == ()


def test_existing_and_example_configs() -> None:
    for name in ("system.yaml", "system_demo_drone.yaml"):
        assert load_system_config(ROOT / "configs" / name).hardware.enabled is False   # unchanged behaviour
    sim = load_system_config(ROOT / "configs" / "system_hardware_sim.yaml").hardware
    esp = load_system_config(ROOT / "configs" / "system_hardware_esp32.example.yaml").hardware
    assert sim.enabled and sim.mode == "simulation"
    assert esp.mode == "esp32" and esp.allow_physical_actuation is False


@pytest.mark.parametrize("bad", [
    {"heartbeat_interval_s": 2.0, "link_timeout_s": 2.0},
    {"telemetry_interval_s": 1.0, "stale_telemetry_s": 1.0},
    {"device_id": "bad id!"},
    {"components": {"actuators": {"a": {"states": ["off", "on"], "safe_state": "closed"}}}},
    {"components": {"motors": {"x": {}}, "sensors": {"x": {}}}},
    {"components": {"motors": {"m": {"max_speed": 1.5}}}},
    {"on_dispatch": [{"name": "set_motor", "target": "motor_1"}]},                    # not permitted
    {"permissions": {"response": ["set_actuator"]},
     "on_dispatch": [{"name": "set_actuator", "target": "nope"}]},                    # unknown target
    {"esp32": {"auth_token_env": "lower-case"}},
    {"mode": "serial"},
])
def test_invalid_configs_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        config(**bad)


def test_valid_messages_parse() -> None:
    model = DeviceModel("pool-prototype-01", config().components)
    hello = parse_device_message(json.dumps(model.hello()))
    tel = parse_device_message(json.dumps(model.telemetry(1.0)))
    assert isinstance(hello, Hello) and set(hello.components["motors"]) == {"motor_1"}
    assert isinstance(tel, Telemetry) and tel.sensors["sensor_2"].value is None       # unavailable stays None
    ack = parse_device_message(json.dumps(msg("ack", 5, cmd_id="c-000001", status="completed")))
    assert isinstance(ack, AckMsg) and ack.status == "completed"
    extra = parse_device_message(json.dumps(msg("heartbeat", 6, rssi=-60)))           # extra fields tolerated
    assert extra.seq == 6


@pytest.mark.parametrize("raw,fragment", [
    ("{not json", "invalid JSON"),
    ("[1, 2]", "JSON object"),
    (json.dumps(msg("bogus", 1)), "unknown message type"),
    (json.dumps({**msg("heartbeat", 1), "proto": 2}), "protocol version"),
    (json.dumps(msg("heartbeat", -1)), "invalid heartbeat"),
    (json.dumps(msg("heartbeat", 1, device_id="has space")), "invalid heartbeat"),
    ('{"type":"telemetry","proto":1,"device_id":"d","seq":1,"sensors":{"s":{"value":NaN}}}', "non-finite"),
    ('{"type":"telemetry","proto":1,"device_id":"d","seq":1,"sensors":{"s":{"value":1e400}}}', "invalid telemetry"),
    (json.dumps(msg("telemetry", 1, motors={"m": {"state": "running", "speed": 2.0}})), "invalid telemetry"),
    (json.dumps(msg("ack", 1, cmd_id="c-1", status="maybe")), "invalid ack"),
    (json.dumps(msg("ack", 1, status="completed")), "invalid ack"),
])
def test_invalid_messages_raise_protocol_error(raw: str, fragment: str) -> None:
    with pytest.raises(ProtocolError, match=fragment):
        parse_device_message(raw)


def test_size_and_binary_limits() -> None:
    with pytest.raises(ProtocolError, match="larger than"):
        parse_device_message(json.dumps(msg("heartbeat", 1, pad="x" * 600)), max_bytes=512)
    with pytest.raises(ProtocolError, match="binary"):
        parse_device_message(b"\x00\x01")


def test_encode_command_refuses_non_finite() -> None:
    text = encode_command("d", 1, "c-1", "set_motor", "motor_1", {"speed": 0.2, "duration_ms": 500})
    assert json.loads(text)["params"] == {"speed": 0.2, "duration_ms": 500}
    with pytest.raises(ProtocolError):
        encode_command("d", 1, "c-1", "set_motor", "motor_1", {"speed": float("nan")})
