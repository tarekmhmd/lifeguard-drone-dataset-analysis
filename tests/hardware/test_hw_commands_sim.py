"""Command manager safety rules and the simulator, driven through HardwareService with a fake clock."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeguard.hardware.models import CommandStatus, LinkState
from tests.hardware.util import config, run, started


def _last(service):
    return service.status().commands[-1]


def test_connects_and_safe_stops_on_connect() -> None:
    service, _ = started()
    status = service.status()
    assert status.link == LinkState.CONNECTED and status.device is not None
    first = status.commands[0]
    assert (first.command.name, first.command.source, first.status) == ("safe_stop", "system", CommandStatus.COMPLETED)
    assert status.telemetry.sensors["sensor_1"].value == 12.5
    unavailable = status.telemetry.sensors["sensor_2"]
    assert unavailable.value is None and unavailable.status == "unavailable"          # never invented
    assert not any(r.command.name in ("set_motor", "set_actuator") for r in status.commands)  # nothing moves at start


def test_timed_motor_run_is_accepted_then_completes_by_itself() -> None:
    service, clock = started()
    rec = service.request("set_motor", "motor_1", {"speed": 0.3, "duration_s": 0.5}, source="operator", by="op")
    assert rec.status == CommandStatus.PENDING and rec.command.params == {"speed": 0.3, "duration_ms": 500}
    run(service, clock, 0.25)
    assert _last(service).status == CommandStatus.ACCEPTED
    assert service.status().telemetry.motors["motor_1"].state == "running"
    run(service, clock, 0.6)
    assert _last(service).status == CommandStatus.COMPLETED
    assert service.status().telemetry.motors["motor_1"].speed == 0.0


@pytest.mark.parametrize("target,params,reason", [
    ("motor_9", {"speed": 0.1, "duration_s": 1}, "unknown motor"),
    ("motor_1", {"speed": 0.9, "duration_s": 1}, "speed must be"),
    ("motor_1", {"speed": True, "duration_s": 1}, "speed must be"),
    ("motor_1", {"speed": float("nan"), "duration_s": 1}, "speed must be"),
    ("motor_1", {"speed": 0.1}, "duration_s must be"),                 # no continuous runs
    ("motor_1", {"speed": 0.1, "duration_s": 10}, "duration_s must be"),
    ("motor_1", {"speed": 0.1, "duration_s": 1, "pin": 4}, "unexpected parameters"),
])
def test_motor_validation(target, params, reason) -> None:
    service, _ = started()
    rec = service.request("set_motor", target, params, source="operator", by="op")
    assert rec.status == CommandStatus.REJECTED and reason in rec.detail


def test_actuator_rules() -> None:
    service, clock = started()
    assert "duration_s must be" in service.request(
        "set_actuator", "indicator_1", {"state": "on"}, source="operator", by="op").detail
    assert "state must be" in service.request(
        "set_actuator", "indicator_1", {"state": "blink", "duration_s": 1}, source="operator", by="op").detail
    rec = service.request("set_actuator", "indicator_1", {"state": "on", "duration_s": 0.5}, source="operator", by="op")
    run(service, clock, 0.2)
    assert service.status().telemetry.actuators["indicator_1"].state == "on"
    run(service, clock, 0.6)                                            # deadman: back to the safe state
    assert service.status().telemetry.actuators["indicator_1"].state == "off"
    assert rec.command.params == {"state": "on", "duration_ms": 500}


def test_permissions_and_identity() -> None:
    service, _ = started()
    assert "not permitted" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                              source="response", by="lifeguard").detail
    assert "not permitted" in service.request("clear_emergency_stop", source="system", by="x").detail
    assert "named operator" in service.request("ping", source="operator", by="  ").detail
    assert "unknown command" in service.request("write_gpio", source="operator", by="op").detail
    assert "takes no target" in service.request("ping", "motor_1", source="operator", by="op").detail


def test_emergency_stop_supersedes_latches_and_needs_an_operator_to_clear() -> None:
    service, clock = started()
    motor = service.request("set_motor", "motor_1", {"speed": 0.3, "duration_s": 2}, source="operator", by="op")
    run(service, clock, 0.2)
    estop = service.emergency_stop("op")
    history = {r.command.cmd_id: r for r in service.status().commands}
    assert history[motor.command.cmd_id].status == CommandStatus.SUPERSEDED
    run(service, clock, 0.2)
    status = service.status()
    assert {r.command.cmd_id: r for r in status.commands}[estop.command.cmd_id].status == CommandStatus.COMPLETED
    assert status.estop_latched and status.telemetry.estop_engaged is True
    assert status.telemetry.motors["motor_1"].speed == 0.0
    assert "latched" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                        source="operator", by="op").detail
    assert service.clear_emergency_stop(" ").status == CommandStatus.REJECTED       # name required
    service.clear_emergency_stop("op")
    run(service, clock, 0.2)
    assert service.status().estop_latched is False
    assert service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                           source="operator", by="op").status == CommandStatus.PENDING


def test_conflicts_and_duplicates() -> None:
    service, clock = started()
    service.request("set_motor", "motor_1", {"speed": 0.2, "duration_s": 2}, source="operator", by="op")
    run(service, clock, 0.1)
    assert "conflict" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                         source="operator", by="op").detail
    service.request("ping", source="operator", by="op")
    assert "duplicate" in service.request("ping", source="operator", by="op").detail


def test_stale_telemetry_blocks_movement_but_not_stops() -> None:
    service, clock = started()
    service.simulator.telemetry_frozen = True
    run(service, clock, 1.5)                                            # heartbeats continue, telemetry stops
    status = service.status()
    assert status.link == LinkState.CONNECTED and status.telemetry_stale
    assert "stale" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                      source="operator", by="op").detail
    assert service.safe_stop("op").status == CommandStatus.PENDING


def test_device_faults_block_movement_and_are_reported() -> None:
    service, clock = started()
    service.simulator.inject_fault("motor_driver_fault")
    run(service, clock, 0.3)
    assert "motor_driver_fault" in service.status().faults
    assert "faults" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                       source="operator", by="op").detail
    service.simulator.clear_faults()
    run(service, clock, 0.3)
    assert service.status().faults == ()


def test_disconnect_reconnect_and_failsafe() -> None:
    service, clock = started()
    motor = service.request("set_motor", "motor_1", {"speed": 0.3, "duration_s": 3}, source="operator", by="op")
    run(service, clock, 0.2)
    service.simulator.go_offline()
    run(service, clock, 0.2)
    status = service.status()
    assert status.link == LinkState.DISCONNECTED and status.telemetry is None       # old state not shown as current
    assert {r.command.cmd_id: r for r in status.commands}[motor.command.cmd_id].status == CommandStatus.FAILED
    assert "not connected" in service.request("set_motor", "motor_1", {"speed": 0.1, "duration_s": 1},
                                              source="operator", by="op").detail
    estop = service.emergency_stop("op")
    assert estop.status == CommandStatus.FAILED and "PHYSICAL e-stop" in estop.detail
    assert service.status().estop_latched                              # host latch set even though unsent
    service.simulator.come_online()
    run(service, clock, 1.5)
    status = service.status()
    assert status.link == LinkState.CONNECTED and status.stats.connects == 2
    # Host latch set while offline -> on reconnect the e-stop is re-asserted (not just a safe stop), so a device
    # that rebooted meanwhile cannot come back unlatched.
    last = status.commands[-1]
    assert last.command.name == "emergency_stop" and last.command.requested_by.startswith("on-connect")
    assert last.status == CommandStatus.COMPLETED and status.telemetry.estop_engaged is True


def test_device_side_deadman_on_host_silence() -> None:
    from lifeguard.hardware.device_model import DeviceModel

    model = DeviceModel("pool-prototype-01", config().components, link_timeout_s=2.0)
    model.host_seen(0.0)
    too_long = model.handle({"type": "command", "cmd_id": "c-0", "name": "set_motor", "target": "motor_1",
                             "params": {"speed": 0.3, "duration_ms": 30000}}, 0.0)
    assert too_long[0]["status"] == "rejected"                          # the device checks limits itself
    model.handle({"type": "command", "cmd_id": "c-1", "name": "set_motor", "target": "motor_1",
                  "params": {"speed": 0.3, "duration_ms": 2900}}, 0.0)
    assert model.motors["motor_1"].value == 0.3
    model.tick(2.5)                          # run not over yet, but the host has been silent > link timeout
    assert model.motors["motor_1"].value == 0.0


def test_link_timeout_drops_a_silent_device() -> None:
    service, clock = started()
    sim = service.simulator
    sim.telemetry_frozen = True
    original = sim.model.heartbeat
    sim.model.heartbeat = lambda now: original(now) | {"seq": 0}       # replayed old seq -> ignored as duplicate
    run(service, clock, 2.6)
    status = service.status()
    assert status.stats.duplicate_or_old_seq > 0 and status.stats.disconnects >= 1


def test_corrupt_message_is_counted_not_fatal() -> None:
    service, clock = started()
    service.simulator.send_corrupt_message()
    run(service, clock, 0.3)
    status = service.status()
    assert status.stats.invalid_messages == 1 and status.link == LinkState.CONNECTED
    assert any("invalid message" in e for e in status.errors)


def test_device_estop_button_latches_host(monkeypatch) -> None:
    service, clock = started()
    service.simulator.model.estop = True                               # physical e-stop pressed on the device
    run(service, clock, 0.3)
    assert service.status().estop_latched


def test_command_timeout_and_estop_resend() -> None:
    service, clock = started(config(emergency_stop_retries=2))
    service.simulator.model.handle = lambda msg, now: []               # device stops answering commands
    ping = service.request("ping", source="operator", by="op")
    run(service, clock, 1.2)
    assert {r.command.cmd_id: r for r in service.status().commands}[ping.command.cmd_id].status == CommandStatus.TIMEOUT
    service.emergency_stop("op")
    run(service, clock, 4.0)
    estop = service.status().commands[-1]
    assert estop.command.name == "emergency_stop" and estop.attempts == 3 and estop.status == CommandStatus.TIMEOUT


def test_command_log(tmp_path: Path) -> None:
    log = tmp_path / "hw" / "commands.jsonl"
    service, clock = started(config(command_log=log))
    service.request("set_motor", "motor_1", {"speed": 0.2, "duration_s": 0.3}, source="operator", by="op")
    run(service, clock, 0.6)
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    motor = [x for x in lines if x["name"] == "set_motor"]
    assert [x["status"] for x in motor] == ["pending", "accepted", "completed"] and motor[0]["by"] == "op"


def test_stop_sends_safe_stop_and_disconnects() -> None:
    service, _ = started()
    service.stop()
    status = service.status()
    assert status.link == LinkState.DISCONNECTED
    assert any(r.command.name == "safe_stop" and r.command.requested_by == "shutdown" for r in status.commands)
