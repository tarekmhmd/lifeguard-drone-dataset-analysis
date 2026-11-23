"""Health watchdog: camera loss, frozen video, low FPS, perception errors, storage; fault notifications."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lifeguard.contracts.enums import HealthComponent, Status
from lifeguard.contracts.errors import ConfigError
from lifeguard.contracts.types import FrameResult, SystemHealth
from lifeguard.system.alerts.service import AlertService, Priority
from lifeguard.system.config import load_system_config
from lifeguard.system.health.config import HealthConfig
from lifeguard.system.health.watchdog import Watchdog, frame_digest
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
T0 = "2026-01-01T00:00:00.000Z"
DT = 0.1


def _image(i: int) -> np.ndarray:
    img = np.zeros((48, 64, 3), np.uint8)
    img[(i * 8) % 48, :, :] = 255  # a moving line: every frame differs
    return img


def _status(states: list[SystemHealth]) -> dict[HealthComponent, Status]:
    return {s.component: s.status for s in states}


def _feed(dog: Watchdog, n: int, start: float = 0.0, frozen: bool = False) -> float:
    t = start
    for i in range(n):
        t = round(start + i * DT, 3)
        dog.on_frame(t, _image(0 if frozen else i))
    return t


def test_healthy_stream(tmp_path: Path) -> None:
    dog = Watchdog(HealthConfig(min_free_disk_mb=0.0), storage_path=tmp_path)
    t = _feed(dog, 60)
    states = _status(dog.check(t, T0))
    assert states == {
        HealthComponent.INGEST: Status.OK,
        HealthComponent.PERCEPTION: Status.OK,
        HealthComponent.STORAGE: Status.OK,
    }


def test_camera_loss_is_an_ingest_error() -> None:
    dog = Watchdog(HealthConfig(camera_timeout_s=3.0))
    t = _feed(dog, 20)
    assert _status(dog.check(t + 2.9, T0))[HealthComponent.INGEST] == Status.OK
    state = next(s for s in dog.check(t + 3.1, T0) if s.component == HealthComponent.INGEST)
    assert state.status == Status.ERROR and "no frame" in state.detail


def test_no_source_at_all_is_an_error_after_timeout() -> None:
    dog = Watchdog(HealthConfig(camera_timeout_s=3.0))
    dog.start(0.0)
    assert _status(dog.check(1.0, T0))[HealthComponent.INGEST] == Status.OK
    assert _status(dog.check(3.5, T0))[HealthComponent.INGEST] == Status.ERROR


def test_frozen_video_is_an_ingest_error() -> None:
    dog = Watchdog(HealthConfig(frozen_after_s=3.0))
    t = _feed(dog, 25, frozen=True)  # 2.4 s of identical frames
    assert _status(dog.check(t, T0))[HealthComponent.INGEST] == Status.OK
    t = _feed(dog, 10, start=t + DT, frozen=True)
    state = next(s for s in dog.check(t, T0) if s.component == HealthComponent.INGEST)
    assert state.status == Status.ERROR and "frozen" in state.detail
    dog.on_frame(t + DT, _image(3))  # picture changes again
    assert _status(dog.check(t + DT, T0))[HealthComponent.INGEST] == Status.OK


def test_low_fps_degrades_perception() -> None:
    dog = Watchdog(HealthConfig(fps_floor=5.0, fps_window_s=5.0))
    dog.start(0.0)
    for i in range(11):  # 2 fps for 5 s
        dog.on_frame(i * 0.5, _image(i))
    assert _status(dog.check(5.0, T0))[HealthComponent.PERCEPTION] == Status.DEGRADED


def test_perception_error_result() -> None:
    dog = Watchdog()
    t = _feed(dog, 5)
    dog.on_result(FrameResult(cam_id="c", session_id="s", frame_idx=0, t_capture=t, t_utc=T0,
                              frame_width=64, frame_height=48, status=Status.ERROR, error="model crashed",
                              latency_ms=0.0))
    state = next(s for s in dog.check(t, T0) if s.component == HealthComponent.PERCEPTION)
    assert state.status == Status.ERROR and "model crashed" in state.detail


def test_storage_checks(tmp_path: Path) -> None:
    assert _status(Watchdog().check(0.0, T0))[HealthComponent.STORAGE] == Status.DEGRADED
    full = Watchdog(HealthConfig(min_free_disk_mb=1e15), storage_path=tmp_path)
    assert _status(full.check(0.0, T0))[HealthComponent.STORAGE] == Status.ERROR
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    unusable = Watchdog(storage_path=blocker / "sub")
    assert _status(unusable.check(0.0, T0))[HealthComponent.STORAGE] == Status.ERROR


def test_frame_digest_detects_change() -> None:
    assert frame_digest(_image(1)) == frame_digest(_image(1))
    assert frame_digest(_image(1)) != frame_digest(_image(2))


def test_alert_service_raises_and_clears_system_faults() -> None:
    alerts = AlertService(RuleBasedRiskEngine())
    dog = Watchdog(HealthConfig(camera_timeout_s=3.0))
    t = _feed(dog, 10)
    assert [n for n in alerts.on_health(dog.check(t, T0)) if n.priority == Priority.ALARM] == []
    faults = alerts.on_health(dog.check(t + 5.0, T0))
    (fault,) = [n for n in faults if n.priority == Priority.ALARM]
    assert "SYSTEM FAULT" in fault.message and "watch the pool directly" in fault.message
    assert alerts.on_health(dog.check(t + 6.0, T0)) == []  # no repeat while unchanged
    dog.on_frame(t + 6.1, _image(7))
    recovered = alerts.on_health(dog.check(t + 6.1, T0))
    assert any(n.priority == Priority.CLEAR for n in recovered)
    assert alerts.health()[HealthComponent.INGEST] == Status.OK


def test_health_and_recording_config_in_system_yaml(tmp_path: Path) -> None:
    cfg = load_system_config(ROOT / "configs" / "system.yaml")
    assert cfg.health == HealthConfig()
    assert cfg.recording.out_dir is None
    path = tmp_path / "system.yaml"
    path.write_text("recording:\n  out_dir: clips\n", encoding="utf-8")
    assert load_system_config(path).recording.out_dir == (tmp_path / "clips").resolve()
    with pytest.raises(ConfigError):
        load_system_config(_write(tmp_path, "health:\n  camera_timeout_s: 0\n"))


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    return path
