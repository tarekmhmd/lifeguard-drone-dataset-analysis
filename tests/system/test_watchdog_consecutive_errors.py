"""Integration checklist: SYSTEM_FAULT after N consecutive engine errors (N = max_consecutive_engine_errors)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.enums import HealthComponent, Status
from lifeguard.contracts.types import FrameResult, SystemHealth
from lifeguard.system.alerts.service import AlertService, Priority
from lifeguard.system.config import load_system_config
from lifeguard.system.health.watchdog import Watchdog
from lifeguard.system.ingest.video import VideoFrame
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
T0 = "2026-01-01T00:00:00.000Z"


def _result(i: int, error: str | None) -> FrameResult:
    return FrameResult(cam_id="c", session_id="s", frame_idx=i, t_capture=float(i) * 0.1, t_utc=T0,
                       frame_width=64, frame_height=48, latency_ms=0.0,
                       status=Status.ERROR if error else Status.OK, error=error)


def _perception(states: list[SystemHealth]) -> SystemHealth:
    return next(s for s in states if s.component == HealthComponent.PERCEPTION)


def _feed(dog: Watchdog, i: int, error: str | None) -> SystemHealth:
    t = i * 0.1
    dog.on_frame(t, np.full((48, 64, 3), i % 256, np.uint8))
    dog.on_result(_result(i, error))
    return _perception(dog.check(t, T0))


def test_fault_only_after_n_consecutive_errors_then_recovery() -> None:
    dog = Watchdog(max_consecutive_errors=3)
    assert _feed(dog, 0, None).status == Status.OK
    first = _feed(dog, 1, "detector crashed")
    assert first.status == Status.DEGRADED and "1 consecutive" in first.detail
    assert _feed(dog, 2, "detector crashed").status == Status.DEGRADED
    fault = _feed(dog, 3, "detector crashed")
    assert fault.status == Status.ERROR and "3 consecutive" in fault.detail and "detector crashed" in fault.detail
    assert _feed(dog, 4, None).status == Status.OK  # one good frame recovers


def test_interrupted_error_run_resets_the_count() -> None:
    dog = Watchdog(max_consecutive_errors=3)
    for i, err in enumerate(["e", "e", None, "e", "e"]):
        state = _feed(dog, i, err)
    assert state.status == Status.DEGRADED


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Watchdog(max_consecutive_errors=0)


def test_fault_and_recovery_notifications() -> None:
    dog = Watchdog(max_consecutive_errors=2)
    alerts = AlertService(RuleBasedRiskEngine())
    notes = []
    for i, err in enumerate([None, "x", "x", "x", None]):
        t = i * 0.1
        dog.on_frame(t, np.full((48, 64, 3), i, np.uint8))
        dog.on_result(_result(i, err))
        notes += alerts.on_health(dog.check(t, T0))
    kinds = [(n.priority, n.incident_id) for n in notes if n.incident_id == "health-perception"]
    assert kinds == [(Priority.INFO, "health-perception"), (Priority.ALARM, "health-perception"),
                     (Priority.CLEAR, "health-perception")]


class _Failing:
    """Engine stand-in whose process() always returns an error frame (never raises, per the contract)."""

    cam_id, session_id = "cam_sim", "s"

    def process(self, frame: np.ndarray, frame_idx: int, t: float, t_utc: str, t_media: float | None) -> FrameResult:
        return FrameResult(cam_id="cam_sim", session_id="s", frame_idx=frame_idx, t_capture=t, t_utc=t_utc,
                           frame_width=64, frame_height=48, status=Status.ERROR, error="model crashed",
                           latency_ms=0.0)


def test_runtime_uses_max_consecutive_engine_errors_from_system_yaml() -> None:
    cfg = load_system_config(ROOT / "configs" / "system.yaml")
    n = cfg.max_consecutive_engine_errors
    rt = LifeguardRuntime(load_ai_config(ROOT / "configs" / "ai.yaml"), cfg, (64, 48), session_id="s",
                          engine=_Failing())  # type: ignore[arg-type]
    image = np.zeros((48, 64, 3), np.uint8)
    statuses = []
    for i in range(n + 2):
        t = i * 1.1  # > the 1 s health period, so every frame is checked
        rt.process(VideoFrame(image, i, t, T0, t))
        statuses.append(_perception(rt.snapshot().health).status)
    assert statuses[n - 2] == Status.DEGRADED and statuses[n - 1] == Status.ERROR
    assert any("SYSTEM FAULT (perception)" in note.message for note in rt.snapshot().notifications)
    rt.close()
