"""Integrated runtime: MOCK perception on synthetic frames through risk, alerts, response, health, clips.

Perception here is the mock fixture replay; everything downstream is the real implementation.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.contracts.enums import AckDecision, HealthComponent, RiskLevel, Status
from lifeguard.contracts.errors import ConfigError
from lifeguard.contracts.io import read_jsonl
from lifeguard.contracts.types import RiskEvent
from lifeguard.system.alerts.service import Priority
from lifeguard.system.config import SystemConfig, load_system_config
from lifeguard.system.ingest.live import LiveCameraSource, SyntheticFrameSource
from lifeguard.system.response import OutcomeKind

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "configs" / "ai.yaml"
SYS = ROOT / "configs" / "system.yaml"
SIZE = (1920, 1080)
T0 = "2026-01-01T00:00:00.000Z"


def _sys(**changes: object) -> SystemConfig:
    cfg = load_system_config(SYS)
    for dotted, value in changes.items():
        target = cfg
        *path, last = dotted.split("__")
        for part in path:
            target = getattr(target, part)
        setattr(target, last, value)
    return cfg


def _runtime(cfg: SystemConfig) -> LifeguardRuntime:
    return LifeguardRuntime(load_ai_config(AI), cfg, SIZE, session_id="rt_test")


def _until_alarm(rt: LifeguardRuntime, source: SyntheticFrameSource) -> tuple[RiskEvent, object]:
    frames = iter(source)
    for frame in frames:
        for event in rt.process(frame):
            if event.to_level == RiskLevel.ALARM:
                return event, frames
    raise AssertionError("no ALARM in the submersion fixture")


def _synthetic(n: int = 1200) -> SyntheticFrameSource:
    return SyntheticFrameSource(*SIZE, n_frames=n, start_t_capture=0.0, start_utc=T0)


def test_confirm_dispatches_simulated_drone_and_release_needs_a_human() -> None:
    rt = _runtime(_sys(response__platform="drone"))
    assert "SIMULATION ONLY" in rt.components["response"]
    assert "PLACEHOLDER" in rt.components["geometry"] and "MOCK" in rt.components["perception"]
    source = _synthetic()
    rt.attach_source(source)
    alarm, frames = _until_alarm(rt, source)
    result, outcome = rt.acknowledge(alarm.event_id, AckDecision.CONFIRM, "lifeguard_1")
    assert result.dispatch is not None and result.dispatch.confirmed_by == "lifeguard_1"
    assert outcome is not None and outcome.kind == OutcomeKind.ACCEPTED

    awaiting = False
    for frame in frames:
        rt.process(frame)
        if any(o.kind == OutcomeKind.AWAITING_CONFIRMATION for o in rt.snapshot().outcomes):
            awaiting = True
            break
    assert awaiting
    assert rt.confirm_release("lifeguard_1") is not None
    for frame in frames:
        rt.process(frame)
        if rt.snapshot().outcomes[-1].kind == OutcomeKind.COMPLETED:
            break
    kinds = [o.kind for o in rt.snapshot().outcomes]
    assert kinds[-1] == OutcomeKind.COMPLETED
    assert any("released" in o.detail for o in rt.snapshot().outcomes)
    rt.close()


def test_manual_platform_and_dismissal() -> None:
    rt = _runtime(_sys())
    source = _synthetic()
    rt.attach_source(source)
    alarm, _ = _until_alarm(rt, source)
    dismissed, outcome = rt.acknowledge(alarm.event_id, AckDecision.DISMISS, "lifeguard_2")
    assert outcome is None and dismissed.dispatch is None
    assert [e.to_level for e in dismissed.events] == [RiskLevel.RESOLVED]
    assert rt.store.hard_negatives()[0]["ack_by"] == "lifeguard_2"
    rt.close()


def test_manual_platform_confirm_means_humans_respond() -> None:
    rt = _runtime(_sys())
    source = _synthetic()
    rt.attach_source(source)
    alarm, _ = _until_alarm(rt, source)
    _, outcome = rt.acknowledge(alarm.event_id, AckDecision.CONFIRM, "lifeguard_1")
    assert outcome is not None and outcome.kind == OutcomeKind.MANUAL
    rt.close()


def test_alarm_clip_is_recorded_and_linked(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    rt = _runtime(_sys(recording__out_dir=tmp_path / "clips", recording__pre_s=1.0, recording__post_s=1.0,
                       alerts__incident_db=tmp_path / "incidents.sqlite3"))
    source = _synthetic(400)
    rt.attach_source(source)
    _, frames = _until_alarm(rt, source)
    for _, frame in zip(range(20), frames):
        rt.process(frame)
    (clip,) = rt.store.clips()
    assert Path(clip["path"]).is_file()
    rt.close()
    assert (tmp_path / "incidents.sqlite3").is_file()


def test_camera_loss_raises_a_system_fault() -> None:
    rt = _runtime(_sys())
    source = _synthetic(20)
    rt.run(source)
    rt.idle(t=1_000.0, t_utc=T0)  # long silence after the last frame
    faults = [n for n in rt.snapshot().notifications if n.priority == Priority.ALARM and "SYSTEM FAULT" in n.message]
    assert faults and "ingest" in faults[0].message
    ingest = next(h for h in rt.snapshot().health if h.component == HealthComponent.INGEST)
    assert ingest.status == Status.ERROR
    rt.close()


def test_unreachable_sitl_falls_back_to_manual_and_says_so() -> None:
    cfg = _sys(response__platform="drone")
    cfg.response.drone.executor = "mavlink_sitl"
    cfg.response.drone.heartbeat_timeout_s = 0.2
    rt = _runtime(cfg)
    assert rt.components["response"].startswith("UNAVAILABLE drone")
    assert rt.response.name == "manual"
    rt.close()


def test_missing_calibration_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _runtime(_sys(geometry__calibration=tmp_path / "absent.json"))


def test_dashboard_style_concurrent_acknowledgement() -> None:
    """Frames are processed on a worker thread while the 'operator' acknowledges from another."""
    rt = _runtime(_sys())
    stop = threading.Event()
    worker = threading.Thread(target=rt.run, args=(_synthetic(),), kwargs={"stop": stop}, daemon=True)
    worker.start()
    deadline = time.monotonic() + 60
    event_id = None
    while time.monotonic() < deadline and event_id is None:
        for incident_id, level in rt.snapshot().open_incidents.values():
            if level == RiskLevel.ALARM:
                event_id = rt.alerts.latest_event_id(incident_id)
        time.sleep(0.01)
    assert event_id is not None
    result, _ = rt.acknowledge(event_id, AckDecision.CONFIRM, "lifeguard_1")
    stop.set()
    worker.join(timeout=30)
    assert result.dispatch is not None
    rt.close()


# --------------------------------------------------------------------------- CLI


def _main(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "lifeguard.app.main", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=180, check=False)


def test_cli_synthetic_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    proc = _main("--ai-config", str(AI), "--system-config", str(SYS), "--synthetic", "--out-events", str(out))
    assert proc.returncode == 0, proc.stderr
    last = proc.stdout.strip().splitlines()[-1]
    assert last.startswith("frames=600 ok=600 degraded=0 error=0 unique_tracks=3 max_submersion_s=40.000")
    assert "alarms=1" in last and "incidents=1" in last
    assert "[ALARM] ALARM:" in proc.stdout and "SYSTEM FAULT" not in proc.stdout
    assert any(e.to_level == RiskLevel.ALARM for e in read_jsonl(out, RiskEvent))


def test_cli_rejects_bad_inputs(tmp_path: Path) -> None:
    assert _main("--ai-config", str(ROOT / "configs" / "ai_yolo.yaml"), "--system-config", str(SYS),
                 "--synthetic").returncode == 2
    assert _main("--ai-config", str(AI), "--system-config", str(SYS),
                 "--video", str(tmp_path / "missing.mp4")).returncode == 2
    bad = tmp_path / "ai.yaml"
    bad.write_text("engine: yolo\nyolo:\n  weights: absent.pt\n", encoding="utf-8")
    pytest.importorskip("cv2")
    import cv2

    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(5):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()
    assert _main("--ai-config", str(bad), "--system-config", str(SYS), "--video", str(video)).returncode == 3


# ------------------------------------------------------------------ live source


class _FakeCapture:
    """Fake camera: a few frames, then a failure (forcing a reconnect)."""

    opened = 0

    def __init__(self, source: object) -> None:
        _FakeCapture.opened += 1
        self._left = 5

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.01)
        if self._left == 0:
            return False, None
        self._left -= 1
        return True, np.full((48, 64, 3), self._left, np.uint8)

    def release(self) -> None:
        pass


def test_live_source_latest_frame_and_reconnect() -> None:
    _FakeCapture.opened = 0
    src = LiveCameraSource(0, reconnect_s=0.05, poll_s=0.05, capture_factory=_FakeCapture)
    frames, nones = [], 0
    for item in src:
        if item is None:
            nones += 1
        else:
            frames.append(item)
        if len(frames) >= 6 or nones > 200:
            break
    src.stop()
    assert len(frames) >= 6
    assert [f.frame_idx for f in frames] == list(range(len(frames)))
    assert all(f.t_media is None for f in frames)
    assert _FakeCapture.opened >= 2 and src.reconnects >= 1
