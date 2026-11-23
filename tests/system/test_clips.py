"""Alarm clip recorder: pre/post window, multiple incidents, disabled and failure paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lifeguard.system.alerts.service import AlertService
from lifeguard.system.recording.clips import ClipRecorder
from lifeguard.system.recording.config import RecordingConfig
from lifeguard.system.risk.engine import RuleBasedRiskEngine

cv2 = pytest.importorskip("cv2")
DT = 0.1


def _frame(i: int, w: int = 320, h: int = 240) -> np.ndarray:
    return np.full((h, w, 3), (i * 5) % 256, np.uint8)


def _frames_in(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_clip_contains_pre_and_post_window(tmp_path: Path) -> None:
    rec = ClipRecorder(RecordingConfig(out_dir=tmp_path, pre_s=2.0, post_s=1.0))
    done: list[Path] = []
    planned = None
    for i in range(60):
        t = round(i * DT, 3)
        done += rec.on_frame(t, _frame(i))
        if i == 40:  # trigger at t = 4.0
            planned = rec.trigger("risk-s1-000007", t)
    assert planned is not None and done == [planned]
    assert planned.is_file()
    assert 29 <= _frames_in(planned) <= 32  # 2 s before + 1 s after at 10 fps (+-1 frame)


def test_large_frames_are_downscaled(tmp_path: Path) -> None:
    rec = ClipRecorder(RecordingConfig(out_dir=tmp_path, pre_s=0.5, post_s=0.5, max_width=160))
    path = None
    for i in range(20):
        rec.on_frame(round(i * DT, 3), _frame(i, 1280, 720))
        if i == 5:
            path = rec.trigger("evt", round(i * DT, 3))
    assert path is not None and path.is_file()
    cap = cv2.VideoCapture(str(path))
    ok, img = cap.read()
    cap.release()
    assert ok and img.shape[1] == 160


def test_close_flushes_unfinished_clips_and_names_are_sanitised(tmp_path: Path) -> None:
    rec = ClipRecorder(RecordingConfig(out_dir=tmp_path, pre_s=1.0, post_s=30.0))
    for i in range(10):
        rec.on_frame(round(i * DT, 3), _frame(i))
    path = rec.trigger("cam:1/../x", 0.9)
    assert path is not None and path.parent == tmp_path and ":" not in path.name and "/" not in path.name
    assert rec.close() == [path] and path.is_file()


def test_disabled_without_out_dir() -> None:
    rec = ClipRecorder(RecordingConfig())
    assert not rec.enabled
    assert rec.on_frame(0.0, _frame(0)) == []
    assert rec.trigger("evt", 0.0) is None


def test_unwritable_location_fails_without_raising(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    rec = ClipRecorder(RecordingConfig(out_dir=blocker / "clips", pre_s=0.2, post_s=0.2))
    for i in range(3):
        rec.on_frame(round(i * DT, 3), _frame(i))
    rec.trigger("evt", 0.2)
    for i in range(3, 8):
        rec.on_frame(round(i * DT, 3), _frame(i))
    assert rec.completed == [] and rec.failed == ["evt"]


def test_clip_is_attached_to_the_incident_log(tmp_path: Path) -> None:
    alerts = AlertService(RuleBasedRiskEngine())
    alerts.attach_clip("risk-s1-000001", str(tmp_path / "a.mp4"), "2026-01-01T00:00:00.000Z")
    (row,) = alerts.store.clips("risk-s1-000001")
    assert row["path"].endswith("a.mp4")
