"""Video ingest, the shared pipeline loop and the run_video CLI (OpenCV-dependent tests skip without cv2)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.app.pipeline import RunSummary, TooManyEngineErrors, run_pipeline
from lifeguard.contracts.io import read_jsonl
from lifeguard.contracts.types import FrameResult
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.ingest.video import VideoFileSource, VideoSourceError
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS, N = 320, 240, 10.0, 30
T0 = "2026-01-01T00:00:00.000Z"


def _write_video(path: Path, n: int = N) -> Path:
    """Write a small MJPG AVI with frame-dependent brightness; skip if this OpenCV cannot write it."""
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer unavailable")
    for i in range(n):
        writer.write(np.full((H, W, 3), (i * 8) % 256, np.uint8))
    writer.release()
    return path


class _Scripted(Detector):
    """Test double: a person at pool (10, 6) for the first ``visible`` frames, then gone (not a model)."""

    def __init__(self, geo: StubPoolGeometry, visible: int, fail: bool = False) -> None:
        self._geo = geo
        self._visible = visible
        self._fail = fail
        self._i = 0

    @property
    def model_version(self) -> str:
        return "scripted"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._fail:
            raise RuntimeError("detector down")
        i = self._i
        self._i += 1
        if i >= self._visible:
            return []
        u, v = self._geo.to_image(10.0, 6.0)
        return [Detection((u - 10, v - 15, u + 10, v + 15), 0.9)]


# ----------------------------------------------------------------------- ingest


def test_video_source_timing_is_deterministic(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")
    runs = []
    for _ in range(2):
        with VideoFileSource(video, start_t_capture=100.0, start_utc=T0) as source:
            assert (source.width, source.height) == (W, H)
            assert source.fps == pytest.approx(FPS)
            runs.append([(f.frame_idx, f.t_media, f.t_capture, f.t_utc) for f in source])
    assert runs[0] == runs[1]
    assert [r[0] for r in runs[0]] == list(range(N))
    assert runs[0][5][1] == pytest.approx(0.5)
    assert runs[0][5][2] == pytest.approx(100.5)
    assert runs[0][10][3] == "2026-01-01T00:00:01.000Z"


def test_video_source_stride_and_max_frames(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")
    with VideoFileSource(video, stride=3, max_frames=4) as source:
        frames = list(source)
    assert [f.frame_idx for f in frames] == [0, 3, 6, 9]
    assert frames[0].image.shape == (H, W, 3) and frames[0].image.dtype == np.uint8


def test_video_source_errors(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError):
        VideoFileSource(tmp_path / "missing.mp4")
    video = _write_video(tmp_path / "clip.avi")
    with pytest.raises(VideoSourceError):
        VideoFileSource(video, stride=0)


# --------------------------------------------------------------------- pipeline


def test_pipeline_on_video_reports_submersion(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")
    geo = StubPoolGeometry(image_width=W, image_height=H)
    engine = RealPerceptionEngine(_Scripted(geo, visible=10), geo, "cam_v", "s1")
    results: list[FrameResult] = []
    with VideoFileSource(video, start_t_capture=0.0, start_utc=T0) as source:
        summary = run_pipeline(source, engine, RuleBasedRiskEngine(), 5, on_result=results.append)
    assert (summary.frames, summary.ok, summary.error) == (N, N, 0)
    assert len(summary.track_uids) == 1
    assert summary.max_submersion_s == pytest.approx(2.0, abs=1e-6)  # last seen 0.9 s, video ends at 2.9 s
    assert len(results) == N


def test_pipeline_stops_on_consecutive_errors(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")
    geo = StubPoolGeometry(image_width=W, image_height=H)
    engine = RealPerceptionEngine(_Scripted(geo, visible=0, fail=True), geo, "cam_v", "s1")
    with VideoFileSource(video) as source, pytest.raises(TooManyEngineErrors):
        run_pipeline(source, engine, RuleBasedRiskEngine(), 5)


def test_summary_line_format() -> None:
    summary = RunSummary(frames=3, ok=2, error=1, max_submersion_s=1.25, risk_events=2, alarms=1)
    summary.track_uids.update({"c:s:1", "c:s:2"})
    assert summary.line() == (
        "frames=3 ok=2 degraded=0 error=1 unique_tracks=2 max_submersion_s=1.250 risk_events=2 alarms=1"
    )


# -------------------------------------------------------------------------- CLI


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lifeguard.app.run_video", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_missing_video_exit_2(tmp_path: Path) -> None:
    proc = _cli(
        "--ai-config", str(ROOT / "configs" / "ai_yolo.yaml"),
        "--system-config", str(ROOT / "configs" / "system.yaml"),
        "--video", str(tmp_path / "missing.mp4"),
    )
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout == ""


def test_cli_missing_weights_exit_3(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")
    ai_cfg = tmp_path / "ai.yaml"
    ai_cfg.write_text("engine: yolo\nyolo:\n  weights: absent.pt\n", encoding="utf-8")
    proc = _cli(
        "--ai-config", str(ai_cfg),
        "--system-config", str(ROOT / "configs" / "system.yaml"),
        "--video", str(video),
    )
    assert proc.returncode == 3, proc.stderr


def test_cli_mock_engine_on_video_hits_error_limit(tmp_path: Path) -> None:
    """The mock engine expects 1920x1080 fixture frames, so a 320x240 video is rejected frame by frame."""
    video = _write_video(tmp_path / "clip.avi")
    out = tmp_path / "frames.jsonl"
    proc = _cli(
        "--ai-config", str(ROOT / "configs" / "ai.yaml"),
        "--system-config", str(ROOT / "configs" / "system.yaml"),
        "--video", str(video),
        "--out-frames", str(out),
    )
    assert proc.returncode == 4, proc.stderr
    assert len(read_jsonl(out, FrameResult)) == 9  # limit 10: the 10th error frame is not written
