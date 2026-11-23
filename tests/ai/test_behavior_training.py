"""Sample building, leave-one-session-out CV, the training CLI and the softmax config path.

Frames come from the REAL engine run on synthetic scenario boxes (test-double detector), so every check
here is about the training *code path*. Nothing here is evidence of real-world behaviour accuracy.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.factory import create_engine
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.sim.scenarios import make_scenario
from lifeguard.ai.temporal.cv import leave_one_session_out
from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES, Sample, build_samples
from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel
from lifeguard.contracts.enums import TrackState
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.contracts.io import write_events_csv, write_jsonl
from lifeguard.contracts.types import FrameResult, GroundTruthEvent, parse_track_uid
from lifeguard.system.geometry.stub import StubPoolGeometry

ROOT = Path(__file__).resolve().parents[2]
FRAME = np.zeros((1080, 1920, 3), np.uint8)


class _ScenarioDetector(Detector):
    """Replays visible synthetic boxes (test double, not a model)."""

    def __init__(self, frames: tuple[FrameResult, ...]) -> None:
        self._frames = frames
        self._i = 0

    @property
    def model_version(self) -> str:
        return "synthetic_boxes"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame = self._frames[self._i]
        self._i += 1
        return [Detection(t.box_xyxy, t.det_conf, t.cls) for t in frame.tracks if t.state == TrackState.TRACKED]


@cache
def _session(kind: str, seed: int) -> tuple[tuple[FrameResult, ...], tuple[GroundTruthEvent, ...]]:
    scenario = make_scenario(kind, seed=seed)
    engine = RealPerceptionEngine(
        _ScenarioDetector(scenario.frames), StubPoolGeometry(), "cam_sim", f"{kind}_s{seed}"
    )
    frames = tuple(engine.process(FRAME, f.frame_idx, f.t_capture, f.t_utc, f.t_media) for f in scenario.frames)
    return frames, scenario.events


def _samples(kind: str, seed: int) -> list[Sample]:
    frames, truths = _session(kind, seed)
    return build_samples(frames, truths)


def test_distress_labels_follow_the_subject_inside_the_event() -> None:
    """Labels stay on the subject, including new track IDs linked to it through merged_from.

    The bobbing subject briefly disappears; the tracker may give it a new ID afterwards, which the
    submersion monitor records in merged_from. Labels must follow that chain and never reach another swimmer.
    """
    frames, _ = _session("distress_like", 0)
    samples = _samples("distress_like", 0)
    distress = [s for s in samples if s.label == "distress"]
    assert distress
    assert all(20.0 <= s.t_media <= 90.0 for s in distress)
    assert {s.label for s in samples} <= {"distress", "normal_swim"}

    chain = {parse_track_uid(distress[0].track_uid)[2]}
    grew = True
    while grew:
        grew = False
        for frame in frames:
            for track in frame.tracks:
                if track.track_id not in chain and chain & set(track.merged_from):
                    chain.add(track.track_id)
                    grew = True
    assert {parse_track_uid(s.track_uid)[2] for s in distress} <= chain


def test_samples_are_strided_per_track() -> None:
    by_uid: dict[str, list[float]] = {}
    for s in _samples("normal_swim", 0):
        by_uid.setdefault(s.track_uid, []).append(s.t_media)
    for times in by_uid.values():
        assert all(b - a >= 0.5 - 1e-9 for a, b in itertools.pairwise(times))


def test_submersion_events_are_not_behaviour_labels() -> None:
    assert {s.label for s in _samples("submersion", 0)} == {"normal_swim"}


def test_unknown_default_label_rejected() -> None:
    frames, truths = _session("idle", 0)
    with pytest.raises(ValueError):
        build_samples(frames, truths, default_label="floating")


def test_leave_one_session_out_covers_every_sample() -> None:
    samples = _samples("distress_like", 0) + _samples("distress_like", 1) + _samples("normal_swim", 0)
    report = leave_one_session_out(samples, lambda x, y: SoftmaxBehaviorModel.fit(x, y, epochs=100))
    assert report.confusion.shape == (len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES))
    assert int(report.confusion.sum()) == len(samples)
    assert set(report.session_accuracy) == {"distress_like_s0", "distress_like_s1", "normal_swim_s0"}
    assert any(line.startswith("class=distress") for line in report.lines())
    with pytest.raises(ValueError):
        leave_one_session_out(_samples("idle", 0))


def test_trained_model_ranks_held_out_distress_above_normal_swimming() -> None:
    """Logic check on synthetic data: a held-out distress session scores higher P(distress)."""
    train = _samples("distress_like", 0) + _samples("normal_swim", 0) + _samples("idle", 0)
    model = SoftmaxBehaviorModel.fit(np.stack([s.x for s in train]), [s.label for s in train])
    distress = [s for s in _samples("distress_like", 1) if s.label == "distress"]
    normal = _samples("normal_swim", 1)
    col = BEHAVIOR_CLASSES.index("distress")
    p_distress = model.predict_proba(np.stack([s.x for s in distress]))[:, col].mean()
    p_normal = model.predict_proba(np.stack([s.x for s in normal]))[:, col].mean()
    assert p_distress > p_normal


def test_train_cli_writes_a_loadable_model(tmp_path: Path) -> None:
    args = [sys.executable, "-m", "lifeguard.ai.temporal.train"]
    for kind, seed in (("distress_like", 0), ("normal_swim", 0)):
        frames, truths = _session(kind, seed)
        write_jsonl(tmp_path / f"{kind}.jsonl", frames)
        write_events_csv(tmp_path / f"{kind}.csv", truths)
        args += ["--frames", str(tmp_path / f"{kind}.jsonl"), "--truth", str(tmp_path / f"{kind}.csv")]
    out = tmp_path / "model.json"
    proc = subprocess.run(
        [*args, "--out", str(out), "--cv", "--epochs", "100"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("samples=")
    assert "heldout=distress_like_s0" in proc.stdout
    assert SoftmaxBehaviorModel.load(out).metadata["sessions"] == ["distress_like_s0", "normal_swim_s0"]


def test_train_cli_rejects_mock_fixtures_without_features(tmp_path: Path) -> None:
    fixtures = ROOT / "tests" / "fixtures" / "synthetic"
    proc = subprocess.run(
        [sys.executable, "-m", "lifeguard.ai.temporal.train",
         "--frames", str(fixtures / "idle_seed0.jsonl"), "--truth", str(fixtures / "idle_seed0_events.csv"),
         "--out", str(tmp_path / "m.json")],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 2


def test_softmax_behavior_config(tmp_path: Path) -> None:
    path = tmp_path / "ai.yaml"
    path.write_text("engine: yolo\nyolo:\n  weights: w.pt\nbehavior: softmax\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_ai_config(path)  # softmax needs behavior_model

    path.write_text(
        "engine: yolo\nyolo:\n  weights: w.pt\nbehavior: softmax\nbehavior_model: m/b.json\n", encoding="utf-8"
    )
    cfg = load_ai_config(path)
    assert cfg.behavior_model == (tmp_path / "m" / "b.json").resolve()
    with pytest.raises(ModelLoadError):  # the model file does not exist
        create_engine(cfg, "cam_a", "s1", geometry=StubPoolGeometry())
