"""M1-04 labels: taxonomy, interval CSV schema, annotator agreement, interval-based training samples."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.sim.scenarios import make_scenario
from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES
from lifeguard.ai.temporal.labels import (
    INTERVAL_COLUMNS,
    BehaviorInterval,
    agreement,
    build_interval_samples,
    load_taxonomy,
    read_intervals,
    write_intervals,
)
from lifeguard.contracts.enums import TrackState
from lifeguard.contracts.errors import ContractValidationError
from lifeguard.contracts.types import FrameResult
from lifeguard.system.geometry.stub import StubPoolGeometry

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "ai" / "annotation_guide.md"


def _iv(label: str, t0: float, t1: float, track: int = 1, annotator: str = "a1", session: str = "s1") -> BehaviorInterval:
    return BehaviorInterval(session_id=session, video_file="v.mp4", track_id=track, t_start=t0, t_end=t1,
                            label=label, annotator=annotator)


def test_taxonomy_covers_the_required_labels_with_observable_criteria() -> None:
    tax = load_taxonomy()
    assert {"normal_swim", "idle", "play", "dive", "distress", "uncertain"} == set(tax.names)
    for d in tax.labels:
        assert d.criteria and d.min_duration_s > 0
        assert d.contract_class is None or d.contract_class in BEHAVIOR_CLASSES
    distress = tax.get("distress")
    assert distress.decision_rule and "never enough" in distress.decision_rule
    assert distress.min_duration_s >= 3.0
    text = " ".join(tax.principles).lower()
    assert "raised arms alone" in text and "submersion alone" in text and "not a drowning person" in text
    assert "drowning" not in tax.names  # never annotated in staged sessions


def test_annotation_guide_exists_and_matches_the_taxonomy() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    for label in load_taxonomy().names:
        assert f"`{label}`" in guide
    assert ",".join(INTERVAL_COLUMNS) in guide


@pytest.mark.parametrize(
    ("label", "t0", "t1", "message"),
    [
        ("flailing", 0.0, 5.0, "unknown label"),
        ("distress", 0.0, 2.0, ">= 3.0 s"),
        ("idle", 5.0, 5.0, "greater than"),
    ],
)
def test_invalid_intervals(label: str, t0: float, t1: float, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _iv(label, t0, t1)


def test_csv_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    rows = [_iv("normal_swim", 0.0, 10.0), _iv("distress", 10.0, 20.0), _iv("idle", 0.0, 5.0, track=2)]
    assert write_intervals(path, rows) == 3
    assert read_intervals(path) == rows
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(INTERVAL_COLUMNS)

    overlap = tmp_path / "overlap.csv"
    write_intervals(overlap, [_iv("normal_swim", 0.0, 10.0), _iv("play", 9.0, 12.0)])
    with pytest.raises(ContractValidationError, match="overlapping"):
        read_intervals(overlap)
    bad = tmp_path / "bad.csv"
    bad.write_text("session,track\n", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="header"):
        read_intervals(bad)
    bad.write_text(",".join(INTERVAL_COLUMNS) + "\ns1,v.mp4,1,0,10,swimming,a1,\n", encoding="utf-8")
    with pytest.raises(ContractValidationError) as info:
        read_intervals(bad)
    assert info.value.line == 2


def test_agreement_perfect_partial_and_none() -> None:
    a = [_iv("normal_swim", 0.0, 10.0), _iv("distress", 10.0, 20.0)]
    same = agreement(a, [_iv(i.label, i.t_start, i.t_end, annotator="a2") for i in a])
    assert same.percent == 1.0 and same.kappa == pytest.approx(1.0)
    b = [_iv("normal_swim", 0.0, 15.0, annotator="a2"), _iv("distress", 15.0, 20.0, annotator="a2")]
    partial = agreement(a, b)
    assert 0.5 < partial.percent < 1.0 and 0.0 < partial.kappa < 1.0
    assert partial.confusion[("distress", "normal_swim")] > 0
    assert agreement(a, [_iv("idle", 0.0, 5.0, track=9, annotator="a2")]).n_points == 0


class _Replay(Detector):
    def __init__(self, frames: tuple[FrameResult, ...]) -> None:
        self._frames, self._i = frames, 0

    @property
    def model_version(self) -> str:
        return "synthetic_boxes"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        f = self._frames[self._i]
        self._i += 1
        return [Detection(t.box_xyxy, t.det_conf, t.cls) for t in f.tracks if t.state == TrackState.TRACKED]


def test_interval_samples_use_only_labelled_time() -> None:
    scenario = make_scenario("normal_swim", seed=0)
    engine = RealPerceptionEngine(_Replay(scenario.frames), StubPoolGeometry(), "cam_sim", "s1")
    frames = [engine.process(np.zeros((1080, 1920, 3), np.uint8), f.frame_idx, f.t_capture, f.t_utc, f.t_media)
              for f in scenario.frames]
    track_ids = sorted({t.track_id for f in frames for t in f.tracks})
    intervals = [_iv("normal_swim", 5.0, 25.0, track=track_ids[0]),
                 _iv("uncertain", 25.0, 30.0, track=track_ids[0]),
                 _iv("idle", 10.0, 20.0, track=track_ids[1])]
    samples = build_interval_samples(frames, intervals, stride_s=0.5)
    by_label = {s.label for s in samples}
    assert by_label == {"normal_swim", "idle"}
    for s in samples:
        tid = int(s.track_uid.rsplit(":", 1)[1])
        assert (tid == track_ids[0] and 5.0 <= s.t_media <= 25.0) or (tid == track_ids[1] and 10.0 <= s.t_media <= 20.0)


def test_labels_cli(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_intervals(a, [_iv("normal_swim", 0.0, 10.0)])
    write_intervals(b, [_iv("normal_swim", 0.0, 10.0, annotator="a2")])

    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "lifeguard.ai.temporal.labels", *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=60, check=False)

    assert cli("validate", str(a)).stdout.startswith("valid intervals=1 normal_swim=1")
    assert "kappa=1.000" in cli("agreement", str(a), str(b)).stdout
    (tmp_path / "x.csv").write_text("nope\n", encoding="utf-8")
    assert cli("validate", str(tmp_path / "x.csv")).returncode == 2
