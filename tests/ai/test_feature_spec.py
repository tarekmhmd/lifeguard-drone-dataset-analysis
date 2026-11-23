"""M1-03 feature spec: valid, consistent with the extractor, the config and the model vector; acceptance checks."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest

from lifeguard.ai.features.config import FeatureConfig
from lifeguard.ai.features.extractor import FeatureExtractor
from lifeguard.ai.features.spec import SPEC_PATH, FeatureSpec, load_feature_spec
from lifeguard.ai.features.submersion import PoolTrackView, SubmersionMonitor
from lifeguard.ai.temporal.dataset import FEATURE_NAMES, VECTOR_SIZE
from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel
from lifeguard.contracts.errors import ModelLoadError
from lifeguard.system.geometry.stub import StubPoolGeometry

DT = 0.1
BOX_H = 60.0


def test_spec_is_valid_and_matches_the_implementation() -> None:
    spec = load_feature_spec()
    cfg = FeatureConfig()
    assert spec.window_s == cfg.window_s and spec.min_span_s == cfg.min_span_s
    assert FEATURE_NAMES == spec.names
    assert spec.model_vector.size == VECTOR_SIZE == 2 * len(spec.names)
    assert {p.name for p in spec.planned_not_implemented}.isdisjoint(spec.names)


def test_every_extractor_output_is_specified_and_every_spec_feature_is_produced() -> None:
    fx = FeatureExtractor()
    produced: set[str] = set()
    for i in range(60):
        t = round(i * DT, 3)
        dy = 15.0 * math.sin(2 * math.pi * 0.8 * t)
        feats = fx.observe(1, t, True, (2.0 + 0.3 * t + 0.2 * math.sin(t), 5.0), True,
                           (100.0, 100.0 + dy, 140.0, 100.0 + BOX_H + dy))
        produced |= feats.keys()
    assert produced == set(load_feature_spec().names)


def test_spec_rejects_duplicates(tmp_path: Path) -> None:
    data = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    data["features"].append(data["features"][0])
    with pytest.raises(ValueError):
        FeatureSpec.model_validate(data)


@pytest.mark.parametrize("freq", [0.5, 0.8, 1.2])
def test_bob_frequency_of_a_sinusoid(freq: float) -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(40):  # 4 s window
        t = round(i * DT, 3)
        dy = 15.0 * math.sin(2 * math.pi * freq * t)
        feats = fx.observe(1, t, True, (5.0, 5.0), True, (100.0, 100.0 + dy, 140.0, 100.0 + BOX_H + dy))
    assert feats["bob_freq_hz"] == pytest.approx(freq, abs=0.1)


def test_no_bob_frequency_for_a_still_box() -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(40):
        feats = fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, (100.0, 100.0, 140.0, 160.0))
    assert "bob_freq_hz" not in feats and feats["bob_std_norm"] == 0.0


def test_m1_03_speed_within_five_percent() -> None:
    """M1-03 acceptance: a synthetic 1.0 m/s straight track is measured within +-5 %."""
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(41):
        t = round(i * DT, 3)
        feats = fx.observe(1, t, True, (2.0 + t, 5.0), True, (100.0, 100.0, 140.0, 160.0))
    assert feats["speed_mps"] == pytest.approx(1.0, rel=0.05)


def test_m1_03_feature_cost_with_twenty_tracks() -> None:
    """M1-03 acceptance: feature + submersion computation < 5 ms per frame with 20 active tracks
    (dev machine). CI runners are slower and shared, so they get a looser bound."""
    fx, monitor = FeatureExtractor(), SubmersionMonitor(StubPoolGeometry())
    budget_ms = 25.0 if os.environ.get("CI") else 5.0
    per_frame: list[float] = []
    for i in range(100):
        t = round(i * DT, 3)
        start = time.perf_counter()
        views = []
        for k in range(20):
            xy = (2.0 + 0.5 * t + 0.01 * k, 1.0 + 0.5 * k)
            fx.observe(k + 1, t, True, xy, True, (10.0 * k, 100.0, 10.0 * k + 40, 160.0))
            views.append(PoolTrackView(k + 1, True, xy, True, 0.0))
        monitor.update(t, views)
        per_frame.append((time.perf_counter() - start) * 1000.0)
    steady = sorted(per_frame[40:])  # after the window is full
    assert steady[len(steady) // 2] < budget_ms, f"median {steady[len(steady) // 2]:.2f} ms"


def test_models_record_and_check_the_spec_version(tmp_path: Path) -> None:
    import numpy as np

    model = SoftmaxBehaviorModel.fit(np.ones((4, VECTOR_SIZE)) * np.arange(4)[:, None],
                                     ["normal_swim", "distress", "normal_swim", "distress"], epochs=5)
    path = tmp_path / "m.json"
    model.save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["feature_spec_version"] == load_feature_spec().spec_version
    doc["feature_spec_version"] = "99.0.0"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="feature spec"):
        SoftmaxBehaviorModel.load(path)
