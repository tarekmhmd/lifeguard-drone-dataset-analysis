"""Softmax behaviour model on toy data: training, determinism, persistence, contract output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES, FEATURE_NAMES, VECTOR_SIZE, vectorize
from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel
from lifeguard.contracts.errors import ModelLoadError


def _toy(n: int = 100, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Two well-separated synthetic clusters: fast swimmers vs motionless bobbing (toy data only)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n):
        rows.append(vectorize({"speed_mps": rng.uniform(0.6, 1.2), "bob_std_norm": rng.uniform(0, 0.05),
                               "visibility_ratio": 1.0}))
        labels.append("normal_swim")
        rows.append(vectorize({"speed_mps": rng.uniform(0, 0.08), "bob_std_norm": rng.uniform(0.15, 0.3),
                               "visibility_ratio": rng.uniform(0.6, 0.9)}))
        labels.append("distress")
    return np.stack(rows), labels


def test_vectorize_layout() -> None:
    v = vectorize({"speed_mps": 0.5, "box_aspect": 1.5})
    assert v.shape == (VECTOR_SIZE,)
    assert v[FEATURE_NAMES.index("speed_mps")] == 0.5
    assert v[len(FEATURE_NAMES) + FEATURE_NAMES.index("speed_mps")] == 1.0
    assert v[len(FEATURE_NAMES) + FEATURE_NAMES.index("bob_std_norm")] == 0.0


def test_fit_separates_toy_clusters() -> None:
    x, y = _toy()
    model = SoftmaxBehaviorModel.fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (len(y), len(BEHAVIOR_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0)
    predicted = [BEHAVIOR_CLASSES[i] for i in proba.argmax(axis=1)]
    assert np.mean([p == t for p, t in zip(predicted, y)]) >= 0.95
    assert model.metadata["class_counts"]["distress"] == 100


def test_fit_is_deterministic() -> None:
    x, y = _toy()
    a = SoftmaxBehaviorModel.fit(x, y)
    b = SoftmaxBehaviorModel.fit(x, y)
    assert a.model_version == b.model_version
    assert np.array_equal(a.predict_proba(x), b.predict_proba(x))


def test_save_load_round_trip(tmp_path: Path) -> None:
    x, y = _toy()
    model = SoftmaxBehaviorModel.fit(x, y, metadata={"note": "toy"})
    path = tmp_path / "m" / "behavior.json"
    model.save(path)
    loaded = SoftmaxBehaviorModel.load(path)
    assert loaded.model_version == model.model_version
    assert np.allclose(loaded.predict_proba(x), model.predict_proba(x))
    assert loaded.metadata["note"] == "toy"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d.update(format="other"), id="format"),
        pytest.param(lambda d: d.update(feature_names=["x"]), id="features"),
        pytest.param(lambda d: d.update(weights=[[1.0]]), id="shape"),
        pytest.param(lambda d: d.update(std=[0.0] * VECTOR_SIZE), id="zero-std"),
    ],
)
def test_load_rejects_bad_files(tmp_path: Path, mutate) -> None:
    x, y = _toy(20)
    path = tmp_path / "m.json"
    SoftmaxBehaviorModel.fit(x, y, epochs=5).save(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ModelLoadError):
        SoftmaxBehaviorModel.load(path)


def test_load_missing_or_corrupt(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        SoftmaxBehaviorModel.load(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError):
        SoftmaxBehaviorModel.load(bad)


def test_predict_returns_contract_behavior() -> None:
    x, y = _toy()
    model = SoftmaxBehaviorModel.fit(x, y)
    b = model.predict(1, {"speed_mps": 0.02, "bob_std_norm": 0.25, "visibility_ratio": 0.7}, in_water=True)
    assert b is not None and b.model_version == model.model_version
    assert b.distress > b.normal_swim
    assert model.predict(1, {"speed_mps": 1.0}, in_water=False) is None
    assert model.predict(1, {}, in_water=True) is None


def test_fit_input_validation() -> None:
    x, y = _toy(5)
    with pytest.raises(ValueError):
        SoftmaxBehaviorModel.fit(x, [*y[:-1], "flailing"])
    with pytest.raises(ValueError):
        SoftmaxBehaviorModel.fit(x[:, :3], y)
