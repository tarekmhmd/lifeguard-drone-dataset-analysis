"""Rules-only behaviour baseline: output shape and qualitative ordering (not accuracy)."""
from __future__ import annotations

import numpy as np
import pytest

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.temporal.rules import MODEL_VERSION, RuleBehaviorBaseline
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import Behavior
from lifeguard.system.geometry.stub import StubPoolGeometry

CLASSES = ("normal_swim", "idle", "play", "dive", "distress", "drowning")


def _top(b: Behavior) -> str:
    return max(CLASSES, key=lambda name: getattr(b, name))


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        pytest.param(
            {"speed_mps": 1.0, "path_speed_mps": 1.0, "progress_ratio": 1.0, "visibility_ratio": 1.0},
            "normal_swim",
            id="swimming",
        ),
        pytest.param({"speed_mps": 0.0, "visibility_ratio": 1.0, "bob_std_norm": 0.0}, "idle", id="calm"),
        pytest.param({"speed_mps": 0.0, "visibility_ratio": 1.0, "bob_std_norm": 0.3}, "distress", id="bobbing"),
        pytest.param({"visibility_ratio": 0.0, "time_since_seen_s": 4.0}, "drowning", id="gone"),
        pytest.param(
            {"speed_mps": 0.05, "path_speed_mps": 1.0, "progress_ratio": 0.05, "visibility_ratio": 1.0},
            "play",
            id="splashing-in-place",
        ),
    ],
)
def test_baseline_ordering(features: dict[str, float], expected: str) -> None:
    behavior = RuleBehaviorBaseline().predict(1, features, in_water=True)
    assert behavior is not None
    assert _top(behavior) == expected
    total = sum(getattr(behavior, name) for name in CLASSES)
    assert total == pytest.approx(1.0, abs=1e-9)
    assert behavior.model_version == MODEL_VERSION


def test_out_of_water_gives_no_behavior() -> None:
    assert RuleBehaviorBaseline().predict(1, {"speed_mps": 0.0}, in_water=False) is None


class _OneSwimmer(Detector):
    """Test double: one person swimming along the pool at 1 m/s (not a model)."""

    def __init__(self) -> None:
        self._geo = StubPoolGeometry()
        self._i = 0

    @property
    def model_version(self) -> str:
        return "scripted"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        u, v = self._geo.to_image(5.0 + 0.1 * self._i, 6.0)
        self._i += 1
        return [Detection((u - 20, v - 30, u + 20, v + 30), 0.9)]


def test_real_engine_fills_behavior_when_classifier_given() -> None:
    engine = RealPerceptionEngine(
        _OneSwimmer(), StubPoolGeometry(), "cam_a", "s1", behavior_classifier=RuleBehaviorBaseline()
    )
    frame = np.zeros((1080, 1920, 3), np.uint8)
    result = None
    for i in range(30):
        t = round(i * 0.1, 3)
        result = engine.process(frame, i, 10.0 + t, add_seconds_utc("2026-01-01T00:00:00.000Z", t), t)
    assert result is not None
    (track,) = result.tracks
    assert track.behavior is not None
    assert _top(track.behavior) == "normal_swim"
    assert result.model_versions["behavior"] == MODEL_VERSION
