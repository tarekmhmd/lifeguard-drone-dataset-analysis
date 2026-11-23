"""Rule-based behaviour baseline (roadmap P5-2, first rung of "rules -> trees -> GRU/TCN").

BASELINE / PLACEHOLDER MODEL. It maps the motion features of ``FeatureExtractor`` to behaviour
probabilities with hand-written formulas loosely based on the instinctive drowning response described in
the Blueprint (little forward progress, vertical bobbing, intermittent disappearance). It is NOT trained
and NOT validated: its only purposes are (1) to exercise the behaviour -> risk integration end to end and
(2) to be the rules-only baseline that a trained model must beat on held-out real sessions.

The risk engine's ``weights.behavior`` stays 0 until such an evaluation exists.
"""
from __future__ import annotations

from collections.abc import Mapping

from lifeguard.ai.temporal.base import BehaviorClassifier
from lifeguard.contracts.types import Behavior

MODEL_VERSION = "rules_baseline_v0"

# Reference values (starting points, not fitted).
_SWIM_SPEED_MPS = 0.5      # speed treated as clearly swimming
_STILL_SPEED_MPS = 0.2     # below this a person is making little progress
_BOB_REF = 0.25            # bob_std_norm treated as strong bobbing
_UNSEEN_REF_S = 3.0        # time unseen treated as fully "gone"
_FLOOR = 0.02              # keeps every class possible


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


class RuleBehaviorBaseline(BehaviorClassifier):
    """Stateless feature -> probability rules. Returns None for people out of the water."""

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    def reset(self) -> None:
        """Stateless: nothing to clear."""

    def predict(self, track_id: int, features: Mapping[str, float], in_water: bool) -> Behavior | None:
        if not in_water:
            return None
        speed = features.get("speed_mps", 0.0)
        path_speed = features.get("path_speed_mps", speed)
        progress = features.get("progress_ratio", 0.0)
        visibility = features.get("visibility_ratio", 1.0)
        bob = _clip01(features.get("bob_std_norm", 0.0) / _BOB_REF)
        unseen = _clip01(features.get("time_since_seen_s", 0.0) / _UNSEEN_REF_S)

        moving = _clip01(speed / _SWIM_SPEED_MPS)
        still = 1.0 - _clip01(speed / _STILL_SPEED_MPS)
        raw = {
            "normal_swim": moving * progress * visibility,
            "idle": still * (1.0 - bob) * visibility,
            "play": _clip01(path_speed / _SWIM_SPEED_MPS) * (1.0 - progress) * visibility,
            "dive": (1.0 - visibility) * (1.0 - unseen) * (1.0 - still),
            "distress": still * bob * visibility,
            "drowning": (1.0 - visibility) * unseen,
        }
        scores = {k: v + _FLOOR for k, v in raw.items()}
        total = sum(scores.values())
        return Behavior(**{k: v / total for k, v in scores.items()}, model_version=MODEL_VERSION)
