"""Temporal behaviour classifier interface (Blueprint M7, roadmap Phase 5)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from lifeguard.contracts.types import Behavior


class BehaviorClassifier(ABC):
    """Per-track window features -> ``Behavior`` probabilities, or None when there is too little evidence.

    Implementations must never claim a drowning has been *detected*: they output probabilities that the
    risk engine weighs (weight 0 until a model is validated on real sessions, roadmap P5-3).
    """

    @abstractmethod
    def predict(self, track_id: int, features: Mapping[str, float], in_water: bool) -> Behavior | None:
        """Classify the current window of one track."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-track state (new session)."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifier written to ``Behavior.model_version`` and ``FrameResult.model_versions``."""
