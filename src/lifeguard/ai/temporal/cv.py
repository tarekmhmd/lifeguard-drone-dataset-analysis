"""Leave-one-session-out cross-validation for behaviour models (Blueprint M7 test protocol).

Window-level metrics only (confusion matrix, per-class recall). Event-level performance of the whole
system is measured with ``lifeguard.app.evaluate`` (missed events, FA/h, latency).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES, Sample
from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel

FitFn = Callable[[np.ndarray, list[str]], SoftmaxBehaviorModel]


@dataclass(frozen=True)
class CVReport:
    """Pooled confusion matrix (rows = true class, columns = predicted) over all held-out sessions."""

    confusion: np.ndarray
    session_accuracy: dict[str, float]

    def recall(self, label: str) -> float | None:
        """Recall of one class, or None if it never occurs in the data."""
        i = BEHAVIOR_CLASSES.index(label)
        total = self.confusion[i].sum()
        return None if total == 0 else float(self.confusion[i, i] / total)

    def lines(self) -> list[str]:
        """Human-readable summary: one line per held-out session, then per-class recall."""
        out = [f"heldout={s} accuracy={a:.3f}" for s, a in sorted(self.session_accuracy.items())]
        for label in BEHAVIOR_CLASSES:
            r = self.recall(label)
            if r is not None:
                out.append(f"class={label} n={int(self.confusion[BEHAVIOR_CLASSES.index(label)].sum())} "
                           f"recall={r:.3f}")
        return out


def leave_one_session_out(samples: Sequence[Sample], fit: FitFn | None = None) -> CVReport:
    """Train on all sessions but one, test on the held-out one, for every session.

    Raises:
        ValueError: With fewer than two sessions.
    """
    fit_fn: FitFn = fit or (lambda x, y: SoftmaxBehaviorModel.fit(x, y))
    sessions = sorted({s.session_id for s in samples})
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out needs at least two sessions")
    confusion = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=int)
    accuracy: dict[str, float] = {}
    for held_out in sessions:
        train = [s for s in samples if s.session_id != held_out]
        test = [s for s in samples if s.session_id == held_out]
        model = fit_fn(np.stack([s.x for s in train]), [s.label for s in train])
        predicted = model.predict_proba(np.stack([s.x for s in test])).argmax(axis=1)
        truth = np.array([BEHAVIOR_CLASSES.index(s.label) for s in test])
        np.add.at(confusion, (truth, predicted), 1)
        accuracy[held_out] = float((predicted == truth).mean())
    return CVReport(confusion, accuracy)
