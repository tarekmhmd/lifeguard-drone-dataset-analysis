"""Multinomial logistic-regression behaviour model in NumPy (roadmap P5-2, a trainable baseline).

REAL, trainable implementation: standardised feature vectors (``dataset.vectorize``) -> linear layer ->
softmax over the six contract behaviour classes, trained by deterministic full-batch gradient descent on
class-weighted cross-entropy with L2 regularisation (Blueprint M7: weighted cross-entropy).

A model is only as good as its labelled data. Until it has been trained and cross-validated on real pool
sessions (roadmap P5-2, data-blocked) its output must not drive alarms: keep ``risk.weights.behavior`` 0.

Saved models are plain JSON (no pickle), so loading a file never executes code.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lifeguard.ai.temporal.base import BehaviorClassifier
from lifeguard.ai.temporal.dataset import (
    BEHAVIOR_CLASSES,
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    VECTOR_SIZE,
    vectorize,
)
from lifeguard.contracts.errors import ModelLoadError
from lifeguard.contracts.types import Behavior

FORMAT = "lifeguard-softmax-behavior"
FORMAT_VERSION = 1
_MIN_STD = 1e-8
_N_CLASSES = len(BEHAVIOR_CLASSES)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxBehaviorModel(BehaviorClassifier):
    """Linear softmax classifier over ``FEATURE_NAMES`` vectors."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if mean.shape != (VECTOR_SIZE,) or std.shape != (VECTOR_SIZE,):
            raise ValueError("mean/std must have one entry per vector element")
        if weights.shape != (VECTOR_SIZE, _N_CLASSES) or bias.shape != (_N_CLASSES,):
            raise ValueError("weights/bias have the wrong shape")
        arrays = (mean, std, weights, bias)
        if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(std <= 0):
            raise ValueError("model parameters must be finite and std > 0")
        self._mean = mean.astype(float)
        self._std = std.astype(float)
        self._w = weights.astype(float)
        self._b = bias.astype(float)
        self.metadata: dict[str, Any] = dict(metadata or {})
        digest = hashlib.sha256(json.dumps(self._params(), sort_keys=True).encode()).hexdigest()
        self._version = f"softmax_v{FORMAT_VERSION}:{digest[:12]}"

    # -------------------------------------------------------------- training

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        labels: Sequence[str],
        *,
        l2: float = 1e-3,
        learning_rate: float = 0.5,
        epochs: int = 500,
        class_weighted: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> SoftmaxBehaviorModel:
        """Train on ``x`` (n x VECTOR_SIZE) with string labels from ``BEHAVIOR_CLASSES``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != VECTOR_SIZE or x.shape[0] != len(labels) or len(labels) == 0:
            raise ValueError("x must be (n, VECTOR_SIZE) with one label per row and n > 0")
        unknown = sorted(set(labels) - set(BEHAVIOR_CLASSES))
        if unknown:
            raise ValueError(f"unknown labels: {unknown}")
        y_idx = np.array([BEHAVIOR_CLASSES.index(lab) for lab in labels])
        y = np.eye(_N_CLASSES)[y_idx]

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < _MIN_STD, 1.0, std)
        xs = (x - mean) / std

        counts = np.bincount(y_idx, minlength=_N_CLASSES).astype(float)
        if class_weighted:
            present = counts > 0
            class_w = np.zeros(_N_CLASSES)
            class_w[present] = counts.sum() / (present.sum() * counts[present])
            sample_w = class_w[y_idx]
        else:
            sample_w = np.ones(len(y_idx))
        sample_w = sample_w / sample_w.sum()

        w = np.zeros((VECTOR_SIZE, _N_CLASSES))
        b = np.zeros(_N_CLASSES)
        for _ in range(epochs):
            grad = (_softmax(xs @ w + b) - y) * sample_w[:, None]
            w -= learning_rate * (xs.T @ grad + l2 * w)
            b -= learning_rate * grad.sum(axis=0)

        meta = {
            "n_samples": len(y_idx),
            "class_counts": {c: int(n) for c, n in zip(BEHAVIOR_CLASSES, counts)},
            "l2": l2,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "class_weighted": class_weighted,
            **dict(metadata or {}),
        }
        return cls(mean, std, w, b, meta)

    # ------------------------------------------------------------- inference

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities (n x 6) in ``BEHAVIOR_CLASSES`` order."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return _softmax(((x - self._mean) / self._std) @ self._w + self._b)

    @property
    def model_version(self) -> str:
        return self._version

    def reset(self) -> None:
        """Stateless at inference time."""

    def predict(self, track_id: int, features: Mapping[str, float], in_water: bool) -> Behavior | None:
        if not in_water or not features:
            return None
        p = self.predict_proba(vectorize(features))[0]
        return Behavior(**dict(zip(BEHAVIOR_CLASSES, (float(v) for v in p))), model_version=self._version)

    # ------------------------------------------------------------ persistence

    def _params(self) -> dict[str, Any]:
        return {
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "weights": self._w.tolist(),
            "bias": self._b.tolist(),
        }

    def save(self, path: Path | str) -> None:
        """Write the model as UTF-8 JSON."""
        doc = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "classes": list(BEHAVIOR_CLASSES),
            "feature_names": list(FEATURE_NAMES),
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "model_version": self._version,
            "metadata": self.metadata,
            **self._params(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, indent=2), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path | str) -> SoftmaxBehaviorModel:
        """Load a model written by :meth:`save`.

        Raises:
            ModelLoadError: Missing file, invalid JSON, another format/version, a different feature or
                class layout, or invalid parameters.
        """
        source = Path(path)
        try:
            doc = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as err:
            raise ModelLoadError(f"behaviour model not found: {source}") from err
        except (OSError, UnicodeError, json.JSONDecodeError) as err:
            raise ModelLoadError(f"cannot read behaviour model {source}: {err}") from err
        if not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("format_version") != FORMAT_VERSION:
            raise ModelLoadError(f"{source} is not a {FORMAT} v{FORMAT_VERSION} file")
        if doc.get("classes") != list(BEHAVIOR_CLASSES) or doc.get("feature_names") != list(FEATURE_NAMES):
            raise ModelLoadError(f"{source} was trained with a different class or feature layout")
        spec = str(doc.get("feature_spec_version", ""))
        if spec.split(".")[0] != FEATURE_SPEC_VERSION.split(".")[0]:
            raise ModelLoadError(f"{source} was trained with feature spec {spec or 'unknown'}, "
                                 f"incompatible with {FEATURE_SPEC_VERSION}; retrain the model")
        try:
            return cls(
                np.array(doc["mean"], dtype=float),
                np.array(doc["std"], dtype=float),
                np.array(doc["weights"], dtype=float),
                np.array(doc["bias"], dtype=float),
                doc.get("metadata"),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise ModelLoadError(f"invalid parameters in {source}: {err}") from err
