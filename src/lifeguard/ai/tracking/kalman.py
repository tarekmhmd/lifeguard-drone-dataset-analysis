"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space, as used by SORT/ByteTrack."""
from __future__ import annotations

import numpy as np

_STD_POSITION = 1.0 / 20.0
_STD_VELOCITY = 1.0 / 160.0
_NDIM = 4


class KalmanFilterXYAH:
    """8-dimensional state (cx, cy, a, h, vcx, vcy, va, vh); one step per processed frame."""

    def __init__(self) -> None:
        """Build the constant-velocity motion and observation matrices."""
        self._motion = np.eye(2 * _NDIM)
        for i in range(_NDIM):
            self._motion[i, _NDIM + i] = 1.0
        self._update = np.eye(_NDIM, 2 * _NDIM)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated measurement (cx, cy, a, h)."""
        mean = np.r_[measurement, np.zeros(_NDIM)]
        h = measurement[3]
        std = np.array([
            2 * _STD_POSITION * h, 2 * _STD_POSITION * h, 1e-2, 2 * _STD_POSITION * h,
            10 * _STD_VELOCITY * h, 10 * _STD_VELOCITY * h, 1e-5, 10 * _STD_VELOCITY * h,
        ])
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state one frame ahead."""
        h = mean[3]
        std = np.array([
            _STD_POSITION * h, _STD_POSITION * h, 1e-2, _STD_POSITION * h,
            _STD_VELOCITY * h, _STD_VELOCITY * h, 1e-5, _STD_VELOCITY * h,
        ])
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def update(
        self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a measurement (cx, cy, a, h)."""
        h = mean[3]
        std = np.array([_STD_POSITION * h, _STD_POSITION * h, 1e-1, _STD_POSITION * h])
        projected_mean = self._update @ mean
        projected_cov = self._update @ cov @ self._update.T + np.diag(np.square(std))
        gain = np.linalg.solve(projected_cov, (cov @ self._update.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + gain @ innovation
        new_cov = cov - gain @ projected_cov @ gain.T
        return new_mean, new_cov


def xyxy_to_xyah(box: tuple[float, float, float, float]) -> np.ndarray:
    """(x1, y1, x2, y2) -> (cx, cy, w/h, h)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=float)


def xyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, a, h, ...) -> (x1, y1, x2, y2)."""
    cx, cy, a, h = (float(v) for v in state[:4])
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
