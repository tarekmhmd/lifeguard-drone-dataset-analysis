"""Detector interface. A detector only localises people; it never decides about drowning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from lifeguard.contracts.enums import DetClass


@dataclass(frozen=True)
class Detection:
    """One detected box in pixel coordinates (x1, y1, x2, y2)."""

    box_xyxy: tuple[float, float, float, float]
    conf: float
    cls: DetClass = DetClass.PERSON


class Detector(ABC):
    """Frame -> detections. Implementations may raise; the perception engine converts errors."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect people in one BGR uint8 frame."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifier of the loaded model (name and weights hash), for traceability."""
