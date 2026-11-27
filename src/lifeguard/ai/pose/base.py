"""Pose estimation interface and the model-independent geometry it relies on.

A ``PoseEstimator`` receives the frame and the boxes of the swimmers that ByteTrack is currently tracking. It
crops each box (plus a margin), estimates COCO-17 keypoints inside the crop, keeps the person in the crop that
overlaps the tracked box best (a crop can contain a neighbouring swimmer) and returns the keypoints in frame
pixels, keyed by track id. It never adds or removes tracks: the person detector and ByteTrack stay the only
source of who is in the pool.

Pose output is evidence for display and logging only. The risk engine does not read ``TrackObs.keypoints``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

COCO_KEYPOINTS = 17
Keypoints = tuple[tuple[float, float, float], ...]
Box = tuple[float, float, float, float]
_MIN_CROP_PX = 8


class PoseEstimator(ABC):
    """Keypoints for tracked people. Implementations must be safe to call on every frame."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifier recorded in ``FrameResult.model_versions['pose']``."""

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray, boxes: Sequence[tuple[int, Box]]) -> dict[int, Keypoints | None]:
        """Keypoints per track id (``None`` when no person could be matched in that track's crop).

        Args:
            frame_bgr: The full frame.
            boxes: ``(track_id, box_xyxy)`` in frame pixels, already clipped to the frame.
        """


def crop_region(box: Box, margin: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer crop around ``box`` enlarged by ``margin`` x its longer side, clipped to the frame.

    Returns None when the clipped crop is smaller than a few pixels (nothing to estimate on).
    """
    x1, y1, x2, y2 = box
    pad = margin * max(x2 - x1, y2 - y1)
    cx1 = int(max(0.0, np.floor(x1 - pad)))
    cy1 = int(max(0.0, np.floor(y1 - pad)))
    cx2 = int(min(float(width), np.ceil(x2 + pad)))
    cy2 = int(min(float(height), np.ceil(y2 + pad)))
    if cx2 - cx1 < _MIN_CROP_PX or cy2 - cy1 < _MIN_CROP_PX:
        return None
    return cx1, cy1, cx2, cy2


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes (0 for degenerate boxes)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_match(target: Box, candidates: Sequence[Box], min_iou: float) -> int | None:
    """Index of the candidate overlapping ``target`` most, or None if none reaches ``min_iou``."""
    best, best_iou = None, min_iou
    for i, cand in enumerate(candidates):
        overlap = box_iou(target, cand)
        if overlap >= best_iou:
            best, best_iou = i, overlap
    return best


def to_frame_keypoints(kpts: np.ndarray, offset: tuple[int, int], width: int, height: int) -> Keypoints:
    """Crop-pixel keypoints (17x3: x, y, conf) -> frame pixels.

    Confidence is clipped to [0, 1]. A point that is not finite or falls outside the frame is reported as
    ``(0, 0, 0)`` (not detected) rather than moved, so no joint is ever invented or displaced. Values are
    rounded (2 decimals for pixels, 3 for confidence) like the rest of ``TrackObs``.
    """
    arr = np.asarray(kpts, dtype=float)
    if arr.shape != (COCO_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints of shape (17, 3), got {arr.shape}")
    ox, oy = offset
    out: list[tuple[float, float, float]] = []
    for x, y, conf in arr:
        fx, fy = x + ox, y + oy
        c = float(min(1.0, max(0.0, conf))) if np.isfinite(conf) else 0.0
        if c == 0.0 or not (np.isfinite(fx) and np.isfinite(fy)) or not (0 <= fx <= width and 0 <= fy <= height):
            out.append((0.0, 0.0, 0.0))
        else:
            out.append((round(float(fx), 2), round(float(fy), 2), round(c, 3)))
    return tuple(out)
