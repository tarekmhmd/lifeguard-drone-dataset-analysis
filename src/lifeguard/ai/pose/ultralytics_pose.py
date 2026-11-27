"""YOLO11-pose keypoints for tracked swimmers (Ultralytics). Optional dependency: ``pip install -e ".[vision]"``.

REAL implementation, used with the official COCO-pretrained ``yolo11s-pose.pt`` (17 COCO keypoints). It is NOT
validated on pool video as a pose reference (see docs/POSE_INTEGRATION.md): keypoints are displayed and
recorded, never used for risk decisions.

Pipeline per frame: one crop per tracked box (box + margin, clipped), all crops in ONE batched ``predict``
call, then per crop the detected person that overlaps the tracked box best. The pose model's own person
boxes are used only for that association; they never create tracks. Weights must be a local file: nothing
is downloaded implicitly.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lifeguard.ai.pose.base import (
    COCO_KEYPOINTS,
    Box,
    Keypoints,
    PoseEstimator,
    best_match,
    crop_region,
    to_frame_keypoints,
)
from lifeguard.contracts.errors import ModelLoadError

_HASH_CHUNK = 1 << 20
_HASH_PREFIX = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsPoseEstimator(PoseEstimator):
    """Crop-based YOLO pose estimation on tracked boxes."""

    def __init__(
        self,
        weights: Path | str,
        conf: float = 0.25,
        imgsz: int = 320,
        device: str | None = None,
        crop_margin: float = 0.15,
        min_iou: float = 0.3,
        max_tracks: int = 8,
        half: bool = False,
    ) -> None:
        """Load the pose model.

        Args:
            weights: Local pose ``.pt`` file (17 COCO keypoints).
            conf: Minimum person confidence of the pose model inside a crop.
            imgsz: Inference size for the crops.
            device: ``"0"``, ``"cpu"`` or None (Ultralytics default).
            crop_margin: Crop enlargement as a fraction of the box's longer side.
            min_iou: Minimum overlap between the tracked box and the pose model's person box (crop pixels).
            max_tracks: At most this many tracks per frame get a pose (highest-confidence first, chosen by
                the caller's ordering); the rest get ``None``.
            half: FP16 inference (CUDA only).

        Raises:
            ModelLoadError: If Ultralytics is missing, the file is missing or not a 17-keypoint pose model.
        """
        path = Path(weights)
        if not path.is_file():
            raise ModelLoadError(f"pose weights not found: {path}")
        try:
            from ultralytics import YOLO  # optional dependency, imported lazily
        except ImportError as err:
            raise ModelLoadError(
                'ultralytics is not installed; install the vision extra: pip install -e ".[vision]"'
            ) from err
        try:
            self._model = YOLO(str(path))
        except Exception as err:  # any loader failure is a model-load failure
            raise ModelLoadError(f"cannot load pose weights {path}: {err}") from err
        kpt_shape = list(getattr(self._model.model, "yaml", {}).get("kpt_shape", []) or [])
        if self._model.task != "pose" or kpt_shape[:1] != [COCO_KEYPOINTS]:
            raise ModelLoadError(
                f"{path} is not a 17-keypoint pose model (task={self._model.task!r}, kpt_shape={kpt_shape})"
            )
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._margin = crop_margin
        self._min_iou = min_iou
        self._max_tracks = max_tracks
        self._half = half
        self._version = f"{path.name}:sha256:{_sha256(path)[:_HASH_PREFIX]}"

    @property
    def model_version(self) -> str:
        return self._version

    def estimate(self, frame_bgr: np.ndarray, boxes: Sequence[tuple[int, Box]]) -> dict[int, Keypoints | None]:
        height, width = frame_bgr.shape[:2]
        out: dict[int, Keypoints | None] = {track_id: None for track_id, _ in boxes}
        jobs: list[tuple[int, Box, tuple[int, int, int, int]]] = []
        for track_id, box in list(boxes)[: self._max_tracks]:
            region = crop_region(box, self._margin, width, height)
            if region is not None:
                jobs.append((track_id, box, region))
        if not jobs:
            return out
        crops = [np.ascontiguousarray(frame_bgr[y1:y2, x1:x2]) for _, _, (x1, y1, x2, y2) in jobs]
        results = self._model.predict(
            crops, conf=self._conf, imgsz=self._imgsz, device=self._device, half=self._half,
            classes=[0], verbose=False,
        )
        for (track_id, box, (cx1, cy1, _, _)), result in zip(jobs, results):
            if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
                continue
            candidates = [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]
            target = (box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1)
            idx = best_match(target, candidates, self._min_iou)
            if idx is None:
                continue
            data = result.keypoints.data[idx].cpu().numpy()      # (17, 3): x, y, conf in crop pixels
            out[track_id] = to_frame_keypoints(data, (cx1, cy1), width, height)
        return out
