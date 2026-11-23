"""YOLO person detector on Ultralytics (roadmap P4-3). Optional dependency: ``pip install -e ".[vision]"``.

REAL implementation. With COCO-pretrained weights it localises people in general; it has NOT been
validated on pool video, and no pool-trained weights exist in this repository (roadmap P4-7, data-blocked).
Note: Ultralytics is AGPL-3.0 (open decision OD-7).

Weights must be a local file: the detector never downloads models implicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.contracts.enums import DetClass
from lifeguard.contracts.errors import ModelLoadError

_HASH_CHUNK = 1 << 20
_HASH_PREFIX = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsDetector(Detector):
    """Wraps ``ultralytics.YOLO`` for single-frame person detection."""

    def __init__(
        self,
        weights: Path | str,
        conf: float = 0.1,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
        class_map: dict[int, DetClass] | None = None,
    ) -> None:
        """Load the model.

        Args:
            weights: Local ``.pt``/``.onnx``/``.engine`` file.
            conf: Minimum confidence. Keep it low (≈0.1) so the tracker's low-confidence stage gets input.
            iou: NMS IoU threshold.
            imgsz: Inference image size.
            device: ``"0"``, ``"cpu"`` or None (Ultralytics default).
            class_map: Model class index -> contract class. Default ``{0: PERSON}`` (COCO).

        Raises:
            ModelLoadError: If Ultralytics is not installed, the weights file is missing, or loading fails.
        """
        path = Path(weights)
        if not path.is_file():
            raise ModelLoadError(f"detector weights not found: {path}")
        try:
            from ultralytics import YOLO  # optional dependency, imported lazily
        except ImportError as err:
            raise ModelLoadError(
                'ultralytics is not installed; install the vision extra: pip install -e ".[vision]"'
            ) from err
        try:
            self._model = YOLO(str(path))
        except Exception as err:  # any loader failure is a model-load failure
            raise ModelLoadError(f"cannot load detector weights {path}: {err}") from err

        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._class_map = dict(class_map) if class_map is not None else {0: DetClass.PERSON}
        self._version = f"{path.name}:sha256:{_sha256(path)[:_HASH_PREFIX]}"

    @property
    def model_version(self) -> str:
        return self._version

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            classes=sorted(self._class_map),
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            det_cls = self._class_map.get(int(cls))
            if det_cls is None:
                continue
            detections.append(
                Detection((float(x1), float(y1), float(x2), float(y2)), float(conf), det_cls)
            )
        return detections
