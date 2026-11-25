"""UltralyticsDetector with a fake ``ultralytics`` module (no real model): class mapping and NMS mode."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from lifeguard.ai.detection.ultralytics_detector import UltralyticsDetector
from lifeguard.contracts.enums import DetClass


class _Tensor:
    def __init__(self, values: list) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self, rows: list[tuple[tuple[float, float, float, float], float, int]]) -> None:
        self.xyxy = _Tensor([r[0] for r in rows])
        self.conf = _Tensor([r[1] for r in rows])
        self.cls = _Tensor([r[2] for r in rows])

    def __len__(self) -> int:
        return len(self.conf.numpy())


class _FakeYOLO:
    """Stands in for ``ultralytics.YOLO``; records the predict() keyword arguments."""

    rows: ClassVar[list] = []
    last_kwargs: ClassVar[dict] = {}

    def __init__(self, path: str) -> None:
        self.path = path

    def predict(self, frame: np.ndarray, **kwargs: object) -> list:
        _FakeYOLO.last_kwargs = kwargs
        return [types.SimpleNamespace(boxes=_Boxes(_FakeYOLO.rows))]


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake weights")
    return path


FRAME = np.zeros((224, 224, 3), np.uint8)


def test_single_class_uses_per_class_nms(weights: Path) -> None:
    _FakeYOLO.rows = [((10, 10, 50, 80), 0.9, 0)]
    det = UltralyticsDetector(weights)
    (d,) = det.detect(FRAME)
    assert d.cls == DetClass.PERSON and d.conf == pytest.approx(0.9)
    assert _FakeYOLO.last_kwargs["agnostic_nms"] is False
    assert _FakeYOLO.last_kwargs["classes"] == [0]


def test_two_classes_mapped_to_person_use_agnostic_nms(weights: Path) -> None:
    _FakeYOLO.rows = [((10, 10, 50, 80), 0.8, 0), ((100, 20, 140, 90), 0.7, 1), ((0, 0, 5, 5), 0.6, 2)]
    det = UltralyticsDetector(weights, class_map={0: DetClass.PERSON, 1: DetClass.PERSON})
    out = det.detect(FRAME)
    assert [d.cls for d in out] == [DetClass.PERSON, DetClass.PERSON]  # class 2 is not mapped: dropped
    assert _FakeYOLO.last_kwargs["agnostic_nms"] is True
    assert _FakeYOLO.last_kwargs["classes"] == [0, 1]
    assert det.model_version.startswith("model.pt:sha256:")


def test_distress_classes_report_their_confidence(weights: Path) -> None:
    _FakeYOLO.rows = [((10, 10, 50, 80), 0.8, 0), ((100, 20, 140, 90), 0.7, 1)]
    det = UltralyticsDetector(
        weights, class_map={0: DetClass.PERSON, 1: DetClass.PERSON}, distress_classes=(1,)
    )
    swim, drown = det.detect(FRAME)
    assert swim.distress == 0.0
    assert drown.distress == drown.conf == 0.7
