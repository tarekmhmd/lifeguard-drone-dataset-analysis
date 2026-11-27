"""The real pretrained YOLO11s-pose file (needs ultralytics and models/yolo11s-pose.pt; skipped otherwise)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lifeguard.common.models import ModelRegistry
from lifeguard.contracts.errors import ModelLoadError

ROOT = Path(__file__).resolve().parents[2]
POSE = ROOT / "models" / "yolo11s-pose.pt"
DETECTOR = ROOT / "models" / "z5cc_s448_final.pt"

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not POSE.is_file(), reason="models/yolo11s-pose.pt not present")


def test_official_weights_load_verify_and_run_on_crops() -> None:
    from lifeguard.ai.pose.ultralytics_pose import UltralyticsPoseEstimator

    ModelRegistry.load(ROOT / "models" / "registry.yaml").verify("pose_yolo11s_coco_v1")
    est = UltralyticsPoseEstimator(POSE, device="cpu")
    assert est.model_version.startswith("yolo11s-pose.pt:sha256:1060bda4a270")
    out = est.estimate(np.zeros((480, 640, 3), np.uint8), [(1, (100.0, 100.0, 200.0, 300.0)),
                                                           (2, (400.0, 50.0, 500.0, 250.0))])
    assert set(out) == {1, 2}
    for kp in out.values():   # a black frame has no person: None, never invented keypoints
        assert kp is None or (len(kp) == 17 and all(0.0 <= c <= 1.0 for _, _, c in kp))


@pytest.mark.skipif(not DETECTOR.is_file(), reason="detector weights not present")
def test_a_detector_checkpoint_is_rejected_as_pose_model() -> None:
    from lifeguard.ai.pose.ultralytics_pose import UltralyticsPoseEstimator

    with pytest.raises(ModelLoadError, match="not a 17-keypoint pose model"):
        UltralyticsPoseEstimator(DETECTOR, device="cpu")
