"""Dashboard pose overlay: confidence-aware joints, no fabricated limbs, scaling, the swimmer-table column."""
from __future__ import annotations

import numpy as np
import pytest

from lifeguard.app.dashboard_controller import POSE_COLOURS, annotate, draw_pose, pose_summary
from lifeguard.contracts.enums import DetClass, RiskLevel, TrackState
from lifeguard.contracts.types import FrameResult, TrackObs

cv2 = pytest.importorskip("cv2")

W, H = 400, 300
UID = "cam_a:s1:1"


def _keypoints(low: set[int] = frozenset(), missing: set[int] = frozenset()):
    """A simple standing figure; ``low`` joints at conf 0.2, ``missing`` joints at (0, 0, 0)."""
    xy = [(200, 60), (195, 55), (205, 55), (190, 58), (210, 58),       # head
          (180, 90), (220, 90), (170, 130), (230, 130), (165, 165), (235, 165),   # arms
          (185, 170), (215, 170), (183, 215), (217, 215), (182, 260), (218, 260)]  # hips, legs
    out = []
    for i, (x, y) in enumerate(xy):
        if i in missing:
            out.append((0.0, 0.0, 0.0))
        else:
            out.append((float(x), float(y), 0.2 if i in low else 0.9))
    return tuple(out)


def _result(keypoints) -> FrameResult:
    track = TrackObs(track_id=1, track_uid=UID, cls=DetClass.PERSON, box_xyxy=(150.0, 40.0, 250.0, 280.0),
                     det_conf=0.9, state=TrackState.TRACKED, pool_xy=None, in_water=False, zone=None,
                     keypoints=keypoints)
    return FrameResult(cam_id="cam_a", session_id="s1", frame_idx=0, t_capture=0.0,
                       t_utc="2026-01-01T00:00:00.000Z", frame_width=W, frame_height=H, latency_ms=1.0,
                       tracks=(track,))


def _bgr(rgb: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    r, g, b = rgb[y, x]
    return int(b), int(g), int(r)


def test_confident_joints_and_limbs_are_drawn() -> None:
    frame = np.zeros((H, W, 3), np.uint8)
    img = annotate(frame, _result(_keypoints()), {UID: (RiskLevel.NORMAL, 0.1)})
    assert _bgr(img, 182, 260) == POSE_COLOURS["left"]       # left ankle joint
    assert _bgr(img, 218, 260) == POSE_COLOURS["right"]      # right ankle joint
    assert _bgr(img, 200, 60) == POSE_COLOURS["head"]        # nose
    assert img[90, 200].any()                                # shoulder-shoulder torso edge midpoint


def test_low_confidence_and_missing_joints_are_not_drawn_and_have_no_limbs() -> None:
    frame = np.zeros((H, W, 3), np.uint8)
    kp = _keypoints(low={15}, missing={16})                  # left ankle uncertain, right ankle missing
    img = annotate(frame, _result(kp), {UID: (RiskLevel.NORMAL, 0.1)})
    assert not img[260, 182].any() and not img[260, 218].any()
    assert not img[238, 182].any() and not img[238, 218].any()   # no knee-ankle edges (midpoints)
    assert not img[0:5, 0:5].any()                               # nothing drawn at the (0, 0) placeholder
    assert pose_summary(kp) == "15/17" and pose_summary(None) == "-"


def test_pose_overlay_follows_display_scaling_and_can_be_switched_off() -> None:
    small = np.zeros((H // 2, W // 2, 3), np.uint8)              # display frame at half the result size
    img = annotate(small, _result(_keypoints()), {UID: (RiskLevel.NORMAL, 0.1)})
    assert _bgr(img, 91, 130) == POSE_COLOURS["left"]             # (182, 260) scaled by 0.5
    off = annotate(np.zeros((H, W, 3), np.uint8), _result(_keypoints()), {UID: (RiskLevel.NORMAL, 0.1)},
                   show_pose=False)
    assert not off[260, 182].any()


def test_draw_pose_counts_drawn_joints() -> None:
    img = np.zeros((H, W, 3), np.uint8)
    assert draw_pose(img, _keypoints(low={0, 1}, missing={2}), 1.0, 1.0) == 14


def test_runtime_lists_pose_component_only_when_enabled(tmp_path) -> None:
    from pathlib import Path

    from lifeguard.ai.config import load_ai_config
    from lifeguard.ai.perception.real_engine import RealPerceptionEngine
    from lifeguard.app.runtime import LifeguardRuntime
    from lifeguard.system.config import load_system_config
    from lifeguard.system.geometry.stub import StubPoolGeometry
    from tests.ai.test_pose import BoxPose
    from tests.ai.test_real_engine import ScriptedDetector

    root = Path(__file__).resolve().parents[2]
    ai = load_ai_config(root / "configs" / "ai.yaml")
    sys_cfg = load_system_config(root / "configs" / "system.yaml")
    off = LifeguardRuntime(ai, sys_cfg, (1920, 1080), session_id="p_off")
    assert "pose" not in off.components
    off.close()
    ai.pose.enabled, ai.pose.weights = True, tmp_path / "pose.pt"
    engine = RealPerceptionEngine(ScriptedDetector([]), StubPoolGeometry(), "cam_sim", "p_on",
                                  pose_estimator=BoxPose())
    on = LifeguardRuntime(ai, sys_cfg, (1920, 1080), session_id="p_on", engine=engine)
    assert on.components["pose"] == "boxpose (display only; not used by the risk engine)"
    on.close()
