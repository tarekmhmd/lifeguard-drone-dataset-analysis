"""Pose integration: geometry helpers, estimator association, engine wiring, failure isolation, config/factory,
and the guarantee that pose never changes tracking or risk decisions.

Model-free: pose estimators here are test doubles (a fake Ultralytics model object for the real estimator's
crop/association code). Loading the real ``yolo11s-pose.pt`` is covered in ``test_pose_real_model.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lifeguard.ai.config import AIConfig, PoseConfig, load_ai_config
from lifeguard.ai.detection.base import Detection
from lifeguard.ai.perception.factory import _pose_estimator
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.pose.base import PoseEstimator, best_match, box_iou, crop_region, to_frame_keypoints
from lifeguard.ai.pose.ultralytics_pose import UltralyticsPoseEstimator
from lifeguard.common.models import ModelIntegrityError, sha256_file
from lifeguard.contracts.enums import RiskLevel, Status, TrackState
from lifeguard.contracts.errors import ModelLoadError
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import FrameResult
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.risk.config import RiskConfig
from lifeguard.system.risk.engine import RuleBasedRiskEngine
from tests.ai.test_real_engine import ScriptedDetector

ROOT = Path(__file__).resolve().parents[2]
T0 = "2026-01-01T00:00:00.000Z"
GEO = StubPoolGeometry()
FRAME = np.zeros((1080, 1920, 3), np.uint8)


def _box_at(x_m: float, y_m: float) -> tuple[float, float, float, float]:
    u, v = GEO.to_image(x_m, y_m)
    return (u - 20.0, v - 30.0, u + 20.0, v + 30.0)


def _kpts_for(box: tuple[float, float, float, float], conf: float = 0.9) -> tuple[tuple[float, float, float], ...]:
    """17 points spread inside ``box`` (a recognisable layout per track)."""
    x1, y1, x2, y2 = box
    return tuple((round(x1 + (x2 - x1) * (i + 1) / 18, 2), round(y1 + (y2 - y1) * (i + 1) / 18, 2), conf)
                 for i in range(17))


class BoxPose(PoseEstimator):
    """Test double: keypoints laid out inside each requested box; records every call."""

    def __init__(self, conf: float = 0.9) -> None:
        self.calls: list[list[tuple[int, tuple[float, float, float, float]]]] = []
        self._conf = conf

    @property
    def model_version(self) -> str:
        return "boxpose"

    def estimate(self, frame_bgr, boxes):
        self.calls.append(list(boxes))
        return {tid: _kpts_for(box, self._conf) for tid, box in boxes}


class FailingPose(BoxPose):
    def estimate(self, frame_bgr, boxes):
        raise RuntimeError("synthetic pose failure")


class MalformedPose(BoxPose):
    def estimate(self, frame_bgr, boxes):
        return {tid: ((1.0, 1.0, 0.5),) * 3 for tid, _ in boxes}   # 3 points instead of 17


def _run(engine: RealPerceptionEngine, n: int) -> list[FrameResult]:
    return [engine.process(FRAME, i, 100.0 + i * 0.1, add_seconds_utc(T0, i * 0.1), i * 0.1) for i in range(n)]


def _two_swimmers(n: int) -> list[list[Detection]]:
    return [[Detection(_box_at(8.0 + 0.05 * i, 4.0), 0.9), Detection(_box_at(16.0, 9.0 - 0.05 * i), 0.8)]
            for i in range(n)]


# ------------------------------------------------------------------ geometry helpers

def test_crop_region_adds_margin_and_clips_to_frame() -> None:
    assert crop_region((100, 200, 140, 300), 0.1, 1920, 1080) == (90, 190, 150, 310)
    assert crop_region((0, 0, 50, 100), 0.5, 1920, 1080) == (0, 0, 100, 150)        # clipped at 0
    assert crop_region((1900, 1050, 1920, 1080), 0.5, 1920, 1080) == (1885, 1035, 1920, 1080)
    assert crop_region((10, 10, 12, 12), 0.0, 1920, 1080) is None                     # too small


def test_keypoints_are_moved_from_crop_to_frame_pixels() -> None:
    k = np.zeros((17, 3))
    k[0] = (5.0, 7.0, 0.8)          # inside
    k[1] = (5.0, 7.0, 1.7)          # confidence clipped to 1
    k[2] = (5.0, 7.0, 0.0)          # not detected stays (0, 0, 0)
    k[3] = (-500.0, 7.0, 0.9)       # lands outside the frame -> dropped, never moved
    k[4] = (np.nan, 7.0, 0.9)       # not finite -> dropped
    out = to_frame_keypoints(k, (100, 200), 1920, 1080)
    assert out[0] == (105.0, 207.0, 0.8)
    assert out[1] == (105.0, 207.0, 1.0)
    assert out[2] == out[3] == out[4] == (0.0, 0.0, 0.0)
    assert len(out) == 17
    with pytest.raises(ValueError):
        to_frame_keypoints(np.zeros((16, 3)), (0, 0), 10, 10)


def test_best_match_picks_the_tracked_person_not_the_neighbour() -> None:
    target = (40, 40, 80, 120)
    neighbour, same = (85, 40, 125, 120), (42, 38, 82, 118)
    assert best_match(target, [neighbour, same], 0.3) == 1
    assert best_match(target, [neighbour], 0.3) is None
    assert box_iou((0, 0, 0, 0), (0, 0, 1, 1)) == 0.0


# ------------------------------------------------- real estimator code with a fake model

def _fake_result(boxes: list[tuple[float, float, float, float]], kpts: list[np.ndarray]) -> SimpleNamespace:
    """Minimal stand-in for an Ultralytics pose ``Results`` object."""
    arr_boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    arr_kpts = np.array(kpts, dtype=float).reshape(-1, 17, 3)
    return SimpleNamespace(boxes=_Boxes(arr_boxes), keypoints=SimpleNamespace(data=[_T(k) for k in arr_kpts]))


class _T:
    def __init__(self, a: np.ndarray) -> None:
        self._a = a

    def cpu(self) -> _T:
        return self

    def numpy(self) -> np.ndarray:
        return self._a


class _Boxes:
    def __init__(self, a: np.ndarray) -> None:
        self.xyxy = _T(a)
        self._n = len(a)

    def __len__(self) -> int:
        return self._n


class _FakeModel:
    """Returns, per crop, a neighbour person at the crop's left edge and the tracked person in the middle."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, crops, **kwargs):
        self.calls.append({"n": len(crops), "shapes": [c.shape for c in crops], **kwargs})
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            neighbour = (0.0, 0.0, w * 0.2, h * 0.5)
            person = (w * 0.15, h * 0.12, w * 0.85, h * 0.88)
            k_neigh = np.tile([1.0, 1.0, 0.9], (17, 1))
            k_person = np.tile([w / 2, h / 2, 0.8], (17, 1))
            out.append(_fake_result([neighbour, person], [k_neigh, k_person]))
        return out


def _estimator(model: _FakeModel, max_tracks: int = 8) -> UltralyticsPoseEstimator:
    est = object.__new__(UltralyticsPoseEstimator)
    est._model, est._conf, est._imgsz, est._device = model, 0.25, 320, None
    est._margin, est._min_iou, est._max_tracks, est._half, est._version = 0.15, 0.3, max_tracks, False, "fake"
    return est


def test_estimator_batches_crops_and_maps_the_matched_person_to_frame_pixels() -> None:
    model = _FakeModel()
    est = _estimator(model)
    boxes = [(1, (100.0, 200.0, 200.0, 400.0)), (2, (1000.0, 500.0, 1100.0, 700.0))]
    out = est.estimate(FRAME, boxes)
    assert len(model.calls) == 1 and model.calls[0]["n"] == 2        # one batched call for all swimmers
    assert model.calls[0]["classes"] == [0]
    for tid, (x1, y1, x2, y2) in boxes:
        crop = crop_region((x1, y1, x2, y2), 0.15, 1920, 1080)
        cx = crop[0] + (crop[2] - crop[0]) / 2
        cy = crop[1] + (crop[3] - crop[1]) / 2
        assert out[tid][0] == (round(cx, 2), round(cy, 2), 0.8)      # the person, not the neighbour (1, 1)


def test_estimator_respects_max_tracks_and_skips_tiny_boxes() -> None:
    model = _FakeModel()
    out = _estimator(model, max_tracks=1).estimate(
        FRAME, [(1, (100.0, 200.0, 200.0, 400.0)), (2, (1000.0, 500.0, 1100.0, 700.0))])
    assert out[1] is not None and out[2] is None and model.calls[0]["n"] == 1
    model = _FakeModel()
    assert _estimator(model).estimate(FRAME, [(3, (5.0, 5.0, 7.0, 7.0))]) == {3: None}
    assert model.calls == []                                           # nothing worth a model call


# ------------------------------------------------------------------- engine wiring

def test_keypoints_are_attached_to_the_correct_track() -> None:
    pose = BoxPose()
    engine = RealPerceptionEngine(ScriptedDetector(_two_swimmers(10)), GEO, "cam_a", "s1", pose_estimator=pose)
    last = _run(engine, 10)[-1]
    assert last.status == Status.OK and len(last.tracks) == 2
    assert FrameResult.model_validate_json(last.model_dump_json()) == last
    for track in last.tracks:
        assert track.keypoints == _kpts_for(track.box_xyxy)            # each track got its own box's pose
    assert last.model_versions["pose"] == "boxpose"
    assert engine.pose_stats.tracks_with_pose == engine.pose_stats.tracks_requested > 0
    # Requests are ordered by detector confidence (0.9 swimmer first), so max_tracks keeps the surest ones.
    first_id = pose.calls[-1][0][0]
    assert next(t for t in last.tracks if t.track_id == first_id).det_conf == pytest.approx(0.9)


def test_lost_tracks_get_no_pose() -> None:
    script = [[Detection(_box_at(10.0, 6.0), 0.9)] for _ in range(5)] + [[] for _ in range(3)]
    pose = BoxPose()
    results = _run(RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1", pose_estimator=pose), 8)
    lost = [t for r in results for t in r.tracks if t.state == TrackState.LOST]
    assert lost and all(t.keypoints is None for t in lost)
    visible_frames = sum(any(t.state == TrackState.TRACKED for t in r.tracks) for r in results)
    assert visible_frames == 4 and len(pose.calls) == visible_frames   # LOST (predicted) boxes get no request


def _strip(result: FrameResult) -> str:
    data = json.loads(result.model_dump_json())
    data.pop("latency_ms")
    data["model_versions"].pop("pose", None)
    for track in data["tracks"]:
        track["keypoints"] = None
    return json.dumps(data, sort_keys=True)


@pytest.mark.parametrize("pose_cls", [BoxPose, FailingPose, MalformedPose])
def test_pose_never_changes_tracks_or_status(pose_cls) -> None:
    script = _two_swimmers(30)
    plain = _run(RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1"), 30)
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1", pose_estimator=pose_cls())
    posed = _run(engine, 30)
    assert all(r.status == Status.OK for r in posed)
    assert [_strip(r) for r in plain] == [_strip(r) for r in posed]
    if pose_cls is FailingPose:
        visible_frames = sum(any(t.state == TrackState.TRACKED for t in r.tracks) for r in posed)
        assert engine.pose_stats.errors == visible_frames > 0
        assert all(t.keypoints is None for r in posed for t in r.tracks)
    if pose_cls is MalformedPose:
        assert all(t.keypoints is None for r in posed for t in r.tracks)


def test_disabled_pose_restores_previous_output_exactly() -> None:
    script = _two_swimmers(10)
    before = _run(RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1"), 10)
    after = _run(RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1", pose_estimator=None,
                                      pose_status="disabled"), 10)
    drop = lambda r: r.model_copy(update={"latency_ms": 0.0})
    assert [drop(r) for r in before] == [drop(r) for r in after]
    assert "pose" not in after[-1].model_versions and all(t.keypoints is None for t in after[-1].tracks)


# ------------------------------------------------------------ risk isolation (safety)

def _events(results: list[FrameResult]) -> list[tuple]:
    risk = RuleBasedRiskEngine(RiskConfig())
    out = []
    for r in results:
        out += [(e.track_uid, e.from_level, e.to_level, e.reason, e.risk, e.t_capture) for e in risk.update(r)]
    return out


def test_pose_cannot_trigger_or_change_risk_decisions() -> None:
    # A steady swimmer: no risk events with or without pose, even with every joint at full confidence.
    steady = [[Detection(_box_at(10.0 + 0.05 * i, 6.0), 0.9)] for i in range(80)]
    plain = _run(RealPerceptionEngine(ScriptedDetector(steady), GEO, "cam_a", "s1"), 80)
    posed = _run(RealPerceptionEngine(ScriptedDetector(steady), GEO, "cam_a", "s1",
                                      pose_estimator=BoxPose(conf=1.0)), 80)
    assert _events(plain) == _events(posed)
    assert not any(e[2] in (RiskLevel.PRE_ALERT, RiskLevel.ALARM) for e in _events(posed))
    # A real submersion: the same escalation, at the same times, with or without pose.
    sub = [[Detection(_box_at(10.0, 6.0), 0.9)] for _ in range(20)] + [[] for _ in range(150)]
    plain = _run(RealPerceptionEngine(ScriptedDetector(sub), GEO, "cam_a", "s1"), 170)
    posed = _run(RealPerceptionEngine(ScriptedDetector(sub), GEO, "cam_a", "s1", pose_estimator=BoxPose()), 170)
    assert _events(plain) == _events(posed)
    assert any(e[2] == RiskLevel.ALARM for e in _events(posed))


def test_risk_engine_ignores_keypoints_entirely() -> None:
    results = _run(RealPerceptionEngine(ScriptedDetector(_two_swimmers(40)), GEO, "cam_a", "s1",
                                        pose_estimator=BoxPose()), 40)
    zeroed = [r.model_copy(update={"tracks": tuple(t.model_copy(update={"keypoints": None}) for t in r.tracks)})
              for r in results]
    assert _events(results) == _events(zeroed)
    assert RiskConfig().weights.model_dump().keys() == {"submersion", "static", "visibility", "deep_zone",
                                                        "behavior"}  # no pose weight exists


# ------------------------------------------------------------------ config and factory

def test_pose_config_defaults_and_validation() -> None:
    assert PoseConfig().enabled is False
    with pytest.raises(ValueError):
        PoseConfig(enabled=True)                                     # no weights / model
    with pytest.raises(ValueError):
        PoseConfig(enabled=True, weights=Path("a.pt"), model="pose_yolo11s_coco_v1")
    with pytest.raises(ValueError):
        AIConfig(engine="mock", mock={"fixture_path": "x.jsonl"},
                 pose={"enabled": True, "model": "pose_yolo11s_coco_v1"})   # name needs a registry
    assert load_ai_config(ROOT / "configs" / "ai_z5cc.yaml").pose.enabled is False
    cfg = load_ai_config(ROOT / "configs" / "ai_z5cc_pose.yaml")
    assert cfg.pose.enabled and cfg.pose.weights == (ROOT / "models" / "yolo11s-pose.pt").resolve()


def _cfg(tmp_path: Path, **pose) -> AIConfig:
    return AIConfig(engine="mock", mock={"fixture_path": tmp_path / "x.jsonl"}, pose=pose)


def test_missing_pose_weights_fall_back_safely_or_refuse(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    est, status = _pose_estimator(_cfg(tmp_path, enabled=True, weights=missing), None)
    assert est is None and status == "unavailable: weights file missing"
    with pytest.raises(ModelLoadError, match="pose weights not found"):
        _pose_estimator(_cfg(tmp_path, enabled=True, weights=missing, allow_missing_weights=False), None)
    assert _pose_estimator(_cfg(tmp_path), None) == (None, "disabled")


def test_pose_checksum_mismatch_always_refuses(tmp_path: Path) -> None:
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"not the registered file")
    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(
        "registry_version: 1\nmodels:\n  - name: pose_yolo11s_coco_v1\n    version: 1.0.0\n    kind: pose\n"
        f"    path: pose.pt\n    sha256: '{'0' * 64}'\n", encoding="utf-8")
    from lifeguard.common.models import ModelRegistry

    registry = ModelRegistry.load(registry_file)
    with pytest.raises(ModelIntegrityError, match="checksum mismatch"):
        _pose_estimator(_cfg(tmp_path, enabled=True, weights=weights), registry)
    by_name = AIConfig(engine="mock", mock={"fixture_path": tmp_path / "x.jsonl"}, model_registry=registry_file,
                       pose={"enabled": True, "model": "pose_yolo11s_coco_v1"})
    with pytest.raises(ModelIntegrityError):
        _pose_estimator(by_name, registry)
    assert sha256_file(weights) != "0" * 64


def test_registry_lists_the_official_pose_weights() -> None:
    from lifeguard.common.models import ModelRegistry

    entry = ModelRegistry.load(ROOT / "models" / "registry.yaml").get("pose_yolo11s_coco_v1")
    assert entry.kind == "pose" and entry.mock is False
    assert entry.sha256 == "1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba"
    assert entry.url == "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt"
