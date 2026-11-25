"""RealPerceptionEngine contract behaviour with a scripted (test-double) detector; config and factory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.factory import create_engine
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.contracts.enums import Status, TrackState
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.contracts.interfaces import PerceptionEngine
from lifeguard.contracts.timeutil import add_seconds_utc
from lifeguard.contracts.types import DISTRESS_FEATURE, FrameResult
from lifeguard.system.geometry.stub import StubPoolGeometry

T0 = "2026-01-01T00:00:00.000Z"
FRAME = np.zeros((1080, 1920, 3), np.uint8)
GEO = StubPoolGeometry()


class ScriptedDetector(Detector):
    """Test double: returns pre-recorded detections in order (not a model)."""

    def __init__(self, script: list[list[Detection]], fail_at: int | None = None) -> None:
        self._script = script
        self._i = 0
        self._fail_at = fail_at

    @property
    def model_version(self) -> str:
        return "scripted"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        i = self._i
        self._i += 1
        if self._fail_at is not None and i == self._fail_at:
            raise RuntimeError("synthetic detector failure")
        return self._script[i] if i < len(self._script) else []


def _box_at(x_m: float, y_m: float) -> tuple[float, float, float, float]:
    u, v = GEO.to_image(x_m, y_m)
    return (u - 20.0, v - 30.0, u + 20.0, v + 30.0)


def _run(engine: RealPerceptionEngine, n: int) -> list[FrameResult]:
    out = []
    for i in range(n):
        t = round(i * 0.1, 3)
        out.append(engine.process(FRAME, i, 100.0 + t, add_seconds_utc(T0, t), t))
    return out


def test_real_engine_implements_contract_and_reports_tracks() -> None:
    script = [[Detection(_box_at(10.0 + 0.1 * i, 6.0), 0.9)] for i in range(20)]
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1")
    assert isinstance(engine, PerceptionEngine)
    results = _run(engine, 20)

    last = results[-1]
    assert last.status == Status.OK
    assert FrameResult.model_validate_json(last.model_dump_json()) == last
    (track,) = last.tracks
    assert track.track_uid == "cam_a:s1:1"
    assert track.state == TrackState.TRACKED
    assert track.in_water and track.pool_xy is not None
    assert track.pool_xy[0] == pytest.approx(11.9, abs=0.01)
    assert "visibility_ratio" in track.features and "speed_mps" in track.features
    assert last.model_versions["engine"] == "real"
    hb = engine.heartbeat()
    assert hb.frames_processed == 20 and hb.fps == pytest.approx(10.0, rel=1e-3)


def test_disappearance_produces_submersion_entries() -> None:
    script = [[Detection(_box_at(10.0, 6.0), 0.9)] for _ in range(10)] + [[] for _ in range(30)]
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1")
    results = _run(engine, 40)
    (sub,) = results[-1].submersions
    assert sub.track_uid == "cam_a:s1:1"
    assert sub.since_t_capture == pytest.approx(100.9)
    assert sub.seconds_submerged == pytest.approx(3.0, abs=1e-6)


def test_person_outside_the_pool_is_not_in_water() -> None:
    script = [[Detection((5.0, 5.0, 45.0, 65.0), 0.9)] for _ in range(5)]  # image corner, outside pool
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1")
    (track,) = _run(engine, 5)[-1].tracks
    assert not track.in_water and track.pool_xy is None and track.zone is None


def test_detector_failure_becomes_error_result_and_engine_continues() -> None:
    script = [[Detection(_box_at(10.0, 6.0), 0.9)] for _ in range(5)]
    engine = RealPerceptionEngine(ScriptedDetector(script, fail_at=2), GEO, "cam_a", "s1")
    results = _run(engine, 5)
    assert results[2].status == Status.ERROR and "synthetic detector failure" in results[2].error
    assert results[2].tracks == () and results[2].submersions == ()
    assert results[4].status == Status.OK


@pytest.mark.parametrize(
    "frame",
    [np.zeros((1080, 1920), np.uint8), np.zeros((1080, 1920, 3), np.float32), "not an array"],
)
def test_invalid_frame_returns_error(frame: object) -> None:
    engine = RealPerceptionEngine(ScriptedDetector([]), GEO, "cam_a", "s1")
    result = engine.process(frame, 0, 1.0, T0)  # type: ignore[arg-type]
    assert result.status == Status.ERROR


def test_frame_size_change_is_an_error() -> None:
    engine = RealPerceptionEngine(ScriptedDetector([]), GEO, "cam_a", "s1")
    assert engine.process(FRAME, 0, 1.0, T0).status == Status.OK
    small = np.zeros((480, 640, 3), np.uint8)
    assert engine.process(small, 1, 1.1, T0).status == Status.ERROR


def test_invalid_ids_rejected() -> None:
    with pytest.raises(ValueError):
        RealPerceptionEngine(ScriptedDetector([]), GEO, "cam:a", "s1")


def test_reset_restarts_track_ids() -> None:
    script = [[Detection(_box_at(10.0, 6.0), 0.9)] for _ in range(10)]
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1")
    _run(engine, 5)
    engine.reset()
    assert engine.heartbeat().frames_processed == 0
    results = _run(engine, 5)
    assert results[-1].tracks[0].track_id == 1


# ------------------------------------------------------------------ config + factory


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ai.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_yolo_config_loads_and_resolves_weights(tmp_path: Path) -> None:
    cfg = load_ai_config(_write(tmp_path, "engine: yolo\nyolo:\n  weights: models/person.pt\n"))
    assert cfg.yolo is not None
    assert cfg.yolo.weights == (tmp_path / "models" / "person.pt").resolve()
    assert cfg.tracker.min_hits == 2


def test_yolo_default_class_ids_is_coco_person(tmp_path: Path) -> None:
    cfg = load_ai_config(_write(tmp_path, "engine: yolo\nyolo:\n  weights: person.pt\n"))
    assert cfg.yolo is not None and cfg.yolo.class_ids == (0,)


def test_yolo_person_class_ids_replace_single_id(tmp_path: Path) -> None:
    text = "engine: yolo\nyolo:\n  weights: z.pt\n  person_class_id: 5\n  person_class_ids: [0, 1]\n"
    cfg = load_ai_config(_write(tmp_path, text))
    assert cfg.yolo is not None and cfg.yolo.class_ids == (0, 1)


@pytest.mark.parametrize("ids", ["[]", "[0, 0]", "[-1]"])
def test_yolo_person_class_ids_rejects_invalid(tmp_path: Path, ids: str) -> None:
    with pytest.raises(ConfigError):
        load_ai_config(_write(tmp_path, f"engine: yolo\nyolo:\n  weights: z.pt\n  person_class_ids: {ids}\n"))


def test_shipped_z5cc_config_loads() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "ai_z5cc.yaml"
    cfg = load_ai_config(path)
    assert cfg.yolo is not None
    assert cfg.yolo.class_ids == (0, 1) and cfg.yolo.imgsz == 448
    assert cfg.yolo.weights.name == "z5cc_s448_final.pt"


def test_yolo_engine_requires_block(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_ai_config(_write(tmp_path, "engine: yolo\n"))


def test_factory_yolo_requires_geometry(tmp_path: Path) -> None:
    cfg = load_ai_config(_write(tmp_path, "engine: yolo\nyolo:\n  weights: person.pt\n"))
    with pytest.raises(ValueError):
        create_engine(cfg, "cam_a", "s1")


def test_factory_yolo_missing_weights_is_model_load_error(tmp_path: Path) -> None:
    """Missing weights (or a missing ultralytics install) is a ModelLoadError, never a download."""
    cfg = load_ai_config(_write(tmp_path, "engine: yolo\nyolo:\n  weights: absent.pt\n"))
    with pytest.raises(ModelLoadError):
        create_engine(cfg, "cam_a", "s1", geometry=GEO)


def test_distress_feature_is_the_two_second_mean_of_matched_detections() -> None:
    script = [[Detection(_box_at(10.0, 6.0), 0.9, distress=0.0 if i < 10 else 0.8)] for i in range(40)]
    engine = RealPerceptionEngine(ScriptedDetector(script), GEO, "cam_a", "s1")
    results = _run(engine, 40)
    early = results[9].tracks[0].features[DISTRESS_FEATURE]
    late = results[-1].tracks[0].features[DISTRESS_FEATURE]
    assert early == 0.0
    assert late == pytest.approx(0.8)          # the last 2 s are all distress detections


def test_distress_class_ids_must_be_person_classes(tmp_path: Path) -> None:
    text = "engine: yolo\nyolo:\n  weights: z.pt\n  person_class_ids: [0, 1]\n  distress_class_ids: [2]\n"
    with pytest.raises(ConfigError):
        load_ai_config(_write(tmp_path, text))
