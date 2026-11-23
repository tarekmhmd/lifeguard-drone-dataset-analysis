"""TP-82..TP-93: MockPerceptionEngine and the engine factory."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.perception.factory import create_engine
from lifeguard.ai.perception.mock_engine import MockPerceptionEngine
from lifeguard.contracts.enums import Status
from lifeguard.contracts.errors import FixtureError, ModelLoadError
from lifeguard.contracts.interfaces import PerceptionEngine
from lifeguard.contracts.io import read_jsonl, write_jsonl
from lifeguard.contracts.types import EngineHeartbeat, FrameResult
from tests.contracts.conftest import make_frame

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"
FRAME = np.zeros((1080, 1920, 3), np.uint8)
T_UTC = "2026-02-02T10:00:00.000Z"


def _tracks_equal(a: tuple, b: tuple) -> bool:
    """Compare two track tuples ignoring the re-stamped track_uid."""
    return [t.model_dump(exclude={"track_uid"}) for t in a] == [
        t.model_dump(exclude={"track_uid"}) for t in b
    ]


def _make_bad_fixture(tmp_path: Path, kind: str) -> Path:
    if kind == "missing":
        return tmp_path / "missing.jsonl"
    if kind == "empty":
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8", newline="\n")
        return path
    if kind == "invalid-line":
        path = tmp_path / "bad.jsonl"
        good = FrameResult(**make_frame()).model_dump_json()
        path.write_text(f"{good}\n{{not json}}\n", encoding="utf-8", newline="\n")
        return path
    path = tmp_path / "mixed.jsonl"
    write_jsonl(
        path,
        [FrameResult(**make_frame()), FrameResult(**make_frame(frame_width=1280))],
    )
    return path


def test_factory_builds_mock() -> None:
    """create_engine builds a MockPerceptionEngine for the committed config."""
    cfg = load_ai_config(ROOT / "configs" / "ai.yaml")
    engine = create_engine(cfg, "cam_test", "s_test")
    assert isinstance(engine, MockPerceptionEngine)
    assert isinstance(engine, PerceptionEngine)


@pytest.mark.parametrize("kind", ["missing", "empty", "invalid-line", "mixed-sizes"])
def test_constructor_fixture_errors(tmp_path: Path, kind: str) -> None:
    """Missing, empty, invalid and mixed-size fixtures raise FixtureError."""
    path = _make_bad_fixture(tmp_path, kind)
    with pytest.raises(FixtureError) as excinfo:
        MockPerceptionEngine(path, "cam_test", "s_test")
    assert isinstance(excinfo.value, ModelLoadError)


def test_replay_restamps() -> None:
    """The first call re-stamps identifiers and metadata from the arguments."""
    engine = MockPerceptionEngine(FIXTURES / "submersion_seed0.jsonl", "cam_test", "s_test")
    result = engine.process(FRAME, 5, 100.0, T_UTC, 1.5)
    assert result.frame_idx == 5
    assert result.t_capture == 100.0
    assert result.t_media == 1.5
    assert result.cam_id == "cam_test"
    assert result.session_id == "s_test"
    assert all(t.track_uid.startswith("cam_test:s_test:") for t in result.tracks)


def test_replay_order_matches_fixture() -> None:
    """The first fixture_length calls replay the fixture lines in order."""
    fixture_frames = read_jsonl(FIXTURES / "normal_swim_seed0.jsonl", FrameResult)
    engine = MockPerceptionEngine(FIXTURES / "normal_swim_seed0.jsonl", "cam_test", "s_test")
    for i in range(50):
        result = engine.process(FRAME, i, 100.0 + i, T_UTC)
        assert _tracks_equal(result.tracks, fixture_frames[i].tracks)


def test_submersion_since_restamped() -> None:
    """Active submersion timers get since_t_capture = t_capture - seconds_submerged."""
    engine = MockPerceptionEngine(FIXTURES / "submersion_seed0.jsonl", "cam_test", "s_test")
    result = engine.process(FRAME, 0, 100.0, T_UTC)
    for _ in range(engine.fixture_length - 1):
        result = engine.process(FRAME, 0, 100.0, T_UTC)
        if result.submersions:
            break
    assert result.submersions
    sub = result.submersions[0]
    assert sub.since_t_capture == pytest.approx(result.t_capture - sub.seconds_submerged, abs=1e-6)


@pytest.mark.parametrize("mode", ["empty", "loop", "error"])
def test_on_exhausted(tmp_path: Path, mode: str) -> None:
    """Once exhausted, the configured on_exhausted behaviour applies."""
    path = tmp_path / "f.jsonl"
    fixture_frames = [FrameResult(**make_frame(frame_idx=i)) for i in range(3)]
    write_jsonl(path, fixture_frames)
    engine = MockPerceptionEngine(path, "cam_test", "s_test", on_exhausted=mode)

    for i in range(engine.fixture_length):
        engine.process(FRAME, i, 100.0 + i, T_UTC)

    assert engine.exhausted == (mode != "loop")
    result = engine.process(FRAME, 99, 200.0, T_UTC)

    if mode == "empty":
        assert result.status is Status.OK and result.tracks == ()
    elif mode == "loop":
        assert _tracks_equal(result.tracks, fixture_frames[0].tracks)
    else:
        assert result.status is Status.ERROR and result.error == "fixture exhausted"


def _raiser(*args: object, **kwargs: object) -> None:
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("frame", "frame_idx", "t_capture", "t_utc", "patch_target"),
    [
        pytest.param(None, 0, 100.0, T_UTC, None, id="none-frame"),
        pytest.param(np.zeros((1080, 1920), np.uint8), 0, 100.0, T_UTC, None, id="2d-frame"),
        pytest.param(np.zeros((1080, 1920, 3), np.float32), 0, 100.0, T_UTC, None, id="float32"),
        pytest.param(np.zeros((100, 100, 3), np.uint8), 0, 100.0, T_UTC, None, id="wrong-size"),
        pytest.param(FRAME, -1, 100.0, T_UTC, None, id="negative-frame-idx"),
        pytest.param(FRAME, 0, 100.0, "bad", None, id="bad-t-utc"),
        pytest.param(FRAME, 0, 100.0, T_UTC, "_restamp", id="internal-runtime-error"),
    ],
)
def test_process_never_raises(
    monkeypatch: pytest.MonkeyPatch,
    frame: object,
    frame_idx: int,
    t_capture: float,
    t_utc: str,
    patch_target: str | None,
) -> None:
    """Invalid inputs and internal errors return error results, never exceptions."""
    engine = MockPerceptionEngine(FIXTURES / "normal_swim_seed0.jsonl", "cam_test", "s_test")
    if patch_target is not None:
        monkeypatch.setattr(MockPerceptionEngine, patch_target, _raiser)
    result = engine.process(frame, frame_idx, t_capture, t_utc)
    assert result.status is Status.ERROR
    assert result.error


def test_reset() -> None:
    """reset() restarts replay from the first fixture line."""
    engine = MockPerceptionEngine(FIXTURES / "normal_swim_seed0.jsonl", "cam_test", "s_test")
    for i in range(10):
        engine.process(FRAME, i, 100.0 + i, T_UTC)
    engine.reset()
    result = engine.process(FRAME, 0, 100.0, T_UTC)

    fixture_frames = read_jsonl(FIXTURES / "normal_swim_seed0.jsonl", FrameResult)
    assert _tracks_equal(result.tracks, fixture_frames[0].tracks)
    assert engine.heartbeat().frames_processed == 1


def test_heartbeat() -> None:
    """heartbeat reports alive, the call count and the last frame index."""
    engine = MockPerceptionEngine(FIXTURES / "normal_swim_seed0.jsonl", "cam_test", "s_test")
    engine.process(FRAME, 7, 100.0, T_UTC)
    engine.process(FRAME, 8, 100.1, T_UTC)

    hb = engine.heartbeat()
    assert isinstance(hb, EngineHeartbeat)
    assert hb.alive is True
    assert hb.frames_processed == 2
    assert hb.last_frame_idx == 8


def test_fixture_status_passthrough(tmp_path: Path) -> None:
    """A degraded fixture line passes its status and error through."""
    path = tmp_path / "f.jsonl"
    write_jsonl(
        path,
        [
            FrameResult(**make_frame(frame_idx=0)),
            FrameResult(**make_frame(frame_idx=1, status=Status.DEGRADED, error="glare")),
            FrameResult(**make_frame(frame_idx=2)),
        ],
    )
    engine = MockPerceptionEngine(path, "cam_test", "s_test")
    engine.process(FRAME, 0, 100.0, T_UTC)
    result = engine.process(FRAME, 1, 100.1, T_UTC)
    assert result.status is Status.DEGRADED
    assert result.error == "glare"


def test_latency_measured() -> None:
    """latency_ms is never negative."""
    engine = MockPerceptionEngine(FIXTURES / "normal_swim_seed0.jsonl", "cam_test", "s_test")
    for i in range(5):
        result = engine.process(FRAME, i, 100.0 + i, T_UTC)
        assert result.latency_ms >= 0


def test_mock_has_no_geometry_dependency() -> None:
    """The mock engine takes no PoolGeometry (decision C-09)."""
    parameters = inspect.signature(MockPerceptionEngine.__init__).parameters
    assert "geometry" not in parameters
