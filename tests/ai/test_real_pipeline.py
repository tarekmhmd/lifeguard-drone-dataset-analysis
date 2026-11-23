"""End-to-end logic test: synthetic boxes -> RealPerceptionEngine (tracker, submersion monitor, features)
-> RuleBasedRiskEngine.

The detector is a test double that replays the *visible* boxes of the synthetic scenarios, so this checks
the tracking/submersion/risk logic chain only. It says nothing about detection accuracy on real video.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.ai.sim.scenarios import make_scenario
from lifeguard.contracts.enums import RiskLevel, RiskReason, Status, TrackState
from lifeguard.contracts.types import FrameResult, RiskEvent
from lifeguard.system.config import load_system_config
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.risk.engine import RuleBasedRiskEngine

ROOT = Path(__file__).resolve().parents[2]
RISK_CONFIG = load_system_config(ROOT / "configs" / "system.yaml").risk
FRAME = np.zeros((1080, 1920, 3), np.uint8)


class _ScenarioDetector(Detector):
    """Replays the visible synthetic boxes frame by frame (test double, not a model)."""

    def __init__(self, frames: tuple[FrameResult, ...]) -> None:
        self._frames = frames
        self._i = 0

    @property
    def model_version(self) -> str:
        return "synthetic_boxes"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        frame = self._frames[self._i]
        self._i += 1
        return [
            Detection(track.box_xyxy, track.det_conf, track.cls)
            for track in frame.tracks
            if track.state == TrackState.TRACKED
        ]


def _run(kind: str) -> tuple[list[FrameResult], list[RiskEvent], object]:
    scenario = make_scenario(kind, seed=0)
    engine = RealPerceptionEngine(_ScenarioDetector(scenario.frames), StubPoolGeometry(), "cam_sim", "real_s0")
    risk = RuleBasedRiskEngine(RISK_CONFIG)
    results: list[FrameResult] = []
    events: list[RiskEvent] = []
    for f in scenario.frames:
        result = engine.process(FRAME, f.frame_idx, f.t_capture, f.t_utc, f.t_media)
        results.append(result)
        events += risk.update(result)
    return results, events, scenario


@pytest.mark.parametrize("kind", ["normal_swim", "idle", "dive", "edge_exit", "submersion", "distress_like"])
def test_every_frame_is_valid_and_ok(kind: str) -> None:
    results, _, _ = _run(kind)
    assert all(r.status == Status.OK for r in results)
    for r in results:
        assert FrameResult.model_validate_json(r.model_dump_json()) == r


def test_submersion_reaches_alarm_at_the_disappearance_point() -> None:
    _, events, scenario = _run("submersion")
    (truth,) = scenario.events
    alarms = [e for e in events if e.to_level == RiskLevel.ALARM]
    assert alarms, "no ALARM in the submersion scenario"
    first = alarms[0]
    assert first.reason == RiskReason.SUBMERSION_TIMEOUT
    assert truth.t_onset + RISK_CONFIG.t_sub_s <= first.t_media <= truth.t_onset + RISK_CONFIG.t_sub_s + 0.5
    assert first.pool_xy is not None
    assert math.dist(first.pool_xy, (truth.pool_x, truth.pool_y)) < 1.0


def test_distress_like_reaches_alarm_inside_the_event() -> None:
    _, events, scenario = _run("distress_like")
    (truth,) = scenario.events
    alarms = [e for e in events if e.to_level == RiskLevel.ALARM]
    assert alarms
    assert truth.t_onset <= alarms[0].t_media <= truth.t_end


@pytest.mark.parametrize("kind", ["normal_swim", "idle", "dive", "edge_exit"])
def test_hard_negatives_never_alarm(kind: str) -> None:
    _, events, _ = _run(kind)
    assert [e for e in events if e.to_level in (RiskLevel.ALARM, RiskLevel.DISPATCH)] == []


def test_dive_reappearance_is_merged_and_timer_cleared() -> None:
    results, _, _ = _run("dive")
    merged = [t for r in results for t in r.tracks if t.merged_from]
    assert merged, "the resurfaced diver was not re-associated"
    assert results[-1].submersions == ()
