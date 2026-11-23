"""TP-60..TP-68: synthetic scenario generator behaviour."""
from __future__ import annotations

import itertools

import pytest

from lifeguard.ai.sim.scenarios import SCENARIO_KINDS, make_scenario
from lifeguard.contracts.enums import GroundTruthType, Status, TrackState
from lifeguard.contracts.types import make_track_uid, parse_track_uid

DURATIONS = {
    "normal_swim": 60.0,
    "idle": 60.0,
    "dive": 40.0,
    "edge_exit": 40.0,
    "submersion": 60.0,
    "distress_like": 90.0,
}


def test_deterministic() -> None:
    """The same seed reproduces the same scenario, and the session id encodes it."""
    for kind in SCENARIO_KINDS:
        first = make_scenario(kind, seed=7)
        second = make_scenario(kind, seed=7)
        assert first == second
        assert first.session_id == f"{kind}_seed7"


@pytest.mark.parametrize("kind", SCENARIO_KINDS)
def test_structure(kind: str) -> None:
    """Every frame is structurally correct with two background swimmers."""
    scenario = make_scenario(kind)
    assert len(scenario.frames) == round(DURATIONS[kind] * 10.0)

    for i, frame in enumerate(scenario.frames):
        assert frame.cam_id == "cam_sim"
        assert frame.session_id == f"{kind}_seed0"
        assert frame.frame_idx == i
        assert frame.status is Status.OK
        assert frame.frame_width == 1920 and frame.frame_height == 1080
        assert frame.t_media == pytest.approx(i / 10.0)
        assert frame.t_capture == pytest.approx(1000.0 + i / 10.0)
        background = [t for t in frame.tracks if t.track_id in (2, 3)]
        assert len(background) == 2
        assert {t.track_id for t in background} == {2, 3}


@pytest.mark.parametrize("kind", SCENARIO_KINDS)
def test_background_never_events(kind: str) -> None:
    """No ground-truth event ever refers to a background track."""
    scenario = make_scenario(kind)
    for event in scenario.events:
        assert parse_track_uid(event.track_uid)[2] == 1


def test_distress_event() -> None:
    """distress_like produces exactly one distress event with the right window."""
    scenario = make_scenario("distress_like")
    assert len(scenario.events) == 1

    event = scenario.events[0]
    assert event.type is GroundTruthType.DISTRESS
    assert event.t_onset == 20.0
    assert event.t_end == 90.0
    assert event.track_uid == make_track_uid("cam_sim", "distress_like_seed0", 1)
    assert event.pool_x is not None and event.pool_y is not None


def test_submersion_scenario() -> None:
    """submersion emits one growing timer from t=20 on and the matching event."""
    scenario = make_scenario("submersion")
    frames = scenario.frames

    for frame in frames:
        if frame.t_media < 20.0:
            assert frame.submersions == ()
        else:
            assert len(frame.submersions) == 1
            sub = frame.submersions[0]
            assert sub.track_uid == make_track_uid("cam_sim", "submersion_seed0", 1)
            assert sub.seconds_submerged == pytest.approx(frame.t_media - 19.9, abs=1e-3)

    seconds = [f.submersions[0].seconds_submerged for f in frames if f.t_media >= 20.0]
    assert all(b > a for a, b in itertools.pairwise(seconds))
    assert seconds[-1] == pytest.approx(40.0, abs=1e-3)

    for frame in frames:
        if 20.0 <= frame.t_media < 22.0:
            subject = [t for t in frame.tracks if t.track_id == 1]
            assert len(subject) == 1 and subject[0].state is TrackState.LOST
        elif frame.t_media >= 22.0:
            assert all(t.track_id != 1 for t in frame.tracks)

    assert len(scenario.events) == 1
    event = scenario.events[0]
    assert event.type is GroundTruthType.SUBMERSION
    assert event.t_onset == 20.0 and event.t_end is None
    assert event.track_uid == make_track_uid("cam_sim", "submersion_seed0", 1)

    first_sub = frames[200].submersions[0]
    assert event.pool_x == first_sub.last_pool_xy[0]
    assert event.pool_y == first_sub.last_pool_xy[1]


def test_submersion_lost_window() -> None:
    """The submersion subject is lost exactly for t in [20, 22)."""
    scenario = make_scenario("submersion")
    lost = [
        frame.t_media
        for frame in scenario.frames
        if any(t.track_id == 1 and t.state is TrackState.LOST for t in frame.tracks)
    ]
    assert lost == [round(i / 10.0, 3) for i in range(200, 220)]


def test_dive_reassociation() -> None:
    """dive reappears as a new track with merged_from=(1,) and no events."""
    scenario = make_scenario("dive")
    assert scenario.events == ()

    merged = {
        t.track_id for f in scenario.frames for t in f.tracks if t.merged_from == (1,)
    }
    assert merged == {4}

    submersion_tids = {
        parse_track_uid(s.track_uid)[2] for f in scenario.frames for s in f.submersions
    }
    assert submersion_tids == {1}


def test_edge_exit() -> None:
    """edge_exit ends the track at t=20 with no lost frames or submersions."""
    scenario = make_scenario("edge_exit")
    assert scenario.events == ()

    for frame in scenario.frames:
        assert frame.submersions == ()
        if frame.t_media > 20.0:
            assert all(t.track_id != 1 for t in frame.tracks)
        else:
            assert any(t.track_id == 1 for t in frame.tracks)


@pytest.mark.parametrize("kind", SCENARIO_KINDS)
def test_no_background_submersion(kind: str) -> None:
    """No Submersion entry ever refers to a background track."""
    scenario = make_scenario(kind)
    for frame in scenario.frames:
        for sub in frame.submersions:
            assert parse_track_uid(sub.track_uid)[2] == 1
