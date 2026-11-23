"""TP-24..TP-29, TP-34..TP-36: invalid contract models raise ValidationError."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lifeguard.contracts.enums import RiskLevel, Status, Zone
from lifeguard.contracts.types import (
    Behavior,
    DispatchCommand,
    FrameResult,
    GroundTruthEvent,
    RiskEvent,
    TrackObs,
)
from tests.contracts.conftest import (
    make_behavior,
    make_dispatch,
    make_frame,
    make_ground_truth,
    make_keypoints,
    make_risk_event,
    make_submersion,
    make_track,
)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"status": Status.OK, "error": "boom"}, id="rule1-ok-with-error"),
        pytest.param(
            {"status": Status.ERROR, "error": "boom"},
            id="rule2-error-with-tracks",
        ),
        pytest.param(
            {"tracks": (make_track(), make_track())},
            id="rule4-duplicate-track-id",
        ),
        pytest.param(
            {"tracks": (make_track(track_uid="cam_north:s1:99"),)},
            id="rule5-uid-mismatch",
        ),
        pytest.param(
            {"tracks": (make_track(box_xyxy=(10.0, 10.0, 2000.0, 100.0)),)},
            id="rule6-box-too-wide",
        ),
        pytest.param(
            {"tracks": (make_track(keypoints=make_keypoints(x=2000.0)),)},
            id="rule7-keypoint-outside",
        ),
        pytest.param(
            {"submersions": (make_submersion(track_uid="other:s1:1"),)},
            id="rule8-submersion-prefix",
        ),
    ],
)
def test_frame_result_rules(overrides: dict[str, object]) -> None:
    """Each SPEC 4.6.4 rule violation raises ValidationError."""
    with pytest.raises(ValidationError):
        FrameResult(**make_frame(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"track_id": 0}, id="track-id-zero"),
        pytest.param(
            {"box_xyxy": (100.0, 100.0, 100.0, 300.0)},
            id="x1-not-less-than-x2",
        ),
        pytest.param(
            {"box_xyxy": (100.0, 300.0, 200.0, 300.0)},
            id="y1-not-less-than-y2",
        ),
        pytest.param(
            {"box_xyxy": (-1.0, 100.0, 200.0, 300.0)},
            id="negative-x1",
        ),
        pytest.param({"det_conf": 1.1}, id="det-conf-above-one"),
        pytest.param(
            {"in_water": True, "pool_xy": None, "zone": None},
            id="in-water-without-pool-xy",
        ),
        pytest.param(
            {"in_water": False, "pool_xy": None, "zone": Zone.DEEP},
            id="zone-without-pool-xy",
        ),
        pytest.param(
            {"keypoints": tuple((100.0, 200.0, 0.9) for _ in range(16))},
            id="sixteen-keypoints",
        ),
        pytest.param(
            {"keypoints": make_keypoints(conf=1.5)},
            id="keypoint-conf-above-one",
        ),
        pytest.param({"merged_from": (1,)}, id="merged-from-contains-self"),
        pytest.param({"features": {"Speed": 0.5}}, id="bad-feature-name"),
        pytest.param({"features": {"speed": float("nan")}}, id="nan-feature-value"),
    ],
)
def test_track_obs_rules(overrides: dict[str, object]) -> None:
    """Each TrackObs rule violation raises ValidationError."""
    with pytest.raises(ValidationError):
        TrackObs(**make_track(**overrides))


def test_behavior_sum() -> None:
    """Probabilities must sum to 1 within 1e-3."""
    with pytest.raises(ValidationError):
        Behavior(**make_behavior(normal_swim=0.6))
    with pytest.raises(ValidationError):
        Behavior(**make_behavior(normal_swim=0.71))

    valid = Behavior(**make_behavior(normal_swim=0.7005))
    assert valid.normal_swim == 0.7005


@pytest.mark.parametrize("cam_id", ["cam:1", "", "a" * 65, "cam 1"])
def test_identifiers(cam_id: str) -> None:
    """Invalid identifiers are rejected."""
    with pytest.raises(ValidationError):
        FrameResult(**make_frame(cam_id=cam_id, tracks=()))


def test_non_finite_rejected() -> None:
    """NaN and infinity are rejected everywhere."""
    with pytest.raises(ValidationError):
        FrameResult(**make_frame(t_capture=float("inf")))
    with pytest.raises(ValidationError):
        TrackObs(**make_track(pool_xy=(float("nan"), 1.0)))


@pytest.mark.parametrize(
    "t_utc",
    [
        "2026-01-01T00:00:00Z",
        "2026-01-01 00:00:00.000Z",
        "2026-01-01T00:00:00.000+00:00",
    ],
)
def test_t_utc_format(t_utc: str) -> None:
    """Malformed UTC timestamps are rejected."""
    with pytest.raises(ValidationError):
        FrameResult(**make_frame(t_utc=t_utc))


def test_risk_event_rules() -> None:
    """RiskEvent cross-field rules are enforced."""
    with pytest.raises(ValidationError):
        RiskEvent(**make_risk_event(to_level=RiskLevel.NORMAL))
    with pytest.raises(ValidationError):
        RiskEvent(**make_risk_event(risk=1.2))
    with pytest.raises(ValidationError):
        RiskEvent(**make_risk_event(track_uid="other:s1:1"))


@pytest.mark.parametrize("confirmed_by", ["", "   ", None])
def test_dispatch_requires_human(confirmed_by: str | None) -> None:
    """A dispatch always requires a non-blank confirmed_by."""
    data = make_dispatch()
    if confirmed_by is None:
        data.pop("confirmed_by")
    else:
        data["confirmed_by"] = confirmed_by

    with pytest.raises(ValidationError):
        DispatchCommand(**data)


def test_ground_truth_rules() -> None:
    """GroundTruthEvent ordering and pool-coordinate pairing are enforced."""
    with pytest.raises(ValidationError):
        GroundTruthEvent(**make_ground_truth(t_end=20.0))
    with pytest.raises(ValidationError):
        GroundTruthEvent(**make_ground_truth(pool_y=None))
