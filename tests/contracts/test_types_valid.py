"""TP-20..TP-23, TP-37, TP-38: valid contract models and helpers."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lifeguard.contracts.enums import Status, TrackState
from lifeguard.contracts.types import FrameResult, make_track_uid, parse_track_uid
from tests.contracts.conftest import (
    make_behavior,
    make_frame,
    make_keypoints,
    make_submersion,
    make_track,
)


def test_minimal_frame_result() -> None:
    """Required fields only, no tracks: status OK and schema_version 1.0.0."""
    frame = FrameResult(**make_frame(tracks=()))
    assert frame.status == Status.OK
    assert frame.schema_version == "1.0.0"


def test_full_frame_result() -> None:
    """Two tracks (one lost, one with keypoints and behaviour) plus one submersion."""
    lost = make_track(
        track_id=2,
        track_uid="cam_north:s1:2",
        state=TrackState.LOST,
        det_conf=0.4,
    )
    with_keypoints = make_track(keypoints=make_keypoints(), behavior=make_behavior())
    frame = FrameResult(
        **make_frame(tracks=(lost, with_keypoints), submersions=(make_submersion(),))
    )

    assert len(frame.tracks) == 2
    assert frame.tracks[0].state == TrackState.LOST
    assert frame.tracks[1].behavior is not None
    assert len(frame.tracks[1].keypoints) == 17
    assert len(frame.submersions) == 1


def test_unknown_field_ignored() -> None:
    """Unknown fields at top level and inside a track are ignored."""
    track = make_track(future_field=1)
    frame = FrameResult(**make_frame(tracks=(track,), future_field=1))

    assert not hasattr(frame, "future_field")
    assert not hasattr(frame.tracks[0], "future_field")


def test_models_are_frozen() -> None:
    """Contract models are immutable."""
    frame = FrameResult(**make_frame())
    with pytest.raises(ValidationError):
        frame.frame_idx = 1


def test_track_uid_helpers() -> None:
    """make_track_uid and parse_track_uid build and split identifiers."""
    uid = make_track_uid("cam_north", "s1", 7)
    assert uid == "cam_north:s1:7"
    assert parse_track_uid(uid) == ("cam_north", "s1", 7)

    for bad in ("a:b", "a:b:0", "a:b:x"):
        with pytest.raises(ValueError, match="track_uid"):
            parse_track_uid(bad)


def test_status_error_shape() -> None:
    """status=error requires a non-empty error and no tracks; a track makes it invalid."""
    frame = FrameResult(**make_frame(status=Status.ERROR, error="boom", tracks=()))
    assert frame.status == Status.ERROR

    with pytest.raises(ValidationError):
        FrameResult(**make_frame(status=Status.ERROR, error="boom"))
