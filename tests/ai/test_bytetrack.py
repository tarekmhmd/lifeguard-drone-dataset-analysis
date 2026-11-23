"""ByteTrack-style tracker: confirmation, ID stability, lost/re-association, low-confidence stage."""
from __future__ import annotations

import numpy as np
import pytest

from lifeguard.ai.detection.base import Detection
from lifeguard.ai.tracking.bytetrack import ByteTracker, TrackedBox, iou_matrix
from lifeguard.ai.tracking.config import TrackerConfig
from lifeguard.contracts.enums import TrackState

DT = 0.1


def _det(x: float, y: float, conf: float = 0.9, w: float = 40.0, h: float = 60.0) -> Detection:
    return Detection((x, y, x + w, y + h), conf)


def _t(i: int) -> float:
    return round(i * DT, 3)


def _ids(out: list[TrackedBox], state: TrackState = TrackState.TRACKED) -> list[int]:
    return [b.track_id for b in out if b.state == state]


def test_iou_matrix_values() -> None:
    ious = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (5, 0, 15, 10), (20, 20, 30, 30)])
    assert ious.shape == (1, 3)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(50 / 150)
    assert ious[0, 2] == 0.0
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)


def test_new_track_reported_after_min_hits() -> None:
    tracker = ByteTracker()
    assert tracker.update([_det(100, 100)], _t(0)) == []
    out = tracker.update([_det(101, 100)], _t(1))
    assert _ids(out) == [1]
    assert out[0].first_seen_t == 0.0


def test_moving_person_keeps_one_id() -> None:
    tracker = ByteTracker()
    ids: set[int] = set()
    for i in range(60):
        out = tracker.update([_det(100 + 5 * i, 200)], _t(i))
        ids.update(_ids(out))
    assert ids == {1}


def test_two_people_get_two_ids() -> None:
    tracker = ByteTracker()
    out: list[TrackedBox] = []
    for i in range(5):
        out = tracker.update([_det(100, 100), _det(600, 400)], _t(i))
    assert _ids(out) == [1, 2]


def test_lost_track_emitted_then_hidden() -> None:
    cfg = TrackerConfig()
    tracker = ByteTracker(cfg)
    tracker.update([_det(100, 100)], _t(0))
    tracker.update([_det(100, 100)], _t(1))  # confirmed, last seen 0.1
    boundary = round((0.1 + cfg.lost_emit_s) / DT)  # frame exactly at the limit: skip (float edge)
    for i in range(2, 40):
        out = tracker.update([], _t(i))
        assert _ids(out) == []
        if i < boundary:
            assert _ids(out, TrackState.LOST) == [1]
        elif i > boundary:
            assert _ids(out, TrackState.LOST) == []


def test_reappearance_within_buffer_keeps_id() -> None:
    tracker = ByteTracker()
    for i in range(3):
        tracker.update([_det(300, 300)], _t(i))
    for i in range(3, 33):  # 3 s under water
        tracker.update([], _t(i))
    out = tracker.update([_det(302, 301)], _t(33))
    assert _ids(out) == [1]


def test_reappearance_after_buffer_gets_new_id() -> None:
    cfg = TrackerConfig()
    tracker = ByteTracker(cfg)
    for i in range(3):
        tracker.update([_det(300, 300)], _t(i))
    gap = round((cfg.track_buffer_s + 1.0) / DT)
    for i in range(3, 3 + gap):
        tracker.update([], _t(i))
    tracker.update([_det(300, 300)], _t(3 + gap))
    out = tracker.update([_det(300, 300)], _t(4 + gap))
    assert _ids(out) == [2]


def test_low_confidence_detection_keeps_existing_track() -> None:
    tracker = ByteTracker()
    for i in range(3):
        tracker.update([_det(300, 300)], _t(i))
    out = tracker.update([_det(301, 300, conf=0.3)], _t(3))
    assert _ids(out) == [1]


def test_low_confidence_detections_never_start_tracks() -> None:
    tracker = ByteTracker()
    for i in range(20):
        assert tracker.update([_det(300, 300, conf=0.3)], _t(i)) == []


def test_lost_track_does_not_drift_into_another_swimmer() -> None:
    """A lost track keeps its last position (zero velocity) instead of being extrapolated."""
    tracker = ByteTracker()
    for i in range(10):
        tracker.update([_det(100 + 10 * i, 300)], _t(i))  # moving right at 10 px/frame
    for i in range(10, 30):
        out = tracker.update([], _t(i))
    lost = [b for b in out if b.state == TrackState.LOST]
    assert lost and lost[0].box_xyxy[0] == pytest.approx(190.0)


def test_reset_restarts_ids() -> None:
    tracker = ByteTracker()
    for i in range(3):
        tracker.update([_det(100, 100)], _t(i))
    tracker.reset()
    tracker.update([_det(500, 500)], _t(0))
    assert _ids(tracker.update([_det(500, 500)], _t(1))) == [1]


def test_invalid_boxes_are_ignored() -> None:
    tracker = ByteTracker()
    bad = Detection((10.0, 10.0, 10.0, 50.0), 0.9)
    nan = Detection((float(np.nan), 0.0, 10.0, 10.0), 0.9)
    for i in range(3):
        assert tracker.update([bad, nan], _t(i)) == []


def test_tracker_config_validation() -> None:
    with pytest.raises(ValueError):
        TrackerConfig(low_thresh=0.6, high_thresh=0.5)
    with pytest.raises(ValueError):
        TrackerConfig(lost_emit_s=10.0, track_buffer_s=5.0)
