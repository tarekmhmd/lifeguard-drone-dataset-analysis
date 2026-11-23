"""Per-track motion and visibility features."""
from __future__ import annotations

import math

import pytest

from lifeguard.ai.features.extractor import FeatureExtractor

DT = 0.1
BOX = (100.0, 100.0, 140.0, 160.0)


def test_straight_swim_has_speed_and_full_progress() -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(21):
        t = round(i * DT, 3)
        feats = fx.observe(1, t, True, (2.0 + 1.0 * t, 5.0), True, BOX)
    assert feats["speed_mps"] == pytest.approx(1.0, rel=1e-6)
    assert feats["path_speed_mps"] == pytest.approx(1.0, rel=1e-6)
    assert feats["progress_ratio"] == pytest.approx(1.0)
    assert feats["visibility_ratio"] == 1.0
    assert feats["time_since_seen_s"] == 0.0
    assert feats["box_aspect"] == pytest.approx(1.5)


def test_motionless_person_has_zero_speed_and_no_progress_ratio() -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(21):
        feats = fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, BOX)
    assert feats["speed_mps"] == 0.0
    assert "progress_ratio" not in feats


def test_invisible_samples_lower_visibility_and_count_time_since_seen() -> None:
    fx = FeatureExtractor()
    for i in range(10):
        fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, BOX)
    feats: dict[str, float] = {}
    for i in range(10, 20):
        feats = fx.observe(1, round(i * DT, 3), False, (5.0, 5.0), True, BOX)
    assert feats["visibility_ratio"] == pytest.approx(0.5)
    assert feats["time_since_seen_s"] == pytest.approx(1.0)


def test_bobbing_box_has_positive_bob_feature() -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(30):
        dy = 15.0 * math.sin(2 * math.pi * 0.8 * i * DT)
        feats = fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, (100.0, 100.0 + dy, 140.0, 160.0 + dy))
    assert feats["bob_std_norm"] > 0.1


def test_out_of_water_has_no_motion_features() -> None:
    fx = FeatureExtractor()
    feats: dict[str, float] = {}
    for i in range(21):
        feats = fx.observe(1, round(i * DT, 3), True, None, False, BOX)
    assert "speed_mps" not in feats


def test_window_drops_old_samples() -> None:
    fx = FeatureExtractor()
    for i in range(10):
        fx.observe(1, round(i * DT, 3), False, (5.0, 5.0), True, BOX)
    feats: dict[str, float] = {}
    for i in range(10, 60):  # 5 s visible; window is 4 s
        feats = fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, BOX)
    assert feats["visibility_ratio"] == 1.0


def test_inherit_moves_history_to_new_id() -> None:
    fx = FeatureExtractor()
    for i in range(10):
        fx.observe(1, round(i * DT, 3), True, (5.0, 5.0), True, BOX)
    fx.inherit(4, (1,))
    assert fx.known_ids == {4}
    feats = fx.observe(4, 1.0, False, (5.0, 5.0), True, BOX)
    assert feats["visibility_ratio"] == pytest.approx(10 / 11)
