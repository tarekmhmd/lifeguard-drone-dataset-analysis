"""Homography calibration: estimation, validation, persistence, CLI and use by the real engine.

The "camera" is a synthetic perspective view (a trapezoid), so the true mapping is known exactly.
"""
from __future__ import annotations

import importlib.metadata
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from lifeguard.ai.detection.base import Detection, Detector
from lifeguard.ai.perception.real_engine import RealPerceptionEngine
from lifeguard.contracts.enums import Zone
from lifeguard.contracts.errors import ConfigError
from lifeguard.system.geometry.calibrated import (
    CalibratedPoolGeometry,
    CheckPoint,
    PoolCalibration,
    apply_homography,
    homography_from_points,
    load_calibration,
    reprojection_error_m,
    save_calibration,
)

ROOT = Path(__file__).resolve().parents[2]
L, W = 25.0, 12.5
CORNERS_PX = [(300.0, 200.0), (1620.0, 200.0), (1900.0, 1000.0), (20.0, 1000.0)]
CORNERS_M = [(0.0, 0.0), (L, 0.0), (L, W), (0.0, W)]


def _calibration(**overrides: object) -> PoolCalibration:
    data: dict[str, object] = {
        "site_id": "test_pool", "image_width": 1920, "image_height": 1080, "length_m": L, "width_m": W,
        "image_points": CORNERS_PX, "exits_m": [(0.0, 2.0)],
    }
    data.update(overrides)
    return PoolCalibration(**data)


def test_homography_is_exact_on_four_corners() -> None:
    h = homography_from_points(CORNERS_PX, CORNERS_M)
    for px, m in zip(CORNERS_PX, CORNERS_M):
        assert apply_homography(h, px) == pytest.approx(m, abs=1e-6)


def test_least_squares_with_noisy_extra_points() -> None:
    h_true = homography_from_points(CORNERS_PX, CORNERS_M)
    h_inv = np.linalg.inv(h_true)
    rng = np.random.default_rng(0)
    pool = [*CORNERS_M, (12.5, 6.25), (5.0, 10.0), (20.0, 3.0)]
    image = [tuple(np.add(apply_homography(h_inv, p), rng.normal(0, 0.5, 2))) for p in pool]
    h = homography_from_points(image, pool)
    error = reprojection_error_m(h, list(zip(image, pool)))
    assert error is not None and error < 0.1


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        pytest.param(CORNERS_PX[:3], CORNERS_M[:3], id="too-few"),
        pytest.param(CORNERS_PX, CORNERS_M[:3], id="mismatched"),
        pytest.param([(0, 0), (1, 1), (2, 2), (3, 3)], CORNERS_M, id="collinear"),
    ],
)
def test_degenerate_inputs_rejected(src: list, dst: list) -> None:
    with pytest.raises(ValueError):
        homography_from_points(src, dst)


def test_calibrated_geometry_round_trip_and_zones() -> None:
    geo = CalibratedPoolGeometry(_calibration())
    for x, y in [(1.0, 1.0), (12.5, 6.25), (24.0, 11.0)]:
        u, v = geo.to_image(x, y)
        assert geo.to_pool(u, v) == pytest.approx((x, y), abs=1e-6)
    assert geo.to_pool(5.0, 5.0) is None          # image corner: outside the pool
    assert geo.to_image(-200.0, 6.0) is None      # far outside the image
    assert geo.zone(20.0, 6.0) == Zone.DEEP and geo.zone(5.0, 6.0) == Zone.SHALLOW
    assert geo.zone(0.5, 6.0) == Zone.EDGE
    assert geo.near_exit(0.3, 2.0) and not geo.near_exit(25.0, 10.5)  # only configured exits
    assert geo.to_local_ned(3.0, 4.0, 5.0) == pytest.approx((3.0, 4.0, -5.0))


def test_no_configured_exits_means_none() -> None:
    geo = CalibratedPoolGeometry(_calibration(exits_m=[]))
    assert not geo.near_exit(0.0, 2.0)


def test_check_points_gate_the_calibration() -> None:
    h_inv = np.linalg.inv(homography_from_points(CORNERS_PX, CORNERS_M))
    centre_px = apply_homography(h_inv, (12.5, 6.25))
    _calibration(check_points=[CheckPoint(image=centre_px, pool=(12.5, 6.25))])  # accepted
    with pytest.raises(ValidationError, match="reprojection error"):
        _calibration(check_points=[CheckPoint(image=centre_px, pool=(13.5, 6.25))])  # 1 m off


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"image_points": [(-5.0, 200.0), *CORNERS_PX[1:]]}, id="outside-image"),
        pytest.param({"image_points": CORNERS_PX[:3]}, id="three-points"),
        pytest.param({"format_version": 99}, id="version"),
    ],
)
def test_invalid_calibrations_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _calibration(**overrides)


def test_save_load_and_load_errors(tmp_path: Path) -> None:
    path = tmp_path / "pool.json"
    save_calibration(path, _calibration())
    assert load_calibration(path) == _calibration()
    assert CalibratedPoolGeometry.from_file(path).site_id == "test_pool"
    with pytest.raises(ConfigError):
        load_calibration(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_calibration(bad)
    bad.write_text('{"site_id": "x"}', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_calibration(bad)


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "lifeguard.system.geometry.calibrate", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)


def test_cli_writes_a_valid_calibration(tmp_path: Path) -> None:
    out = tmp_path / "pool_test.json"
    corners = ";".join(f"{u},{v}" for u, v in CORNERS_PX)
    proc = _cli("--site-id", "test_pool", "--length", "25", "--width", "12.5", "--size", "1920x1080",
                "--corners", corners, "--exit", "0,2", "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("calibration=")
    assert load_calibration(out).exits_m == [(0.0, 2.0)]


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    base = ["--site-id", "p", "--length", "25", "--width", "12.5", "--size", "1920x1080",
            "--out", str(tmp_path / "p.json")]
    assert _cli(*base, "--corners", "1,2;3,4").returncode == 2                       # too few corners
    assert _cli(*base, "--corners", "0,0;1,1;2,2;3,3").returncode == 2               # collinear
    assert _cli(*base).returncode == 2                                                # no corners
    assert not (tmp_path / "p.json").exists()


def _headless_opencv() -> bool:
    names = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    return "opencv-python-headless" in names and "opencv-python" not in names


@pytest.mark.skipif(not _headless_opencv(), reason="a GUI OpenCV would open a window and wait for clicks")
def test_cli_interactive_without_gui_fails_cleanly(tmp_path: Path) -> None:
    import cv2

    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((1080, 1920, 3), np.uint8))
    proc = _cli("--site-id", "p", "--length", "25", "--width", "12.5", "--image", str(image),
                "--interactive", "--out", str(tmp_path / "p.json"))
    assert proc.returncode == 2
    assert "--corners" in proc.stderr


class _OneBox(Detector):
    def __init__(self, box: tuple[float, float, float, float]) -> None:
        self._box = box

    @property
    def model_version(self) -> str:
        return "scripted"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return [Detection(self._box, 0.9)]


def test_real_engine_uses_calibrated_coordinates() -> None:
    geo = CalibratedPoolGeometry(_calibration())
    u, v = geo.to_image(10.0, 6.0)
    engine = RealPerceptionEngine(_OneBox((u - 15, v - 20, u + 15, v + 20)), geo, "cam_a", "s1")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    result = None
    for i in range(3):
        result = engine.process(frame, i, float(i), "2026-01-01T00:00:00.000Z")
    (track,) = result.tracks
    assert track.in_water and math.dist(track.pool_xy, (10.0, 6.0)) < 0.01
