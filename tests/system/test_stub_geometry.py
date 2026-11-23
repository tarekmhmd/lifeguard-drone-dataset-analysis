"""TP-50..TP-58: StubPoolGeometry behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from lifeguard.contracts.enums import Zone
from lifeguard.contracts.interfaces import PoolGeometry
from lifeguard.system.geometry.stub import StubPoolGeometry


def test_is_pool_geometry() -> None:
    """StubPoolGeometry implements the PoolGeometry interface."""
    assert isinstance(StubPoolGeometry(), PoolGeometry)


def test_corner_mapping() -> None:
    """Pool corners map to the expected image pixels."""
    geometry = StubPoolGeometry()
    assert geometry.to_image(0.0, 0.0) == pytest.approx((192.0, 108.0))
    assert geometry.to_image(25.0, 12.5) == pytest.approx((1728.0, 972.0))


def test_round_trip() -> None:
    """to_pool is the exact inverse of to_image for interior points."""
    geometry = StubPoolGeometry()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = float(rng.uniform(0.0, geometry.length_m))
        y = float(rng.uniform(0.0, geometry.width_m))
        u, v = geometry.to_image(x, y)
        assert geometry.to_pool(u, v) == pytest.approx((x, y), abs=1e-6)


def test_outside() -> None:
    """Points outside the calibrated area return None or False."""
    geometry = StubPoolGeometry()
    assert geometry.to_pool(10.0, 10.0) is None
    assert geometry.contains(-0.1, 1.0) is False
    assert geometry.zone(30.0, 5.0) is None


def test_zones() -> None:
    """Zones follow the edge band and the deep/split line."""
    geometry = StubPoolGeometry()
    assert geometry.zone(0.5, 6.0) is Zone.EDGE
    assert geometry.zone(5.0, 6.0) is Zone.SHALLOW
    assert geometry.zone(20.0, 6.0) is Zone.DEEP


def test_is_interior() -> None:
    """is_interior respects the margin from every edge."""
    geometry = StubPoolGeometry()
    assert geometry.is_interior(5.0, 6.0, 1.0) is True
    assert geometry.is_interior(0.5, 6.0, 1.0) is False


def test_near_exit() -> None:
    """near_exit is True only within the margin of an exit point."""
    geometry = StubPoolGeometry()
    assert geometry.near_exit(0.5, 2.0, 1.0) is True
    assert geometry.near_exit(12.0, 6.0, 1.0) is False


def test_to_local_ned() -> None:
    """NED conversion applies the configured yaw and offsets."""
    geometry = StubPoolGeometry()
    assert geometry.to_local_ned(3.0, 4.0, 5.0) == (3.0, 4.0, -5.0)

    rotated = StubPoolGeometry(ned_yaw_deg=90.0)
    assert rotated.to_local_ned(3.0, 4.0, 5.0) == pytest.approx((-4.0, 3.0, -5.0))


def test_invalid_construction_and_inputs() -> None:
    """Invalid construction and non-finite inputs raise ValueError."""
    with pytest.raises(ValueError):
        StubPoolGeometry(length_m=0.0)
    with pytest.raises(ValueError):
        StubPoolGeometry(margin_frac=0.5)

    geometry = StubPoolGeometry()
    with pytest.raises(ValueError):
        geometry.contains(float("nan"), 1.0)
