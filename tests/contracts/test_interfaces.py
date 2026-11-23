"""TP-14: ABCs cannot be instantiated and their signatures match SPEC 4.7."""
from __future__ import annotations

import inspect

import pytest

from lifeguard.contracts.interfaces import PerceptionEngine, PoolGeometry, RiskEngine


def test_abcs_cannot_instantiate() -> None:
    """Each interface ABC raises TypeError on instantiation."""
    for abc_cls in (PoolGeometry, PerceptionEngine, RiskEngine):
        with pytest.raises(TypeError):
            abc_cls()


def _params(member) -> list[str]:
    return list(inspect.signature(member).parameters)


def test_method_signatures() -> None:
    """Every abstract method exposes the SPEC 4.7 parameter names in order."""
    cases = {
        PoolGeometry.to_pool: ["self", "u", "v"],
        PoolGeometry.to_image: ["self", "x", "y"],
        PoolGeometry.contains: ["self", "x", "y"],
        PoolGeometry.zone: ["self", "x", "y"],
        PoolGeometry.is_interior: ["self", "x", "y", "margin_m"],
        PoolGeometry.near_exit: ["self", "x", "y", "margin_m"],
        PoolGeometry.to_local_ned: ["self", "x", "y", "altitude_m"],
        PoolGeometry.site_id.fget: ["self"],
        PoolGeometry.length_m.fget: ["self"],
        PoolGeometry.width_m.fget: ["self"],
        PerceptionEngine.process: ["self", "frame_bgr", "frame_idx", "t_capture", "t_utc", "t_media"],
        PerceptionEngine.reset: ["self"],
        PerceptionEngine.heartbeat: ["self"],
        PerceptionEngine.cam_id.fget: ["self"],
        PerceptionEngine.session_id.fget: ["self"],
        RiskEngine.update: ["self", "frame_result"],
        RiskEngine.acknowledge: ["self", "ack"],
        RiskEngine.reset: ["self"],
    }
    for member, expected in cases.items():
        assert _params(member) == expected

    assert inspect.signature(PoolGeometry.is_interior).parameters["margin_m"].default == 1.0
    assert inspect.signature(PoolGeometry.near_exit).parameters["margin_m"].default == 1.0
    assert inspect.signature(PerceptionEngine.process).parameters["t_media"].default is None
