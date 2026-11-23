"""TP-40/TP-41: JSON serialisation round-trips and enums serialise as values."""
from __future__ import annotations

from collections.abc import Callable

import pytest

from lifeguard.contracts.types import (
    Acknowledgement,
    Behavior,
    ContractModel,
    DispatchCommand,
    EngineHeartbeat,
    FrameResult,
    GroundTruthEvent,
    RiskEvent,
    Submersion,
    SystemHealth,
    TrackObs,
)
from tests.contracts.conftest import (
    make_ack,
    make_behavior,
    make_dispatch,
    make_frame,
    make_ground_truth,
    make_health,
    make_heartbeat,
    make_risk_event,
    make_submersion,
    make_track,
)

CASES: list[tuple[type[ContractModel], Callable[..., dict[str, object]]]] = [
    (Behavior, make_behavior),
    (TrackObs, make_track),
    (Submersion, make_submersion),
    (FrameResult, make_frame),
    (RiskEvent, make_risk_event),
    (Acknowledgement, make_ack),
    (DispatchCommand, make_dispatch),
    (SystemHealth, make_health),
    (EngineHeartbeat, make_heartbeat),
    (GroundTruthEvent, make_ground_truth),
]


@pytest.mark.parametrize(("model", "builder"), CASES)
def test_json_round_trip(
    model: type[ContractModel],
    builder: Callable[..., dict[str, object]],
) -> None:
    """Every model round-trips through model_dump_json / model_validate_json."""
    instance = model(**builder())
    assert model.model_validate_json(instance.model_dump_json()) == instance


def test_enum_serialised_as_value() -> None:
    """Enums appear as their string value, never as a Python repr."""
    json_str = FrameResult(**make_frame()).model_dump_json()
    assert '"status":"ok"' in json_str
    assert "Status" not in json_str
