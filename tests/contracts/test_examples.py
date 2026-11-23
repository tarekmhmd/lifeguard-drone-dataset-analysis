"""TP-30..TP-33: hand-written contract example fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lifeguard.contracts.io import read_events_csv
from lifeguard.contracts.types import (
    ContractModel,
    DispatchCommand,
    FrameResult,
    RiskEvent,
    SystemHealth,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "contract_examples"

VALID_EXAMPLES: list[tuple[str, type[ContractModel]]] = [
    ("frame_result_valid.json", FrameResult),
    ("risk_event_valid.json", RiskEvent),
    ("dispatch_command_valid.json", DispatchCommand),
    ("system_health_valid.json", SystemHealth),
]


@pytest.mark.parametrize(("filename", "model"), VALID_EXAMPLES)
def test_valid_examples_parse(filename: str, model: type[ContractModel]) -> None:
    """Every *_valid.json parses with its model."""
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    instance = model.model_validate(data)
    assert isinstance(instance, model)


@pytest.mark.parametrize(
    "filename",
    ["frame_result_invalid_uid.json", "frame_result_invalid_box.json"],
)
def test_invalid_examples_fail(filename: str) -> None:
    """The invalid examples raise ValidationError."""
    data = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        FrameResult.model_validate(data)


def test_major_version_rejected() -> None:
    """A different major schema_version is rejected and named in the message."""
    data = json.loads((FIXTURES / "frame_result_major_v2.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError) as excinfo:
        FrameResult.model_validate(data)
    assert "2.0.0" in str(excinfo.value)


def test_events_csv_example() -> None:
    """The example events CSV parses to at least one GroundTruthEvent."""
    events = read_events_csv(FIXTURES / "ground_truth_events_valid.csv")
    assert len(events) >= 1
