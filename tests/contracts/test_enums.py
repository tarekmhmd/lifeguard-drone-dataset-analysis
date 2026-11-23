"""TP-11/TP-12: enum values, ordering and JSON serialisation."""
from __future__ import annotations

import json

from lifeguard.contracts.enums import (
    AckDecision,
    DetClass,
    GroundTruthType,
    HealthComponent,
    RiskLevel,
    RiskReason,
    Status,
    TrackState,
    Zone,
)

EXPECTED_VALUES: dict[type, list[str]] = {
    Status: ["ok", "degraded", "error"],
    DetClass: ["person", "head"],
    TrackState: ["tracked", "lost"],
    Zone: ["shallow", "deep", "edge"],
    RiskLevel: ["NORMAL", "WATCH", "PRE_ALERT", "ALARM", "DISPATCH", "RESOLVED"],
    RiskReason: [
        "RISK_ABOVE_THRESHOLD",
        "RISK_BELOW_THRESHOLD",
        "SUBMERSION_TIMEOUT",
        "LIFEGUARD_CONFIRMED",
        "LIFEGUARD_DISMISSED",
        "ACK_TIMEOUT_ESCALATION",
        "INCIDENT_CLOSED",
        "COOLDOWN_ELAPSED",
        "TRACK_EXITED",
    ],
    AckDecision: ["confirm", "dismiss"],
    HealthComponent: [
        "ingest",
        "perception",
        "risk",
        "alerts",
        "dashboard",
        "mission",
        "storage",
        "system",
    ],
    GroundTruthType: ["submersion", "distress", "drowning"],
}


def test_enum_values_exact() -> None:
    """Every enum exposes exactly the SPEC values in declaration order."""
    for enum_cls, expected in EXPECTED_VALUES.items():
        assert [member.value for member in enum_cls] == expected


def test_enums_are_str() -> None:
    """A risk level serialises through json as its string value."""
    assert json.dumps(RiskLevel.ALARM) == '"ALARM"'
