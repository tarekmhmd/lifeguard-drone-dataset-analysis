"""TP-15: the public contract API re-exports exactly the SPEC 4.10 names."""
from __future__ import annotations

from lifeguard import contracts

EXPECTED = {
    "SCHEMA_VERSION",
    "check_schema_version",
    "LifeguardError",
    "ContractVersionError",
    "ContractValidationError",
    "ConfigError",
    "ModelLoadError",
    "FixtureError",
    "Status",
    "DetClass",
    "TrackState",
    "Zone",
    "RiskLevel",
    "RiskReason",
    "AckDecision",
    "HealthComponent",
    "GroundTruthType",
    "format_utc",
    "utc_now",
    "add_seconds_utc",
    "ContractModel",
    "Behavior",
    "TrackObs",
    "Submersion",
    "FrameResult",
    "RiskEvent",
    "Acknowledgement",
    "DispatchCommand",
    "SystemHealth",
    "EngineHeartbeat",
    "GroundTruthEvent",
    "make_track_uid",
    "parse_track_uid",
    "PoolGeometry",
    "PerceptionEngine",
    "RiskEngine",
    "write_jsonl",
    "read_jsonl",
    "write_events_csv",
    "read_events_csv",
}


def test_public_api_exact() -> None:
    """The public API exports exactly the SPEC 4.10 name set."""
    assert set(contracts.__all__) == EXPECTED
