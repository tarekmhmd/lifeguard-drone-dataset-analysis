"""Contract enums shared across lifeguard modules."""
from __future__ import annotations

import enum


class Status(str, enum.Enum):
    """Frame or component status."""

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class DetClass(str, enum.Enum):
    """Detection class."""

    PERSON = "person"
    HEAD = "head"


class TrackState(str, enum.Enum):
    """Tracker state for a person."""

    TRACKED = "tracked"
    LOST = "lost"


class Zone(str, enum.Enum):
    """Pool zone."""

    SHALLOW = "shallow"
    DEEP = "deep"
    EDGE = "edge"


class RiskLevel(str, enum.Enum):
    """Incident risk level."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    PRE_ALERT = "PRE_ALERT"
    ALARM = "ALARM"
    DISPATCH = "DISPATCH"
    RESOLVED = "RESOLVED"


class RiskReason(str, enum.Enum):
    """Reason for a risk-level transition."""

    RISK_ABOVE_THRESHOLD = "RISK_ABOVE_THRESHOLD"
    RISK_BELOW_THRESHOLD = "RISK_BELOW_THRESHOLD"
    SUBMERSION_TIMEOUT = "SUBMERSION_TIMEOUT"
    LIFEGUARD_CONFIRMED = "LIFEGUARD_CONFIRMED"
    LIFEGUARD_DISMISSED = "LIFEGUARD_DISMISSED"
    ACK_TIMEOUT_ESCALATION = "ACK_TIMEOUT_ESCALATION"
    INCIDENT_CLOSED = "INCIDENT_CLOSED"
    COOLDOWN_ELAPSED = "COOLDOWN_ELAPSED"
    TRACK_EXITED = "TRACK_EXITED"


class AckDecision(str, enum.Enum):
    """Lifeguard decision for an alert."""

    CONFIRM = "confirm"
    DISMISS = "dismiss"


class HealthComponent(str, enum.Enum):
    """System health component."""

    INGEST = "ingest"
    PERCEPTION = "perception"
    RISK = "risk"
    ALERTS = "alerts"
    DASHBOARD = "dashboard"
    MISSION = "mission"
    STORAGE = "storage"
    SYSTEM = "system"


class GroundTruthType(str, enum.Enum):
    """Ground-truth event type."""

    SUBMERSION = "submersion"
    DISTRESS = "distress"
    DROWNING = "drowning"
