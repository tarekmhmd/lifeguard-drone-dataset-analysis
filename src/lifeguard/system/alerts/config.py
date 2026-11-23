"""Alert service configuration (Blueprint M9). Starting values; agree them with the lifeguard team."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AlertConfig(BaseModel):
    """How often an unacknowledged ALARM is repeated, and where incidents are stored."""

    model_config = ConfigDict(extra="forbid")

    # T_ack: an ALARM nobody has acknowledged is repeated (louder / other channels) every this many
    # seconds. Decision D-03: the repeat never dispatches the drone.
    realert_interval_s: float = Field(15.0, gt=0.0)
    # SQLite incident log; null keeps it in memory (lost at exit).
    incident_db: Path | None = None
