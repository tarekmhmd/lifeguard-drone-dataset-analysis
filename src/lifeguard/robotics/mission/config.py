"""Mission planner configuration (Blueprint M10). Simulation only; starting values to agree with the team."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MissionConfig(BaseModel):
    """Launch point, altitudes, stand-off distance, geofence margin and pre-flight minimums."""

    model_config = ConfigDict(extra="forbid")

    home_pool_xy: tuple[float, float] = (-1.0, -1.0)   # launch pad on the deck, pool metres
    cruise_altitude_m: float = Field(6.0, gt=0.0)      # transit altitude above the water
    hover_altitude_m: float = Field(3.0, gt=0.0)       # altitude at the release point
    max_altitude_m: float = Field(10.0, gt=0.0)        # never planned above this (fence ceiling)
    standoff_m: float = Field(1.5, gt=0.0)             # hover this far from the swimmer, never over them
    fence_margin_m: float = Field(2.0, ge=0.0)         # geofence = pool rectangle grown by this margin
    min_battery_pct: float = Field(40.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _ordered_altitudes(self) -> MissionConfig:
        if not self.hover_altitude_m <= self.cruise_altitude_m <= self.max_altitude_m:
            raise ValueError("altitudes must satisfy hover <= cruise <= max")
        return self
