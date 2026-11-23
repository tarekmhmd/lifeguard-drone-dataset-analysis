"""Which pool geometry the runtime uses."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GeometryConfig(BaseModel):
    """A calibration file (real geometry) or, when null, an explicitly labelled placeholder rectangle."""

    model_config = ConfigDict(extra="forbid")

    calibration: Path | None = None                  # JSON from python -m lifeguard.system.geometry.calibrate
    placeholder_length_m: float = Field(25.0, gt=0)  # used ONLY without a calibration (logged as PLACEHOLDER)
    placeholder_width_m: float = Field(12.5, gt=0)
