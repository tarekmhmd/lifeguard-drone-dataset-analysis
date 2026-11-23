"""Health watchdog configuration (Blueprint M13). Starting values; tune to the camera and hardware."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthConfig(BaseModel):
    """Thresholds of the health checks."""

    model_config = ConfigDict(extra="forbid")

    camera_timeout_s: float = Field(3.0, gt=0.0)    # no frame for this long -> INGEST error
    frozen_after_s: float = Field(3.0, gt=0.0)      # identical frames for this long -> INGEST error
    fps_floor: float = Field(5.0, ge=0.0)           # processing rate below this -> PERCEPTION degraded
    fps_window_s: float = Field(5.0, gt=0.0)        # window over which the rate is measured
    min_free_disk_mb: float = Field(1024.0, ge=0.0)  # STORAGE error below this
