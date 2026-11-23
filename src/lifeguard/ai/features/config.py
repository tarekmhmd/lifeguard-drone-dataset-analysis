"""Feature extraction and submersion monitor configuration (Blueprint M6). Starting values to tune."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    """Window length and pool-space margins."""

    model_config = ConfigDict(extra="forbid")

    window_s: float = Field(4.0, gt=0.0)            # ring buffer for per-track features
    min_span_s: float = Field(0.5, gt=0.0)          # minimum time span for motion features
    interior_margin_m: float = Field(1.0, ge=0.0)   # disappearances closer to an edge start no timer
    exit_margin_m: float = Field(1.0, ge=0.0)       # disappearances near ladders/steps start no timer
    # A reappearance this close ends the timer. Blueprint M6 suggests 1.5 m; 2.0 m allows for drift while
    # the tracker confirms the resurfaced swimmer (one extra frame). Tune on staged clips.
    reappear_radius_m: float = Field(2.0, gt=0.0)
