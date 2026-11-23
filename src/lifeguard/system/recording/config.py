"""Alarm clip recording configuration."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RecordingConfig(BaseModel):
    """Where and how alarm clips are written."""

    model_config = ConfigDict(extra="forbid")

    out_dir: Path | None = None              # null disables clip recording
    pre_s: float = Field(10.0, ge=0.0)       # seconds kept before the trigger
    post_s: float = Field(5.0, ge=0.0)       # seconds recorded after the trigger
    max_width: int = Field(640, ge=64)       # frames are down-scaled to this width to bound memory
    codec: str = Field("mp4v", min_length=4, max_length=4)
