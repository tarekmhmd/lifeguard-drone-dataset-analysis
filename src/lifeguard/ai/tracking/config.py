"""Tracker configuration. Starting values from ByteTrack; re-tune on annotated pool clips (roadmap P4-4)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrackerConfig(BaseModel):
    """ByteTrack-style association thresholds and track lifetimes."""

    model_config = ConfigDict(extra="forbid")

    high_thresh: float = Field(0.5, gt=0.0, le=1.0)       # first-stage detections
    low_thresh: float = Field(0.1, ge=0.0, lt=1.0)        # second stage keeps partially submerged swimmers
    new_track_thresh: float = Field(0.6, gt=0.0, le=1.0)  # minimum confidence to start a track
    match_iou: float = Field(0.2, gt=0.0, le=1.0)         # stage 1: tracked + lost tracks vs high detections
    low_match_iou: float = Field(0.5, gt=0.0, le=1.0)     # stage 2: tracked tracks vs low detections
    tentative_iou: float = Field(0.3, gt=0.0, le=1.0)     # tentative tracks vs leftover high detections
    min_hits: int = Field(2, ge=1)                        # matches before a track is reported
    track_buffer_s: float = Field(5.0, gt=0.0)            # lost tracks kept for re-association
    lost_emit_s: float = Field(2.0, ge=0.0)               # lost tracks reported with state=lost (C-08)

    @model_validator(mode="after")
    def _ordered(self) -> TrackerConfig:
        """Low threshold below high threshold; emitting lost tracks only while they are kept."""
        if self.low_thresh >= self.high_thresh:
            raise ValueError("low_thresh must be below high_thresh")
        if self.lost_emit_s > self.track_buffer_s:
            raise ValueError("lost_emit_s must not exceed track_buffer_s")
        return self
