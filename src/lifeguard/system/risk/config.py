"""Risk engine configuration (Blueprint M8). All values are starting points to tune, not validated constants."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskWeights(BaseModel):
    """Weights of the evidence terms in the instantaneous risk score."""

    model_config = ConfigDict(extra="forbid")

    submersion: float = Field(0.40, ge=0.0)
    static: float = Field(0.25, ge=0.0)
    visibility: float = Field(0.20, ge=0.0)
    deep_zone: float = Field(0.15, ge=0.0)
    # Temporal classifier evidence; stays 0 until a validated classifier exists (roadmap P5-3).
    behavior: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _positive_total(self) -> RiskWeights:
        """At least one weight must be positive."""
        if self.total <= 0.0:
            raise ValueError("risk weights must not all be zero")
        return self

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return self.submersion + self.static + self.visibility + self.deep_zone + self.behavior


class RiskConfig(BaseModel):
    """Thresholds, dwell times and evidence parameters of the rule-based risk engine."""

    model_config = ConfigDict(extra="forbid")

    # Hard rule: a submersion longer than this forces ALARM from any lower level.
    t_sub_s: float = Field(10.0, gt=0.0)

    # Evidence and smoothing.
    tau_s: float = Field(1.5, gt=0.0)
    window_s: float = Field(4.0, gt=0.0)
    progress_ref_mps: float = Field(0.2, gt=0.0)
    min_progress_span_s: float = Field(1.0, gt=0.0)
    weights: RiskWeights = Field(default_factory=RiskWeights)

    # FSM thresholds on the smoothed risk R.
    t_watch: float = Field(0.4, gt=0.0, lt=1.0)
    t_pre_alert: float = Field(0.6, gt=0.0, lt=1.0)
    t_alarm: float = Field(0.8, gt=0.0, lt=1.0)
    hysteresis: float = Field(0.1, gt=0.0)

    # Minimum time a condition must hold before a transition fires.
    hold_watch_s: float = Field(1.0, ge=0.0)
    hold_pre_alert_s: float = Field(2.0, ge=0.0)
    hold_alarm_s: float = Field(2.0, ge=0.0)
    hold_down_s: float = Field(3.0, ge=0.0)

    # Lifecycle.
    cooldown_s: float = Field(30.0, ge=0.0)
    track_timeout_s: float = Field(2.0, gt=0.0)
    # Open decision OD-1: escalate an un-acknowledged PRE_ALERT to ALARM; null disables it.
    pre_alert_ack_timeout_s: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> RiskConfig:
        """Thresholds must increase and the hysteresis band must stay above zero."""
        if not self.t_watch < self.t_pre_alert < self.t_alarm:
            raise ValueError("thresholds must satisfy t_watch < t_pre_alert < t_alarm")
        if self.hysteresis >= self.t_watch:
            raise ValueError("hysteresis must be smaller than t_watch")
        return self
