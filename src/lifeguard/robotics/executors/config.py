"""Drone executor configuration (SIMULATION ONLY)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DroneConfig(BaseModel):
    """Executor selection, failsafe thresholds and the simulated release channel."""

    model_config = ConfigDict(extra="forbid")

    executor: Literal["simulated", "mavlink_sitl"] = "simulated"
    # MAVLink endpoint; only local SITL endpoints are accepted (physical deployment is disabled).
    endpoint: str = "udp:127.0.0.1:14550"
    heartbeat_timeout_s: float = Field(10.0, gt=0.0)   # connect timeout
    link_timeout_s: float = Field(3.0, gt=0.0)         # no heartbeat for this long -> failsafe RTL
    rtl_battery_pct: float = Field(25.0, ge=0.0, le=100.0)
    arrive_tolerance_m: float = Field(0.5, gt=0.0)
    # The release waits for a human confirmation by default (architecture section 16: human in the loop).
    require_release_confirmation: bool = True
    release_servo: int = Field(9, ge=1, le=16)          # SERVOn output wired to the (simulated) latch
    release_pwm: int = Field(1900, ge=800, le=2200)
    # Pure-Python simulator parameters.
    sim_speed_mps: float = Field(5.0, gt=0.0)
    sim_vertical_mps: float = Field(2.0, gt=0.0)
    sim_drain_pct_per_s: float = Field(0.2, ge=0.0)


class ResponseConfig(BaseModel):
    """Which response platform the runtime uses (an open supervisor decision; default: none)."""

    model_config = ConfigDict(extra="forbid")

    platform: Literal["none", "drone"] = "none"
    drone: DroneConfig = Field(default_factory=DroneConfig)
