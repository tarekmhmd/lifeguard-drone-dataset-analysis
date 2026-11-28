"""Member 2 system configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from lifeguard.common.config import load_yaml_model
from lifeguard.contracts.errors import ConfigError
from lifeguard.contracts.types import Ident
from lifeguard.hardware.config import HardwareConfig, load_hardware_config_safe
from lifeguard.robotics.executors.config import ResponseConfig
from lifeguard.robotics.mission.config import MissionConfig
from lifeguard.system.alerts.config import AlertConfig
from lifeguard.system.geometry.config import GeometryConfig
from lifeguard.system.health.config import HealthConfig
from lifeguard.system.recording.config import RecordingConfig
from lifeguard.system.risk.config import RiskConfig


class SystemConfig(BaseModel):
    """Member 2 configuration: ids, logging, engine error threshold, risk, alerts, health, recording, mission."""

    model_config = ConfigDict(extra="forbid")

    cam_id: Ident = "cam_sim"
    session_id: Ident | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None
    max_consecutive_engine_errors: int = Field(10, ge=1)
    latency_budget_ms: float = Field(100.0, gt=0)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    # ESP32 pool prototype (docs/HARDWARE_INTEGRATION.md). Disabled by default; never feeds the risk engine.
    # Either inline (``hardware:``) or a dedicated file (``hardware_file: hardware.yaml``, relative to this YAML).
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    hardware_file: Path | None = None
    # Why ``hardware_file`` could not be used (missing/invalid); the runtime then shows a hardware FAULT and the
    # AI pipeline runs normally. Set by load_system_config, never read from YAML.
    _hardware_error: str | None = PrivateAttr(default=None)

    @property
    def hardware_error(self) -> str | None:
        return self._hardware_error


def _resolve(base: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    return (base / value).resolve()


def load_system_config(path: Path | str) -> SystemConfig:
    """Load SystemConfig and resolve relative file paths against the YAML directory."""
    config_path = Path(path)
    config = load_yaml_model(config_path, SystemConfig)
    base = config_path.parent
    config.log_file = _resolve(base, config.log_file)
    config.alerts.incident_db = _resolve(base, config.alerts.incident_db)
    config.recording.out_dir = _resolve(base, config.recording.out_dir)
    config.geometry.calibration = _resolve(base, config.geometry.calibration)
    config.hardware.command_log = _resolve(base, config.hardware.command_log)
    if config.hardware_file is not None:
        if "hardware" in config.model_fields_set:
            raise ConfigError(f"{config_path}: use either an inline 'hardware:' block or 'hardware_file:', not both")
        config.hardware_file = _resolve(base, config.hardware_file)
        loaded, problem = load_hardware_config_safe(config.hardware_file)   # never stops the AI pipeline
        if loaded is not None:
            config.hardware = loaded
        else:
            config._hardware_error = problem
    return config
