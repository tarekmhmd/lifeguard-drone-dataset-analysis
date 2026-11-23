"""Member 1 AI configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lifeguard.ai.features.config import FeatureConfig
from lifeguard.ai.tracking.config import TrackerConfig
from lifeguard.common.config import load_yaml_model


class MockEngineConfig(BaseModel):
    """Mock engine settings."""

    model_config = ConfigDict(extra="forbid")

    fixture_path: Path
    on_exhausted: Literal["empty", "loop", "error"] = "empty"


class YoloEngineConfig(BaseModel):
    """Real engine detector settings (Ultralytics YOLO, optional ``vision`` extra).

    Give either ``weights`` (a file path) or ``model`` (a name in the model registry, M2-12).
    """

    model_config = ConfigDict(extra="forbid")

    weights: Path | None = None
    model: str | None = None
    conf: float = Field(0.1, gt=0.0, lt=1.0)   # low on purpose: feeds ByteTrack's low-confidence stage
    iou: float = Field(0.7, gt=0.0, le=1.0)
    imgsz: int = Field(640, ge=32)
    device: str | None = None
    person_class_id: int = Field(0, ge=0)        # COCO "person"

    @model_validator(mode="after")
    def _one_source(self) -> YoloEngineConfig:
        if (self.weights is None) == (self.model is None):
            raise ValueError("yolo needs exactly one of 'weights' (path) or 'model' (registry name)")
        return self


class AIConfig(BaseModel):
    """Member 1 configuration: engine selection plus per-engine settings."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["mock", "yolo"]
    mock: MockEngineConfig | None = None
    yolo: YoloEngineConfig | None = None
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    # Temporal behaviour model for the real engine: "none", the unvalidated rules baseline, or a trained
    # softmax model (JSON from ``python -m lifeguard.ai.temporal.train``) given as a path or a registry name.
    behavior: Literal["none", "rules", "softmax"] = "none"
    behavior_model: Path | None = None
    behavior_model_name: str | None = None
    # Model registry (M2-12). When set, every model file the factory loads must be registered with a
    # matching sha256, otherwise start-up is refused with a SYSTEM FAULT.
    model_registry: Path | None = None

    @model_validator(mode="after")
    def _engine_block_required(self) -> AIConfig:
        """The selected engine requires its settings block."""
        if self.engine == "mock" and self.mock is None:
            raise ValueError("engine 'mock' requires mock configuration")
        if self.engine == "yolo" and self.yolo is None:
            raise ValueError("engine 'yolo' requires yolo configuration")
        if self.behavior == "softmax" and self.behavior_model is None and self.behavior_model_name is None:
            raise ValueError("behavior 'softmax' requires behavior_model or behavior_model_name")
        named = (self.yolo is not None and self.yolo.model is not None) or self.behavior_model_name is not None
        if named and self.model_registry is None:
            raise ValueError("models referenced by name need model_registry")
        return self


def load_ai_config(path: Path | str) -> AIConfig:
    """Load AIConfig and resolve relative paths against the YAML directory."""
    config_path = Path(path)
    config = load_yaml_model(config_path, AIConfig)
    base = config_path.parent

    def resolve(value: Path | None) -> Path | None:
        return None if value is None or value.is_absolute() else (base / value).resolve()

    if config.mock is not None:
        config.mock.fixture_path = resolve(config.mock.fixture_path) or config.mock.fixture_path
    if config.yolo is not None and config.yolo.weights is not None:
        config.yolo.weights = resolve(config.yolo.weights) or config.yolo.weights
    if config.behavior_model is not None:
        config.behavior_model = resolve(config.behavior_model) or config.behavior_model
    if config.model_registry is not None:
        config.model_registry = resolve(config.model_registry) or config.model_registry
    return config
