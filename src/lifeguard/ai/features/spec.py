"""Feature specification (M1-03): the versioned description of ``TrackObs.features``.

``feature_spec.json`` (next to this module) is the single source of truth for feature names and their order;
the behaviour model's input vector (``lifeguard.ai.temporal.dataset``) is built from it, and trained models
store ``feature_spec_version`` so an incompatible spec is detected at load time.

Versioning: patch = wording only; minor = a feature appended (old models still load, they ignore it only if
retrained); major = a feature removed, renamed, re-ordered or its definition changed (retrain required).
Promoting this spec into the shared contract (``FeatureSpec``, decision C-12) is a minor contract version
and needs both members' approval; until then it lives with its owner (Member 1, ``lifeguard.ai``).
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

SPEC_PATH = Path(__file__).with_name("feature_spec.json")
_NAME = r"^[a-z][a-z0-9_]{0,63}$"  # contract FeatureName pattern


class FeatureDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_NAME)
    unit: str
    definition: str = Field(min_length=10)
    inputs: list[str]
    available_when: str
    range: tuple[float | None, float | None]


class PlannedFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_NAME)
    blocked_by: str


class ModelVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layout: str
    missing_value: float
    size: int


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    producer: str
    status: str
    window_s: float = Field(gt=0)
    step_s: float = Field(gt=0)
    min_span_s: float = Field(gt=0)
    per_frame: bool
    notes: list[str]
    features: list[FeatureDef]
    related_contract_fields: dict[str, str]
    model_vector: ModelVector
    planned_not_implemented: list[PlannedFeature]

    @field_validator("features")
    @classmethod
    def _unique(cls, value: list[FeatureDef]) -> list[FeatureDef]:
        names = [f.name for f in value]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        return value

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def major(self) -> int:
        return int(self.spec_version.split(".")[0])


@cache
def load_feature_spec(path: Path = SPEC_PATH) -> FeatureSpec:
    """Load and validate the feature spec (cached)."""
    return FeatureSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
