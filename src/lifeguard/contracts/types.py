"""Contract data models for Phase 1 (schema version 1.0.0)."""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lifeguard.contracts.enums import (
    AckDecision,
    DetClass,
    GroundTruthType,
    HealthComponent,
    RiskLevel,
    RiskReason,
    Status,
    TrackState,
    Zone,
)
from lifeguard.contracts.errors import ContractVersionError
from lifeguard.contracts.timeutil import UTC_PATTERN
from lifeguard.contracts.version import SCHEMA_VERSION, check_schema_version

_BEHAVIOR_SUM_TOLERANCE = 1e-3
_COCO_KEYPOINTS = 17


class ContractModel(BaseModel):
    """Base model: unknown fields ignored, instances frozen, non-finite floats rejected."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


Prob = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
Ident = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")]
UtcStr = Annotated[str, Field(pattern=UTC_PATTERN)]
SemVer = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
FeatureName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
# ``TrackObs.features`` key: mean confidence of the detector's distress class over the last 2 s of the
# track's matched detections (0.0 for detectors without a distress class). Read by the risk engine.
DISTRESS_FEATURE: str = "det_distress_2s"


def _check_version(value: str) -> str:
    """Validate a schema version and convert the project error into a pydantic error."""
    try:
        check_schema_version(value)
    except ContractVersionError as err:
        raise ValueError(str(err)) from err
    return value


class Behavior(ContractModel):
    """Output of the temporal behaviour classifier."""

    normal_swim: Prob
    idle: Prob
    play: Prob
    dive: Prob
    distress: Prob
    drowning: Prob
    model_version: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> Behavior:
        """Reject behaviour vectors whose probabilities do not sum to 1."""
        total = self.normal_swim + self.idle + self.play + self.dive + self.distress + self.drowning
        if abs(total - 1.0) > _BEHAVIOR_SUM_TOLERANCE:
            raise ValueError("behavior probabilities must sum to 1")
        return self


class TrackObs(ContractModel):
    """One person (track) observed in one frame."""

    track_id: int = Field(ge=1)
    track_uid: str
    merged_from: tuple[Annotated[int, Field(ge=1)], ...] = ()
    cls: DetClass
    box_xyxy: tuple[float, float, float, float]
    det_conf: Prob
    state: TrackState
    pool_xy: tuple[float, float] | None
    in_water: bool
    zone: Zone | None
    keypoints: tuple[tuple[float, float, float], ...] | None = None
    features: dict[FeatureName, float] = Field(default_factory=dict)
    behavior: Behavior | None = None

    @model_validator(mode="after")
    def _validate_track(self) -> TrackObs:
        """Enforce the TrackObs cross-field rules from SPEC section 4.6.2."""
        if self.track_id in self.merged_from:
            raise ValueError("merged_from must not contain track_id")

        x1, y1, x2, y2 = self.box_xyxy
        if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
            raise ValueError("box_xyxy must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")

        if self.in_water and self.pool_xy is None:
            raise ValueError("in_water requires pool_xy to be set")
        if self.pool_xy is None and self.zone is not None:
            raise ValueError("zone must be None when pool_xy is None")

        if self.keypoints is not None:
            if len(self.keypoints) != _COCO_KEYPOINTS:
                raise ValueError("keypoints must contain exactly 17 entries")
            for _, _, conf in self.keypoints:
                if not 0.0 <= conf <= 1.0:
                    raise ValueError("keypoint confidence must be within [0, 1]")

        if any(not math.isfinite(value) for value in self.features.values()):
            raise ValueError("feature values must be finite")

        return self


class Submersion(ContractModel):
    """An active submersion timer reported by perception."""

    track_uid: str
    last_pool_xy: tuple[float, float]
    seconds_submerged: NonNegFloat
    since_t_capture: float


class FrameResult(ContractModel):
    """Perception output for one processed frame."""

    schema_version: SemVer = SCHEMA_VERSION
    cam_id: Ident
    session_id: Ident
    frame_idx: Annotated[int, Field(ge=0)]
    t_capture: Annotated[float, Field(ge=0.0)]
    t_utc: UtcStr
    t_media: Annotated[float, Field(ge=0.0)] | None = None
    frame_width: Annotated[int, Field(ge=1)]
    frame_height: Annotated[int, Field(ge=1)]
    status: Status = Status.OK
    error: str | None = None
    latency_ms: NonNegFloat
    tracks: tuple[TrackObs, ...] = ()
    submersions: tuple[Submersion, ...] = ()
    model_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_compatible(cls, value: str) -> str:
        return _check_version(value)

    @model_validator(mode="after")
    def _validate_status(self) -> FrameResult:
        """Enforce the status/error shape rules (SPEC 4.6.4 rules 1-2)."""
        if self.status == Status.OK and self.error is not None:
            raise ValueError("status ok requires error to be None")
        if self.status == Status.ERROR:
            if not self.error:
                raise ValueError("status error requires a non-empty error")
            if self.tracks or self.submersions:
                raise ValueError("status error requires empty tracks and submersions")
        return self

    @model_validator(mode="after")
    def _validate_track_identity(self) -> FrameResult:
        """Enforce unique track ids and correct track uids (rules 4-5)."""
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track_id values must be unique within tracks")
        for track in self.tracks:
            expected = make_track_uid(self.cam_id, self.session_id, track.track_id)
            if track.track_uid != expected:
                raise ValueError("track_uid must equal cam_id:session_id:track_id")
        return self

    @model_validator(mode="after")
    def _validate_track_geometry(self) -> FrameResult:
        """Enforce box and keypoint bounds inside the frame (rules 6-7)."""
        for track in self.tracks:
            if track.box_xyxy[2] > self.frame_width or track.box_xyxy[3] > self.frame_height:
                raise ValueError("box must fit inside the frame")
            if track.keypoints is None:
                continue
            for x, y, conf in track.keypoints:
                if conf == 0:
                    continue
                if not (0 <= x <= self.frame_width and 0 <= y <= self.frame_height):
                    raise ValueError("keypoint must lie inside the frame")
        return self

    @model_validator(mode="after")
    def _validate_submersions(self) -> FrameResult:
        """Enforce submersion uniqueness and ownership (rule 8)."""
        uids = [submersion.track_uid for submersion in self.submersions]
        if len(uids) != len(set(uids)):
            raise ValueError("submersion track_uid values must be unique")
        prefix = f"{self.cam_id}:{self.session_id}:"
        for uid in uids:
            if not uid.startswith(prefix):
                raise ValueError("submersion track_uid must belong to this camera and session")
        return self


class RiskEvent(ContractModel):
    """Risk-engine event describing a risk-level transition."""

    schema_version: SemVer = SCHEMA_VERSION
    event_id: Ident
    cam_id: Ident
    session_id: Ident
    track_uid: str
    from_level: RiskLevel
    to_level: RiskLevel
    risk: Prob
    reason: RiskReason
    pool_xy: tuple[float, float] | None = None
    t_capture: Annotated[float, Field(ge=0.0)]
    t_utc: UtcStr
    t_media: float | None = None
    clip_path: str | None = None
    snapshot_path: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version_compatible(cls, value: str) -> str:
        return _check_version(value)

    @model_validator(mode="after")
    def _validate_event(self) -> RiskEvent:
        """Enforce the RiskEvent cross-field rules from SPEC section 4.6.5."""
        if self.from_level == self.to_level:
            raise ValueError("to_level must differ from from_level")
        prefix = f"{self.cam_id}:{self.session_id}:"
        if not self.track_uid.startswith(prefix):
            raise ValueError("track_uid must belong to this camera and session")
        return self


class Acknowledgement(ContractModel):
    """Lifeguard decision for a risk event."""

    event_id: Ident
    track_uid: str
    decision: AckDecision
    by: Annotated[str, Field(min_length=1, max_length=64)]
    t_utc: UtcStr

    @field_validator("by")
    @classmethod
    def _by_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("by must not be blank")
        return value


class DispatchCommand(ContractModel):
    """Human-approved dispatch command for the mission manager."""

    schema_version: SemVer = SCHEMA_VERSION
    event_id: Ident
    track_uid: str
    target_pool_xy: tuple[float, float]
    confirmed_by: Annotated[str, Field(min_length=1, max_length=64)]
    t_utc: UtcStr

    @field_validator("schema_version")
    @classmethod
    def _schema_version_compatible(cls, value: str) -> str:
        return _check_version(value)

    @field_validator("confirmed_by")
    @classmethod
    def _confirmed_by_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("confirmed_by must not be blank")
        return value


class SystemHealth(ContractModel):
    """Health report from any component."""

    schema_version: SemVer = SCHEMA_VERSION
    component: HealthComponent
    status: Status
    detail: str = ""
    t_utc: UtcStr
    t_capture: float | None = None


class EngineHeartbeat(ContractModel):
    """Return value of PerceptionEngine.heartbeat()."""

    alive: bool
    frames_processed: Annotated[int, Field(ge=0)]
    last_frame_idx: int | None = None
    last_latency_ms: Annotated[float, Field(ge=0.0)] | None = None
    fps: Annotated[float, Field(ge=0.0)] | None = None
    gpu_mem_mb: Annotated[float, Field(ge=0.0)] | None = None
    detail: str = ""


class GroundTruthEvent(ContractModel):
    """Evaluation label for a submersion, distress or drowning event."""

    event_id: Ident
    session_id: Ident
    video_file: str = ""
    type: GroundTruthType
    t_onset: Annotated[float, Field(ge=0.0)]
    t_end: float | None = None
    track_uid: str | None = None
    pool_x: float | None = None
    pool_y: float | None = None

    @model_validator(mode="after")
    def _validate_event(self) -> GroundTruthEvent:
        """Enforce the GroundTruthEvent rules from SPEC section 4.6.10."""
        if self.t_end is not None and self.t_end <= self.t_onset:
            raise ValueError("t_end must be greater than t_onset")
        if (self.pool_x is None) != (self.pool_y is None):
            raise ValueError("pool_x and pool_y must both be set or both be None")
        return self


def make_track_uid(cam_id: str, session_id: str, track_id: int) -> str:
    """Build the contract track identifier."""
    return f"{cam_id}:{session_id}:{track_id}"


def parse_track_uid(track_uid: str) -> tuple[str, str, int]:
    """Split a track identifier into camera, session and numeric track id.

    Args:
        track_uid: Identifier in ``cam_id:session_id:track_id`` form.

    Returns:
        The ``(cam_id, session_id, track_id)`` tuple.

    Raises:
        ValueError: If ``track_uid`` does not contain exactly three parts or the
            track id is not an integer greater than or equal to 1.
    """
    parts = track_uid.split(":")
    if len(parts) != 3:
        raise ValueError(f"malformed track_uid: {track_uid!r}")
    cam_id, session_id, raw_track_id = parts
    try:
        track_id = int(raw_track_id)
    except ValueError as err:
        raise ValueError(f"malformed track_uid (non-integer track_id): {track_uid!r}") from err
    if track_id < 1:
        raise ValueError(f"malformed track_uid (track_id < 1): {track_uid!r}")
    return cam_id, session_id, track_id
