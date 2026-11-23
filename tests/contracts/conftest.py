"""Shared builders for contract-model tests (TEST_PLAN section 4)."""
from __future__ import annotations

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

_T_UTC = "2026-12-10T12:00:00.000Z"


def make_keypoints(
    x: float = 100.0,
    y: float = 200.0,
    conf: float = 0.9,
) -> tuple[tuple[float, float, float], ...]:
    """Return a valid COCO-length (17-entry) keypoint tuple."""
    return tuple((x, y, conf) for _ in range(17))


def make_behavior(**overrides: object) -> dict[str, object]:
    """Return a valid Behavior dict whose probabilities sum to 1."""
    data: dict[str, object] = {
        "normal_swim": 0.7,
        "idle": 0.1,
        "play": 0.05,
        "dive": 0.05,
        "distress": 0.05,
        "drowning": 0.05,
        "model_version": "behavior_v1",
    }
    data.update(overrides)
    return data


def make_track(**overrides: object) -> dict[str, object]:
    """Return a valid TrackObs dict with one tracked person."""
    data: dict[str, object] = {
        "track_id": 1,
        "track_uid": "cam_north:s1:1",
        "cls": DetClass.PERSON,
        "box_xyxy": (100.0, 200.0, 180.0, 320.0),
        "det_conf": 0.9,
        "state": TrackState.TRACKED,
        "pool_xy": (5.0, 6.0),
        "in_water": True,
        "zone": Zone.SHALLOW,
    }
    data.update(overrides)
    return data


def make_submersion(**overrides: object) -> dict[str, object]:
    """Return a valid Submersion dict."""
    data: dict[str, object] = {
        "track_uid": "cam_north:s1:1",
        "last_pool_xy": (5.0, 6.0),
        "seconds_submerged": 3.2,
        "since_t_capture": 96.8,
    }
    data.update(overrides)
    return data


def make_frame(**overrides: object) -> dict[str, object]:
    """Return a valid FrameResult dict: 1920x1080, cam_north, session s1, one track."""
    data: dict[str, object] = {
        "cam_id": "cam_north",
        "session_id": "s1",
        "frame_idx": 0,
        "t_capture": 100.0,
        "t_utc": _T_UTC,
        "frame_width": 1920,
        "frame_height": 1080,
        "latency_ms": 12.5,
        "tracks": (make_track(),),
    }
    data.update(overrides)
    return data


def make_risk_event(**overrides: object) -> dict[str, object]:
    """Return a valid RiskEvent dict."""
    data: dict[str, object] = {
        "event_id": "evt_1",
        "cam_id": "cam_north",
        "session_id": "s1",
        "track_uid": "cam_north:s1:1",
        "from_level": RiskLevel.NORMAL,
        "to_level": RiskLevel.WATCH,
        "risk": 0.55,
        "reason": RiskReason.RISK_ABOVE_THRESHOLD,
        "t_capture": 100.0,
        "t_utc": _T_UTC,
    }
    data.update(overrides)
    return data


def make_ack(**overrides: object) -> dict[str, object]:
    """Return a valid Acknowledgement dict."""
    data: dict[str, object] = {
        "event_id": "evt_1",
        "track_uid": "cam_north:s1:1",
        "decision": AckDecision.CONFIRM,
        "by": "lifeguard_1",
        "t_utc": _T_UTC,
    }
    data.update(overrides)
    return data


def make_dispatch(**overrides: object) -> dict[str, object]:
    """Return a valid DispatchCommand dict."""
    data: dict[str, object] = {
        "event_id": "evt_1",
        "track_uid": "cam_north:s1:1",
        "target_pool_xy": (5.0, 6.0),
        "confirmed_by": "lifeguard_1",
        "t_utc": _T_UTC,
    }
    data.update(overrides)
    return data


def make_health(**overrides: object) -> dict[str, object]:
    """Return a valid SystemHealth dict."""
    data: dict[str, object] = {
        "component": HealthComponent.PERCEPTION,
        "status": Status.OK,
        "detail": "ok",
        "t_utc": _T_UTC,
    }
    data.update(overrides)
    return data


def make_heartbeat(**overrides: object) -> dict[str, object]:
    """Return a valid EngineHeartbeat dict."""
    data: dict[str, object] = {
        "alive": True,
        "frames_processed": 10,
        "last_frame_idx": 9,
        "last_latency_ms": 12.5,
        "fps": 30.0,
        "gpu_mem_mb": 512.0,
        "detail": "mock",
    }
    data.update(overrides)
    return data


def make_ground_truth(**overrides: object) -> dict[str, object]:
    """Return a valid GroundTruthEvent dict."""
    data: dict[str, object] = {
        "event_id": "evt_1",
        "session_id": "s1",
        "video_file": "",
        "type": GroundTruthType.SUBMERSION,
        "t_onset": 20.0,
        "t_end": None,
        "track_uid": "cam_north:s1:1",
        "pool_x": 5.0,
        "pool_y": 6.0,
    }
    data.update(overrides)
    return data
