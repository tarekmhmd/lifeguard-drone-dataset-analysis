"""Real perception engine: detector -> ByteTrack -> pool mapping -> submersion monitor -> features.

REAL implementation of the ``PerceptionEngine`` contract (roadmap P4-6). Its accuracy depends on the
detector weights and the pool calibration; with COCO-pretrained weights and ``StubPoolGeometry`` it is a
working pipeline, not a validated drowning detector. ``TrackObs.behavior`` is filled only when a
``BehaviorClassifier`` is supplied (roadmap Phase 5); otherwise it stays ``None``.

Pixel -> pool point: the box centre is projected with ``PoolGeometry.to_pool``; a person counts as
``in_water`` when that point lies inside the pool. This is an approximation until head boxes and a
calibrated homography are available (roadmap P4-2, Blueprint M2 recommends the head-box centre).
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from lifeguard.ai.detection.base import Detector
from lifeguard.ai.features.config import FeatureConfig
from lifeguard.ai.features.extractor import FeatureExtractor
from lifeguard.ai.features.submersion import PoolTrackView, SubmersionMonitor
from lifeguard.ai.pose.base import Keypoints, PoseEstimator
from lifeguard.ai.temporal.base import BehaviorClassifier
from lifeguard.ai.tracking.bytetrack import ByteTracker, TrackedBox
from lifeguard.ai.tracking.config import TrackerConfig
from lifeguard.contracts.enums import Status, TrackState
from lifeguard.contracts.interfaces import PerceptionEngine, PoolGeometry
from lifeguard.contracts.timeutil import UTC_PATTERN, utc_now
from lifeguard.contracts.types import (
    DISTRESS_FEATURE,
    EngineHeartbeat,
    FrameResult,
    Submersion,
    TrackObs,
    make_track_uid,
)

logger = logging.getLogger(__name__)

_IDENT_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_TRACKER_VERSION = "bytetrack_np_v1"
_FEATURES_VERSION = "rules_v1"
_FPS_SMOOTHING = 0.9
# Detector distress evidence (``DISTRESS_FEATURE``): mean ``Detection.distress`` over this window.
_DISTRESS_WINDOW_S = 2.0
_DISTRESS_MIN_SPAN_S = 1.9   # the track must be this old before the mean is reported


class RealPerceptionEngine(PerceptionEngine):
    """Frame -> FrameResult with real detection, tracking, submersion timers and motion features."""

    def __init__(
        self,
        detector: Detector,
        geometry: PoolGeometry,
        cam_id: str,
        session_id: str,
        tracker_config: TrackerConfig | None = None,
        feature_config: FeatureConfig | None = None,
        behavior_classifier: BehaviorClassifier | None = None,
        pose_estimator: PoseEstimator | None = None,
        pose_status: str = "disabled",
    ) -> None:
        """Build the pipeline.

        ``pose_estimator`` (optional) fills ``TrackObs.keypoints`` for visible tracks. It is display and
        recording only: the risk engine does not read keypoints, and a pose failure leaves the keypoints
        ``None`` for that frame without affecting detection, tracking or the result status.

        Raises:
            ValueError: If cam_id or session_id is not a valid identifier.
        """
        if not _IDENT_PATTERN.fullmatch(cam_id):
            raise ValueError(f"invalid cam_id: {cam_id!r}")
        if not _IDENT_PATTERN.fullmatch(session_id):
            raise ValueError(f"invalid session_id: {session_id!r}")
        self._detector = detector
        self._geometry = geometry
        self._cam_id = cam_id
        self._session_id = session_id
        self._tracker = ByteTracker(tracker_config)
        feature_config = feature_config or FeatureConfig()
        self._monitor = SubmersionMonitor(geometry, feature_config)
        self._features = FeatureExtractor(feature_config)
        self._distress: dict[int, deque[tuple[float, float]]] = {}
        self._frame_size: tuple[int, int] | None = None
        self._frames_processed = 0
        self._last_frame_idx: int | None = None
        self._last_latency_ms: float | None = None
        self._last_t_capture: float | None = None
        self._fps: float | None = None
        self._model_versions = {
            "engine": "real",
            "detector": detector.model_version,
            "tracker": _TRACKER_VERSION,
            "features": _FEATURES_VERSION,
        }
        self._behavior = behavior_classifier
        if behavior_classifier is not None:
            self._model_versions["behavior"] = behavior_classifier.model_version
        self._pose = pose_estimator
        self.pose_stats = PoseStats()
        if pose_estimator is not None:
            self._model_versions["pose"] = pose_estimator.model_version
        elif pose_status != "disabled":
            self._model_versions["pose"] = pose_status   # e.g. "unavailable: weights file missing"

    @property
    def cam_id(self) -> str:
        return self._cam_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def pose_status(self) -> str:
        """Pose model version, its unavailability reason, or "disabled"."""
        return self._model_versions.get("pose", "disabled")

    def reset(self) -> None:
        """Start a new session: clear tracks, timers, feature buffers and counters."""
        self._tracker.reset()
        self._monitor.reset()
        self._features.reset()
        self._distress.clear()
        if self._behavior is not None:
            self._behavior.reset()
        self._frame_size = None
        self._frames_processed = 0
        self._last_frame_idx = None
        self._last_latency_ms = None
        self._last_t_capture = None
        self._fps = None
        self.pose_stats = PoseStats()

    def heartbeat(self) -> EngineHeartbeat:
        return EngineHeartbeat(
            alive=True,
            frames_processed=self._frames_processed,
            last_frame_idx=self._last_frame_idx,
            last_latency_ms=self._last_latency_ms,
            fps=self._fps,
            gpu_mem_mb=None,
            detail=f"real: {self._model_versions['detector']}",
        )

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None = None,
    ) -> FrameResult:
        """Process one frame. Never raises: failures become ``status=error`` results."""
        start = time.perf_counter()
        try:
            problem = self._input_error(frame_bgr, frame_idx, t_capture, t_utc)
            if problem is not None:
                return self._error(problem, start, frame_bgr, frame_idx, t_capture, t_utc, t_media)
            result = self._run(frame_bgr, frame_idx, t_capture, t_utc, t_media, start)
        except Exception as err:
            logger.exception("RealPerceptionEngine.process failed")
            return self._error(
                f"{type(err).__name__}: {err}", start, frame_bgr, frame_idx, t_capture, t_utc, t_media
            )
        self._frames_processed += 1
        self._last_frame_idx = frame_idx
        self._last_latency_ms = result.latency_ms
        if self._last_t_capture is not None and t_capture > self._last_t_capture:
            inst = 1.0 / (t_capture - self._last_t_capture)
            self._fps = inst if self._fps is None else _FPS_SMOOTHING * self._fps + (1 - _FPS_SMOOTHING) * inst
        self._last_t_capture = t_capture
        return result

    def _distress_mean(self, box: TrackedBox, t: float) -> float:
        """Mean detector distress of the track over the last ``_DISTRESS_WINDOW_S`` (unmatched frames = 0)."""
        window = self._distress.setdefault(box.track_id, deque())
        # A frame without a matched detection counts as no distress evidence (conservative).
        window.append((t, box.distress if box.state == TrackState.TRACKED else 0.0))
        while window and window[0][0] < t - _DISTRESS_WINDOW_S:
            window.popleft()
        # Report only a full window: a mean over a track's first frames is a single-frame decision.
        if box.first_seen_t > t - _DISTRESS_MIN_SPAN_S or not window:
            return 0.0
        return round(sum(v for _, v in window) / len(window), 4)

    # ----------------------------------------------------------------- pipeline

    def _run(
        self,
        frame: np.ndarray,
        frame_idx: int,
        t: float,
        t_utc: str,
        t_media: float | None,
        start: float,
    ) -> FrameResult:
        height, width = frame.shape[:2]
        detections = self._detector.detect(frame)
        boxes = self._tracker.update(detections, t)

        mapped = [(box, *self._to_pool(box)) for box in boxes]
        views = [
            PoolTrackView(
                track_id=box.track_id,
                visible=box.state == TrackState.TRACKED,
                pool_xy=xy,
                in_water=in_water,
                first_seen_t=box.first_seen_t,
            )
            for box, xy, in_water in mapped
        ]
        timers = self._monitor.update(t, views)
        poses = self._estimate_poses(frame, boxes, width, height)

        tracks: list[TrackObs] = []
        for box, xy, in_water in mapped:
            clipped = _clip_box(box.box_xyxy, width, height)
            if clipped is None:
                continue
            merged = self._monitor.merged_from(box.track_id)
            if merged:
                self._features.inherit(box.track_id, merged)
            features = self._features.observe(
                box.track_id, t, box.state == TrackState.TRACKED, xy, in_water, clipped
            )
            features[DISTRESS_FEATURE] = self._distress_mean(box, t)
            behavior = (
                self._behavior.predict(box.track_id, features, in_water)
                if self._behavior is not None
                else None
            )
            tracks.append(TrackObs(
                track_id=box.track_id,
                track_uid=make_track_uid(self._cam_id, self._session_id, box.track_id),
                merged_from=merged,
                cls=box.cls,
                box_xyxy=clipped,
                det_conf=round(min(1.0, max(0.0, box.conf)), 3),
                state=box.state,
                pool_xy=xy,
                in_water=in_water,
                zone=self._geometry.zone(*xy) if xy is not None else None,
                keypoints=poses.get(box.track_id),
                features=features,
                behavior=behavior,
            ))

        live = {b.track_id for b in boxes} | {timer.track_id for timer in timers}
        self._features.forget(self._features.known_ids - live)
        for gone in set(self._distress) - live:
            del self._distress[gone]

        submersions = tuple(
            Submersion(
                track_uid=make_track_uid(self._cam_id, self._session_id, timer.track_id),
                last_pool_xy=timer.xy,
                seconds_submerged=round(max(0.0, t - timer.start_t), 3),
                since_t_capture=timer.start_t,
            )
            for timer in timers
        )
        return FrameResult(
            cam_id=self._cam_id,
            session_id=self._session_id,
            frame_idx=frame_idx,
            t_capture=t,
            t_utc=t_utc,
            t_media=t_media if t_media is not None and t_media >= 0 else None,
            frame_width=width,
            frame_height=height,
            status=Status.OK,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tracks=tuple(tracks),
            submersions=submersions,
            model_versions=dict(self._model_versions),
        )

    def _estimate_poses(
        self, frame: np.ndarray, boxes: list[TrackedBox], width: int, height: int
    ) -> dict[int, Keypoints | None]:
        """Keypoints for the visible tracks (highest confidence first). Never raises."""
        if self._pose is None:
            return {}
        visible = sorted(
            (b for b in boxes if b.state == TrackState.TRACKED), key=lambda b: b.conf, reverse=True
        )
        requests = [(b.track_id, c) for b in visible if (c := _clip_box(b.box_xyxy, width, height)) is not None]
        if not requests:
            return {}
        start = time.perf_counter()
        try:
            poses = self._pose.estimate(frame, requests)
        except Exception as err:  # noqa: BLE001 - pose must never take detection, tracking or risk down
            self.pose_stats.errors += 1
            if self.pose_stats.errors == 1 or self.pose_stats.errors % 100 == 0:
                logger.warning("pose estimation failed (%d so far), keypoints omitted: %s: %s",
                               self.pose_stats.errors, type(err).__name__, err)
            return {}
        ms = (time.perf_counter() - start) * 1000.0
        wanted = {track_id for track_id, _ in requests}
        poses = {
            tid: (kp if _valid_keypoints(kp, width, height) else None)
            for tid, kp in poses.items() if tid in wanted
        }
        stats = self.pose_stats
        stats.frames += 1
        stats.total_ms += ms
        stats.last_ms = ms
        stats.tracks_requested += len(requests)
        stats.tracks_with_pose += sum(1 for k in poses.values() if k is not None)
        return poses

    def _to_pool(self, box: TrackedBox) -> tuple[tuple[float, float] | None, bool]:
        """Project the box centre to pool metres; in_water when the point is inside the pool."""
        x1, y1, x2, y2 = box.box_xyxy
        xy = self._geometry.to_pool((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if xy is None:
            return None, False
        rounded = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        return rounded, self._geometry.contains(*rounded)

    # ------------------------------------------------------------ validation

    def _input_error(self, frame: object, frame_idx: object, t_capture: object, t_utc: object) -> str | None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return "frame_bgr must be an HxWx3 numpy array"
        if frame.dtype != np.uint8:
            return "frame_bgr dtype must be uint8"
        size = (frame.shape[1], frame.shape[0])
        if self._frame_size is None:
            self._frame_size = size
        elif size != self._frame_size:
            return f"frame size changed from {self._frame_size} to {size}"
        if not isinstance(frame_idx, int) or isinstance(frame_idx, bool) or frame_idx < 0:
            return "frame_idx must be a non-negative integer"
        if not isinstance(t_capture, (int, float)) or not math.isfinite(t_capture) or t_capture < 0:
            return "t_capture must be a finite number >= 0"
        if not isinstance(t_utc, str) or re.fullmatch(UTC_PATTERN, t_utc) is None:
            return "t_utc must match UTC_PATTERN"
        return None

    def _error(
        self,
        message: str,
        start: float,
        frame: object,
        frame_idx: object,
        t_capture: object,
        t_utc: object,
        t_media: object,
    ) -> FrameResult:
        logger.error("perception error: %s", message)
        if isinstance(frame, np.ndarray) and frame.ndim >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
            width, height = int(frame.shape[1]), int(frame.shape[0])
        elif self._frame_size is not None:
            width, height = self._frame_size
        else:
            width, height = 1, 1
        ok_idx = isinstance(frame_idx, int) and not isinstance(frame_idx, bool) and frame_idx >= 0
        ok_t = isinstance(t_capture, (int, float)) and math.isfinite(t_capture) and t_capture >= 0
        ok_utc = isinstance(t_utc, str) and re.fullmatch(UTC_PATTERN, t_utc) is not None
        ok_media = isinstance(t_media, (int, float)) and math.isfinite(t_media) and t_media >= 0
        return FrameResult(
            cam_id=self._cam_id,
            session_id=self._session_id,
            frame_idx=frame_idx if ok_idx else 0,
            t_capture=t_capture if ok_t else 0.0,
            t_utc=t_utc if ok_utc else utc_now(),
            t_media=t_media if ok_media else None,
            frame_width=width,
            frame_height=height,
            status=Status.ERROR,
            error=message,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            model_versions=dict(self._model_versions),
        )


@dataclass
class PoseStats:
    """Pose counters since the last reset (for benchmarks and the dashboard)."""

    frames: int = 0             # frames on which the estimator ran
    tracks_requested: int = 0   # visible tracks sent to the estimator
    tracks_with_pose: int = 0   # of those, tracks that received keypoints
    errors: int = 0             # estimator exceptions (keypoints omitted for that frame)
    total_ms: float = 0.0
    last_ms: float | None = None

    @property
    def mean_ms(self) -> float | None:
        return self.total_ms / self.frames if self.frames else None


def _valid_keypoints(kp: object, width: int, height: int) -> bool:
    """True if ``kp`` satisfies the TrackObs/FrameResult keypoint rules (malformed output is dropped)."""
    if kp is None or not isinstance(kp, tuple) or len(kp) != 17:
        return False
    for point in kp:
        if not isinstance(point, tuple) or len(point) != 3:
            return False
        x, y, conf = point
        if not all(math.isfinite(v) for v in (x, y, conf)) or not 0.0 <= conf <= 1.0:
            return False
        if conf > 0 and not (0 <= x <= width and 0 <= y <= height):
            return False
    return True


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip to the frame and round to 2 decimals; None if nothing valid remains."""
    x1 = round(min(max(box[0], 0.0), float(width)), 2)
    y1 = round(min(max(box[1], 0.0), float(height)), 2)
    x2 = round(min(max(box[2], 0.0), float(width)), 2)
    y2 = round(min(max(box[3], 0.0), float(height)), 2)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)
