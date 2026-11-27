"""Integrated runtime (roadmap Phase 7): one object that runs the whole system.

    frames -> PerceptionEngine (detection, tracking, features, behaviour, submersion timers)
           -> RiskEngine -> AlertService (incidents, notifications, SQLite log)
           -> ResponsePlatform (manual or simulated drone) on a lifeguard CONFIRM only
    + Watchdog (health faults) + ClipRecorder (alarm clips)

The AI core never depends on the response platform. All public methods are thread-safe so the operator
dashboard can call ``acknowledge`` / ``confirm_release`` / ``snapshot`` while ``run`` processes frames.

Optional components that are unavailable (no calibration, no clip directory, drone executor unreachable)
are reported in ``components`` and the startup log as UNAVAILABLE/PLACEHOLDER, distinct from failures
(which raise SYSTEM FAULT notifications through the watchdog).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lifeguard.ai.config import AIConfig
from lifeguard.ai.perception.factory import create_engine
from lifeguard.app.pipeline import RunSummary
from lifeguard.contracts.enums import AckDecision, RiskLevel, Status
from lifeguard.contracts.interfaces import PerceptionEngine, PoolGeometry
from lifeguard.contracts.timeutil import utc_now
from lifeguard.contracts.types import Acknowledgement, FrameResult, RiskEvent, SystemHealth
from lifeguard.system.alerts.service import AckResult, AlertService, Notification
from lifeguard.system.alerts.store import IncidentStore
from lifeguard.system.config import SystemConfig
from lifeguard.system.geometry.calibrated import CalibratedPoolGeometry
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.health.watchdog import Watchdog
from lifeguard.system.ingest.video import VideoFrame
from lifeguard.system.recording.clips import ClipRecorder
from lifeguard.system.response import ManualResponse, ResponseOutcome, ResponsePlatform
from lifeguard.system.risk.engine import RuleBasedRiskEngine

logger = logging.getLogger(__name__)

_HEALTH_PERIOD_S = 1.0
_HISTORY = 200
_PREVIEW_WIDTH = 960


def build_geometry(config: SystemConfig, image_width: int, image_height: int) -> tuple[PoolGeometry, str]:
    """Calibrated geometry from the configured file, or an explicitly labelled placeholder.

    Raises:
        ConfigError: the calibration file is missing or invalid.
    """
    if config.geometry.calibration is not None:
        geo = CalibratedPoolGeometry.from_file(config.geometry.calibration)
        if (geo.image_width, geo.image_height) != (image_width, image_height):
            logger.warning("calibration is for %dx%d but frames are %dx%d", geo.image_width, geo.image_height,
                           image_width, image_height)
        return geo, f"calibrated ({config.geometry.calibration.name})"
    geo = StubPoolGeometry(
        length_m=config.geometry.placeholder_length_m,
        width_m=config.geometry.placeholder_width_m,
        image_width=image_width,
        image_height=image_height,
    )
    return geo, "PLACEHOLDER rectangle (not calibrated: positions approximate)"


def build_response(config: SystemConfig, geometry: PoolGeometry) -> tuple[ResponsePlatform, str]:
    """The configured response platform; a drone that cannot start falls back to manual response."""
    if config.response.platform == "none":
        return ManualResponse(), "manual (fixed-camera / alert-only)"
    from lifeguard.robotics.executors.base import ExecutorError
    from lifeguard.robotics.response import DroneResponse

    try:
        drone = DroneResponse(geometry, config.mission, drone=config.response.drone)
    except (ExecutorError, ValueError) as err:
        logger.error("drone response UNAVAILABLE (%s); falling back to manual response", err)
        return ManualResponse(), f"UNAVAILABLE drone ({err}); manual response"
    return drone, f"{drone.name} [SIMULATION ONLY]"


@dataclass
class Snapshot:
    """Thread-safe copy of the runtime state for the dashboard."""

    frame: np.ndarray | None
    result: FrameResult | None
    levels: dict[str, tuple[RiskLevel, float]]
    open_incidents: dict[str, tuple[str, RiskLevel]]
    notifications: list[Notification]
    outcomes: list[ResponseOutcome]
    health: list[SystemHealth]
    components: dict[str, str]
    summary: RunSummary
    response_status: dict[str, str]


@dataclass
class _Counters:
    summary: RunSummary = field(default_factory=RunSummary)
    last_health_t: float | None = None
    last_t: float = 0.0


class LifeguardRuntime:
    """Owns every component; feed it frames (``process``) or a source (``run``)."""

    def __init__(
        self,
        ai_config: AIConfig,
        system_config: SystemConfig,
        image_size: tuple[int, int],
        session_id: str | None = None,
        engine: PerceptionEngine | None = None,
        response: ResponsePlatform | None = None,
    ) -> None:
        """Build the system. Raises ModelLoadError / ConfigError / ValueError for unusable configuration."""
        cfg = system_config
        self.session_id = session_id or cfg.session_id or datetime.now(timezone.utc).strftime("sess_%Y%m%dT%H%M%SZ")
        self.geometry, geometry_note = build_geometry(cfg, *image_size)
        self.engine = engine or create_engine(ai_config, cfg.cam_id, self.session_id, geometry=self.geometry)
        self.risk = RuleBasedRiskEngine(cfg.risk)
        self.store = IncidentStore(cfg.alerts.incident_db or ":memory:")
        self.alerts = AlertService(self.risk, cfg.alerts, self.store)
        storage = cfg.recording.out_dir or (cfg.alerts.incident_db.parent if cfg.alerts.incident_db else None)
        self.watchdog = Watchdog(cfg.health, storage_path=storage,
                                 max_consecutive_errors=cfg.max_consecutive_engine_errors)
        self.clips = ClipRecorder(cfg.recording)
        if response is not None:
            self.response, response_note = response, response.name
        else:
            self.response, response_note = build_response(cfg, self.geometry)
        self.components = {
            "perception": f"{ai_config.engine}" + (" (MOCK fixture replay)" if ai_config.engine == "mock" else ""),
            "behaviour": f"{ai_config.behavior} (risk weight {cfg.risk.weights.behavior})",
            "geometry": geometry_note,
            "response": response_note,
            "recording": str(cfg.recording.out_dir) if self.clips.enabled else "disabled",
            "incident_log": str(cfg.alerts.incident_db) if cfg.alerts.incident_db else "in memory",
        }
        if ai_config.pose.enabled:  # listed only when configured, so pose-off sessions report as before
            status = getattr(self.engine, "pose_status", None) or "not supported by this engine"
            self.components["pose"] = f"{status} (display only; not used by the risk engine)"
        for name, value in self.components.items():
            logger.info("component %-13s %s", name, value)
        self._lock = threading.RLock()
        self._c = _Counters()
        self._frame: np.ndarray | None = None
        self._result: FrameResult | None = None
        self._health: list[SystemHealth] = []
        self._notes: deque[Notification] = deque(maxlen=_HISTORY)
        self._undelivered: list[Notification] = []
        self._outcomes: deque[ResponseOutcome] = deque(maxlen=_HISTORY)
        self._clip_owner: dict[Path, str] = {}
        self._frozen_check = True

    # ---------------------------------------------------------------- frames

    def process(self, frame: VideoFrame) -> list[RiskEvent]:
        """Run one frame through the whole system; returns the risk transitions it caused."""
        with self._lock:
            t = frame.t_capture
            self._c.last_t = t
            self.watchdog.on_frame(t, frame.image if self._frozen_check else None)
            for path in self.clips.on_frame(t, frame.image):
                self._attach_clip(path, frame.t_utc)

            result = self.engine.process(frame.image, frame.frame_idx, t, frame.t_utc, frame.t_media)
            self.watchdog.on_result(result)
            self._count(result)
            self.alerts.on_frame(result)
            self.response.on_frame(result)

            events = self.risk.update(result)
            self._add_notes(self.alerts.handle(events))
            for event in events:
                self._c.summary.risk_events += 1
                self._outcomes.extend(self.response.on_risk_event(event))
                if event.to_level == RiskLevel.ALARM:
                    self._c.summary.alarms += 1
                    found = self.alerts.incident_for_event(event.event_id)
                    if found is not None:
                        path = self.clips.trigger(found[0], t)
                        if path is not None:
                            self._clip_owner[path] = found[0]
            self._add_notes(self.alerts.tick(t, frame.t_utc))
            self._outcomes.extend(self.response.step(t))
            self._health_check(t, frame.t_utc)
            self._frame = frame.image
            self._result = result
            return events

    def idle(self, t: float | None = None, t_utc: str | None = None) -> None:
        """No new frame: keep health checks, alert repeats and the response platform running."""
        with self._lock:
            t = time.monotonic() if t is None else t
            t_utc = t_utc or utc_now()
            self._add_notes(self.alerts.tick(t, t_utc))
            self._outcomes.extend(self.response.step(t))
            self._health_check(t, t_utc, force=True)

    def attach_source(self, source: object) -> None:
        """Start monitoring a source. Synthetic sources disable the frozen-frame check (black frames)."""
        with self._lock:
            self._frozen_check = bool(getattr(source, "frozen_check", True))
            self.watchdog.start(time.monotonic())

    def run(self, source: Iterable[VideoFrame | None], stop: threading.Event | None = None,
            realtime_fps: float | None = None) -> RunSummary:
        """Consume a source until it ends or ``stop`` is set. ``realtime_fps`` paces file replay."""
        self.attach_source(source)
        next_due = time.monotonic()
        for item in source:
            if stop is not None and stop.is_set():
                break
            if item is None:
                self.idle()
                continue
            self.process(item)
            if realtime_fps:
                next_due += 1.0 / realtime_fps
                time.sleep(max(0.0, next_due - time.monotonic()))
        return self._c.summary

    # ------------------------------------------------------- operator actions

    def acknowledge(self, event_id: str, decision: AckDecision, by: str) -> tuple[AckResult, ResponseOutcome | None]:
        """A lifeguard decision on an incident. A CONFIRM on an ALARM dispatches the response platform."""
        with self._lock:
            found = self.alerts.incident_for_event(event_id)
            track_uid = found[1] if found is not None else "unknown:unknown:1"
            ack = Acknowledgement(event_id=event_id, track_uid=track_uid, decision=decision, by=by,
                                  t_utc=utc_now())
            result = self.alerts.acknowledge(ack)
            self._add_notes(result.notifications)
            outcome = None
            for event in result.events:
                self._outcomes.extend(self.response.on_risk_event(event))
            if result.dispatch is not None:
                people = [
                    t.pool_xy for t in (self._result.tracks if self._result else ())
                    if t.in_water and t.pool_xy is not None and t.track_uid != result.dispatch.track_uid
                ]
                outcome = self.response.dispatch(result.dispatch, self._c.last_t, people)
                self._outcomes.append(outcome)
                logger.info("response: %s", outcome.detail)
            return result, outcome

    def confirm_release(self, by: str) -> ResponseOutcome | None:
        """Human confirmation of the simulated flotation release (drone platform only)."""
        with self._lock:
            outcome = self.response.confirm_release(by)
            if outcome is not None:
                self._outcomes.append(outcome)
            return outcome

    def snapshot(self) -> Snapshot:
        with self._lock:
            frame = self._frame
            if frame is not None and frame.shape[1] > _PREVIEW_WIDTH:
                step = int(np.ceil(frame.shape[1] / _PREVIEW_WIDTH))
                frame = frame[::step, ::step]
            return Snapshot(
                frame=None if frame is None else frame.copy(),
                result=self._result,
                levels=self.risk.levels(),
                open_incidents=self.alerts.open_incidents(),
                notifications=list(self._notes),
                outcomes=list(self._outcomes),
                health=list(self._health),
                components=dict(self.components),
                summary=self._c.summary,
                response_status=self.response.status(),
            )

    def close(self) -> None:
        """Flush clips and close the incident log and the response platform."""
        with self._lock:
            for path in self.clips.close():
                self._attach_clip(path, utc_now())
            self.response.close()
            self.store.close()

    # --------------------------------------------------------------- helpers

    def _count(self, result: FrameResult) -> None:
        s = self._c.summary
        s.frames += 1
        if result.status == Status.ERROR:
            s.error += 1
        elif result.status == Status.DEGRADED:
            s.degraded += 1
        else:
            s.ok += 1
        s.track_uids.update(t.track_uid for t in result.tracks)
        for sub in result.submersions:
            s.max_submersion_s = max(s.max_submersion_s, sub.seconds_submerged)

    def _health_check(self, t: float, t_utc: str, force: bool = False) -> None:
        last = self._c.last_health_t
        if not force and last is not None and t - last < _HEALTH_PERIOD_S:
            return
        self._c.last_health_t = t
        self._health = self.watchdog.check(t, t_utc)
        self._add_notes(self.alerts.on_health(self._health))

    def _attach_clip(self, path: Path, t_utc: str) -> None:
        owner = self._clip_owner.pop(path, None)
        if owner is not None:
            self.alerts.attach_clip(owner, str(path), t_utc)

    def _add_notes(self, notes: Iterable[Notification]) -> None:
        for note in notes:
            self._notes.append(note)
            self._undelivered.append(note)

    def drain_notifications(self) -> list[Notification]:
        """Notifications not yet delivered by a headless caller (each returned once)."""
        with self._lock:
            out, self._undelivered = self._undelivered, []
            return out
