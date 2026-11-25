"""ByteTrack-style multi-object tracker in pure NumPy (roadmap P4-4, Blueprint M5).

Follows ByteTrack (Zhang et al., ECCV 2022): Kalman prediction, a first association of high-confidence
detections with tracked *and* lost tracks, then a second association of low-confidence detections with
the remaining tracked tracks, so partially submerged swimmers keep their ID.

Deliberate differences from the reference implementation (documented, pool-specific):
- assignment is greedy by descending IoU instead of the Hungarian algorithm (no SciPy/lap dependency);
- a track that becomes lost has its velocity set to zero: a swimmer who goes under stays where they went
  under, so the lost box is not extrapolated across the pool where it could capture another swimmer;
- reported boxes are the last matched detection boxes; the Kalman state is used only for association.

Track IDs are integers starting at 1 and are never reused within a session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lifeguard.ai.detection.base import Detection
from lifeguard.ai.tracking.config import TrackerConfig
from lifeguard.ai.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah
from lifeguard.contracts.enums import DetClass, TrackState

_TENTATIVE = "tentative"
_TRACKED = "tracked"
_LOST = "lost"
_MIN_HEIGHT_PX = 1.0
_MIN_ASPECT = 1e-3


@dataclass(frozen=True)
class TrackedBox:
    """Tracker output for one person in one frame."""

    track_id: int
    box_xyxy: tuple[float, float, float, float]
    conf: float
    cls: DetClass
    state: TrackState
    first_seen_t: float
    last_seen_t: float
    distress: float = 0.0   # Detection.distress of the last matched detection


class _Track:
    """Internal mutable track."""

    def __init__(self, det: Detection, t: float, kf: KalmanFilterXYAH) -> None:
        self.track_id: int | None = None
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(det.box_xyxy))
        self.box = det.box_xyxy
        self.conf = det.conf
        self.cls = det.cls
        self.distress = det.distress
        self.status = _TENTATIVE
        self.hits = 1
        self.first_seen_t = t
        self.last_seen_t = t

    def predicted_box(self) -> tuple[float, float, float, float]:
        return xyah_to_xyxy(self.mean)


def _valid_box(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return all(np.isfinite(box)) and x2 > x1 and y2 > y1


def iou_matrix(
    a: list[tuple[float, float, float, float]], b: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """Pairwise IoU between two lists of xyxy boxes, shape (len(a), len(b))."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    aa = np.asarray(a, dtype=float)[:, None, :]
    bb = np.asarray(b, dtype=float)[None, :, :]
    ix1 = np.maximum(aa[..., 0], bb[..., 0])
    iy1 = np.maximum(aa[..., 1], bb[..., 1])
    ix2 = np.minimum(aa[..., 2], bb[..., 2])
    iy2 = np.minimum(aa[..., 3], bb[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (aa[..., 2] - aa[..., 0]) * (aa[..., 3] - aa[..., 1])
    area_b = (bb[..., 2] - bb[..., 0]) * (bb[..., 3] - bb[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def _greedy_match(
    tracks: list[_Track], dets: list[Detection], min_iou: float
) -> tuple[list[tuple[_Track, Detection]], list[_Track], list[Detection]]:
    """Greedy one-to-one assignment by descending IoU (ties broken by input order)."""
    ious = iou_matrix([t.predicted_box() for t in tracks], [d.box_xyxy for d in dets])
    candidates = sorted(
        ((float(ious[i, j]), i, j) for i in range(len(tracks)) for j in range(len(dets))
         if ious[i, j] >= min_iou),
        key=lambda c: (-c[0], c[1], c[2]),
    )
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[_Track, Detection]] = []
    for _, i, j in candidates:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        matches.append((tracks[i], dets[j]))
    rest_t = [t for i, t in enumerate(tracks) if i not in used_t]
    rest_d = [d for j, d in enumerate(dets) if j not in used_d]
    return matches, rest_t, rest_d


class ByteTracker:
    """Detections per frame -> persistent track IDs. ``t`` is the frame's ``t_capture`` (seconds)."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        self._kf = KalmanFilterXYAH()
        self._tracks: list[_Track] = []
        self._next_id = 1

    @property
    def config(self) -> TrackerConfig:
        return self._config

    def reset(self) -> None:
        """Forget every track and restart IDs at 1."""
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection], t: float) -> list[TrackedBox]:
        """Associate one frame of detections and return tracked and recently lost tracks."""
        cfg = self._config
        for trk in self._tracks:
            trk.mean, trk.cov = self._kf.predict(trk.mean, trk.cov)
            trk.mean[3] = max(trk.mean[3], _MIN_HEIGHT_PX)
            trk.mean[2] = max(trk.mean[2], _MIN_ASPECT)

        dets = [d for d in detections if d.conf >= cfg.low_thresh and _valid_box(d.box_xyxy)]
        high = [d for d in dets if d.conf >= cfg.high_thresh]
        low = [d for d in dets if d.conf < cfg.high_thresh]

        confirmed = [t_ for t_ in self._tracks if t_.status in (_TRACKED, _LOST)]
        tentative = [t_ for t_ in self._tracks if t_.status == _TENTATIVE]

        # Stage 1: high detections vs tracked + lost tracks.
        matches, rest_tracks, rest_high = _greedy_match(confirmed, high, cfg.match_iou)
        for trk, det in matches:
            self._apply(trk, det, t)

        # Stage 2: low detections vs still-unmatched *tracked* tracks.
        still_tracked = [t_ for t_ in rest_tracks if t_.status == _TRACKED]
        matches2, unmatched_tracked, _ = _greedy_match(still_tracked, low, cfg.low_match_iou)
        for trk, det in matches2:
            self._apply(trk, det, t)
        for trk in unmatched_tracked:
            trk.status = _LOST
            trk.mean[4:] = 0.0  # a swimmer who goes under stays where they went under

        # Stage 3: tentative tracks vs leftover high detections.
        matches3, unmatched_tentative, rest_high = _greedy_match(tentative, rest_high, cfg.tentative_iou)
        for trk, det in matches3:
            self._apply(trk, det, t)
            if trk.hits >= cfg.min_hits:
                self._confirm(trk)
        dropped = {id(t_) for t_ in unmatched_tentative}

        # New tracks from confident leftover detections.
        for det in rest_high:
            if det.conf < cfg.new_track_thresh:
                continue
            trk = _Track(det, t, self._kf)
            if cfg.min_hits <= 1:
                self._confirm(trk)
            self._tracks.append(trk)

        # Forget tentative misses and lost tracks beyond the buffer.
        self._tracks = [
            t_ for t_ in self._tracks
            if id(t_) not in dropped
            and not (t_.status == _LOST and t - t_.last_seen_t > cfg.track_buffer_s)
        ]
        return self._output(t)

    def _apply(self, trk: _Track, det: Detection, t: float) -> None:
        trk.mean, trk.cov = self._kf.update(trk.mean, trk.cov, xyxy_to_xyah(det.box_xyxy))
        trk.box = det.box_xyxy
        trk.conf = det.conf
        trk.cls = det.cls
        trk.distress = det.distress
        trk.hits += 1
        trk.last_seen_t = t
        if trk.status == _LOST:
            trk.status = _TRACKED

    def _confirm(self, trk: _Track) -> None:
        trk.status = _TRACKED
        trk.track_id = self._next_id
        self._next_id += 1

    def _output(self, t: float) -> list[TrackedBox]:
        out: list[TrackedBox] = []
        for trk in self._tracks:
            if trk.track_id is None:
                continue
            if trk.status == _TRACKED:
                state = TrackState.TRACKED
            elif trk.status == _LOST and t - trk.last_seen_t <= self._config.lost_emit_s:
                state = TrackState.LOST
            else:
                continue
            out.append(TrackedBox(
                track_id=trk.track_id,
                box_xyxy=trk.box,
                conf=trk.conf,
                cls=trk.cls,
                state=state,
                first_seen_t=trk.first_seen_t,
                last_seen_t=trk.last_seen_t,
                distress=trk.distress,
            ))
        return sorted(out, key=lambda b: b.track_id)
