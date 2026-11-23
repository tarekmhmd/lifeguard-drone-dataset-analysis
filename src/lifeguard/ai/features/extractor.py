"""Per-track motion and visibility features over a sliding window (Blueprint M6).

Feature names written to ``TrackObs.features`` (all finite floats; a feature is omitted when there is not
enough data). They describe motion only; they are inputs to the temporal and risk layers, not a decision.

| name | definition |
| --- | --- |
| ``speed_mps`` | net pool displacement / time span over visible in-water samples |
| ``path_speed_mps`` | path length / time span over the same samples |
| ``progress_ratio`` | net displacement / path length (near 0 = moving in place) |
| ``bob_std_norm`` | std of the box-centre y divided by the mean box height (vertical bobbing) |
| ``bob_freq_hz`` | dominant frequency of the box-centre y oscillation (needs >= 8 visible samples over >= 2 s) |
| ``visibility_ratio`` | fraction of window samples in which the track was visible |
| ``time_since_seen_s`` | seconds since the track was last visible |
| ``box_aspect`` | box height / width of the current box |

The authoritative, versioned description is ``feature_spec.json`` in this package (M1-03), loaded and
cross-checked by ``lifeguard.ai.features.spec``.
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from lifeguard.ai.features.config import FeatureConfig

_MIN_BOB_SAMPLES = 3
_MIN_BOB_NORM = 0.02       # below this the box is essentially still: no meaningful bobbing frequency
_MIN_FREQ_SAMPLES = 8
_MIN_FREQ_SPAN_S = 2.0
_FFT_PAD = 8               # zero-padding factor for a finer frequency estimate


@dataclass(frozen=True)
class _Sample:
    t: float
    visible: bool
    pool_xy: tuple[float, float] | None
    in_water: bool
    box_xyxy: tuple[float, float, float, float]


class FeatureExtractor:
    """Ring buffer per track ID; ``observe`` returns the feature dict for the current frame."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._config = config or FeatureConfig()
        self._buffers: dict[int, deque[_Sample]] = {}
        self._last_visible: dict[int, float] = {}

    def reset(self) -> None:
        self._buffers.clear()
        self._last_visible.clear()

    @property
    def known_ids(self) -> set[int]:
        """Track IDs that currently have a buffer."""
        return set(self._buffers)

    def forget(self, track_ids: set[int]) -> None:
        """Drop buffers of tracks that no longer exist."""
        for track_id in track_ids:
            self._buffers.pop(track_id, None)
            self._last_visible.pop(track_id, None)

    def inherit(self, new_id: int, old_ids: tuple[int, ...]) -> None:
        """Start ``new_id``'s buffer from the most recent old buffer (re-associated swimmer)."""
        if new_id in self._buffers:
            return
        for old in old_ids:
            if old in self._buffers:
                self._buffers[new_id] = self._buffers.pop(old)
                if old in self._last_visible:
                    self._last_visible[new_id] = self._last_visible.pop(old)
                return

    def observe(
        self,
        track_id: int,
        t: float,
        visible: bool,
        pool_xy: tuple[float, float] | None,
        in_water: bool,
        box_xyxy: tuple[float, float, float, float],
    ) -> dict[str, float]:
        buf = self._buffers.setdefault(track_id, deque())
        buf.append(_Sample(t, visible, pool_xy, in_water, box_xyxy))
        while buf and buf[0].t < t - self._config.window_s:
            buf.popleft()
        if visible:
            self._last_visible[track_id] = t

        features: dict[str, float] = {
            "visibility_ratio": sum(1 for s in buf if s.visible) / len(buf),
            "time_since_seen_s": t - self._last_visible.get(track_id, t),
        }
        x1, y1, x2, y2 = box_xyxy
        if x2 > x1:
            features["box_aspect"] = (y2 - y1) / (x2 - x1)

        moving = [s for s in buf if s.visible and s.in_water and s.pool_xy is not None]
        if len(moving) >= 2 and moving[-1].t - moving[0].t >= self._config.min_span_s:
            span = moving[-1].t - moving[0].t
            path = sum(
                math.dist(a.pool_xy, b.pool_xy)  # type: ignore[arg-type]
                for a, b in itertools.pairwise(moving)
            )
            net = math.dist(moving[0].pool_xy, moving[-1].pool_xy)  # type: ignore[arg-type]
            features["speed_mps"] = net / span
            features["path_speed_mps"] = path / span
            if path > 0:
                features["progress_ratio"] = min(1.0, net / path)

        seen = [s for s in buf if s.visible]
        if len(seen) >= _MIN_BOB_SAMPLES:
            centres = np.array([(s.box_xyxy[1] + s.box_xyxy[3]) / 2.0 for s in seen])
            heights = np.array([s.box_xyxy[3] - s.box_xyxy[1] for s in seen])
            mean_h = float(heights.mean())
            if mean_h > 0:
                features["bob_std_norm"] = float(centres.std()) / mean_h
                if features["bob_std_norm"] >= _MIN_BOB_NORM:
                    freq = _dominant_frequency([s.t for s in seen], centres)
                    if freq is not None:
                        features["bob_freq_hz"] = freq

        return {k: float(v) for k, v in features.items() if math.isfinite(v)}


def _dominant_frequency(times: list[float], values: np.ndarray) -> float | None:
    """Dominant oscillation frequency (Hz) of an irregularly sampled signal; None if too little data.

    The signal is resampled on a uniform grid (median sample spacing), de-meaned, zero-padded and
    transformed with a real FFT; the strongest non-DC bin wins. At least one full cycle must fit in the span.
    """
    if len(times) < _MIN_FREQ_SAMPLES:
        return None
    t = np.asarray(times, dtype=float)
    span = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t)))
    if span < _MIN_FREQ_SPAN_S or dt <= 0:
        return None
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    signal = np.interp(grid, t, values) - float(np.mean(values))
    n = len(signal) * _FFT_PAD
    spectrum = np.abs(np.fft.rfft(signal, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)
    valid = freqs >= 1.0 / span
    if not np.any(valid):
        return None
    return float(freqs[valid][np.argmax(spectrum[valid])])
