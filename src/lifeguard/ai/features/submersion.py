"""Submersion detector (Blueprint M6, rule-based, no training).

A timer starts when a person who was last seen *in the water*, in the pool interior and away from any exit
stops being visible. It reports ``seconds_submerged = t - t_last_visible`` every frame until the person
reappears within ``reappear_radius_m`` of the disappearance point:
- the same track ID reappearing there ends the timer;
- a *new* track (first seen after the disappearance) reappearing there ends the timer and is reported
  with ``merged_from`` containing the old ID (decision C-08).

Safety choices: a track that already existed before the disappearance can never end someone else's timer
(a swimmer passing over the spot must not cancel it), and the same ID reappearing *far* from the spot does
not end the timer either (a likely ID switch). The ALARM decision itself belongs to the risk engine (C-07).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from lifeguard.ai.features.config import FeatureConfig
from lifeguard.contracts.interfaces import PoolGeometry


@dataclass(frozen=True)
class PoolTrackView:
    """What the monitor needs to know about one tracker output in one frame."""

    track_id: int
    visible: bool
    pool_xy: tuple[float, float] | None
    in_water: bool
    first_seen_t: float


@dataclass(frozen=True)
class SubmersionTimer:
    """An active timer: the person was last seen at ``xy`` at time ``start_t``."""

    track_id: int
    start_t: float
    xy: tuple[float, float]


@dataclass
class _LastSeen:
    t: float
    xy: tuple[float, float] | None
    in_water: bool


class SubmersionMonitor:
    """Per-camera submersion timers and re-association (``merged_from``) bookkeeping."""

    def __init__(self, geometry: PoolGeometry, config: FeatureConfig | None = None) -> None:
        self._geometry = geometry
        self._config = config or FeatureConfig()
        self._last_seen: dict[int, _LastSeen] = {}
        self._timers: dict[int, SubmersionTimer] = {}
        self._merged: dict[int, tuple[int, ...]] = {}

    def reset(self) -> None:
        self._last_seen.clear()
        self._timers.clear()
        self._merged.clear()

    @property
    def timers(self) -> dict[int, SubmersionTimer]:
        """Active timers by the ID of the person who went under."""
        return dict(self._timers)

    def merged_from(self, track_id: int) -> tuple[int, ...]:
        """Old IDs that ``track_id`` was re-associated with (empty if none)."""
        return self._merged.get(track_id, ())

    def update(self, t: float, views: list[PoolTrackView]) -> list[SubmersionTimer]:
        """Process one frame of tracker output; return the active timers (sorted by ID)."""
        visible = {v.track_id: v for v in views if v.visible}

        # 1. Reappearances end timers.
        for view in visible.values():
            if view.pool_xy is None:
                continue
            timer = self._closest_timer(view)
            if timer is None:
                continue
            del self._timers[timer.track_id]
            if timer.track_id != view.track_id:
                old = (timer.track_id, *self._merged.get(timer.track_id, ()))
                merged = tuple(dict.fromkeys((*self._merged.get(view.track_id, ()), *old)))
                self._merged[view.track_id] = tuple(i for i in merged if i != view.track_id)

        # 2. Disappearances start timers.
        for track_id, seen in list(self._last_seen.items()):
            if track_id in visible:
                continue
            if (
                track_id not in self._timers
                and seen.in_water
                and seen.xy is not None
                and self._starts_timer(seen.xy)
            ):
                self._timers[track_id] = SubmersionTimer(track_id, seen.t, seen.xy)
            del self._last_seen[track_id]

        # 3. Remember where every visible person was last seen.
        for view in visible.values():
            self._last_seen[view.track_id] = _LastSeen(t, view.pool_xy, view.in_water)

        return [self._timers[k] for k in sorted(self._timers)]

    def _starts_timer(self, xy: tuple[float, float]) -> bool:
        cfg = self._config
        return self._geometry.is_interior(xy[0], xy[1], cfg.interior_margin_m) and not (
            self._geometry.near_exit(xy[0], xy[1], cfg.exit_margin_m)
        )

    def _closest_timer(self, view: PoolTrackView) -> SubmersionTimer | None:
        """The nearest timer this visible track may end, or None."""
        assert view.pool_xy is not None
        best: tuple[float, SubmersionTimer] | None = None
        for timer in self._timers.values():
            eligible = timer.track_id == view.track_id or view.first_seen_t > timer.start_t
            if not eligible:
                continue
            distance = math.hypot(view.pool_xy[0] - timer.xy[0], view.pool_xy[1] - timer.xy[1])
            if distance > self._config.reappear_radius_m:
                continue
            if best is None or distance < best[0]:
                best = (distance, timer)
        return None if best is None else best[1]
