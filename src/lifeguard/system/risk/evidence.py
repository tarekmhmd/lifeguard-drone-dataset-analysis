"""Per-track evidence terms and the exponentially smoothed risk score (Blueprint M8).

The evidence terms are computed from the contract fields that exist today (pool position, track state,
zone, submersion timers, optional behaviour). When Member 1's feature extractor (roadmap P4-5) fills
``TrackObs.features``, richer terms can be added here without changing the FSM.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from lifeguard.contracts.enums import Zone
from lifeguard.system.risk.config import RiskConfig


@dataclass(frozen=True)
class Observation:
    """What the risk engine knows about one track in one frame."""

    t: float
    visible: bool
    pool_xy: tuple[float, float] | None
    in_water: bool
    zone: Zone | None
    seconds_submerged: float = 0.0
    behavior_score: float = 0.0
    detector_distress: float = 0.0


@dataclass(frozen=True)
class EvidenceTerms:
    """Evidence terms, each scaled to [0, 1]."""

    submersion: float
    static: float
    visibility: float
    deep_zone: float
    behavior: float


class TrackEvidence:
    """Sliding window of observations for one track and its smoothed risk R."""

    def __init__(self, config: RiskConfig) -> None:
        """Start with an empty window and R = 0."""
        self._config = config
        self._window: deque[Observation] = deque()
        self._last_t: float | None = None
        self.risk = 0.0
        self.last_terms: EvidenceTerms | None = None

    def observe(self, obs: Observation) -> float:
        """Add one observation, update R and return it."""
        self._window.append(obs)
        horizon = obs.t - self._config.window_s
        while self._window and self._window[0].t < horizon:
            self._window.popleft()

        terms = self._terms(obs)
        self.last_terms = terms
        r = self._instant_risk(terms)

        dt = 0.0 if self._last_t is None else max(0.0, obs.t - self._last_t)
        alpha = math.exp(-dt / self._config.tau_s)
        self.risk = min(1.0, max(0.0, alpha * self.risk + (1.0 - alpha) * r))
        self._last_t = obs.t
        return self.risk

    def _instant_risk(self, terms: EvidenceTerms) -> float:
        """Weighted mean of the evidence terms, in [0, 1]."""
        w = self._config.weights
        total = (
            w.submersion * terms.submersion
            + w.static * terms.static
            + w.visibility * terms.visibility
            + w.deep_zone * terms.deep_zone
            + w.behavior * terms.behavior
        )
        return min(1.0, max(0.0, total / w.total))

    def _terms(self, obs: Observation) -> EvidenceTerms:
        """Compute all evidence terms from the current window."""
        cfg = self._config
        submersion = min(1.0, obs.seconds_submerged / cfg.t_sub_s)
        visible = sum(1 for o in self._window if o.visible)
        visibility = 1.0 - visible / len(self._window)
        deep_zone = 1.0 if obs.in_water and obs.zone == Zone.DEEP else 0.0
        static = self._static_term() if obs.in_water else 0.0
        behavior = min(1.0, max(0.0, obs.behavior_score))
        return EvidenceTerms(submersion, static, visibility, deep_zone, behavior)

    def _static_term(self) -> float:
        """1 - min(1, net speed / reference speed) over visible in-water samples; 0 if too little data."""
        samples = [o for o in self._window if o.visible and o.in_water and o.pool_xy is not None]
        if len(samples) < 2:
            return 0.0
        first, last = samples[0], samples[-1]
        span = last.t - first.t
        if span < self._config.min_progress_span_s:
            return 0.0
        dx = last.pool_xy[0] - first.pool_xy[0]  # type: ignore[index]
        dy = last.pool_xy[1] - first.pool_xy[1]  # type: ignore[index]
        speed = math.hypot(dx, dy) / span
        return 1.0 - min(1.0, speed / self._config.progress_ref_mps)
