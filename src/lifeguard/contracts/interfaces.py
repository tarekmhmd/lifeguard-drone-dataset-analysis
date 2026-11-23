"""Abstract interfaces shared across lifeguard modules."""
from abc import ABC, abstractmethod

import numpy as np

from lifeguard.contracts.enums import Zone
from lifeguard.contracts.types import Acknowledgement, EngineHeartbeat, FrameResult, RiskEvent


class PoolGeometry(ABC):
    """Maps between image pixels and pool coordinates."""

    @property
    @abstractmethod
    def site_id(self) -> str:
        """Identifier of the pool site."""

    @property
    @abstractmethod
    def length_m(self) -> float:
        """Pool length in metres."""

    @property
    @abstractmethod
    def width_m(self) -> float:
        """Pool width in metres."""

    @abstractmethod
    def to_pool(self, u: float, v: float) -> tuple[float, float] | None:
        """Image pixel -> pool metres; None if the pixel maps outside the calibrated area."""

    @abstractmethod
    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """Pool metres -> image pixel; None if the result is outside the image."""

    @abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the pool polygon (boundary counts as inside)."""

    @abstractmethod
    def zone(self, x: float, y: float) -> Zone | None:
        """None outside the pool; EDGE within the edge band; otherwise SHALLOW or DEEP."""

    @abstractmethod
    def is_interior(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        """Inside the pool AND at least margin_m from every pool edge."""

    @abstractmethod
    def near_exit(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        """True if within margin_m of any exit point (ladder/steps)."""

    @abstractmethod
    def to_local_ned(self, x: float, y: float, altitude_m: float) -> tuple[float, float, float]:
        """Pool metres + altitude above water -> drone local NED (north, east, down)."""


class PerceptionEngine(ABC):
    """Processes frames and returns FrameResult observations."""

    @abstractmethod
    def process(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        t_capture: float,
        t_utc: str,
        t_media: float | None = None,
    ) -> FrameResult:
        """Process one frame; MUST NOT raise, returning an error result on failure."""

    @abstractmethod
    def reset(self) -> None:
        """Clear tracks, buffers and counters (start of a new session)."""

    @abstractmethod
    def heartbeat(self) -> EngineHeartbeat:
        """Return engine health and counters."""

    @property
    @abstractmethod
    def cam_id(self) -> str:
        """Camera identifier."""

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Session identifier."""


class RiskEngine(ABC):
    """Turns FrameResult streams into RiskEvent transitions (Member 2, later phase)."""

    @abstractmethod
    def update(self, frame_result: FrameResult) -> list[RiskEvent]:
        """Consume one FrameResult and return any risk-level transitions."""

    @abstractmethod
    def acknowledge(self, ack: Acknowledgement) -> list[RiskEvent]:
        """Apply a lifeguard acknowledgement and return any resulting transitions."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the risk engine state."""
