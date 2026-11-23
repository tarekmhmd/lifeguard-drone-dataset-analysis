"""Deterministic rectangular pool implementing the PoolGeometry interface."""
import math

from lifeguard.contracts.enums import Zone
from lifeguard.contracts.interfaces import PoolGeometry


def _require_finite(name: str, *values: float) -> None:
    """Raise ValueError if any coordinate is NaN or infinite."""
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} requires finite coordinates")


class StubPoolGeometry(PoolGeometry):
    """A rectangular pool with an affine pixel mapping and simple zones."""

    def __init__(
        self,
        length_m: float = 25.0,
        width_m: float = 12.5,
        image_width: int = 1920,
        image_height: int = 1080,
        margin_frac: float = 0.1,
        deep_from_x_m: float | None = None,
        edge_band_m: float = 1.0,
        exits_m: tuple[tuple[float, float], ...] | None = None,
        site_id: str = "stub_pool",
        ned_yaw_deg: float = 0.0,
        ned_north_offset_m: float = 0.0,
        ned_east_offset_m: float = 0.0,
    ) -> None:
        """Build the stub pool.

        Args:
            length_m: Pool length in metres.
            width_m: Pool width in metres.
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            margin_frac: Fraction of the image left as a margin around the pool.
            deep_from_x_m: X beyond which the pool is DEEP; defaults to half the length.
            edge_band_m: Distance from the pool edge counted as EDGE.
            exits_m: Exit (ladder/steps) points in pool metres.
            site_id: Site identifier.
            ned_yaw_deg: Yaw rotation of the pool frame in the drone NED frame.
            ned_north_offset_m: North offset of the pool origin.
            ned_east_offset_m: East offset of the pool origin.
        """
        _require_finite(
            "StubPoolGeometry",
            length_m,
            width_m,
            margin_frac,
            edge_band_m,
            ned_yaw_deg,
            ned_north_offset_m,
            ned_east_offset_m,
        )
        if length_m <= 0:
            raise ValueError("length_m must be > 0")
        if width_m <= 0:
            raise ValueError("width_m must be > 0")
        if image_width < 2:
            raise ValueError("image_width must be >= 2")
        if image_height < 2:
            raise ValueError("image_height must be >= 2")
        if not 0 <= margin_frac < 0.5:
            raise ValueError("margin_frac must be in [0, 0.5)")
        if edge_band_m < 0:
            raise ValueError("edge_band_m must be >= 0")
        if deep_from_x_m is not None:
            _require_finite("deep_from_x_m", deep_from_x_m)

        self._site_id = site_id
        self._length_m = length_m
        self._width_m = width_m
        self._image_width = image_width
        self._image_height = image_height
        self._deep_from_x_m = length_m / 2.0 if deep_from_x_m is None else deep_from_x_m
        self._edge_band_m = edge_band_m
        self._exits_m = ((0.0, 2.0), (length_m, width_m - 2.0)) if exits_m is None else exits_m
        self._ned_yaw_deg = ned_yaw_deg
        self._ned_north_offset_m = ned_north_offset_m
        self._ned_east_offset_m = ned_east_offset_m

        self._u0 = margin_frac * image_width
        self._u1 = (1.0 - margin_frac) * image_width
        self._v0 = margin_frac * image_height
        self._v1 = (1.0 - margin_frac) * image_height

    @property
    def site_id(self) -> str:
        """Site identifier."""
        return self._site_id

    @property
    def length_m(self) -> float:
        """Pool length in metres."""
        return self._length_m

    @property
    def width_m(self) -> float:
        """Pool width in metres."""
        return self._width_m

    @property
    def image_width(self) -> int:
        """Image width in pixels."""
        return self._image_width

    @property
    def image_height(self) -> int:
        """Image height in pixels."""
        return self._image_height

    def to_pool(self, u: float, v: float) -> tuple[float, float] | None:
        """Inverse of :meth:`to_image`; None if the result is outside the pool."""
        _require_finite("to_pool", u, v)
        x = (u - self._u0) / (self._u1 - self._u0) * self._length_m
        y = (v - self._v0) / (self._v1 - self._v0) * self._width_m
        if not (0 <= x <= self._length_m and 0 <= y <= self._width_m):
            return None
        return (x, y)

    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        """Map pool metres to pixels; None if the result is outside the image."""
        _require_finite("to_image", x, y)
        u = self._u0 + x / self._length_m * (self._u1 - self._u0)
        v = self._v0 + y / self._width_m * (self._v1 - self._v0)
        if not (0 <= u <= self._image_width and 0 <= v <= self._image_height):
            return None
        return (u, v)

    def contains(self, x: float, y: float) -> bool:
        """True if the point is inside the pool rectangle (boundary included)."""
        _require_finite("contains", x, y)
        return 0 <= x <= self._length_m and 0 <= y <= self._width_m

    def zone(self, x: float, y: float) -> Zone | None:
        """EDGE near the border, DEEP beyond deep_from_x_m, otherwise SHALLOW."""
        _require_finite("zone", x, y)
        if not self.contains(x, y):
            return None
        distance_to_edge = min(x, self._length_m - x, y, self._width_m - y)
        if distance_to_edge < self._edge_band_m:
            return Zone.EDGE
        if x >= self._deep_from_x_m:
            return Zone.DEEP
        return Zone.SHALLOW

    def is_interior(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        """True if inside the pool and at least margin_m from every edge."""
        _require_finite("is_interior", x, y, margin_m)
        if not self.contains(x, y):
            return False
        return min(x, self._length_m - x, y, self._width_m - y) >= margin_m

    def near_exit(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        """True if within margin_m of any exit point."""
        _require_finite("near_exit", x, y, margin_m)
        return any(math.hypot(x - ex, y - ey) <= margin_m for ex, ey in self._exits_m)

    def to_local_ned(self, x: float, y: float, altitude_m: float) -> tuple[float, float, float]:
        """Map pool metres plus altitude to drone local NED (north, east, down)."""
        _require_finite("to_local_ned", x, y, altitude_m)
        theta = math.radians(self._ned_yaw_deg)
        n = x * math.cos(theta) - y * math.sin(theta) + self._ned_north_offset_m
        e = x * math.sin(theta) + y * math.cos(theta) + self._ned_east_offset_m
        return (n, e, -altitude_m)
