"""Calibrated pool geometry from a planar homography (Blueprint M2, roadmap P4-2).

REAL implementation. The water surface is a plane, so image pixels map to pool metres with a 3x3
homography H estimated (normalised DLT) from >= 4 correspondences, usually the four pool corners:
``(0, 0) -> (L, 0) -> (L, W) -> (0, W)`` in pool metres (decision C-18: origin at corner 0, X along the
length, Y across). Pool-space queries (zones, interior, exits, pool -> drone NED) reuse the same rules as
the stub, but on the real rectangle size.

A calibration file is validated before use: matching point counts, >= 4 non-collinear points inside the
image, a non-singular H, and (when check points such as lane marks are given) a reprojection error below
``max_reprojection_error_m`` (Blueprint M2 acceptance: 0.3 m). The pool must be rectangular; other shapes
need a polygon extension (documented limitation).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lifeguard.contracts.enums import Zone
from lifeguard.contracts.errors import ConfigError
from lifeguard.contracts.interfaces import PoolGeometry
from lifeguard.system.geometry.stub import StubPoolGeometry

FORMAT = "lifeguard-pool-calibration"
FORMAT_VERSION = 1
_MIN_POINTS = 4
_EPS = 1e-9

Point = tuple[float, float]


def _normaliser(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    scale = math.sqrt(2.0) / max(np.sqrt(((points - centre) ** 2).sum(axis=1)).mean(), _EPS)
    return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])


def homography_from_points(src: list[Point], dst: list[Point]) -> np.ndarray:
    """H with ``dst ~ H @ src`` (normalised DLT, least squares for > 4 points).

    Raises:
        ValueError: fewer than 4 points, mismatched counts, collinear points or a singular result.
    """
    if len(src) != len(dst) or len(src) < _MIN_POINTS:
        raise ValueError("need at least 4 matching point pairs")
    a = np.asarray(src, dtype=float)
    b = np.asarray(dst, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("points must be finite")
    for pts in (a, b):
        centred = pts - pts.mean(axis=0)
        if np.linalg.matrix_rank(centred, tol=1e-6 * max(1.0, float(np.abs(centred).max()))) < 2:
            raise ValueError("points are collinear or coincident")
    ta, tb = _normaliser(a), _normaliser(b)
    ah = (ta @ np.c_[a, np.ones(len(a))].T).T
    bh = (tb @ np.c_[b, np.ones(len(b))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(ah, bh):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tb) @ h @ ta
    if abs(np.linalg.det(h)) < _EPS or abs(h[2, 2]) < _EPS:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, point: Point) -> Point | None:
    """Map one point; None when it maps to infinity (behind the camera horizon)."""
    x, y, w = h @ np.array([point[0], point[1], 1.0])
    if abs(w) < _EPS:
        return None
    return (float(x / w), float(y / w))


class CheckPoint(BaseModel):
    """A known mark (e.g. a lane-line crossing) used to measure the reprojection error."""

    model_config = ConfigDict(extra="forbid")

    image: Point
    pool: Point


class PoolCalibration(BaseModel):
    """Persisted calibration of one camera over one rectangular pool."""

    model_config = ConfigDict(extra="forbid")

    format: str = FORMAT
    format_version: int = FORMAT_VERSION
    site_id: str = Field(min_length=1, max_length=64)
    image_width: int = Field(ge=2)
    image_height: int = Field(ge=2)
    length_m: float = Field(gt=0)
    width_m: float = Field(gt=0)
    image_points: list[Point]
    pool_points: list[Point] | None = None       # default: the four corners in C-18 order
    deep_from_x_m: float | None = None           # default: half the length
    edge_band_m: float = Field(1.0, ge=0)
    exits_m: list[Point] = Field(default_factory=list)
    check_points: list[CheckPoint] = Field(default_factory=list)
    max_reprojection_error_m: float = Field(0.3, gt=0)
    ned_yaw_deg: float = 0.0
    ned_north_offset_m: float = 0.0
    ned_east_offset_m: float = 0.0

    @property
    def corners(self) -> list[Point]:
        return self.pool_points or [(0.0, 0.0), (self.length_m, 0.0), (self.length_m, self.width_m),
                                    (0.0, self.width_m)]

    @model_validator(mode="after")
    def _validate(self) -> PoolCalibration:
        if self.format != FORMAT or self.format_version != FORMAT_VERSION:
            raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} file")
        if len(self.image_points) != len(self.corners):
            raise ValueError("image_points and pool_points must have the same length (4 corners by default)")
        for u, v in self.image_points:
            if not (0 <= u <= self.image_width and 0 <= v <= self.image_height):
                raise ValueError(f"image point ({u}, {v}) lies outside the {self.image_width}x{self.image_height} image")
        h = homography_from_points(self.image_points, self.corners)
        error = reprojection_error_m(h, [(c.image, c.pool) for c in self.check_points])
        if error is not None and error > self.max_reprojection_error_m:
            raise ValueError(f"reprojection error {error:.3f} m exceeds {self.max_reprojection_error_m} m")
        return self

    def homography(self) -> np.ndarray:
        """Image pixels -> pool metres."""
        return homography_from_points(self.image_points, self.corners)


def reprojection_error_m(h: np.ndarray, pairs: list[tuple[Point, Point]]) -> float | None:
    """Largest distance (m) between projected image points and their known pool positions; None if no pairs."""
    if not pairs:
        return None
    worst = 0.0
    for image_pt, pool_pt in pairs:
        mapped = apply_homography(h, image_pt)
        worst = max(worst, math.inf if mapped is None else math.dist(mapped, pool_pt))
    return worst


def save_calibration(path: Path | str, calibration: PoolCalibration) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(calibration.model_dump_json(indent=2), encoding="utf-8", newline="\n")


def load_calibration(path: Path | str) -> PoolCalibration:
    """Load and validate a calibration file. Raises ConfigError naming the file."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        return PoolCalibration.model_validate(data)
    except FileNotFoundError as err:
        raise ConfigError(f"calibration file not found: {source}") from err
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise ConfigError(f"cannot read calibration file {source}: {err}") from err
    except ValidationError as err:
        raise ConfigError(f"invalid calibration in {source}: {err}") from err


class CalibratedPoolGeometry(PoolGeometry):
    """``PoolGeometry`` backed by a validated calibration."""

    def __init__(self, calibration: PoolCalibration) -> None:
        self.calibration = calibration
        self._h = calibration.homography()
        self._h_inv = np.linalg.inv(self._h)
        self._pool = StubPoolGeometry(
            length_m=calibration.length_m,
            width_m=calibration.width_m,
            image_width=calibration.image_width,
            image_height=calibration.image_height,
            deep_from_x_m=calibration.deep_from_x_m,
            edge_band_m=calibration.edge_band_m,
            # No configured exits means none: never assume ladder positions for a real pool (an assumed
            # exit would suppress submersion timers near it).
            exits_m=tuple(calibration.exits_m),
            site_id=calibration.site_id,
            ned_yaw_deg=calibration.ned_yaw_deg,
            ned_north_offset_m=calibration.ned_north_offset_m,
            ned_east_offset_m=calibration.ned_east_offset_m,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> CalibratedPoolGeometry:
        return cls(load_calibration(path))

    @property
    def site_id(self) -> str:
        return self.calibration.site_id

    @property
    def length_m(self) -> float:
        return self.calibration.length_m

    @property
    def width_m(self) -> float:
        return self.calibration.width_m

    @property
    def image_width(self) -> int:
        return self.calibration.image_width

    @property
    def image_height(self) -> int:
        return self.calibration.image_height

    def to_pool(self, u: float, v: float) -> tuple[float, float] | None:
        mapped = apply_homography(self._h, (u, v))
        if mapped is None or not self._pool.contains(*mapped):
            return None
        return mapped

    def to_image(self, x: float, y: float) -> tuple[float, float] | None:
        mapped = apply_homography(self._h_inv, (x, y))
        if mapped is None:
            return None
        u, v = mapped
        if not (0 <= u <= self.image_width and 0 <= v <= self.image_height):
            return None
        return mapped

    def contains(self, x: float, y: float) -> bool:
        return self._pool.contains(x, y)

    def zone(self, x: float, y: float) -> Zone | None:
        return self._pool.zone(x, y)

    def is_interior(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        return self._pool.is_interior(x, y, margin_m)

    def near_exit(self, x: float, y: float, margin_m: float = 1.0) -> bool:
        return self._pool.near_exit(x, y, margin_m)

    def to_local_ned(self, x: float, y: float, altitude_m: float) -> tuple[float, float, float]:
        return self._pool.to_local_ned(x, y, altitude_m)
