"""Create a pool calibration file (roadmap P4-2).

Usage (non-interactive: read the four corner pixels from any image viewer)::

    python -m lifeguard.system.geometry.calibrate --site-id my_pool --length 25 --width 12.5 \
        --image frame.png --corners "212,180;1705,175;1880,950;40,960" \
        [--exit 0,2 --exit 25,10.5] [--check "960,560,12.5,6.25"] [--deep-from-x 12.5] \
        --out configs/pool_my_pool.json

Corner order (decision C-18): pool corner (0, 0), then (L, 0), then (L, W), then (0, W).
``--image`` may also be a video file (``--frame N`` picks the frame). ``--size WxH`` replaces ``--image``.
``--interactive`` opens a window to click the four corners; it needs an OpenCV build with GUI support
(``opencv-python``, not ``-headless``). Exit codes: 0 written, 2 bad input or invalid calibration.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from lifeguard.common.logging_setup import configure_logging
from lifeguard.system.geometry.calibrated import (
    CalibratedPoolGeometry,
    CheckPoint,
    PoolCalibration,
    save_calibration,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2


def _pairs(text: str, n: int) -> list[tuple[float, ...]]:
    out = []
    for chunk in text.split(";"):
        values = tuple(float(v) for v in chunk.split(","))
        if len(values) != n:
            raise ValueError(f"expected {n} comma-separated numbers in {chunk!r}")
        out.append(values)
    return out


def _load_image(path: Path, frame: int) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is not None:
        return image
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read an image or frame {frame} from {path}")
    return image


def _click_corners(image: np.ndarray) -> list[tuple[float, float]]:
    """Collect four left-clicks in an OpenCV window. Raises RuntimeError without GUI support."""
    import cv2

    clicks: list[tuple[float, float]] = []
    labels = ["(0, 0)", "(L, 0)", "(L, W)", "(0, W)"]
    window = "click pool corners (Esc cancels)"

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x), float(y)))

    try:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        while len(clicks) < 4:
            view = image.copy()
            for i, (x, y) in enumerate(clicks):
                cv2.circle(view, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.putText(view, labels[i], (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
            cv2.putText(view, f"click corner {labels[len(clicks)]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2)
            cv2.imshow(window, view)
            if cv2.waitKey(30) == 27:
                raise RuntimeError("calibration cancelled")
    except cv2.error as err:
        raise RuntimeError(
            "this OpenCV build has no GUI support (opencv-python-headless?); use --corners instead"
        ) from err
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return clicks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lifeguard.system.geometry.calibrate",
                                description="Create a pool calibration (homography) file.")
    p.add_argument("--site-id", required=True)
    p.add_argument("--length", type=float, required=True, help="Pool length (m), along X.")
    p.add_argument("--width", type=float, required=True, help="Pool width (m), along Y.")
    p.add_argument("--image", type=Path, help="Calibration image or video file.")
    p.add_argument("--frame", type=int, default=0, help="Frame index when --image is a video.")
    p.add_argument("--size", help="Image size WxH when no image is given.")
    p.add_argument("--corners", help='"u,v;u,v;u,v;u,v" in the order (0,0),(L,0),(L,W),(0,W).')
    p.add_argument("--interactive", action="store_true", help="Click the corners (needs OpenCV GUI).")
    p.add_argument("--exit", action="append", default=[], help="Exit (ladder/steps) position x,y in metres.")
    p.add_argument("--check", action="append", default=[], help="Check point u,v,x,y (e.g. lane mark).")
    p.add_argument("--deep-from-x", type=float, default=None)
    p.add_argument("--edge-band", type=float, default=1.0)
    p.add_argument("--max-error", type=float, default=0.3)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging("INFO")
    try:
        image = _load_image(args.image, args.frame) if args.image is not None else None
        if image is not None:
            height, width = image.shape[:2]
        elif args.size:
            width, height = (int(v) for v in args.size.lower().split("x"))
        else:
            raise ValueError("give --image or --size")
        if args.interactive:
            if image is None:
                raise ValueError("--interactive needs --image")
            corners = _click_corners(image)
        elif args.corners:
            corners = [(u, v) for u, v in _pairs(args.corners, 2)]
        else:
            raise ValueError("give --corners or --interactive")
        calibration = PoolCalibration(
            site_id=args.site_id,
            image_width=width,
            image_height=height,
            length_m=args.length,
            width_m=args.width,
            image_points=corners,
            deep_from_x_m=args.deep_from_x,
            edge_band_m=args.edge_band,
            exits_m=[(x, y) for x, y in (_pairs(e, 2)[0] for e in args.exit)],
            check_points=[CheckPoint(image=(u, v), pool=(x, y))
                          for u, v, x, y in (_pairs(c, 4)[0] for c in args.check)],
            max_reprojection_error_m=args.max_error,
        )
    except (ValueError, RuntimeError, ImportError) as err:
        logger.error("%s", err)
        return EXIT_BAD_ARGS

    save_calibration(args.out, calibration)
    geo = CalibratedPoolGeometry(calibration)
    centre = geo.to_image(args.length / 2, args.width / 2)
    print(f"calibration={args.out} site={args.site_id} image={width}x{height} "
          f"pool_centre_px={None if centre is None else tuple(round(c, 1) for c in centre)}")
    if not calibration.exits_m:
        logger.warning("no exits given: submersion timers are never suppressed near ladders/steps")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
