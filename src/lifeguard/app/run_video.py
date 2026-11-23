"""Run the real pipeline on a recorded video: video -> perception -> risk engine (roadmap P4/P7).

Usage::

    python -m lifeguard.app.run_video --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml \
        --video pool.mp4 [--pool-length 25 --pool-width 12.5] [--stride N] [--max-frames N] \
        [--out-frames frames.jsonl] [--out-events events.jsonl]

Intended for ``engine: yolo`` (needs the ``vision`` extra and local weights). Pool geometry is the
PLACEHOLDER ``StubPoolGeometry`` (a rectangle filling the central 80 % of the image) until calibration
exists (roadmap P4-2), so pool coordinates, zones and in-water decisions are only approximate.

Exit codes (SPEC 8): 0 success, 2 bad arguments/configuration/video, 3 model load error,
4 too many consecutive engine errors.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.perception.factory import create_engine
from lifeguard.app.pipeline import TooManyEngineErrors, run_pipeline
from lifeguard.common.logging_setup import configure_logging
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.contracts.types import ContractModel
from lifeguard.system.config import load_system_config
from lifeguard.system.geometry.stub import StubPoolGeometry
from lifeguard.system.ingest.video import VideoFileSource, VideoSourceError
from lifeguard.system.risk.engine import RuleBasedRiskEngine

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_MODEL_LOAD = 3
EXIT_ENGINE_ERRORS = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lifeguard.app.run_video",
        description="Run detection, tracking, submersion timers and the risk engine on a video file.",
    )
    parser.add_argument("--ai-config", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pool-length", type=float, default=25.0, help="Placeholder pool length (m).")
    parser.add_argument("--pool-width", type=float, default=12.5, help="Placeholder pool width (m).")
    parser.add_argument("--fps", type=float, default=None, help="Override the video frame rate.")
    parser.add_argument("--stride", type=int, default=1, help="Process every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out-frames", type=Path, default=None, help="Write FrameResult JSONL.")
    parser.add_argument("--out-events", type=Path, default=None, help="Write RiskEvent JSONL.")
    return parser


def _jsonl(stack: ExitStack, path: Path | None) -> IO[str] | None:
    """Open a UTF-8, LF-terminated output file owned by ``stack``, or return None."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))


def _writer(handle: IO[str] | None) -> Callable[[ContractModel], None] | None:
    """A callback that appends one JSONL line per contract message, or None."""
    if handle is None:
        return None

    def write(item: ContractModel) -> None:
        handle.write(item.model_dump_json() + "\n")

    return write


def main(argv: list[str] | None = None) -> int:
    """Run the command and return the exit code."""
    args = _build_parser().parse_args(argv)

    try:
        ai_cfg = load_ai_config(args.ai_config)
        sys_cfg = load_system_config(args.system_config)
    except ConfigError as err:
        configure_logging("INFO")
        logger.error("configuration error: %s", err)
        return EXIT_BAD_ARGS
    configure_logging(sys_cfg.log_level, sys_cfg.log_file)
    session_id = sys_cfg.session_id or datetime.now(timezone.utc).strftime("sess_%Y%m%dT%H%M%SZ")

    with ExitStack() as stack:
        try:
            source = stack.enter_context(
                VideoFileSource(args.video, fps=args.fps, stride=args.stride, max_frames=args.max_frames)
            )
            geometry = StubPoolGeometry(
                length_m=args.pool_length,
                width_m=args.pool_width,
                image_width=source.width,
                image_height=source.height,
            )
        except (VideoSourceError, ValueError) as err:
            logger.error("cannot use video: %s", err)
            return EXIT_BAD_ARGS
        logger.warning(
            "using PLACEHOLDER rectangular pool geometry %.1f x %.1f m; positions are approximate",
            args.pool_length,
            args.pool_width,
        )

        try:
            engine = create_engine(ai_cfg, sys_cfg.cam_id, session_id, geometry=geometry)
        except ModelLoadError as err:
            logger.error("engine load failed: %s", err)
            return EXIT_MODEL_LOAD
        except ValueError as err:
            logger.error("cannot build engine: %s", err)
            return EXIT_BAD_ARGS

        risk = RuleBasedRiskEngine(sys_cfg.risk)
        on_result = _writer(_jsonl(stack, args.out_frames))
        on_event = _writer(_jsonl(stack, args.out_events))
        try:
            summary = run_pipeline(
                source,
                engine,
                risk,
                sys_cfg.max_consecutive_engine_errors,
                on_result=on_result,
                on_event=on_event,
            )
        except TooManyEngineErrors as err:
            logger.error("%s", err)
            return EXIT_ENGINE_ERRORS

    print(summary.line())
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
