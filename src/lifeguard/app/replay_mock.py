"""Phase 1 integration command: replay the mock perception engine end to end (SPEC 7.3).

Usage::

    python -m lifeguard.app.replay_mock --ai-config configs/ai.yaml \
        --system-config configs/system.yaml [--frames N] [--out PATH.jsonl]

Exit codes (SPEC 8): 0 = success, 2 = bad arguments or configuration,
3 = model/fixture load error, 4 = too many consecutive engine errors.

This command contains no risk logic; it is the Phase 1 stand-in for ``app/main.py``.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lifeguard.ai.config import load_ai_config
from lifeguard.ai.perception.factory import create_engine
from lifeguard.common.logging_setup import configure_logging
from lifeguard.contracts.enums import Status
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.contracts.io import write_jsonl
from lifeguard.contracts.timeutil import utc_now
from lifeguard.contracts.types import FrameResult
from lifeguard.system.config import load_system_config

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_MODEL_LOAD = 3
EXIT_ENGINE_ERRORS = 4

# Media clock used for the synthetic replay (SPEC 7.3 step 4: t_media = i / 10.0).
_MEDIA_FPS = 10.0


def _positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        number = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from err
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1: {value!r}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m lifeguard.app.replay_mock",
        description="Replay the mock perception engine on black frames and summarise the results.",
    )
    parser.add_argument("--ai-config", type=Path, required=True, help="Path to ai.yaml.")
    parser.add_argument("--system-config", type=Path, required=True, help="Path to system.yaml.")
    parser.add_argument(
        "--frames",
        type=_positive_int,
        default=None,
        help="Number of frames to process (default: the engine fixture length).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSONL output path.")
    return parser


def _generate_session_id() -> str:
    """Generate a session id of the form ``sess_YYYYMMDDTHHMMSSZ`` (SPEC 7.1)."""
    return datetime.now(timezone.utc).strftime("sess_%Y%m%dT%H%M%SZ")


def _summary_line(results: list[FrameResult]) -> str:
    """Build the single summary line required by SPEC 7.3 step 8."""
    n_ok = sum(1 for r in results if r.status == Status.OK)
    n_deg = sum(1 for r in results if r.status == Status.DEGRADED)
    n_err = sum(1 for r in results if r.status == Status.ERROR)
    unique_tracks = {t.track_uid for r in results for t in r.tracks}
    max_submersion = max(
        (s.seconds_submerged for r in results for s in r.submersions),
        default=0.0,
    )
    return (
        f"frames={len(results)} ok={n_ok} degraded={n_deg} error={n_err} "
        f"unique_tracks={len(unique_tracks)} max_submersion_s={max_submersion:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the mock replay and return the process exit code."""
    args = _build_parser().parse_args(argv)

    # Step 1: configs, logging, session id.
    try:
        ai_cfg = load_ai_config(args.ai_config)
        sys_cfg = load_system_config(args.system_config)
    except ConfigError as err:
        configure_logging("INFO")
        logger.error("configuration error: %s", err)
        return EXIT_BAD_ARGS

    configure_logging(sys_cfg.log_level, sys_cfg.log_file)
    session_id = sys_cfg.session_id if sys_cfg.session_id is not None else _generate_session_id()
    logger.info("replay_mock starting: cam_id=%s session_id=%s", sys_cfg.cam_id, session_id)

    # Step 2: engine.
    try:
        engine = create_engine(ai_cfg, sys_cfg.cam_id, session_id)
    except ModelLoadError as err:
        logger.error("engine load failed: %s", err)
        return EXIT_MODEL_LOAD

    # Step 3: number of frames. fixture_length / expected_frame_size are MockPerceptionEngine
    # properties (not part of the PerceptionEngine ABC); Phase 1 only builds the mock engine.
    n_frames = args.frames
    if n_frames is None:
        n_frames = getattr(engine, "fixture_length", None)
        if n_frames is None:
            logger.error("engine has no fixture_length; pass --frames explicitly")
            return EXIT_BAD_ARGS
    width, height = engine.expected_frame_size  # type: ignore[attr-defined]
    frame = np.zeros((height, width, 3), np.uint8)

    # Steps 4-6: process, count consecutive errors, re-validate.
    results: list[FrameResult] = []
    consecutive_errors = 0
    for i in range(n_frames):
        result = engine.process(
            frame,
            frame_idx=i,
            t_capture=time.monotonic(),
            t_utc=utc_now(),
            t_media=i / _MEDIA_FPS,
        )
        if result.status == Status.ERROR:
            consecutive_errors += 1
            if consecutive_errors >= sys_cfg.max_consecutive_engine_errors:
                logger.error(
                    "%d consecutive engine errors (limit %d) at frame %d; last error: %s",
                    consecutive_errors,
                    sys_cfg.max_consecutive_engine_errors,
                    i,
                    result.error,
                )
                return EXIT_ENGINE_ERRORS
        else:
            consecutive_errors = 0
        results.append(FrameResult.model_validate_json(result.model_dump_json()))

    # Step 7: optional output.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        written = write_jsonl(args.out, results)
        logger.info("wrote %d results to %s", written, args.out)

    # Step 8: summary.
    print(_summary_line(results))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
