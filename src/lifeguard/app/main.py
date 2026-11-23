"""Run the integrated lifeguard system (roadmap Phase 7).

Usage::

    # Full stack without a camera or model weights: MOCK perception (fixture replay) on synthetic frames
    python -m lifeguard.app.main --ai-config configs/ai.yaml --system-config configs/system.yaml --synthetic

    # Recorded pool video with the real engine (needs the vision extra and local weights)
    python -m lifeguard.app.main --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml \
        --video pool.mp4 [--realtime]

    # Live camera (index or RTSP URL); Ctrl+C stops cleanly
    python -m lifeguard.app.main --ai-config configs/ai_yolo.yaml --system-config configs/system.yaml \
        --camera 0

Headless: alerts are logged and printed; operator decisions (confirm/dismiss) need the dashboard
(``streamlit run src/lifeguard/app/dashboard.py``). Exit codes: 0 ok, 2 bad configuration or input,
3 model/fixture load error.
"""
from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import IO

from lifeguard.ai.config import AIConfig, load_ai_config
from lifeguard.ai.perception.mock_engine import MockPerceptionEngine
from lifeguard.app.runtime import LifeguardRuntime
from lifeguard.common.logging_setup import configure_logging
from lifeguard.common.models import ModelIntegrityError, integrity_fault
from lifeguard.contracts.errors import ConfigError, ModelLoadError
from lifeguard.system.config import SystemConfig, load_system_config
from lifeguard.system.ingest.live import LiveCameraSource, SyntheticFrameSource, parse_source
from lifeguard.system.ingest.video import VideoFileSource, VideoFrame, VideoSourceError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_MODEL_LOAD = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lifeguard.app.main", description="Run the lifeguard system.")
    p.add_argument("--ai-config", type=Path, required=True)
    p.add_argument("--system-config", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Recorded video file.")
    src.add_argument("--camera", help="Camera index (0) or stream URL (rtsp://...).")
    src.add_argument("--synthetic", action="store_true", help="Black frames for the MOCK engine (demo/test).")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--realtime", action="store_true", help="Pace file/synthetic replay at its frame rate.")
    p.add_argument("--out-events", type=Path, default=None, help="Write RiskEvent JSONL.")
    return p


def open_source(
    args: argparse.Namespace, ai_cfg: AIConfig
) -> tuple[Iterable[VideoFrame | None], tuple[int, int], float | None]:
    """Open the selected input -> (source, (width, height), fps). Raises VideoSourceError / ModelLoadError."""
    if args.video is not None:
        source = VideoFileSource(args.video, max_frames=args.max_frames)
        return source, (source.width, source.height), source.fps
    if args.camera is not None:
        import cv2  # size probe; the reader thread opens its own capture

        cap = cv2.VideoCapture(parse_source(args.camera))
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise VideoSourceError(f"camera {args.camera!r} not available")
        return LiveCameraSource(parse_source(args.camera)), (image.shape[1], image.shape[0]), None
    if ai_cfg.engine != "mock" or ai_cfg.mock is None:
        raise ValueError("--synthetic needs engine: mock (black frames carry no image content)")
    probe = MockPerceptionEngine(ai_cfg.mock.fixture_path, "probe", "probe")
    n = args.max_frames or probe.fixture_length
    return SyntheticFrameSource(*probe.expected_frame_size, n_frames=n), probe.expected_frame_size, 10.0


def run(args: argparse.Namespace, ai_cfg: AIConfig, sys_cfg: SystemConfig, stop: threading.Event | None = None) -> int:
    try:
        source, size, fps = open_source(args, ai_cfg)
        runtime = LifeguardRuntime(ai_cfg, sys_cfg, size)
    except ModelIntegrityError as err:
        fault = integrity_fault(err)
        logger.error("SYSTEM FAULT (%s): %s", fault.component.value, fault.detail)
        print(f"[ALARM] SYSTEM FAULT ({fault.component.value}): {fault.detail}. The system did not start.")
        return EXIT_MODEL_LOAD
    except ModelLoadError as err:
        logger.error("model load failed: %s", err)
        return EXIT_MODEL_LOAD
    except (VideoSourceError, ConfigError, ValueError, ImportError) as err:
        logger.error("cannot start: %s", err)
        return EXIT_BAD_ARGS

    with ExitStack() as stack:
        stack.callback(runtime.close)
        if isinstance(source, (VideoFileSource,)):
            stack.callback(source.close)
        if isinstance(source, LiveCameraSource):
            stack.callback(source.stop)
        events_out: IO[str] | None = None
        if args.out_events is not None:
            args.out_events.parent.mkdir(parents=True, exist_ok=True)
            events_out = stack.enter_context(args.out_events.open("w", encoding="utf-8", newline="\n"))
        runtime.attach_source(source)
        try:
            for item in source:
                if stop is not None and stop.is_set():
                    break
                if item is None:
                    runtime.idle()
                    continue
                for event in runtime.process(item):
                    if events_out is not None:
                        events_out.write(event.model_dump_json() + "\n")
                for note in runtime.drain_notifications():
                    print(f"[{note.priority.value.upper()}] {note.message}")
                if args.realtime and fps:
                    stop_wait = stop or threading.Event()
                    stop_wait.wait(1.0 / fps)
        except KeyboardInterrupt:
            logger.info("stopped by operator")
        summary = runtime.snapshot().summary
        incidents = len(runtime.store.incidents())
    print(f"{summary.line()} incidents={incidents}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ai_cfg = load_ai_config(args.ai_config)
        sys_cfg = load_system_config(args.system_config)
    except ConfigError as err:
        configure_logging("INFO")
        logger.error("configuration error: %s", err)
        return EXIT_BAD_ARGS
    configure_logging(sys_cfg.log_level, sys_cfg.log_file)
    return run(args, ai_cfg, sys_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
