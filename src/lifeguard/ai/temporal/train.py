"""Train the softmax behaviour model from labelled sessions (roadmap P5-2).

Usage (``--frames``/``--truth`` are given once per session, in the same order)::

    python -m lifeguard.ai.temporal.train \
        --frames s1_frames.jsonl --truth s1_events.csv \
        --frames s2_frames.jsonl --truth s2_events.csv \
        --out models/behavior_softmax.json [--cv] [--default-label normal_swim]

The frames must contain ``TrackObs.features``: produce them with
``python -m lifeguard.app.run_video ... --out-frames`` (real engine). The mock fixtures have no features.
Exit codes: 0 ok, 2 bad arguments or input files.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from lifeguard.ai.temporal.cv import leave_one_session_out
from lifeguard.ai.temporal.dataset import BEHAVIOR_CLASSES, Sample, build_samples
from lifeguard.ai.temporal.labels import build_interval_samples, read_intervals
from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel
from lifeguard.common.logging_setup import configure_logging
from lifeguard.contracts.errors import ContractValidationError, ContractVersionError
from lifeguard.contracts.io import read_events_csv, read_jsonl
from lifeguard.contracts.types import FrameResult

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lifeguard.ai.temporal.train",
        description="Train the softmax behaviour model on labelled sessions.",
    )
    parser.add_argument("--frames", type=Path, action="append", required=True)
    parser.add_argument("--truth", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Model JSON to write.")
    parser.add_argument("--default-label", choices=BEHAVIOR_CLASSES, default="normal_swim")
    parser.add_argument("--stride", type=float, default=0.5, help="Seconds between samples of a track.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cv", action="store_true", help="Also run leave-one-session-out CV.")
    parser.add_argument("--intervals", action="store_true",
                        help="--truth files are M1-04 behaviour-interval CSVs (all five labels) instead of event CSVs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run training and return the exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging("INFO")
    if len(args.frames) != len(args.truth):
        logger.error("give exactly one --truth per --frames")
        return EXIT_BAD_ARGS

    samples: list[Sample] = []
    try:
        for frames_path, truth_path in zip(args.frames, args.truth):
            frames = read_jsonl(frames_path, FrameResult)
            if args.intervals:
                session = build_interval_samples(frames, read_intervals(truth_path), stride_s=args.stride)
            else:
                truths = read_events_csv(truth_path)
                session = build_samples(frames, truths, default_label=args.default_label, stride_s=args.stride)
            logger.info("%s: %d samples", frames_path.name, len(session))
            samples += session
    except (FileNotFoundError, ContractValidationError, ContractVersionError, ValueError) as err:
        logger.error("%s", err)
        return EXIT_BAD_ARGS
    if not samples:
        logger.error("no samples: do the frames contain TrackObs.features (real engine output)?")
        return EXIT_BAD_ARGS

    counts = {c: sum(1 for s in samples if s.label == c) for c in BEHAVIOR_CLASSES}
    print("samples=" + str(len(samples)) + " " + " ".join(f"{c}={n}" for c, n in counts.items() if n))

    if args.cv:
        try:
            report = leave_one_session_out(
                samples, lambda x, y: SoftmaxBehaviorModel.fit(x, y, epochs=args.epochs)
            )
        except ValueError as err:
            logger.error("cross-validation: %s", err)
            return EXIT_BAD_ARGS
        for line in report.lines():
            print(line)

    model = SoftmaxBehaviorModel.fit(
        np.stack([s.x for s in samples]),
        [s.label for s in samples],
        epochs=args.epochs,
        metadata={"sessions": sorted({s.session_id for s in samples})},
    )
    model.save(args.out)
    print(f"model={args.out} version={model.model_version}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
