"""Replay FrameResult JSONL through the risk engine and score it against ground-truth events (M2-06).

Usage::

    python -m lifeguard.app.evaluate --frames tests/fixtures/synthetic/submersion_seed0.jsonl \
        --truth tests/fixtures/synthetic/submersion_seed0_events.csv --system-config configs/system.yaml \
        [--l-max 30] [--radius 2.0] [--level ALARM|PRE_ALERT|WATCH] [--sweep t_sub_s=6,8,10,12]

The frames can come from the synthetic fixtures (a LOGIC check only) or from ``run_video --out-frames``
on a labelled recording (a real measurement for that recording). One input = one monitored session.
Prints one summary line per risk configuration. Exit codes: 0 ok, 2 bad arguments or input files.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from lifeguard.common.logging_setup import configure_logging
from lifeguard.contracts.enums import RiskLevel
from lifeguard.contracts.errors import ConfigError, ContractValidationError, ContractVersionError
from lifeguard.contracts.io import read_events_csv, read_jsonl
from lifeguard.contracts.types import FrameResult, RiskEvent
from lifeguard.system.config import load_system_config
from lifeguard.system.eval.events import EvalConfig, evaluate_events
from lifeguard.system.risk.config import RiskConfig
from lifeguard.system.risk.engine import RuleBasedRiskEngine

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_BAD_ARGS = 2

_LEVELS_AT_OR_ABOVE: dict[str, frozenset[RiskLevel]] = {
    "ALARM": frozenset({RiskLevel.ALARM}),
    "PRE_ALERT": frozenset({RiskLevel.PRE_ALERT, RiskLevel.ALARM}),
    "WATCH": frozenset({RiskLevel.WATCH, RiskLevel.PRE_ALERT, RiskLevel.ALARM}),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lifeguard.app.evaluate",
        description="Event-level evaluation of the risk engine (FA/h, missed events, latency).",
    )
    parser.add_argument("--frames", type=Path, required=True, help="FrameResult JSONL (one session).")
    parser.add_argument("--truth", type=Path, required=True, help="Ground-truth events CSV.")
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--l-max", type=float, default=30.0, help="Detection window after onset (s).")
    parser.add_argument("--radius", type=float, default=2.0, help="Location match radius (m).")
    parser.add_argument("--level", choices=sorted(_LEVELS_AT_OR_ABOVE), default="ALARM")
    parser.add_argument("--sweep", default=None, help="KEY=V1,V2,... over a numeric risk setting.")
    return parser


def _parse_sweep(spec: str | None, base: RiskConfig) -> list[tuple[str, RiskConfig]]:
    """Return (label, config) pairs. Raises ValueError for a bad spec or an invalid resulting config."""
    if spec is None:
        return [("", base)]
    key, sep, values = spec.partition("=")
    key = key.strip()
    numeric = {
        name for name, info in RiskConfig.model_fields.items()
        if name != "weights" and info.annotation in (float, int)
    }
    if not sep or key not in numeric:
        raise ValueError(f"--sweep must be KEY=V1,V2 with KEY one of: {', '.join(sorted(numeric))}")
    out: list[tuple[str, RiskConfig]] = []
    for raw in values.split(","):
        value = float(raw)
        try:
            cfg = RiskConfig.model_validate({**base.model_dump(), key: value})
        except ValidationError as err:
            raise ValueError(f"{key}={value} is not a valid risk configuration: {err}") from err
        out.append((f"{key}={value} ", cfg))
    return out


def _monitored_seconds(frames: list[FrameResult]) -> float:
    times = [f.t_media for f in frames if f.t_media is not None]
    if len(times) < 2:
        raise ValueError("need at least two frames with t_media to measure the monitored duration")
    period = times[1] - times[0]
    return (max(times) - min(times)) + max(period, 0.0)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and return the exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging("WARNING")
    try:
        sys_cfg = load_system_config(args.system_config)
        frames = read_jsonl(args.frames, FrameResult)
        truths = read_events_csv(args.truth)
        monitored = _monitored_seconds(frames)
        runs = _parse_sweep(args.sweep, sys_cfg.risk)
        eval_cfg = EvalConfig(
            l_max_s=args.l_max,
            match_radius_m=args.radius,
            alarm_levels=_LEVELS_AT_OR_ABOVE[args.level],
        )
    except (ConfigError, ContractValidationError, ContractVersionError, FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return EXIT_BAD_ARGS

    for label, risk_cfg in runs:
        engine = RuleBasedRiskEngine(risk_cfg)
        events: list[RiskEvent] = []
        for frame in frames:
            events += engine.update(frame)
        report = evaluate_events(events, truths, monitored, eval_cfg)
        print(f"{label}{report.line()}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
