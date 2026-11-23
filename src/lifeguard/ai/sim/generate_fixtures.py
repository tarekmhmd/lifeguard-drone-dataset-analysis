"""CLI to generate and verify synthetic scenario fixtures."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from lifeguard.ai.sim.scenarios import SCENARIO_KINDS, make_scenario
from lifeguard.contracts.io import write_events_csv, write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate or verify synthetic scenario fixtures."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Frames per second (default 10)."
    )
    parser.add_argument("--check", action="store_true", help="Verify instead of writing.")
    return parser


def _check(out_dir: Path, seed: int, fps: float) -> int:
    """Regenerate fixtures and compare them byte-for-byte with the files on disk.

    Returns 0 when identical, 1 when any file differs or is missing.
    """
    mismatches: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for kind in SCENARIO_KINDS:
            scenario = make_scenario(kind, seed=seed, fps=fps)
            jsonl_name = f"{kind}_seed{seed}.jsonl"
            csv_name = f"{kind}_seed{seed}_events.csv"
            write_jsonl(tmp_path / jsonl_name, scenario.frames)
            write_events_csv(tmp_path / csv_name, scenario.events)
            for name in (jsonl_name, csv_name):
                expected = out_dir / name
                if not expected.is_file() or expected.read_bytes() != (tmp_path / name).read_bytes():
                    mismatches.append(name)

    for name in mismatches:
        print(name, file=sys.stderr)
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    """Run the fixture CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out
    if args.check:
        return _check(out_dir, args.seed, args.fps)

    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in SCENARIO_KINDS:
        scenario = make_scenario(kind, seed=args.seed, fps=args.fps)
        jsonl_name = f"{kind}_seed{args.seed}.jsonl"
        csv_name = f"{kind}_seed{args.seed}_events.csv"
        frames = write_jsonl(out_dir / jsonl_name, scenario.frames)
        events = write_events_csv(out_dir / csv_name, scenario.events)
        print(f"{jsonl_name}: {frames} lines")
        print(f"{csv_name}: {events + 1} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
