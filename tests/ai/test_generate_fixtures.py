"""TP-69/TP-70: fixture generation CLI."""
from __future__ import annotations

from pathlib import Path

import pytest

from lifeguard.ai.sim import generate_fixtures
from lifeguard.ai.sim.scenarios import SCENARIO_KINDS, make_scenario
from lifeguard.contracts.io import read_events_csv, read_jsonl
from lifeguard.contracts.types import FrameResult


def test_generate_writes_and_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Generation writes 6 JSONL + 6 CSV files that round-trip and re-verify."""
    out = tmp_path / "synthetic"
    assert generate_fixtures.main(["--out", str(out), "--seed", "0", "--fps", "10"]) == 0

    for kind in SCENARIO_KINDS:
        scenario = make_scenario(kind, seed=0, fps=10.0)
        jsonl = out / f"{kind}_seed0.jsonl"
        csv = out / f"{kind}_seed0_events.csv"
        assert jsonl.is_file() and csv.is_file()
        assert read_jsonl(jsonl, FrameResult) == list(scenario.frames)
        assert read_events_csv(csv) == list(scenario.events)

    captured = capsys.readouterr().out
    for kind in SCENARIO_KINDS:
        assert f"{kind}_seed0.jsonl" in captured
        assert f"{kind}_seed0_events.csv" in captured

    assert (
        generate_fixtures.main(["--out", str(out), "--seed", "0", "--fps", "10", "--check"])
        == 0
    )


def test_check_detects_drift(tmp_path: Path) -> None:
    """--check returns 1 after a fixture file is changed."""
    out = tmp_path / "synthetic"
    generate_fixtures.main(["--out", str(out), "--seed", "0", "--fps", "10"])

    (out / "normal_swim_seed0.jsonl").write_text("corrupt\n", encoding="utf-8")
    assert (
        generate_fixtures.main(["--out", str(out), "--seed", "0", "--fps", "10", "--check"])
        == 1
    )
