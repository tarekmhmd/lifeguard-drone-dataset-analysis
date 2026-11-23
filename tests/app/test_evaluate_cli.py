"""evaluate CLI on the committed synthetic fixtures (a logic check of the risk engine, not accuracy)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"
SYSTEM = ROOT / "configs" / "system.yaml"


def _run(kind: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, "-m", "lifeguard.app.evaluate",
            "--frames", str(FIXTURES / f"{kind}_seed0.jsonl"),
            "--truth", str(FIXTURES / f"{kind}_seed0_events.csv"),
            "--system-config", str(SYSTEM),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_submersion_detected_without_false_alarms() -> None:
    proc = _run("submersion")
    assert proc.returncode == 0, proc.stderr
    (line,) = proc.stdout.splitlines()
    assert line.startswith("events=1 detected=1 missed=0 alarms=1 false_alarms=0 fa_per_hour=0.000 ")
    assert "median_latency_s=10.000" in line


def test_negative_scenario_has_no_events_and_no_alarms() -> None:
    proc = _run("dive")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("events=0 detected=0 missed=0 alarms=0 false_alarms=0 ")


def test_sweep_prints_one_line_per_value() -> None:
    proc = _run("submersion", "--sweep", "t_sub_s=5,10,40")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert [ln.split()[0] for ln in lines] == ["t_sub_s=5.0", "t_sub_s=10.0", "t_sub_s=40.0"]
    assert "detected=1" in lines[0] and "median_latency_s=5.000" in lines[0]
    assert "detected=0 missed=1" in lines[2]  # 40 s is never exceeded in a 60 s clip that starts at 20 s


def test_bad_sweep_key_exit_2() -> None:
    assert _run("submersion", "--sweep", "weights=1").returncode == 2
    assert _run("submersion", "--sweep", "t_watch=0.9").returncode == 2  # violates t_watch < t_pre_alert


def test_missing_frames_exit_2(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable, "-m", "lifeguard.app.evaluate",
            "--frames", str(tmp_path / "missing.jsonl"),
            "--truth", str(FIXTURES / "submersion_seed0_events.csv"),
            "--system-config", str(SYSTEM),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 2
