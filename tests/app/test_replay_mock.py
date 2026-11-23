"""TP-100..TP-104: mock replay integration command (SPEC 7.3), run via subprocess."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from lifeguard.ai.sim.scenarios import SCENARIO_KINDS
from lifeguard.contracts.enums import Status
from lifeguard.contracts.io import read_jsonl, write_jsonl
from lifeguard.contracts.types import FrameResult

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"
SUMMARY_RE = re.compile(
    r"^frames=(\d+) ok=(\d+) degraded=(\d+) error=(\d+) unique_tracks=(\d+) "
    r"max_submersion_s=(\d+\.\d{3})$"
)
ZERO_SUBMERSION_KINDS = {"normal_swim", "idle", "edge_exit"}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the replay CLI in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, "-m", "lifeguard.app.replay_mock", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _write_ai_config(tmp_path: Path, fixture: Path) -> Path:
    """Write a tmp ai.yaml pointing at ``fixture`` (absolute path)."""
    path = tmp_path / "ai.yaml"
    path.write_text(
        f'engine: mock\nmock:\n  fixture_path: "{fixture.as_posix()}"\n  on_exhausted: empty\n',
        encoding="utf-8",
    )
    return path


def _write_system_config(tmp_path: Path, max_errors: int = 10) -> Path:
    """Write a tmp system.yaml with a fixed session id."""
    path = tmp_path / "system.yaml"
    path.write_text(
        "cam_id: cam_sim\n"
        "session_id: test_session\n"
        "log_level: INFO\n"
        "log_file: null\n"
        f"max_consecutive_engine_errors: {max_errors}\n",
        encoding="utf-8",
    )
    return path


def _summary(stdout: str) -> re.Match[str]:
    """Assert stdout is exactly one summary line and return its match."""
    lines = stdout.splitlines()
    assert len(lines) == 1, stdout
    match = SUMMARY_RE.match(lines[0])
    assert match is not None, lines[0]
    return match


def test_replay_default_configs() -> None:
    """TP-100: default configs exit 0 with one summary line and max_submersion_s > 0."""
    proc = _run(
        "--ai-config", str(CONFIGS / "ai.yaml"),
        "--system-config", str(CONFIGS / "system.yaml"),
    )
    assert proc.returncode == 0, proc.stderr
    match = _summary(proc.stdout)
    assert float(match.group(6)) > 0


def test_replay_writes_valid_jsonl(tmp_path: Path) -> None:
    """TP-101: --out writes a JSONL with one valid FrameResult per processed frame."""
    out = tmp_path / "out.jsonl"
    proc = _run(
        "--ai-config", str(CONFIGS / "ai.yaml"),
        "--system-config", str(CONFIGS / "system.yaml"),
        "--out", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    frames = int(_summary(proc.stdout).group(1))
    results = read_jsonl(out, FrameResult)
    assert len(results) == frames
    assert [r.frame_idx for r in results] == list(range(frames))


@pytest.mark.parametrize("kind", SCENARIO_KINDS)
def test_replay_each_fixture(tmp_path: Path, kind: str) -> None:
    """TP-102: every committed fixture replays with exit 0; non-submersion kinds report 0.000."""
    ai_cfg = _write_ai_config(tmp_path, FIXTURES / f"{kind}_seed0.jsonl")
    sys_cfg = _write_system_config(tmp_path)
    proc = _run("--ai-config", str(ai_cfg), "--system-config", str(sys_cfg))
    assert proc.returncode == 0, proc.stderr
    match = _summary(proc.stdout)
    if kind in ZERO_SUBMERSION_KINDS:
        assert match.group(6) == "0.000"


def test_replay_missing_fixture_exit_3(tmp_path: Path) -> None:
    """TP-103: an ai.yaml pointing at a missing fixture exits with code 3."""
    ai_cfg = _write_ai_config(tmp_path, tmp_path / "does_not_exist.jsonl")
    sys_cfg = _write_system_config(tmp_path)
    proc = _run("--ai-config", str(ai_cfg), "--system-config", str(sys_cfg))
    assert proc.returncode == 3, proc.stderr
    assert proc.stdout == ""


def test_replay_consecutive_errors_exit_4(tmp_path: Path) -> None:
    """TP-104: 20 status=error lines with max_consecutive_engine_errors=5 exit with code 4."""
    fixture = tmp_path / "errors.jsonl"
    write_jsonl(
        fixture,
        [
            FrameResult(
                cam_id="cam_sim",
                session_id="test_session",
                frame_idx=i,
                t_capture=float(i),
                t_utc="2026-01-01T00:00:00.000Z",
                frame_width=640,
                frame_height=480,
                status=Status.ERROR,
                error="synthetic engine failure",
                latency_ms=0.0,
            )
            for i in range(20)
        ],
    )
    ai_cfg = _write_ai_config(tmp_path, fixture)
    sys_cfg = _write_system_config(tmp_path, max_errors=5)
    proc = _run("--ai-config", str(ai_cfg), "--system-config", str(sys_cfg))
    assert proc.returncode == 4, proc.stderr
    assert proc.stdout == ""
