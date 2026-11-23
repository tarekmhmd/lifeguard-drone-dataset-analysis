"""TP-110: the GitHub Actions workflow parses and runs the SPEC 9.2 commands."""
from __future__ import annotations

from pathlib import Path

import yaml

CI_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

REQUIRED_COMMANDS = (
    'python -m pip install -e ".[dev]"',
    "python -m ruff check src tests",
    "python -m pytest",
    "python -m lifeguard.ai.sim.generate_fixtures --out tests/fixtures/synthetic --check",
    "python -m lifeguard.app.replay_mock --ai-config configs/ai.yaml --system-config configs/system.yaml",
)


def test_ci_yaml_parses() -> None:
    """The workflow is valid YAML, covers both OSes and runs every required command."""
    data = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    triggers = data.get("on", data.get(True))
    assert triggers is not None
    assert "push" in triggers and "pull_request" in triggers

    jobs = data["jobs"]
    assert len(jobs) >= 1
    job = next(iter(jobs.values()))

    matrix = job["strategy"]["matrix"]
    assert "windows-latest" in matrix["os"]
    assert "ubuntu-latest" in matrix["os"]
    assert "3.11" in [str(v) for v in matrix["python-version"]]

    uses = [step.get("uses", "") for step in job["steps"]]
    assert any(u.startswith("actions/checkout") for u in uses)
    assert any(u.startswith("actions/setup-python") for u in uses)

    runs = [step["run"].strip() for step in job["steps"] if "run" in step]
    for command in REQUIRED_COMMANDS:
        assert command in runs, f"missing CI step: {command}"


def test_ci_installs_dashboard_and_checks_models() -> None:
    """The dashboard extra is installed before pytest (so its test runs), and the model registry is verified."""
    data = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    steps = next(iter(data["jobs"].values()))["steps"]
    runs = [step["run"].strip() for step in steps if "run" in step]
    dashboard = runs.index('python -m pip install -e ".[dashboard]"')
    assert dashboard < runs.index("python -m pytest")
    assert "python -m lifeguard.common.models verify" in runs
    linux = next(s for s in steps if "libgl1" in s.get("run", ""))
    assert linux["if"] == "runner.os == 'Linux'"


def test_fixtures_are_checked_out_byte_exact() -> None:
    """Fixtures are compared byte for byte and checksummed: git must never convert their line endings."""
    attributes = (CI_PATH.parents[2] / ".gitattributes").read_text(encoding="utf-8")
    assert "tests/fixtures/** -text" in attributes
