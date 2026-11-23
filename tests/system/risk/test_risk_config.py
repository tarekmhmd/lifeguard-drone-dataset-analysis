"""Risk engine configuration: defaults, validation and the committed system.yaml block."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lifeguard.contracts.errors import ConfigError
from lifeguard.system.config import load_system_config
from lifeguard.system.risk.config import RiskConfig, RiskWeights

ROOT = Path(__file__).resolve().parents[3]


def test_defaults_are_valid() -> None:
    """The default configuration validates and keeps the behaviour weight at 0."""
    cfg = RiskConfig()
    assert cfg.t_watch < cfg.t_pre_alert < cfg.t_alarm
    assert cfg.weights.behavior == 0.0
    assert cfg.pre_alert_ack_timeout_s is None


def test_committed_system_yaml_risk_block_matches_defaults() -> None:
    """Every threshold lives in configs/system.yaml, and the file matches the documented defaults."""
    config = load_system_config(ROOT / "configs" / "system.yaml")
    assert config.risk == RiskConfig()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"t_watch": 0.7, "t_pre_alert": 0.6}, id="watch-above-pre-alert"),
        pytest.param({"t_pre_alert": 0.85}, id="pre-alert-above-alarm"),
        pytest.param({"hysteresis": 0.4}, id="hysteresis-too-large"),
        pytest.param({"t_sub_s": 0.0}, id="zero-t-sub"),
        pytest.param({"pre_alert_ack_timeout_s": 0.0}, id="zero-ack-timeout"),
    ],
)
def test_invalid_risk_config_rejected(overrides: dict[str, float]) -> None:
    """Out-of-order thresholds and non-positive durations are rejected."""
    with pytest.raises(ValidationError):
        RiskConfig(**overrides)


def test_all_zero_weights_rejected() -> None:
    """At least one evidence weight must be positive."""
    with pytest.raises(ValidationError):
        RiskWeights(submersion=0.0, static=0.0, visibility=0.0, deep_zone=0.0, behavior=0.0)


def test_unknown_risk_key_rejected_by_loader(tmp_path: Path) -> None:
    """A misspelled risk key fails loudly (extra='forbid')."""
    path = tmp_path / "system.yaml"
    path.write_text("risk:\n  t_watchh: 0.4\n", encoding="utf-8", newline="\n")
    with pytest.raises(ConfigError):
        load_system_config(path)
