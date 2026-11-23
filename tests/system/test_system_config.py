"""TP-04/TP-05: SystemConfig loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from lifeguard.contracts.errors import ConfigError
from lifeguard.system.config import load_system_config

ROOT = Path(__file__).resolve().parents[2]


def test_default_system_yaml_loads() -> None:
    """The committed system.yaml loads with its documented defaults."""
    config = load_system_config(ROOT / "configs" / "system.yaml")
    assert config.cam_id == "cam_sim"
    assert config.session_id is None
    assert config.max_consecutive_engine_errors == 10


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("cam_id: a:b\n", id="bad-cam-id"),
        pytest.param("max_consecutive_engine_errors: 0\n", id="zero-errors"),
        pytest.param("cam_id: cam_sim\nfoo: bar\n", id="unknown-key"),
    ],
)
def test_system_config_rejects_bad_values(tmp_path: Path, content: str) -> None:
    """Bad values and unknown keys raise ConfigError through the loader."""
    path = tmp_path / "system.yaml"
    path.write_text(content, encoding="utf-8", newline="\n")
    with pytest.raises(ConfigError):
        load_system_config(path)
