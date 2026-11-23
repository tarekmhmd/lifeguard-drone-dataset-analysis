"""TP-80/TP-81: AIConfig loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from lifeguard.ai.config import load_ai_config
from lifeguard.contracts.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def test_ai_yaml_loads() -> None:
    """The committed ai.yaml loads; the fixture path is absolute and exists."""
    config = load_ai_config(ROOT / "configs" / "ai.yaml")
    assert config.engine == "mock"
    assert config.mock is not None
    assert config.mock.fixture_path.is_absolute()
    assert config.mock.fixture_path.exists()


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("engine: coco\n", id="unknown-engine"),
        pytest.param("engine: mock\n", id="mock-without-block"),
        pytest.param(
            "engine: mock\nmock:\n  fixture_path: x.jsonl\n  on_exhausted: empty\nfoo: 1\n",
            id="unknown-key",
        ),
    ],
)
def test_ai_config_rejects(tmp_path: Path, content: str) -> None:
    """Unknown engines, missing mock blocks and unknown keys raise ConfigError."""
    path = tmp_path / "ai.yaml"
    path.write_text(content, encoding="utf-8", newline="\n")
    with pytest.raises(ConfigError):
        load_ai_config(path)
