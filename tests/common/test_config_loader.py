"""TP-03: load_yaml_model reports every configuration failure as ConfigError."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from lifeguard.common.config import load_yaml_model
from lifeguard.contracts.errors import ConfigError


class _StrictModel(BaseModel):
    """Minimal model with unknown keys forbidden."""

    model_config = ConfigDict(extra="forbid")

    value: int


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        pytest.param("missing.yaml", None, id="missing-file"),
        pytest.param("syntax.yaml", "value: [unclosed\n", id="yaml-syntax-error"),
        pytest.param("list.yaml", "- one\n- two\n", id="top-level-list"),
        pytest.param("unknown.yaml", "value: 1\nextra: 2\n", id="unknown-key"),
    ],
)
def test_load_yaml_model_errors(
    tmp_path: Path,
    filename: str,
    content: str | None,
) -> None:
    """Every bad configuration raises ConfigError naming the file."""
    path = tmp_path / filename
    if content is not None:
        path.write_text(content, encoding="utf-8", newline="\n")

    with pytest.raises(ConfigError) as excinfo:
        load_yaml_model(path, _StrictModel)

    assert path.name in str(excinfo.value)
