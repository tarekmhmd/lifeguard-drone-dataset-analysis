"""YAML configuration loading for the Phase 1 lifeguard package."""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from lifeguard.contracts.errors import ConfigError

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(path: Path | str, model: type[ModelT]) -> ModelT:
    """Load a YAML file and validate it as a pydantic model.

    Args:
        path: Path to the YAML configuration file.
        model: Pydantic model class used to validate the parsed mapping.

    Returns:
        The validated model instance.

    Raises:
        ConfigError: If the file is missing, cannot be read, contains invalid
            YAML, does not contain a top-level mapping, or fails validation.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise ConfigError(f"configuration file not found: {config_path}") from err
    except (OSError, UnicodeError) as err:
        raise ConfigError(f"cannot read configuration file: {config_path}: {err}") from err

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" (line {mark.line + 1}, column {mark.column + 1})"
        raise ConfigError(
            f"invalid YAML in configuration file {config_path}{location}: {err}"
        ) from err

    if not isinstance(data, dict):
        raise ConfigError(
            f"configuration file {config_path} must contain a top-level mapping, "
            f"got {type(data).__name__}"
        )

    try:
        return model.model_validate(data)
    except ValidationError as err:
        fields = ", ".join(".".join(str(part) for part in item["loc"]) for item in err.errors())
        raise ConfigError(f"invalid configuration in {config_path}: {fields}: {err}") from err
