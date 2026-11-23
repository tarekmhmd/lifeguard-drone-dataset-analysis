"""Exception hierarchy for the Phase 1 lifeguard package."""
from __future__ import annotations


class LifeguardError(Exception):
    """Base class for all project errors."""


class ContractVersionError(LifeguardError):
    """Incompatible schema_version."""


class ContractValidationError(LifeguardError):
    """Wraps pydantic.ValidationError with file/line context."""

    source: str | None
    line: int | None

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        line: int | None = None,
    ) -> None:
        """Store the message and optional source file/line context.

        Args:
            message: Human-readable description of the validation failure.
            source: File path or other origin of the invalid data.
            line: 1-based line number in ``source``.
        """
        self.source = source
        self.line = line
        super().__init__(self._format(message))

    def _format(self, message: str) -> str:
        """Append the available source and line context to the message."""
        context: list[str] = []
        if self.source is not None:
            context.append(self.source)
        if self.line is not None:
            context.append(f"line {self.line}")
        if not context:
            return message
        return f"{message} ({', '.join(context)})"


class ConfigError(LifeguardError):
    """Missing/invalid configuration file or field."""


class ModelLoadError(LifeguardError):
    """Perception engine could not load its models/resources."""


class FixtureError(ModelLoadError):
    """Mock fixture missing, unreadable or invalid."""
