"""Contract schema version constants and compatibility check."""
from __future__ import annotations

import re

from lifeguard.contracts.errors import ContractVersionError

SCHEMA_VERSION: str = "1.0.0"
SCHEMA_MAJOR: int = 1

_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def check_schema_version(version: str) -> None:
    """Validate a contract schema version against the supported major version.

    Args:
        version: Version string in ``X.Y.Z`` form.

    Returns:
        None when the version is well formed and its major part matches.

    Raises:
        ContractVersionError: If ``version`` is not semver ``X.Y.Z`` or its
            major part differs from :data:`SCHEMA_MAJOR`.
    """
    if _SEMVER_PATTERN.fullmatch(version) is None:
        raise ContractVersionError(
            f"invalid schema_version {version!r}; expected X.Y.Z (current {SCHEMA_VERSION})"
        )
    major = int(version.split(".", 1)[0])
    if major != SCHEMA_MAJOR:
        raise ContractVersionError(
            f"incompatible schema_version {version}: major {major} != {SCHEMA_MAJOR} "
            f"(current {SCHEMA_VERSION})"
        )
