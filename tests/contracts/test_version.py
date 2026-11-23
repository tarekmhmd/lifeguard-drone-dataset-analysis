"""TP-10: schema version compatibility checks."""
from __future__ import annotations

import pytest

from lifeguard.contracts.errors import ContractVersionError
from lifeguard.contracts.version import check_schema_version


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        pytest.param("1.0.0", True, id="current"),
        pytest.param("1.4.2", True, id="newer-minor-and-patch"),
        pytest.param("2.0.0", False, id="newer-major"),
        pytest.param("0.9.0", False, id="older-major"),
        pytest.param("1.0", False, id="missing-patch"),
        pytest.param("abc", False, id="not-semver"),
    ],
)
def test_check_schema_version(version: str, accepted: bool) -> None:
    """Compatible semver is accepted; malformed or other-major versions raise."""
    if accepted:
        assert check_schema_version(version) is None
    else:
        with pytest.raises(ContractVersionError):
            check_schema_version(version)
