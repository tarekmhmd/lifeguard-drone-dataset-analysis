"""TP-13: UTC formatting, parsing and millisecond rounding."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from lifeguard.contracts.timeutil import UTC_PATTERN, add_seconds_utc, format_utc, utc_now


def test_format_utc_truncates_and_rejects_naive() -> None:
    """format_utc truncates to milliseconds and rejects naive datetimes."""
    aware = datetime(2026, 12, 10, 12, 34, 56, 123456, tzinfo=timezone.utc)
    assert format_utc(aware) == "2026-12-10T12:34:56.123Z"

    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc(aware.replace(tzinfo=None))


def test_add_seconds_utc_rounds_to_nearest_millisecond() -> None:
    """Seconds are added with nearest-millisecond rounding."""
    assert add_seconds_utc("2026-01-01T00:00:00.000Z", 0.1) == "2026-01-01T00:00:00.100Z"
    assert add_seconds_utc("2026-01-01T00:00:00.000Z", 0.7) == "2026-01-01T00:00:00.700Z"


def test_utc_now_matches_pattern() -> None:
    """utc_now returns a value matching UTC_PATTERN."""
    assert re.fullmatch(UTC_PATTERN, utc_now())
