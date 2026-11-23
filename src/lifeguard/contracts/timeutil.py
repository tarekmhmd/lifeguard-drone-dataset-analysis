"""UTC formatting and parsing helpers for the contract."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

UTC_PATTERN: str = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"

_MS_PER_SECOND = 1000


def format_utc(dt: datetime) -> str:
    """Format a timezone-aware datetime as UTC with millisecond precision.

    Args:
        dt: Timezone-aware datetime.

    Returns:
        ``YYYY-MM-DDTHH:MM:SS.mmmZ``, with microseconds truncated to milliseconds.

    Raises:
        ValueError: If ``dt`` is naive (has no UTC offset).
    """
    if dt.utcoffset() is None:
        raise ValueError("format_utc requires a timezone-aware datetime")
    utc = dt.astimezone(timezone.utc)
    milliseconds = utc.microsecond // _MS_PER_SECOND
    return (
        f"{utc.year:04d}-{utc.month:02d}-{utc.day:02d}"
        f"T{utc.hour:02d}:{utc.minute:02d}:{utc.second:02d}"
        f".{milliseconds:03d}Z"
    )


def utc_now() -> str:
    """Return the current UTC time formatted by :func:`format_utc`."""
    return format_utc(datetime.now(timezone.utc))


def add_seconds_utc(t_utc: str, seconds: float) -> str:
    """Add seconds to a UTC_PATTERN timestamp using nearest-millisecond rounding.

    Args:
        t_utc: Timestamp matching :data:`UTC_PATTERN`.
        seconds: Seconds to add; may be negative.

    Returns:
        The resulting timestamp in ``YYYY-MM-DDTHH:MM:SS.mmmZ`` form.

    Raises:
        ValueError: If ``t_utc`` does not match :data:`UTC_PATTERN`.
    """
    if re.fullmatch(UTC_PATTERN, t_utc) is None:
        raise ValueError(f"invalid UTC timestamp: {t_utc!r}")
    parsed = datetime.fromisoformat(t_utc[:-1] + "+00:00")
    delta = timedelta(milliseconds=round(seconds * _MS_PER_SECOND))
    return format_utc(parsed + delta)
