"""Root logging configuration shared by every lifeguard component."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

# Log line format used by every lifeguard component.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Accepted level names mapped to the standard logging levels.
_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class _Utf8LfFileHandler(logging.FileHandler):
    """Append UTF-8 log lines with LF newlines on every platform."""

    def _open(self) -> TextIO:
        return open(self.baseFilename, self.mode, encoding=self.encoding, newline="\n")


def configure_logging(level: str = "INFO", log_file: Path | str | None = None) -> None:
    """Configure the root logger with stderr output and an optional UTF-8 log file.

    Calling this function repeatedly does not duplicate handlers: existing root
    handlers are removed before the new ones are installed.

    Args:
        level: Logging level name, for example ``"DEBUG"`` or ``"INFO"``.
        log_file: Optional path for the UTF-8 log file. Parent directories are
            created when needed.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    try:
        numeric_level = _LEVELS[level.upper()]
    except KeyError as err:
        raise ValueError(f"invalid logging level: {level!r}") from err

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _Utf8LfFileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric_level)
