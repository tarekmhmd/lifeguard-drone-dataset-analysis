"""TP-02: logging configuration is idempotent and writes UTF-8 log files."""
from __future__ import annotations

import logging
from pathlib import Path

from lifeguard.common.logging_setup import configure_logging


def test_configure_logging_idempotent() -> None:
    """Two calls leave exactly one stderr handler at the requested level."""
    configure_logging("DEBUG")
    configure_logging("DEBUG")

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    assert root.level == logging.DEBUG


def test_log_file_utf8(tmp_path: Path) -> None:
    """The file handler writes non-ASCII text as UTF-8."""
    log_path = tmp_path / "x.log"
    configure_logging("INFO", log_file=log_path)

    logging.getLogger("lifeguard.test").info("سباح ✓")

    assert log_path.exists()
    assert "سباح ✓" in log_path.read_text(encoding="utf-8")
