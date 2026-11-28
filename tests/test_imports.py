"""TP-01: Phase 1 packages import and the package version is 0.1.0."""
from __future__ import annotations

import importlib

import pytest

MODULES: tuple[str, ...] = (
    "lifeguard",
    "lifeguard.contracts",
    "lifeguard.common",
    "lifeguard.ai",
    "lifeguard.ai.perception",
    "lifeguard.ai.sim",
    "lifeguard.system",
    "lifeguard.system.geometry",
    "lifeguard.robotics",
    "lifeguard.hardware",
    "lifeguard.app",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_import_all_packages(module_name: str) -> None:
    """Import each Phase 1 package; the root package exposes version 0.1.0."""
    module = importlib.import_module(module_name)
    if module_name == "lifeguard":
        assert module.__version__ == "0.1.0"
