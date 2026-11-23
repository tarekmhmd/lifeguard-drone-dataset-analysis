"""Operator dashboard: controller (headless) and a Streamlit AppTest smoke test of the real page.

Operator actions must reach the runtime: Confirm/Dismiss change the risk engine state and the incident log.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lifeguard.app.dashboard_controller import (
    DashboardController,
    SourceChoice,
    annotate,
    health_rows,
    track_rows,
)
from lifeguard.contracts.enums import AckDecision, RiskLevel

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "configs" / "ai.yaml"
SYS = ROOT / "configs" / "system.yaml"


def _wait_for_alarm(ctl: DashboardController, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert ctl.runtime is not None
        for incident_id, level in ctl.runtime.snapshot().open_incidents.values():
            if level == RiskLevel.ALARM:
                return incident_id
        time.sleep(0.02)
    raise AssertionError("no ALARM within the timeout")


def test_controller_runs_and_operator_actions_reach_the_runtime() -> None:
    ctl = DashboardController()
    assert ctl.start(AI, SYS, SourceChoice("synthetic", realtime=False)), ctl.error
    try:
        incident_id = _wait_for_alarm(ctl)
        assert ctl.acknowledge(incident_id, AckDecision.CONFIRM, "   ") == "enter the lifeguard's name first"
        message = ctl.acknowledge(incident_id, AckDecision.CONFIRM, "lifeguard_1")
        assert "ALARM -> DISPATCH" in message and "lifeguards respond in person" in message
        assert ctl.runtime is not None
        assert ctl.runtime.store.dispatches()[0].confirmed_by == "lifeguard_1"
        assert ctl.acknowledge(incident_id, AckDecision.CONFIRM, "lifeguard_1").startswith("no change")
        snap = ctl.runtime.snapshot()
        rows = track_rows(snap)
        assert rows and {"track", "level", "risk", "state"} <= rows[0].keys()
        assert {r["component"] for r in health_rows(snap)} == {"ingest", "perception", "storage"}
        image = annotate(snap.frame, snap.result, snap.levels)
        assert image is not None and image.ndim == 3
    finally:
        ctl.stop()
    assert ctl.runtime is None and not ctl.running


def test_controller_reports_start_failures_instead_of_crashing(tmp_path: Path) -> None:
    ctl = DashboardController()
    assert not ctl.start(AI, SYS, SourceChoice("video", str(tmp_path / "missing.mp4")))
    assert ctl.error is not None and "missing.mp4" in ctl.error
    assert not ctl.start(tmp_path / "absent.yaml", SYS, SourceChoice("synthetic"))
    assert "ConfigError" in ctl.error
    assert ctl.acknowledge("x", AckDecision.CONFIRM, "a") == "system not running"


def test_annotate_without_frame_or_tracks() -> None:
    assert annotate(None, None, {}) is None
    out = annotate(np.zeros((10, 20, 3), np.uint8), None, {})
    assert out is not None and out.shape == (10, 20, 3)


def test_streamlit_page_start_and_confirm() -> None:
    """Drives the real Streamlit page: Start (synthetic), wait for an ALARM, press Confirm."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "lifeguard" / "app" / "dashboard.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.checkbox[0].uncheck()  # no real-time pacing
    at.sidebar.text_input(key="operator").input("lifeguard_1")
    next(b for b in at.sidebar.button if b.label == "Start").click()
    at.run()
    assert not at.exception

    deadline = time.monotonic() + 60
    alarm_shown = False
    while time.monotonic() < deadline and not alarm_shown:
        time.sleep(0.2)
        at.run()
        alarm_shown = any(m.value.startswith("**ALARM**") for m in at.markdown)
    assert alarm_shown, "the page never showed an ALARM incident"
    card = next(m.value for m in at.markdown if m.value.startswith("**ALARM**"))
    assert "**ALARM** · swimmer" in card and "Ã" not in card  # text encoding regression
    next(b for b in at.button if b.label == "Confirm emergency").click()
    at.run()
    assert not at.exception
    assert any("ALARM -> DISPATCH" in i.value for i in at.info)
    next(b for b in at.sidebar.button if b.label == "Stop").click()
    at.run()
