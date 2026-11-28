"""Operator dashboard (Streamlit, roadmap P3-5). Run with::

    streamlit run src/lifeguard/app/dashboard.py

Needs ``pip install -e ".[dashboard]"``. One runtime is shared by every browser tab of this server (a single
operator station). Alerts are machine-generated and unconfirmed; the operator decides.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from lifeguard.app import hardware_ui
from lifeguard.app.dashboard_controller import (
    DashboardController,
    SourceChoice,
    annotate,
    clean_input,
    hardware_badges,
    health_rows,
    track_rows,
)
from lifeguard.app.hardware_hub import shared_hub
from lifeguard.app.hardware_view import hardware_state
from lifeguard.app.runtime import Snapshot
from lifeguard.contracts.enums import AckDecision, RiskLevel
from lifeguard.hardware.models import HardwareStatus
from lifeguard.system.alerts.service import Priority

ROOT = Path(__file__).resolve().parents[3]
REFRESH_S = 1.0


@st.cache_resource
def controller(shared: bool = False) -> DashboardController:
    """One controller (one runtime) per Streamlit process. ``shared``: the two-page app, where the hardware
    service belongs to the process-wide hub and is the same one the Hardware page shows."""
    return DashboardController(hub=shared_hub() if shared else None)


def sidebar(ctl: DashboardController) -> None:
    st.sidebar.header("System")
    # Defaults can be preset per launch (e.g. a demo): LIFEGUARD_AI_CONFIG / LIFEGUARD_SYSTEM_CONFIG.
    ai_default = os.environ.get("LIFEGUARD_AI_CONFIG", str(ROOT / "configs" / "ai.yaml"))
    sys_default = os.environ.get("LIFEGUARD_SYSTEM_CONFIG", str(ROOT / "configs" / "system.yaml"))
    ai_cfg = st.sidebar.text_input("AI config", ai_default)
    sys_cfg = st.sidebar.text_input("System config", sys_default)
    kind = st.sidebar.radio("Input", ["synthetic", "video", "camera"],
                            help="synthetic = MOCK engine replaying a fixture (no camera, no weights)")
    value = ""
    if kind == "video":
        value = st.sidebar.text_input("Video file")
    elif kind == "camera":
        value = st.sidebar.text_input("Camera index or URL", "0")
    realtime = st.sidebar.checkbox("Real-time pacing", value=True)
    start, stop = st.sidebar.columns(2)
    if start.button("Start", type="primary", disabled=ctl.running):
        ctl.start(Path(clean_input(ai_cfg)), Path(clean_input(sys_cfg)),
                  SourceChoice(kind, clean_input(value), realtime))
    # Enabled while a session exists, also after a replay has ended (closes it and its incident log).
    if stop.button("Stop", disabled=ctl.runtime is None):
        ctl.stop()
    st.sidebar.text_input("Lifeguard name", key="operator")
    if ctl.error:
        st.sidebar.error(ctl.error)


def incidents_panel(ctl: DashboardController, snap: Snapshot) -> None:
    st.subheader("Open incidents")
    if not snap.open_incidents:
        st.caption("none")
    for uid, (incident_id, level) in sorted(snap.open_incidents.items(), key=lambda kv: kv[1][0]):
        box = st.container(border=True)
        box.markdown(f"**{level.value}** · swimmer `{uid}` · incident `{incident_id}`  \n"
                     "_Machine-generated alert, unconfirmed: look at the swimmer._")
        confirm, dismiss = box.columns(2)
        operator = st.session_state.get("operator", "")
        confirmable = level in (RiskLevel.WATCH, RiskLevel.PRE_ALERT, RiskLevel.ALARM)
        dismissable = confirmable or level == RiskLevel.DISPATCH
        if confirmable and confirm.button("Confirm emergency", key=f"c-{incident_id}", type="primary"):
            st.session_state["last_action"] = ctl.acknowledge(incident_id, AckDecision.CONFIRM, operator)
        if dismissable and dismiss.button("Dismiss (false alarm)", key=f"d-{incident_id}"):
            st.session_state["last_action"] = ctl.acknowledge(incident_id, AckDecision.DISMISS, operator)
    if st.session_state.get("last_action"):
        st.info(st.session_state["last_action"])


def response_panel(ctl: DashboardController, snap: Snapshot) -> None:
    st.subheader("Response")
    st.json(snap.response_status, expanded=False)
    if snap.outcomes:
        last = snap.outcomes[-1]
        st.write(f"`{last.kind.value}` {last.detail}")
        if last.kind.value == "awaiting_confirmation" and st.button("Confirm (simulated) release", type="primary"):
            st.session_state["last_action"] = ctl.confirm_release(st.session_state.get("operator", ""))


def hardware_panel(ctl: DashboardController, hw: HardwareStatus) -> None:
    """ESP32 pool prototype: same service, panel and controls as the Hardware Demo window (compact)."""
    st.subheader("Hardware")
    show = {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}
    for level, text in hardware_badges(hw):
        show[level](text)
    state = hardware_state(hw)
    st.markdown(f"**Connection: {state.label}** - {state.detail}")
    operator = st.session_state.get("operator", "")
    hardware_ui.controls(ctl.hardware_console, hw, "hw", operator)
    hardware_ui.details(hw, compact=True)
    st.caption("Full hardware view: the Hardware page (same service)" if ctl.hub is not None else
               "Full hardware window: `python -m streamlit run src/lifeguard/app/hardware_demo.py`")


@st.fragment(run_every=REFRESH_S)
def live_view(shared: bool = False) -> None:
    ctl = controller(shared)
    rt = ctl.runtime
    if rt is None:
        st.info("Not running. Choose an input in the sidebar and press Start.")
        return
    snap = rt.snapshot()
    left, right = st.columns([3, 2])
    with left:
        image = annotate(snap.frame, snap.result, snap.levels)
        if image is not None:
            st.image(image, caption=f"frame {snap.result.frame_idx if snap.result else '-'}", width="stretch")
        st.caption(snap.summary.line())
        st.subheader("Swimmers")
        st.dataframe(track_rows(snap), hide_index=True, width="stretch")
    with right:
        faults = [n for n in snap.notifications[-20:] if n.priority == Priority.ALARM]
        if faults:
            st.error(faults[-1].message)
        incidents_panel(ctl, snap)
        response_panel(ctl, snap)
        st.subheader("Health")
        st.dataframe(health_rows(snap), hide_index=True, width="stretch")
        if snap.hardware is not None:   # only when hardware.enabled is true in the system config
            hardware_panel(ctl, snap.hardware)
    with st.expander("Notifications"):
        for note in reversed(snap.notifications[-50:]):
            st.write(f"`{note.priority.value}` {note.t_utc} {note.message}")
    with st.expander("Incident history"):
        st.dataframe(rt.store.incidents(), hide_index=True, width="stretch")
    with st.expander("Components"):
        st.json(snap.components)


def page(shared: bool = False) -> None:
    """The AI dashboard page (standalone script, or the "AI Dashboard" page of the two-page app)."""
    st.title("Pool lifeguard assistant")
    st.caption("Decision support only. It does not replace lifeguards. Drone response is simulation only.")
    sidebar(controller(shared))
    live_view(shared)


def main() -> None:
    st.set_page_config(page_title="Pool lifeguard assistant", layout="wide")
    page()


if __name__ == "__main__":      # ``streamlit run`` executes this file as __main__; importing it renders nothing
    main()
