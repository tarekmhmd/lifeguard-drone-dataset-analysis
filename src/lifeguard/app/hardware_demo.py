"""Hardware Demo window (Streamlit): the ESP32 pool prototype on its own, simulated or real. Run with::

    python -m streamlit run src/lifeguard/app/hardware_demo.py

Configuration: ``configs/hardware.yaml`` (or the file named by ``LIFEGUARD_HARDWARE_CONFIG``). ``mode:
simulation`` shows an explicitly labelled software device; ``mode: esp32`` talks to the real ESP32 at the
configured address. Same ``HardwareService``, protocol and safety logic as the AI dashboard; switching between
simulation and the real device is a configuration change only.

This file is also the "Hardware" page of the two-page app (``lifeguard_app.py``, ``page(shared=True)``): there the
service is the process-wide shared one (``hardware_hub``) that the AI Dashboard page uses too.

Note: in ``esp32`` mode with ``role: server`` only one program can listen on the configured port. Two separate
``streamlit run`` processes never share a service; use the two-page app to see AI and hardware together.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from lifeguard.app import hardware_ui
from lifeguard.app.hardware_hub import shared_hub
from lifeguard.app.hardware_view import DEFAULT_HARDWARE_CONFIG, HardwareConsole, HardwareDemo
from lifeguard.hardware.service import HardwareService

REFRESH_S = 0.5


@st.cache_resource
def demo(path: str) -> HardwareDemo:
    """Standalone window: one hardware service per configuration file for this Streamlit process."""
    return HardwareDemo(path)


def _default_path() -> str:
    return os.environ.get("LIFEGUARD_HARDWARE_CONFIG", str(DEFAULT_HARDWARE_CONFIG))


def render(service: HardwareService, operator: str, key: str = "demo") -> None:
    """The full hardware view of one service (identical in the standalone window and the shared page)."""
    console = HardwareConsole(service)
    hw = service.status()
    left, right = st.columns([3, 2])
    with left:
        hardware_ui.status_header(hw)
        endpoint = hw.transport.endpoint if hw.transport else service.config.esp32.endpoint
        st.markdown(f"**Device id:** `{hw.device_id}` &nbsp; **Firmware:** "
                    f"`{(hw.device.firmware or 'unknown') if hw.device else '-'}` &nbsp; **Address:** `{endpoint}`")
        peer = hw.transport.peer if hw.transport is not None else None
        st.caption(f"Device connected from {peer}" if peer else "No device connected")   # always one line
        with st.container():
            hardware_ui.controls(console, hw, key, operator)
        if not hw.config_error:
            with st.expander("Test an output (short, bounded runs)", expanded=False):
                hardware_ui.output_tests(console, hw, key, operator)
            if hw.simulated:
                with st.expander("Simulate failures (SIMULATION ONLY)", expanded=False):
                    hardware_ui.simulation_controls(console, hw, key)
    with right:
        hardware_ui.details(hw)


# ------------------------------------------------------------------ standalone window (own service)
def sidebar() -> str:
    st.sidebar.header("Hardware configuration")
    path = st.sidebar.text_input("Hardware config file", _default_path())
    if st.sidebar.button("Reload configuration"):
        demo(path).close()
        demo.clear()
        st.rerun()
    st.sidebar.text_input("Operator name", key="hw_operator")
    st.sidebar.caption("Edit the YAML and press Reload: simulation <-> real ESP32 is a configuration change only.")
    return path


@st.fragment(run_every=REFRESH_S)
def live(path: str) -> None:
    d = demo(path)
    if d.service is None:
        st.info(f"Hardware integration is DISABLED in {d.path} (hardware.enabled: false). "
                "The AI demo runs without hardware. Set enabled: true and press Reload configuration.")
        return
    render(d.service, st.session_state.get("hw_operator", ""))


# ------------------------------------------------------------------ shared page (process-wide service)
def shared_sidebar() -> str:
    hub = shared_hub()
    st.sidebar.header("Shared hardware")
    path = st.sidebar.text_input("Hardware config file", _default_path(), key="shared_hw_path")
    if st.sidebar.button("Reload configuration", key="shared_hw_reload",
                         help="Rebuilds the shared service from this file (refused while the AI runtime runs)"):
        st.session_state["shared_hw_msg"] = hub.reload(path)
    if st.session_state.get("shared_hw_msg"):
        st.sidebar.caption(st.session_state["shared_hw_msg"])
    st.sidebar.text_input("Operator name", key="hw_operator")
    st.sidebar.caption("One hardware service for this whole application: the AI Dashboard page, every tab and "
                       "every browser connected to this server see the same connection, e-stop and commands.")
    return path


@st.fragment(run_every=REFRESH_S)
def shared_live(path: str) -> None:
    hub = shared_hub()
    service, reason = hub.for_hardware_page(path)     # created once per process; afterwards always the same
    if service is None:
        st.info(f"{reason}. The AI pages run without hardware. Set enabled: true and press Reload configuration.")
        return
    used = "in use by the AI runtime" if hub.ai_attached else "AI runtime not running"
    st.caption(f"Shared hardware service - configuration: {hub.source} - {used}")
    render(service, st.session_state.get("hw_operator", ""), key="shared")


def page(shared: bool = False) -> None:
    st.title("Lifeguard Hardware - ESP32 pool prototype")
    if shared:
        shared_live(shared_sidebar())
        return
    path = sidebar()
    st.caption(f"Configuration: {Path(path)}")
    live(path)


def main() -> None:
    st.set_page_config(page_title="Lifeguard Hardware", layout="wide")
    page()


if __name__ == "__main__":      # ``streamlit run`` executes this file as __main__; importing it renders nothing
    main()
