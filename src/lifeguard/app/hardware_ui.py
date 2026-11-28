"""Streamlit rendering of the hardware panel; used by the AI dashboard and the Hardware Demo window.

All data comes from ``HardwareStatus`` and all actions go through ``HardwareConsole`` (-> ``HardwareService``,
where every safety check applies). Controls that cannot work right now are disabled with the reason shown.
"""
from __future__ import annotations

import html
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

from lifeguard.app.hardware_view import (
    HardwareConsole,
    clear_blockers,
    hardware_rows,
    hardware_state,
    motion_blockers,
)
from lifeguard.hardware.models import HardwareStatus

_COLOURS = {"success": "#1e8e3e", "warning": "#e37400", "error": "#c5221f", "neutral": "#5f6368"}


def _box(text: str, sub: str, colour: str, big: bool = True) -> None:
    size = "2.0rem" if big else "1.2rem"
    st.markdown(
        f"<div style='background:{colour};color:#fff;border-radius:10px;padding:14px 18px;margin:4px 0 10px 0'>"
        f"<div style='font-size:{size};font-weight:700;letter-spacing:.03em'>{html.escape(text)}</div>"
        f"<div style='font-size:.95rem;opacity:.95'>{html.escape(sub)}</div></div>",
        unsafe_allow_html=True)


def _say(key: str, message: str, redraw: bool = True) -> None:
    """Remember the action's result; ``redraw`` updates the status at once (top-level controls only: a full
    rerun would collapse the expander the operator is working in, and the 0.5 s refresh follows anyway)."""
    st.session_state[f"{key}-msg"] = message
    if not redraw:
        return
    try:
        st.rerun(scope="fragment")         # allowed only while the fragment reruns on its own (timer)
    except StreamlitAPIException:
        st.rerun()                         # the click came with a full-page run


def _table(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Display copy with text cells: columns mixing numbers and 'unavailable' would not render otherwise."""
    return [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in rows]


def status_header(hw: HardwareStatus, big: bool = True) -> None:
    """Large connection indicator, the mode label and the e-stop banner.

    One container: banners that come and go must not shift (and so re-create / collapse) the widgets below.
    """
    with st.container():
        _status_header(hw, big)


def _status_header(hw: HardwareStatus, big: bool) -> None:
    if hw.config_error:
        _box("CONFIGURATION ERROR", hw.config_error, _COLOURS["error"], big)
        return
    if hw.simulated:
        _box("SIMULATION", "simulated ESP32 in software - no network, nothing physical moves", "#1a73e8", big=False)
    else:
        gate = "ENABLED" if hw.physical_actuation_allowed else "DISABLED (stop commands only)"
        _box("REAL ESP32", f"physical hardware link - physical actuation {gate}", "#202124", big=False)
    state = hardware_state(hw)
    _box(state.label, state.detail, _COLOURS[state.level], big)
    if hw.estop_latched:
        _box("EMERGENCY STOP ACTIVE", "movement refused until an authorised operator clears it", "#a50e0e", big)
    if hw.faults:
        st.error(f"Device faults: {', '.join(hw.faults)}")
    if hw.link.value == "connected" and hw.telemetry_stale:
        st.warning("Telemetry is STALE: movement refused; anything moving is safe-stopped.")


def controls(console: HardwareConsole, hw: HardwareStatus, key: str, operator: str) -> None:
    """Connect / Disconnect / Refresh / EMERGENCY STOP / Safe stop and the guarded e-stop reset."""
    if hw.config_error:
        st.caption("Fix the hardware configuration file and reload; no hardware control is possible.")
        return
    # Always full width and never disabled: an emergency stop must be readable and reachable at any window size.
    if st.button("EMERGENCY STOP", key=f"{key}-estop", type="primary", width="stretch"):
        _say(key, console.command("estop", operator))
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Connect", key=f"{key}-connect", disabled=hw.running, width="stretch"):
        _say(key, console.connect(operator))
    if c2.button("Disconnect", key=f"{key}-disconnect", disabled=not hw.running, width="stretch"):
        _say(key, console.disconnect(operator))
    if c3.button("Refresh", key=f"{key}-refresh", width="stretch"):
        _say(key, "status refreshed")
    if c4.button("Safe stop", key=f"{key}-safe", width="stretch"):
        _say(key, console.command("safe_stop", operator))
    # Always rendered (stable layout: an element appearing here would re-create, i.e. collapse, the expanders below).
    st.caption(f"Last action: {st.session_state.get(f'{key}-msg') or '-'}")
    st.caption("A software stop does not replace the physical emergency stop on the prototype.")
    with st.expander("Clear emergency stop (authorised operator)", expanded=hw.estop_latched):
        blockers = clear_blockers(hw)
        for reason in blockers:
            st.caption(f"Unavailable: {reason}")
        who = st.text_input("Operator name", value=operator, key=f"{key}-clear-name")
        if hw.estop_clear_operators:
            st.caption(f"Authorised operators: {', '.join(hw.estop_clear_operators)}")
        pin = st.text_input("PIN", type="password", key=f"{key}-clear-pin") if hw.estop_clear_needs_pin else None
        sure = st.checkbox("The pool area is safe and the physical e-stop is released", key=f"{key}-clear-ok")
        if st.button("Clear e-stop", key=f"{key}-clear", disabled=bool(blockers)):
            _say(key, console.command("clear_estop", who, pin=pin, confirmed=sure))


def output_tests(console: HardwareConsole, hw: HardwareStatus, key: str, operator: str) -> None:
    """Short, bounded test runs of one motor / actuator (all gates enforced by the command manager)."""
    service = console.service
    if service is None or hw.config_error:
        return
    comps = service.config.components
    blockers = motion_blockers(hw)
    for reason in blockers:
        st.caption(f"Unavailable: {reason}")
    disabled = bool(blockers) or not operator.strip()
    if not operator.strip():
        st.caption("Enter an operator name to send test commands.")
    if comps.motors:
        m1, m2, m3, m4 = st.columns([2, 2, 2, 1])
        motor = m1.selectbox("Motor", list(comps.motors), key=f"{key}-motor")
        spec = comps.motors[motor]
        speed = m2.slider("Speed", -spec.max_speed, spec.max_speed, min(0.2, spec.max_speed), 0.05,
                          key=f"{key}-speed")
        secs = m3.slider("Duration (s)", 0.1, spec.max_duration_s, min(1.0, spec.max_duration_s), 0.1,
                         key=f"{key}-dur")
        if m4.button("Run", key=f"{key}-run", disabled=disabled):
            run_params = {"speed": speed, "duration_s": secs}
            _say(key, console.command("set_motor", operator, motor, run_params), redraw=False)
    if comps.actuators:
        a1, a2, a3, a4 = st.columns([2, 2, 2, 1])
        act = a1.selectbox("Actuator", list(comps.actuators), key=f"{key}-act")
        spec_a = comps.actuators[act]
        state = a2.selectbox("State", list(spec_a.states), key=f"{key}-state")
        secs_a = a3.slider("Duration (s) ", 0.1, spec_a.max_duration_s, min(1.0, spec_a.max_duration_s), 0.1,
                           key=f"{key}-adur")
        if a4.button("Set", key=f"{key}-set", disabled=disabled):
            params: dict[str, Any] = {"state": state}
            if state != spec_a.safe_state:
                params["duration_s"] = secs_a
            _say(key, console.command("set_actuator", operator, act, params), redraw=False)
    p1, p2 = st.columns(2)
    if p1.button("Ping", key=f"{key}-ping", disabled=not operator.strip()):
        _say(key, console.command("ping", operator), redraw=False)
    if p2.button("Request status", key=f"{key}-status", disabled=not operator.strip()):
        _say(key, console.command("request_status", operator), redraw=False)
    if st.session_state.get(f"{key}-msg"):
        st.caption(f"Last action: {st.session_state[f'{key}-msg']}")


def simulation_controls(console: HardwareConsole, hw: HardwareStatus, key: str) -> None:
    """Failure injection for the simulated device only (never shown for a real ESP32)."""
    if not hw.simulated or hw.config_error:
        return
    st.caption("SIMULATION ONLY - these act on the software device, never on real hardware.")
    actions = [("Device offline", "go_offline"), ("Device online", "come_online"),
               ("Freeze telemetry", "freeze_telemetry"), ("Resume telemetry", "resume_telemetry"),
               ("Inject motor fault", "inject_fault"), ("Clear faults", "clear_faults"),
               ("Corrupt message", "corrupt_message"), ("Press physical e-stop", "press_physical_estop"),
               ("Release physical e-stop", "release_physical_estop"),
               ("Firmware ignores commands", "ignore_commands"), ("Firmware answers again", "answer_commands")]
    cols = st.columns(2)
    for i, (label, action) in enumerate(actions):
        if cols[i % 2].button(label, key=f"{key}-sim-{action}", width="stretch"):
            _say(key, console.simulate(action), redraw=False)
    if st.session_state.get(f"{key}-msg"):
        st.caption(f"Last action: {st.session_state[f'{key}-msg']}")


def details(hw: HardwareStatus, compact: bool = False) -> None:
    """Telemetry, link, commands and communication log (missing data shown as missing)."""
    rows = hardware_rows(hw)
    source = "SIMULATED telemetry" if hw.simulated else "device telemetry"
    if hw.telemetry is None:
        st.info("No telemetry received" + ("" if hw.link.value == "connected" else " (device not connected)"))
    else:
        age = f"{hw.telemetry_age_s:.1f} s old" if hw.telemetry_age_s is not None else ""
        st.caption(f"{source}, {age}{' - STALE' if hw.telemetry_stale else ''}")
        for title, name in (("Motors", "motors"), ("Actuators", "actuators"), ("Sensors", "sensors")):
            if rows[name]:
                st.markdown(f"**{title}**")
                st.dataframe(_table(rows[name]), hide_index=True, width="stretch")
        if not hw.telemetry.motors:
            st.caption("Motor status: not reported by the device.")
    with st.expander("Link and communication", expanded=not compact):
        st.dataframe(_table(rows["link"]), hide_index=True, width="stretch")
    with st.expander("Commands and acknowledgements", expanded=not compact):
        if rows["commands"]:
            st.dataframe(_table(rows["commands"]), hide_index=True, width="stretch")
        else:
            st.caption("No commands yet.")
    with st.expander("Device communication log", expanded=False):
        st.caption("Link events, hello/welcome, commands and acks; telemetry and heartbeats are counted, not "
                   "listed; tokens are never shown.")
        st.dataframe(_table(rows["comm"]), hide_index=True, width="stretch")
    if hw.errors:
        with st.expander(f"Hardware errors ({len(hw.errors)})", expanded=False):
            for err in reversed(hw.errors):
                st.write(f"- {err}")
