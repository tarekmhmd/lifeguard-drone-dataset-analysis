"""AI Lifeguard - two-page application: "AI Dashboard" and "Hardware" share ONE hardware service. Run with::

    python -m streamlit run src/lifeguard/app/lifeguard_app.py

Both pages run in this one Streamlit process and use the process-wide ``hardware_hub``: the same ESP32 connection,
e-stop latch, telemetry and command history, also when the pages are open in different tabs, windows or browsers
connected to this server. The AI runtime (one per process, as in the standalone dashboard) borrows that service,
including for its ``on_dispatch`` commands; stopping the AI leaves the hardware running on the Hardware page.

The standalone ``dashboard.py`` and ``hardware_demo.py`` commands still work as before (each with its own service).
"""
from pathlib import Path

import streamlit as st

_PAGES = Path(__file__).resolve().parent / "app_pages"

st.set_page_config(page_title="AI Lifeguard", layout="wide")
st.navigation([
    st.Page(_PAGES / "ai_dashboard.py", title="AI Dashboard", default=True),    # served at /
    st.Page(_PAGES / "hardware.py", title="Hardware", url_path="hardware"),     # served at /hardware
]).run()
