# Demo day — quick start (one page)

Full procedure, configuration blocks and troubleshooting: [`DEMO_DAY_RUNBOOK.md`](DEMO_DAY_RUNBOOK.md).
Physical ESP32 / motors / pool camera: **NOT VERIFIED** until the checklist is done with the hardware team.

**One-time (already done? skip):** create the demo system config

```powershell
cd E:\lifeguardDron\lifeguard-system
Copy-Item configs\system_demo_drone.yaml configs\system_demo_hardware.yaml
Add-Content configs\system_demo_hardware.yaml "`nhardware_file: hardware.yaml"
```

**Before power-up:** physical e-stop PRESSED · outputs disconnected · `configs\hardware.yaml` has
`allow_physical_actuation: false`.

**Real ESP32:** in `configs\hardware.yaml` set `mode: esp32`, `device_id`, and either `role: server` +
`bind_host: <laptop IP from ipconfig>` + `port`, or `role: client` + `device_host: <ESP32 IP>` + `device_port`,
plus the real component ids. **No ESP32:** leave `mode: simulation` (blue SIMULATION banner).

**Launch** (one PowerShell window, keep it open):

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
# only if the team uses a token:  $env:LIFEGUARD_HW_TOKEN = "<token>"
E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.common.models verify --registry models/registry.yaml
$env:LIFEGUARD_AI_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml"
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_demo_hardware.yaml"
E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/lifeguard_app.py
```

`verify` must print 9× `ok`. Browser: http://localhost:8501 (AI Dashboard) and http://localhost:8501/hardware.

| Step | Action | Success looks like |
| --- | --- | --- |
| 1 | Hardware page | REAL ESP32 (or SIMULATION) banner → **CONNECTED** "identity verified", telemetry age < 2 s |
| 2 | AI sidebar: Input **camera**, index `0`/`1` or `rtsp://...` **+ Enter**, Lifeguard name **+ Enter**, **Start** | live image, boxes + track ids + skeletons, `error=0` |
| 3 | Hardware page **EMERGENCY STOP** | **EMERGENCY STOP ACTIVE**, `emergency_stop completed`; AI panel "EMERGENCY STOP LATCHED" |
| 4 | Physical e-stop test, release it, Clear e-stop (name + tick box) | `clear_emergency_stop completed`, banner gone |
| 5 | AI: **Stop**, Input **video** `E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4` + Enter, **Start** | PRE_ALERT, then ALARM (~20 s) |
| 6 | **Confirm emergency** (twice if it was a PRE_ALERT) | DISPATCH, `[SIMULATION] mission-0001 started`; hardware row if `on_dispatch` is set |
| 7 | Shutdown: AI **Stop** → Hardware **Disconnect** → `Ctrl+C` → e-stop pressed → power off | DISCONNECTED, `safe_stop` by `shutdown` completed |

**If something fails:** wrong folder → `cd` · `No module named ultralytics` → use `.venv-train` ·
`No module named lifeguard.app.hardware_hub` → set `PYTHONPATH` · FAULT "cannot listen" → port busy / wrong
`bind_host` · stays DISCONNECTED → IP, port, firewall, same Wi-Fi · CONFIGURATION ERROR → read the reason, fix
`hardware.yaml`, Reload (AI stopped) · camera not available → other index, close the Camera app · nothing detected on
the prototype → detector not validated on that scene → use the demo videos.

**Fallback (always works):** `mode: simulation` + demo videos (`demo_06` alarm, `demo_11` early PRE_ALERT,
`demo_05` normal swim, `demo_01` pose). Say clearly that the hardware shown is simulated.

**Never:** enable `allow_physical_actuation` without the team's approval and a tested physical e-stop; claim
physical results you did not observe.
