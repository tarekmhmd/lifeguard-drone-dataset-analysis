# Demo-day runbook — camera + AI + ESP32 hardware (professor demonstration)

Written 2026-11-28 from the code in `E:\lifeguardDron\lifeguard-system` (not from plans). Every command below was
run on this laptop on 2026-11-28 unless it is marked **NOT VERIFIED**. One-page version:
[`DEMO_DAY_QUICKSTART.md`](DEMO_DAY_QUICKSTART.md). Hardware protocol reference:
[`HARDWARE_INTEGRATION.md`](HARDWARE_INTEGRATION.md).

## 0. What is verified and what is not

| Item | Status (2026-11-28) |
| --- | --- |
| Real AI stack on a recorded pool clip: Z5cc detector + ByteTrack + pose + risk engine | ✅ `demo_06`: 574 frames, 0 errors, 3 tracks, 1 ALARM (submersion) |
| Detector-distress PRE_ALERT → ALARM → lifeguard CONFIRM → DISPATCH → simulated drone | ✅ scripted run with real models |
| `on_dispatch` hardware command after CONFIRM (simulated device) | ✅ `set_actuator indicator_1` accepted |
| Two-page app (AI Dashboard + Hardware) with real models and a video | ✅ 258 frames, 0 errors, hardware CONNECTED, AI attached to the shared service |
| Model registry integrity (all 9 entries) | ✅ `verify` → all `ok` |
| Laptop camera index 0 | ✅ opens, 640×480 (OpenCV default) |
| Hardware software against a **fake** ESP32 over real WebSockets (both roles) | ✅ automated tests (121 hardware tests, 653 total) |
| **The team's pool camera** | ⚠️ NOT VERIFIED (index/URL unknown) |
| **Detector on the prototype scene** (miniature pool, figures, lab lighting) | ⚠️ NOT VERIFIED — the detector was trained on real side-view pool video |
| **Physical ESP32, firmware, motors, actuators, sensors, wiring, network** | ⚠️ NOT VERIFIED — nothing physical has been tested |
| Physical e-stop, link-loss failsafe on the device | ⚠️ NOT VERIFIED (firmware/electrical responsibility) |

Say this on the day: *"The software side, including the hardware link, is tested against a simulated and a fake
ESP32. The physical prototype is tested only as far as the checklist in section 5 got today."*

## 1. What the system actually is (as implemented)

```
Camera index / RTSP URL / video file
  -> YOLO11s Z5cc pool detector (configs/ai_z5cc_pose.yaml, models/z5cc_s448_final.pt, imgsz 448)
  -> ByteTrack (project implementation: src/lifeguard/ai/tracking/bytetrack.py)
  -> pose keypoints on tracked boxes (models/yolo11s-pose.pt) - DISPLAY ONLY, never used for risk
  -> rule-based risk engine (submersion timing; detector "drowning" 2 s mean -> PRE_ALERT if enabled)
  -> alert service -> dashboard card -> lifeguard CONFIRM (human decision)
  -> response platform: manual (system.yaml) or SIMULATED drone (system_demo_drone.yaml)
     + hardware on_dispatch commands (only if configured) via the shared HardwareService
  -> ESP32 over WebSocket (or the simulated device), shown on the Hardware page
```

Two Python environments exist — use the right one:

| Environment | Has | Use for |
| --- | --- | --- |
| `E:\lifeguardDron\.venv-train` | ultralytics 8.3.40, torch 2.6 CUDA (GTX 1650), streamlit, websockets | **The demo** (real models). Its installed `lifeguard` package points to the other copy `E:\lifeguard-github-upload`, which has **no hardware code** → always set `PYTHONPATH` to this repository's `src` (commands below do). |
| `E:\lifeguardDron\.venv` | pytest, streamlit, websockets; **no ultralytics/torch** | Running the test suite; mock/synthetic demo only |

## 2. Physical setup

### 2.1 Confirmed by the software (you can rely on these)

- The computer runs everything AI-related; the ESP32 only exchanges JSON messages over **one WebSocket over
  Wi-Fi/LAN** (no serial, no MQTT, no discovery). Protocol: `HARDWARE_INTEGRATION.md` §4 and §10.
- Two link roles (`hardware.esp32.role` in `configs/hardware.yaml`):
  - `server` (default): the **computer listens** on `bind_host:port`, the **ESP32 connects** to the computer's IP.
    The firmware must contain the computer's IP, port and path (`/lifeguard-hw`).
  - `client`: the **computer connects** to the ESP32's WebSocket server at `device_host:device_port`.
  The firmware decides which one it implements — **ask the hardware team** (section 8).
- On every (re)connect the computer sends `safe_stop` (or `emergency_stop` while its e-stop latch is set).
- With `allow_physical_actuation: false` (default) the computer **never sends** `set_motor` / `set_actuator`
  to a real ESP32 — only `ping`, `request_status`, `safe_stop`, `emergency_stop`, `clear_emergency_stop`.
- The camera is any device OpenCV can open: a USB/built-in camera **index** (`0`, `1`, ...) or a **stream URL**
  (e.g. an IP camera's `rtsp://...` URL). Selected in the dashboard sidebar, not in a YAML file.

### 2.2 Must come from the hardware team (do NOT guess)

GPIO pins, motor driver type, supply voltages, current limits, fuse ratings, wiring of the physical e-stop, which
outputs exist and their safe states, sensor types/units/ranges, the ESP32's IP address (or DHCP reservation), the
Wi-Fi network, the firmware's role (server/client), port, path, device id, token, firmware version string. None of
these are defined anywhere in this repository; the component names `motor_1`, `indicator_1`, `sensor_1/2` are
**placeholders**.

### 2.3 Connection layout (fill in the blanks with the team)

```
[Pool camera] --USB or LAN (RTSP)--> [Laptop: AI app, port 8501 browser UI]
                                           |  Wi-Fi / LAN (same network, isolated SSID recommended)
                                           |  WebSocket ws://<IP>:<port>/lifeguard-hw
                                      [ESP32] --GPIO--> [motor drivers / relays]  --> motors, pumps, indicator
                                         ^                       ^
                                         |                       |
                              [sensors]  +      [PHYSICAL E-STOP: cuts actuator power directly, not via the ESP32]
```

Network preparation on the laptop (PowerShell):

- Find the laptop's IP on the demo network: `ipconfig` → "IPv4 Address" of the Wi-Fi/Ethernet adapter in use.
- `role: server` only — allow inbound TCP on the chosen port for the **Private** profile (run PowerShell as
  administrator; replace `<PORT>`):
  `New-NetFirewallRule -DisplayName "Lifeguard ESP32" -Direction Inbound -Protocol TCP -LocalPort <PORT> -Action Allow -Profile Private`
  (**NOT VERIFIED** on this laptop; make sure the Wi-Fi network is set to *Private* in Windows.)
- `role: client` only — check the ESP32 answers: `Test-NetConnection <ESP32_IP> -Port <PORT>` →
  `TcpTestSucceeded : True`.

## 3. Configuration (exact files and supported keys only)

### A. Camera — there is no camera YAML

The camera is chosen at start time:

- Dashboard sidebar → **Input: camera** → **Camera index or URL** (default `0`) → press Enter → Start.
- Headless: `--camera 0` or `--camera rtsp://...` (`python -m lifeguard.app.main --help`).
- `parse_source` (`src/lifeguard/system/ingest/live.py`): digits = device index; anything else is passed to
  OpenCV unchanged (RTSP/HTTP URL or a device path).

**Not configurable today (missing features — do not look for keys):** camera resolution, camera FPS, exposure,
codec, OpenCV backend. The camera delivers OpenCV's default (measured: laptop camera index 0 → 640×480). Live
camera processing runs as fast as frames arrive (no pacing). The detector letterboxes to 448 px either way.

Related keys that DO exist (in the system YAML, block `health:` — defaults are fine):
`camera_timeout_s: 3.0` (no frame → INGEST error), `frozen_after_s: 3.0` (identical frames → error),
`fps_floor: 5.0`. Optional pool calibration: `geometry.calibration: <file>.json`, created with
`python -m lifeguard.system.geometry.calibrate` (without it the runtime uses and announces a PLACEHOLDER rectangle;
positions are approximate — acceptable for the demo).

### B. Hardware — `E:\lifeguardDron\lifeguard-system\configs\hardware.yaml`

This is the only hardware file you edit. Today it is the shipped simulation default (`enabled: true`,
`mode: simulation`). For the real ESP32, change only the lines marked **REPLACE**; everything else can stay.
Placeholders in `<...>` deliberately fail validation (the app then shows CONFIGURATION ERROR instead of
connecting somewhere wrong).

```yaml
hardware:
  enabled: true                    # REQUIRED  (false = AI only)
  mode: esp32                      # REQUIRED  REPLACE: simulation -> esp32 for the real device
  device_id: pool-prototype-01     # REQUIRED  REPLACE with the firmware's DEVICE_ID (letters, digits, _ . -)
  auto_connect: true               # optional  open the link at start-up (false: press Connect)
  allow_physical_actuation: false  # optional  KEEP false until section 6 step "first movement" is approved

  telemetry_interval_s: 0.5        # optional  (sent to the device in "welcome")
  heartbeat_interval_s: 1.0        # optional
  link_timeout_s: 3.0              # optional  must be > heartbeat_interval_s
  hello_timeout_s: 2.0             # optional
  stale_telemetry_s: 2.0           # optional  must be > telemetry_interval_s
  command_timeout_s: 2.0           # optional

  emergency_stop_retries: 3        # optional
  emergency_stop_retry_s: 0.5      # optional
  safe_stop_on_connect: true       # optional  keep true
  safe_stop_on_shutdown: true      # optional  keep true
  safe_stop_on_stale_telemetry: true  # optional keep true
  estop_clear:
    operators: []                  # optional  e.g. ["your_name"]: only these names may clear the e-stop
    pin_env: null                  # optional  e.g. LIFEGUARD_HW_CLEAR_PIN (environment variable NAME, never the PIN)

  esp32:
    protocol: websocket
    role: server                   # REQUIRED for esp32  REPLACE if the firmware is a WebSocket server: client
    # --- role: server (laptop listens, ESP32 connects) ---
    bind_host: <LAPTOP_LAN_IP>     # REPLACE with the laptop IP from ipconfig (or 0.0.0.0 = all interfaces)
    port: 8765                     # REPLACE if the team chose another port (1024-65535)
    allowed_peers: []              # optional  recommended: ["<ESP32_IP>"] once the ESP32 IP is known
    # --- role: client (laptop connects to the ESP32) ---
    device_host: null              # REQUIRED for role client: REPLACE with "<ESP32_IP>"
    device_port: 8765              # role client: REPLACE with the firmware's port
    reconnect:                     # optional  role client back-off
      initial_backoff_s: 1.0
      max_backoff_s: 30.0
      multiplier: 2.0
      max_attempts: 0              # 0 = keep trying
    # --- both roles ---
    path: /lifeguard-hw            # REPLACE only if the firmware uses another path
    auth_token_env: null           # optional, REQUIRED if allow_physical_actuation is true on a network link:
                                   #   LIFEGUARD_HW_TOKEN  (the NAME of an environment variable)
    open_timeout_s: 3.0            # optional
    close_timeout_s: 1.0           # optional
    max_message_bytes: 16384       # optional

  components:                      # REPLACE all ids/limits with the team's real components (must match hello)
    motors:
      motor_1: {max_speed: 0.5, max_duration_s: 3.0, description: "placeholder motor channel"}
    actuators:
      indicator_1: {states: ["off", "on"], safe_state: "off", max_duration_s: 5.0, description: "placeholder"}
    sensors:
      sensor_1: {unit: null, description: "placeholder sensor"}   # sim_value is used in simulation only

  on_dispatch: []                  # optional  see the dispatch example below
  command_log: null                # optional  e.g. ../events/hardware_commands.jsonl
```

Rules the code enforces (you get CONFIGURATION ERROR with the reason otherwise): `role: client` needs
`device_host`; `allow_physical_actuation: true` on a non-loopback link needs `auth_token_env`;
`clear_emergency_stop` can never be granted to `response`/`system`; unknown keys are rejected.

Token (only if `auth_token_env` is set) — in the **same PowerShell window** that launches the app:
`$env:LIFEGUARD_HW_TOKEN = "<token from the hardware team>"`. If it is missing the Hardware page shows a FAULT.

**Dispatch → hardware example** (verified in simulation): replace the `on_dispatch: []` line with

```yaml
  permissions:
    response: [set_actuator]       # the dispatch path may ONLY send set_actuator
  on_dispatch:
    - {name: set_actuator, target: indicator_1, params: {state: "on", duration_s: 5}}
```

`target` must be a configured actuator (REPLACE `indicator_1` with the team's real output, e.g. a warning light).
In `mode: esp32` this command is **refused** ("physical actuation is disabled") until
`allow_physical_actuation: true` — that refusal is itself a safe, demonstrable result.

### C. AI — `E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml` (use as is)

```yaml
engine: yolo
yolo:
  weights: ../models/z5cc_s448_final.pt   # registry: detector_yolo11s_z5cc_v1 (sha256 8136a671...)
  conf: 0.1                       # low on purpose (ByteTrack uses low-confidence boxes)
  iou: 0.7
  imgsz: 448                      # must match training
  device: null                    # optional: "0" = first GPU, "cpu"; null = Ultralytics default (GPU if available)
  person_class_ids: [0, 1]
  distress_class_ids: [1]
pose:
  enabled: true                   # optional: false (or configs/ai_z5cc.yaml) = no skeletons
  weights: ../models/yolo11s-pose.pt      # registry: pose_yolo11s_coco_v1
  conf: 0.25
  imgsz: 320
  device: null
  crop_margin: 0.15
  min_iou: 0.3
  max_tracks: 8
  half: false
  allow_missing_weights: true     # missing file -> runs without pose; checksum mismatch -> refuses to start
behavior: none
model_registry: ../models/registry.yaml
```

Nothing to replace. Optional: `device: "0"` to force the GTX 1650. Alternatives that exist: `configs/ai_z5cc.yaml`
(same without pose), `configs/ai_ddt.yaml` (overhead-camera detector), `configs/ai.yaml` (MOCK replay, no model).
Model files present in `models/`: `z5cc_s448_final.pt`, `yolo11s-pose.pt`, `ddt_v2_final.pt`, checked against
`models/registry.yaml` at every start (mismatch → SYSTEM FAULT, the system does not start).

### D. System — create `configs\system_demo_hardware.yaml` (one-time, a copy + one line)

No existing file combines the demo risk settings (`detector_distress_threshold: 0.5`, simulated drone) with the
hardware file, so create one (verified 2026-11-28 with an identical file in a temporary folder):

```powershell
cd E:\lifeguardDron\lifeguard-system
Copy-Item configs\system_demo_drone.yaml configs\system_demo_hardware.yaml
Add-Content configs\system_demo_hardware.yaml "`nhardware_file: hardware.yaml"
```

Result: `system_demo_drone.yaml` + `hardware_file: hardware.yaml`. Do not also add an inline `hardware:` block
(the loader refuses both). If you prefer the alert-only response (no simulated drone), use the existing
`configs/system_hardware.yaml` instead (PRE_ALERT from the detector is then off).

## 4. Demonstration-day sequence (from power-off)

| # | Do | You should see |
| --- | --- | --- |
| 1 | Power the laptop, plug in the charger, connect to the demo network. Open PowerShell. `cd E:\lifeguardDron\lifeguard-system` | Prompt in that folder |
| 2 | **Hardware power, safely:** physical e-stop **pressed**, outputs disconnected or motors mechanically free, then power the prototype as the hardware team instructs | Nothing moves; ESP32 powered (LED per the team) |
| 3 | ESP32 network: `ipconfig` (laptop IP); `role: client`: `Test-NetConnection <ESP32_IP> -Port <PORT>` | `TcpTestSucceeded : True` (client role); for server role the check is step 10 |
| 4 | Camera: plug it in; test it alone (Windows Camera app, then close the Camera app!) | A live picture; the app is closed afterwards (only one program can hold the camera) |
| 5 | Config: edit `configs\hardware.yaml` (section 3B); if a token is used: `$env:LIFEGUARD_HW_TOKEN = "..."` in this window | — |
| 6 | Models: `$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"` then `E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.common.models verify --registry models/registry.yaml` | 9 lines `ok ...`, no `MISMATCH`/`missing` |
| 7 | Launch (keep this window open): see the box below | Browser at http://localhost:8501, page **AI Dashboard**, sidebar AI config `...ai_z5cc_pose.yaml`, System config `...system_demo_hardware.yaml` |
| 8 | Open **Hardware** (left navigation, or a 2nd tab at http://localhost:8501/hardware) | Black **REAL ESP32** banner "physical actuation DISABLED (stop commands only)"; then **CONNECTED** "pool-prototype-01 (<firmware>) at <ESP32 IP>: identity verified" (or **SIMULATION** if mode is still simulation) |
| 9 | AI Dashboard sidebar: Input **camera**, Camera index or URL `0`/`1`/`rtsp://...` + **Enter**, type your name in **Lifeguard name** + Enter, press **Start** | Live image with boxes and track ids, skeletons on tracked people, caption `frames=... error=0`, Swimmers table |
| 10 | Hardware page: telemetry and link | Telemetry age < 2 s, motors/sensors tables as the device reports them; "Device connected from <IP>"; Commands table: `safe_stop completed` (on-connect). AI Dashboard → Hardware panel: **Connection: CONNECTED** |
| 11 | **E-STOP test** (outputs still disabled): Hardware page → **EMERGENCY STOP** | Red **EMERGENCY STOP ACTIVE**; Commands: `emergency_stop completed` with the device's detail; AI Dashboard panel: "EMERGENCY STOP LATCHED". Then the PHYSICAL e-stop: telemetry shows e-stop engaged. Clear: release physical e-stop → expander "Clear emergency stop" → name, tick "The pool area is safe...", **Clear e-stop** → `clear_emergency_stop completed`, banner gone |
| 12 | **Alert + dispatch**: switch Input to **video** `E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4` (+Enter) → Start (camera alerts are scene-dependent, see §7) | ~20 s: orange **PRE_ALERT** (detector distress) then **ALARM** card in "Open incidents". With your name in the sidebar → **Confirm emergency** (on a PRE_ALERT it goes to ALARM; press **Confirm emergency** again) → "ALARM -> DISPATCH", response panel `[SIMULATION] mission-0001 started, hover at (x, y)`; with `on_dispatch` configured: `hardware: set_actuator/<target>: ...` and on the Hardware page a `set_actuator` row, source `response`, by your name (REJECTED "physical actuation is disabled" while that is false — expected) |
| 13 | **Shutdown**: AI sidebar **Stop** → Hardware page **Disconnect** → close the browser → `Ctrl+C` in PowerShell → physical e-stop pressed → power off the prototype → laptop | Stop: "Not running"; Disconnect: big **DISCONNECTED**, Commands `safe_stop` by `shutdown` completed; PowerShell returns to the prompt |

Launch box (step 7):

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
$env:LIFEGUARD_AI_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml"
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_demo_hardware.yaml"
E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/lifeguard_app.py
```

Sidebar text fields: **type and press Enter** — Streamlit keeps the old value otherwise (`DEMO.md`).

## 5. Pre-demonstration checklist (do it the day before AND one hour before)

| ✓ | Test | How | Expected | If it fails |
| --- | --- | --- | --- | --- |
| ☐ | Code + configs load | Launch box (step 7) | App opens, no red error | see failure table |
| ☐ | Model loading | step 6 `verify` | 9× `ok` | restore weights from `E:\lifeguardDron\Z5cc_work_backup\...\last.pt` (see `DEMO.md`); never edit the sha256 |
| ☐ | Camera | step 9 with the real camera | live image, `error=0`, no "camera ... not available" | other index (`1`), close Camera/Teams/Zoom, re-plug; RTSP: test URL in VLC |
| ☐ | Detection + tracking | camera on a person / the prototype | boxes + stable track ids | **detector not validated on this scene** — if nothing is detected, use the video fallback (§7) |
| ☐ | Pose overlay | same | skeletons on tracked people; Swimmers column `pose` | missing → `yolo11s-pose.pt` absent (warning in log) or AI config without pose |
| ☐ | ESP32 connection | Hardware page | CONNECTED, identity verified, correct firmware | failure table "DISCONNECTED/RECONNECTING" |
| ☐ | Hardware telemetry | Hardware page | age < `stale_telemetry_s`, values match reality (team checks with a meter) | "STALE": firmware not sending telemetry at the welcome interval |
| ☐ | E-STOP (software + physical) | step 11 | latched, device acks, outputs dead, clear works only after release | stop: do not demo hardware, use simulation (§7) |
| ☐ | Link-loss failsafe | unplug ESP32 Wi-Fi / power while an output is ON (only once actuation is approved) | device stops by itself within `link_timeout_s`; page shows RECONNECTING then CONNECTED | firmware issue — do not demo motors |
| ☐ | Alert + dispatch | step 12 with `demo_06` | PRE_ALERT, ALARM, CONFIRM → DISPATCH, simulated mission, hardware row | wrong system config (check sidebar), `on_dispatch` not set |
| ☐ | Safe shutdown | step 13 | DISCONNECTED, `safe_stop` by `shutdown` completed | if the ESP32 was already gone: `safe_stop` not sent (expected), use the physical e-stop |
| ☐ | Fallback works | §7 commands | simulation + videos run | — |

**Common failures**

| Symptom | Cause → fix |
| --- | --- |
| `File does not exist: src\lifeguard\app\lifeguard_app.py` | wrong folder → `cd E:\lifeguardDron\lifeguard-system` |
| `No module named 'ultralytics'` / torch | used `.venv` → use `E:\lifeguardDron\.venv-train\Scripts\python.exe` |
| `No module named 'lifeguard.app.hardware_hub'` / `lifeguard.hardware` | `PYTHONPATH` not set in this window (`.venv-train` points to the old copy) |
| SYSTEM FAULT ... checksum / model missing | model file changed or missing → restore the original file |
| Frame-size fault / mock engine runs | AI config field still `ai.yaml` → type `...ai_z5cc_pose.yaml` + Enter |
| `HardwareShareConflict` on Start | System config's hardware ≠ shared service → use `system_demo_hardware.yaml`, or press Disconnect on the Hardware page first |
| Hardware **FAULT** "cannot listen on ..." | port busy (another app / a standalone `hardware_demo.py` still running) or `bind_host` is not this laptop's IP |
| Hardware **FAULT** about `auth_token_env` | token variable not set in the launch window |
| **CONFIGURATION ERROR** | read the reason (bad key/value, `<...>` placeholder left) → fix `hardware.yaml` → Hardware page **Reload configuration** (AI stopped) |
| Stays **DISCONNECTED** (server role) | ESP32 not reaching the laptop: wrong IP/port/path in firmware, firewall, different network, Wi-Fi set to Public |
| **RECONNECTING** with "connect ... failed" (client role) | wrong `device_host`/`device_port`, ESP32 off/not on network |
| Connects then drops; log "unexpected device id" / "authentication failed" | `device_id` or token differs from the firmware |
| Motor/actuator commands "did not announce" | the firmware's hello lacks that component id → align ids with the team |
| Clear e-stop unavailable | not latched / not connected / telemetry stale / physical e-stop still pressed (device rejects) |
| Video plays slower than real time with pose | expected on GTX 1650 (≈23–25 FPS); `configs/ai_z5cc.yaml` without pose is faster |

## 6. Safety rules for the demonstration

1. **Start with outputs disabled**: `allow_physical_actuation: false` (default). The software then cannot command
   any movement on a real ESP32; only stop commands and status requests go out.
2. **Physical e-stop first**: before any power-up and before any motor test, the team demonstrates that the
   physical e-stop cuts actuator power **directly** (not through the ESP32). Software stops do not guarantee safety.
3. **Test the software E-STOP (step 11) before any movement.**
4. **First movement only with the hardware team's approval**: set `allow_physical_actuation: true` (plus
   `auth_token_env` on a network link), restart, then Hardware page → "Test an output" with the minimum speed and
   shortest duration, one output at a time, hands clear, someone on the physical e-stop.
5. Every run is time-limited by `max_duration_s`; nothing runs continuously; connection, shutdown and stale
   telemetry all trigger a safe stop.
6. **Simulation vs real** is always labelled: blue **SIMULATION** banner and "SIMULATED telemetry" vs black
   **REAL ESP32** banner. If a real link fails, the app shows DISCONNECTED/FAULT — it never switches to
   simulation by itself.
7. Never claim physical results you did not observe on the day; the AI alerts are decision support, confirmed by a
   human; the drone is simulation only.

## 7. Fallback demonstration (prototype or camera unavailable)

Everything below is verified. Keep `configs\hardware.yaml` with `mode: simulation` (or switch back to it and press
**Reload configuration** with the AI stopped).

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
$env:LIFEGUARD_AI_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\ai_z5cc_pose.yaml"
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_demo_hardware.yaml"
E:\lifeguardDron\.venv-train\Scripts\python.exe -m streamlit run src/lifeguard/app/lifeguard_app.py
```

- Input **video**: `E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4` (PRE_ALERT + ALARM + dispatch),
  `demo_11_drown_iVMA7GhSJjM.mp4` (early PRE_ALERT), `demo_05_swim_L0KTqPloUiU.mp4` (no alarm),
  `demo_01_swim_I-EKqLdbysk.mp4` (pose skeletons). Scenario notes: `docs/DEMO.md`.
- Hardware page (simulation): connection, E-STOP, clear, and the "Simulate failures" buttons (device offline →
  RECONNECTING, stale telemetry, motor fault, physical e-stop pressed, firmware ignores commands).
- Rehearsing the real link without the prototype: a copy of `hardware.yaml` with `mode: esp32` + in a 2nd window
  `E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.hardware.fake_device --hardware-config <that copy>`
  (the fake ESP32 is labelled "fake-esp32-0.1" as firmware; it is still software, not hardware).

## 8. Information still needed from the hardware team

1. Firmware role: does the ESP32 **connect to the laptop** (`server`) or **run a WebSocket server** (`client`)?
2. ESP32 IP address (DHCP reservation or static), port, path, Wi-Fi network used on the day.
3. `DEVICE_ID` string and firmware version string.
4. Whether a token is used (and the token itself, handed over privately).
5. Exact component list: motor ids, actuator ids + states + safe state, sensor ids + units + valid ranges —
   identical to what the firmware sends in `hello.components`.
6. Limits: max speed (normalised 0–1) and max run time per motor/actuator.
7. Confirmation that the firmware implements `HARDWARE_INTEGRATION.md` §10 (validation, one ack per command,
   `accepted`→`completed` for timed runs, deadman timers, link-loss failsafe at `link_timeout_ms`, e-stop latch,
   refuse clear while the physical e-stop is pressed, report `estop_engaged`, boot with outputs off).
8. Electrical: supply voltages, driver enable lines defaulting to off, fuses/current limits, physical e-stop wiring
   that cuts actuator power directly, water/electricity separation.
9. Which output should react to a confirmed dispatch (for `on_dispatch`), if any.
10. A bench test record (section 9, last block) signed off before motors are enabled.

## 9. Tests that must pass before presenting

```powershell
cd E:\lifeguardDron\lifeguard-system
# Software tests (use .venv: it has pytest)              expected: 653 passed, 2 skipped (2026-11-28)
E:\lifeguardDron\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
E:\lifeguardDron\.venv\Scripts\python.exe -m ruff check src tests                        # All checks passed!
# Demo environment
$env:PYTHONPATH = "E:\lifeguardDron\lifeguard-system\src"
E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.common.models verify --registry models/registry.yaml   # 9x ok
E:\lifeguardDron\.venv-train\Scripts\python.exe -m lifeguard.app.main --ai-config configs/ai_z5cc_pose.yaml --system-config configs/system_demo_hardware.yaml --video E:\lifeguardDron\demo\demo_06_drown_T5mDQeDkca0.mp4
#   expected last line like: frames=574 ok=574 degraded=0 error=0 ... alarms=1   (+ incidents=1)
```

Physical bench tests (**must be done with the team; none done yet**): ESP32 connects and shows CONNECTED with the
right firmware; telemetry values match a measurement; `ping` completes; software E-STOP latches and the device
acks; physical e-stop cuts power and is reported; clear is refused while pressed; link loss (Wi-Fi off) stops
outputs within `link_timeout_s`; only then, with approval, one minimum-speed, shortest-duration run per output.
