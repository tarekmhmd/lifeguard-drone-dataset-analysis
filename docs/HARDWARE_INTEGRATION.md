# Hardware integration — ESP32 pool prototype

Status (2026-11-28): **software implemented and tested in simulation and against a fake ESP32 over real loopback
WebSockets (both link roles). No physical ESP32, firmware, motor, sensor or wiring has been tested.** The host side
is configuration-driven: a device that implements sections 4 and 10 is connected by editing
[`configs/hardware.yaml`](../configs/hardware.yaml) only (section 15) — physical safety still depends on the
firmware, the electrical design and the prototype being built and validated (section 14).

Code: [`src/lifeguard/hardware/`](../src/lifeguard/hardware/) · tests: [`tests/hardware/`](../tests/hardware/) ·
Hardware Demo window: [`src/lifeguard/app/hardware_demo.py`](../src/lifeguard/app/hardware_demo.py) ·
configs: [`configs/hardware.yaml`](../configs/hardware.yaml) (the one file to edit),
[`configs/system_hardware.yaml`](../configs/system_hardware.yaml) (AI dashboard + hardware.yaml); inline examples
[`configs/system_hardware_sim.yaml`](../configs/system_hardware_sim.yaml),
[`configs/system_hardware_esp32.example.yaml`](../configs/system_hardware_esp32.example.yaml).

## 1. Hardware architecture

```
Camera -> Detection -> Tracking -> Features -> Risk engine -> Incident / Alert -> lifeguard CONFIRM
                                                                                     |
                                       ResponsePlatform (manual / simulated drone)   |   (existing, unchanged)
                                                     |                               v
                                       HardwareLinkedResponse  <-- only if hardware.on_dispatch is configured
                                                     |
 Dashboard operator (E-STOP, safe stop, ...) ---> HardwareService ---------------------------------------
                                                     |  CommandManager: permission, ranges, actuation gate,
                                                     |  link, stale telemetry, e-stop latch, faults, conflicts
                                                     |  TelemetryManager: latest state, staleness, faults
                                                     v
                                       HardwareInterface (one of)
                              SimulatedHardware            Esp32Adapter
                     (in-memory DeviceModel)         (WebSocket server, ESP32 connects)
                                                     |
                                    ESP32 firmware -> motor drivers / actuators / sensors (prototype)
```

Design decisions:

- **Reuse, not duplicate.** The existing `ResponsePlatform` ([`system/response.py`](../src/lifeguard/system/response.py))
  is the "authorised response manager": it only ever receives a dispatch that a lifeguard confirmed, and its
  docstring already names poolside actuators as its extension point. `HardwareLinkedResponse` wraps the configured
  platform and adds hardware commands; the incident system and the risk engine are untouched.
- **Isolation.** Perception (`lifeguard.ai`), the risk engine (`lifeguard.system.risk`) and the contracts never import
  `lifeguard.hardware` (checked by a test). Telemetry is displayed and logged; it is **not** an input to risk.
- **One protocol path.** The simulator runs the same JSON messages through the same `DeviceSession` code as the
  real link; only the transport differs.
- **Background service.** The hardware loop runs on its own thread (20 Hz), so a slow or dead link never slows the
  video pipeline; every failure is contained and reported as a hardware FAULT.

Modules: `config.py` (configuration), `models.py` (telemetry/command types), `protocol.py` (message contract),
`interface.py` (common interface), `session.py` (protocol state machine), `transport.py` (WebSocket server),
`esp32.py` (ESP32 adapter), `device_model.py` (reference device behaviour), `simulator.py`, `telemetry.py`,
`commands.py` (command manager), `service.py`, `response_bridge.py`, `fake_device.py` (fake ESP32 client).

## 2. ESP32's role

The ESP32 is the prototype's **controller and communication interface**: it connects to the computer, reports
sensor/motor/actuator state, and executes **validated** commands on its outputs, enforcing its own limits and
timers. It does not run AI, does not decide about swimmers and is not the pool itself; the physical prototype
(miniature pool, motors, pumps, objects, sensors) is a separate design that the ESP32 drives.

## 3. Communication protocol — WebSocket over Wi-Fi (ESP32 = client)

| Option | Verdict |
| --- | --- |
| **WebSocket (chosen)** | One persistent full-duplex TCP connection for telemetry and commands, ordered and reliable delivery, no broker to install or operate, standard ESP32 client libraries exist [Assumption: choose and verify one when writing the firmware], and the Python side only needs the optional `websockets` package (already present in both project environments). The ESP32 is the client, so its DHCP address does not matter; it needs the computer's address. |
| MQTT over Wi-Fi | The best fit if several devices or consumers share data, but it needs a broker (e.g. Mosquitto) running and maintained, plus `paho-mqtt` (not installed). Not needed for one device and one host. Can be added later behind the same `Transport` interface. |
| HTTP | Request/response only; telemetry push would need polling or a second channel. |
| Serial (USB) | Useful for bench debugging; needs `pyserial` (not installed). Not implemented; a `Transport` can be added for it without touching the rest. |

Two link roles, same messages (`esp32.role`):

- `server` (default): the computer listens on `ws://<bind_host>:<port><path>` (default
  `ws://127.0.0.1:8765/lifeguard-hw`) and the ESP32 connects to it; the ESP32 needs the computer's address.
  One device at a time (a second client is refused); `esp32.allowed_peers` can restrict the slot to the ESP32's IP.
- `client`: the computer connects to the ESP32's WebSocket server at `ws://<device_host>:<device_port><path>` and
  reconnects with exponential back-off (`esp32.reconnect`: initial / max interval, multiplier, optional attempt
  limit). The firmware then runs a WebSocket server and still sends `hello` first on every connection.

There is **no network discovery** (no mDNS/broadcast): the address comes from the configuration.

Maximum message size `max_message_bytes` (16 KiB); bounded opening/closing handshakes (`open_timeout_s`,
`close_timeout_s`). Security: no TLS (local network only, isolated Wi-Fi recommended); optional shared token whose
**environment variable name** is configured (`esp32.auth_token_env`), never the token itself (required for
physical actuation over a non-loopback link). The token authenticates the device to the host only; in `client` role
the firmware cannot authenticate the host (restrict it to the host's IP).

## 4. Message format (protocol version 1)

JSON text messages, one per WebSocket frame. Every message has `type`, `proto` (= 1), `device_id`, and a per-sender
`seq` (non-negative, increasing within one connection; restart at 0 after reconnect). Unknown extra fields are
ignored. NaN/Infinity are rejected. `null` means "not available" and is never replaced by a guess.

Device → host:

```json
{"type": "hello", "proto": 1, "device_id": "pool-prototype-01", "seq": 1, "firmware": "0.1.0",
 "token": "<only if configured>",
 "components": {"motors": ["motor_1"], "actuators": ["indicator_1"], "sensors": ["sensor_1"]}}

{"type": "telemetry", "proto": 1, "device_id": "pool-prototype-01", "seq": 2, "t_device_ms": 51234,
 "sensors": {"sensor_1": {"value": 12.5, "unit": "cm", "fault": false}, "sensor_2": {"value": null}},
 "motors": {"motor_1": {"state": "running", "speed": 0.3}},
 "actuators": {"indicator_1": {"state": "off"}},
 "faults": [], "estop_engaged": false, "power": {"voltage_v": null, "battery_pct": null}}

{"type": "heartbeat", "proto": 1, "device_id": "pool-prototype-01", "seq": 3, "t_device_ms": 52234}

{"type": "ack", "proto": 1, "device_id": "pool-prototype-01", "seq": 4, "cmd_id": "c-000007",
 "status": "accepted", "detail": "running"}
```

`motors.*.state`: `stopped | running | fault | unknown` (anything else is shown as `unknown`); `speed` in [-1, 1].
`faults`: codes of `[A-Za-z0-9_.:-]`, e.g. `sensor_fault:sensor_1`. Unknown `components` kinds in `hello` are
ignored. `ack.status`: `accepted` (started a timed action), `completed`, `rejected`, `failed`.

Command ids look like `c-<4 hex>-<6 digits>` (the hex part changes on every host start, so ids never repeat).

Acknowledgement state machine (host side): a command written to the link is **not** executed until the device's
`ack` for its `cmd_id` says so.

```
PENDING  --accepted-->  ACCEPTED  --completed-->  COMPLETED
PENDING | ACCEPTED  --completed | rejected | failed-->  final
PENDING  --no ack within command_timeout_s-->  TIMEOUT   (emergency_stop: re-sent every emergency_stop_retry_s first)
ACCEPTED --no completion by the end of duration_ms + command_timeout_s (untimed: command_timeout_s)-->  TIMEOUT
any active --link lost--> FAILED;  --safe_stop / emergency_stop--> SUPERSEDED
```

Duplicate `accepted`, acks for finished or unknown ids, and acks from an older connection are ignored.

Host → device:

```json
{"type": "welcome", "proto": 1, "device_id": "pool-prototype-01", "seq": 1,
 "heartbeat_interval_ms": 1000, "telemetry_interval_ms": 500, "link_timeout_ms": 3000}

{"type": "command", "proto": 1, "device_id": "pool-prototype-01", "seq": 2, "cmd_id": "c-000007",
 "name": "set_motor", "target": "motor_1", "params": {"speed": 0.3, "duration_ms": 2000}}

{"type": "heartbeat", "proto": 1, "device_id": "pool-prototype-01", "seq": 3, "t_host_ms": 104250}

{"type": "error", "proto": 1, "device_id": "pool-prototype-01", "seq": 4, "detail": "authentication failed"}
```

Commands:

| `name` | `target` | `params` | Moves something? |
| --- | --- | --- | --- |
| `ping` | – | – | no |
| `request_status` | – | – (device answers with an ack and an immediate telemetry) | no |
| `safe_stop` | – | – (all motors stop, actuators to their safe state) | no (stops) |
| `emergency_stop` | – | – (as safe stop, and latch) | no (stops, latches) |
| `clear_emergency_stop` | – | – (releases the software latch only) | no |
| `set_motor` | motor id | `speed` in [-max_speed, max_speed], `duration_ms` ≤ max_duration | **yes** |
| `set_actuator` | actuator id | `state`; `duration_ms` ≤ max_duration unless `state` is the safe state | **yes** |

Sequence: connection opens → device sends `hello` (within `hello_timeout_s`) → host checks id/token → `welcome` →
host sends `safe_stop` (on-connect; `emergency_stop` instead while the host e-stop latch is set) → telemetry and
heartbeats flow both ways → commands get `ack`s. Wrong id or token, or a second `hello` on the same connection:
`error` and the host closes. Commands to components the device did not announce in `hello` are refused.

## 5. Hardware simulator

`mode: simulation` runs `SimulatedHardware`: the reference device model ([`device_model.py`](../src/lifeguard/hardware/device_model.py))
behind an in-memory link. It simulates configured motors, actuators and sensors, timed runs, the e-stop latch, the
link-loss failsafe and device faults. Sensors with `sim_value` report that value; sensors without one report
`null` (unavailable). Hooks (tests and demos): `inject_fault`, `clear_faults`, `set_sensor`, `go_offline` /
`come_online`, `telemetry_frozen` (stale data), `send_corrupt_message`. It is **not** the AI video simulation
(mock perception) and **not** ArduPilot SITL (simulated drone).

## 6. Configuration

Preferably the dedicated file [`configs/hardware.yaml`](../configs/hardware.yaml) (one top-level key `hardware:`),
referenced from a system YAML by `hardware_file: hardware.yaml` (relative to that YAML) and read directly by the
Hardware Demo window. An inline `hardware:` block in the system YAML still works (not both). Validation is strict
([`hardware/config.py`](../src/lifeguard/hardware/config.py)): unknown keys, bad addresses, `clear_emergency_stop`
for `response`/`system`, client role without `device_host`, or physical actuation over a network without a token
are rejected with the reason. A missing or invalid hardware file never stops the AI: the hardware panel shows
CONFIGURATION ERROR (FAULT) and nothing connects — simulation is never substituted.

Keys added for plug & play (the table below lists the original ones): `auto_connect` (true), `hello_timeout_s` (2),
`emergency_stop_retry_s` (0.5), `safe_stop_on_shutdown` (true), `safe_stop_on_stale_telemetry` (true),
`estop_clear.operators` ([] = any named operator), `estop_clear.pin_env` (null), `esp32.role` (server),
`esp32.allowed_peers` ([]), `esp32.device_host` / `device_port`, `esp32.open_timeout_s` (3), `esp32.close_timeout_s`
(1), `esp32.reconnect.{initial_backoff_s: 1, max_backoff_s: 30, multiplier: 2, max_attempts: 0 = keep trying}`.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Start the hardware service with the runtime |
| `mode` | `simulation` | `simulation` or `esp32` |
| `device_id` | `pool-prototype-01` | Must equal the firmware's id |
| `allow_physical_actuation` | `false` | In `esp32` mode, `set_motor` / `set_actuator` are refused unless `true` |
| `telemetry_interval_s` / `heartbeat_interval_s` | 0.5 / 1.0 | Sent to the device in `welcome` |
| `link_timeout_s` | 3.0 | Nothing heard → link lost (host closes; device must safe-stop) |
| `stale_telemetry_s` | 2.0 | Older telemetry → movement refused |
| `command_timeout_s` | 2.0 | No acknowledgement → TIMEOUT |
| `emergency_stop_retries` | 3 | Re-sends of an unacknowledged e-stop |
| `safe_stop_on_connect` | `true` | Send `safe_stop` whenever the device (re)connects |
| `components.motors.<id>` | – | `max_speed` (0–1], `max_duration_s` |
| `components.actuators.<id>` | – | `states`, `safe_state`, `max_duration_s` |
| `components.sensors.<id>` | – | `unit`, `valid_min`, `valid_max` (flag implausible values), `sim_value` (simulation only) |
| `permissions` | operator: all; response: none; system: ping/status/safe_stop/emergency_stop | Who may send what |
| `on_dispatch` | `[]` | Commands sent after a lifeguard-confirmed dispatch (must be permitted for `response`) |
| `esp32.bind_host` / `port` / `path` | `127.0.0.1` / 8765 / `/lifeguard-hw` | WebSocket server |
| `esp32.auth_token_env` | `null` | Environment variable holding the shared token |
| `command_log` | `null` | JSON-lines log of every command and status change |
| `log_messages` | `false` | DEBUG-log every raw message |

All component names in the examples (`motor_1`, `indicator_1`, `sensor_1`) are **placeholders**.

## 7. Run in simulation mode

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_hardware_sim.yaml"
python -m streamlit run src/lifeguard/app/dashboard.py
```

Start any input (synthetic, video or camera). The right column shows **Hardware**: "SIMULATION mode", the link,
sensors/motors/actuators, faults, recent commands, and the EMERGENCY STOP / Safe stop / Clear e-stop buttons.
The CLI (`python -m lifeguard.app.main ... --system-config configs/system_hardware_sim.yaml`) starts the same service.

## 8. Enable physical ESP32 mode

1. Copy `configs/system_hardware_esp32.example.yaml`; replace every placeholder (device id, components, limits).
2. Set `esp32.bind_host` to this computer's LAN address (or `0.0.0.0`), pick `port`, allow it in the firewall for
   the local network only.
3. Optional token: set an environment variable (e.g. `LIFEGUARD_HW_TOKEN`) and put its **name** in `auth_token_env`.
4. `pip install -e ".[hardware]"` (installs `websockets`; already present in the existing environments).
5. Start with `allow_physical_actuation: false`: verify connection, telemetry, stop commands and the physical e-stop.
6. Only after section 11's hardware protections exist and are tested: set `allow_physical_actuation: true`.

Rehearse without hardware: run the dashboard with the esp32 config, then
`python -m lifeguard.hardware.fake_device --system-config <that config>` in another terminal.

## 9. Connect a real ESP32 later

Flash firmware that implements sections 4 and 10 with the same `device_id`, the computer's address/port/path and
(if used) the token; power it; it joins Wi-Fi and connects. The dashboard should show "Connected", the firmware
version and live telemetry. Then test, in order: telemetry values against reality, `ping`, `safe_stop`,
`emergency_stop` (software) and the **physical** e-stop, link-loss failsafe (unplug Wi-Fi / stop the host),
then — with actuation enabled — one short low-speed run per output.

## 10. Firmware responsibilities (integration specification)

The firmware must:

1. Join Wi-Fi (credentials in firmware/provisioning, never in this repository); connect as a WebSocket client to
   `ws://host:port/path`; reconnect with back-off forever.
2. Send `hello` on every connection (id, protocol 1, firmware version, component ids, token if configured).
3. Apply `welcome` intervals; send `telemetry` every `telemetry_interval_ms` and `heartbeat` every
   `heartbeat_interval_ms`; restart `seq` at 0 per connection.
4. Report only measured values; `null` for anything not measured; set `fault` flags and `faults` codes honestly.
5. **Validate every command itself** (known target, ranges, duration); answer every command with one `ack`
   (`rejected` with a reason when refused).
6. **Deadman timers**: every motor run and non-safe actuator state ends after its `duration_ms` (then `ack`
   `completed`); never run continuously.
7. **Link-loss failsafe**: no host message for `link_timeout_ms` → safe stop.
8. `safe_stop` → all outputs safe; `emergency_stop` → outputs safe **and** latch (refuse movement) until
   `clear_emergency_stop`; report `estop_engaged`. A physical e-stop must be reported as `estop_engaged: true`, and
   `clear_emergency_stop` must be answered `rejected` while the physical e-stop is still engaged.
9. Boot with all outputs off/safe; never move on boot or reconnect.
10. Refuse movement while any fault is active.
11. Acks: timed runs → `accepted` then `completed` when the run ends; everything else → exactly one final ack
    (`completed` / `rejected` / `failed`). Send `hello` once per connection (a second one closes the link).
12. Announce every component id in `hello.components` (`motors`, `actuators`, `sensors`); unannounced ones are
    refused by the host.
13. `role: client` only: run a WebSocket server on the configured port/path, accept one host at a time, and accept
    connections only from the host's IP.

[`device_model.py`](../src/lifeguard/hardware/device_model.py) implements these rules in Python and is the
executable reference; [`fake_device.py`](../src/lifeguard/hardware/fake_device.py) shows the message loop.

## 11. Safety requirements

Implemented in software (tested in simulation and against the fake ESP32): default simulation mode; physical
actuation off unless explicitly enabled; no movement on start (safe stop on every connect); bounded durations only;
range checks from the component config; command permissions per source (the AI has none; the response bridge none
unless configured; it only acts after a lifeguard CONFIRM; `clear_emergency_stop` can never be granted to it);
rejection when disconnected, telemetry stale, e-stop latched, faults active or the component was not announced;
conflict and duplicate rejection; command timeouts (no command stays ACCEPTED forever); e-stop never blocked,
latched immediately on the host, re-sent every `emergency_stop_retry_s` (also after a failed send) up to
`emergency_stop_retries`, and re-asserted on every reconnect while the host latch is set; e-stop clear only by an
authorised operator (optional name list and PIN from an environment variable), with an explicit safety confirmation,
fresh telemetry, and only after the device acknowledges it; stale telemetry while an output runs → safe stop;
one `hello` per connection; optional peer allow-list; safe stop on shutdown/disconnect; every command logged; a dead,
silent or unreachable device never blocks video processing or status reads (H1).

**A software stop does not guarantee physical safety** (Wi-Fi loss, firmware crash, a stuck driver). The prototype
must also have, in hardware: a **physical emergency-stop switch that cuts actuator power directly** (not through the
ESP32); motor-driver **enable lines** that default to disabled on reset/brown-out (pull-downs); current limiting
and fuses; electrical isolation and enclosures appropriate for water near electronics (low-voltage supply, RCD/GFCI
on any mains equipment); mechanical guards on moving parts; and a supervised, documented test procedure.

## 12. Troubleshooting

| Symptom | Check |
| --- | --- |
| Panel shows FAULT "websockets is not installed" | `pip install -e ".[hardware]"` |
| FAULT "cannot listen on …" | Port in use or bind address not on this computer; change `esp32.port` / `bind_host` |
| FAULT about `auth_token_env` | The environment variable is not set in the process that runs the dashboard |
| Stays "Disconnected" | ESP32 Wi-Fi, host address/port/path in the firmware, firewall, same network |
| Connects then drops | Wrong `device_id` or token (see last error), no heartbeats, link timeout too short |
| Movement refused "physical actuation is disabled" | Intended in esp32 mode until `allow_physical_actuation: true` |
| Refused "telemetry is stale" | Telemetry not arriving within `stale_telemetry_s` |
| Refused "emergency stop is latched" | Release the physical e-stop, then Clear e-stop (operator name required) |
| Invalid-message counter rising | Firmware JSON format (section 4); enable `log_messages` for raw messages |

## 13. Current limitations

- No physical ESP32, firmware, electrical design or prototype has been built or tested.
- The firmware does not exist yet; the Python reference model is not firmware.
- One device per host; WebSocket only (no MQTT/serial transports implemented); no TLS.
- Component types, limits and units are placeholders.
- Hardware telemetry is not used anywhere in risk assessment (by design); any such use is a separate future,
  validated decision.
- Motor/actuator test runs are in the Hardware Demo window ("Test an output"), bounded by the configured limits and
  refused in `esp32` mode until `allow_physical_actuation: true`.
- No network discovery; no TLS; the host does not authenticate itself to the device (client role); operator
  identity is a typed name (optionally an allow-list + PIN), not a login.
- In `esp32` / `role: server` only one program can listen on the port: run either the Hardware Demo window or the AI
  dashboard with hardware, not both (the second shows a FAULT "cannot listen").

## 14. Steps remaining before physical deployment

1. Finalise the prototype design: ESP32 board model, motors and drivers, actuators, sensors (models, units, ranges),
   power supply, GPIO assignment, physical e-stop and interlocks.
2. Fill in the component configuration (ids, limits, units, valid ranges, safe states).
3. Write and flash firmware implementing sections 4 and 10; test it against `fake_device`-equivalent behaviour.
4. Network setup: dedicated Wi-Fi/SSID, fixed host address or DNS, firewall rule, optional token.
5. Bench tests with outputs disconnected: link, telemetry, stop commands, link-loss failsafe, physical e-stop.
6. Electrical and safety review of the prototype (water + electricity), supervised.
7. Enable `allow_physical_actuation` and test each output at minimum speed/duration.
8. Record results in this document; until then the hardware status remains "software-ready, physically untested".

## 15. Plug & play: modes, windows and connecting the team's ESP32

Everything is chosen in [`configs/hardware.yaml`](../configs/hardware.yaml); no Python change is needed.

| Mode | How | What you see |
| --- | --- | --- |
| AI only (no ESP32) | AI dashboard with `configs/system.yaml` (default), or `enabled: false` in hardware.yaml | No hardware panel; AI unchanged |
| Simulated ESP32 | `enabled: true`, `mode: simulation` (shipped default) | Blue **SIMULATION** label, simulated telemetry, failure buttons |
| Real ESP32 | `enabled: true`, `mode: esp32` + the `esp32:` section | Black **REAL ESP32** label, real telemetry and acks only |

Launch (Windows PowerShell, from `E:\lifeguardDron\lifeguard-system`):

```powershell
# Hardware Demo window (reads configs/hardware.yaml; another file: $env:LIFEGUARD_HARDWARE_CONFIG = "...")
E:\lifeguardDron\.venv\Scripts\python.exe -m streamlit run src/lifeguard/app/hardware_demo.py

# AI dashboard with the same hardware file
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_hardware.yaml"
E:\lifeguardDron\.venv\Scripts\python.exe -m streamlit run src/lifeguard/app/dashboard.py
```

The Hardware Demo window shows a large connection state (CONNECTED / CONNECTING / RECONNECTING / DISCONNECTED /
FAULT / CONFIGURATION ERROR) and an EMERGENCY STOP ACTIVE banner, device id, firmware, address and peer, last
communication, telemetry age and staleness, motors/actuators/sensors as reported, commands with their
acknowledgements, a communication log (tokens never shown), errors and reconnection progress. Controls: EMERGENCY
STOP, Connect, Disconnect, Refresh, Safe stop, guarded Clear e-stop, bounded output tests and (simulation only)
failure injection. The AI dashboard shows the same panel in compact form, from the same `HardwareService`.

When the team's ESP32 arrives:

1. Firmware implements sections 4 and 10 with the same `device_id`.
2. In hardware.yaml: `mode: esp32`; choose the role —
   `server`: set `esp32.bind_host` to this computer's LAN IP (or `0.0.0.0`), `port`, optionally `allowed_peers:
   ["<ESP32 IP>"]`, and flash the computer's address into the firmware; or `client`: set `esp32.device_host` /
   `device_port` to the ESP32's address.
3. Optional (required for physical actuation over the network): set an environment variable with the token and put
   its NAME in `esp32.auth_token_env`; flash the same token.
4. Open the Hardware Demo window (or press Reload configuration). It connects automatically (`auto_connect`), checks
   the hello (id, token, protocol, components), shows CONNECTED and the real telemetry. Pulling the plug shows
   RECONNECTING at once; the link recovers by itself.
5. Only after section 11's hardware protections exist and pass the bench tests: `allow_physical_actuation: true`.

Rehearse today without hardware: set `mode: esp32` in a copy of hardware.yaml, open the window with it, and run
`python -m lifeguard.hardware.fake_device --hardware-config <that file>` in a second terminal (it plays the ESP32 in
either role).

### Two-page application (AI Dashboard + Hardware, one shared hardware service)

```powershell
cd E:\lifeguardDron\lifeguard-system
$env:LIFEGUARD_SYSTEM_CONFIG = "E:\lifeguardDron\lifeguard-system\configs\system_hardware.yaml"
E:\lifeguardDron\.venv\Scripts\python.exe -m streamlit run src/lifeguard/app/lifeguard_app.py
```

Pages: AI Dashboard at `/`, Hardware at `/hardware` (open them in separate tabs or windows side by side). Both use
ONE `HardwareService` held by the process-wide hub ([`app/hardware_hub.py`](../src/lifeguard/app/hardware_hub.py)):
one connection, one e-stop latch, one telemetry stream, one command history, for every page, tab and browser
connected to this server. The Hardware page creates the service from `hardware.yaml` if the AI is not running; the
AI runtime borrows the same instance (also for `on_dispatch`) and stopping the AI leaves the hardware running. An AI
system config with *different* hardware settings is refused while the shared service runs (use
`system_hardware.yaml`, which points at the same `hardware.yaml`, or press Disconnect first); Reload configuration is
refused while the AI uses the service. Two separate `streamlit run` processes never share a service.
