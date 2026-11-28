"""Fake ESP32: the reference device model on a real WebSocket (development and testing, no hardware).

It plays the firmware's side of the protocol for either link role (``esp32.role``):

- ``role: server`` (host listens): the fake device CONNECTS to ``ws://<bind_host>:<port><path>``;
- ``role: client`` (host connects): the fake device LISTENS on ``device_host:device_port`` like ESP32 firmware
  with a WebSocket server would.

Run the hardware demo (or the AI dashboard) with ``mode: esp32``, then in another terminal::

    python -m lifeguard.hardware.fake_device --hardware-config configs/hardware.yaml

It sends ``hello`` (with the token from ``esp32.auth_token_env`` if configured), periodic ``telemetry`` and
``heartbeat``, answers commands exactly like ``device_model.DeviceModel`` and reconnects if the link drops. It
moves nothing. Its loop is the template for the firmware (docs/HARDWARE_INTEGRATION.md §10).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from typing import Any

from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.device_model import DeviceModel

logger = logging.getLogger(__name__)


def _talk(ws: Any, model: DeviceModel, config: HardwareConfig, token: str | None, stop: threading.Event) -> None:
    """One connection: hello, then telemetry / heartbeats / command replies until the link ends."""
    model.seq = 0
    now = time.monotonic()
    model.host_seen(now)
    ws.send(json.dumps(model.hello(token)))
    next_tel = now
    next_hb = now + config.heartbeat_interval_s
    while not stop.is_set():
        try:
            raw = ws.recv(timeout=0.05)
        except TimeoutError:
            raw = None
        now = time.monotonic()
        if raw is not None:
            msg = json.loads(raw)
            if msg.get("type") == "error":
                logger.error("host refused: %s", msg.get("detail"))
            for reply in model.handle(msg, now):
                ws.send(json.dumps(reply))
        for reply in model.tick(now):
            ws.send(json.dumps(reply))
        if now >= next_tel:
            ws.send(json.dumps(model.telemetry(now)))
            next_tel = now + config.telemetry_interval_s
        if now >= next_hb:
            ws.send(json.dumps(model.heartbeat(now)))
            next_hb = now + config.heartbeat_interval_s


def _model(config: HardwareConfig) -> DeviceModel:
    return DeviceModel(config.device_id, config.components, link_timeout_s=config.link_timeout_s,
                       firmware="fake-esp32-0.1")


def run(config: HardwareConfig, url: str, stop: threading.Event, token: str | None = None,
        reconnect_s: float = 1.0, model: DeviceModel | None = None) -> None:
    """Device as WebSocket CLIENT (host ``role: server``): connect, talk, reconnect, until ``stop`` is set."""
    from websockets.sync.client import connect

    model = model or _model(config)
    while not stop.is_set():
        try:
            with connect(url, open_timeout=3, proxy=None) as ws:
                _talk(ws, model, config, token, stop)
        except Exception as err:  # noqa: BLE001 - a device just retries
            logger.info("fake device: %s; reconnecting in %.1f s", err, reconnect_s)
        stop.wait(reconnect_s)


def serve(config: HardwareConfig, host: str, port: int, stop: threading.Event, token: str | None = None,
          model: DeviceModel | None = None, ready: threading.Event | None = None) -> None:
    """Device as WebSocket SERVER (host ``role: client``): one host connection at a time, until ``stop``."""
    from websockets.sync.server import serve as ws_serve

    model = model or _model(config)
    busy = threading.Lock()

    def handler(ws: Any) -> None:
        path = getattr(getattr(ws, "request", None), "path", "") or ""
        if path.split("?")[0] != config.esp32.path:
            ws.close(code=1008, reason="wrong path")
            return
        if not busy.acquire(blocking=False):
            ws.close(code=1013, reason="a host is already connected")
            return
        try:
            _talk(ws, model, config, token, stop)
        except Exception as err:  # noqa: BLE001 - the host went away; wait for the next one
            logger.info("fake device: host connection ended: %s", err)
        finally:
            busy.release()

    with ws_serve(handler, host, port, close_timeout=1.0) as server:
        threading.Thread(target=lambda: (stop.wait(), server.shutdown()), daemon=True).start()
        logger.info("fake ESP32 %s listening on ws://%s:%d%s", config.device_id, host, port, config.esp32.path)
        if ready is not None:
            ready.set()
        server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    from lifeguard.hardware.config import load_hardware_config
    from lifeguard.system.config import load_system_config

    parser = argparse.ArgumentParser(prog="python -m lifeguard.hardware.fake_device", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hardware-config", help="hardware YAML (e.g. configs/hardware.yaml)")
    source.add_argument("--system-config", help="system YAML with a hardware block or hardware_file")
    parser.add_argument("--host", help="server role: host to connect to (default: esp32.bind_host)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.hardware_config:
        config = load_hardware_config(args.hardware_config)
    else:
        sys_cfg = load_system_config(args.system_config)
        if sys_cfg.hardware_error:
            parser.error(sys_cfg.hardware_error)
        config = sys_cfg.hardware
    link = config.esp32
    token = os.environ.get(link.auth_token_env, "") if link.auth_token_env else None
    stop = threading.Event()
    try:
        if link.role == "client":
            host = link.device_host or "127.0.0.1"
            logger.info("fake ESP32 %s acting as the device's WebSocket server (host role: client)",
                        config.device_id)
            serve(config, host, link.device_port, stop, token=token or None)
        else:
            host = args.host or ("127.0.0.1" if link.bind_host in ("0.0.0.0", "") else link.bind_host)
            url = f"ws://{host}:{link.port}{link.path}"
            logger.info("fake ESP32 %s -> %s (Ctrl+C to stop)", config.device_id, url)
            run(config, url, stop, token=token or None)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
