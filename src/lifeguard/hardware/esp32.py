"""ESP32 adapter (``mode: esp32``): the device session over a WebSocket server on this computer.

Protocol choice (docs/HARDWARE_INTEGRATION.md §3): WebSocket over Wi-Fi, with the ESP32 as client. Reliable
ordered delivery (TCP), one full-duplex connection for telemetry and commands, no broker to install, and the
Python side needs only the optional ``websockets`` package. The shared token (optional) is read from the
environment variable named by ``esp32.auth_token_env``; it is never stored in a config file.

VERIFICATION STATUS: tested against a fake device over a real loopback WebSocket; NOT yet run against a real ESP32.
"""
from __future__ import annotations

import os

from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.interface import HardwareError
from lifeguard.hardware.session import DeviceSession
from lifeguard.hardware.transport import Transport, WebSocketClientTransport, WebSocketServerTransport


def read_token(config: HardwareConfig) -> str | None:
    """The shared token from the configured environment variable; None if no token is configured."""
    name = config.esp32.auth_token_env
    if name is None:
        return None
    token = os.environ.get(name, "")
    if not token:
        raise HardwareError(f"esp32.auth_token_env is set but environment variable {name} is empty or missing")
    return token


class Esp32Adapter(DeviceSession):
    """The physical device link. ``transport`` can be injected (tests)."""

    def __init__(self, config: HardwareConfig, transport: Transport | None = None) -> None:
        super().__init__(config, transport or build_transport(config), kind="esp32", token=read_token(config))


def build_transport(config: HardwareConfig) -> Transport:
    """The WebSocket transport for ``esp32.role`` (everything comes from the configuration)."""
    link = config.esp32
    if link.role == "client":
        r = link.reconnect
        return WebSocketClientTransport(link.endpoint, link.max_message_bytes, link.open_timeout_s,
                                        link.close_timeout_s, r.initial_backoff_s, r.max_backoff_s, r.multiplier,
                                        r.max_attempts)
    return WebSocketServerTransport(link.bind_host, link.port, link.path, link.max_message_bytes,
                                    link.open_timeout_s, link.close_timeout_s, link.allowed_peers)
