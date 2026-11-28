"""Device session: the protocol state machine shared by the ESP32 adapter and the simulator.

Per connection: wait for a valid ``hello`` (right device id, protocol 1, token if configured) -> ``welcome`` ->
CONNECTED. Then: parse every message, drop duplicates / out-of-order sequence numbers, send host heartbeats,
declare the link lost when nothing is heard for ``link_timeout_s`` (and close it, so the device reconnects).
Invalid messages are counted and logged; they never raise.
"""
from __future__ import annotations

import hmac
import logging
from collections import deque

from lifeguard.contracts.timeutil import utc_now
from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.interface import HardwareInterface
from lifeguard.hardware.models import (
    Ack,
    CommEntry,
    DeviceInfo,
    HardwareCommand,
    HardwareEvent,
    LinkState,
    LinkStats,
)
from lifeguard.hardware.protocol import (
    AckMsg,
    Heartbeat,
    Hello,
    ProtocolError,
    Telemetry,
    encode_command,
    encode_error,
    encode_heartbeat,
    encode_welcome,
    parse_device_message,
)
from lifeguard.hardware.telemetry import normalise
from lifeguard.hardware.transport import Transport

logger = logging.getLogger(__name__)
_LOG_FIRST_N = 5
_LOG_EVERY = 50


class DeviceSession(HardwareInterface):
    """Protocol handling over any ``Transport``."""

    def __init__(self, config: HardwareConfig, transport: Transport, kind: str, token: str | None = None) -> None:
        self._cfg = config
        self._transport = transport
        self._kind = kind
        self._token = token
        self._state = LinkState.DISCONNECTED
        self._device: DeviceInfo | None = None
        self._stats = LinkStats()
        self._seen_conn = transport.connection_id
        self._last_seq: int | None = None
        self._hello_deadline: float | None = None
        self._last_hb_sent: float | None = None
        self._host_seq = 0
        # Communication log: link events, hello/welcome/error, commands and acks (telemetry/heartbeats counted only).
        self.comm_log: deque[CommEntry] = deque(maxlen=200)

    # ----------------------------------------------------------------- interface
    @property
    def kind(self) -> str:
        return self._kind

    @property
    def link_state(self) -> LinkState:
        return self._state

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    @property
    def stats(self) -> LinkStats:
        return self._stats

    @property
    def transport(self) -> Transport:
        return self._transport

    def connect(self, now: float) -> None:
        self._transport.start()

    def disconnect(self) -> None:
        self._transport.stop()
        self._state = LinkState.DISCONNECTED
        self._device = None

    def send_command(self, command: HardwareCommand, now: float) -> bool:
        if self._state != LinkState.CONNECTED:
            return False
        text = encode_command(self._cfg.device_id, self._next_seq(), command.cmd_id, command.name, command.target,
                              command.params)
        ok = self._send(text)
        target = f" {command.target}" if command.target else ""
        self._note("out", "command", f"{command.cmd_id} {command.name}{target} {command.params or ''}".rstrip()
                   + ("" if ok else " (not queued: link closed)"))
        return ok

    def _note(self, direction: str, kind: str, text: str) -> None:
        self.comm_log.append(CommEntry(utc_now(), direction, kind, text))

    def poll(self, now: float) -> list[HardwareEvent]:
        events: list[HardwareEvent] = []
        conn_id = self._transport.connection_id
        if conn_id != self._seen_conn:                       # a (re)connection: start a fresh session
            if self._state == LinkState.CONNECTED:
                events.append(self._lost(now, "device reconnected"))
            self._seen_conn = conn_id
            self._device, self._last_seq = None, None
            self._state = LinkState.CONNECTING
            self._hello_deadline = now + self._cfg.hello_timeout_s
            self._note("link", "open", f"connection #{conn_id} from {self._transport.peer or 'device'}; "
                                       "waiting for hello")
        if not self._transport.connected and self._state in (LinkState.CONNECTED, LinkState.CONNECTING):
            if self._state == LinkState.CONNECTED:
                events.append(self._lost(now, "connection closed"))
            else:
                self._note("link", "closed", "connection closed before a valid hello")
            self._state = LinkState.DISCONNECTED
        for raw in self._transport.receive(now):
            event = self._handle(raw, now)
            if event is not None:
                events.append(event)
        if (self._state == LinkState.CONNECTING and self._hello_deadline is not None
                and now > self._hello_deadline):
            self._error("no valid hello in time; closing the connection")
            self._transport.drop("no hello")
            self._state = LinkState.DISCONNECTED
        if self._state == LinkState.CONNECTED:
            last = self._stats.last_heard_t
            if last is not None and now - last > self._cfg.link_timeout_s:
                events.append(self._lost(now, f"nothing heard for {now - last:.1f} s (link timeout)"))
                self._transport.drop("link timeout")
                self._state = LinkState.DISCONNECTED
            elif ((self._last_hb_sent is None or now - self._last_hb_sent >= self._cfg.heartbeat_interval_s)
                  and self._send(encode_heartbeat(self._cfg.device_id, self._next_seq(), int(now * 1000)))):
                self._last_hb_sent = now
        return events

    # ----------------------------------------------------------------- internals
    def _next_seq(self) -> int:
        self._host_seq += 1
        return self._host_seq

    def _send(self, text: str) -> bool:
        ok = self._transport.send(text)
        if ok:
            self._stats.messages_out += 1
            if self._cfg.log_messages:
                logger.debug("hardware -> %s", text)
        return ok

    def _error(self, detail: str) -> HardwareEvent:
        self._stats.last_error = detail
        n = self._stats.invalid_messages
        if n <= _LOG_FIRST_N or n % _LOG_EVERY == 0:
            logger.warning("hardware link: %s", detail)
        self._note("link", "error", detail)
        return HardwareEvent("error", 0.0, detail)

    def _lost(self, now: float, detail: str) -> HardwareEvent:
        self._stats.disconnects += 1
        self._device = None
        logger.warning("hardware device %s disconnected: %s", self._cfg.device_id, detail)
        self._note("link", "disconnected", detail)
        return HardwareEvent("disconnected", now, detail)

    def _handle(self, raw: str | bytes, now: float) -> HardwareEvent | None:
        self._stats.messages_in += 1
        if self._cfg.log_messages:
            logger.debug("hardware <- %r", raw)
        try:
            msg = parse_device_message(raw, self._cfg.esp32.max_message_bytes)
        except ProtocolError as err:
            self._stats.invalid_messages += 1
            event = self._error(f"invalid message: {err}")
            return HardwareEvent("error", now, event.detail)
        if msg.device_id != self._cfg.device_id:
            self._stats.invalid_messages += 1
            detail = f"unexpected device id {msg.device_id!r} (expected {self._cfg.device_id!r})"
            self._error(detail)
            if isinstance(msg, Hello):
                self._send(encode_error(self._cfg.device_id, self._next_seq(), "unexpected device id"))
                self._transport.drop("unexpected device id")
                self._state = LinkState.DISCONNECTED
            return HardwareEvent("error", now, detail)
        if isinstance(msg, Hello):
            return self._on_hello(msg, now)
        if self._state != LinkState.CONNECTED:
            self._stats.invalid_messages += 1
            return HardwareEvent("error", now, self._error(f"{msg.type} before hello ignored").detail)
        if self._last_seq is not None and msg.seq <= self._last_seq:
            self._stats.duplicate_or_old_seq += 1
            return None
        self._last_seq = msg.seq
        self._stats.last_heard_t = now
        if isinstance(msg, Telemetry):
            return HardwareEvent("telemetry", now, telemetry=normalise(msg, self._cfg.components, now, self._kind))
        if isinstance(msg, Heartbeat):
            return HardwareEvent("heartbeat", now)
        if isinstance(msg, AckMsg):
            self._note("in", "ack", f"{msg.cmd_id} {msg.status}" + (f": {msg.detail}" if msg.detail else ""))
            return HardwareEvent("ack", now, ack=Ack(msg.cmd_id, msg.status, msg.detail))
        return None

    def _refuse(self, now: float, reason: str, detail: str) -> HardwareEvent:
        """Send an ``error`` and close the connection (the device may reconnect)."""
        was_connected = self._state == LinkState.CONNECTED
        self._stats.invalid_messages += 1
        self._send(encode_error(self._cfg.device_id, self._next_seq(), reason))
        self._note("out", "error", reason)
        self._transport.drop(reason)
        self._state = LinkState.DISCONNECTED
        if was_connected:
            self._error(detail)
            return self._lost(now, detail)
        return HardwareEvent("error", now, self._error(detail).detail)

    def _on_hello(self, msg: Hello, now: float) -> HardwareEvent:
        self._note("in", "hello", f"device {msg.device_id}, firmware {msg.firmware or 'unknown'}, "
                                  f"token {'present' if msg.token else 'absent'}")   # never the token itself
        if self._state == LinkState.CONNECTED:
            # One hello per connection: a second one would reset the sequence window (replay) -> close.
            return self._refuse(now, "duplicate hello", "duplicate hello on an established connection; closed")
        if self._token is not None and not hmac.compare_digest((msg.token or "").encode("utf-8"),
                                                               self._token.encode("utf-8")):
            return self._refuse(now, "authentication failed", "hello with a wrong or missing token rejected")
        self._seen_conn = self._transport.connection_id     # this hello establishes the current connection
        self._device = DeviceInfo(msg.device_id, msg.firmware, msg.proto,
                                  {k: tuple(v) for k, v in msg.components.items()})
        self._last_seq = msg.seq
        self._stats.last_heard_t = now
        self._stats.connects += 1
        self._state = LinkState.CONNECTED
        cfg = self._cfg
        self._send(encode_welcome(cfg.device_id, self._next_seq(), int(cfg.heartbeat_interval_s * 1000),
                                  int(cfg.telemetry_interval_s * 1000), int(cfg.link_timeout_s * 1000)))
        self._last_hb_sent = now
        missing = [c for kind, ids in (("motors", cfg.components.motors), ("actuators", cfg.components.actuators),
                                       ("sensors", cfg.components.sensors))
                   for c in ids if c not in self._device.components.get(kind, ())]
        detail = f"device {msg.device_id} connected (firmware {msg.firmware or 'unknown'})"
        if missing:
            detail += (f"; configured components not announced by the device (commands to them are refused): "
                       f"{', '.join(missing)}")
            logger.warning("hardware: %s", detail)
        logger.info("hardware: %s", detail)
        self._note("out", "welcome", "identity verified; " + detail)
        return HardwareEvent("connected", now, detail, device=self._device)
