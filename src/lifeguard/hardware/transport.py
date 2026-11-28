"""Message transports: how JSON text messages reach the host.

Two WebSocket roles carry the same protocol (``esp32.role`` in the configuration):

- ``WebSocketServerTransport`` (``role: server``, default): this computer listens; the ESP32 connects as a client.
  One device connection at a time; a second client is refused while one is connected; optional peer allow-list.
- ``WebSocketClientTransport`` (``role: client``): this computer connects to the ESP32's WebSocket server at a
  configured address and reconnects with exponential back-off (bounded interval, optional attempt limit).

There is no network discovery: the address comes from the configuration.

Both need the ``websockets`` package (``pip install -e ".[hardware]"``), imported lazily so the AI application
runs without it.

Non-blocking contract (H1): ``send``, ``receive`` and ``drop`` return immediately whatever the peer does.
Outgoing messages go to a per-connection writer thread; connections are closed on a closer thread, bounded by the
close timeout (then the socket is aborted). A silent, half-open or non-reading device can never hold the caller.
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from lifeguard.hardware.interface import HardwareError
from lifeguard.hardware.models import TransportInfo

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 1.0        # default websockets ``close_timeout`` (configurable: esp32.close_timeout_s)
OPEN_TIMEOUT_S = 3.0         # default websockets ``open_timeout`` (configurable: esp32.open_timeout_s)
_STOP_JOIN_S = 5.0           # upper bound for server shutdown and thread joins in ``stop``
_OUTBOX_MAX = 256            # queued messages per connection; full = the device is not reading -> drop it


class Transport(ABC):
    """Carries text messages between the host and one device. ``send``/``receive``/``drop`` must not block."""

    @abstractmethod
    def start(self) -> None:
        """Begin accepting / creating the device connection. Raises HardwareError on failure."""

    @abstractmethod
    def stop(self) -> None:
        """Close everything. Never raises."""

    @abstractmethod
    def send(self, text: str) -> bool:
        """Send one message to the connected device; False if none is connected or the send failed."""

    @abstractmethod
    def receive(self, now: float) -> list[str | bytes]:
        """All messages received since the last call (non-blocking)."""

    @abstractmethod
    def drop(self, reason: str) -> None:
        """Close the current device connection (the device is expected to reconnect)."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """A device connection is open."""

    @property
    @abstractmethod
    def connection_id(self) -> int:
        """Increments for every new device connection (lets the session detect reconnects)."""

    @property
    def peer(self) -> str | None:
        """Address of the connected device, if known."""
        return None

    def info(self) -> TransportInfo:
        """What the transport is doing, for the dashboard. Must not block."""
        return TransportInfo(role="unknown", endpoint="", active=self.connected, peer=self.peer)


class _Link:
    """One device connection: the websockets connection, its send queue and its closing state."""

    def __init__(self, ws: Any, conn_id: int, peer: str | None) -> None:
        self.ws, self.conn_id, self.peer = ws, conn_id, peer
        self.outbox: queue.Queue[str] = queue.Queue(maxsize=_OUTBOX_MAX)
        self.closing = threading.Event()       # set when the connection ends (writer and reader stop)
        self.close_started = False             # a closer thread was started (guarded by the transport lock)


class _WebSocketLinks(Transport):
    """Connection bookkeeping shared by both roles: one current link, inbox, writer and bounded closing."""

    def __init__(self, max_message_bytes: int, open_timeout_s: float, close_timeout_s: float) -> None:
        self._max = max_message_bytes
        self._open_timeout = open_timeout_s
        self._close_timeout = close_timeout_s
        self._force_close_s = close_timeout_s + 0.5      # abort the socket if a close is still stuck
        self._lock = threading.Lock()          # guards _link, _conn_id, _closers (never held during I/O)
        self._link: _Link | None = None
        self._conn_id = 0
        self._closers: list[threading.Thread] = []
        self._inbox: queue.Queue[tuple[int, str | bytes]] = queue.Queue(maxsize=1000)
        self._last_error = ""

    # ------------------------------------------------------------- one connection, start to end
    def _claim(self, ws: Any) -> _Link | None:
        remote = getattr(ws, "remote_address", None)
        with self._lock:
            if self._link is not None:
                return None
            self._conn_id += 1
            link = _Link(ws, self._conn_id, f"{remote[0]}:{remote[1]}" if remote else None)
            self._link = link
            return link

    def _serve_link(self, link: _Link) -> None:
        """Runs on the connection's thread until it ends: reads into the inbox, a writer thread sends."""
        writer = threading.Thread(target=self._write_loop, args=(link,), name="hardware-ws-writer", daemon=True)
        writer.start()
        try:
            for message in link.ws:
                if link.closing.is_set():
                    break
                try:
                    self._inbox.put_nowait((link.conn_id, message))
                except queue.Full:
                    logger.warning("hardware inbox full; dropping a message")
        except Exception as err:  # noqa: BLE001 - any connection error just ends this device session
            logger.info("hardware connection closed: %s", err)
        finally:
            self._detach(link)
            link.closing.set()                 # the writer exits within its poll interval
            writer.join(timeout=self._force_close_s)

    def _write_loop(self, link: _Link) -> None:
        """Sends queued messages; a blocked send only blocks this thread (the close path aborts the socket)."""
        while not link.closing.is_set():
            try:
                text = link.outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if link.closing.is_set():
                return
            try:
                link.ws.send(text)
            except Exception as err:  # noqa: BLE001 - a failed send ends the connection, never raises
                logger.warning("hardware send failed: %s", err)
                self._close_link(link, "send failed")
                return

    def _detach(self, link: _Link) -> bool:
        with self._lock:
            if self._link is link:
                self._link = None
                return True
            return False

    def _close_link(self, link: _Link, reason: str) -> None:
        """Detach now, close in the background (bounded); safe to call repeatedly from any thread."""
        self._detach(link)
        with self._lock:
            if link.close_started:
                return
            link.close_started = True
            link.closing.set()
            closer = threading.Thread(target=self._close, args=(link, reason),
                                      name=f"hardware-ws-close-{link.conn_id}", daemon=True)
            self._closers = [t for t in self._closers if t.is_alive()] + [closer]
        closer.start()

    def _close(self, link: _Link, reason: str) -> None:
        """Closing handshake bounded by the close timeout; abort the socket if even that is stuck."""
        def abort() -> None:
            try:
                link.ws.socket.shutdown(socket.SHUT_RDWR)     # unblocks a send stuck on a non-reading peer
            except OSError:
                pass
            try:
                link.ws.close_socket()
            except Exception as err:  # noqa: BLE001 - already closed
                logger.debug("hardware connection abort: %s", err)

        watchdog = threading.Timer(self._force_close_s, abort)
        watchdog.daemon = True
        watchdog.start()
        try:
            link.ws.close(code=1000, reason=reason[:120])
        except Exception as err:  # noqa: BLE001 - closing an already broken connection
            logger.debug("hardware connection close: %s", err)
        finally:
            watchdog.cancel()

    def _join_closers(self) -> None:
        with self._lock:
            closers, self._closers = self._closers, []
        for closer in closers:
            closer.join(timeout=self._force_close_s + 0.5)
            if closer.is_alive():
                logger.error("hardware link: a connection did not close in time (%s)", closer.name)

    # ------------------------------------------------------------- Transport API
    def send(self, text: str) -> bool:
        """Queue one message for the current connection (never blocks)."""
        with self._lock:
            link = self._link
        if link is None:
            return False
        try:
            link.outbox.put_nowait(text)
            return True
        except queue.Full:
            logger.warning("hardware send queue full: the device is not reading; closing the connection")
            self._close_link(link, "send queue full")
            return False

    def receive(self, now: float) -> list[str | bytes]:
        """Messages of the current connection only (anything left from an older connection is discarded)."""
        with self._lock:
            current = self._conn_id
        out: list[str | bytes] = []
        while True:
            try:
                conn_id, message = self._inbox.get_nowait()
            except queue.Empty:
                return out
            if conn_id == current:
                out.append(message)

    def drop(self, reason: str) -> None:
        """Close the current connection without waiting."""
        with self._lock:
            link = self._link
        if link is not None:
            self._close_link(link, reason)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._link is not None

    @property
    def connection_id(self) -> int:
        with self._lock:
            return self._conn_id

    @property
    def peer(self) -> str | None:
        with self._lock:
            link = self._link
        return link.peer if link is not None else None


class WebSocketServerTransport(_WebSocketLinks):
    """``role: server``: WebSocket server on ``ws://bind_host:port/path``; the ESP32 connects to it."""

    def __init__(self, bind_host: str, port: int, path: str, max_message_bytes: int,
                 open_timeout_s: float = OPEN_TIMEOUT_S, close_timeout_s: float = CLOSE_TIMEOUT_S,
                 allowed_peers: tuple[str, ...] = ()) -> None:
        super().__init__(max_message_bytes, open_timeout_s, close_timeout_s)
        self._host, self._port, self._path = bind_host, port, path
        self._allowed = frozenset(allowed_peers)
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Bound (host, port); useful when ``port`` was 0 in tests."""
        server = self._server
        if server is None:
            return self._host, self._port
        host, port = server.socket.getsockname()[:2]
        return host, port

    def start(self) -> None:
        """Listen. Idempotent while running; after ``stop`` it listens again (same port)."""
        if self._server is not None:
            return
        try:
            from websockets.sync.server import serve
        except ImportError as err:
            raise HardwareError('websockets is not installed: pip install -e ".[hardware]"') from err
        try:
            server = serve(self._handler, self._host, self._port, max_size=self._max,
                           open_timeout=self._open_timeout, close_timeout=self._close_timeout)
        except OSError as err:
            self._last_error = f"cannot listen on {self._host}:{self._port}: {err}"
            raise HardwareError(self._last_error) from err
        self._inbox = queue.Queue(maxsize=1000)
        self._last_error = ""
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="hardware-ws", daemon=True)
        self._thread.start()
        logger.info("hardware link listening on ws://%s:%d%s", self._host, self.address[1], self._path)

    def _handler(self, ws: Any) -> None:
        """Runs on a websockets connection thread for the whole connection."""
        path = getattr(getattr(ws, "request", None), "path", None)
        if path is not None and path.split("?")[0] != self._path:
            ws.close(code=1008, reason="wrong path")
            return
        remote = getattr(ws, "remote_address", None)
        if self._allowed and (remote is None or remote[0] not in self._allowed):
            logger.warning("hardware link: refused connection from %s (not in esp32.allowed_peers)",
                           remote[0] if remote else "unknown")
            self._last_error = f"refused peer {remote[0] if remote else 'unknown'} (not in allowed_peers)"
            ws.close(code=1008, reason="peer not allowed")
            return
        link = self._claim(ws)
        if link is None:
            ws.close(code=1013, reason="a device is already connected")     # outside the lock (bounded)
            return
        self._serve_link(link)

    def stop(self) -> None:
        """Close the connection and the server, bounded. Never raises; logs what is left."""
        try:
            self.drop("host shutting down")
            server, self._server = self._server, None
            thread, self._thread = self._thread, None
            if server is not None:
                shutdown = threading.Thread(target=server.shutdown, name="hardware-ws-shutdown", daemon=True)
                shutdown.start()                 # closes the listening socket first, then remaining connections
                shutdown.join(timeout=_STOP_JOIN_S)
                if shutdown.is_alive():
                    logger.error("hardware link: server shutdown did not finish within %.0f s", _STOP_JOIN_S)
            if thread is not None:
                thread.join(timeout=_STOP_JOIN_S)
                if thread.is_alive():
                    logger.error("hardware link: server thread did not stop within %.0f s", _STOP_JOIN_S)
            self._join_closers()
        except Exception as err:  # noqa: BLE001 - shutdown must never raise
            logger.warning("hardware link shutdown: %s", err)

    def info(self) -> TransportInfo:
        return TransportInfo(role="server", endpoint=f"ws://{self._host}:{self._port}{self._path}",
                             active=self._server is not None, peer=self.peer, last_error=self._last_error)


class WebSocketClientTransport(_WebSocketLinks):
    """``role: client``: connect to the ESP32's WebSocket server; reconnect with exponential back-off.

    Connection attempts run on a background thread (``open_timeout`` bounds each one), so nothing here waits on
    the network in the caller. ``max_attempts`` > 0 gives up after that many failures in a row; ``start`` again
    (dashboard Connect) resets the count.
    """

    def __init__(self, url: str, max_message_bytes: int, open_timeout_s: float = OPEN_TIMEOUT_S,
                 close_timeout_s: float = CLOSE_TIMEOUT_S, initial_backoff_s: float = 1.0,
                 max_backoff_s: float = 30.0, multiplier: float = 2.0, max_attempts: int = 0) -> None:
        super().__init__(max_message_bytes, open_timeout_s, close_timeout_s)
        self._url = url
        self._initial, self._max_backoff, self._mult, self._max_attempts = (
            initial_backoff_s, max_backoff_s, multiplier, max_attempts)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._attempt = 0
        self._next_at: float | None = None
        self._gave_up = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            from websockets.sync.client import connect  # noqa: F401 - availability check only
        except ImportError as err:
            raise HardwareError('websockets is not installed: pip install -e ".[hardware]"') from err
        self._stop = threading.Event()
        self._inbox = queue.Queue(maxsize=1000)
        self._attempt, self._next_at, self._gave_up, self._last_error = 0, None, False, ""
        self._thread = threading.Thread(target=self._run, args=(self._stop,), name="hardware-ws-connect",
                                        daemon=True)
        self._thread.start()
        logger.info("hardware link: connecting to %s", self._url)

    def _run(self, stop: threading.Event) -> None:
        from websockets.sync.client import connect

        delay = self._initial
        while not stop.is_set():
            self._next_at = None
            try:
                ws = connect(self._url, open_timeout=self._open_timeout, close_timeout=self._close_timeout,
                             max_size=self._max, proxy=None, legacy=True)   # legacy: return the connection
            except Exception as err:  # noqa: BLE001 - every failure is retried (or counted toward giving up)
                self._attempt += 1
                self._last_error = f"connect to {self._url} failed: {type(err).__name__}: {err}"
                if self._attempt <= 3 or self._attempt % 20 == 0:
                    logger.warning("hardware link: %s (attempt %d)", self._last_error, self._attempt)
                if self._max_attempts and self._attempt >= self._max_attempts:
                    self._gave_up = True
                    logger.error("hardware link: giving up after %d attempts (press Connect to retry)",
                                 self._attempt)
                    return
                self._next_at = time.monotonic() + delay
                stop.wait(delay)
                delay = min(delay * self._mult, self._max_backoff)
                continue
            if stop.is_set():
                ws.close()
                return
            self._attempt, delay, self._last_error = 0, self._initial, ""
            link = self._claim(ws)
            if link is None:                         # cannot happen with one connector; be safe anyway
                ws.close()
                continue
            logger.info("hardware link: connected to %s", self._url)
            self._serve_link(link)                   # returns when the connection ends (or is dropped)
            if not stop.is_set():
                self._next_at = time.monotonic() + self._initial
                stop.wait(self._initial)             # the device restarts its server: retry quickly

    def stop(self) -> None:
        try:
            self._stop.set()
            self.drop("host shutting down")
            thread, self._thread = self._thread, None
            if thread is not None:
                thread.join(timeout=self._open_timeout + self._force_close_s + 1.0)
                if thread.is_alive():
                    logger.error("hardware link: connector thread did not stop in time")
            self._join_closers()
        except Exception as err:  # noqa: BLE001 - shutdown must never raise
            logger.warning("hardware link shutdown: %s", err)

    def info(self) -> TransportInfo:
        next_at = self._next_at
        retry = None if next_at is None else max(0.0, next_at - time.monotonic())
        active = self._thread is not None and self._thread.is_alive()
        return TransportInfo(role="client", endpoint=self._url, active=active, peer=self.peer,
                             attempt=self._attempt, next_retry_in_s=retry, last_error=self._last_error,
                             gave_up=self._gave_up)
