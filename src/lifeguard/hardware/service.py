"""Hardware service: owns the adapter, the telemetry and command managers and a background loop.

The runtime creates it only when ``hardware.enabled`` is true. It never raises into the AI application: if the
link cannot start (missing package, port in use, missing token) the service reports FAULT and the AI pipeline
runs on unchanged. Hardware telemetry is displayed and logged only; it does not feed the risk engine.

Locking (a dead or slow link must never hold up status reads, the dashboard or the video pipeline):

- ``_lock`` guards the service state: command manager, telemetry, errors and the *published* link view
  (state, device, counters). It is only ever held for short in-memory work, never around adapter or network I/O.
  ``status()`` and ``request()`` take only this lock.
- ``_io_lock`` serialises every call into the adapter (protocol session + transport), which is not thread-safe.
  Order: ``_io_lock`` may be held while taking ``_lock``, never the reverse, so the two cannot deadlock.
- The command manager does not touch the adapter: an accepted command is queued and sent by whoever holds
  ``_io_lock`` next (the caller, if the link is idle, else the service thread, woken at once). A command that
  was superseded, finished, or belongs to an older connection by then is not sent.
- ``_lifecycle`` serialises ``start``/``stop``.
"""
from __future__ import annotations

import dataclasses
import hmac
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from lifeguard.contracts.timeutil import utc_now
from lifeguard.hardware.commands import CommandManager
from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.interface import HardwareInterface
from lifeguard.hardware.models import (
    CommandRecord,
    CommandStatus,
    CommEntry,
    DeviceInfo,
    HardwareCommand,
    HardwareEvent,
    HardwareStatus,
    LinkState,
    LinkStats,
    TransportInfo,
)
from lifeguard.hardware.telemetry import TelemetryManager

logger = logging.getLogger(__name__)
_PERIOD_S = 0.05
_SHUTDOWN_ACK_WAIT_S = 1.0
_THREAD_JOIN_S = 2.0          # bounded wait for the service thread in stop()
_IO_LOCK_WAIT_S = 1.0         # bounded wait for a busy adapter in stop() (it is closed anyway afterwards)


def build_adapter(config: HardwareConfig) -> HardwareInterface:
    """The adapter for the configured mode (imports are local so simulation needs no extra packages)."""
    if config.mode == "esp32":
        from lifeguard.hardware.esp32 import Esp32Adapter

        return Esp32Adapter(config)
    from lifeguard.hardware.simulator import SimulatedHardware

    return SimulatedHardware(config)


class _CommandLink(HardwareInterface):
    """What the command manager sees: the published link state and an outgoing queue. Does no I/O."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self.state = LinkState.DISCONNECTED
        self.epoch = 0                                  # the adapter's ``connects`` counter when published
        self.queue: deque[tuple[str, int]] = deque()    # (cmd_id, epoch) waiting to be sent
        self.announced: dict[str, tuple[str, ...]] | None = None   # components in the device's hello

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def link_state(self) -> LinkState:
        return self.state

    def send_command(self, command: HardwareCommand, now: float) -> bool:
        if self.state != LinkState.CONNECTED:
            return False
        self.queue.append((command.cmd_id, self.epoch))
        return True

    def connect(self, now: float) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def poll(self, now: float) -> list[HardwareEvent]:
        return []

    @property
    def device(self) -> DeviceInfo | None:
        return None

    @property
    def stats(self) -> LinkStats:
        return LinkStats()


class HardwareService:
    """Thread-safe facade used by the runtime, the response bridge and the dashboard."""

    def __init__(self, config: HardwareConfig, adapter: HardwareInterface | None = None,
                 clock: Callable[[], float] = time.monotonic, config_error: str | None = None) -> None:
        self.config = config
        self.config_error = config_error
        self._clock = clock
        self._lock = threading.RLock()        # service state (short sections only, no I/O)
        self._io_lock = threading.Lock()      # the adapter (may block on a misbehaving link)
        self._lifecycle = threading.Lock()    # start / stop
        self._errors: deque[str] = deque(maxlen=20)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._running = False
        self._fault: str | None = None
        self._fault_permanent = False         # adapter could not be built (configuration): start() cannot fix it
        self.telemetry = TelemetryManager(config.stale_telemetry_s)
        self._host_log: deque[CommEntry] = deque(maxlen=100)
        self._comm: tuple[CommEntry, ...] = ()
        self._transport_info: TransportInfo | None = None
        self._stale_stop_sent = False
        if config_error is not None:             # never substitute simulation for a broken configuration
            self.adapter: HardwareInterface = _NullAdapter("unconfigured")
            self._set_fault(f"hardware configuration error: {config_error}")
            self._fault_permanent = True
        else:
            try:
                self.adapter = adapter or build_adapter(config)
            except Exception as err:  # noqa: BLE001 - e.g. a missing auth token: report, never crash the app
                self.adapter = _NullAdapter(config.mode)
                self._set_fault(f"hardware adapter unavailable: {err}")
                self._fault_permanent = True
        self._link = _CommandLink(self.adapter.kind)
        self._device: DeviceInfo | None = None
        self._stats = LinkStats()
        self.commands = CommandManager(config, self._link, self.telemetry)
        with self._io_lock, self._lock:
            self._publish()

    @classmethod
    def unavailable(cls, detail: str) -> HardwareService:
        """A service that only reports a configuration FAULT (missing/invalid hardware file); it never connects."""
        return cls(HardwareConfig(enabled=True), config_error=detail)

    # ----------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return self._running

    def connect(self, by: str = "operator") -> bool:
        """Operator Connect: open the link (idempotent; restarts a client connector that gave up)."""
        self._host_note("connect", f"Connect requested by {by or 'operator'}")
        if self._running:
            transport = getattr(self.adapter, "transport", None)
            info = transport.info() if transport is not None else None
            if info is not None and info.role == "client" and info.gave_up:
                with self._io_lock:
                    transport.start()          # type: ignore[union-attr]  # new connector, attempts reset
                return True
        return self.start()

    def disconnect(self, by: str = "operator") -> None:
        """Operator Disconnect: safe-stop (if connected), close the link and stop the service thread."""
        self._host_note("disconnect", f"Disconnect requested by {by or 'operator'}")
        self.stop()

    def start(self, background: bool = True) -> bool:
        """Open the link and (by default) start the polling thread. Never raises; True if running afterwards.

        Idempotent: starting a running service does nothing. After ``stop`` it starts again (same port). After
        a failed start (e.g. port in use) the FAULT stays until a later ``start`` succeeds; anything the failed
        attempt opened is closed. If the adapter could not even be built (configuration), it never starts.
        """
        with self._lifecycle:
            if self._running:
                logger.info("hardware service already running; start() ignored")
                return True
            if self._fault_permanent:
                return False
            with self._io_lock:
                try:
                    self.adapter.connect(self._clock())
                except Exception as err:  # noqa: BLE001
                    _quiet_disconnect(self.adapter)          # release whatever the failed attempt opened
                    with self._lock:
                        self._set_fault(f"hardware link could not start: {err}")
                        self._publish()
                    return False
                with self._lock:
                    self._fault = None
                    self._publish()
            self._running = True
            logger.info("hardware service started: mode=%s device=%s physical_actuation=%s", self.config.mode,
                        self.config.device_id, self.config.allow_physical_actuation)
            if background:
                self._stop.clear()
                self._wake.clear()
                self._thread = threading.Thread(target=self._loop, name="hardware-service", daemon=True)
                self._thread.start()
            return True

    def stop(self) -> None:
        """Safe-stop the device if connected, then close the link. Never raises; bounded in time."""
        with self._lifecycle:
            try:
                self._shutdown()
            except Exception as err:  # noqa: BLE001
                logger.warning("hardware service stop: %s", err)

    def _shutdown(self) -> None:
        record = None
        if self._running and self.config.safe_stop_on_shutdown:
            with self._lock:
                if self._fault is None and self._link.state == LinkState.CONNECTED:
                    record = self.commands.request("safe_stop", source="system", by="shutdown", now=self._clock())
            self._try_flush()
        if record is not None and record.status == CommandStatus.PENDING:
            deadline = time.monotonic() + _SHUTDOWN_ACK_WAIT_S
            while time.monotonic() < deadline and not self._final(record.command.cmd_id):
                if self._thread is None:
                    self.step()
                time.sleep(0.02)
        self._stop.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=_THREAD_JOIN_S)
            if thread.is_alive():
                self._note_error(f"hardware service thread did not stop within {_THREAD_JOIN_S:.0f} s "
                                 "(link busy); closing the link anyway")
        got = self._io_lock.acquire(timeout=_IO_LOCK_WAIT_S)
        try:
            self.adapter.disconnect()            # transports bound their own shutdown; never left listening
            with self._lock:
                self._publish()
        finally:
            if got:
                self._io_lock.release()
        if self._running:
            self._host_note("stopped", "link closed; service stopped")
        self._running = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._wake.wait(_PERIOD_S)
            self._wake.clear()

    def step(self, now: float | None = None) -> None:
        """One iteration: poll the link, route events, check command timeouts, send. Never raises."""
        with self._io_lock:                       # adapter I/O: does not block status() or request()
            now = self._clock() if now is None else now
            try:
                events = self.adapter.poll(now)
            except Exception as err:  # a bug here must not take the AI application down
                logger.exception("hardware link poll failed")
                events = []
                self._note_error(f"internal error: {type(err).__name__}: {err}")
            with self._lock:                      # short, in-memory only
                try:
                    self._publish()
                    for event in events:
                        self._route(event, now)
                    self.commands.tick(now)
                    self._stale_guard(now)
                except Exception as err:
                    logger.exception("hardware service step failed")
                    self._errors.append(f"internal error: {type(err).__name__}: {err}")
            self._flush(now)

    def _route(self, event: HardwareEvent, now: float) -> None:
        if event.kind == "telemetry" and event.telemetry is not None:
            self.telemetry.update(event.telemetry)
            self.commands.on_telemetry_estop(event.telemetry.estop_engaged)
        elif event.kind == "ack" and event.ack is not None:
            self.commands.on_ack(event.ack, now)
        elif event.kind == "connected":
            self._stale_stop_sent = False
            if self.commands.estop_latched:          # recovery: the device must be latched too (it may have rebooted)
                self.commands.request("emergency_stop", source="system", by="on-connect (host latch set)", now=now)
            elif self.config.safe_stop_on_connect:   # nothing keeps moving from a previous session
                self.commands.request("safe_stop", source="system", by="on-connect", now=now)
        elif event.kind == "disconnected":
            self.telemetry.clear()
            self.commands.on_link_lost(now)
            self._errors.append(f"disconnected: {event.detail}")
        elif event.kind == "error":
            self._errors.append(event.detail)

    def _stale_guard(self, now: float) -> None:
        """Telemetry went stale while something moves -> safe stop once (we can no longer see the device)."""
        if not self.telemetry.is_stale(now):
            self._stale_stop_sent = False
            return
        if (self.config.safe_stop_on_stale_telemetry and not self._stale_stop_sent
                and self._link.state == LinkState.CONNECTED and self.commands.has_active_actuation()):
            self._stale_stop_sent = True
            logger.error("hardware telemetry stale while an output is active: sending safe_stop")
            self.commands.request("safe_stop", source="system", by="stale-telemetry", now=now)

    # ----------------------------------------------------------------- commands
    def request(self, name: str, target: str | None = None, params: dict[str, Any] | None = None, *,
                source: str, by: str, pin: str | None = None) -> CommandRecord:
        with self._lock:
            now = self._clock()
            if self._fault is not None:
                return _rejected(name, target, source, by, self._fault, now)
            if name == "clear_emergency_stop":
                problem = self._authorize_clear(by, pin)
                if problem is not None:
                    return _rejected(name, target, source, by, problem, now)
            record = self.commands.request(name, target, params, source=source, by=by, now=now)
        self._try_flush(now)
        return record

    def _authorize_clear(self, by: str, pin: str | None) -> str | None:
        """Operator authorisation for clearing the e-stop (permissions are checked by the command manager)."""
        rules = self.config.estop_clear
        name = by.strip()
        if rules.operators and name.lower() not in {o.lower() for o in rules.operators}:
            return f"operator {name!r} is not authorised to clear the emergency stop (hardware.estop_clear.operators)"
        if rules.pin_env is not None:
            expected = os.environ.get(rules.pin_env, "")
            if not expected:
                return f"clearing needs the PIN in environment variable {rules.pin_env}, which is not set"
            if not hmac.compare_digest((pin or "").encode("utf-8"), expected.encode("utf-8")):
                return "wrong PIN: emergency stop not cleared"
        return None

    def emergency_stop(self, by: str, source: str = "operator") -> CommandRecord:
        logger.error("EMERGENCY STOP requested by %s (%s)", by or "-", source)
        return self.request("emergency_stop", source=source, by=by or source)

    def safe_stop(self, by: str, source: str = "operator") -> CommandRecord:
        return self.request("safe_stop", source=source, by=by or source)

    def clear_emergency_stop(self, by: str, pin: str | None = None) -> CommandRecord:
        return self.request("clear_emergency_stop", source="operator", by=by, pin=pin)

    # ----------------------------------------------------------------- simulation controls
    @property
    def simulator(self) -> Any:
        """The simulated device's test/demo hooks, or None in esp32 mode."""
        return getattr(self.adapter, "device_sim", None)

    SIMULATION_ACTIONS: tuple[str, ...] = (
        "go_offline", "come_online", "freeze_telemetry", "resume_telemetry", "inject_fault", "clear_faults",
        "corrupt_message", "press_physical_estop", "release_physical_estop", "ignore_commands", "answer_commands",
    )

    def simulate(self, action: str, arg: str = "") -> str:
        """Drive the simulated device (demo/test). Thread-safe; refused outside simulation mode."""
        sim = self.simulator
        if sim is None:
            return "not available: hardware.mode is not 'simulation' (a real device cannot be simulated)"
        if action not in self.SIMULATION_ACTIONS:
            return f"unknown simulation action {action!r}"
        with self._io_lock:                       # the simulated link is used by the service thread too
            if action == "go_offline":
                sim.go_offline()
            elif action == "come_online":
                sim.come_online()
            elif action in ("freeze_telemetry", "resume_telemetry"):
                sim.telemetry_frozen = action == "freeze_telemetry"
            elif action == "inject_fault":
                sim.inject_fault(arg or "motor_driver_fault")
            elif action == "clear_faults":
                sim.clear_faults()
            elif action == "corrupt_message":
                sim.send_corrupt_message()
            elif action in ("press_physical_estop", "release_physical_estop"):
                sim.press_physical_estop(action == "press_physical_estop")
            else:
                sim.ignore_commands(action == "ignore_commands")
        text = f"simulated: {action.replace('_', ' ')}" + (f" ({arg})" if arg else "")
        self._host_note("simulation", text)
        return text

    def _host_note(self, kind: str, text: str) -> None:
        with self._lock:
            self._host_log.append(CommEntry(utc_now(), "host", kind, text))

    def status(self) -> HardwareStatus:
        """A consistent snapshot. Takes only the state lock: never waits for link I/O."""
        with self._lock:
            now = self._clock()
            link = LinkState.FAULT if self._fault is not None else self._link.state
            heard = self._stats.last_heard_t
            comm = tuple(sorted((*self._comm, *self._host_log), key=lambda e: e.t_utc))[-100:]
            return HardwareStatus(
                enabled=True, mode="unconfigured" if self.config_error else self.config.mode,
                device_id=self.config.device_id,
                physical_actuation_allowed=self.config.mode == "esp32" and self.config.allow_physical_actuation,
                link=link, device=self._device, telemetry=self.telemetry.latest,
                telemetry_age_s=self.telemetry.age_s(now), telemetry_stale=self.telemetry.is_stale(now),
                estop_latched=self.commands.estop_latched, faults=self.telemetry.faults,
                stats=dataclasses.replace(self._stats), commands=self.commands.history[-30:],
                errors=tuple(self._errors), running=self._running, transport=self._transport_info,
                last_heard_age_s=None if heard is None or link != LinkState.CONNECTED else max(0.0, now - heard),
                comm_log=comm, config_error=self.config_error,
                estop_clear_needs_pin=self.config.estop_clear.pin_env is not None,
                estop_clear_operators=self.config.estop_clear.operators,
            )

    def describe(self) -> str:
        """One line for the runtime's component list."""
        if self._fault is not None:
            return f"FAULT ({self._fault}); AI pipeline unaffected"
        if self.config.mode == "simulation":
            return f"simulation ({self.config.device_id}); no physical hardware"
        gate = "ENABLED" if self.config.allow_physical_actuation else "disabled"
        link = self.config.esp32
        return (f"esp32 {self.config.device_id} on ws://{link.bind_host}:{link.port}{link.path}; "
                f"physical actuation {gate}")

    # ----------------------------------------------------------------- internals
    def _publish(self) -> None:
        """Copy the adapter's link view for lock-free readers. Caller holds ``_io_lock`` and ``_lock``."""
        stats = self.adapter.stats
        self._link.state = self.adapter.link_state
        self._link.epoch = stats.connects
        self._device = self.adapter.device
        self._link.announced = dict(self._device.components) if self._device is not None else None
        self._stats = dataclasses.replace(stats)
        log = getattr(self.adapter, "comm_log", None)
        if log is not None:
            self._comm = tuple(log)
        transport = getattr(self.adapter, "transport", None)
        try:
            self._transport_info = transport.info() if transport is not None else None
        except Exception as err:  # noqa: BLE001 - display only
            logger.debug("hardware transport info: %s", err)

    def _try_flush(self, now: float | None = None) -> None:
        """Send queued commands now if the adapter is free; otherwise wake the service thread to do it."""
        if self._io_lock.acquire(blocking=False):
            try:
                self._flush(self._clock() if now is None else now)
            finally:
                self._io_lock.release()
        else:
            self._wake.set()

    def _flush(self, now: float) -> None:
        """Send every queued command that is still due. Caller holds ``_io_lock`` (not ``_lock``)."""
        while True:
            with self._lock:
                batch = self._due(now)
            if not batch:
                return
            failed: list[str] = []
            for record in batch:
                try:
                    ok = self.adapter.send_command(record.command, now)
                except Exception as err:  # noqa: BLE001 - reported as a failed send
                    logger.warning("hardware send of %s raised: %s", record.command.cmd_id, err)
                    ok = False
                if not ok:
                    failed.append(record.command.cmd_id)
            with self._lock:
                for cmd_id in failed:
                    self.commands.on_send_failed(cmd_id, now)
                self._publish()

    def _due(self, now: float) -> list[CommandRecord]:
        """Pop the queue: commands still pending on the same connection. Caller holds both locks."""
        out: list[CommandRecord] = []
        seen: set[str] = set()
        while self._link.queue:
            cmd_id, epoch = self._link.queue.popleft()
            if cmd_id in seen:
                continue
            seen.add(cmd_id)
            record = self.commands.pending(cmd_id)
            if record is None:
                continue                          # superseded / finished / failed before it went out
            if epoch != self._link.epoch:
                self.commands.on_send_failed(cmd_id, now, "link changed before the command could be sent")
                continue
            out.append(record)
        return out

    def _final(self, cmd_id: str) -> bool:
        with self._lock:
            return all(r.command.cmd_id != cmd_id for r in self.commands.active)

    def _note_error(self, detail: str) -> None:
        logger.error("%s", detail)
        with self._lock:
            self._errors.append(detail)

    def _set_fault(self, detail: str) -> None:
        self._fault = detail
        self._errors.append(detail)
        logger.error("%s; the AI pipeline continues without hardware", detail)


def _quiet_disconnect(adapter: HardwareInterface) -> None:
    try:
        adapter.disconnect()
    except Exception as err:  # noqa: BLE001 - cleanup after a failed start
        logger.debug("hardware adapter cleanup: %s", err)


def _rejected(name: str, target: str | None, source: str, by: str, detail: str, now: float) -> CommandRecord:
    command = HardwareCommand("c-none", name, target, {}, source, by)
    return CommandRecord(command, CommandStatus.REJECTED, detail, t_created=now, t_done=now)


class _NullAdapter(HardwareInterface):
    """Stand-in when the real adapter could not be built (always disconnected)."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._stats = LinkStats()

    @property
    def kind(self) -> str:
        return self._mode

    def connect(self, now: float) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def poll(self, now: float) -> list[Any]:
        return []

    def send_command(self, command: HardwareCommand, now: float) -> bool:
        return False

    @property
    def link_state(self) -> LinkState:
        return LinkState.FAULT

    @property
    def device(self) -> None:
        return None

    @property
    def stats(self) -> LinkStats:
        return self._stats
