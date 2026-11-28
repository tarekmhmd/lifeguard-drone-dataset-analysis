"""Process-level owner of the ONE shared hardware service (two-page app: AI Dashboard + Hardware).

Why a module-level hub and not Streamlit session state: session state is per browser tab, so it would give every
tab its own service (duplicate ESP32 connections, port conflicts). This hub lives once per Python process, so every
page, tab and browser session of the same ``streamlit run`` process sees the same ``HardwareService``: one link,
one e-stop latch, one telemetry stream, one command history. (Two separate ``streamlit run`` processes are two
processes: they do not share, and in esp32 mode the second one reports a FAULT instead of connecting twice.)

Lifecycle, all under one lock:

- the first user (Hardware page, or an AI start whose config enables hardware) creates the service from the
  configuration and starts it if ``auto_connect``;
- later users get the SAME instance when their hardware configuration is identical;
- a different configuration is refused while the service is running (no silent reconnect, no second instance);
  a stopped service (operator pressed Disconnect) may be replaced;
- the AI runtime only *uses* the service: stopping the AI does not stop the hardware (the Hardware page keeps it);
- ``reload`` (Hardware page) is refused while the AI runtime uses the service;
- ``shutdown`` (process exit, tests) safe-stops and closes it.
"""
from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path

from lifeguard.contracts.errors import LifeguardError
from lifeguard.hardware.config import HardwareConfig, load_hardware_config_safe
from lifeguard.hardware.service import HardwareService
from lifeguard.system.config import SystemConfig

logger = logging.getLogger(__name__)


class HardwareShareConflict(LifeguardError):
    """The request needs a different hardware configuration than the running shared service."""


class HardwareHub:
    """Holds at most one shared ``HardwareService`` per process (see module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._service: HardwareService | None = None
        self._config: HardwareConfig | None = None      # None for a configuration-error service
        self._error: str | None = None
        self._source = ""
        self._ai_users = 0

    # ------------------------------------------------------------------ queries
    @property
    def service(self) -> HardwareService | None:
        with self._lock:
            return self._service

    @property
    def source(self) -> str:
        """Where the shared service's configuration came from (shown on the Hardware page)."""
        with self._lock:
            return self._source

    @property
    def ai_attached(self) -> bool:
        with self._lock:
            return self._ai_users > 0

    # ------------------------------------------------------------------ Hardware page
    def for_hardware_page(self, path: Path | str) -> tuple[HardwareService | None, str | None]:
        """The shared service, created from ``path`` if there is none yet.

        Returns ``(service, None)``, or ``(None, reason)`` when the file disables hardware.
        """
        with self._lock:
            if self._service is not None:
                return self._service, None
            config, error = load_hardware_config_safe(path)
            if config is not None and not config.enabled:
                return None, f"hardware integration is disabled in {path} (hardware.enabled: false)"
            self._install(config, error, str(path))
            return self._service, None

    def reload(self, path: Path | str) -> str:
        """Replace the shared service with one built from ``path`` (Hardware page "Reload configuration")."""
        with self._lock:
            if self._ai_users:
                return "refused: the AI runtime is using the hardware service; stop the AI first"
            self._drop()
            service, reason = self.for_hardware_page(path)
            return reason or f"reloaded from {path}" + ("" if service is None else f" ({service.config.mode})")

    # ------------------------------------------------------------------ AI runtime
    def acquire_for_ai(self, system_config: SystemConfig, source: str) -> HardwareService | None:
        """The service the AI runtime must use for ``system_config``; None if its hardware is disabled.

        Raises ``HardwareShareConflict`` if a running shared service has a different configuration.
        """
        if system_config.hardware_error is not None:
            config: HardwareConfig | None = None
            error: str | None = system_config.hardware_error
        elif system_config.hardware.enabled:
            config, error = system_config.hardware, None
        else:
            return None                                   # AI only: the shared service (if any) is left alone
        with self._lock:
            if self._service is not None and not self._matches(config, error):
                if self._service.running:
                    raise HardwareShareConflict(
                        f"the shared hardware service is running with the configuration from {self._source}, and "
                        f"the AI configuration {source} specifies different hardware settings. Use a system config "
                        "that references the same hardware file (e.g. configs/system_hardware.yaml), or press "
                        "Disconnect on the Hardware page first.")
                logger.info("shared hardware service (stopped) replaced by the AI configuration %s", source)
                self._drop()
            if self._service is None:
                self._install(config, error, source)
            self._ai_users += 1
            return self._service

    def release_ai(self) -> None:
        """The AI runtime stopped using the service (the service keeps running for the Hardware page)."""
        with self._lock:
            self._ai_users = max(0, self._ai_users - 1)

    # ------------------------------------------------------------------ shutdown
    def shutdown(self) -> None:
        """Safe-stop and close the shared service (process exit, tests)."""
        with self._lock:
            self._drop()
            self._ai_users = 0

    # ------------------------------------------------------------------ internals
    def _matches(self, config: HardwareConfig | None, error: str | None) -> bool:
        return (self._config == config) if error is None else (self._error == error)

    def _install(self, config: HardwareConfig | None, error: str | None, source: str) -> None:
        if error is not None or config is None:
            service = HardwareService.unavailable(error or "no hardware configuration")
        else:
            service = HardwareService(config)
            if config.auto_connect:
                service.start()
        self._service, self._config, self._error, self._source = service, config, error, source
        logger.info("shared hardware service created from %s", source)

    def _drop(self) -> None:
        service, self._service = self._service, None
        self._config, self._error, self._source = None, None, ""
        if service is not None:
            service.stop()


_HUB = HardwareHub()
atexit.register(_HUB.shutdown)


def shared_hub() -> HardwareHub:
    """The process-wide hub used by the two-page application."""
    return _HUB
