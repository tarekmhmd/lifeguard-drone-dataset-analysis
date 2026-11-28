"""Common hardware interface: implemented by the simulator and the ESP32 adapter.

The AI pipeline never uses this directly: commands go through ``CommandManager`` (validation, permissions,
safety gates) inside ``HardwareService``. Adapters only move messages and report link state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from lifeguard.hardware.models import DeviceInfo, HardwareCommand, HardwareEvent, LinkState, LinkStats


class HardwareInterface(ABC):
    """One hardware device behind a link. All methods are called from the service thread."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """``simulation`` or ``esp32``."""

    @abstractmethod
    def connect(self, now: float) -> None:
        """Open the link (non-blocking). Raises ``HardwareError`` if it cannot even start (e.g. port in use)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the link and release resources. Never raises."""

    @abstractmethod
    def poll(self, now: float) -> list[HardwareEvent]:
        """Exchange pending messages and report connection changes, telemetry, acknowledgements and errors."""

    @abstractmethod
    def send_command(self, command: HardwareCommand, now: float) -> bool:
        """Transmit one command; False if the link cannot carry it now (caller marks it FAILED)."""

    @property
    @abstractmethod
    def link_state(self) -> LinkState:
        """Current connection state."""

    @property
    @abstractmethod
    def device(self) -> DeviceInfo | None:
        """Identity reported by the connected device, None when not connected."""

    @property
    @abstractmethod
    def stats(self) -> LinkStats:
        """Communication counters."""


class HardwareError(RuntimeError):
    """The hardware link could not be started or used."""
