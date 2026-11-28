"""Hardware integration (ESP32 pool prototype): interface, simulator, ESP32 adapter, command and telemetry managers.

Independent of the AI pipeline: perception, tracking and the risk engine never import this package, and hardware
telemetry never feeds risk decisions. Disabled by default (``hardware.enabled: false``); when enabled, the default
mode is ``simulation`` and physical actuation is refused unless explicitly allowed. See docs/HARDWARE_INTEGRATION.md.
"""
from lifeguard.hardware.config import HardwareConfig
from lifeguard.hardware.service import HardwareService

__all__ = ["HardwareConfig", "HardwareService"]
