"""Python driver for the CleverHand EMG wristband (Teensy 4.0 + ADS1293 modules)."""

from .ads1293 import Config
from .protocol import ControllerError, find_port
from .stream import Stream
from .wristband import Wristband

__all__ = ["Wristband", "Config", "Stream", "ControllerError", "find_port"]
__version__ = "1.0.0"
