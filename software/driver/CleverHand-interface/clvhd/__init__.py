"""Python driver for the CleverHand EMG wristband (Teensy 4.0 + ADS1293 modules)."""

from .ads1293 import Config
from .contact import check_contact
from .protocol import ControllerError, find_port
from .wristband import Wristband

__all__ = ["Wristband", "Config", "ControllerError", "find_port", "check_contact"]
__version__ = "1.0.0"
