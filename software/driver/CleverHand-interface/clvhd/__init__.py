"""Python driver for the CleverHand EMG wristband (Teensy 4.0 + ADS1293 modules)."""

from .ads1293 import Config
from .contact import check_contact
from .protocol import ControllerError, find_port
from .wristband import Wristband

__all__ = ["Config", "ControllerError", "Wristband", "check_contact", "find_port"]
__version__ = "1.0.0"
