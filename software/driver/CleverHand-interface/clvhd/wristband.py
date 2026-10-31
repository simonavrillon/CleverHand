"""High-level interface to the wristband: Teensy controller + chained ADS1293 modules."""

import copy

from . import ads1293
from .protocol import INIT_TOO_MANY, Controller, ControllerError, find_port
from .stream import Stream


class Wristband:
    """Connect to the Teensy, address the modules and configure them.

    >>> with Wristband(expected_modules=13) as wb:
    ...     wb.configure()
    ...     with wb.stream() as s:
    ...         t_host, t_device, data = s.read(0.05)

    ``expected_modules``: raises unless exactly this many modules are found
    (0 accepts any number). Addressing only runs once per power-up, so on a
    mismatch power-cycle the wristband (Teensy included) and retry.
    ``spi_clock``: the Teensy starts at 4 MHz, too slow to read 13 modules
    within a sample period (469 us): about 40% of their samples are dropped.
    """

    def __init__(self, port=None, expected_modules=13, spi_clock=8_000_000):
        port = port or find_port()
        if port is None:
            raise ControllerError("No Teensy found: pass the serial port explicitly")
        self.port = port
        self.ctrl = Controller(port)
        try:
            self.firmware = self.ctrl.check_version()
            self.nb_modules, status = self.ctrl.address_modules()
            if status == INIT_TOO_MANY:
                print("More than 15 modules on the bus: only the first 15 are used")
            if expected_modules and self.nb_modules != expected_modules:
                raise ControllerError(
                    f"Found {self.nb_modules} modules, expected {expected_modules}. "
                    "Power-cycle the wristband (Teensy included) and retry; if it "
                    f"persists, check the ribbon cable after module {self.nb_modules}"
                )
            if self.nb_modules == 0:
                raise ControllerError(
                    "No module found. Check that the modules are powered and that the ribbon "
                    "cable is plugged into the Teensy, then power-cycle the wristband and retry"
                )
            revid = self.ctrl.read_regs(self.mask, ads1293.REVID, 1)[:, 0]
            not_ads = [m for m in range(self.nb_modules) if revid[m] != 0x01]
            if not_ads:
                raise ControllerError(f"Module(s) {not_ads} do not answer as ADS1293")
            for m in range(self.nb_modules):
                for led in range(2):
                    self.ctrl.set_led(m, led, (0, 0, 0))
            self.set_spi_clock(spi_clock)
        except Exception:
            self.ctrl.close()
            raise
        self.config = None

    @property
    def mask(self):
        return (1 << self.nb_modules) - 1

    def configure(self, config=None, **kwargs):
        """Configure every module with ``config`` (default: ``ads1293.Config()``);
        keyword arguments override its fields, e.g. ``configure(R3=[12, 12, 12])``.
        The passed ``config`` is not modified."""
        cfg = copy.deepcopy(config) if config is not None else ads1293.Config()
        for key, val in kwargs.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unknown configuration field {key!r}")
            setattr(cfg, key, val)
        ads1293.configure(self.ctrl, self.nb_modules, cfg)
        self.config = cfg
        return cfg

    def stream(self):
        """Aligned stream of all modules (use ``with wb.stream() as s``)."""
        if self.config is None:
            self.configure()
        return Stream(self.ctrl, self.nb_modules, self.config)

    def start_conversions(self):
        self.ctrl.write_regs(self.mask, ads1293.CONFIG, [ads1293.START_CONV])

    def stop_conversions(self):
        self.ctrl.write_regs(self.mask, ads1293.CONFIG, [ads1293.STANDBY])

    def read_errors(self):
        """Error flags of every module ('' when none). The flags latch until read."""
        regs = self.ctrl.read_regs(self.mask, ads1293.ERROR_LOD, 7)
        return [ads1293.describe_errors(r) for r in regs]

    def set_led(self, module, led, rgb):
        """Set LED ``led`` (0 or 1) of ``module`` (0-based) to ``rgb`` = (r, g, b), 0-255."""
        self.ctrl.set_led(module, led, rgb)

    def set_spi_clock(self, hz):
        """Teensy-module SPI clock (4 MHz at power-up, ADS1293 max 20 MHz); returns the clock set."""
        return self.ctrl.set_spi_clock(hz)

    def close(self):
        self.ctrl.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
