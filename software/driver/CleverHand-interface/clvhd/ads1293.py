"""ADS1293 configuration and data conversion (datasheet TI SNAS602C)."""

from dataclasses import dataclass, field

import numpy as np

# registers
CONFIG = 0x00
FLEX_CH_CN = 0x01  # 0x01-0x03, one per channel
CMDET_EN = 0x0A
RLD_CN = 0x0C
OSC_CN = 0x12
AFE_RES = 0x13
AFE_SHDN_CN = 0x14
ERROR_LOD = 0x18  # 0x18-0x1E: LOD, STATUS, RANGE1-3, SYNC, MISC
R2_RATE = 0x21
R3_RATE_CH = 0x22  # 0x22-0x24
R1_RATE = 0x25
DRDYB_SRC = 0x27
DATA_STATUS = 0x30
DATA_CH_ECG = 0x37  # 0x37-0x3F, 3 bytes per channel, MSB first
REVID = 0x40

START_CONV, STANDBY = 0x01, 0x02

VREF = 2.4
INA_GAIN = 3.5

R2_CODES = {4: 0b0001, 5: 0b0010, 6: 0b0100, 8: 0b1000}
R3_CODES = {4: 1, 6: 2, 8: 4, 12: 8, 16: 16, 32: 32, 64: 64, 128: 128}


@dataclass
class Config:
    """Configuration of every ADS1293 module.

    The defaults suit DRY_ELEC_6CH_THIN: its electrodes are ordered E4 E5 E6
    E1 E2 E3 along the strip (5.08 mm pitch) and E6 is the extRef/bias net
    shared by all modules, so the channels use the three neighbouring pairs.
    The modules carry a 4.000 MHz crystal (FOX 4000), not the datasheet's
    4.096 MHz, so every rate and bandwidth is 1/1.024 of the datasheet's:
    200 kHz with R1=4, R2=4, R3=6 gives 2083 sps, 420 Hz bandwidth and
    3.4 uVrms noise (datasheet table 10).
    """
    route: list = field(default_factory=lambda: [[4, 5], [1, 2], [2, 3]])  # (+, -) input of each channel
    enable: list = field(default_factory=lambda: [True, True, True])
    high_res: list = field(default_factory=lambda: [True, True, True])
    high_freq: list = field(default_factory=lambda: [True, True, True])  # clock_hz / 20 (else / 40) modulator
    R1: list = field(default_factory=lambda: [4, 4, 4])  # 2 or 4
    R2: int = 4  # 4, 5, 6 or 8
    R3: list = field(default_factory=lambda: [6, 6, 6])  # 4, 6, 8, 12, 16, 32, 64 or 128
    # module whose right-leg drive biases the body through rld_input (None:
    # no bias). IN6 is the extRef net shared by all modules, so only one
    # module may drive it.
    rld_module: int = 0
    rld_input: int = 6
    # share module 0's clock over its CLK pin (needs the EMG J1 -> COM_MOD
    # J19/AUX1 wire on every node); otherwise each module uses its own crystal
    shared_clock: bool = False
    clock_hz: float = 4_000_000  # module crystal

    def odr(self, ch=0):
        """ECG output data rate of a channel (sps)."""
        fs = self.clock_hz / (20 if self.high_freq[ch] else 40)
        return fs / (self.R1[ch] * self.R2 * self.R3[ch])

    def used_inputs(self):
        """Inputs used by the enabled channels (bit i: IN(i+1))."""
        mask = 0
        for ch in range(3):
            if self.enable[ch]:
                for pin in self.route[ch]:
                    if 1 <= pin <= 6:
                        mask |= 1 << (pin - 1)
        return mask

    def drdy_mask(self):
        """DATA_STATUS bits set when every enabled ECG channel has a new sample."""
        return sum(0x20 << ch for ch in range(3) if self.enable[ch])

    def adc_max(self):
        """ECG ADC_MAX of each channel (datasheet tables 8-11)."""
        out = []
        for ch in range(3):
            r3_6_or_12 = self.R3[ch] in (6, 12)
            out.append({4: (0xF30000, 0x800000), 8: (0xF30000, 0x800000),
                        5: (0xB964F0, 0xC35000), 6: (0xE6A900, 0xF30000)}
                       [self.R2][0 if r3_6_or_12 else 1])
        return np.array(out, dtype=np.float64)

    def validate(self):
        if self.R2 not in R2_CODES:
            raise ValueError(f"R2 must be one of {sorted(R2_CODES)}")
        for ch in range(3):
            if self.R1[ch] not in (2, 4):
                raise ValueError("R1 must be 2 or 4")
            if self.R3[ch] not in R3_CODES:
                raise ValueError(f"R3 must be one of {sorted(R3_CODES)}")
            for pin in self.route[ch]:
                if not 0 <= pin <= 6:
                    raise ValueError("route inputs must be 1-6 (0: unused)")
        if not 1 <= self.rld_input <= 6:
            raise ValueError("rld_input must be 1-6")
        if len({self.odr(ch) for ch in range(3) if self.enable[ch]}) > 1:
            raise ValueError("enabled channels must share the same data rate")
        if self.drdy_mask() == 0:
            raise ValueError("no channel enabled")


def register_values(cfg):
    """Registers written to every module (except RLD_CN and OSC_CN), in write order."""
    regs = [(CONFIG, STANDBY)]
    for ch in range(3):
        pos, neg = cfg.route[ch]
        val = pos | (neg << 3)
        if pos == neg:
            val |= 0xC0  # inputs shorted (zero test signal)
        regs.append((FLEX_CH_CN + ch, val))
    regs.append((CMDET_EN, cfg.used_inputs()))
    # AFE_SHDN_CN: bit ch shuts the INA down, bit ch+3 the modulator
    regs.append((AFE_SHDN_CN, sum(0b001001 << ch for ch in range(3) if not cfg.enable[ch])))
    regs.append((AFE_RES, sum((1 << ch) * bool(cfg.high_res[ch]) for ch in range(3))
                 | sum((8 << ch) * bool(cfg.high_freq[ch]) for ch in range(3))))
    regs.append((R1_RATE, sum(1 << ch for ch in range(3) if cfg.R1[ch] == 2)))
    regs.append((R2_RATE, R2_CODES[cfg.R2]))
    for ch in range(3):
        regs.append((R3_RATE_CH + ch, R3_CODES[cfg.R3[ch]]))
    regs.append((DRDYB_SRC, 0x08))  # data ready follows channel 1 ECG, as in the datasheet examples
    return regs


def configure(ctrl, nb_modules, cfg):
    """Write ``cfg`` to the modules and read every register back."""
    cfg.validate()
    if cfg.rld_module is not None and not 0 <= cfg.rld_module < nb_modules:
        raise ValueError(f"rld_module must be None or 0-{nb_modules - 1}")
    every = (1 << nb_modules) - 1
    expected = {}
    # Configure from standby, as in the datasheet examples: power-down may
    # also stop the crystal oscillator, which then needs 15 ms to restart.
    for reg, val in register_values(cfg):
        ctrl.write_regs(every, reg, [val])
        expected[reg] = np.full(nb_modules, val)

    # Right-leg drive: SELRLD (bits 2:0) connects the output to an input,
    # SHDN_RLD (bit 3) powers unused amplifiers down
    rld = np.full(nb_modules, 0x08)
    if cfg.rld_module is not None:
        rld[cfg.rld_module] = cfg.rld_input
    for m in range(nb_modules):
        ctrl.write_regs(1 << m, RLD_CN, [int(rld[m])])
    expected[RLD_CN] = rld

    # OSC_CN: STRTCLK (bit 2) only once the clock runs. The crystal
    # oscillator runs from power-up; a shared clock needs the master's CLK
    # output (bit 0) before the others switch to it (SHDN_OSC, bit 1).
    if cfg.shared_clock and nb_modules > 1:
        others = every & ~1
        ctrl.write_regs(1, OSC_CN, [0x05])
        ctrl.write_regs(others, OSC_CN, [0x02])
        ctrl.write_regs(others, OSC_CN, [0x06])
        expected[OSC_CN] = np.array([0x05] + [0x06] * (nb_modules - 1))
    else:
        ctrl.write_regs(every, OSC_CN, [0x04])
        expected[OSC_CN] = np.full(nb_modules, 0x04)

    for reg, val in expected.items():
        got = ctrl.read_regs(every, reg, 1)[:, 0]
        bad = np.nonzero(got != val)[0]
        if len(bad):
            raise RuntimeError(f"Register 0x{reg:02X} did not take on module(s) {list(bad)}")


def ecg_volts(codes, adc_max, enable):
    """Convert ECG bytes (..., 3 channels, 3 bytes MSB first) to volts, NaN for disabled channels."""
    codes = codes.astype(np.uint32)
    value = (codes[..., 0] << 16) | (codes[..., 1] << 8) | codes[..., 2]
    volts = (value / adc_max - 0.5) * 2 * VREF / INA_GAIN
    volts[..., ~np.asarray(enable, dtype=bool)] = np.nan
    return volts


_STATUS_BITS = ["common-mode out of range", "right-leg drive near rail", "low battery",
                "lead off", "ch1 out of range", "ch2 out of range", "ch3 out of range",
                "digital sync error"]
_RANGE_BITS = {0: "INA out of range", 1: "INA (+) near (+) rail", 2: "INA (+) near (-) rail",
               3: "INA (-) near (+) rail", 4: "INA (-) near (-) rail", 6: "modulator over range"}


def describe_errors(err):
    """Text of the 7 error registers (ERROR_LOD..ERROR_MISC) of one module, '' if none."""
    out = [_STATUS_BITS[b] for b in range(8) if err[1] >> b & 1]
    for ch in range(3):
        out += [f"ch{ch + 1} {text}" for b, text in _RANGE_BITS.items() if err[2 + ch] >> b & 1]
    return ", ".join(out)
