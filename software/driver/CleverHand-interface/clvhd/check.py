"""Bring-up checks: controller link, addressing, SPI reliability, error flags, streaming timing.

    python -m clvhd.check [--port PORT] [--modules 12] [--seconds 10] [--spi HZ]
"""

import argparse
import sys
import time

import numpy as np

from . import ads1293
from .wristband import Wristband


class Report:
    def __init__(self):
        self.failures = 0

    def __call__(self, ok, text):
        print(("  [ OK ] " if ok else "  [FAIL] ") + text)
        self.failures += not ok


def spi_errors(wb, hz, n=2000):
    """Wrong REVID reads per module at the given SPI clock."""
    wb.set_spi_clock(hz)
    errors = np.zeros(wb.nb_modules, dtype=int)
    for _ in range(n):
        try:
            errors += wb.ctrl.read_regs(wb.mask, ads1293.REVID, 1)[:, 0] != 0x01
        except Exception:
            errors += 1
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
    p.add_argument("--modules", type=int, default=12, help="expected number of modules")
    p.add_argument("--seconds", type=float, default=10, help="streaming test duration")
    p.add_argument("--spi", type=int, default=0, help="SPI clock to test (default: 4 and 8 MHz, keep the best)")
    args = p.parse_args(argv)
    check = Report()

    print("1. Controller and addressing")
    wb = Wristband(args.port, expected_modules=0)
    check(wb.ctrl.mirror(), f"firmware v{wb.firmware[0]}.{wb.firmware[1]}, mirror command echoes its bytes")
    check(wb.nb_modules == args.modules,
          f"{wb.nb_modules} modules found (expected {args.modules}), all answer as ADS1293")

    print("2. SPI reliability (REVID reads)")
    best = 0
    for hz in ([args.spi] if args.spi else [4_000_000, 8_000_000]):
        err = spi_errors(wb, hz)
        detail = " ".join(f"M{m}:{e}" for m, e in enumerate(err) if e)
        check(err.sum() == 0, f"{hz / 1e6:g} MHz: {err.sum()} bad reads {detail}")
        if err.sum() == 0:
            best = hz
    if best == 0:
        print("  No clean SPI clock: try a lower one, or add a 33 ohm resistor on SCLK at Teensy pin 13")
        best = 4_000_000
    wb.set_spi_clock(best)
    print(f"  using {best / 1e6:g} MHz")

    print("3. ADS1293 error flags (wear the band with the bias electrode for this one)")
    cfg = wb.configure()
    wb.start_conversions()
    time.sleep(1.0)
    wb.read_errors()  # the flags latch: clear the start-up ones
    time.sleep(0.5)
    for m, err in enumerate(wb.read_errors()):
        check(err == "", f"module {m}: {err or 'no error'}")
    wb.stop_conversions()

    rate = cfg.odr()
    print(f"4. Streaming for {args.seconds:g} s at {rate:.2f} sps")
    frames = nan_frames = 0
    with wb.stream() as s:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            _, t, data = s.read(0.02)
            frames += len(t)
            nan_frames += int(np.isnan(data).any(axis=1).sum())
        stats, crc = s.stats(), s.crc_errors
    wb.close()

    check(crc == 0, f"{crc} frames with a bad CRC")
    expected = args.seconds * rate
    print("  module   samples  dropped  dup      rate (sps)   ppm")
    for m, st in enumerate(stats):
        print(f"  {m:6d}{st['samples']:10d}{st['dropped']:9d}{st['duplicates']:5d}"
              f"{st['rate_hz']:15.3f}{st['ppm']:8.1f}")
    for m, st in enumerate(stats):
        check(abs(st["samples"] - expected) < 0.02 * expected,
              f"module {m}: {st['samples']} samples (expected ~{expected:.0f})")
        check(st["dropped"] == 0, f"module {m}: {st['dropped']} dropped (frames with a bad CRC "
                                  "count as dropped; otherwise raise the SPI clock)")
        # duplicates mean a module's channels updated further apart than the
        # firmware's settle time (STREAM_SETTLE_US), so one sample was sent twice
        check(st["duplicates"] < st["samples"] / 100, f"module {m}: {st['duplicates']} duplicate frames")
        check(abs(st["rate_hz"] / rate - 1) < 0.005, f"module {m}: rate within 0.5% of nominal")
    spread = max(st["ppm"] for st in stats) - min(st["ppm"] for st in stats)
    slip = f"one sample slip every {1e6 / (spread * rate):.1f} s" if spread > 0 else "no slip"
    print(f"  module clocks spread over {spread:.1f} ppm: about {slip} without the alignment")
    check(frames > 0.95 * expected, f"{frames} aligned samples, {nan_frames} with a missing value")

    print("All checks passed" if check.failures == 0 else f"{check.failures} check(s) failed")
    return 1 if check.failures else 0


if __name__ == "__main__":
    sys.exit(main())
