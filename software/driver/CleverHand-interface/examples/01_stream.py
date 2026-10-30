"""Connect to the wristband, stream for a few seconds and print the stream statistics.

    python 01_stream.py [--port PORT] [--modules 12]
"""
import argparse
import time

import numpy as np

import clvhd

p = argparse.ArgumentParser()
p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
p.add_argument("--modules", type=int, default=12, help="expected number of modules")
args = p.parse_args()

with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    # Each module has 3 ADCs, each routed to a (+, -) pair of the 6 inputs.
    # The defaults use the three neighbouring electrode pairs of
    # DRY_ELEC_6CH_THIN, 2083 sps (R1=4, R2=4, R3=6 at 200 kHz) and the
    # right-leg drive of module 0 on IN6 (the shared reference net).
    # Any field of clvhd.Config can be overridden, e.g. wb.configure(R3=[12, 12, 12]).
    cfg = wb.configure()
    print(f"{wb.nb_modules} modules, {cfg.odr():.2f} sps")

    with wb.stream() as s:
        t_end = time.monotonic() + 4
        while time.monotonic() < t_end:
            t_host, t_device, data = s.read(0.25)  # data: (N, 3 * modules) in V, NaN when missing
            if len(t_device):
                print(f"t = {t_device[-1]:.3f} s, {len(t_device)} samples, "
                      f"RMS of ch1 per module (uV): {np.round(1e6 * np.nanstd(data[:, 0::3], axis=0), 1)}")
        for m, st in enumerate(s.stats()):
            print(f"module {m}: {st['samples']} samples, {st['dropped']} dropped, "
                  f"{st['rate_hz']:.3f} sps ({st['ppm']:+.1f} ppm)")
    print("Errors:", wb.read_errors())
