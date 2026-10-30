"""Record the wristband to a .npz file.

    python 03_record.py [seconds] [--out recording.npz] [--port PORT] [--modules 12]

The file holds t_host and t_device (s, one per sample), data ((N, 3 * modules)
in V, NaN where a sample is missing), fs, channel labels and the configuration.
"""
import argparse
import dataclasses
import json
import time

import numpy as np

import clvhd

p = argparse.ArgumentParser()
p.add_argument("seconds", type=float, nargs="?", default=10, help="duration of the recording")
p.add_argument("--out", default="recording.npz", help="output file")
p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
p.add_argument("--modules", type=int, default=12, help="expected number of modules")
args = p.parse_args()

with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    cfg = wb.configure()
    chunks = []
    with wb.stream() as s:
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            chunks.append(s.read(0.1))
            print(f"\r{time.monotonic() - t0:5.1f} / {args.seconds:g} s", end="", flush=True)
        stats, labels = s.stats(), s.channel_labels
    errors = wb.read_errors()
print()

t_host, t_device, data = (np.concatenate(c) for c in zip(*chunks))
np.savez_compressed(args.out, t_host=t_host, t_device=t_device, data=data, fs=cfg.odr(),
                    labels=labels, config=json.dumps(dataclasses.asdict(cfg)),
                    stats=json.dumps(stats), errors=json.dumps(errors))
print(f"{len(t_device)} samples x {data.shape[1]} channels written to {args.out}")
for m, st in enumerate(stats):
    print(f"  module {m}: {st['samples']} samples, {st['dropped']} dropped, {st['ppm']:+.1f} ppm"
          + (f", errors: {errors[m]}" if errors[m] else ""))
