"""Stream the wristband to LSL (one channel per module and ADC, in mV).

    python -m clvhd.lsl [--port PORT] [--modules 12] [--name EMG] [--spi HZ]
"""

import argparse

import pylsl

from .wristband import Wristband


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
    p.add_argument("--modules", type=int, default=12, help="expected number of modules")
    p.add_argument("--name", default="EMG", help="LSL stream name")
    p.add_argument("--spi", type=int, default=0, help="SPI clock (Hz)")
    args = p.parse_args(argv)

    with Wristband(args.port, expected_modules=args.modules, spi_clock=args.spi or None) as wb:
        cfg = wb.configure()
        # timestamps on the LSL clock, so they line up with other streams
        stream = wb.stream(host_clock=pylsl.local_clock)
        info = pylsl.StreamInfo(args.name, "EMG", 3 * wb.nb_modules, cfg.odr(),
                                pylsl.cf_double64, "clvhd-ads1293")
        chans = info.desc().append_child("channels")
        for label in stream.channel_labels:
            ch = chans.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "millivolts")
            ch.append_child_value("type", "EMG")
        outlet = pylsl.StreamOutlet(info)
        print(f"Streaming {3 * wb.nb_modules} channels at {cfg.odr():.2f} sps to LSL '{args.name}' (Ctrl-C to stop)")
        with stream:
            try:
                while True:
                    t_host, _, data = stream.read(0.02)
                    if len(t_host):
                        outlet.push_chunk((data * 1e3).tolist(), t_host.tolist())
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
