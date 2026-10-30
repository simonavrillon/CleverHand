# CleverHand interface

Host driver and controller firmware for the CleverHand EMG wristband: a Teensy 4.0 controller and a chain of ADS1293 EMG modules.

- `clvhd/`: pure-Python driver (needs only `numpy` and `pyserial`, Python ≥ 3.11, macOS/Linux/Windows).
- `firmware/uctrl_teensy/`: Teensy 4.0 firmware (v4.1). Flash it with the Arduino IDE and Teensyduino; the driver refuses older firmware.
- `docs/serial_protocol.md`: serial protocol between the host and the Teensy. `docs/ads1293.pdf`: ADS1293 datasheet.

## Install

With [uv](https://docs.astral.sh/uv/) (uses Python 3.13 from `.python-version` and the versions pinned in `uv.lock`):

```bash
uv sync --all-extras                       # creates .venv with the driver, LSL, plotting and processing
uv run examples/02_live_plot.py            # run anything inside that environment
```

Or with pip, in any Python ≥ 3.11 environment: `pip install -e ".[lsl,plot,process]"`.

## Usage

```python
import clvhd

with clvhd.Wristband(expected_modules=12) as wb:    # port: first Teensy found, or Wristband("/dev/tty.usbmodem...")
    wb.configure()                                  # defaults, or e.g. wb.configure(R3=[12, 12, 12])
    with wb.stream() as s:
        t_host, t_device, data = s.read(0.05)       # data: (N, 3 * modules) in V, NaN when missing
```

- `Wristband(...)` checks the firmware version, addresses the modules (raises unless `expected_modules` are found; power-cycle the band and retry) and turns their LEDs off.
- `configure()` writes a `clvhd.Config` to every module and reads every register back. The defaults route the three neighbouring electrode pairs of DRY_ELEC_6CH_THIN ({4,5}, {1,2}, {2,3}), run at 2083 sps (200 kHz from the 4.000 MHz crystal, R1=4, R2=4, R3=6: 420 Hz bandwidth, 3.4 µVrms) and make module 0 drive the right-leg drive on IN6, the reference net shared by all modules.
- `stream()` streams every sample of every module. Each module samples on its own crystal, so the stream fits each module's sample period and offset, detects dropped or duplicated samples, and resamples all modules onto module 0's sample times. A background thread drains the serial port, so `read()` can be called at any pace; `read(timeout)` waits `timeout` s and returns everything aligned so far. `stats()` gives each module's samples, drops, duplicates, rate and ppm offset.
- `t_host` is on `time.monotonic()`, or on the clock passed as `wb.stream(host_clock=...)` (e.g. `pylsl.local_clock`).
- Also: `read_errors()`, `set_led(module, led, rgb)`, `set_spi_clock(hz)`, `start_conversions()` / `stop_conversions()`. While streaming, only `set_led` works (it can cost about one sample per module); the others raise until the stream stops.

## Tools

```bash
uv run clvhd-check [--port PORT] [--modules 12]   # bring-up checks: addressing, SPI at 4/8 MHz, error flags, stream timing
uv run clvhd-lsl [--port PORT] [--name EMG]       # stream to LSL (mV, LSL timestamps)
```

`examples/`: `01_stream.py` (connect and stream), `02_live_plot.py` (live plot with channel toggles, channel offset, band-pass and LED buttons), `03_record.py` (record to `.npz`), `04_process.py` (EMG envelopes from a recording).

## Tests

`tests/teensy_sim.py` simulates the Teensy and 12 modules on a pseudo-terminal (macOS/Linux): different clock rates, up to 350 us of timestamp jitter, the `micros()` wraparound, the controller's 1 s boot delay, and optional dropped, duplicated and corrupted frames.

```bash
uv run pytest
```
