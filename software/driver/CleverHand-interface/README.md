# CleverHand interface

Host driver and controller firmware for the CleverHand EMG wristband: a Teensy 4.0 controller and a chain of ADS1293 EMG modules.

- `clvhd/`: pure-Python driver (needs only `numpy` and `pyserial`, Python ≥ 3.11, macOS/Linux/Windows).
- `firmware/uctrl_teensy/`: Teensy 4.0 firmware (v4.1). Flash it with the Arduino IDE and Teensyduino, then unplug the USB once: flashing restarts only the Teensy, and the modules keep their addresses until they lose power. The driver refuses older firmware.
- `docs/serial_protocol.md`: serial protocol between the host and the Teensy. `docs/ads1293.pdf`: ADS1293 datasheet.

## Install

With [uv](https://docs.astral.sh/uv/) (uses Python 3.13 from `.python-version` and the versions pinned in `uv.lock`):

```bash
uv sync --all-extras                       # creates .venv with the driver, plotting and processing
uv run examples/01_live_plot.py            # run anything inside that environment
```

Or with pip, in any Python ≥ 3.11 environment: `pip install -e ".[plot,process]"`.

## Usage

```python
import clvhd

with clvhd.Wristband(expected_modules=13) as wb:    # port: first Teensy found, or Wristband("/dev/tty.usbmodem...")
    wb.configure()                                  # defaults, or e.g. wb.configure(R3=[12, 12, 12])
    with wb.stream() as s:
        t_host, t_device, data = s.read(0.05)       # data: (N, 3 * modules) in V, NaN when missing
```

- `Wristband(...)` checks the firmware version, addresses the modules (raises unless `expected_modules` are found; power-cycle the band and retry), sets the SPI clock to 8 MHz (the Teensy's 4 MHz power-up clock drops samples from 13 modules) and turns their LEDs off.
- `configure()` writes a `clvhd.Config` to every module and reads every register back. The defaults route the three neighbouring electrode pairs of DRY_ELEC_6CH_THIN ({4,5}, {1,2}, {2,3}), run at 2083 sps (200 kHz from the 4.000 MHz crystal, R1=4, R2=4, R3=6: 420 Hz bandwidth, 3.4 µVrms) and make module 0 drive the right-leg drive on IN6, the reference net shared by all modules.
- `stream()` streams every sample of every module, except the first 8, taken while the ADS1293 decimation filters settle. Each module samples on its own crystal, so the stream fits each module's sample period and offset, detects dropped or duplicated samples, and resamples all modules onto module 0's sample times. A background thread drains the serial port, so `read()` can be called at any pace; `read(timeout)` waits `timeout` s and returns everything aligned so far. `stats()` gives each module's samples, drops, duplicates, rate and ppm offset.
- `t_host` is on `time.monotonic()`.
- `clvhd.check_contact(data)` flags each channel of about 0.5 s of raw data (in V) as `"flat"` (0 V DC and only the ADC noise: both electrodes at the same potential, usually shorted by water, sweat or solder, or both off the skin), `"poor contact"` (more than 150 mV of DC offset: one electrode barely touching) or `""`. `clvhd-check` runs it on the last second of its stream and `01_live_plot.py` shows it next to each module, with the totals on top.
- `configure(lead_off=True)` turns on the ADS1293 DC lead-off detection: 8 nA into every used input, and `read_errors()` reports `INx lead off` for an electrode off the skin, which tells it from two shorted electrodes (both read flat). Keep it off while recording: 8 nA through dry electrodes (~25 MΩ) shifts the channels by up to 200 mV. `clvhd-check --lead-off` runs it in step 3 and turns it off for the stream.
- Also: `read_errors()`, `set_led(module, led, rgb)`, `set_spi_clock(hz)`, `start_conversions()` / `stop_conversions()`. While streaming, only `set_led` works (it can cost about one sample per module); the others raise until the stream stops.

## Tools

```bash
uv run clvhd-check [--port PORT] [--modules 13] [--lead-off]   # bring-up checks: addressing, SPI at 4/8 MHz, error flags, stream timing, contact
```

`examples/`: `01_live_plot.py` (live plot, 20-450 Hz band-pass and 50 Hz notch filtered, ±1 mV per module, with electrode contact status and LED buttons), `02_record.py` (record to `.npz`), `03_process.py` (EMG envelopes from a recording).

## Tests

`tests/teensy_sim.py` simulates the Teensy and 13 modules on a pseudo-terminal (macOS/Linux): different clock rates, up to 350 us of timestamp jitter, the `micros()` wraparound, the controller's 1 s boot delay, and optional dropped, duplicated and corrupted frames.

```bash
uv run pytest
```
