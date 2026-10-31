# CleverHand driver (`clvhd`) and Teensy firmware

Host driver and controller firmware for the **CleverHand EMG wristband**: a **Teensy 4.0**
controller chained to **13 EMG modules** (each a COM_MOD + ADS1293 board with 6 dry
electrodes, the DRY_ELEC_6CH_THIN), worn around the forearm.

| Folder | Content |
|---|---|
| `clvhd/` | the pure-Python host driver (needs only `numpy` and `pyserial`, Python ≥ 3.11, macOS/Linux/Windows) |
| `firmware/uctrl_teensy/` | the Teensy controller firmware (serial protocol v4.1) |
| `examples/` | live-plot, hand-tracking and cursor-control windows |
| `tests/` | driver tests against a simulated Teensy on a pseudo-terminal |
| `docs/` | `serial_protocol.md` (host ↔ Teensy protocol), `experiments.md` (03/04 design notes), `ads1293.pdf` (the ADS1293 datasheet) |

## Contents

- [Quick start](#quick-start)
- [Install](#install)
- [Using the driver](#using-the-driver)
- [Configuration](#configuration)
- [Streaming and alignment](#streaming-and-alignment)
- [Electrode contact check](#electrode-contact-check)
- [Command-line tools](#command-line-tools)
- [Examples](#examples)
- [Recording to a BIDS dataset](#recording-to-a-bids-dataset)
- [Camera hand tracking](#camera-hand-tracking)
- [Tests](#tests)

## Quick start

From this folder, with the wristband plugged in:

```bash
uv sync --all-extras              # create .venv (Python 3.13, pinned versions): driver + plotting + hand tracking + BIDS export
uv run clvhd-check                # bring-up check: firmware v4.1, 13 modules, SPI, streaming, electrode contact
uv run examples/01_live_plot.py   # live plot of every module; the Record button saves to a BIDS dataset
```

First time on a new Teensy? **Flash the firmware first** — [docs/flashing.md](../../../docs/flashing.md),
part 2 (Arduino IDE + Teensyduino over USB). The driver refuses firmware older than v4.1.
After flashing, power-cycle the whole wristband once: the modules take their address only at
power-up, so the Teensy alone rebooting would find 0 modules.

## Install

With [uv](https://docs.astral.sh/uv/) (uses Python 3.13 from `.python-version` and the versions pinned in `uv.lock`):

```bash
uv sync --all-extras
uv run examples/01_live_plot.py   # run anything inside that environment
```

Or with pip, in any Python ≥ 3.11 environment: `pip install -e ".[plot,hand,bids]"`.

Extras: `plot` (the example windows), `hand` (MediaPipe hand tracking from a camera; downloads its
model to `~/.cache/clvhd` on first use), `bids` (BIDS export, EDF or BDF via pyedflib).

## Using the driver

```python
import clvhd

with clvhd.Wristband(expected_modules=13) as wb:    # port: first Teensy found, or Wristband("/dev/tty.usbmodem...")
    wb.configure()                                  # defaults, or e.g. wb.configure(R3=[12, 12, 12])
    with wb.stream() as s:
        t_host, t_device, data = s.read(0.05)       # data: (N, 3 * modules) in V, NaN when missing
```

- `Wristband(...)` checks the firmware version, addresses the modules (raises unless `expected_modules`
  are found; power-cycle the band and retry), sets the SPI clock to 8 MHz (the Teensy's 4 MHz power-up
  clock drops samples from 13 modules) and turns their LEDs off.
- `t_host` is on `time.monotonic()`; `t_device` on the Teensy's clock.
- Also: `read_errors()`, `set_led(module, led, rgb)`, `set_spi_clock(hz)`, `start_conversions()` /
  `stop_conversions()`. While streaming, only `set_led` works (it can cost about one sample per
  module); the others raise until the stream stops.

## Configuration

`configure()` writes a `clvhd.Config` to every module and reads every register back. The defaults
suit the DRY_ELEC_6CH_THIN board:

- the three channels use the three **neighbouring electrode pairs** of the strip
  ({4,5}, {1,2}, {2,3} — the physical order on the board is E4 E5 E6 E1 E2 E3);
- **2083 samples/s** per channel (200 kHz modulator from the 4.000 MHz crystal, R1=4, R2=4, R3=6:
  420 Hz bandwidth, 3.4 µVrms noise);
- module 0 drives the **right-leg drive** on IN6, the reference net shared by all modules.

`configure(lead_off=True)` turns on the ADS1293 DC lead-off detection: 8 nA into every used input,
and `read_errors()` reports `INx lead off` for an electrode off the skin, which tells it from two
shorted electrodes (both read flat). Keep it off while recording: 8 nA through dry electrodes
(~25 MΩ) shifts the channels by up to 200 mV. `clvhd-check --lead-off` runs it in step 3 and turns
it off for the stream.

## Streaming and alignment

`stream()` streams every sample of every module, except the first 8, taken while the ADS1293
decimation filters settle. Each module samples on its own crystal, so the stream fits each module's
sample period and offset, detects dropped or duplicated samples, and resamples all modules onto
module 0's sample times. A background thread drains the serial port, so `read()` can be called at
any pace; `read(timeout)` waits `timeout` s and returns everything aligned so far. `stats()` gives
each module's samples, drops, duplicates, rate and ppm offset.

## Electrode contact check

`clvhd.check_contact(data)` flags each channel of about 0.5 s of raw data (in V) as `"flat"`
(0 V DC and only the ADC noise: both electrodes at the same potential, usually shorted by water,
sweat or solder, or both off the skin), `"poor contact"` (more than 150 mV of DC offset: one
electrode barely touching) or `""`. `clvhd-check` runs it on the last second of its stream and
`01_live_plot.py` shows it next to each module, with the totals on top.

## Command-line tools

```bash
uv run clvhd-check [--port PORT] [--modules 13] [--lead-off]   # bring-up checks: addressing, SPI at 4/8 MHz, error flags, stream timing, contact
uv run clvhd-export data/sourcedata/sub-01/*_clvhd.json        # write the BIDS files of recordings left in sourcedata/
uv run clvhd-movements data/sub-01/emg/sub-01_task-gestures_run-01_events.tsv  # each cued gesture's movement, from the camera
```

## Examples

The windows share one look, from `clvhd/ui.py`:

- `01_live_plot.py`: live plot, 20-450 Hz band-pass filtered with notches at 50 Hz and its
  harmonics, ±1 mV per module, with electrode contact status and LED buttons, and a **Record**
  button that saves the EMG to a BIDS dataset.
- `02_live_emg_hand.py`: the live plot of `01_live_plot.py` next to the camera image, its MediaPipe
  hand skeleton (each finger in its trace's colour), and the finger movements on one time axis: each
  finger's flexion (the sum of its joint angles, near 0 flat, about 250° in a fist) and the
  thumb–index pinch aperture (cm). Its **Record** button saves the EMG and the hand kinematics; the
  motion file holds the landmarks, each finger's flexion and the pinch aperture. With
  `--protocol examples/protocols/finger_gestures.json` (`clvhd.cues`), a second window
  (`--cue-fullscreen` for the participant's screen) cues finger gestures, each done once from the
  position shown on the card (open hand for a pinch, closed hand for an extension): thumb pinches
  with each finger, with index and middle, and with all fingers; extensions of each finger, of
  thumb and index, of thumb, index and middle, and of all fingers; and rest (15 classes), each once
  per block in a random order (10 blocks, about 16 min). The cards show the hand
  (`clvhd/assets/hand.svg`) with the moving fingertips in white: a dashed line from the thumb for
  pinches, an arrow along the finger for extensions. Each trial shows the gesture's card in grey
  with a countdown (prepare, 3 s: 3, 2, 1), lit (go: do it once, 1.5 s), then a relaxed hand (rest,
  2 s). Space starts it and records the EMG, the camera and the cue events in one run
  (`_events.tsv`: cue, go, rest and break, with the gesture, its fingers and motion, block and
  trial, stamped when the card is drawn); Space also ends the breaks between blocks, and Escape
  stops it, keeping what was recorded. The protocol
  file sets the gestures (or new ones), repetitions and timings. `clvhd-movements <the run's
  _events.tsv>` then finds each go's real movement from the camera's fingertips (onset, offset,
  reaction time, or no movement / no hand), in `derivatives/movements/`: the windows to train
  classifiers on.
- `03_cursor_centerout.py`: center-out myoelectric cursor control with a fixed Wiener filter. A cued
  calibration (each of the 8 targets lit `--calib-reps` times while a guide cursor travels to it in
  2 s; the participant contracts towards it, then relaxes for 2 s at the centre) fits the decoder
  (`clvhd.decoder`, shared with 04): the log of the 60 Hz EMG envelope (20-450 Hz band) over the
  last 100 ms, each channel z-scored, plus a bias, with a ridge penalty picked by cross-validation
  over the cues (its cross-validated R² is shown; C calibrates again). Its output goes through a
  gain per target direction (each direction reaches the calibration's speed, undoing the ridge's
  shrinkage), a dead-zone (the decoded speed at rest, or `--deadzone`), and with `--speed-power`
  (try 1.5) a speed curve: the calibrated contraction keeps its speed, stronger ones go faster than
  proportionally, weaker ones slower. Calibrating with light contractions leaves room to
  accelerate. The test then runs `--reps` shuffled blocks of the 8 targets: 1 s at the centre with
  the target shown, then 10 s to hold the cursor 0.5 s in it. Calibration and test are runs of
  `acq-calib` and `acq-online` in the BIDS dataset (`emg/` with `_events.tsv` holding each reach's
  outcome, time to target, acquisition time and path efficiency; `beh/`), with the decoder and the
  envelopes in `derivatives/centerout/`; `--decoder` reuses a saved decoder and skips the
  calibration. A contact check at the start leaves flat and poor-contact channels out of the
  decoder. Escape quits; the phase in progress is not saved. Same task field and `--field-cm` as
  04, in the examples' dark palette: the targets take the module colours of the live plot in turn
  (but green, which marks an acquired target), the cursor is white.
- `04_cursor_coadapt.py`: co-adaptive myoelectric cursor tracking, after Madduri et al. (Nat. Mach.
  Intell. 2026): a linear decoder of the EMG, refit every 20 s on the intended velocity with an
  effort penalty λ and blended with the previous one by SmoothBatch; 6 five-minute trials, the
  decoder fixed (the control), slow or fast learning rate, once per block, each ending with 20 s on
  the calibration's decoder, unannounced (a probe of who adapted: the participant or the decoder).
  Every 20 s the console prints the window's tracking error, effort (envelope above rest, 1 = as in
  the calibration) and the decoder's gain and direction against the calibration's; they are saved
  with each trial, and an interrupted trial's data go to `derivatives/coadapt/` as
  `_desc-interrupted…npz`. Each trial goes to the BIDS dataset (`--bids-root`, default `data/`):
  `emg/` with `_events.tsv`, `beh/` and `derivatives/coadapt/` (envelopes and every decoder); a
  rerun resumes at the first unsaved trial. Not in the paper: the decoder is 03's
  (`clvhd.decoder.WienerFilter`, from a `03_cursor_centerout.py` calibration at the start of the
  same session, `--decoder <its _desc-decoder.npz>`, required), without its output stage and with
  its constant term kept (only the EMG weights are refitted), and every trial starts from its fit
  instead of the paper's random D1 / D2, so the cursor follows the participant from the first
  second; and λ is relative to the features' energy (low 1, high 10). As in the authors'
  published analysis code (the paper's equation reads per frame), the intended velocity is
  (target − cursor) / 0.5 s, so the updates refine the decoder at the task's speeds. Smaller
  differences (decoder history, edge reset, frame rate): [docs/experiments.md](docs/experiments.md).
  The EMG files keep every channel. The task runs in
  a 22.7 × 15.8 cm area (an 11-inch iPad), centred on the screen; `--field-cm 46.5 24.5` gives the
  paper's display. Same palette as 03 (`clvhd.ui`): dark background, the task field outlined, a
  white cursor, the target in a module colour that changes each trial (the paper's were red and blue
  on white).

  `--protocol gain` (in its own `--session`): the calibration's decoder, never refitted, its EMG
  weights times a gain that a staircase moves every 20 s towards the largest gain whose tracking
  error stays within 10 % of the reference trials' (where the participant needs the least
  contraction without losing accuracy). 9 two-minute trials (`--trial-seconds`, default 120): two
  references at gain 1, then three pairs of a staircase starting low (0.5) and one starting high
  (2), each pair shuffled, each start's staircase carried over between its trials, then a last
  reference; no probe, and no break between blocks (Space after each trial, as between the
  others). The console also prints each gain; the files get a `gain` column in `beh/`,
  `gain_update` events, and in the derivative the gains, the reference error and the staircase's
  state (a rerun resumes from it; `optimal_decoders` stays empty). Every window also saves the remnant (cursor velocity not at the
  target's frequencies) and the null-space share (activity the decoder maps to no motion). Design:
  [docs/experiments.md](docs/experiments.md).

## Recording to a BIDS dataset

Recording (01 and 02): type the subject, session (optional) and task in the window, or pass
`--subject`, `--session` and `--task`. Each recording takes the next free run. It goes to
`--bids-root` (default `data/` in this folder, git-ignored) in the layout and conventions of
MUedit2's BIDS export (BIDS = a standard folder layout for neuroimaging datasets, so recordings are
easy to share and analyse):

- `emg/`: EDF in µV (`--format bdf` for 24 bits; EDF and BDF are standard biosignal file formats),
  sidecars, bipolar channels with their electrodes, module-local electrode coordinates; channels the
  contact check flags on the first second are marked bad.
- `motion/` (02 only, `tracksys-mediapipe`): one row per processed camera frame, with its time from
  the first EMG sample in a LATENCY channel.
- `_scans.tsv`: the shared `acq_time`.
- `sourcedata/`: the stream as it arrived, written during the recording. Every sample (host and
  device times, the channels in float32 V, NaN where missing) and every camera frame are appended
  to `<label>_clvhd-emg.dat` and `<label>_clvhd-hand.dat`, and pushed to disk every 200 ms.
  `<label>_clvhd.json` describes them. Memory use stays flat however long the recording: about
  1.3 GB of disk per hour with 13 modules.

On Stop, the BIDS files are built from `sourcedata/`, about 50 s of signal at a time, in the
background, so the plots keep going. Once they read back complete (the EDF holds every sample of
every channel, the motion file every frame), the recording is deleted from `sourcedata/`;
`--keep-sourcedata` keeps it. If the export fails, the recording stays. Closing the window saves a
recording still running. After a crash, the recording is still in `sourcedata/` up to its last
200 ms: `uv run clvhd-export data/sourcedata/sub-01/<label>_clvhd.json` writes its BIDS files
(`--keep` to keep the source). EDF is MUedit2's default and a third smaller than BDF. Its 16 bits
span each channel's range in the recording, and `_channels.tsv` gives the resulting step
(`resolution`). Steady contact gives steps under 1 µV, whose rounding noise is lost under the
ADC's 3.4 µV. A channel that hits the ADC's limit (an electrode lifting) gets steps of about
21 µV for the whole file; the record bar then names the coarse channels, so you can record again
with `--format bdf`. Other options (`--muscle 0=ECR`, `--placement`, `--age`, `--sex`,
`--handedness`, `--acq`, `--task-description`) fill the sidecars and `participants.tsv`. The
ADS1293 error flags cannot be read while streaming, so the recordings leave them out.

## Camera hand tracking

The camera and the EMG share one clock: `clvhd.hand.HandTracker` reads the camera on one thread and
stamps each frame with `time.monotonic()`, like the stream's `t_host`. A second thread runs
MediaPipe on the latest frame, skipping frames when it is slower than the camera, so results never
lag. OpenCV and MediaPipe release the GIL, so these threads do not slow the serial reader. Frame
times are when the frame reached the host, about one frame period after its exposure. MediaPipe 1.x
sends usage logs to Google (`play.googleapis.com/log`); block that host if your ethics approval
requires it.

## Tests

One file per `clvhd` module (`test_<module>.py`, as in MUedit2); shared fixtures live in
`tests/conftest.py` and helpers in underscore-prefixed modules (`_teensy_sim.py`,
`_synthetic.py`). `tests/_teensy_sim.py` simulates the Teensy and 13 modules on a pseudo-terminal
(macOS/Linux): different clock rates, up to 350 us of timestamp jitter, the `micros()` wraparound,
the controller's 1 s boot delay, and optional dropped, duplicated and corrupted frames.

```bash
uv run pytest
```

## Lint and type-check

The package, the examples and the tests are linted and type-checked like in MUedit2
(config in `pyproject.toml`; ruff and mypy are in the `dev` dependency group):

```bash
uv run ruff check . && uv run ruff format .   # lint and format (import order, pyupgrade, bug finders, ...)
uv run mypy                                    # types over clvhd/, examples/ and tests/
```

Working on the code? Read [AGENTS.md](AGENTS.md) first (conventions, rules, and what to run
before finishing). Experiment design notes (03, 04): [docs/experiments.md](docs/experiments.md).
