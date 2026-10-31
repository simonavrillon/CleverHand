# Running experiments 03 and 04: cursor control

How to run a cursor session: the center-out calibration and test of
`examples/03_cursor_centerout.py`, then the tracking experiment of
`examples/04_cursor_coadapt.py`, which uses the decoder 03 fits. What the tasks do and every
option: the [README](../README.md), "Examples"; why they are designed this way:
[experiments.md](experiments.md).

Commands run from `software/driver/CleverHand-interface/`, the wristband plugged in. The
participant-facing screens are in French.

## Before the session

```bash
uv sync --all-extras        # once
uv run clvhd-check          # bring-up checks
```

Place the band and check the channels in the live plot (`uv run examples/01_live_plot.py`).
03 also checks the contact itself, below.

## 1. Calibration and test (03)

```bash
uv run examples/03_cursor_centerout.py --subject 01 --session 01
```

1. **Contact check**: 3 s at rest. Flat and poor-contact channels are left out of the decoder.
2. **Calibration**: the 8 targets lit one at a time (`--calib-reps`, default 3 each), a guide
   cursor travelling to each in 2 s. The participant contracts as if pushing the cursor there,
   then relaxes for 2 s when it jumps back to the centre.
3. The decoder is fitted and its cross-validated R² shown. **Space** starts the test, **C**
   calibrates again.
4. **Test**: `--reps` (default 3) shuffled blocks of the 8 targets; each reach holds the cursor
   1 s at the centre, then gives 10 s to hold it 0.5 s in the target.

**Escape** quits; the phase in progress is not saved. To skip the calibration with a decoder
fitted earlier: `--decoder <its _desc-decoder.npz>`.

The decoder is saved as
`data/derivatives/centerout/sub-01/ses-01/emg/sub-01_ses-01_task-centerout_acq-calib_run-01_desc-decoder.npz`.
After a recalibration (C), use the last run's file:

```bash
ls data/derivatives/centerout/sub-01/ses-01/emg/*_desc-decoder.npz
```

## 2. Tracking (04)

04 needs a decoder from a 03 calibration of the same session (`--decoder`, required). A
session's plan is made for one protocol: run each protocol in its own `--session`, with its
own 03 calibration.

**Coadapt protocol** (Madduri et al. 2026): 6 trials of 5 minutes, the decoder adapting every
20 s; the 3 learning rates (fixed, slow, fast) once per block, shuffled, each trial ending with
20 s on the calibration's decoder, unannounced (the probe); a 5-minute break halfway
(`--break-seconds`).

```bash
uv run examples/04_cursor_coadapt.py --subject 01 --session 01 \
    --decoder data/derivatives/centerout/sub-01/ses-01/emg/sub-01_ses-01_task-centerout_acq-calib_run-01_desc-decoder.npz
```

**Gain protocol**: 9 trials of 2 minutes, the calibration's decoder at a gain a staircase moves
every 20 s. Two references at gain 1, three pairs of a low- and a high-start trial (shuffled),
a last reference; no probe, no break between blocks.

```bash
uv run examples/03_cursor_centerout.py --subject 01 --session 02
uv run examples/04_cursor_coadapt.py --subject 01 --session 02 --protocol gain \
    --decoder data/derivatives/centerout/sub-01/ses-02/emg/sub-01_ses-02_task-centerout_acq-calib_run-01_desc-decoder.npz
```

During either protocol:

- **Space** starts each trial; the participant rests between trials as long as needed.
- **Escape** quits. The trial in progress is run again next time; what it recorded goes to
  `derivatives/coadapt/…_desc-interrupted….npz`.
- **Resuming**: the same command continues at the first unsaved trial (in the gain protocol,
  each staircase from its saved state).
- The console prints every 20 s the window's tracking error, effort, decoder gain and
  direction, remnant and null-space share; in the gain protocol, each new gain.

## Useful options (03 and 04)

| Option | What it does |
|---|---|
| `--windowed` | a window instead of full screen |
| `--screen-width-cm 52.7` | the screen's visible width, to draw in real cm |
| `--field-cm 46.5 24.5` | the task field (default: an 11-inch iPad, 22.7 × 15.8 cm; this: the paper's display) |
| `--port PORT`, `--modules 13` | the wristband's serial port and module count |
| `--bids-root` | where the dataset goes (default `data/`) |
| 03: `--speed-power 1.5`, `--deadzone` | the test's speed curve and dead-zone |
| 04: `--trial-seconds`, `--trials N` | trial length; stop after N trials |
| 04: `--autostart` | start each trial without waiting for Space |

## Output

In `data/` (or `--bids-root`):

- 03, per phase (`acq-calib`, `acq-online`): `emg/` with `_events.tsv` (each reach's outcome,
  time to target, acquisition time, path efficiency), `beh/`, and in `derivatives/centerout/`
  the decoder and the envelopes.
- 04, per trial: `emg/` with `_events.tsv`, `beh/`, and in `derivatives/coadapt/` the
  envelopes, every decoder, each window's measures and the trajectory; in the gain protocol
  also the gains, the reference error and the staircase's state. The session plan is
  `derivatives/coadapt/sub-01/ses-01/sub-01_ses-01_task-cursor_desc-plan.json`.
