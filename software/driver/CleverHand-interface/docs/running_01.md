# Running experiment 01: live EMG plot and recording

How to run `examples/01_live_plot.py`: the live plot of every module's channels, band-pass
filtered, with the electrode contact status and LED buttons, and a **Record** button that saves
the EMG to a BIDS dataset. What the window shows and every option: the [README](../README.md),
"Examples".

Commands run from `software/driver/CleverHand-interface/`, the wristband plugged in.

## Before the session

```bash
uv sync --all-extras     # once (the plot extra brings pyqtgraph)
uv run clvhd-check       # bring-up checks
```

- Place the band. The plot shows every channel next to its module (20-450 Hz band-pass, notches
  at the mains frequency and its harmonics, ±1 mV per module), with each channel's contact
  status and the totals on top: flat or poor-contact channels mean the electrodes need
  re-placing.
- The LED buttons set each module's two LEDs, e.g. to mark a module during the session.

## Recording

```bash
uv run examples/01_live_plot.py --subject 01 --session 01 --task rest \
    --muscle 0=ECR --age 30 --sex F --handedness right
```

- Subject, session and task can also be typed in the window.
- **Record** starts and stops a recording; each recording takes the next run number.
- `--muscle MODULE=NAME` (repeat) names the muscle under a module, `--placement` describes how
  the band was placed, `--age` / `--sex` / `--handedness` the participant; all go to the BIDS
  sidecars (`participants.tsv`, the EMG sidecar).

## Options

| Option | What it does |
|---|---|
| `--window 5` | time window shown (s) |
| `--duration 0` | close after this many seconds (0: never) |
| `--powerline 50` | mains frequency notched, and its harmonics below 450 Hz (0: none) |
| `--port PORT`, `--modules 13` | the wristband's serial port and module count |
| `--bids-root` | where the dataset goes (default `data/`) |
| `--format edf` | EMG file format: `edf` (16 bits, default) or `bdf` (24 bits) |
| `--keep-sourcedata` | keep the stream in `sourcedata/` once the BIDS files are written |

## Output

In `data/` (or `--bids-root`), per run: `emg/` with the EMG file (`_emg.edf` or `.bdf`), its
sidecar `_emg.json`, `_channels.tsv/.json` (per-channel status, interpolated samples,
resolution) and `_electrodes.tsv`; `participants.tsv` and the session's `_scans.tsv`. The stream
is written to `sourcedata/` while recording and deleted once the BIDS files are written, unless
`--keep-sourcedata`. No `motion/` or `beh/`: those come from 02 and 03/04.

See also: [running_02.md](running_02.md) (EMG + hand tracking from a camera),
[running_03_04.md](running_03_04.md) (cursor control).
