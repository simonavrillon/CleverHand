# Running experiment 02: EMG and hand tracking

How to run a recording session with `examples/02_live_emg_hand.py`: the live EMG of every
module next to the camera, its hand skeleton and the finger movements, recorded together to
a BIDS dataset. What the window shows and every option: the [README](../README.md),
"Examples".

Commands run from `software/driver/CleverHand-interface/`. The participant-facing screens
are in French.

## Before the session

```bash
uv sync --all-extras        # once: the hand extra brings MediaPipe
uv run clvhd-check          # wristband plugged in: bring-up checks
```

- Place the band, then check every module's channels in the live plot (`uv run
  examples/01_live_plot.py`, or 02 itself): flat or noisy channels mean poor contact.
- Place the camera so the whole hand stays in view; the skeleton drawn on the image shows
  whether it is tracked.

## Free recording

```bash
uv run examples/02_live_emg_hand.py --subject 01 --session 01 --task fingers \
    --muscle 0=FDS --muscle 6=EDC --age 30 --sex F --handedness right
```

- Subject, session and task can also be typed in the window.
- **Record** starts and stops a recording; each recording takes the next run number.
- `--muscle MODULE=NAME` (repeat) names the muscle under a module, `--placement` describes
  how the band was placed; both go to the BIDS sidecars.

## Cued finger gestures (classifier datasets)

```bash
uv run examples/02_live_emg_hand.py --subject 01 --session 01 --task fingers \
    --protocol examples/protocols/finger_gestures.json --cue-fullscreen
```

- A second window shows the cue cards; `--cue-fullscreen` puts them full screen on the
  participant's display.
- 15 gestures (thumb pinches, finger extensions, rest), each once per block in a random order,
  10 blocks: about 16 minutes. Each trial: the card in grey with a countdown (prepare, 3 s),
  lit (go: do the gesture once, 1.5 s), then a relaxed hand (rest, 2 s).
- **Space** starts the protocol and records the EMG, the camera and the cue events in one run;
  **Space** also ends each break between blocks. **Escape** stops, keeping what was recorded.
- `--seed` fixes the trial order; the protocol file sets the gestures, repetitions and timings.

## After the session

Find each cued movement's real onset and offset from the camera:

```bash
uv run clvhd-movements data/sub-01/ses-01/emg/sub-01_ses-01_task-fingers_run-01_events.tsv
```

It writes `derivatives/movements/`: the windows to train classifiers on (onset, offset,
reaction time, or no movement / no hand).

## Output

In `data/` (or `--bids-root`), per run: `emg/` (EMG, `--format edf` or `bdf`) with
`_events.tsv` for the cued protocol, and `motion/` (the hand landmarks, each finger's flexion,
the thumb–index pinch aperture). The stream is written to `sourcedata/` while recording and
deleted once the BIDS files are written, unless `--keep-sourcedata`.
