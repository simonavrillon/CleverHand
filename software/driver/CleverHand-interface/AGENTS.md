# Driver conventions (`clvhd` + firmware + examples + tests)

Rules for `software/driver/CleverHand-interface/`. Ruff and mypy (config in
`pyproject.toml`) already enforce formatting, imports and the bug-finder rules — run
`uv run ruff check . && uv run ruff format .` and `uv run mypy` before finishing; this file
covers what they can't. User docs: [README.md](README.md). Wire protocol:
[docs/serial_protocol.md](docs/serial_protocol.md). Experiment rationale:
[docs/experiments.md](docs/experiments.md).

## Layout

```text
clvhd/                  the driver package
  protocol.py           serial protocol: Controller (request/reply), find_port, frame constants
  ads1293.py            ADS1293 registers, Config, conversion to volts, error flags
  wristband.py          Wristband: connect, address, configure, stream (the public API, with __init__.py)
  stream.py             acquisition: frame parsing, per-module clock fit, alignment (Stream)
  filters.py            causal EMG band-pass (scipy, imported lazily)
  contact.py            electrode contact check from raw samples
  bids.py               BIDS export: write_emg/motion/events/beh, RawRecording, export_recording, clvhd-export
  hand.py               MediaPipe hand landmarks on background threads (clvhd.hand.HandTracker)
  cues.py               cued finger gestures (Protocol, cards, movement_windows), clvhd-movements
  decoder.py            WienerFilter, the cursor decoder of examples 03/04
  centerout.py          center-out task pieces (targets, Calibration, reach_metrics)
  coadapt.py            co-adaptation task pieces (Envelope, Trajectory, optimal_decoder, ...)
  check.py              clvhd-check bring-up checks
  ui.py                 the example windows' shared look and widgets (pyqtgraph, imported lazily)
firmware/uctrl_teensy/  Teensy 4.0 firmware (Arduino; .clang-format)
examples/               01 live plot, 02 + hand, 03 center-out, 04 co-adaptation (participant UI in French)
tests/                  pytest suite; _teensy_sim.py simulates the Teensy on a pseudo-terminal
```

## Layers and imports

- Core driver (`protocol`, `ads1293`, `wristband`, `stream`, `contact`) needs only `numpy`
  and `pyserial`. Everything heavier (scipy, pyedflib, pyqtgraph, mediapipe, cv2) is an
  optional extra, imported inside the function that needs it — keep it that way.
- `clvhd/__init__.py` exports only the public API (`Wristband`, `Config`, `ControllerError`,
  `check_contact`, `find_port`); import submodules elsewhere, so `import clvhd` stays light.
- The UI/examples import the package, never the other way round.

## The serial-protocol boundary

`clvhd/protocol.py`, `firmware/uctrl_teensy/uctrl_teensy.ino` (+ `clvHd_util.hpp`),
`tests/_teensy_sim.py` and `docs/serial_protocol.md` describe the same wire protocol
(firmware v4.1): change all of them together, in one commit. Command bytes, frame layout
(22 bytes: sync, type, module, counter, `micros()`, status, 9 ECG bytes, CRC-8) and the
reply statuses must match on both sides; the sim is the executable spec for the tests.
Host constant names drop the firmware's `CLVHD_` prefix (`INIT_STUCK` = `CLVHD_INIT_STUCK`).

## Style

- Short docstrings (one line when that says it); comments only what the code can't say: a
  datasheet/protocol rule, a unit fact, an ordering constraint, why a constant has its value.
  Long rationale goes to `docs/`, not the docstring.
- Same name for the same concept everywhere: `nb_modules` (never `nb`/`n_mod` in Python),
  `t_host`/`t_device` for the two clocks, `mask` with bit `i` = module `i`, `data` in V,
  `u` for envelopes in mV (`U` stacked over frames). Grep before naming.
- Module-level constants are `UPPER_CASE` with a short `#` comment when the value needs a
  reason; private helpers start with `_`; dataclasses for configs and records.
- Don't mutate a caller's object (e.g. `Wristband.configure(config=...)` copies it).
- Errors the user can act on raise `ControllerError` with a specific, plain-sentence message
  (they reach the user verbatim). Anything else is a bug.

## Tests

- One file per `clvhd` module: `test_<module>.py`; add to the existing file before creating
  one. Group a behaviour in `class TestThing:`; test names are sentences about behaviour.
- Shared fixtures in `tests/conftest.py` (`make_sim`/`sim`/`faulty_sim` run the `_teensy_sim.py`
  pseudo-terminal Teensy; `load_example` imports an example, its `__main__` block skipped, to
  test what it saves in the file of the `clvhd` module it uses); other shared helpers in underscore-prefixed modules
  (`_teensy_sim.py`, `_synthetic.py`). Reuse `clvhd.bids.read_tsv` instead of re-defining it.
- Prefer synthetic data and the simulated Teensy — no hardware in CI. Seed randomness
  (`np.random.default_rng(0)`); `pytest.approx` for floats; write files only to `tmp_path`.
- `pytest.importorskip` for optional extras (scipy, pyedflib, pyqtgraph);
  `QT_QPA_PLATFORM=offscreen` is set at module level in the GUI tests.

## Run

```bash
uv sync --all-extras
uv run pytest                                        # full suite, no hardware needed
uv run ruff check . && uv run ruff format .
uv run mypy
uv run clvhd-check                                   # with the wristband plugged in
uv run examples/01_live_plot.py
```

## Before you finish

`uv run ruff check . && uv run ruff format .`, `uv run mypy`, `uv run pytest`. Update the one
doc that owns anything user-visible you changed (see the root AGENTS.md's documentation map).
