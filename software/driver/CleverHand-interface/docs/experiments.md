# Experiment design notes (examples 03 and 04)

Why the cursor experiments are built the way they are. `clvhd/centerout.py` and
`clvhd/coadapt.py` hold the task pieces without the display; `clvhd/decoder.py` the decoder
they share. This file holds the rationale that doesn't fit in a docstring.

## The decoder (`clvhd.decoder.WienerFilter`)

- Features: each channel's envelope `u` (mV) as `log(u + floor)`, the floor its median at rest
  on the calibration, z-scored on the calibration, over the last `LAGS` frames (100 ms at
  60 Hz), plus a constant. The log turns each electrode's gain into an offset and steadies the
  noise (on recorded calibrations: +0.07-0.1 R^2, and decoders that transfer between sessions,
  where linear envelopes failed); the z-score makes the ridge penalty relative to the data,
  whatever the envelope's scale.
- `WienerFilter.fit`: ridge regression of the velocity on them, the penalty picked by
  cross-validation over the cues (5 folds of consecutive cues).
- Output stage (03 only): a gain per target direction, interpolated by the decoded velocity's
  angle, undoing the ridge's shrinkage direction by direction (on the calibration's
  cross-validated predictions, the decoded speed towards each target matches the label's; a
  gain per half-axis would not: the diagonals set it); a dead-zone (the 75th percentile of the
  decoded speed at rest); and a speed curve
  `speed * ((s - deadzone) / (speed - deadzone)) ** power` anchored at the calibration's speed:
  the calibrated contraction keeps its speed, a stronger one goes faster than proportionally
  when `power` > 1, a weaker one slower (finer positioning).

Units: velocities in cm/s, envelopes in mV.

## Center-out (03_cursor_centerout.py)

Calibration (`acq-calib`): the 8 targets lit one at a time, each `--calib-reps` times in
shuffled blocks. A guide cursor travels from the centre to the lit target in 2 s; the
participant contracts as if pushing the cursor there, and relaxes when it jumps back to the
centre (2 s rest). The velocity labels are the guide's: `distance / CUE_S` towards the target
during the cue, 0 at rest. Frames within `REACTION_S` of a cue's start or end are left out of
the fit. Test (`acq-online`): `--reps` blocks of the 8 targets; each reach holds the cursor at
the centre 1 s with the target shown, then gives 10 s to hold it 0.5 s in the target.
`clvhd.centerout.reach_metrics` scores each reach (success, time to target, acquisition time,
path efficiency, entries).

## Co-adaptation (04_cursor_coadapt.py), after Madduri et al., Nat. Mach. Intell. 8, 372 (2026)

The paper's pieces, in `clvhd/coadapt.py`:

- `Envelope`: EMG linear envelope at the 60 Hz display rate (the paper's dissertation, ch. 3:
  band-pass, rectification, 40 Hz low-pass, moving average down to 60 Hz; a wider band than
  theirs, `BAND`).
- `Trajectory`: the pseudo-random sum-of-sines target (magnitudes 1/f^2), with the 5 s speed
  ramp.
- `optimal_decoder` / `smoothbatch`: the decoder update every 20 s,
  D* = argmin ||D x - (tau - y)/H||^2 + lambda ||D||^2 on the last 20 s, then
  D = alpha D- + (1 - alpha) D* (SmoothBatch, Orsborn et al. 2012).
- `conditions` / `session_plan`: the learning rates (fixed, slow, fast) at the low penalty,
  each once in each of two blocks of shuffled trials.
- `window_metrics`: what each 20 s window shows: tracking error, effort, and how the decoder's
  EMG part differs from the calibration's (gain, direction).

Not in the paper:

- The decoder is the calibration's (`clvhd.decoder.WienerFilter`, x its features: log
  envelopes, z-scored, over 100 ms, and a constant), and every trial starts from its fit
  instead of the paper's random D1 / D2, which hardly moved the cursor for the first 20 s
  while the participant, seeing nothing, contracted harder than they would later.
- The intended velocity is (tau - y)/H, the error closed in `HORIZON_S` (0.5 s), not in one
  frame (the paper's dt): a 2 cm error asks for 4 cm/s instead of 120 cm/s. With the paper's
  label, lambda alone brought the fits back to cursor speeds, so it decided whether a good
  decoder shrank to nothing or blew up (replaying recorded EMG: 6 to 160 cm/s within a
  minute, whatever the input's scale). With H, the fits ask for the speeds the task needs,
  and refine the decoder rather than rescale it: on in-control EMG, D* runs at 3-4 cm/s,
  the target's speed.
- The constant term (the velocity added whatever the EMG) stays the calibration's: only the
  EMG weights are refitted. Refitted, it learnt the last 20 s's mean error and applied it
  for the next 20 s, when that error was gone: the cursor drifted, the error flipped, and
  so did the constant (recorded, slow / low: up to the target's own speed by 140 s).
- A fixed-decoder condition (alpha = 1: D* is fitted and logged, never used), the control
  that separates the decoder's adaptation from the participant's practice; and every trial
  ends with `PROBE_S` s on the calibration's decoder, unannounced: the participant still
  in control then means the decoder learnt them, out of control means they learnt it.
- lambda is relative, like the calibration's ridge: times the mean feature energy of the
  20 s. Low (1, about what the calibration's cross-validation picks) and high (10) keep the
  paper's tenfold contrast, and its sense: a higher penalty gives a smaller decoder, so more
  effort for the same speed.
