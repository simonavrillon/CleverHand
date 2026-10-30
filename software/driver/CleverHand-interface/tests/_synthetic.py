"""Synthetic cued calibrations for the decoder tests."""

import numpy as np

from clvhd import centerout
from clvhd.coadapt import FRAME_RATE


def simulated(rng, scale, n_channels=12, noise=0.002):
    """Envelopes of a cued calibration: each channel tuned to a direction, plus ``noise``, in mV
    times ``scale``; and the labels, cues and mask."""
    pos = centerout.targets(5.0)
    cal = centerout.Calibration(centerout.block_order(rng, 3), pos)
    t = np.arange(0, cal.duration, 1 / FRAME_RATE)
    cues = [cal.at(x) for x in t]
    V = np.array([c["velocity"] for c in cues])
    tuning = rng.normal(size=(n_channels, 2))
    U = 0.01 + np.clip(V @ tuning.T, 0, None) * 0.004 + noise * rng.random((len(t), n_channels))
    return U * scale, V, np.array([c["cue"] for c in cues]), np.array([c["fit"] for c in cues])
