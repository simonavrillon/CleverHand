"""Electrode contact check from raw (unfiltered) samples."""

import warnings

import numpy as np

# A channel whose two electrodes sit at the same potential (shorted together
# by water, sweat or solder, or both floating off the skin) reads 0 V DC with
# only the ADC noise, ~3 uV rms (Config.lead_off tells the two apart). A
# channel on skin reads mV of DC offset and tens of uV of noise at least.
FLAT_RMS = 8e-6  # V
FLAT_DC = 0.5e-3  # V
# An electrode with poor contact drifts towards a rail: hundreds of mV of DC
# between the pair, against tens of mV for a good one.
OFFSET_DC = 150e-3  # V


def check_contact(data):
    """Contact status of each channel of `data` (N samples x channels, in V, raw).

    Returns one string per channel: "" (in contact), "flat" (both electrodes
    at the same potential: shorted together, or both off the skin), "poor
    contact" (large DC offset: one electrode barely touching) or "no data"
    (only NaN: the module sent nothing). Use about 0.5 s of data or more.
    """
    data = np.asarray(data, dtype=float)
    empty = np.isnan(data).all(axis=0)
    with warnings.catch_warnings():  # "Mean of empty slice" on the empty channels
        warnings.simplefilter("ignore", RuntimeWarning)
        dc = np.nanmean(data, axis=0)
        rms = np.nanstd(data, axis=0)
    status = np.full(data.shape[1], "", dtype=object)
    status[np.abs(dc) > OFFSET_DC] = "poor contact"
    status[(rms < FLAT_RMS) & (np.abs(dc) < FLAT_DC)] = "flat"
    status[empty] = "no data"
    return list(status)
