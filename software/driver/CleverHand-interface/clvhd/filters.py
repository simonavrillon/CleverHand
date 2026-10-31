"""Causal EMG filtering for the live displays and decoders (needs scipy).

``emg_band``: a band-pass with notches at the mains frequency and its harmonics within the
band (dry electrodes pick up mains). ``CausalFilter`` runs it sample block by sample block,
one state per channel, starting at steady state on the first complete sample and filtering
missing samples (NaN) as the channel's last value: a jump to 0 would ring.
"""

import numpy as np

BAND = (20, 450)  # Hz


def fill_gaps(last, data):
    """``data`` with each missing sample replaced by its channel's previous value (``last``: the row before)."""
    rows = np.vstack([last, data])
    idx = np.maximum.accumulate(np.where(np.isnan(rows), 0, np.arange(len(rows))[:, None]), axis=0)
    return rows[idx, np.arange(rows.shape[1])][1:]


def emg_band(fs, band=BAND, powerline=50.0):
    """Second-order sections: 4th-order low-pass at ``band[1]``, notches at ``powerline`` and its
    harmonics below it (none when 0), 4th-order high-pass at ``band[0]``."""
    from scipy.signal import butter, iirnotch, tf2sos

    stages = [butter(4, band[1], btype="low", fs=fs, output="sos")]
    if powerline:
        stages += [
            tf2sos(*iirnotch(f, 30, fs=fs)) for f in np.arange(powerline, band[1], powerline)
        ]
    stages.append(butter(4, band[0], btype="high", fs=fs, output="sos"))
    return np.concatenate(stages)


class CausalFilter:
    """``sos`` applied to each channel of successive blocks, with its state kept between them."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = self.last = None

    def __call__(self, x):
        """``(k, y)``: ``y`` the filtered ``x[k:]`` (N, channels), ``k`` the first complete sample
        of the first call (later calls: 0; ``len(x)`` while none has arrived)."""
        from scipy.signal import sosfilt, sosfilt_zi

        k = 0
        if len(x) == 0:
            return 0, x
        if self.zi is None:
            complete = np.flatnonzero(~np.isnan(x).any(axis=1))
            if len(complete) == 0:
                return len(x), x[:0]
            k = complete[0]
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[k]
            self.last = x[k]
        x = fill_gaps(self.last, x[k:])
        self.last = x[-1]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return k, y
