"""Load a recording from 02_record.py and plot the EMG envelopes.

    python 03_process.py [recording.npz]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

rec = np.load(sys.argv[1] if len(sys.argv) > 1 else "recording.npz")
fs = float(rec["fs"])
t = rec["t_device"] - rec["t_device"][0]  # uniform: the samples are aligned
data = rec["data"].copy()
labels = list(rec["labels"])
print(f"{data.shape[1]} channels, {len(t)} samples at {fs:.2f} Hz ({t[-1]:.1f} s)")

# fill missing samples by linear interpolation (all-missing channels: 0)
for c in range(data.shape[1]):
    nan = np.isnan(data[:, c])
    data[:, c] = np.interp(t, t[~nan], data[~nan, c]) if (~nan).any() else 0

band = signal.butter(4, [20, 450], btype="bandpass", fs=fs, output="sos")  # surface EMG band
notch_b, notch_a = signal.iirnotch(50, Q=10, fs=fs)  # mains (use 60 Hz where relevant)
smooth = signal.butter(4, 10, btype="low", fs=fs, output="sos")

x = signal.sosfiltfilt(band, data, axis=0)
x = signal.filtfilt(notch_b, notch_a, x, axis=0)
envelope = signal.sosfiltfilt(smooth, np.abs(x), axis=0)  # rectify and smooth

fig, ax = plt.subplots(figsize=(10, 6))
offset = 1.2 * np.nanmax(envelope)
for c in range(data.shape[1]):
    ax.plot(t, envelope[:, c] * 1e6 + c * offset * 1e6, lw=0.8)
ax.set_yticks(np.arange(data.shape[1]) * offset * 1e6, labels, fontsize=6)
ax.set_xlabel("Time (s)")
ax.set_title("EMG envelopes (20-450 Hz, rectified, 10 Hz low-pass)")
ax.set_xlim(t[0], t[-1])
plt.tight_layout()
plt.show()
