"""Causal EMG filtering (clvhd.filters)."""

import numpy as np
import pytest

pytest.importorskip("scipy")
from clvhd.filters import CausalFilter, emg_band, fill_gaps

FS = 2000 / 0.96  # the ADS1293 rate of the default configuration, about


def test_fill_gaps_holds_each_channel_last_value():
    data = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan]])
    assert np.array_equal(fill_gaps([5.0, 6.0], data), [[5, 1], [2, 1], [2, 1]])


def test_blocks_filter_as_one_starting_at_the_first_complete_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 3))
    x[:5, 1] = np.nan  # incomplete first samples: skipped
    x[1000:1020, 2] = np.nan  # a gap: filtered as the last value
    whole_k, whole = CausalFilter(emg_band(FS))(x)
    f = CausalFilter(emg_band(FS))
    parts = [f(x[a:a + 128]) for a in range(0, len(x), 128)]
    assert whole_k == parts[0][0] == 5 and all(k == 0 for k, _ in parts[1:])
    assert np.allclose(np.vstack([y for _, y in parts]), whole)
    assert np.isfinite(whole).all()


def test_nothing_until_a_complete_sample():
    f = CausalFilter(emg_band(FS))
    k, y = f(np.full((10, 2), np.nan))
    assert k == 10 and y.shape == (0, 2)
    assert f(np.zeros((0, 2)))[1].shape == (0, 2)


def test_notches_the_mains_and_its_harmonics():
    t = np.arange(int(4 * FS)) / FS
    _, y = CausalFilter(emg_band(FS, powerline=50))(np.column_stack(
        [np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 150 * t), np.sin(2 * np.pi * 100 * t + 1)]))
    assert np.abs(y[-int(FS):]).max(axis=0).max() < 0.05  # after the notches settle
