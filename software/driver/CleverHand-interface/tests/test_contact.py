"""The electrode contact check (``clvhd.contact``)."""

import numpy as np

import clvhd


def test_contact_check_flags_flat_on_skin_and_lifted_channels():
    # values measured on the band: flat pairs read 0 V and ~3 uV of ADC noise,
    # pairs on skin tens of mV and tens of uV, a lifted electrode -344 mV
    rng = np.random.default_rng(0)
    n = 2083
    data = np.column_stack(
        [
            rng.normal(0, 3e-6, n),
            -0.030 + rng.normal(0, 30e-6, n),
            -0.344 + rng.normal(0, 130e-6, n),
        ]
    )
    data[[5, 50], 1] = np.nan  # missing samples are ignored
    assert clvhd.check_contact(data) == ["flat", "", "poor contact"]


def test_contact_check_flags_channels_without_data():
    data = np.full((500, 2), np.nan)
    data[:, 1] = -0.030 + np.random.default_rng(0).normal(0, 30e-6, 500)
    assert clvhd.check_contact(data) == ["no data", ""]
