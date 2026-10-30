"""The serial protocol (``clvhd.protocol``)."""

import numpy as np

from clvhd.protocol import crc8_rows


def test_crc8_matches_a_bytewise_implementation():
    rows = np.random.default_rng(0).integers(0, 256, (50, 20), dtype=np.uint8)
    for row, crc in zip(rows, crc8_rows(rows), strict=True):
        c = 0
        for b in row:
            c ^= int(b)
            for _ in range(8):
                c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        assert c == crc
