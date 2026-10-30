"""The camera hand landmarks (``clvhd.hand``)."""

import numpy as np
import pytest

from clvhd.hand import FINGERS, finger_flexion, pinch_aperture


def straight_hand():
    """Hand world landmarks (m): every finger a straight line out of the wrist, fanned."""
    hand = np.zeros((21, 3))
    for k, chain in enumerate(FINGERS.values()):
        direction = np.array([np.sin(0.3 * (k - 2)), -np.cos(0.3 * (k - 2)), 0])
        start = 1 if chain[0] == 1 else 0  # the thumb's chain starts at its CMC, off the wrist
        for j, i in enumerate(chain):
            hand[i] = 0.03 * (j + start) * direction
    return hand


def test_finger_flexion_and_pinch_aperture():
    hand = straight_hand()
    assert np.allclose(finger_flexion(hand), 0, atol=1e-6)
    bent = hand.copy()  # the index bent 90 deg at its PIP, in the hand's plane
    seg = hand[6] - hand[5]
    perp = np.array([-seg[1], seg[0], 0])
    bent[7], bent[8] = hand[6] + perp, hand[6] + 2 * perp
    flex = finger_flexion(bent)
    assert flex[1] == pytest.approx(90) and np.allclose(np.delete(flex, 1), 0, atol=1e-6)
    assert pinch_aperture(hand) == pytest.approx(100 * np.linalg.norm(hand[4] - hand[8]))
    stacked = np.stack([hand, bent])  # vectorised over frames; NaN where no hand
    assert finger_flexion(stacked).shape == (2, 5)
    assert np.isnan(finger_flexion(np.full((21, 3), np.nan))).all()
