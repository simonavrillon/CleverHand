"""The center-out task pieces (``clvhd.centerout``)."""

import numpy as np

from clvhd import centerout
from clvhd.coadapt import FRAME_RATE


def test_targets_are_on_a_circle_counter_clockwise():
    p = centerout.targets(5.0)
    assert p.shape == (8, 2)
    assert np.allclose(np.linalg.norm(p, axis=1), 5.0)
    assert np.allclose(p[0], [5, 0]) and np.allclose(p[2], [0, 5])


def test_block_order_has_each_target_once_per_block():
    order = centerout.block_order(np.random.default_rng(0), 3)
    assert len(order) == 24
    for block in order.reshape(3, 8):
        assert sorted(block) == list(range(8))


def test_calibration_schedule():
    pos = centerout.targets(4.0)
    cal = centerout.Calibration([2, 5], pos)
    assert cal.duration == 2 * (centerout.REST_S + centerout.CUE_S) + centerout.REST_S
    rest = cal.at(1.0)
    assert rest["target"] == -1 and rest["fit"] and not rest["velocity"].any()
    mid = cal.at(centerout.REST_S + centerout.CUE_S / 2)
    assert mid["target"] == 2 and mid["fit"]
    assert np.allclose(mid["guide"], pos[2] / 2)
    assert np.allclose(mid["velocity"], pos[2] / centerout.CUE_S)
    assert not cal.at(centerout.REST_S + 0.1)["fit"]  # just after the cue lit
    assert not cal.at(centerout.REST_S + centerout.CUE_S + 0.1)["fit"]  # just after it ended
    assert cal.at(cal.duration - 0.5)["cue"] == 2  # the final rest


def test_reach_metrics():
    target, r = np.array([4.0, 0.0]), 1.0
    t = np.arange(0, 4, 1 / FRAME_RATE)
    x = np.clip(t * 2, 0, 4)  # straight there in 2 s, then stays
    m = centerout.reach_metrics(t, np.column_stack([x, 0 * x]), target, r, hold=0.5)
    assert m["success"] and m["entries"] == 1
    assert np.isclose(m["time_to_target"], 1.5, atol=1 / FRAME_RATE)
    assert np.isclose(m["acquisition_time"], 2.0, atol=1 / FRAME_RATE)
    assert np.isclose(m["path_efficiency"], 1.0, atol=0.02)
    # in and out too briefly, then away: not acquired
    x = np.where(t < 1.6, t * 2, 0.0)
    m = centerout.reach_metrics(t, np.column_stack([x, 0 * x]), target, r, hold=0.5)
    assert not m["success"] and m["entries"] == 1 and np.isnan(m["acquisition_time"])
