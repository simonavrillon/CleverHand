"""The center-out task pieces (``clvhd.centerout``) and what example 03 saves of them."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clvhd import bids, centerout
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


def test_example_writes_the_reaches_as_events(tmp_path, load_example):
    """03's events start at their onset (ready, reach: stamped at their end), in tsv columns."""
    example = load_example(
        "03_cursor_centerout.py", "--subject", "01", "--bids-root", str(tmp_path)
    )

    ent = bids.Entities("01", "centerout", acquisition="online", run="01")
    emg = ent.folder(tmp_path) / "emg" / f"{ent.label()}_emg.edf"
    emg.parent.mkdir(parents=True)
    emg.touch()
    metrics = {
        "success": 1,
        "time_to_target": 1.2,
        "acquisition_time": 1.7,
        "path_efficiency": 0.9,
        "entries": 1,
    }
    frame = {"t": 101.0, "phase_t": 1.0, "state": "reach", "target": 2, "fit": 0}
    exp = SimpleNamespace(
        ent=ent,
        export=lambda: {"emg": emg, "acq_time": "2026-10-31T09:00:00"},
        raw=SimpleNamespace(t0=100.0),
        stream=SimpleNamespace(nominal_rate=2000.0, channel_labels=["m0_ch1"]),
        events=[
            {"t": 102.0, "trial_type": "ready", "duration": 1.0, "target": 2},
            {"t": 104.0, "trial_type": "reach", "duration": 1.7, "target": 2, **metrics},
        ],
        positions=centerout.targets(5.0),
        frames=[{**frame, "y": np.zeros(2), "v": np.zeros(2)}],
        phase="online",
        beh_sidecar=dict,
        wf=SimpleNamespace(
            gains=np.ones(8),
            deadzone=0.5,
            power=1.0,
            speed=4.0,
            save=np.savez,
        ),
        channels=[0],
    )
    assert example.save_phase(exp, np.zeros((1, 1)), {})

    # what 04 reads of the calibration's decoder file (decoder.channel_indices, load_plan)
    deriv = ent.folder(tmp_path / "derivatives" / "centerout") / "emg"
    with np.load(deriv / f"{ent.label()}_desc-decoder.npz") as f:
        assert {"decoder_channels", "band", "envelopes", "target"} <= set(f.files)
        assert list(f["decoder_channels"]) == ["m0_ch1"] and f["target"].tolist() == [2]

    ready, reach = bids.read_tsv(emg.with_name(f"{ent.label()}_events.tsv"))
    assert list(reach) == example.EVENT_COLUMNS
    assert float(ready["onset"]) == 1.0 and ready["sample"] == "2000"
    assert float(reach["onset"]) == pytest.approx(2.3)
    assert float(reach["target_x"]) == 0.0 and float(reach["target_y"]) == 5.0
    assert reach["success"] == "1"
