"""The co-adaptation pieces (``clvhd.coadapt``) and what example 04 saves of them."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clvhd import bids, coadapt


def test_plan_runs_every_condition_once_per_block():
    plan = coadapt.session_plan(np.random.default_rng(0))
    assert len(plan) == 6
    key = lambda c: (c["learning_rate"], c["penalty"])
    for block in (plan[:3], plan[3:]):
        assert sorted(map(key, block)) == sorted(map(key, coadapt.conditions()))
    assert {c["learning_rate"]: c["alpha"] for c in plan} == {
        "fixed": 1.0,
        "slow": 0.75,
        "fast": 0.25,
    }
    assert {c["penalty"] for c in plan} == {"low"} and plan[0]["lambda"] == 1.0


def test_optimal_decoder_keeps_the_constant_and_minimises_the_cost():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(size=(10, 600)), np.ones(600)])  # features, the constant last
    V, lam, c = rng.normal(size=(2, 600)) * 5 + [[3.0], [-2.0]], 1.0, np.array([0.5, -0.2])
    D = coadapt.optimal_decoder(X, V, lam, c)
    assert np.array_equal(D[:, -1], c)  # the constant term is kept, not refitted
    pen = lam * (X[:-1] ** 2).sum() / 10  # times the mean feature energy
    cost = lambda W: ((W @ X[:-1] + c[:, None] - V) ** 2).sum() + pen * (W**2).sum()
    for _ in range(20):  # no nearby weights do better
        assert cost(D[:, :-1]) <= cost(D[:, :-1] + rng.normal(size=(2, 10)) * 1e-3)
    # a larger penalty gives smaller weights, whatever the features' scale
    assert np.linalg.norm(coadapt.optimal_decoder(X, V, 10.0, c)[:, :-1]) < np.linalg.norm(
        D[:, :-1]
    )
    Xs = X.copy()
    Xs[:-1] *= 1e-3
    assert np.allclose(coadapt.optimal_decoder(Xs, V, lam, c)[:, :-1] * 1e-3, D[:, :-1])


def test_window_metrics():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=(500, 6)), np.ones(500)])
    D0 = rng.normal(size=(2, 7))
    m = coadapt.window_metrics(np.ones((500, 2)), np.zeros((500, 2)), np.full(500, 0.5), X, D0, D0)
    assert np.isclose(m["error"], np.sqrt(2)) and m["effort"] == 0.5
    assert np.isclose(m["gain"], 1) and np.isclose(m["direction"], 1)
    D = D0.copy()
    D[:, :-1] *= 2
    D[:, -1] += 5  # faster, same mapping; the constant ignored
    m = coadapt.window_metrics(np.zeros((500, 2)), np.zeros((500, 2)), np.ones(500), X, D, D0)
    assert np.isclose(m["gain"], 2) and np.isclose(m["direction"], 1)


def test_smoothbatch_weights_the_previous_decoder_by_alpha():
    prev, new = np.ones((2, 3)), np.zeros((2, 3))
    assert np.allclose(coadapt.smoothbatch(prev, new, 0.75), 0.75)
    assert np.allclose(coadapt.smoothbatch(prev, new, 0.25), 0.25)


def test_trajectory_amplitude_ramp_and_velocity():
    traj = coadapt.Trajectory(np.random.default_rng(3), amplitude=(15, 8))
    t = np.arange(0, coadapt.TRIAL_S, 1 / coadapt.FRAME_RATE)
    pos = traj.position(t)
    assert np.allclose(np.abs(pos).max(axis=0), (15, 8), rtol=1e-6)
    # at rest at the start, speeding up over the ramp
    v = traj.velocity(t)
    assert np.allclose(v[0], 0)
    assert np.allclose(np.gradient(pos, t, axis=0)[10:-10], v[10:-10], rtol=0.02, atol=0.05)
    # 1/f^2 magnitudes: the slow component dominates each axis
    spec = np.abs(np.fft.rfft(pos[int(coadapt.RAMP_S * 60) :, 0]))
    freqs = np.fft.rfftfreq(len(pos) - int(coadapt.RAMP_S * 60), 1 / 60)
    peak = lambda f: spec[np.argmin(np.abs(freqs - f))]
    assert peak(0.10) / peak(0.25) == pytest.approx((0.25 / 0.10) ** 2, rel=0.3)


def test_cursor_resets_after_being_held_at_the_edge():
    cursor = coadapt.Cursor(half_size=(20, 10), radius=0.5)
    t, dt = 0.0, 1 / 60
    reset_at = None
    while t < 5:
        if cursor.step(np.array([100.0, 0]), dt, t):
            reset_at = t
            break
        t += dt
    assert cursor.y[0] == 0 if reset_at else False
    assert reset_at is not None and coadapt.STUCK_S < reset_at < coadapt.STUCK_S + 0.5


def test_envelope_of_a_burst_in_microvolts():
    pytest.importorskip("scipy")
    fs = 2000
    env = coadapt.Envelope(fs, 2, powerline=50)
    rng = np.random.default_rng(4)
    t = np.arange(0, 4, 1 / fs)
    x = np.column_stack([rng.normal(size=len(t)) * 50e-6 * (t > 2), np.full(len(t), 0.1)])  # V
    x[100, 0] = np.nan  # a missing sample
    frames = []
    for k in range(0, len(t), fs // 60):
        env.add(x[k : k + fs // 60])
        frames.append(env.frame())
    values = np.array(frames)
    assert np.isfinite(values).all()
    rest, active = values[60:110, 0].mean(), values[-30:, 0].mean()
    # white noise of 50 uV rms keeps (450 - 20) / 1000 of its power in the 20-450 Hz band
    # (less the notches): 33 uV rms, a rectified mean of about 26 uV (the envelope is in mV)
    assert rest < 0.01 * active and 22e-3 < active < 30e-3
    assert values[-1, 1] < 1e-3  # DC offset removed


def test_example_saves_a_trial(tmp_path, load_example):
    """04's events, beh/ and derivative of a trial, with every event and the condition's names."""
    argv = ["--subject", "01", "--bids-root", str(tmp_path), "--decoder", "calib.npz"]
    example = load_example("04_cursor_coadapt.py", *argv)
    ent = example.trial_entities(0)
    emg = ent.folder(tmp_path) / "emg" / f"{ent.label()}_emg.edf"
    emg.parent.mkdir(parents=True)
    emg.touch()
    cond = coadapt.conditions()[1]
    frame = {
        "t": 101.0,
        "trial_t": 1.0,
        "tau": np.array([1.0, 2.0]),
        "y": np.zeros(2),
        "v": np.zeros(2),
        "u": np.ones(2),
        "dt": 1 / 60,
        "decoder_index": 1,
        "reset": 0,
        "effort": 0.5,
        "probe": 0,
    }
    window = {"start": 0.0, "end": 20.0, "probe": False, "decoder_index": 0}
    exp = SimpleNamespace(
        trial=0,
        export=lambda: {"emg": emg, "acq_time": "2026-10-31T09:00:00"},
        raw=SimpleNamespace(t0=100.0),
        stream=SimpleNamespace(nominal_rate=2000.0, channel_labels=["m0_ch1", "m0_ch2"]),
        t_start=100.5,
        cond=cond,
        update_times=[120.5],
        probe_time=380.5,
        resets=[200.0],
        frames=[frame],
        beh_sidecar=dict,
        target_radius=0.7,
        plan={"decoder_file": "calib.npz", "effort_reference": 0.2, "band": [20, 450]},
        windows=[{**window, "error": 1.0, "effort": 0.5, "gain": 1.0, "direction": 1.0}],
        decoders=[np.zeros((2, 3))],
        optimal=[np.zeros((2, 3))],
        channels=[0, 1],
        traj=SimpleNamespace(phases=np.zeros(4), scale=1.0),
    )
    assert example.save_trial(exp, 400.5)

    rows = bids.read_tsv(emg.with_name(f"{ent.label()}_events.tsv"))
    assert list(rows[0]) == example.EVENT_COLUMNS
    kinds = {r["trial_type"]: r for r in rows}
    assert set(kinds) == {
        "trial_start",
        "ramp_end",
        "decoder_update",
        "probe_start",
        "cursor_reset",
        "trial_end",
    }
    start = kinds["trial_start"]
    assert float(start["onset"]) == 0.5 and float(start["duration"]) == 300.0
    assert start["learning_rate"] == cond["learning_rate"] and float(start["lambda"]) == 1.0
    assert kinds["decoder_update"]["value"] == "1"

    beh = ent.folder(tmp_path) / "beh" / f"{ent.label()}_beh.tsv"
    (row,) = bids.read_tsv(beh)
    assert list(row) == list(example.BEH_COLUMNS) and row["decoder_index"] == "1"
    sidecar = json.loads(beh.with_suffix(".json").read_text())
    assert sidecar["Condition"] == cond and sidecar["DecoderFile"] == "calib.npz"

    deriv = ent.folder(tmp_path / "derivatives" / "coadapt") / "emg"
    with np.load(deriv / f"{ent.label()}_desc-coadapt.npz") as f:
        assert json.loads(str(f["condition"])) == cond
        assert f["decoder_file"] == "calib.npz" and f["window_decoder_index"].tolist() == [0]
        assert f["envelopes"].shape == (1, 2) and f["target_position"].tolist() == [[1.0, 2.0]]
