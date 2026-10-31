"""The co-adaptation pieces (``clvhd.coadapt``) and what example 04 saves of them."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clvhd import bids, coadapt
from tests._simulated_user import SimulatedUser, fixed_gain, run_trial


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


def test_gain_plan_brackets_interleaved_staircases_with_references():
    plan = coadapt.gain_session_plan(np.random.default_rng(0))
    starts = [c["start"] for c in plan]
    n_ref = coadapt.GAIN_REFERENCES
    assert starts[:n_ref] == ["reference"] * n_ref and starts[-1] == "reference"
    pairs = starts[n_ref:-1]
    assert len(pairs) == 2 * coadapt.GAIN_RUNS
    for k in range(0, len(pairs), 2):  # each pair runs both starts
        assert sorted(pairs[k : k + 2]) == sorted(coadapt.GAIN_STARTS)


def test_window_metrics():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=(500, 6)), np.ones(500)])
    D0 = rng.normal(size=(2, 7))  # one lag of 6 channels
    t = 10 + np.arange(500) / 60
    v = np.column_stack([np.sin(2 * np.pi * 0.10 * t), np.sin(2 * np.pi * 0.35 * t)])
    Z = np.abs(X[:, :-1])
    m = coadapt.window_metrics(
        np.ones((500, 2)), np.zeros((500, 2)), np.full(500, 0.5), X, D0, D0, t, v, Z
    )
    assert np.isclose(m["error"], np.sqrt(2)) and m["effort"] == 0.5
    assert np.isclose(m["gain"], 1) and np.isclose(m["direction"], 1)
    assert np.allclose(m["remnant"], 0, atol=1e-9)
    assert m["null_space"] == pytest.approx(coadapt.null_space_share(Z, D0))
    D = D0.copy()
    D[:, :-1] *= 2
    D[:, -1] += 5  # faster, same mapping; the constant ignored
    m = coadapt.window_metrics(
        np.zeros((500, 2)), np.zeros((500, 2)), np.ones(500), X, D, D0, t, v, Z
    )
    assert np.isclose(m["gain"], 2) and np.isclose(m["direction"], 1)
    assert m["null_space"] == pytest.approx(coadapt.null_space_share(Z, D0))  # gain-free


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


class TestGainStaircase:
    def test_raises_the_gain_below_the_target_ratio_and_lowers_it_above(self):
        st = coadapt.GainStaircase(1.0, e_ref=2.0, tol=0.1, a0=1.0)
        assert st.update(2.2) == pytest.approx(1.0)  # at the target: stays
        assert st.update(2.0) == pytest.approx(np.exp(0.1))  # ratio 1: +0.1 in log gain
        assert st.update(2.4) < np.exp(0.1)

    def test_steps_are_clipped_and_shrink_at_each_crossing(self):
        st = coadapt.GainStaircase(1.0, e_ref=1.0, tol=0.1, a0=1.0, a_min=0.25)
        assert st.update(0.0) == pytest.approx(1.2)  # +1.1 asked, 20 % allowed
        assert st.update(1.15) == pytest.approx(1.2 * np.exp(-0.5 * 0.05))  # crossed: halved
        assert st.crossings == 1
        for ratio in (1.0, 1.2, 1.0, 1.2):
            st.update(ratio)
        g = st.gain
        assert st.update(1.0) == pytest.approx(g * np.exp(0.25 * 0.1))  # down to the floor

    def test_continues_from_its_state(self):
        st = coadapt.GainStaircase(1.0, e_ref=1.0)
        for ratio in (1.0, 1.3, 1.0):
            st.update(ratio)
        again = coadapt.GainStaircase(e_ref=1.0, **st.state())
        assert again.update(1.3) == pytest.approx(st.update(1.3))

    def test_settles_where_a_simulated_user_misses_the_target_by_tol(self):
        """From both starts, on the crossover-model user of tests/_simulated_user.py: the
        error ratio at the end is 1 + tol, the gain moved towards it, and from the low start
        the effort fell."""
        e_ref, _ = fixed_gain(
            np.random.default_rng(0), 1.0, SimulatedUser(np.random.default_rng(1))
        )
        for start, g0 in coadapt.GAIN_STARTS.items():
            ratios, finals = [], []
            for seed in range(3):
                st = coadapt.GainStaircase(g0, e_ref)
                gains, windows = run_trial(
                    np.random.default_rng(10 + seed),
                    st,
                    SimulatedUser(np.random.default_rng(20 + seed)),
                )
                ratios.append(windows[-4:, 0].mean() / e_ref)
                finals.append(gains[-1])
                if start == "low":
                    assert windows[-3:, 1].mean() < 0.7 * windows[0, 1]
            assert np.mean(ratios) == pytest.approx(1 + coadapt.GAIN_TOL, abs=0.06), start
            assert 0.8 < np.mean(finals) < 2.0, start


def test_remnant_is_what_the_target_frequencies_leave_unexplained():
    rng = np.random.default_rng(5)
    t = np.arange(10, 25, 1 / 60)  # a 15 s window, not a whole number of periods
    v = np.column_stack(
        [
            np.sin(2 * np.pi * 0.10 * t + 1) + 0.5 * np.cos(2 * np.pi * 0.25 * t),
            np.sin(2 * np.pi * 0.15 * t) - np.sin(2 * np.pi * 0.35 * t + 2),
        ]
    )
    assert np.allclose(coadapt.remnant(t, v + 3), 0, atol=1e-12)  # a constant velocity too
    noise = rng.normal(size=v.shape)
    r = coadapt.remnant(t, v + noise)
    assert np.allclose(r, noise.var(axis=0) / (v + noise).var(axis=0), rtol=0.05)
    assert coadapt.remnant(t, noise) == pytest.approx([1, 1], abs=0.02)
    # the other axis's frequencies count as remnant (less what their sines share over 15 s)
    assert coadapt.remnant(t, v[:, ::-1])[0] > 0.8


def test_null_space_share_is_the_activity_the_decoder_maps_to_nothing():
    rng = np.random.default_rng(6)
    W = rng.normal(size=(2, 5))  # per channel, over 2 lags: D sums them
    D = np.column_stack([0.3 * W, 0.7 * W, [1.0, -1.0]])
    along = rng.normal(size=(100, 2)) @ W  # in the span of the weights
    other = rng.normal(size=(100, 5))
    other -= other @ np.linalg.pinv(W) @ W
    assert coadapt.null_space_share(along, D) == pytest.approx(0, abs=1e-12)
    assert coadapt.null_space_share(other, D) == pytest.approx(1)
    both = coadapt.null_space_share(along + other, D)
    assert both == pytest.approx((other**2).sum() / ((along + other) ** 2).sum())
    assert np.isnan(coadapt.null_space_share(np.zeros((10, 5)), D))


PLAN = {"decoder_file": "calib.npz", "effort_reference": 0.2, "band": [20, 450]}
WINDOW = {
    "start": 0.0,
    "end": 20.0,
    "probe": False,
    "decoder_index": 0,
    "error": 1.0,
    "effort": 0.5,
    "gain": 1.0,
    "direction": 1.0,
    "remnant": np.zeros(2),
    "null_space": 0.5,
}


def _save(example, root, trial, cond, **fields):
    """04's save_trial of a one-frame ``trial`` of ``cond``; ``fields`` replace the
    experiment's. Returns the trial's entities and EMG file."""
    ent = example.trial_entities(trial)
    emg = ent.folder(root) / "emg" / f"{ent.label()}_emg.edf"
    emg.parent.mkdir(parents=True, exist_ok=True)
    emg.touch()
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
        "gain": 1.0,
    }
    exp = {
        "trial": trial,
        "export": lambda: {"emg": emg, "acq_time": "2026-10-31T09:00:00"},
        "raw": SimpleNamespace(t0=100.0),
        "stream": SimpleNamespace(nominal_rate=2000.0, channel_labels=["m0_ch1", "m0_ch2"]),
        "t_start": 100.5,
        "cond": cond,
        "update_times": [120.5],
        "probe_time": 380.5,
        "resets": [200.0],
        "frames": [frame],
        "beh_sidecar": dict,
        "target_radius": 0.7,
        "plan": PLAN,
        "windows": [WINDOW],
        "decoders": [np.zeros((2, 3))],
        "optimal": [np.zeros((2, 3))],
        "channels": [0, 1],
        "traj": SimpleNamespace(phases=np.zeros(4), scale=1.0),
        "gains": [1.0],
        "e_ref": None,
        "staircase": None,
    }
    assert example.save_trial(SimpleNamespace(**{**exp, **fields}), 400.5)
    return ent, emg


def test_example_saves_a_trial(tmp_path, load_example):
    """04's events, beh/ and derivative of a trial, with every event and the condition's names."""
    argv = ["--subject", "01", "--bids-root", str(tmp_path), "--decoder", "calib.npz"]
    example = load_example("04_cursor_coadapt.py", *argv)
    cond = coadapt.conditions()[1]
    ent, emg = _save(example, tmp_path, 0, cond)

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
    assert start["protocol"] == "coadapt"
    assert kinds["decoder_update"]["value"] == "1"

    beh = ent.folder(tmp_path) / "beh" / f"{ent.label()}_beh.tsv"
    (row,) = bids.read_tsv(beh)
    assert list(row) == list(example.BEH_COLUMNS) and row["decoder_index"] == "1"
    sidecar = json.loads(beh.with_suffix(".json").read_text())
    assert sidecar["Condition"] == cond and sidecar["DecoderFile"] == "calib.npz"
    assert sidecar["Protocol"] == "coadapt" and sidecar["ProbeS"] == coadapt.PROBE_S

    deriv = ent.folder(tmp_path / "derivatives" / "coadapt") / "emg"
    with np.load(deriv / f"{ent.label()}_desc-coadapt.npz") as f:
        assert json.loads(str(f["condition"])) == cond
        assert f["decoder_file"] == "calib.npz" and f["window_decoder_index"].tolist() == [0]
        assert f["envelopes"].shape == (1, 2) and f["target_position"].tolist() == [[1.0, 2.0]]
        assert f["window_remnant"].shape == (1, 2) and str(f["staircase"]) == ""


def test_example_gain_protocol_carries_the_staircase_over(tmp_path, load_example):
    """04 --protocol gain: e_ref from the saved references, each start's staircase continued
    from its previous trial, and what a staircase trial saves."""
    argv = ["--subject", "01", "--bids-root", str(tmp_path), "--decoder", "calib.npz"]
    example = load_example("04_cursor_coadapt.py", *argv, "--protocol", "gain")
    assert example.args.trial_seconds == coadapt.GAIN_TRIAL_S
    conds = coadapt.gain_session_plan(np.random.default_rng(0))
    plan = {**PLAN, "conditions": conds}
    no_probe = {"plan": plan, "probe_time": None}
    for k in range(coadapt.GAIN_REFERENCES):
        windows = [{**WINDOW, "error": 1.0 + k}]
        _save(example, tmp_path, k, conds[k], update_times=[], windows=windows, **no_probe)
    first = coadapt.GAIN_REFERENCES
    assert example.reference_error(plan, first) == pytest.approx(1.5)
    start = conds[first]["start"]
    state = example.staircase_state(plan, first, start)
    assert state == {"gain": coadapt.GAIN_STARTS[start]}
    st = coadapt.GainStaircase(e_ref=1.5, **state)
    g = st.update(1.5)
    ent, emg = _save(
        example,
        tmp_path,
        first,
        conds[first],
        gains=[state["gain"], g],
        e_ref=1.5,
        staircase=st,
        **no_probe,
    )
    later = next(k for k in range(first + 1, len(conds)) if conds[k]["start"] == start)
    assert example.staircase_state(plan, later, start) == st.state()

    kinds = {r["trial_type"]: r for r in bids.read_tsv(emg.with_name(f"{ent.label()}_events.tsv"))}
    assert "probe_start" not in kinds and "decoder_update" not in kinds
    assert kinds["trial_start"]["protocol"] == "gain" and kinds["trial_start"]["start"] == start
    assert float(kinds["trial_start"]["gain"]) == state["gain"]
    assert float(kinds["gain_update"]["value"]) == pytest.approx(g, abs=1e-4)
    beh = ent.folder(tmp_path) / "beh" / f"{ent.label()}_beh.tsv"
    sidecar = json.loads(beh.with_suffix(".json").read_text())
    assert sidecar["Protocol"] == "gain" and sidecar["ReferenceErrorCm"] == 1.5
    with np.load(example.derivative_path(first)) as f:
        assert f["gains"].tolist() == [state["gain"], g] and float(f["reference_error"]) == 1.5
