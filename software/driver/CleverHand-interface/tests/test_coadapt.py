"""The co-adaptation pieces (``clvhd.coadapt``)."""

import numpy as np
import pytest

from clvhd import coadapt


def test_plan_runs_every_condition_once_per_block():
    plan = coadapt.session_plan(np.random.default_rng(0))
    assert len(plan) == 6
    key = lambda c: (c["rate"], c["penalty"])
    for block in (plan[:3], plan[3:]):
        assert sorted(map(key, block)) == sorted(map(key, coadapt.conditions()))
    assert {c["rate"]: c["alpha"] for c in plan} == {"fixed": 1.0, "slow": 0.75, "fast": 0.25}
    assert {c["penalty"] for c in plan} == {"low"} and plan[0]["lam"] == 1.0


def test_optimal_decoder_keeps_the_constant_and_minimises_the_cost():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(size=(10, 600)), np.ones(600)])  # features, the constant last
    V, lam, c = rng.normal(size=(2, 600)) * 5 + [[3.0], [-2.0]], 1.0, np.array([0.5, -0.2])
    D = coadapt.optimal_decoder(X, V, lam, c)
    assert np.array_equal(D[:, -1], c)  # the constant term is kept, not refitted
    pen = lam * (X[:-1] ** 2).sum() / 10  # times the mean feature energy
    cost = lambda W: ((W @ X[:-1] + c[:, None] - V) ** 2).sum() + pen * (W ** 2).sum()
    for _ in range(20):  # no nearby weights do better
        assert cost(D[:, :-1]) <= cost(D[:, :-1] + rng.normal(size=(2, 10)) * 1e-3)
    # a larger penalty gives smaller weights, whatever the features' scale
    assert np.linalg.norm(coadapt.optimal_decoder(X, V, 10.0, c)[:, :-1]) < np.linalg.norm(D[:, :-1])
    Xs = X.copy(); Xs[:-1] *= 1e-3
    assert np.allclose(coadapt.optimal_decoder(Xs, V, lam, c)[:, :-1] * 1e-3, D[:, :-1])


def test_window_metrics():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=(500, 6)), np.ones(500)])
    D0 = rng.normal(size=(2, 7))
    m = coadapt.window_metrics(np.ones((500, 2)), np.zeros((500, 2)), np.full(500, 0.5), X, D0, D0)
    assert np.isclose(m["error"], np.sqrt(2)) and m["effort"] == 0.5
    assert np.isclose(m["gain"], 1) and np.isclose(m["direction"], 1)
    D = D0.copy(); D[:, :-1] *= 2; D[:, -1] += 5  # faster, same mapping; the constant ignored
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
    spec = np.abs(np.fft.rfft(pos[int(coadapt.RAMP_S * 60):, 0]))
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
        env.add(x[k:k + fs // 60])
        frames.append(env.frame())
    values = np.array(frames)
    assert np.isfinite(values).all()
    rest, active = values[60:110, 0].mean(), values[-30:, 0].mean()
    # white noise of 50 uV rms keeps (450 - 20) / 1000 of its power in the 20-450 Hz band
    # (less the notches): 33 uV rms, a rectified mean of about 26 uV (the envelope is in mV)
    assert rest < 0.01 * active and 22e-3 < active < 30e-3
    assert values[-1, 1] < 1e-3  # DC offset removed
