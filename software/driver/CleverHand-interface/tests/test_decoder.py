"""The Wiener filter decoder (``clvhd.decoder``), on synthetic cued calibrations."""

import numpy as np
import pytest

from clvhd import decoder
from tests._synthetic import simulated


class TestWienerFilter:
    # ── fit ─────────────────────────────────────────────────────────────

    def test_decodes_and_ignores_the_envelope_scale(self):
        U, V, groups, mask = simulated(np.random.default_rng(1), 1.0)
        wf = decoder.WienerFilter.fit(U, V, groups, mask)
        assert (wf.cv_r2 > 0.8).all()
        assert (decoder.r2(V[mask], wf.predict(U)[mask]) > 0.8).all()
        # the same EMG a thousand times smaller (µV-level envelopes in mV): the same decoder output
        small = decoder.WienerFilter.fit(U * 1e-3, V, groups, mask)
        assert small.ridge == wf.ridge
        assert np.allclose(small.predict(U * 1e-3), wf.predict(U), atol=1e-6)

    def test_rest_makes_no_motion(self):
        U, V, groups, mask = simulated(np.random.default_rng(2), 1.0)
        wf = decoder.WienerFilter.fit(U, V, groups, mask)
        rest = wf.predict(U)[mask & ~V.any(axis=1)]
        assert np.abs(rest.mean(axis=0)).max() < 0.3  # cm/s, against the 2.5 cm/s of the cues

    def test_step_push_and_save_load_round_trip(self, tmp_path):
        U, V, groups, mask = simulated(np.random.default_rng(3), 1.0)
        wf = decoder.WienerFilter.fit(U, V, groups, mask)
        x = [wf.push(u) for u in U[:200]]  # the features step() uses
        assert len(x[0]) == wf.lags * U.shape[1] + 1 and x[0][-1] == 1
        assert np.allclose(np.array(x) @ wf.weights, wf.predict(U[:200], linear=True))
        wf.reset()  # the pushes above advanced the history
        online = np.array([wf.step(u) for u in U[:200]])
        assert np.allclose(online, wf.predict(U[:200]))
        wf.save(tmp_path / "d.npz", channels=["a"])
        back = decoder.WienerFilter.load(tmp_path / "d.npz")
        assert back.lags == wf.lags and back.ridge == wf.ridge
        assert np.allclose(back.predict(U), wf.predict(U))

    # ── output stage ────────────────────────────────────────────────────

    def test_direction_gains_balance_the_speeds(self):
        U, V, groups, mask = simulated(np.random.default_rng(4), 1.0, noise=0.03)  # noisy: shrunk
        wf = decoder.WienerFilter.fit(U, V, groups, mask)
        assert wf.gains.shape == (8,) and (wf.gains > 1).all()  # the ridge shrinks, the gains undo it
        lin, out = wf.predict(U, linear=True), wf._gain(wf.predict(U, linear=True))
        k = np.round(np.arctan2(V[:, 1], V[:, 0]) / (np.pi / 4)).astype(int) % 8
        moving = mask & V.any(axis=1)
        speed = lambda P: np.array([np.linalg.norm(P[moving & (k == d)].mean(axis=0)) for d in range(8)])
        assert np.abs(speed(out) - wf.speed).max() < np.abs(speed(lin) - wf.speed).max()
        assert np.abs(speed(out) / wf.speed - 1).max() < 0.2

    def test_gain_interpolates_between_directions(self):
        wf = decoder.WienerFilter(np.zeros((1, 2)), [0.0], [1.0], lags=1, gains=np.arange(1.0, 9.0))
        assert np.allclose(wf._gain(np.array([1.0, 0.0])), [1, 0])  # target 0: gain 1
        assert np.allclose(wf._gain(np.array([0.0, 2.0])), [0, 6])  # target 2: gain 3
        half = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])  # between targets 0 and 1
        assert np.allclose(wf._gain(half), 1.5 * half)
        below = np.array([[np.cos(np.pi / 8), -np.sin(np.pi / 8)]])  # between targets 7 and 0
        assert np.allclose(wf._gain(below), 4.5 * below)

    def test_speed_curve(self):
        wf = decoder.WienerFilter(np.zeros((1, 2)), [0.0], [1.0], lags=1, speed=3.0, deadzone=0.5,
                                    power=2.0)
        d = np.array([0.6, 0.8])  # a unit direction
        assert np.allclose(wf.output(0.4 * d), 0)  # in the dead-zone
        assert np.allclose(wf.output(3.0 * d), 3.0 * d)  # the calibration speed is kept
        assert np.linalg.norm(wf.output(4.0 * d)) > 4.0  # stronger: faster than proportionally
        assert np.linalg.norm(wf.output(2.0 * d)) < 2.0  # weaker: slower, for fine positioning
        assert np.allclose(wf.output(np.array([[0.0, 0.0]])), 0)
        wf.power, wf.deadzone = 1.0, 0.0
        assert np.allclose(wf.output(2.0 * d), 2.0 * d)

    # ── files ───────────────────────────────────────────────────────────

    def test_old_decoder_file_is_refused(self, tmp_path):
        np.savez(tmp_path / "old.npz", weights=np.zeros((7, 2)), mean=np.zeros(1), sd=np.ones(1),
                 lags=6, ridge=0.1, cv_r2=[0.5, 0.5])  # linear envelopes, no output stage
        with pytest.raises(ValueError, match="calibrate again"):
            decoder.WienerFilter.load(tmp_path / "old.npz")
