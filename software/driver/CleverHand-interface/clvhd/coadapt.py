"""Co-adaptive myoelectric cursor control, after Madduri et al., Nat. Mach. Intell. 8, 372 (2026).

The pieces of their experiment, without the display (``examples/04_cursor_coadapt.py``): the
EMG ``Envelope`` at the 60 Hz display rate, the pseudo-random sum-of-sines ``Trajectory``, the
20 s decoder update (``optimal_decoder`` / ``smoothbatch``), the conditions (``conditions`` /
``session_plan``) and the per-window metrics (``window_metrics``). How this deviates from the
paper: docs/experiments.md. Also 04's gain protocol: the calibration's decoder at the gain of
a ``GainStaircase`` (``gain_session_plan``).

Units: positions in cm on the screen, velocities in cm/s, envelopes in mV.
"""

import numpy as np

from .filters import BAND, CausalFilter, emg_band  # BAND: the envelope's (the paper's: 40-130 Hz)

FRAME_RATE = 60.0  # Hz, display and decoder rate
TRIAL_S = 300.0
RAMP_S = 5.0
UPDATE_S = 20.0  # decoder update period, on the previous UPDATE_S of data
STUCK_S = 3.33  # cursor held at the screen edge this long: reset to the centre
# the task field (cm), whatever the screen: an 11-inch iPad in landscape (2360 x 1640 px at
# 264 ppi). The target's path and speeds scale with it; the paper's display (HP Compaq
# L2206tm) was 46.5 x 24.5 cm (--field-cm 46.5 24.5)
FIELD_CM = (22.7, 15.8)
# SmoothBatch alpha: weight of the previous decoder (fixed: never updated, the control)
LEARNING_RATES = {"fixed": 1.0, "slow": 0.75, "fast": 0.25}
# lambda_D, times the mean feature energy (the paper: 1e2, 1e3)
PENALTIES = {"low": 1.0, "high": 10.0}
PENALTY = "low"  # the sessions'
HORIZON_S = 0.5  # the intended velocity closes the cursor's error in this time
PROBE_S = 20.0  # each trial's last seconds, on the calibration's decoder
# target: sums of sines at prime multiples of 0.05 Hz, distinct per axis
FREQS_X = (0.10, 0.25)
FREQS_Y = (0.15, 0.35)
# gain staircase: the largest decoder gain whose error stays within GAIN_TOL of the reference
GAIN_TOL = 0.1
# first step, in log gain per unit of error ratio, and its floor (to keep following the
# user's learning): on the simulated user, the smallest that reach the threshold from 0.5
# within 14 updates (tests/_simulated_user.py)
GAIN_A0 = 3.0
GAIN_A_MIN = 0.75
GAIN_STEP_MAX = float(np.log(1.2))  # at most 20 % per update: small overshoots
SETTLE_S = 5.0  # left out of each window after a gain change, while the user re-tunes
GAIN_STARTS = {"low": 0.5, "high": 2.0}  # times the calibration's gain
# gain protocol: short trials against fatigue, the staircase carried over between a start's
# trials; reference trials (gain 1) first for e_ref, and one last
GAIN_TRIAL_S = 120.0
GAIN_RUNS = 3  # trials per start
GAIN_REFERENCES = 2


def conditions():
    """The decoder conditions, every learning rate at ``PENALTY``: dicts of learning_rate, alpha, penalty, lambda."""
    return [
        {"learning_rate": r, "alpha": alpha, "penalty": PENALTY, "lambda": PENALTIES[PENALTY]}
        for r, alpha in LEARNING_RATES.items()
    ]


def gain_session_plan(rng, runs=GAIN_RUNS):
    """Trial conditions of the gain protocol: ``GAIN_REFERENCES`` references, then ``runs``
    pairs of a low- and a high-start staircase trial (each pair shuffled), then a reference:
    dicts of start ("reference", or a key of ``GAIN_STARTS``)."""
    plan = [{"start": "reference"} for _ in range(GAIN_REFERENCES)]
    for _ in range(runs):
        pair = [{"start": start} for start in GAIN_STARTS]
        rng.shuffle(pair)
        plan += pair
    return [*plan, {"start": "reference"}]


def session_plan(rng, blocks=2):
    """Trial conditions: each block runs every condition once, in a random order."""
    plan = []
    for _ in range(blocks):
        block = conditions()
        rng.shuffle(block)
        plan += block
    return plan


class Trajectory:
    """Target position (cm from the screen centre) over trial time.

    Each axis sums two sines with random phases, each of magnitude 1/f^2 ("the
    magnitude of each frequency component was normalized by the frequency
    squared"), scaled so the target spans ``amplitude`` (cm per axis) at most.
    Over the first ``ramp`` s the target speeds up from rest: time runs at
    ``speed(t) = t / ramp`` of its rate.
    """

    def __init__(self, rng, amplitude, duration=TRIAL_S, ramp=RAMP_S):
        self.ramp = ramp
        self.phases = [rng.uniform(0, 2 * np.pi, len(f)) for f in (FREQS_X, FREQS_Y)]
        peak = np.abs(self._unscaled(self.warp(np.arange(0, duration, 1 / FRAME_RATE)))).max(axis=0)
        self.scale = np.asarray(amplitude, dtype=float) / peak

    def speed(self, t):
        return np.clip(np.asarray(t, dtype=float) / self.ramp, 0, 1) if self.ramp > 0 else 1.0

    def warp(self, t):
        """Trajectory time reached at trial time ``t`` (s), with the ramp."""
        t = np.asarray(t, dtype=float)
        if self.ramp <= 0:
            return t
        return np.where(t < self.ramp, t * t / (2 * self.ramp), t - self.ramp / 2)

    def _unscaled(self, s):
        """(N, 2) sums of sines at trajectory times ``s``, before the scaling to the amplitude."""
        s = np.atleast_1d(s)[:, None]
        out = []
        for freqs, phases in zip((FREQS_X, FREQS_Y), self.phases, strict=True):
            f = np.asarray(freqs)
            out.append((np.sin(2 * np.pi * f * s + phases) / f**2).sum(axis=1))
        return np.column_stack(out)

    def position(self, t):
        """(2,) target position at trial time ``t``, or (N, 2) for an array."""
        p = self._unscaled(self.warp(t)) * self.scale
        return p[0] if np.ndim(t) == 0 else p

    def velocity(self, t):
        """Target velocity (cm/s) at trial time ``t``."""
        s = np.atleast_1d(self.warp(t))[:, None]
        parts = []
        for freqs, phases in zip((FREQS_X, FREQS_Y), self.phases, strict=True):
            f = np.asarray(freqs)
            parts.append((2 * np.pi * np.cos(2 * np.pi * f * s + phases) / f).sum(axis=1))
        v = np.column_stack(parts) * self.scale * np.atleast_1d(self.speed(t))[:, None]
        return v[0] if np.ndim(t) == 0 else v


def optimal_decoder(X, V, lam, constant):
    """D* (2, p): the EMG weights W = argmin ||W X' + c - V||_F^2 + lam * e ||W||_F^2 for
    features X (p, T), the constant last (X' the others, e their mean energy), intended
    velocities V (2, T) and the constant term c (2,), kept as given: D* = [W, c].

    The paper minimises its absolute-lambda version, constant included, with SciPy from a
    least-squares start; it is a ridge regression, so its minimiser has a closed form,
    computed here exactly.
    """
    Xe = X[:-1]
    G = Xe @ Xe.T
    W = np.linalg.solve(
        G + lam * np.trace(G) / len(G) * np.eye(len(G)), Xe @ (V - np.asarray(constant)[:, None]).T
    ).T
    return np.column_stack([W, constant])


def smoothbatch(D_prev, D_star, alpha):
    """SmoothBatch update: alpha weighs the previous decoder (0.75 slow, 0.25 fast)."""
    return alpha * D_prev + (1 - alpha) * D_star


def window_metrics(tau, y, effort, X, D, D0, t, v, Z):
    """One window's dict: ``error`` (mean distance target - cursor, cm), ``effort`` (mean);
    the EMG part of decoder D against the calibration's D0 on its features X (T, p): ``gain``
    (mean speed ratio) and ``direction`` (mean correlation of their outputs per axis; 1: the
    same mapping, rescaled); the ``remnant`` (2,) of the velocities ``v`` at trial times
    ``t``, and the ``null_space`` share of the activity ``Z`` (T, C) under D."""
    P, P0 = X[:, :-1] @ D[:, :-1].T, X[:, :-1] @ D0[:, :-1].T
    s0 = np.linalg.norm(P0, axis=1).mean()
    ok = [a for a in range(2) if P[:, a].std() > 0 and P0[:, a].std() > 0]
    return {
        "error": float(np.linalg.norm(np.asarray(tau) - np.asarray(y), axis=1).mean()),
        "effort": float(np.mean(effort)),
        "gain": float(np.linalg.norm(P, axis=1).mean() / s0) if s0 > 0 else np.nan,
        "direction": float(np.mean([np.corrcoef(P[:, a], P0[:, a])[0, 1] for a in ok]))
        if ok
        else np.nan,
        "remnant": remnant(t, v),
        "null_space": null_space_share(Z, D),
    }


class GainStaircase:
    """The largest decoder gain whose tracking error stays within ``tol`` of the reference
    ``e_ref``: stochastic approximation on log gain (Robbins-Monro), its step shrinking at each
    crossing of the target ratio 1 + ``tol`` (Kesten), down to ``a_min``. ``crossings`` and
    ``last`` continue a staircase (``state()``) from an earlier trial."""

    def __init__(
        self,
        gain,
        e_ref,
        tol=GAIN_TOL,
        a0=GAIN_A0,
        a_min=GAIN_A_MIN,
        max_step=GAIN_STEP_MAX,
        crossings=0,
        last=0.0,
    ):
        self.gain, self.e_ref, self.tol = float(gain), float(e_ref), tol
        self.a0, self.a_min, self.max_step = a0, a_min, max_step
        self.crossings = int(crossings)
        self.last = float(last)  # the previous window's error ratio minus the target

    def state(self):
        """What continues this staircase: gain, crossings, last."""
        return {"gain": self.gain, "crossings": self.crossings, "last": self.last}

    def update(self, error):
        """The gain after a window of mean tracking ``error`` (cm)."""
        d = error / self.e_ref - (1 + self.tol)
        if d * self.last < 0:
            self.crossings += 1
        self.last = d if d else self.last
        a = max(self.a0 / (1 + self.crossings), self.a_min)
        self.gain *= float(np.exp(np.clip(-a * d, -self.max_step, self.max_step)))
        return self.gain


def remnant(t, v, freqs=(FREQS_X, FREQS_Y)):
    """Per axis (2,), the share of the variance of cursor velocities ``v`` (T, 2) at trial times
    ``t`` (s, after the ramp) that sines at that axis's target ``freqs`` don't explain."""
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    out = np.full(2, np.nan)
    for axis, f in enumerate(freqs):
        w = 2 * np.pi * np.asarray(f) * t[:, None]
        A = np.column_stack([np.cos(w), np.sin(w), np.ones(len(t))])
        r = v[:, axis] - A @ np.linalg.lstsq(A, v[:, axis], rcond=None)[0]
        var = v[:, axis].var()
        out[axis] = (r**2).mean() / var if var > 0 else np.nan
    return out


def null_space_share(Z, D):
    """Share of the energy of activity ``Z`` (T, C: the decoder's features of each channel above
    rest) that decoder ``D`` (2, lags * C + 1) maps to no velocity, its lags summed:
    co-contraction and task-irrelevant activity."""
    Z = np.asarray(Z, dtype=float)
    W = np.asarray(D)[:, :-1].reshape(2, -1, Z.shape[1]).sum(axis=1)
    null = Z - Z @ (np.linalg.pinv(W) @ W)  # the projection is symmetric
    total = (Z**2).sum()
    return float((null**2).sum() / total) if total > 0 else np.nan


class Envelope:
    """Causal EMG linear envelope (mV) for online control.

    ``band`` with the mains notches (``clvhd.filters.emg_band``), rectification, 40 Hz low-pass.
    ``frame()`` is the moving average down to the display rate: the mean envelope of the
    samples added since the previous call. (Averaging by arrival rather than by sample time
    loses nothing: the aligned stream delivers samples tens of ms late.) The paper's band
    was 40-130 Hz (its Quattrocento's 10-130 Hz, then a 40 Hz high-pass); ``BAND`` keeps
    more of the dry electrodes' EMG power, and decodes as well on recorded calibrations.
    """

    def __init__(self, fs, n_channels, powerline=50.0, band=BAND):
        from scipy.signal import butter

        self.band = CausalFilter(emg_band(fs, band, powerline))
        self.smooth = butter(4, 40, btype="low", fs=fs, output="sos")
        self.zi_smooth = np.zeros((self.smooth.shape[0], 2, n_channels))
        self.sum = np.zeros(n_channels)
        self.count = 0
        self.value = None  # last frame's envelope

    def add(self, data):
        """Filter new samples (``data`` in V, NaN where missing) and add their envelope."""
        from scipy.signal import sosfilt

        _, y = self.band(data * 1e3)  # mV
        if len(y) == 0:
            return
        y, self.zi_smooth = sosfilt(self.smooth, np.abs(y), axis=0, zi=self.zi_smooth)
        self.sum += y.sum(axis=0)
        self.count += len(y)

    def frame(self):
        """Mean envelope (mV) of the samples added since the last call: the previous value when
        none was added, None before the first sample."""
        if self.count:
            self.value = self.sum / self.count
            self.sum = np.zeros_like(self.sum)
            self.count = 0
        return None if self.value is None else self.value.copy()


class Cursor:
    """Velocity-controlled cursor (cm from the screen centre), clamped to the screen.

    Held at an edge for ``STUCK_S`` s, it is reset to the centre (``step`` then returns True).
    """

    def __init__(self, half_size, radius):
        self.limit = np.asarray(half_size, dtype=float) - radius
        self.y = np.zeros(2)
        self.stuck_since = None

    def step(self, v, dt, t):
        self.y = np.clip(self.y + v * dt, -self.limit, self.limit)
        if np.any(np.abs(self.y) >= self.limit):
            if self.stuck_since is None:
                self.stuck_since = t
            elif t - self.stuck_since > STUCK_S:
                self.y = np.zeros(2)
                self.stuck_since = None
                return True
        else:
            self.stuck_since = None
        return False
