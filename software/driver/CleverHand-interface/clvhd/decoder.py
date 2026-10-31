"""The cursor decoder of the examples: a Wiener filter of the EMG envelope.

``examples/03_cursor_centerout.py`` fits it on a cued calibration and uses it fixed, with an
output stage; ``examples/04_cursor_coadapt.py`` starts from that fit and refits its weights
every 20 s (``clvhd.coadapt``). Features: each channel's envelope u as log(u + floor), z-scored,
over the last ``LAGS`` frames, plus a constant; ``WienerFilter.fit`` picks the ridge penalty by
cross-validation over the cues. Design notes: docs/experiments.md.

Units: velocities in cm/s, envelopes in mV.
"""

import numpy as np

LAGS = 6  # history: 100 ms at 60 Hz
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # relative penalties tried, times the mean feature energy
FOLDS = 5
N_DIRECTIONS = 8  # of the direction gains: the center-out targets'
GAIN_RANGE = (0.5, 5.0)  # direction gains kept within
DEADZONE_PERCENTILE = (
    75  # of the decoded speed at rest on the calibration (95: about half its speed)
)


def lagged(Z, lags):
    """(T, lags * C) features: each row holds ``Z`` (T, C) at t, t - 1, ..., zeros before the start."""
    T, C = Z.shape
    X = np.zeros((T, lags * C))
    for k in range(lags):
        X[k:, k * C : (k + 1) * C] = Z[: T - k]
    return X


def _ridge(X, V, ridge):
    """Weights (p + 1, 2) of V ~ [X 1] W, every weight but the bias penalised by
    ``ridge`` times the mean feature energy."""
    A = np.column_stack([X, np.ones(len(X))])
    G = A.T @ A
    pen = np.full(A.shape[1], ridge * np.trace(G[:-1, :-1]) / max(1, X.shape[1]))
    pen[-1] = 0
    return np.linalg.solve(G + np.diag(pen), A.T @ V)


def r2(V, P):
    """R^2 of predictions ``P`` of ``V``, per column."""
    return 1 - ((V - P) ** 2).sum(axis=0) / ((V - V.mean(axis=0)) ** 2).sum(axis=0)


def direction_gains(P, V, n=N_DIRECTIONS):
    """Gains (n,) of the n target directions (0 to the right, then counter-clockwise): on the
    frames whose label points to each, the mean decoded velocity along it times its gain is
    the mean label speed (1 when the decoder points the wrong way on average, or none was
    cued; within ``GAIN_RANGE``)."""
    g = np.ones(n)
    moving = V.any(axis=1)
    k = np.round(np.arctan2(V[:, 1], V[:, 0]) / (2 * np.pi / n)).astype(int) % n
    for d in range(n):
        cued = moving & (k == d)
        a = 2 * np.pi * d / n
        decoded = (P[cued] @ [np.cos(a), np.sin(a)]).mean() if cued.any() else 0.0
        if decoded > 0:
            g[d] = np.clip(np.linalg.norm(V[cued], axis=1).mean() / decoded, *GAIN_RANGE)
    return g


class WienerFilter:
    """Linear decoder of the velocity from the envelope's last ``lags`` frames, then the output
    stage (direction ``gains``, ``deadzone``, speed curve of ``power`` anchored at ``speed``).

    ``fit`` on the calibration, then ``step(u)`` once per frame online (it keeps the history),
    or ``predict(U)`` on a whole recording; ``push(u)`` gives the frame's features alone, for
    weights of one's own (04's). ``save`` / ``load``: an .npz.
    """

    def __init__(
        self,
        weights,
        mean,
        sd,
        lags=LAGS,
        ridge=None,
        cv_r2=None,
        gains=None,
        speed=np.nan,
        deadzone=0.0,
        power=1.0,
        floor=None,
    ):
        self.weights = np.asarray(weights, dtype=float)  # (lags * C + 1, 2), the bias last
        self.mean, self.sd = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
        self.floor = None if floor is None else np.asarray(floor, dtype=float)
        self.lags, self.ridge = int(lags), ridge
        self.cv_r2 = None if cv_r2 is None else np.asarray(cv_r2, dtype=float)
        self.gains = np.ones(N_DIRECTIONS) if gains is None else np.asarray(gains, dtype=float)
        self.speed, self.deadzone, self.power = float(speed), float(deadzone), float(power)
        self.reset()

    @staticmethod
    def _clip(U):
        """Envelopes ``U`` without NaN or negative values (the low-pass rings below 0)."""
        return np.clip(np.nan_to_num(np.asarray(U, dtype=float)), 0, None)

    def _z(self, U):
        E = self._clip(U)
        return ((E if self.floor is None else np.log(E + self.floor)) - self.mean) / self.sd

    @classmethod
    def fit(cls, U, V, groups, in_fit=None, lags=LAGS, ridges=RIDGES, folds=FOLDS):
        """Fit on envelopes ``U`` (T, C) and velocities ``V`` (T, 2), on the frames of ``in_fit``.

        The floor is each channel's median envelope on the frames of ``in_fit`` labelled rest
        (V = 0). The ridge is the one of ``ridges`` with the best mean cross-validated R^2,
        ``folds`` folds of consecutive ``groups`` (the cue of each frame); the history spans
        the whole recording, so the lags of the first frames of a fold are its neighbour's
        frames. The output stage is set on that ridge's cross-validated predictions: what a
        new reach sees.
        """
        V, groups = np.asarray(V, dtype=float), np.asarray(groups)
        in_fit = np.ones(len(V), bool) if in_fit is None else np.asarray(in_fit, bool)
        E = cls._clip(U)
        rest = in_fit & ~V.any(axis=1)
        floor = np.median(E[rest if rest.any() else in_fit], axis=0)
        floor = np.maximum(floor, 1e-6 * max(floor.max(), 1e-30))
        F = np.log(E + floor)
        mean = F[in_fit].mean(axis=0)
        sd = F[in_fit].std(axis=0)
        sd = np.where(sd > 1e-9 * max(sd.max(), 1e-30), sd, 1.0)  # a constant channel stays 0
        X = lagged((F - mean) / sd, lags)
        X, V, groups = X[in_fit], V[in_fit], groups[in_fit]
        cues = np.unique(groups)
        chunks = np.array_split(cues, min(folds, len(cues)))
        fold_of = np.zeros(len(groups), int)
        for fold, chunk in enumerate(chunks):
            fold_of[np.isin(groups, chunk)] = fold
        scores, predictions = [], []
        for ridge in ridges:
            P = np.zeros_like(V)
            for fold in range(len(chunks)):
                test = fold_of == fold
                W = _ridge(X[~test], V[~test], ridge)
                P[test] = np.column_stack([X[test], np.ones(test.sum())]) @ W
            scores.append(r2(V, P))
            predictions.append(P)
        best = int(np.argmax([s.mean() for s in scores]))
        wf = cls(
            _ridge(X, V, ridges[best]), mean, sd, lags, ridges[best], scores[best], floor=floor
        )
        wf.set_output(predictions[best], V)
        return wf

    def set_output(self, P, V):
        """Gains, speed and dead-zone from decoded velocities ``P`` (T, 2, before the output
        stage) of labels ``V`` (T, 2): the calibration's."""
        self.gains = direction_gains(P, V)
        moving = V.any(axis=1)
        self.speed = float(np.median(np.linalg.norm(V[moving], axis=1))) if moving.any() else np.nan
        rest = np.linalg.norm(self._gain(P[~moving]), axis=1)
        dz = np.percentile(rest, DEADZONE_PERCENTILE) if len(rest) else 0.0
        self.deadzone = float(min(dz, 0.5 * self.speed)) if np.isfinite(self.speed) else 0.0

    def _gain(self, v):
        """``v`` times the gain of its direction, interpolated between the two nearest targets'."""
        n = len(self.gains)
        pos = np.arctan2(v[..., 1], v[..., 0]) / (2 * np.pi / n) % n
        i = np.floor(pos).astype(int)
        g = self.gains[i % n] * (1 - (pos - i)) + self.gains[(i + 1) % n] * (pos - i)
        return v * np.asarray(g)[..., None]

    def output(self, v):
        """The output stage on linear velocities ``v`` (2,) or (T, 2)."""
        v = self._gain(np.asarray(v, dtype=float))
        if not np.isfinite(self.speed) or (self.deadzone == 0 and self.power == 1):
            return v
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        out = (
            self.speed
            * (np.clip(s - self.deadzone, 0, None) / (self.speed - self.deadzone)) ** self.power
        )
        return v * np.divide(out, s, out=np.zeros_like(s), where=s > 0)

    def reset(self):
        """Forget the history: the next frames see rest (the calibration mean) before them."""
        self.history = np.zeros((self.lags, len(self.mean)))

    def push(self, u):
        """Add this frame's envelope ``u`` (C,) to the history; its features (lags * C + 1,),
        the constant last."""
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = self._z(u)
        return np.append(self.history.ravel(), 1.0)

    def activity(self, U):
        """Features (T, C) of envelopes ``U`` (T, C) above rest (the floor), 0 at or below it."""
        rest = self._z(self.floor if self.floor is not None else np.zeros_like(self.mean))
        return np.clip(self._z(U) - rest, 0, None)

    def step(self, u):
        """Velocity (2,) for this frame's envelope ``u`` (C,)."""
        return self.output(self.push(u) @ self.weights)

    def predict(self, U, linear=False):
        """Velocities (T, 2) of envelopes ``U`` (T, C), from rest; ``linear``: before the output
        stage."""
        X = lagged(self._z(U), self.lags)
        v = np.column_stack([X, np.ones(len(X))]) @ self.weights
        return v if linear else self.output(v)

    def save(self, path, **extra):
        np.savez(
            path,
            weights=self.weights,
            mean=self.mean,
            sd=self.sd,
            lags=self.lags,
            ridge=np.nan if self.ridge is None else self.ridge,
            cv_r2=np.full(2, np.nan) if self.cv_r2 is None else self.cv_r2,
            gains=self.gains,
            speed=self.speed,
            deadzone=self.deadzone,
            power=self.power,
            floor=np.full(len(self.mean), np.nan) if self.floor is None else self.floor,
            **extra,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(
                f["weights"],
                f["mean"],
                f["sd"],
                int(f["lags"]),
                float(f["ridge"]),
                f["cv_r2"],
                f["gains"],
                float(f["speed"]),
                float(f["deadzone"]),
                float(f["power"]),
                f["floor"],
            )


def channel_indices(path, labels):
    """Indices in ``labels`` of the channels the decoder saved at ``path`` was fitted on."""
    with np.load(path) as f:
        names = [str(c) for c in f["decoder_channels"]]
    missing = [n for n in names if n not in labels]
    if missing:
        raise ValueError(f"{path} uses channels this wristband does not have: {', '.join(missing)}")
    return [labels.index(n) for n in names]
