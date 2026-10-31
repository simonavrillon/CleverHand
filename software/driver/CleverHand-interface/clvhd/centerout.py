"""Center-out myoelectric cursor control with a fixed Wiener filter.

The pieces of the experiment, without the display (``examples/03_cursor_centerout.py``):

- ``targets`` / ``block_order``: the 8 targets on a circle, each once per block in a random order.
- ``Calibration``: the cued block the decoder is fitted on. Each cue rests ``REST_S`` s at the
  centre, then lights a target while a guide cursor travels to it in ``CUE_S`` s; the
  participant contracts as if pushing the cursor there, and relaxes when it is back. The
  velocity labels are the guide's: ``distance / CUE_S`` towards the target during the cue, 0 at
  rest. Frames within ``REACTION_S`` of a cue's start or end are left out of the fit.
- The decoder, fitted on the calibration and fixed afterwards: ``clvhd.decoder.WienerFilter``.
- ``reach_metrics``: success, time to target, acquisition time, path efficiency and target
  entries of one reach.

Units: positions in cm from the screen centre, velocities in cm/s, envelopes in mV.
"""

import numpy as np

N_TARGETS = 8
REST_S = 2.0  # calibration: rest at the centre before each cue
CUE_S = 2.0  # calibration: the guide's travel to the target
REACTION_S = 0.3  # calibration frames this close to a cue's start or end are not fitted
READY_S = 1.0  # online: the cursor held at the centre with the target shown, before the go
REACH_S = 10.0  # online: time allowed to acquire the target
HOLD_S = 0.5  # online: time in the target that acquires it
FEEDBACK_S = 0.5  # online: the target's colour shows the outcome


def targets(distance, n=N_TARGETS):
    """(n, 2) target centres on a circle of radius ``distance``, the first to the right, then
    counter-clockwise."""
    a = 2 * np.pi * np.arange(n) / n
    return distance * np.column_stack([np.cos(a), np.sin(a)])


def block_order(rng, reps, n=N_TARGETS):
    """Target indices: ``reps`` blocks, each target once per block in a random order."""
    return np.concatenate([rng.permutation(n) for _ in range(reps)]) if reps else np.zeros(0, int)


class Calibration:
    """The cued calibration of ``order`` (target indices) on ``positions`` ((n, 2) cm), a final
    rest closing it. ``at(t)`` gives the cue at time ``t`` from its start."""

    def __init__(self, order, positions, rest=REST_S, cue=CUE_S):
        self.order, self.positions = np.asarray(order), np.asarray(positions, dtype=float)
        self.rest, self.cue = rest, cue
        self.period = rest + cue
        self.duration = len(self.order) * self.period + rest

    def at(self, t):
        """dict: ``cue`` (index in ``order``, its rest included; ``len(order)`` for the final
        rest), ``target`` (target index, -1 at rest), ``guide`` (guide cursor, cm), ``velocity``
        (label, cm/s) and ``fit`` (False near a transition)."""
        k = min(int(t // self.period), len(self.order))
        within = t - k * self.period
        moving = k < len(self.order) and within >= self.rest
        near = min(within, abs(within - self.rest)) < REACTION_S
        if not moving:
            return {
                "cue": k,
                "target": -1,
                "guide": np.zeros(2),
                "velocity": np.zeros(2),
                "fit": not near,
            }
        target = int(self.order[k])
        goal = self.positions[target]
        return {
            "cue": k,
            "target": target,
            "guide": goal * min(1.0, (within - self.rest) / self.cue),
            "velocity": goal / self.cue,
            "fit": not near,
        }


def reach_metrics(t, cursor, target, radius, hold=HOLD_S):
    """One reach from the go (``t`` (T,) s from the go, ``cursor`` (T, 2) cm, ``target`` (2,) cm,
    acquired when within ``radius`` of it for ``hold`` s).

    dict: ``success``; ``time_to_target`` (first entry, s), ``acquisition_time`` (end of the
    hold, s), NaN when never reached / acquired; ``path_efficiency`` (straight distance from the
    start to the target's edge over the path length to the entry that was held, NaN when not
    acquired); ``entries`` (times the cursor entered the target).
    """
    t, cursor = np.asarray(t, dtype=float), np.asarray(cursor, dtype=float)
    inside = np.linalg.norm(cursor - target, axis=1) <= radius
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    out = {
        "success": False,
        "time_to_target": t[starts[0]] if len(starts) else np.nan,
        "acquisition_time": np.nan,
        "path_efficiency": np.nan,
        "entries": len(starts),
    }
    for s in starts:
        after = np.flatnonzero(~inside[s:])
        end = s + after[0] if len(after) else len(t)
        done = np.flatnonzero(t[s:end] - t[s] >= hold - 1e-9)
        if len(done):
            out["success"], out["acquisition_time"] = True, t[s + done[0]]
            path = np.linalg.norm(np.diff(cursor[: s + 1], axis=0), axis=1).sum()
            straight = max(0.0, np.linalg.norm(target - cursor[0]) - radius)
            out["path_efficiency"] = straight / path if path > 0 else np.nan
            break
    return out
