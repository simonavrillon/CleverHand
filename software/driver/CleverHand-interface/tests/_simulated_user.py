"""A simulated user of the cursor task, for the gain staircase tests (crossover model).

The user is a gain with a delay on the error they see (McRuer's crossover model for a velocity
cursor): they re-tune their gain towards ``crossover / g`` after each change of the decoder
gain g, but not below the gain their EMG resolution allows. Their command carries an additive
noise floor and noise proportional to it; the cursor moves at g times the noisy command.
"""

import numpy as np

from clvhd import coadapt
from clvhd.coadapt import FRAME_RATE


class SimulatedUser:
    def __init__(
        self,
        rng,
        crossover=3.0,  # rad/s, the loop bandwidth the user keeps
        delay=0.25,  # s, visuomotor plus processing
        adapt_s=3.0,  # s, time constant of the re-tuning after a gain change
        min_kp=1.5,  # 1/s, the resolution floor: command per cm of error
        noise=1.0,  # cm/s, sd of the additive command noise
        cv=0.2,  # noise proportional to the command
        noise_s=0.3,  # s, the noise's correlation time (the envelope's slow fluctuations)
    ):
        self.rng, self.crossover, self.adapt_s, self.min_kp = rng, crossover, adapt_s, min_kp
        self.noise, self.cv, self.noise_s = noise, cv, noise_s
        self.errors = np.zeros((max(1, round(delay * FRAME_RATE)), 2))  # a ring, at self.next
        self.next = 0
        self.kp = crossover  # their gain (command per cm of error), tuned to g = 1
        self.n = np.zeros(2)  # the low-passed unit noise

    def step(self, error, g, dt):
        """Command (2,) at this frame, from the ``error`` (target - cursor, cm) seen now."""
        seen = self.errors[self.next].copy()  # the error of ``delay`` ago
        self.errors[self.next] = error
        self.next = (self.next + 1) % len(self.errors)
        wanted = max(self.crossover / g, self.min_kp)
        self.kp += (wanted - self.kp) * dt / self.adapt_s
        rate = dt / self.noise_s
        self.n += -rate * self.n + np.sqrt(2 * rate) * self.rng.normal(size=2)  # unit variance
        a = self.kp * seen
        return a + self.n * (self.noise + self.cv * np.abs(a))


def run_trial(rng, staircase, user, duration=coadapt.TRIAL_S - coadapt.PROBE_S, half=(11, 8)):
    """A tracking trial: the staircase updates the gain every ``UPDATE_S`` on the window's mean
    error after ``SETTLE_S``. Returns the gains and each window's (error, effort)."""
    traj = coadapt.Trajectory(rng, 0.7 * np.asarray(half))
    cursor = coadapt.Cursor(half, 0.5)
    dt, gains, windows = 1 / FRAME_RATE, [staircase.gain], []
    window, next_update, changed = [], coadapt.UPDATE_S, 0.0
    for trial_t in np.arange(0, duration, dt):
        tau = traj.position(trial_t)
        a = user.step(tau - cursor.y, staircase.gain, dt)
        cursor.step(staircase.gain * a * traj.speed(trial_t), dt, trial_t)
        if trial_t > max(coadapt.RAMP_S, changed + coadapt.SETTLE_S):
            window.append((np.linalg.norm(tau - cursor.y), np.linalg.norm(a)))
        if trial_t >= next_update:
            error, effort = np.mean(window, axis=0)
            windows.append((error, effort))
            gains.append(staircase.update(error))
            window, next_update, changed = [], next_update + coadapt.UPDATE_S, trial_t
    return np.array(gains), np.array(windows)


class FixedGain:
    """A staircase that never moves."""

    def __init__(self, gain):
        self.gain = gain

    def update(self, error):
        return self.gain


def fixed_gain(rng, g, user, duration=120.0):
    """Mean (error, effort) at a fixed gain ``g``, after the ramp and the user's re-tuning."""
    _, windows = run_trial(rng, FixedGain(g), user, duration)
    return windows[1:].mean(axis=0)
