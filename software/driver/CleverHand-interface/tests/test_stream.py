"""The aligned stream (``clvhd.stream``): the clock fit and frame indexing on synthetic
timestamps, and end-to-end against the simulated Teensy."""

import time

import numpy as np
import pytest

import clvhd
from clvhd.stream import SETTLE_SAMPLES, LinearClock, ModuleAligner


def record(port, seconds):
    """Record ``seconds`` s from the band on ``port``: (t, data, per-module stats, crc errors)."""
    with clvhd.Wristband(port, expected_modules=13) as wb:
        wb.configure()
        with wb.stream() as s:
            chunks = []
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                _, t_device, data = s.read(0.02)
                chunks.append((t_device, data))
            stats, crc = s.stats(), s.crc_errors
    t_device = np.concatenate([c[0] for c in chunks])
    data = np.concatenate([c[1] for c in chunks])
    return t_device, data, stats, crc


class TestLinearClock:
    def test_fits_a_line(self):
        clock = LinearClock(1 - 1e-5)
        x = np.arange(100000)
        y = 1234.5 + 468.75 * x + np.random.default_rng(1).uniform(0, 100, len(x))
        for i in range(0, len(x), 40):
            clock.update(x[i : i + 40], y[i : i + 40])
        assert abs(clock.slope() - 468.75) < 1e-3
        assert abs(clock.inverse(clock.predict(5000.25)) - 5000.25) < 1e-6


class TestModule:
    def test_settling_samples_are_dropped(self):
        # the first samples after START_CONV swing from -550 mV while the filters settle
        T = 1e6 / (200000 / 96)
        mod = ModuleAligner(T, 1 - 1 / (20 * 2083), 4000)
        n = 400
        ts = (1e6 + np.arange(n) * T).astype(np.uint32)
        v = np.zeros((n, 3))
        v[:SETTLE_SAMPLES] = -0.55
        for i in range(0, n, 5):  # batches shorter than the settling
            mod.add(ts[i : i + 5], v[i : i + 5])
        assert mod.samples == n - SETTLE_SAMPLES and mod.dropped == 0
        assert np.all(mod.v == 0)

    def test_indexing_tolerates_poll_jitter(self):
        # the Teensy stamps a sample when it polls it: up to a whole poll round
        # (~350 us at 4 MHz) after the sample, against a 469 us period
        # Drops and duplicates (80 us late) are injected, some among the first frames.
        T = 1e6 / (204800 / 96)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = 30 * 2133
            t = 1e6 + np.arange(n) * T * (1 + 30e-6) + rng.uniform(0, 350, n)
            drop = np.zeros(n, dtype=bool)
            drop[[40, 41, 100]] = True
            drop[rng.choice(n, 30, replace=False)] = True
            dup = np.flatnonzero(~drop)[[60, 200, 5000]]
            t = np.sort(np.concatenate([t[~drop], t[dup] + 80]))
            ts = (t.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
            v = np.zeros((len(t), 3))
            mod = ModuleAligner(T, 1 - 1 / (20 * 2133), 4000)
            mod.to_settle = 0  # count every frame
            for i in range(0, len(t), 40):
                mod.add(ts[i : i + 40], v[i : i + 40])
            assert (mod.samples, mod.dropped, mod.duplicates) == (n - drop.sum(), drop.sum(), 3)


class TestStreamAgainstSim:
    def test_stream_is_complete_and_aligned(self, sim):
        # 5 s spans the simulated micros() wraparound
        t_device, data, stats, crc = record(sim.port, 5)
        rate = 200000 / 96
        assert crc == 0
        for st in stats:
            assert abs(st["samples"] - 5 * rate) < 0.03 * 5 * rate
            assert st["dropped"] == 0 and st["duplicates"] == 0
        # no gap at the wrap
        assert np.all(np.diff(t_device) > 0) and np.max(np.diff(t_device)) < 1.5 / rate
        # the fitted rates match the simulated clock offsets
        ppm = np.array(sim.ppm())
        expected = ((1 + ppm * 1e-6) / (1 + ppm[0] * 1e-6) - 1) * 1e6
        assert np.allclose([st["ppm"] for st in stats], expected, atol=3)
        # every module samples the same signal: aligned, they agree
        ok = ~np.isnan(data).any(axis=1)
        assert ok.mean() > 0.99
        ch1 = data[ok][2000:, 0::3]
        mismatch = np.sqrt(
            np.mean((ch1[:, 1:] - ch1[:, [0]]) ** 2, axis=0) / np.mean(ch1[:, 0] ** 2)
        )
        assert mismatch.max() < 0.01

    def test_faults_are_detected(self, faulty_sim):
        _, _, stats, crc = record(faulty_sim.port, 5)
        injected = faulty_sim.injected()
        assert sum(st["duplicates"] for st in stats) == injected["dup"]
        assert abs(crc - injected["crc"]) <= 1
        # a frame dropped by a bad CRC is a missing sample
        assert abs(sum(st["dropped"] for st in stats) - (injected["drop"] + injected["crc"])) <= 2

    def test_commands_while_streaming(self, sim):
        with clvhd.Wristband(sim.port, expected_modules=13) as wb:
            wb.configure()
            with wb.stream() as s:
                s.read(0.2)
                wb.set_led(3, 1, (120, 0, 16))  # 120 is 'x': must not stop the stream
                with pytest.raises(clvhd.ControllerError, match="stop the stream"):
                    wb.read_errors()
                n = sum(len(s.read(0.1)[1]) for _ in range(5))
            assert n > 0.8 * 0.5 * 2083
            assert wb.read_errors() == [""] * 13  # usable again once stopped
        assert [4, 2, 120, 0, 16] in sim.leds()

    def test_slow_reader_loses_nothing(self, sim):
        # a consumer that only reads every 100 ms (e.g. a GUI): the background
        # reader keeps draining the port, so no module falls behind
        with clvhd.Wristband(sim.port, expected_modules=13) as wb:
            wb.configure()
            with wb.stream() as s:
                n = 0
                t0 = time.monotonic()
                while time.monotonic() - t0 < 3:
                    time.sleep(0.1)
                    n += len(s.read(0)[1])
                elapsed = time.monotonic() - t0
                stats = s.stats()
        rate = 200000 / 96
        assert n > 0.9 * elapsed * rate
        assert min(st["samples"] for st in stats) > 0.95 * max(st["samples"] for st in stats)
        assert all(st["dropped"] == 0 for st in stats)
