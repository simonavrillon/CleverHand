"""Driver tests against the simulated Teensy (tests/teensy_sim.py).

    uv run pytest
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import clvhd
from clvhd.protocol import crc8_rows
from clvhd import ads1293
from clvhd.stream import LinearClock, _Module

SIM = Path(__file__).with_name("teensy_sim.py")


class Sim:
    def __init__(self, tmp_path, faults, boot_s=0):
        path_file = tmp_path / "pty.txt"
        self.log = tmp_path / "sim.log"
        env = dict(os.environ, FAULTS="1" if faults else "0", BOOT_S=str(boot_s))
        self.proc = subprocess.Popen([sys.executable, str(SIM), str(path_file)], env=env,
                                     stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        for _ in range(100):
            if path_file.exists() and path_file.read_text():
                break
            time.sleep(0.05)
        self.port = path_file.read_text()

    def ppm(self):
        line = next(l for l in self.log.read_text().splitlines() if l.startswith("sim on"))
        return json.loads(line.split("ppm ", 1)[1])

    def injected(self):
        line = [l for l in self.log.read_text().splitlines() if l.startswith("stream stopped")][-1]
        return json.loads(line.split("stopped ", 1)[1])

    def leds(self):
        return [json.loads(l.split(" ", 1)[1]) for l in self.log.read_text().splitlines()
                if l.startswith("led ")]

    def close(self):
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def sim(tmp_path):
    s = Sim(tmp_path, faults=False)
    yield s
    s.close()


@pytest.fixture
def faulty_sim(tmp_path):
    s = Sim(tmp_path, faults=True)
    yield s
    s.close()


def record(port, seconds):
    with clvhd.Wristband(port, expected_modules=12) as wb:
        wb.configure()
        with wb.stream() as s:
            chunks = []
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                _, t, data = s.read(0.02)
                chunks.append((t, data))
            stats, crc = s.stats(), s.crc_errors
    t = np.concatenate([c[0] for c in chunks])
    data = np.concatenate([c[1] for c in chunks])
    return t, data, stats, crc


def test_crc8_matches_bytewise():
    rows = np.random.default_rng(0).integers(0, 256, (50, 20), dtype=np.uint8)
    for row, crc in zip(rows, crc8_rows(rows)):
        c = 0
        for b in row:
            c ^= int(b)
            for _ in range(8):
                c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        assert c == crc


def test_linear_clock_fits_a_line():
    clock = LinearClock(1 - 1e-5)
    x = np.arange(100000)
    y = 1234.5 + 468.75 * x + np.random.default_rng(1).uniform(0, 100, len(x))
    for i in range(0, len(x), 40):
        clock.update(x[i:i + 40], y[i:i + 40])
    assert abs(clock.slope() - 468.75) < 1e-3
    assert abs(clock.inverse(clock.predict(5000.25)) - 5000.25) < 1e-6


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        clvhd.Config(R2=7).validate()
    with pytest.raises(ValueError):
        clvhd.Config(enable=[False, False, False]).validate()


def test_config_rejects_missing_rld_module():
    # checked before anything is sent to the controller
    with pytest.raises(ValueError, match="rld_module"):
        ads1293.configure(None, 12, clvhd.Config(rld_module=12))
    with pytest.raises(ValueError, match="rld_input"):
        clvhd.Config(rld_input=7).validate()


def test_indexing_tolerates_poll_jitter():
    # the Teensy stamps a sample when it polls it: up to a whole poll round
    # (~350 us at 4 MHz, 12 modules) after the sample, against a 469 us period
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
        mod = _Module(T, 1 - 1 / (20 * 2133), 4000)
        for i in range(0, len(t), 40):
            mod.add(ts[i:i + 40], v[i:i + 40])
        assert (mod.samples, mod.dropped, mod.duplicates) == (n - drop.sum(), drop.sum(), 3)


def test_connect_while_controller_boots(tmp_path):
    # right after a power-up the Teensy answers only after its 1 s setup()
    s = Sim(tmp_path, faults=False, boot_s=1)
    try:
        with clvhd.Wristband(s.port, expected_modules=12) as wb:
            assert wb.firmware == (4, 1) and wb.ctrl.mirror()
    finally:
        s.close()


def test_connect_and_configure(sim):
    with clvhd.Wristband(sim.port, expected_modules=12) as wb:
        assert wb.nb_modules == 12
        assert wb.ctrl.mirror()
        cfg = wb.configure()
        assert abs(cfg.odr() - 2083.33) < 0.01
        assert wb.read_errors() == [""] * 12
        assert wb.set_spi_clock(8_000_000) == 8_000_000


def test_wrong_module_count_raises(sim):
    with pytest.raises(clvhd.ControllerError, match="expected 11"):
        clvhd.Wristband(sim.port, expected_modules=11)


def test_stream_is_complete_and_aligned(sim):
    # 5 s spans the simulated micros() wraparound
    t, data, stats, crc = record(sim.port, 5)
    rate = 200000 / 96
    assert crc == 0
    for st in stats:
        assert abs(st["samples"] - 5 * rate) < 0.03 * 5 * rate
        assert st["dropped"] == 0 and st["duplicates"] == 0
    assert np.all(np.diff(t) > 0) and np.max(np.diff(t)) < 1.5 / rate  # no gap at the wrap
    # the fitted rates match the simulated clock offsets
    ppm = np.array(sim.ppm())
    expected = ((1 + ppm * 1e-6) / (1 + ppm[0] * 1e-6) - 1) * 1e6
    assert np.allclose([st["ppm"] for st in stats], expected, atol=3)
    # every module samples the same signal: aligned, they agree
    ok = ~np.isnan(data).any(axis=1)
    assert ok.mean() > 0.99
    ch1 = data[ok][2000:, 0::3]
    mismatch = np.sqrt(np.mean((ch1[:, 1:] - ch1[:, [0]]) ** 2, axis=0) / np.mean(ch1[:, 0] ** 2))
    assert mismatch.max() < 0.01


def test_faults_are_detected(faulty_sim):
    t, data, stats, crc = record(faulty_sim.port, 5)
    injected = faulty_sim.injected()
    assert sum(st["duplicates"] for st in stats) == injected["dup"]
    assert abs(crc - injected["crc"]) <= 1
    # a frame dropped by a bad CRC is a missing sample
    assert abs(sum(st["dropped"] for st in stats) - (injected["drop"] + injected["crc"])) <= 2


def test_commands_while_streaming(sim):
    with clvhd.Wristband(sim.port, expected_modules=12) as wb:
        wb.configure()
        with wb.stream() as s:
            s.read(0.2)
            wb.set_led(3, 1, (120, 0, 16))  # 120 is 'x': must not stop the stream
            with pytest.raises(clvhd.ControllerError, match="stop the stream"):
                wb.read_errors()
            n = sum(len(s.read(0.1)[1]) for _ in range(5))
        assert n > 0.8 * 0.5 * 2083
        assert wb.read_errors() == [""] * 12  # usable again once stopped
    assert [4, 2, 120, 0, 16] in sim.leds()


def test_slow_reader_loses_nothing(sim):
    # a consumer that only reads every 100 ms (e.g. a GUI): the background
    # reader keeps draining the port, so no module falls behind
    with clvhd.Wristband(sim.port, expected_modules=12) as wb:
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
