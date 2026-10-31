"""Shared fixtures for the test suite: simulated Teensys (the ``_teensy_sim.py`` script on a
pseudo-terminal), and the examples imported as modules."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SIM = Path(__file__).with_name("_teensy_sim.py")
EXAMPLES = Path(__file__).parents[1] / "examples"


class Sim:
    """A simulated Teensy; its stdout goes to ``log`` (``ppm()``, ``injected()``, ``leds()`` read it)."""

    def __init__(self, folder, faults, boot_s=0):
        folder.mkdir(parents=True, exist_ok=True)
        self.path_file = folder / "pty.txt"
        self.log = folder / "sim.log"
        env = dict(os.environ, FAULTS="1" if faults else "0", BOOT_S=str(boot_s))
        # the log file stays open for the sim's lifetime: pass its fd to Popen
        self._log_fd = os.open(self.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        self.proc = subprocess.Popen(
            [sys.executable, str(SIM), str(self.path_file)],
            env=env,
            stdout=self._log_fd,
            stderr=subprocess.STDOUT,
        )
        for _ in range(100):
            if self.path_file.exists() and self.path_file.read_text():
                break
            time.sleep(0.05)
        self.port = self.path_file.read_text()

    def ppm(self):
        line = next(x for x in self.log.read_text().splitlines() if x.startswith("sim on"))
        return json.loads(line.split("ppm ", 1)[1])

    def injected(self):
        line = [x for x in self.log.read_text().splitlines() if x.startswith("stream stopped")][-1]
        return json.loads(line.split("stopped ", 1)[1])

    def leds(self):
        return [
            json.loads(x.split(" ", 1)[1])
            for x in self.log.read_text().splitlines()
            if x.startswith("led ")
        ]

    def close(self):
        self.proc.kill()
        self.proc.wait()
        os.close(self._log_fd)


@pytest.fixture
def make_sim(tmp_path):
    """Factory: ``make_sim(faults=..., boot_s=...)``; every Sim is closed at the end of the test."""
    sims: list[Sim] = []

    def make(faults=False, boot_s=0):
        s = Sim(tmp_path / f"sim{len(sims)}", faults, boot_s)
        sims.append(s)
        return s

    yield make
    for s in sims:
        s.close()


@pytest.fixture
def sim(make_sim):
    return make_sim()


@pytest.fixture
def faulty_sim(make_sim):
    return make_sim(faults=True)


@pytest.fixture
def load_example(monkeypatch):
    """Factory: ``load_example("03_cursor_centerout.py", *argv)``, the example's module with
    ``argv`` parsed (its ``__main__`` block, which opens the wristband, does not run)."""

    def load(name, *argv):
        pytest.importorskip("pyqtgraph")
        monkeypatch.setattr(sys, "argv", [name, *argv])
        spec = importlib.util.spec_from_file_location(f"example_{name[:2]}", EXAMPLES / name)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
