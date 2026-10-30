"""The example windows' shared pieces (``clvhd.ui``)."""

import argparse
import os
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clvhd import ui


def test_record_refuses_while_saving(tmp_path):
    """The record button is disabled while saving, but setChecked would still start a recording."""
    app = ui.application()
    p = argparse.ArgumentParser()
    ui.add_record_arguments(p)
    args = p.parse_args(["--bids-root", str(tmp_path), "--subject", "01", "--task", "rest"])
    bar = ui.RecordBar(SimpleNamespace(listeners=[], powerline=50.0), None, args)
    done = threading.Event()
    bar.saver = threading.Thread(target=done.wait)  # the last recording still saving
    bar.saver.start()
    try:
        assert bar.record(True) == "busy saving the last recording"
        assert not bar.recording and not bar.button.isChecked()
    finally:
        done.set()
        bar.saver.join()
    assert app is not None
