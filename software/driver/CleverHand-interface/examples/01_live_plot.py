"""Live plot of every module, 20-450 Hz band-pass filtered with notches at 50 Hz and its harmonics,
with electrode contact status (per channel and in total), LED buttons, and a record button that
saves the EMG to a BIDS dataset (see clvhd/bids.py).

    python 01_live_plot.py [--port PORT] [--modules 13] [--window 5]
        [--bids-root ~/Documents/CleverHand] [--subject 01] [--session 01] [--task rest]
        [--muscle 0=ECR] [--age 30 --sex F --handedness right]

Subject, session and task can also be typed in the window; each recording takes the next run.
"""
import argparse

from pyqtgraph.Qt import QtCore, QtWidgets

import clvhd
from clvhd import ui

p = argparse.ArgumentParser()
ui.add_device_arguments(p)
p.add_argument("--window", type=float, default=5, help="time window shown (s)")
p.add_argument("--duration", type=float, default=0, help="close after this many seconds (0: never)")
ui.add_record_arguments(p)
args = p.parse_args()

with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    with wb.stream() as stream:
        app = ui.application()
        window = QtWidgets.QWidget()
        window.setWindowTitle("CleverHand EMG")
        ui.style(window)
        panel = ui.EmgPanel(wb, stream, args.window, args.powerline)
        recorder = ui.RecordBar(panel, wb, args)
        layout = QtWidgets.QVBoxLayout(window)
        layout.addWidget(recorder)
        layout.addWidget(panel)
        window.resize(1200, 900)
        window.show()
        if args.duration:
            QtCore.QTimer.singleShot(int(args.duration * 1000), app.quit)
        app.exec()
        recorder.finish()
