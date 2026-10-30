"""Live plot of every module, with channel toggles, scale, channel offset, optional band-pass and LED buttons.

    python 02_live_plot.py [--port PORT] [--modules 12] [--window 5]
"""
import argparse
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import butter, sosfilt

import clvhd

p = argparse.ArgumentParser()
p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
p.add_argument("--modules", type=int, default=12, help="expected number of modules")
p.add_argument("--window", type=float, default=5, help="time window shown (s)")
p.add_argument("--duration", type=float, default=0, help="close after this many seconds (0: never)")
args = p.parse_args()

COLORS = [(230, 60, 60), (60, 180, 60), (60, 120, 230)]


class LivePlot(QtWidgets.QWidget):
    def __init__(self, wb, stream):
        super().__init__()
        self.wb, self.stream = wb, stream
        self.nb = wb.nb_modules
        self.fs = stream.nominal_rate
        self.n = int(args.window * self.fs)
        self.t = np.full(self.n, np.nan)
        self.y = np.full((self.n, 3 * self.nb), np.nan)  # mV
        self.sos = self.zi = None
        self.leds_on = [False] * self.nb
        self.setWindowTitle("CleverHand EMG")

        # controls
        controls = QtWidgets.QHBoxLayout()
        self.channel_boxes = []
        for ch in range(3):
            box = QtWidgets.QCheckBox(f"ch{ch + 1}")
            box.setChecked(True)
            box.stateChanged.connect(self.update_visibility)
            controls.addWidget(box)
            self.channel_boxes.append(box)
        controls.addWidget(QtWidgets.QLabel("scale (mV):"))
        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(["0.05", "0.1", "0.5", "1", "2"])
        self.scale.setCurrentText("0.5")
        self.scale.currentTextChanged.connect(self.update_range)
        controls.addWidget(self.scale)
        self.offset_box = QtWidgets.QCheckBox("offset channels")  # ch1 one scale up, ch3 one down
        self.offset_box.setChecked(True)
        self.offset_box.stateChanged.connect(self.update_range)
        controls.addWidget(self.offset_box)
        self.filter_box = QtWidgets.QCheckBox("band-pass")
        self.filter_box.stateChanged.connect(self.update_filter)
        controls.addWidget(self.filter_box)
        self.low, self.high = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin, val in ((self.low, 20), (self.high, 450)):
            spin.setRange(0.5, 0.45 * self.fs)
            spin.setValue(val)
            spin.setSuffix(" Hz")
            spin.valueChanged.connect(self.update_filter)
            controls.addWidget(spin)
        controls.addStretch()

        # one plot per module, with a button to light up its LEDs
        self.graphs = pg.GraphicsLayoutWidget()
        self.plots, self.curves = [], []
        for m in range(self.nb):
            plot = self.graphs.addPlot(row=m, col=0)
            plot.setLabel("left", f"M{m}")
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.setDownsampling(auto=True, mode="peak")
            plot.setClipToView(True)
            plot.setMouseEnabled(x=False, y=False)
            if m < self.nb - 1:
                plot.hideAxis("bottom")
            self.plots.append(plot)
            self.curves += [plot.plot(pen=pg.mkPen(c, width=1), connect="finite") for c in COLORS]
        leds = QtWidgets.QVBoxLayout()
        for m in range(self.nb):
            button = QtWidgets.QPushButton(f"LED {m}")
            button.clicked.connect(lambda _, m=m: self.toggle_led(m))
            leds.addWidget(button)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.graphs, stretch=1)
        body.addLayout(leds)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(body)
        self.resize(1200, 900)
        self.update_range()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(30)

    def offset(self, ch):
        return (1 - ch) * float(self.scale.currentText()) if self.offset_box.isChecked() else 0.0

    def update_range(self, *_):
        s = float(self.scale.currentText())
        if self.offset_box.isChecked():
            s *= 2
        for plot in self.plots:
            plot.setYRange(-s, s, padding=0)
        self.redraw()

    def redraw(self):
        valid = ~np.isnan(self.t)
        for i, curve in enumerate(self.curves):
            if curve.isVisible():
                curve.setData(self.t[valid], self.y[valid, i] + self.offset(i % 3))

    def update_visibility(self, *_):
        for i, curve in enumerate(self.curves):
            curve.setVisible(self.channel_boxes[i % 3].isChecked())

    def update_filter(self, *_):
        if self.filter_box.isChecked() and self.low.value() < self.high.value():
            self.sos = butter(4, [self.low.value(), self.high.value()], btype="band",
                              fs=self.fs, output="sos")
            self.zi = np.zeros((self.sos.shape[0], 2, 3 * self.nb))  # one state per channel
        else:
            self.sos = None

    def toggle_led(self, m):
        self.leds_on[m] = not self.leds_on[m]
        for led in range(2):
            self.wb.set_led(m, led, (0, 16, 16) if self.leds_on[m] else (0, 0, 0))

    def update_data(self):
        _, t, data = self.stream.read(0)  # everything received since the last update
        if len(t) == 0:
            return
        data = data * 1e3  # mV
        if self.sos is not None:
            missing = np.isnan(data)  # filtered as 0, shown as gaps
            data, self.zi = sosfilt(self.sos, np.nan_to_num(data), axis=0, zi=self.zi)  # state carries over
            data[missing] = np.nan
        k = min(len(t), self.n)
        self.t = np.roll(self.t, -k)
        self.y = np.roll(self.y, -k, axis=0)
        self.t[-k:], self.y[-k:] = t[-k:], data[-k:]
        self.redraw()
        for plot in self.plots:
            plot.setXRange(self.t[-1] - args.window, self.t[-1], padding=0)


with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    with wb.stream() as stream:
        app = QtWidgets.QApplication(sys.argv)
        window = LivePlot(wb, stream)
        window.show()
        if args.duration:
            QtCore.QTimer.singleShot(int(args.duration * 1000), app.quit)
        app.exec()
