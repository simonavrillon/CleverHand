"""Live plot of every module, 20-450 Hz band-pass and 50 Hz notch filtered, with electrode contact
status (per channel and in total) and LED buttons.

    python 01_live_plot.py [--port PORT] [--modules 13] [--window 5]
"""
import argparse
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

import clvhd

p = argparse.ArgumentParser()
p.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
p.add_argument("--modules", type=int, default=13, help="expected number of modules")
p.add_argument("--window", type=float, default=5, help="time window shown (s)")
p.add_argument("--duration", type=float, default=0, help="close after this many seconds (0: never)")
args = p.parse_args()

# one colour per module (cycled past 13), its channels a shade apart; flat channels muted
MODULE_COLORS = ["#9dd0ff", "#9de9be", "#ffd99f", "#e7c1ff", "#ffb7c3", "#d4f09a", "#ffb98a",
                 "#aab4ff", "#8fe6f0", "#f0f0a0", "#ff9ee6", "#e0c9a6", "#ff9a9a"]
FLAT_COLOR = "#b7b7b7"
BACKGROUND_COLOR, TEXT_COLOR = "#1f1f1f", "#ffffff"
FONT = "Open Sans"  # the system font when not installed
OFFSET = 0.5  # mV between channels: ch1 at +0.5, ch2 at 0, ch3 at -0.5, in a -1 to 1 mV plot
CONTACT_WINDOW = 1.0  # s of raw data the contact check looks at
LED_ON = (0, 16, 16)  # dim: the LEDs share the Teensy's 3.3 V with the modules
BUTTON_SIZE = 30  # px
BULB_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" '
            'fill="none" stroke="{}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M8.5 14.5a6 6 0 1 1 7 0c-.6.5-1 1.2-1 2v.5h-5v-.5c0-.8-.4-1.5-1-2z"/>'
            '<path d="M9.5 20h5"/><path d="M10.5 22.5h3"/></svg>')

pg.setConfigOptions(background=BACKGROUND_COLOR, foreground=TEXT_COLOR)


def module_color(m):
    return MODULE_COLORS[m % len(MODULE_COLORS)]


def channel_color(m, ch):
    """ch1 slightly lighter, ch3 slightly darker than the module's colour."""
    c = pg.mkColor(module_color(m))
    return [c.lighter(115), c, c.darker(125)][ch].name()


def led_button(m):
    """Round button with a bulb in the module's colour, filled while its LEDs are on."""
    icon = QtGui.QIcon()
    for color, state in ((module_color(m), QtGui.QIcon.State.Off), (BACKGROUND_COLOR, QtGui.QIcon.State.On)):
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(BULB_SVG.format(color).encode(), "SVG")
        icon.addPixmap(pixmap, QtGui.QIcon.Mode.Normal, state)
    button = QtWidgets.QToolButton()
    button.setCheckable(True)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(BUTTON_SIZE * 3 // 5, BUTTON_SIZE * 3 // 5))
    button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
    button.setToolTip(f"LEDs of M{m}")
    button.setStyleSheet(
        f"QToolButton {{ border: 1px solid {module_color(m)}; border-radius: {BUTTON_SIZE // 2}px;"
        f" background: transparent; }}"
        f"QToolButton:checked {{ background-color: {module_color(m)}; }}")
    return button


def fill_gaps(last, data):
    """`data` with each missing sample replaced by its channel's previous value (`last`: the row before)."""
    rows = np.vstack([last, data])
    idx = np.maximum.accumulate(np.where(np.isnan(rows), 0, np.arange(len(rows))[:, None]), axis=0)
    return rows[idx, np.arange(rows.shape[1])][1:]


class LivePlot(QtWidgets.QWidget):
    def __init__(self, wb, stream):
        super().__init__()
        self.wb, self.stream = wb, stream
        nb_modules = wb.nb_modules
        fs = stream.nominal_rate
        self.t = np.full(int(args.window * fs), np.nan)
        self.y = np.full((len(self.t), 3 * nb_modules), np.nan)  # mV, filtered
        self.raw = np.full((int(CONTACT_WINDOW * fs), 3 * nb_modules), np.nan)  # V, for the contact check
        self.sos = np.concatenate([butter(4, [20, 450], btype="band", fs=fs, output="sos"),
                                   tf2sos(*iirnotch(50, 30, fs=fs))])
        # filter state and last sample of each channel, set from the first complete sample
        self.zi = self.last = None
        self.setWindowTitle("CleverHand EMG")
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")
        self.contact_total = QtWidgets.QLabel()

        # per module: the traces, then each channel's contact status level with its trace and the
        # LED button level with ch2; the status column fits the longest status, a gap and the button
        self.graphs = pg.GraphicsLayoutWidget()
        widest = QtGui.QFontMetrics(QtGui.QFont(FONT)).horizontalAdvance("ch1 poor contact")
        self.graphs.ci.layout.setColumnFixedWidth(1, widest + BUTTON_SIZE + 20)
        self.curves, self.status_texts = [], []
        for m in range(nb_modules):
            last_row = m == nb_modules - 1
            traces = self.add_plot(m, 0, last_row)
            traces.setLabel("left", f"M{m}", color=module_color(m))
            traces.getAxis("left").setTicks([[(-1, "-1"), (1, "1")]])
            traces.setDownsampling(auto=True, mode="peak")
            traces.setClipToView(True)
            if m == 0:
                self.first_traces = traces
            else:
                traces.setXLink(self.first_traces)
            self.curves += [traces.plot(pen=pg.mkPen(channel_color(m, ch), width=1), connect="finite")
                            for ch in range(3)]

            status = self.add_plot(m, 1, last_row)
            status.hideAxis("left")
            status.setXRange(0, 1, padding=0)
            if last_row:  # its time axis kept but invisible, so the plot area matches the traces'
                status.getAxis("bottom").setPen(BACKGROUND_COLOR)
                status.getAxis("bottom").setTextPen(BACKGROUND_COLOR)
            for ch in range(3):
                text = pg.TextItem(anchor=(0, 0.5))
                text.setPos(0, (1 - ch) * OFFSET)
                status.addItem(text)
                self.status_texts.append(text)
            button = led_button(m)
            button.toggled.connect(lambda on, m=m: self.set_leds(m, on))
            proxy = QtWidgets.QGraphicsProxyWidget(status)  # outside the view box, which clips clicks
            proxy.setWidget(button)
            status.vb.sigResized.connect(lambda vb, proxy=proxy: self.place_button(proxy, vb))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.contact_total, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.graphs)
        self.resize(1200, 900)

        self.contact_timer = QtCore.QTimer()
        self.contact_timer.timeout.connect(self.update_contact)
        self.contact_timer.start(500)
        self.data_timer = QtCore.QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(30)

    def add_plot(self, row, col, last_row):
        """A fixed -1 to 1 plot; only the last row shows the time axis."""
        plot = self.graphs.addPlot(row=row, col=col)
        plot.setYRange(-1, 1, padding=0)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        if not last_row:
            plot.hideAxis("bottom")
        return plot

    @staticmethod
    def place_button(proxy, vb):
        """At the right edge of the view box, centred on y = 0 (ch2)."""
        r = vb.mapRectToItem(proxy.parentItem(), vb.boundingRect())
        proxy.setPos(r.right() - BUTTON_SIZE, r.center().y() - BUTTON_SIZE / 2)

    def set_leds(self, m, on):
        for led in range(2):
            self.wb.set_led(m, led, LED_ON if on else (0, 0, 0))

    def update_contact(self):
        if np.isnan(self.raw[0]).all():
            return  # less than CONTACT_WINDOW received yet
        status = clvhd.check_contact(self.raw)
        self.contact_total.setText(f"{status.count('')} good, {status.count('poor contact')} poor contact, "
                                   f"{status.count('flat')} flat")
        for i, state in enumerate(status):
            color = FLAT_COLOR if state == "flat" else channel_color(i // 3, i % 3)
            self.curves[i].setPen(pg.mkPen(color, width=1))
            self.status_texts[i].setText(f"ch{i % 3 + 1} {state or 'ok'}", color=color)

    def update_data(self):
        _, t, data = self.stream.read(0)  # everything received since the last update
        if len(t) == 0:
            return
        k = min(len(t), len(self.raw))
        self.raw = np.roll(self.raw, -k, axis=0)
        self.raw[-k:] = data[-k:]
        data = data * 1e3  # mV
        if self.zi is None:  # start the filters at steady state on the first complete sample
            complete = np.flatnonzero(~np.isnan(data).any(axis=1))
            if len(complete) == 0:
                return
            t, data = t[complete[0]:], data[complete[0]:]
            self.zi = sosfilt_zi(self.sos)[:, :, None] * data[0]  # one state per channel
            self.last = data[0]
        # missing samples are filtered as the channel's last value (a jump to 0 would ring)
        # and shown as gaps
        missing = np.isnan(data)
        data = fill_gaps(self.last, data)
        self.last = data[-1]
        data, self.zi = sosfilt(self.sos, data, axis=0, zi=self.zi)
        data[missing] = np.nan
        k = min(len(t), len(self.t))
        self.t = np.roll(self.t, -k)
        self.y = np.roll(self.y, -k, axis=0)
        self.t[-k:], self.y[-k:] = t[-k:], data[-k:]
        shown = ~np.isnan(self.t)
        for i, curve in enumerate(self.curves):
            curve.setData(self.t[shown], self.y[shown, i] + (1 - i % 3) * OFFSET)
        self.first_traces.setXRange(self.t[-1] - args.window, self.t[-1], padding=0)  # the others follow


with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    with wb.stream() as stream:
        app = QtWidgets.QApplication(sys.argv)
        app.setFont(QtGui.QFont(FONT))
        window = LivePlot(wb, stream)
        window.show()
        if args.duration:
            QtCore.QTimer.singleShot(int(args.duration * 1000), app.quit)
        app.exec()
