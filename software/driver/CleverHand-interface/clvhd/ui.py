"""Look of the example windows: palette, font, and the live EMG panel of every module.

Needs the ``plot`` extra (pyqtgraph, PyQt5, scipy). Call ``application()`` for
the QApplication: it applies the dark theme and the font to every window.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import bids
from .coadapt import FIELD_CM, FRAME_RATE, Envelope
from .contact import check_contact
from .filters import BAND, CausalFilter, emg_band
from .hand import stack

# one colour per module (cycled past 13), its channels a shade apart; flat channels muted
MODULE_COLORS = ["#9dd0ff", "#9de9be", "#ffd99f", "#e7c1ff", "#ffb7c3", "#d4f09a", "#ffb98a",
                 "#aab4ff", "#8fe6f0", "#f0f0a0", "#ff9ee6", "#e0c9a6", "#ff9a9a"]
FLAT_COLOR = "#b7b7b7"
BACKGROUND_COLOR, TEXT_COLOR = "#1f1f1f", "#ffffff"
FONT = "Open Sans"  # the system font when not installed
FONT_MESSAGE = 24    # point size of every text screen's main text (02's cue window, 03, 04)
FONT_INFO = 12       # the small info line under it, for the experimenter
FONT_COUNTDOWN = 40  # the cue window's seconds countdown
OFFSET = 0.5  # mV between channels: ch1 at +0.5, ch2 at 0, ch3 at -0.5, in a -1 to 1 mV plot
CONTACT_WINDOW = 1.0  # s of raw data the contact check looks at
LED_ON = (0, 16, 16)  # dim: the LEDs share the Teensy's 3.3 V with the modules
BUTTON_SIZE = 30  # px
RECORD_COLOR = MODULE_COLORS[12]
# the cursor tasks (03, 04): the field outlined in the flat grey on the background, a white
# cursor, targets in the module colours in turn, an acquired target in module 1's green
CURSOR_COLOR = TEXT_COLOR
HIT_COLOR = MODULE_COLORS[1]
TARGET_COLORS = [c for c in MODULE_COLORS if c != HIT_COLOR]
# the task field drawn as a tablet's screen in light lines, its targets as app icons
TABLET_BEZEL_CM = 0.9  # the frame around the screen (the window fits it, with 0.5 cm to spare)
ICON_CORNER = 0.225  # an icon's corner radius, fraction of its side
BULB_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" '
            'fill="none" stroke="{}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M8.5 14.5a6 6 0 1 1 7 0c-.6.5-1 1.2-1 2v.5h-5v-.5c0-.8-.4-1.5-1-2z"/>'
            '<path d="M9.5 20h5"/><path d="M10.5 22.5h3"/></svg>')

pg.setConfigOptions(background=BACKGROUND_COLOR, foreground=TEXT_COLOR, imageAxisOrder="row-major")


def application():
    """The QApplication, in the examples' font."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont(FONT))
    return app


def style(widget):
    """Dark background and light text on ``widget`` and its children."""
    widget.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")


def draw_tablet(qp, screen, px_per_cm):
    """A tablet around the task field ``screen`` (QRectF, px), in light lines on the
    background: its edge and the screen's, both with rounded corners, and the front camera
    between them. The painter is then clipped to the screen."""
    b = TABLET_BEZEL_CM * px_per_cm
    body = screen.adjusted(-b, -b, b, b)
    qp.setPen(QtGui.QPen(QtGui.QColor(FLAT_COLOR), 1.5))
    qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    qp.drawRoundedRect(body, 1.8 * b, 1.8 * b)
    qp.drawRoundedRect(screen, 0.8 * b, 0.8 * b)
    qp.drawEllipse(QtCore.QPointF(body.center().x(), body.top() + b / 2), 0.12 * b, 0.12 * b)
    path = QtGui.QPainterPath()
    path.addRoundedRect(screen, 0.8 * b, 0.8 * b)
    qp.setClipPath(path)


def draw_icon(qp, center, size, color, outline=False):
    """An app icon: a square of side ``size`` px with rounded corners, centred on ``center``
    (QPointF), filled with ``color``; ``outline``: its outline only."""
    r = QtCore.QRectF(center.x() - size / 2, center.y() - size / 2, size, size)
    if outline:
        qp.setPen(QtGui.QPen(QtGui.QColor(color), 2))
        qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    else:
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.setBrush(QtGui.QColor(color))
    qp.drawRoundedRect(r, ICON_CORNER * size, ICON_CORNER * size)


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


def fixed_plot(layout, row, col, y_range, show_time):
    """A plot with a fixed y range and no mouse interaction; the time axis only when ``show_time``."""
    plot = layout.addPlot(row=row, col=col)
    plot.setYRange(*y_range, padding=0)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    if not show_time:
        plot.hideAxis("bottom")
    return plot


class EmgPanel(QtWidgets.QWidget):
    """Every module's channels, 20-450 Hz band-pass and mains-notch filtered (clvhd.filters), -1 to 1 mV per
    module, with each channel's contact status (and the totals on top) and LED buttons.

    It draws what arrived every 30 ms. Its time axis counts s from ``t0``, the host time
    (``time.monotonic()``) of the first sample shown: ``time_plot`` is the plot other plots
    can link their x axis to, plotting ``t - t0``.
    """

    def __init__(self, wb, stream, window=5.0, powerline=50.0):
        super().__init__()
        self.wb, self.stream, self.window, self.powerline = wb, stream, window, powerline
        nb_modules = wb.nb_modules
        fs = stream.nominal_rate
        self.t = np.full(int(window * fs), np.nan)
        self.y = np.full((len(self.t), 3 * nb_modules), np.nan)  # mV, filtered
        self.raw = np.full((int(CONTACT_WINDOW * fs), 3 * nb_modules), np.nan)  # V, for the contact check
        self.filter = CausalFilter(emg_band(fs, BAND, powerline))
        self.t0 = None  # host time of the first complete sample
        self.contact_total = QtWidgets.QLabel()
        self.listeners = []  # called with (t_host, t_device, data) of every read

        # per module: the traces, then each channel's contact status level with its trace and the
        # LED button level with ch2; the status column fits the longest status, a gap and the button
        self.graphs = pg.GraphicsLayoutWidget()
        widest = QtGui.QFontMetrics(QtGui.QFont(FONT)).horizontalAdvance("ch1 poor contact")
        self.graphs.ci.layout.setColumnFixedWidth(1, widest + BUTTON_SIZE + 20)
        self.curves, self.status_texts = [], []
        self.led_buttons = []  # (proxy, status plot) per module
        for m in range(nb_modules):
            last_row = m == nb_modules - 1
            traces = fixed_plot(self.graphs, m, 0, (-1, 1), last_row)
            traces.setLabel("left", f"M{m}", color=module_color(m))
            traces.getAxis("left").setTicks([[(-1, "-1"), (1, "1")]])
            traces.setDownsampling(auto=True, mode="peak")
            traces.setClipToView(True)
            if m == 0:
                self.time_plot = traces
            else:
                traces.setXLink(self.time_plot)
            self.curves += [traces.plot(pen=pg.mkPen(channel_color(m, ch), width=1), connect="finite")
                            for ch in range(3)]

            status = fixed_plot(self.graphs, m, 1, (-1, 1), last_row)
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
            self.led_buttons.append((proxy, status))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.contact_total, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.graphs)

        self.contact_timer = QtCore.QTimer()
        self.contact_timer.timeout.connect(self.update_contact)
        self.contact_timer.start(500)
        self.data_timer = QtCore.QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(30)

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
        status = check_contact(self.raw)
        self.contact_total.setText(f"{status.count('')} good, {status.count('poor contact')} poor contact, "
                                   f"{status.count('flat')} flat"
                                   + (f", {n} no data" if (n := status.count("no data")) else ""))
        for i, state in enumerate(status):
            color = FLAT_COLOR if state in ("flat", "no data") else channel_color(i // 3, i % 3)
            self.curves[i].setPen(pg.mkPen(color, width=1))
            self.status_texts[i].setText(f"ch{i % 3 + 1} {state or 'ok'}", color=color)

    def update_data(self):
        t, t_device, data = self.stream.read(0)  # everything received since the last update
        for listener in self.listeners:
            listener(t, t_device, data)
        self.add(t, data)

    def add(self, t, data):
        """Draw samples ``data`` (V, NaN where missing) at host times ``t``."""
        if len(t) == 0:
            return
        k = min(len(t), len(self.raw))
        self.raw = np.roll(self.raw, -k, axis=0)
        self.raw[-k:] = data[-k:]
        start, filtered = self.filter(data * 1e3)  # mV
        if len(filtered) == 0:
            return
        t, data = t[start:], np.where(np.isnan(data[start:]), np.nan, filtered)  # gaps shown
        if self.t0 is None:
            self.t0 = t[0]
        k = min(len(t), len(self.t))
        self.t = np.roll(self.t, -k)
        self.y = np.roll(self.y, -k, axis=0)
        self.t[-k:], self.y[-k:] = t[-k:] - self.t0, data[-k:]
        shown = ~np.isnan(self.t)
        for i, curve in enumerate(self.curves):
            curve.setData(self.t[shown], self.y[shown, i] + (1 - i % 3) * OFFSET)
        self.time_plot.setXRange(self.t[-1] - self.window, self.t[-1], padding=0)  # the others follow


def _add_storage_arguments(g):
    """Where the recordings go and in which format."""
    g.add_argument("--bids-root", type=lambda s: Path(s).expanduser(), default=str(bids.DEFAULT_ROOT),
                   help="BIDS dataset folder, created if missing (default: %(default)s)")
    g.add_argument("--format", choices=sorted(bids.FORMATS), default="edf",
                   help="EMG file format: edf (16 bits, default) or bdf (24 bits)")
    g.add_argument("--keep-sourcedata", action="store_true",
                   help="keep the stream in sourcedata/ once the BIDS files are written")


def add_record_arguments(parser):
    """The command-line options of a RecordBar: the dataset and what describes the recordings."""
    g = parser.add_argument_group("recording (BIDS)")
    _add_storage_arguments(g)
    g.add_argument("--subject", default="", help="subject label (editable in the window)")
    g.add_argument("--session", default="", help="session label (editable in the window)")
    g.add_argument("--task", default="", help="task label (editable in the window)")
    g.add_argument("--acq", help="acquisition label")
    g.add_argument("--task-description")
    g.add_argument("--muscle", action="append", default=[], metavar="MODULE=NAME",
                   help="muscle under a module, e.g. 0=ECR (repeat)")
    g.add_argument("--placement", help="how the band was placed (EMGPlacementSchemeDescription)")
    g.add_argument("--age", type=int)
    g.add_argument("--sex", choices=["M", "F", "O"])
    g.add_argument("--handedness", choices=["right", "left", "ambidextrous"])


def _stats_since(before, after):
    """Per-module stream statistics of a recording: the counts between ``before`` and ``after``."""
    return [{**a, **{k: a[k] - b[k] for k in ("samples", "dropped", "duplicates")}}
            for b, a in zip(before, after, strict=True)]


class RecordBar(QtWidgets.QWidget):
    """Subject, session and task fields, the run the next recording gets, and a record button.

    While recording it streams every sample the EMG panel reads, and the hand
    frames passed to ``add_frames``, to ``sourcedata/`` (``clvhd.bids.RawRecording``),
    pushed to the disk every 200 ms: the RAM use stays flat however long the
    recording. On stop it writes the BIDS files from there
    (``clvhd.bids.export_recording``) on a background thread, so the plots keep
    running, then deletes them from sourcedata/ (unless --keep-sourcedata). ``camera_fps``:
    record the hand frames as motion (None: EMG only). ``events``: the columns of the events
    ``add_event`` records with them (``_events.tsv``, described by ``event_descriptions``;
    None: no events). ``record(on)`` starts or stops it as the button does.
    Call ``finish()`` when the window closes: it saves a recording still running.
    """

    def __init__(self, panel, wb, args, camera_fps=None, events=None, event_descriptions=None):
        super().__init__()
        self.panel, self.wb, self.args, self.camera_fps = panel, wb, args, camera_fps
        self.events = events
        self.root = args.bids_root
        self.export = {  # what describes every recording, kept in its sourcedata header
            "participant": {"age": args.age, "sex": args.sex, "handedness": args.handedness},
            "muscles": dict(s.split("=", 1) for s in args.muscle),
            "powerline_freq": panel.powerline, "placement_description": args.placement,
            "task_description": args.task_description, "file_format": args.format,
            "event_descriptions": event_descriptions}
        self.recording, self.saver, self.result = False, None, None
        panel.listeners.append(self._add_samples)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fields = {}
        for key, width in (("subject", 60), ("session", 60), ("task", 120)):
            layout.addWidget(QtWidgets.QLabel(key))
            edit = QtWidgets.QLineEdit(getattr(args, key) or "")
            edit.setFixedWidth(width)
            edit.setStyleSheet(f"border: 1px solid {FLAT_COLOR}; border-radius: 4px; padding: 2px 4px;")
            edit.textChanged.connect(self.refresh)
            layout.addWidget(edit)
            self.fields[key] = edit
        self.run_label = QtWidgets.QLabel()
        layout.addWidget(self.run_label)
        self.button = QtWidgets.QPushButton("Record")
        self.button.setCheckable(True)
        self.button.setStyleSheet(
            f"QPushButton {{ border: 1px solid {RECORD_COLOR}; border-radius: {BUTTON_SIZE // 2}px;"
            f" color: {RECORD_COLOR}; background: transparent; padding: 0 16px;"
            f" min-height: {BUTTON_SIZE - 2}px; }}"
            f"QPushButton:checked {{ background-color: {RECORD_COLOR}; color: {BACKGROUND_COLOR}; }}"
            f"QPushButton:disabled {{ border-color: {FLAT_COLOR}; color: {FLAT_COLOR}; }}")
        self.button.toggled.connect(self.toggle)
        layout.addWidget(self.button)
        self.status = QtWidgets.QLabel()
        layout.addWidget(self.status)
        layout.addStretch()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(200)
        self.refresh()

    def entities(self):
        """The recording's entities from the fields; ValueError with a message when invalid."""
        f = {k: e.text().strip() for k, e in self.fields.items()}
        if not f["subject"] or not f["task"]:
            raise ValueError("enter a subject and a task to record")
        return bids.Entities(f["subject"], f["task"], f["session"] or None, self.args.acq)

    def refresh(self):
        """The run number and the button for the fields as they are."""
        if self.recording or self.saver:
            return
        try:
            ent = self.entities()
        except ValueError as e:
            self.run_label.setText("")
            self.button.setEnabled(False)
            self.status.setText(str(e))
            return
        self.run_label.setText(f"run {bids.next_run(self.root, ent)}")
        self.button.setEnabled(True)
        if not self.result:
            self.status.setText(f"to {self.root}")

    def toggle(self, on):
        if on:
            self.start()
        else:
            self.stop()

    def start(self):
        ent = self.entities()
        self.ent = bids.Entities(ent.subject, ent.task, ent.session, ent.acquisition,
                                 bids.next_run(self.root, ent))
        stream = self.panel.stream
        self.raw = bids.RawRecording(self.root, self.ent, self.wb.config, stream.channel_labels,
                                     hand=bool(self.camera_fps), camera_fps=self.camera_fps,
                                     firmware=self.wb.firmware, export=self.export,
                                     events=self.events)
        self.with_hand = 0
        self.stats_before = stream.stats()
        self.t_start = time.monotonic()
        self.recording, self.result = True, None
        for edit in self.fields.values():
            edit.setEnabled(False)
        self.button.setText("Stop")
        self.tick()

    def stop(self):
        self.recording = False
        self.button.setEnabled(False)
        self.button.setText("Saving")
        self.status.setText(f"saving {self.ent.label()}")
        self.raw.close(_stats_since(self.stats_before, self.panel.stream.stats()))
        self.saver = threading.Thread(target=self._save, name="clvhd-save", args=(self.raw,))
        self.saver.start()

    def _save(self, raw):
        try:
            self.result = bids.export_recording(raw, keep_source=self.args.keep_sourcedata)
        except Exception as e:  # shown in the bar
            self.result = e

    def _done(self):
        self.saver = None
        if isinstance(self.result, Exception):
            self.status.setText(f"not saved: {self.result}")
        else:
            result = self.result
            assert result is not None  # set by _save before _done runs
            bad = sum(bool(c) for c in result["contact"])
            coarse = result["coarse"]
            worst = max(coarse, key=coarse.get) if coarse else None
            self.status.setText(f"saved {self.ent.label()} ({result['duration']:.1f} s"
                                + (f", {bad} channels marked bad" if bad else "") + ")"
                                + (f"; {len(coarse)} channels coarse in EDF, up to "
                                   f"{coarse[worst]:.1f} uV steps ({worst}): --format bdf keeps "
                                   "24 bits" if coarse else ""))
        for edit in self.fields.values():
            edit.setEnabled(True)
        self.button.blockSignals(True)
        self.button.setChecked(False)
        self.button.blockSignals(False)
        self.button.setText("Record")
        self.refresh()

    def tick(self):
        if self.recording:
            elapsed = time.monotonic() - self.t_start
            self.status.setText(f"recording  {int(elapsed // 60)}:{elapsed % 60:04.1f}"
                                + (f", hand in {self.with_hand} of {self.raw.frames} frames"
                                   if self.camera_fps else ""))
            self.raw.flush()
        elif self.saver and not self.saver.is_alive():
            self._done()

    def record(self, on):
        """Start (the fields' entities, as the button) or stop recording; on start, the reason
        it cannot (None when it does)."""
        if on and self.saver:  # the disabled button would still start a recording
            return "busy saving the last recording"
        if on and not self.recording:
            try:
                self.entities()
            except ValueError as e:
                return str(e)
        if on != self.button.isChecked():
            self.button.setChecked(on)
        return None

    def add_event(self, t_host, trial_type, duration=0.0, **values):
        """An event at host time ``t_host`` in the recording running (none: dropped)."""
        if self.recording:
            self.raw.append_event(t_host, trial_type, duration, **values)

    def _add_samples(self, t_host, t_device, data):
        if self.recording:
            self.raw.append(t_host, t_device, data)

    def add_frames(self, frames):
        """Hand frames (``HandTracker.read()``) to record with the EMG."""
        if self.recording and frames:
            self.raw.append_frames(stack(frames))
            self.with_hand += sum(not np.isnan(f.score) for f in frames)

    def finish(self):
        """Save a recording still running and wait for the saving to end."""
        if self.recording:
            self.stop()
        if self.saver:
            self.saver.join()
            self._done()


def add_device_arguments(parser):
    """The command-line options of the wristband: its port, modules and the mains notch."""
    parser.add_argument("--port", help="serial port of the Teensy (default: first Teensy found)")
    parser.add_argument("--modules", type=int, default=13, help="expected number of modules")
    parser.add_argument("--powerline", type=float, default=50, help="mains frequency notched (Hz, 0: none)")


def add_task_arguments(parser, task):
    """The command-line options of the cursor tasks (03, 04); ``task``: the default task label."""
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session")
    parser.add_argument("--task", default=task, help="BIDS task label")
    add_device_arguments(parser)
    parser.add_argument("--seed", type=int, help="random seed (default: random)")
    parser.add_argument("--screen-width-cm", type=float,
                        help="visible width of the screen, to draw in cm (default: as the screen "
                             "reports it)")
    parser.add_argument("--field-cm", type=float, nargs=2, default=FIELD_CM, metavar=("WIDTH", "HEIGHT"),
                        help="task field, centred on the screen (default: an 11-inch iPad, "
                             "%(default)s; the co-adaptation paper's display: 46.5 24.5)")
    parser.add_argument("--cursor-radius-cm", type=float, default=0.4)
    parser.add_argument("--windowed", action="store_true", help="a window instead of full screen")
    _add_storage_arguments(parser)


class TaskWindow(QtWidgets.QWidget):
    """A cursor task (03, 04): the task field (``--field-cm``, cm from the screen centre, y up)
    drawn as a tablet, and the EMG envelope of the decoder's ``channels`` at 60 Hz.

    The stream is read every 4 ms and its envelope kept running between recordings (the
    filters stay settled); while ``state`` is "running" it is also streamed to the
    ``RawRecording`` of ``start_recording``, and ``frame(now)`` (the subclass's) runs on a
    60 Hz grid, skipping missed frames rather than bunching them. Otherwise ``idle(now)``
    repaints the text screens. ``layout_task()`` (the subclass's) follows each resize.
    """

    def __init__(self, wb, stream, args, title, channels, band=BAND):
        super().__init__()
        self.wb, self.stream, self.args = wb, stream, args
        self.channels = np.array(channels)
        labels = stream.channel_labels
        left_out = [lb for c, lb in enumerate(labels) if c not in set(channels)]
        self.channel_info = (f"Decoder: {len(channels)} of {len(labels)} channels"
                             + (f", left out: {', '.join(left_out)}" if left_out else ""))
        self.envelope = Envelope(stream.nominal_rate, len(labels), args.powerline, band)
        self.state, self.frames, self.last_paint = "ready", [], 0.0
        # set for real by start_recording(); declared here so tick() can read them
        self.next_frame = self.last_frame = self.last_flush = 0.0
        self.geometry_info = None
        self.setWindowTitle(title)
        self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(4)

    def open(self):
        if self.args.windowed:
            self.resize(1280, 720)
            self.show()
        else:
            self.showFullScreen()

    def layout_task(self):
        """Place the task in the field (``self.half``, cm); a note for the geometry printout."""
        return ""

    def resizeEvent(self, event):
        screen = self.screen()
        width_cm = self.args.screen_width_cm or screen.physicalSize().width() / 10 * \
            self.width() / screen.size().width()
        self.px_per_cm = self.width() / width_cm
        window_cm = np.array([self.width(), self.height()]) / self.px_per_cm
        field = np.array(self.args.field_cm, dtype=float)
        # a smaller window: the field scaled to fit it, with the tablet's frame
        shrink = min(1.0, *(window_cm / (field + 2 * TABLET_BEZEL_CM + 1.0)))
        self.half = field * shrink / 2
        w, h = 2 * self.half * self.px_per_cm
        self.field = QtCore.QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        note = self.layout_task()
        geometry = (f"Task field {2 * self.half[0]:.1f} x {2 * self.half[1]:.1f} cm, {w:.0f} x {h:.0f} px "
                    f"({self.px_per_cm:.1f} px/cm){note}: check its width with a ruler, else set "
                    "--screen-width-cm" + (" (shrunk to fit the window)" if shrink < 1 else ""))
        if geometry != self.geometry_info:
            self.geometry_info = geometry
            print(geometry, flush=True)

    def to_px(self, pos):
        return QtCore.QPointF(self.width() / 2 + pos[0] * self.px_per_cm,
                              self.height() / 2 - pos[1] * self.px_per_cm)

    def draw_cursor(self, qp, pos):
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.setBrush(QtGui.QColor(CURSOR_COLOR))
        r = self.args.cursor_radius_cm * self.px_per_cm
        qp.drawEllipse(self.to_px(pos), r, r)

    def draw_message(self, qp, text, channel_info=True):
        """A text screen: ``text`` centred, and the decoder's channels below, for the experimenter."""
        qp.fillRect(self.rect(), QtGui.QColor(BACKGROUND_COLOR))
        qp.setPen(QtGui.QColor(TEXT_COLOR))
        font = qp.font()
        font.setPointSize(FONT_MESSAGE)
        qp.setFont(font)
        qp.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, text)
        if channel_info:
            font.setPointSize(FONT_INFO)
            qp.setFont(font)
            qp.setPen(QtGui.QColor(FLAT_COLOR))
            qp.drawText(self.rect().adjusted(40, 0, -40, -30),
                        QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom
                        | QtCore.Qt.TextFlag.TextWordWrap, self.channel_info)

    def tick(self):
        if not hasattr(self, "half"):
            return  # not laid out yet
        now = time.monotonic()
        t, t_device, data = self.stream.read(0)
        self.envelope.add(data)
        if self.state != "running":
            self.idle(now)
            return
        self.raw.append(t, t_device, data)
        if now - self.last_flush > 0.2:
            self.raw.flush()
            self.last_flush = now
        if now >= self.next_frame:
            self.frame(now)
            self.next_frame += max(1, np.ceil((now - self.next_frame) * FRAME_RATE)) / FRAME_RATE
            self.update()

    def idle(self, now):
        if now - self.last_paint > 0.2:
            self.last_paint = now
            self.update()

    def start_recording(self, ent, task_description):
        """Stream the EMG to ``sourcedata/`` as ``ent``, and run the frames from now."""
        args = self.args
        self.raw = bids.RawRecording(
            args.bids_root, ent, self.wb.config, self.stream.channel_labels, firmware=self.wb.firmware,
            export={"powerline_freq": args.powerline, "task_description": task_description,
                    "file_format": args.format})
        self.frames = []
        self.envelope.frame()  # drop what came in before
        self.t_start = self.next_frame = self.last_frame = self.last_flush = time.monotonic()
        self.state = "running"

    def next_input(self, now):
        """``(u, dt)``: the decoder channels' envelope (mV) since the last frame, and the time since it."""
        u = self.envelope.frame()
        u = np.zeros(len(self.channels)) if u is None else np.nan_to_num(u[self.channels])
        dt = now - self.last_frame if self.frames else 1 / FRAME_RATE
        self.last_frame = now
        return u, dt

    def frame(self, now):
        raise NotImplementedError

    def export(self):
        """Close the recording and write its BIDS EMG (``clvhd.bids.export_recording``); None
        (printed) when it holds no sample."""
        self.raw.close()
        if not self.raw.samples:
            print(f"{self.raw.header_path.name}: no EMG sample was recorded, not saved")
            return None
        return bids.export_recording(self.raw, keep_source=self.args.keep_sourcedata)

    def beh_sidecar(self):
        """What every beh/ sidecar of the task holds: the geometry and the decoder's channels."""
        labels = self.stream.channel_labels
        return {"FieldWidthCm": float(2 * self.half[0]), "FieldHeightCm": float(2 * self.half[1]),
                "ScreenWidthCm": float(self.width() / self.px_per_cm),
                "ScreenHeightCm": float(self.height() / self.px_per_cm),
                "CursorRadiusCm": self.args.cursor_radius_cm, "FrameRate": FRAME_RATE,
                "DecoderChannels": [labels[c] for c in self.channels]}
