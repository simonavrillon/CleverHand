"""Live EMG of every module (as 01_live_plot.py) next to the camera, its MediaPipe hand
skeleton, and the finger movements (each finger's flexion, the thumb-index pinch), with a
record button that saves the EMG and the hand kinematics to a BIDS dataset (emg/ and motion/,
see clvhd/bids.py); with --protocol, cued gestures for classifier datasets.

    python 02_live_emg_hand.py [--port PORT] [--modules 13] [--camera 0] [--window 5]
        [--protocol protocols/finger_gestures.json [--cue-fullscreen] [--seed 1]]
        [--bids-root ~/Documents/CleverHand] [--subject 01] [--session 01] [--task fingers]
        [--muscle 0=FDS] [--age 30 --sex F --handedness right]

Subject, session and task can also be typed in the window; each recording takes the next run.

Each finger's flexion is the sum of its joint angles (near 0 flat, about 250 deg in a fist),
the pinch the distance between the thumb and index tips (clvhd.hand.finger_flexion,
pinch_aperture), from MediaPipe's 3D hand landmarks; the skeleton draws each finger in its
trace's colour.

--protocol (a JSON file, clvhd.cues.Protocol) opens a second window, for the participant,
with the cards of its gestures (clvhd.cues: thumb pinches with each finger, with index and
middle, with all; extensions of each finger, thumb + index, thumb + index + middle, all;
rest), each performed once from the starting position shown on the card (open hand for a
pinch, closed hand for an extension): the card in grey with a countdown
(prepare), lit (go: do it once), then the relaxed hand (rest). Space starts it, recording the EMG, the camera and the cue
events in one run (the task: the protocol's unless typed); Space also ends the breaks between
blocks; Escape stops it and saves what was recorded. The events (_events.tsv: cue, go, rest
and break, with the gesture, its fingers and motion, block and trial) are stamped when their
card is drawn, on the samples' clock. After the recording, clvhd-movements finds each go's
real movement from the camera (derivatives/movements/), the windows to train on.

Three threads feed the window: the driver's serial reader, the camera reader
and the MediaPipe thread (clvhd.hand.HandTracker). Both streams are stamped
with time.monotonic(), so the EMG and the finger movements share the time axis.
"""
import argparse
import itertools
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import clvhd
from clvhd import cues, ui
from clvhd.hand import FINGERS, HandTracker, finger_flexion, pinch_aperture

p = argparse.ArgumentParser()
ui.add_device_arguments(p)
p.add_argument("--camera", default="0", help="camera index or video file")
p.add_argument("--protocol", help="cued gestures: a protocol JSON file (clvhd.cues.Protocol)")
p.add_argument("--cue-fullscreen", action="store_true", help="the cards full screen")
p.add_argument("--seed", type=int, help="random seed of the trial order (default: random)")
p.add_argument("--window", type=float, default=5, help="time window shown (s)")
p.add_argument("--duration", type=float, default=0, help="close after this many seconds (0: never)")
ui.add_record_arguments(p)
args = p.parse_args()

FINGER_COLORS = dict(zip(FINGERS, ui.MODULE_COLORS, strict=False))  # the skeleton's fingers as their traces
PINCH_COLOR = ui.TEXT_COLOR
FLEXION_RANGE = (0, 270)  # deg: straight to a fist
PINCH_RANGE = (0, 15)  # cm
# the palm's edges (pairs of landmark indices), drawn in the flat grey; each finger's chain in its colour
PALM = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


class Window(QtWidgets.QWidget):
    def __init__(self, wb, stream, tracker, events=None, event_descriptions=None):
        super().__init__()
        self.tracker = tracker
        self.setWindowTitle("CleverHand EMG + hand")
        ui.style(self)
        self.emg = ui.EmgPanel(wb, stream, args.window, args.powerline)
        self.recorder = ui.RecordBar(self.emg, wb, args, camera_fps=tracker.nominal_fps,
                                     events=events, event_descriptions=event_descriptions)
        self.hand_t = np.full(int(args.window * 120), np.nan)  # room for 120 fps
        self.fingers = np.full((len(self.hand_t), len(FINGERS) + 1), np.nan)  # flexions, pinch
        self.with_hand = 0

        self.camera_status = QtWidgets.QLabel()
        graphs = pg.GraphicsLayoutWidget()
        self.view = graphs.addViewBox(row=0, col=0, lockAspect=True, invertY=True, enableMouse=False)
        self.image = pg.ImageItem()
        self.view.addItem(self.image)
        # the skeleton: the palm, then each finger in its trace's colour
        self.bones = [(PALM, pg.PlotDataItem(pen=pg.mkPen(ui.FLAT_COLOR, width=3), connect="pairs"))]
        self.bones += [(list(itertools.pairwise(chain)),
                        pg.PlotDataItem(pen=pg.mkPen(FINGER_COLORS[name], width=3), connect="pairs"))
                       for name, chain in FINGERS.items()]
        for _, item in self.bones:
            self.view.addItem(item)

        flexion_plot = ui.fixed_plot(graphs, 1, 0, FLEXION_RANGE, show_time=False)
        flexion_plot.setLabel("left", "fingers (deg)")
        flexion_plot.getAxis("left").setTicks([[(v, str(v)) for v in (0, 90, 180, 270)]])
        flexion_plot.setXLink(self.emg.time_plot)
        legend = flexion_plot.addLegend(offset=(5, 5), labelTextColor=ui.TEXT_COLOR, colCount=5)
        legend.setBrush(ui.BACKGROUND_COLOR)
        self.curves = [flexion_plot.plot(pen=pg.mkPen(c, width=2), name=name, connect="finite")
                       for name, c in FINGER_COLORS.items()]
        pinch_plot = ui.fixed_plot(graphs, 2, 0, PINCH_RANGE, show_time=True)
        pinch_plot.setLabel("left", "pinch (cm)")
        pinch_plot.getAxis("left").setTicks([[(v, str(v)) for v in (0, 5, 10, 15)]])
        pinch_plot.setXLink(self.emg.time_plot)
        self.curves.append(pinch_plot.plot(pen=pg.mkPen(PINCH_COLOR, width=2), connect="finite"))
        graphs.ci.layout.setRowStretchFactor(0, 4)
        graphs.ci.layout.setRowStretchFactor(1, 3)
        graphs.ci.layout.setRowStretchFactor(2, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.camera_status, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        right.addWidget(graphs)
        panels = QtWidgets.QHBoxLayout()
        panels.addWidget(self.emg, stretch=3)
        panels.addLayout(right, stretch=2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.recorder)
        layout.addLayout(panels)
        self.resize(1700, 900)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_hand)
        self.timer.start(30)

    def update_hand(self):
        frames = self.tracker.read()
        self.recorder.add_frames(frames)
        self.with_hand += sum(not np.isnan(f.score) for f in frames)
        if self.emg.t0 is None:
            frames = []  # no time axis before the first EMG sample
        if frames:
            k = min(len(frames), len(self.hand_t))
            self.hand_t = np.roll(self.hand_t, -k)
            self.fingers = np.roll(self.fingers, -k, axis=0)
            self.hand_t[-k:] = [f.t - self.emg.t0 for f in frames[-k:]]
            world = np.array([f.hand_world for f in frames[-k:]])
            self.fingers[-k:] = np.column_stack([finger_flexion(world), pinch_aperture(world)])
            shown = ~np.isnan(self.hand_t)
            for i, curve in enumerate(self.curves):
                curve.setData(self.hand_t[shown], self.fingers[shown, i])
        st = self.tracker.stats()
        self.camera_status.setText(f"camera: {st['captured']} frames, {st['processed']} processed, "
                                   f"hand in {self.with_hand}")
        image, frame = self.tracker.latest()
        if image is None:
            return
        h, w = image.shape[:2]
        self.image.setImage(image[:, :, ::-1])  # BGR to RGB
        for pairs, item in self.bones:
            if np.isnan(frame.hand_image[0, 0]):
                item.setData([], [])
            else:
                xy = np.array([frame.hand_image[i, :2] for pair in pairs for i in pair]) * (w, h)
                item.setData(xy[:, 0], xy[:, 1])


EVENT_COLUMNS = ["gesture", "gesture_name", "fingers", "motion", "block", "trial"]


def event_descriptions(protocol):
    specs = protocol.specs()
    return {
        "onset": {"Description": "When the card was drawn (Qt paint, within a display frame of "
                                 "it showing), from the first EMG sample.", "Units": "s"},
        "trial_type": {"Levels": {
            "cue": "The gesture's card in grey: prepare.",
            "go": "The card lit: perform the gesture once, starting open (pinch) or closed "
                  "(extension) as the card shows.",
            "rest": "A relaxed hand: rest.",
            "break": "A break between blocks, ended by the participant (duration n/a)."}},
        "gesture": {"Description": "The gesture cued (clvhd.cues.GESTURES)",
                    "Levels": {g: spec["name"] for g, spec in specs.items()}},
        "gesture_name": {"Description": "Its name on the card"},
        "fingers": {"Description": "The fingers moving, + separated"},
        "motion": {"Description": "Their motion: pinch (their tips meeting the thumb's, from an "
                                  "open hand), extend (straightening from a closed hand), rest"},
        "block": {"Description": "Block (one repetition of every gesture), from 1"},
        "trial": {"Description": "Trial in the recording, from 1"},
        "Protocol": {"Name": protocol.name, "PrepareS": protocol.prepare, "GoS": protocol.go,
                     "RestS": protocol.rest, "Repetitions": protocol.repetitions},
    }


class CueWindow(QtWidgets.QWidget):
    """The protocol's cards, for the participant, recording through ``recorder``."""

    def __init__(self, protocol, recorder):
        super().__init__()
        self.protocol, self.recorder = protocol, recorder
        self.specs = protocol.specs()
        self.order = protocol.trial_order(np.random.default_rng(args.seed))
        self.state, self.k, self.message = "ready", 0, ""
        self.pending = None  # the event of the card to draw: stamped when it is drawn
        self.phase_end = None
        self.setWindowTitle(f"Gestures: {protocol.name}")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        ui.style(self)
        if not recorder.fields["task"].text().strip():
            recorder.fields["task"].setText(protocol.task)
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.timer.start(5)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space and self.state == "ready":
            self.message = self.recorder.record(True) or ""
            if not self.message:
                self.k = 0
                self.begin("cue")
            self.update()
        elif key == QtCore.Qt.Key.Key_Space and self.state == "break":
            self.begin("cue")
        elif key == QtCore.Qt.Key.Key_Escape and self.state not in ("ready", "done"):
            self.finish(f"Arrêté à l'essai {self.k + 1} sur {len(self.order)} : ce qui a été "
                        "enregistré est sauvegardé")

    def begin(self, state):
        """Show ``state``'s card; its event and timing start when it is drawn."""
        self.state, self.phase_end = state, None
        block, gesture = self.order[self.k]
        spec = self.specs[gesture]
        values = {"gesture": gesture, "gesture_name": spec["name"],
                  "fingers": "+".join(spec.get("fingers", [])) or "n/a",
                  "motion": spec["motion"],
                  "block": block + 1, "trial": self.k + 1}
        duration = {"cue": self.protocol.prepare, "go": self.protocol.go,
                    "rest": self.protocol.rest, "break": None}[state]
        if state == "break":  # before the next block: its number only
            values = {"block": block + 1}
        self.pending = (state, duration, values)
        self.update()

    def tick(self):
        if self.state == "cue":
            self.update()  # the countdown
        if self.phase_end is None or time.monotonic() < self.phase_end:
            return
        if self.state == "cue":
            self.begin("go")
        elif self.state == "go":
            self.begin("rest")
        elif self.state == "rest":
            self.k += 1
            if self.k == len(self.order):
                self.finish(f"Terminé : {len(self.order)} essais enregistrés. Merci !")
            elif self.order[self.k][0] != self.order[self.k - 1][0] and \
                    self.order[self.k][0] % self.protocol.break_every == 0:
                self.begin("break")
            else:
                self.begin("cue")

    def finish(self, message):
        self.state, self.phase_end, self.pending, self.message = "done", None, None, message
        self.recorder.record(False)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QtGui.QColor(ui.BACKGROUND_COLOR))
        w, h = self.width(), self.height()
        font = qp.font()
        if self.state in ("ready", "done", "break"):
            font.setPointSize(ui.FONT_MESSAGE)
            qp.setFont(font)
            qp.setPen(QtGui.QColor(ui.TEXT_COLOR))
            if self.state == "ready":
                text = (f"{self.protocol.name} : {len(self.specs)} gestes x "
                        f"{self.protocol.repetitions}, environ {self.protocol.duration() / 60:.0f} min\n\n"
                        "Quand la main s'allume, faites son geste une seule fois :\n"
                        "pincement, commencez la main ouverte ; extension, commencez la main "
                        "fermée.\nLes doigts marqués en blanc rejoignent le pouce (pincement) ou "
                        "suivent la flèche (extension). Relâchez ensuite la main.\n\n"
                        "Appuyez sur Espace pour commencer")
            elif self.state == "break":
                text = (f"Pause\n\nbloc {self.order[self.k][0] + 1} sur {self.protocol.repetitions} "
                        "ensuite\n\nEspace pour continuer")
            else:
                text = ""
            qp.drawText(self.rect().adjusted(40, 40, -40, -40), QtCore.Qt.AlignmentFlag.AlignCenter,
                        text + ("\n\n" + self.message if self.message else ""))
            if self.state == "break":
                self.stamp()
            return
        block, gesture = self.order[self.k]
        spec = self.specs[gesture] if self.state != "rest" else cues.GESTURES["rest"]
        lit = self.state == "go"
        accent = ui.TARGET_COLORS[self.k % len(ui.TARGET_COLORS)]
        font.setPointSize(ui.FONT_MESSAGE)
        qp.setFont(font)
        qp.setPen(QtGui.QColor(accent if lit else ui.TEXT_COLOR))
        qp.drawText(QtCore.QRectF(0, 0.04 * h, w, 0.1 * h), QtCore.Qt.AlignmentFlag.AlignCenter,
                    "repos" if self.state == "rest" else spec["name"])
        cues.draw_card(qp, QtCore.QRectF(0.1 * w, 0.14 * h, 0.8 * w, 0.66 * h), spec, lit,
                       accent=accent, dim=ui.FLAT_COLOR)
        if self.state == "cue":  # the seconds to the go
            left = self.protocol.prepare if self.phase_end is None else self.phase_end - time.monotonic()
            font.setPointSize(ui.FONT_COUNTDOWN)
            qp.setFont(font)
            qp.setPen(QtGui.QColor(ui.TEXT_COLOR))
            qp.drawText(QtCore.QRectF(0, 0.8 * h, w, 0.12 * h), QtCore.Qt.AlignmentFlag.AlignCenter,
                        str(max(1, int(np.ceil(left)))))
        elif self.state == "go":  # the starting position, where the countdown was
            motion = spec.get("motion")
            hint = ({"pinch": "main ouverte", "extend": "main fermée"}.get(motion)
                    if isinstance(motion, str) else None)
            if hint:
                font.setPointSize(ui.FONT_MESSAGE)
                qp.setFont(font)
                qp.setPen(QtGui.QColor(accent))
                qp.drawText(QtCore.QRectF(0, 0.8 * h, w, 0.12 * h),
                            QtCore.Qt.AlignmentFlag.AlignCenter, hint)
        font.setPointSize(ui.FONT_INFO)
        qp.setFont(font)
        qp.setPen(QtGui.QColor(ui.FLAT_COLOR))
        qp.drawText(QtCore.QRectF(0, 0.92 * h, w, 0.06 * h), QtCore.Qt.AlignmentFlag.AlignCenter,
                    f"bloc {block + 1} / {self.protocol.repetitions}    essai {self.k + 1} / "
                    f"{len(self.order)}    Échap pour arrêter")
        self.stamp()

    def stamp(self):
        """The pending card was just drawn: its event, and when its phase ends."""
        if self.pending is None:
            return
        state, duration, values = self.pending
        self.pending = None
        now = time.monotonic()
        self.recorder.add_event(now, state, duration, **values)
        if duration is not None:
            self.phase_end = now + duration


camera = int(args.camera) if args.camera.isdigit() else args.camera
with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    protocol = cues.Protocol.load(args.protocol) if args.protocol else None
    with HandTracker(camera) as tracker, wb.stream() as stream:
        app = ui.application()
        window = Window(wb, stream, tracker, EVENT_COLUMNS if protocol else None,
                        event_descriptions(protocol) if protocol else None)
        window.show()
        if protocol:
            cue_window = CueWindow(protocol, window.recorder)
            if args.cue_fullscreen:
                cue_window.showFullScreen()
            else:
                cue_window.resize(900, 800)
                cue_window.show()
        if args.duration:
            QtCore.QTimer.singleShot(int(args.duration * 1000), app.quit)
        app.exec()
        window.recorder.finish()
