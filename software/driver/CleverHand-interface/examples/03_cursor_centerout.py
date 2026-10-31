"""Center-out myoelectric cursor control with a fixed Wiener filter.

    python 03_cursor_centerout.py --subject 01 [--session 01] [--bids-root data]
        [--port PORT] [--modules 13] [--calib-reps 3] [--reps 3] [--decoder FILE]
        [--speed-power 1.5] [--deadzone 0.6] [--screen-width-cm 52.7] [--windowed]

Calibration, then test, with a decoder that does not adapt in between:

1. Calibration (``acq-calib``): the 8 targets lit one at a time, each ``--calib-reps`` times in
   shuffled blocks. A guide cursor travels from the centre to the lit target in 2 s; the
   participant contracts as if pushing the cursor there, and relaxes when it jumps back to
   the centre (2 s rest). A Wiener filter (clvhd.decoder: the log of the 60 Hz EMG
   envelope over the last 100 ms, each channel z-scored, plus a bias) is fitted on it, its ridge penalty picked by
   cross-validation over the cues; the cross-validated R^2 is shown. C runs the calibration
   again, Space goes on. Its output then goes through a gain per target direction (so each
   direction reaches the calibration's speed, undoing the ridge's shrinkage), a dead-zone
   (the decoded speed at rest, or --deadzone) and, with --speed-power p > 1, a speed curve:
   the calibrated contraction keeps its speed, a stronger one goes faster than
   proportionally, a weaker one slower. Calibrating with light contractions leaves room to
   go faster.
2. Test (``acq-online``): ``--reps`` blocks of the 8 targets, shuffled. Each reach: the
   cursor held at the centre for 1 s with the target shown, then the go (the target lights
   up): the decoded velocity moves the cursor, and holding it 0.5 s in the target acquires
   it (green); after 10 s the reach times out (grey). The targets take the module colours
   of the live plot in turn, but green.

``--decoder`` skips the calibration with a decoder saved by an earlier run. Escape quits (the
phase in progress is not saved). When a session starts, the wristband's contact is checked on
3 s at rest: flat and poor-contact channels are left out of the decoder (the EMG files keep
every channel).

Per phase, in the BIDS dataset: emg/ (raw EMG, EDF or BDF, built at the end from the stream
written to sourcedata/ as it arrives, then deleted) and _events.tsv (calibration cues; reaches
with their outcome, time to target, acquisition time, path efficiency and target entries),
beh/ (the 60 Hz task: target, cursor, velocity) and derivatives/centerout/ (.npz: the
envelopes, and the decoder fitted on the calibration).
"""
import argparse
import time

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import clvhd
from clvhd import bids, centerout, coadapt, ui
from clvhd.centerout import FEEDBACK_S, HOLD_S, REACH_S, READY_S
from clvhd.decoder import WienerFilter, channel_indices

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ui.add_task_arguments(p, task="centerout")
p.add_argument("--calib-reps", type=int, default=3, help="calibration cues per target")
p.add_argument("--reps", type=int, default=3, help="test reaches per target")
p.add_argument("--decoder", help="decoder .npz of an earlier calibration: skip the calibration")
p.add_argument("--distance", type=float, default=0.7,
               help="target distance from the centre, fraction of the field's half height")
p.add_argument("--target-radius-cm", type=float, default=1.0)
p.add_argument("--speed-power", type=float,
               help="speed curve exponent: above 1, contractions stronger than the calibration's "
                    "go faster than proportionally (try 1.5; default: the decoder's, 1 when "
                    "calibrating)")
p.add_argument("--deadzone", type=float,
               help="decoded speeds below this make no motion (cm/s; default: the decoder's, "
                    "from its calibration at rest)")
args = p.parse_args()

PIPELINE = "centerout"
CONTACT_S = 3.0  # s at rest the contact check looks at, before the session
TASK_DESCRIPTION = ("Center-out myoelectric cursor control: reach 8 targets with a cursor whose "
                    "velocity is decoded from the EMG by a Wiener filter fitted on a cued "
                    "calibration and then fixed.")
# the cursor tasks' look (clvhd.ui): the field a tablet's screen, the targets app icons, each
# lit in the next module colour, an acquired one green, a timed-out one grey, the others
# outlined in the flat grey
HIT_COLOR, TARGET_COLORS = ui.HIT_COLOR, ui.TARGET_COLORS
EVENT_COLUMNS = ["onset", "duration", "sample", "trial_type", "target", "target_x", "target_y",
                 "success", "time_to_target", "acquisition_time", "path_efficiency", "entries"]
EVENT_DESCRIPTIONS = {
    "onset": {"Description": "Time from the first EMG sample of the file.", "Units": "s"},
    "duration": {"Units": "s"},
    "sample": {"Description": "EMG sample at the onset (zero-based)."},
    "trial_type": {"Levels": {
        "cue": "Calibration: the target lit, the guide cursor travelling to it (duration).",
        "ready": "Test: the cursor held at the centre, the target shown.",
        "reach": "Test: from the go to the acquisition or the timeout (duration)."}},
    "target": {"Description": "Target index: 0 to the right, then counter-clockwise by 45 deg."},
    "target_x": {"Description": "Target centre, right of the screen centre.", "Units": "cm"},
    "target_y": {"Description": "Target centre, above the screen centre.", "Units": "cm"},
    "success": {"Description": "1 when the target was acquired (held 0.5 s) before the timeout."},
    "time_to_target": {"Description": "First entry in the target from the go.", "Units": "s"},
    "acquisition_time": {"Description": "End of the 0.5 s hold from the go.", "Units": "s"},
    "path_efficiency": {"Description": "Straight distance from the centre to the target's edge "
                                       "over the cursor's path to the entry that was held."},
    "entries": {"Description": "Times the cursor entered the target."},
}
BEH_COLUMNS = {
    "time": ("Time from the first EMG sample of the run's EMG file", "s"),
    "phase_time": ("Time from the start of the calibration or test", "s"),
    "state": ("rest or cue (calibration); ready, reach or feedback (test)", None),
    "target": ("Index of the target shown, n/a when none", None),
    "cursor_x": ("Cursor after this frame (the guide during the calibration), right of the "
                 "screen centre", "cm"),
    "cursor_y": ("As cursor_x, above the screen centre", "cm"),
    "velocity_x": ("Calibration: the label (the guide's velocity); test: the decoded velocity "
                   "(direction gain, dead-zone, speed curve)", "cm/s"),
    "velocity_y": ("As velocity_x, vertical", "cm/s"),
    "fit": ("Calibration: 1 when the frame was in the decoder's fit", None),
}


class Experiment(ui.TaskWindow):
    def __init__(self, wb, stream, channels, decoder):
        super().__init__(wb, stream, args, "Center-out", channels)
        self.decoder = decoder
        self.rng = np.random.default_rng(args.seed)
        self.phase = "online" if decoder else "calib"
        self.summary = ""

    def layout_task(self):
        self.positions = centerout.targets(args.distance * self.half[1])
        self.limit = self.half - args.cursor_radius_cm
        return f", targets at {args.distance * self.half[1]:.1f} cm"

    def icon(self, qp, pos, color, outline=False):
        """A target at ``pos`` (cm): an app icon whose sides touch its hit circle."""
        ui.draw_icon(qp, self.to_px(pos), 2 * args.target_radius_cm * self.px_per_cm, color, outline)

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.state == "running":
            qp.fillRect(self.rect(), QtGui.QColor(ui.BACKGROUND_COLOR))
            ui.draw_tablet(qp, self.field, self.px_per_cm)
            for k, pos in enumerate(self.positions):
                if k == self.shown:
                    color = TARGET_COLORS[self.lit % len(TARGET_COLORS)]
                    fill = ({"hit": HIT_COLOR, "miss": ui.FLAT_COLOR}.get(self.outcome, color)
                            if self.outcome is not None else color)
                    # before the go, the target is outlined only
                    lit = self.sub != "ready"
                    self.icon(qp, pos, fill if lit else color, outline=not lit)
                else:
                    self.icon(qp, pos, ui.FLAT_COLOR, outline=True)
            self.draw_cursor(qp, self.cursor)
            return
        if self.state == "done":
            text = f"Terminé, merci !\n\n{self.summary}\n\nÉchap pour fermer"
        elif self.state == "fitted":
            text = f"{self.summary}\n\nEspace : commencer le test\nC : calibrer à nouveau"
        elif self.phase == "calib":
            text = ("Calibration\n\nQuand une cible s'allume, contractez comme pour pousser le\n"
                    "curseur vers elle, et maintenez pendant son trajet.\n"
                    "Relâchez quand il est revenu au centre.\n\nEspace pour commencer")
        else:
            text = ("Test\n\nMettez le curseur dans la cible quand elle s'allume,\n"
                    "et restez-y jusqu'à ce qu'elle devienne verte.\n\nEspace pour commencer")
        self.draw_message(qp, text, channel_info=self.state != "done")

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            if self.state == "running":
                print(f"\n{self.phase} interrupted, not saved")
            QtWidgets.QApplication.quit()
        elif key == QtCore.Qt.Key.Key_Space and self.state in ("ready", "fitted"):
            if self.state == "fitted":
                self.phase = "online"
            self.start_phase()
        elif key == QtCore.Qt.Key.Key_C and self.state == "fitted":
            self.phase = "calib"
            self.start_phase()

    def start_phase(self):
        ent = bids.Entities(args.subject, args.task, args.session, self.phase)
        self.ent = bids.Entities(args.subject, args.task, args.session, self.phase,
                                 bids.next_run(args.bids_root, ent))
        self.cursor, self.shown, self.sub, self.outcome = np.zeros(2), -1, "rest", None
        self.events = []
        self.lit = -1  # targets lit so far, minus one: the lit one's colour
        if self.phase == "calib":
            self.cal = centerout.Calibration(centerout.block_order(self.rng, args.calib_reps),
                                             self.positions)
        else:
            self.order = centerout.block_order(self.rng, args.reps)
            self.reach, self.reaches = 0, []
            self.decoder.reset()
            self.begin_reach(0.0)
        self.start_recording(self.ent, TASK_DESCRIPTION)
        print(f"{self.ent.label()} started")

    def begin_reach(self, pt):
        self.sub, self.sub_t0, self.outcome = "ready", pt, None
        self.shown = int(self.order[self.reach])
        self.lit += 1
        self.cursor = np.zeros(2)

    def frame(self, now):
        pt = now - self.t_start
        u, dt = self.next_input(now)
        if self.phase == "calib":
            self.calib_frame(now, pt, u)
        else:
            self.online_frame(now, pt, u, dt)

    def record(self, now, pt, u, v, fit=0):
        self.frames.append({"t": now, "pt": pt, "state": self.sub,
                            "target": self.shown if self.shown >= 0 else None,
                            "y": self.cursor.copy(), "v": np.asarray(v, dtype=float), "u": u,
                            "fit": int(fit)})

    def calib_frame(self, now, pt, u):
        c = self.cal.at(pt)
        if c["target"] >= 0 and self.shown < 0:  # a cue lit
            self.lit += 1
            self.events.append({"t": now, "trial_type": "cue", "duration": self.cal.cue,
                                "target": c["target"]})
        self.shown, self.sub = c["target"], "cue" if c["target"] >= 0 else "rest"
        self.cursor = c["guide"]
        self.record(now, pt, u, c["velocity"], c["fit"])
        self.frames[-1]["cue"] = c["cue"]
        if pt >= self.cal.duration:
            self.end_phase(now)

    def online_frame(self, now, pt, u, dt):
        v = self.decoder.step(u)  # the history runs through ready and feedback too
        if self.sub == "ready":
            if pt - self.sub_t0 >= READY_S:
                self.sub, self.go, self.go_index, self.entered = "reach", pt, len(self.frames), None
                self.events.append({"t": now, "trial_type": "ready", "duration": READY_S,
                                    "target": self.shown})
        elif self.sub == "reach":
            self.cursor = np.clip(self.cursor + v * dt, -self.limit, self.limit)
        self.record(now, pt, u, v)
        if self.sub == "reach":
            # the hold, frame by frame; the metrics once, at the reach's end
            inside = np.linalg.norm(self.cursor - self.positions[self.shown]) <= args.target_radius_cm
            entered = self.entered if inside else None  # leaving the target resets the hold
            if inside and entered is None:
                entered = pt
            self.entered = entered
            held = entered is not None and pt - entered >= HOLD_S - 1e-9
            if held or pt - self.go >= REACH_S:
                reach = self.frames[self.go_index:]
                m = centerout.reach_metrics([f["pt"] - self.go for f in reach], [f["y"] for f in reach],
                                            self.positions[self.shown], args.target_radius_cm, HOLD_S)
                self.outcome = "hit" if m["success"] else "miss"
                self.reaches.append({"target": self.shown, **m})
                self.events.append({"t": now, "trial_type": "reach", "duration": pt - self.go,
                                    "target": self.shown,
                                    **{k: int(x) if k in ("success", "entries") else x
                                       for k, x in m.items()}})
                print(f"  reach {self.reach + 1}/{len(self.order)}, target {self.shown}: "
                      + (f"acquired in {m['acquisition_time']:.2f} s, path efficiency "
                         f"{m['path_efficiency']:.2f}" if m["success"] else "timeout"))
                self.sub, self.sub_t0 = "feedback", pt
        elif self.sub == "feedback" and pt - self.sub_t0 >= FEEDBACK_S:
            self.reach += 1
            if self.reach == len(self.order):
                self.end_phase(now)
            else:
                self.begin_reach(pt)

    def end_phase(self, now):
        self.state = "saving"
        f = self.frames
        U = np.array([fr["u"] for fr in f])
        extra = {}
        if self.phase == "calib":
            V = np.array([fr["v"] for fr in f])
            groups = np.array([fr["cue"] for fr in f])
            mask = np.array([fr["fit"] for fr in f], bool)
            self.decoder = WienerFilter.fit(U, V, groups, mask)
            set_output_options(self.decoder)
            self.summary = f"Calibration terminée : {describe(self.decoder)}"
            extra = {"fit": mask, "cue": groups}
        else:
            ok = [r for r in self.reaches if r["success"]]
            self.summary = (f"{len(ok)} cibles acquises sur {len(self.reaches)}"
                            + (f", en {np.median([r['acquisition_time'] for r in ok]):.1f} s "
                               f"(médiane), efficacité du trajet "
                               f"{np.median([r['path_efficiency'] for r in ok]):.2f}" if ok else ""))
        if save_phase(self, U, extra):
            print(f"{self.ent.label()} saved. {self.summary}")
        self.state = "fitted" if self.phase == "calib" else "done"


def save_phase(exp, U, extra):
    """The phase's BIDS files and derivatives; False when it recorded nothing."""
    root = args.bids_root
    ent = exp.ent
    written = exp.export()
    if written is None:
        return False
    t0 = exp.raw.t0
    fs = exp.stream.nominal_rate

    rows = []
    for e in exp.events:  # stamped at their end (the go, the acquisition or timeout): onset at the start
        start = e["t"] - e["duration"] if e["trial_type"] in ("ready", "reach") else e["t"]
        pos = np.round(exp.positions[e["target"]], 4) + 0.0  # no -0
        cols = {k: v for k, v in e.items() if k != "t"}
        rows.append(bids.event_row(start, t0, fs, **{**cols, "duration": round(e["duration"], 6),
                                                     "target_x": pos[0], "target_y": pos[1]}))
    bids.write_events(written["emg"].with_name(f"{ent.label()}_events.tsv"), rows, EVENT_COLUMNS,
                      EVENT_DESCRIPTIONS)

    f = exp.frames
    beh = bids.write_beh(
        root, ent, BEH_COLUMNS,
        [[fr["t"] - t0, fr["pt"], fr["state"], fr["target"], *fr["y"], *fr["v"],
          fr["fit"] if exp.phase == "calib" else None] for fr in f],
        Phase={"calib": "calibration", "online": "test"}[exp.phase],
        **exp.beh_sidecar(),
        TargetsCm=np.round(exp.positions, 4).tolist(), TargetRadiusCm=args.target_radius_cm,
        DirectionGains=np.round(exp.decoder.gains, 4).tolist(),
        DeadzoneCmPerS=round(exp.decoder.deadzone, 4), SpeedCurvePower=exp.decoder.power,
        CalibrationSpeedCmPerS=round(exp.decoder.speed, 4))
    bids.write_scans(root, ent, [written["emg"], beh], written["acq_time"])

    deriv_root = bids.write_derivative_description(root, PIPELINE, "clvhd 03_cursor_centerout.py")
    deriv = ent.folder(deriv_root) / "emg"
    deriv.mkdir(parents=True, exist_ok=True)
    # the decoder in use (fitted on this calibration, or the test's), with what the phase decoded
    labels = exp.stream.channel_labels
    exp.decoder.save(
        deriv / f"{ent.label()}_desc-decoder.npz",
        channels=[labels[c] for c in exp.channels], band=np.array(coadapt.BAND),
        time=np.array([fr["t"] - t0 for fr in f]), u=U,
        cursor=np.array([fr["y"] for fr in f]), velocity=np.array([fr["v"] for fr in f]),
        target=np.array([-1 if fr["target"] is None else fr["target"] for fr in f]),
        positions=exp.positions,
        envelope=f"u: EMG envelope in mV ({coadapt.BAND[0]}-{coadapt.BAND[1]} Hz band, mains "
                 "notch, rectified, 40 Hz low-pass, averaged over each 60 Hz frame)", **extra)
    return True


def set_output_options(decoder):
    """The command line's speed curve and dead-zone on ``decoder``."""
    if args.speed_power is not None:
        decoder.power = args.speed_power
    if args.deadzone is not None:
        decoder.deadzone = min(args.deadzone, 0.9 * decoder.speed)  # the curve needs speed above it


def describe(decoder):
    r2 = decoder.cv_r2
    return (f"R² en validation croisée : x {r2[0]:.2f}, y {r2[1]:.2f} (ridge {decoder.ridge:g})\n"
            f"gains de direction de {decoder.gains.min():.2f} à {decoder.gains.max():.2f}, "
            f"zone morte {decoder.deadzone:.2f} cm/s, courbe de vitesse puissance {decoder.power:g} "
            f"à {decoder.speed:.2f} cm/s")


def contact_check(stream):
    """The channels in contact, from CONTACT_S s of the stream, the forearm relaxed."""
    print(f"Contact check: keep the forearm relaxed for {CONTACT_S:g} s", flush=True)
    chunks, n, t_end = [], 0, time.monotonic() + CONTACT_S + 10
    while n < CONTACT_S * stream.nominal_rate:
        if time.monotonic() > t_end:
            raise SystemExit("Too few samples for the contact check: is the wristband streaming?")
        data = stream.read(0.1)[2]
        chunks.append(data)
        n += len(data)
    contact = clvhd.check_contact(np.concatenate(chunks))
    labels = stream.channel_labels
    for c, st in enumerate(contact):
        if st:
            print(f"  {labels[c]}: {st}, left out of the decoder")
    channels = [c for c, st in enumerate(contact) if not st]
    if not channels:
        raise SystemExit("No channel in contact: check the wristband")
    print(f"Decoder on {len(channels)} of {len(labels)} channels")
    return channels


with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    with wb.stream() as stream:
        decoder = None
        if args.decoder:
            try:
                decoder = WienerFilter.load(args.decoder)
                channels = channel_indices(args.decoder, list(stream.channel_labels))
            except ValueError as e:
                raise SystemExit(e) from None
            set_output_options(decoder)
            print(f"Decoder from {args.decoder}, {len(channels)} channels: {describe(decoder)}")
        else:
            channels = contact_check(stream)
        bids.write_dataset(args.bids_root)
        bids.write_participant(args.bids_root, args.subject)
        app = ui.application()
        exp = Experiment(wb, stream, channels, decoder)
        exp.open()
        app.exec()
