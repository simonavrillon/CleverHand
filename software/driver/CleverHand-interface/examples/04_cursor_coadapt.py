"""Co-adaptive myoelectric cursor tracking, after Madduri et al., Nat. Mach. Intell. 8, 372 (2026).

    python 04_cursor_coadapt.py --subject 01 --decoder CALIBRATION [--session 01]
        [--bids-root data]
        [--port PORT] [--modules 13] [--screen-width-cm 52.7] [--windowed]

The participant moves a white cursor with their forearm EMG to follow a coloured
target along a pseudo-random sum-of-sines path (the examples' palette, clvhd.ui: a
module colour per trial, on the dark background, the task field outlined; the paper's
was a red target and a blue cursor on white). The cursor velocity is a
linear (Wiener filter) decoder of the 60 Hz EMG envelope, v = D x, which
adapts every 20 s: D* = argmin ||D x - (tau - y)/H||^2 + lambda ||D||^2 on the
last 20 s, then D = alpha D- + (1 - alpha) D* (SmoothBatch). Six 5-minute trials in
two blocks (break between them) run the 3 conditions once per block, shuffled: the
decoder fixed (alpha = 1, the control: D* fitted and logged, never used), slow
(0.75) or fast (0.25), at the low penalty. Every trial ends with 20 s on the
calibration's decoder, unannounced (the probe): still in control then, the decoder
learnt the participant; out of control, the participant learnt the decoder. The
participant is not told the conditions. Space starts a trial, Escape quits (the trial
in progress is run again; what it recorded is kept in derivatives/coadapt/ as
_desc-interrupted...npz); run the command again to resume at the first trial not
saved. Every 20 s the console prints the window's tracking error, effort (the
envelope above rest, 1 = the calibration's contractions), and the decoder's EMG part
against the calibration's: gain and direction (1: the same mapping).

Not in the paper: the decoder is the one of 03_cursor_centerout.py
(clvhd.decoder.WienerFilter), from its calibration at the start of the same session
(--decoder, its _desc-decoder.npz): its channels (those in contact), envelope band and
features x (log envelopes, z-scored, over the last 100 ms, and a constant), and every
trial starts from its fit, so the cursor follows the participant from the start (the
paper's random D1 / D2 hardly moved it for 20 s, while the participant, seeing nothing,
contracted harder than they would later, and the first fit learnt from that). The
output stage of 03 is left out: the updates refit D's EMG weights, its constant term
staying the calibration's (refitted, it chased the last 20 s's mean error and made the
cursor drift). The intended velocity
is (tau - y)/H, the error closed in H = 0.5 s rather than in one frame (the paper's
dt, which asked for hundreds of cm/s and left lambda alone to bring the fits back to
cursor speeds), and lambda is relative, like the calibration's ridge: low 1, high 10
times the features' mean energy (clvhd.coadapt). The channels, shown on the start and break
screens, are the same for every trial (the EMG files still hold every channel).

Per trial, in the BIDS dataset: emg/ (raw EMG, EDF or BDF, built at the end from the
stream written to sourcedata/ as it arrives, then deleted) and _events.tsv (trial
start with its condition, end of the speed ramp, decoder updates, probe, cursor
resets), beh/ (the 60 Hz task: target, cursor, velocity, effort) and
derivatives/coadapt/ (.npz with the EMG envelopes, every decoder D and D*, the 20 s
windows' metrics, and the trajectory). The session plan (conditions, decoder) is in
derivatives/coadapt/.../_desc-plan.json.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import clvhd
from clvhd import bids, coadapt, decoder, ui
from clvhd.coadapt import RAMP_S, UPDATE_S

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ui.add_task_arguments(p, task="cursor")
p.add_argument("--trial-seconds", type=float, default=coadapt.TRIAL_S)
p.add_argument("--break-seconds", type=float, default=300, help="break between the blocks")
p.add_argument("--trials", type=int, default=0, help="stop after this many trials (0: all)")
p.add_argument("--amplitude", type=float, default=0.7,
               help="target excursion, fraction of the half field on each axis")
p.add_argument("--decoder", required=True, metavar="CALIBRATION",
               help="_desc-decoder.npz of a 03_cursor_centerout.py calibration of this session")
p.add_argument("--autostart", action="store_true", help="start each trial without waiting for space")
args = p.parse_args()

PIPELINE = "coadapt"
TASK_DESCRIPTION = ("Co-adaptive myoelectric cursor tracking (Madduri et al. 2026): follow a "
                    "target with a cursor whose velocity is decoded from the EMG by an adaptive "
                    "linear decoder.")
# the cursor tasks' look (clvhd.ui), as in 03: the field a tablet's screen, the target an app
# icon in the next module colour each trial, the cursor white
EVENT_COLUMNS = ["onset", "duration", "sample", "trial_type", "value", "learning_rate", "alpha",
                 "penalty", "lambda"]
EVENT_DESCRIPTIONS = {
    "onset": {"Description": "Time from the first EMG sample of the file.", "Units": "s"},
    "duration": {"Units": "s"},
    "sample": {"Description": "EMG sample at the onset (zero-based)."},
    "trial_type": {"Levels": {
        "trial_start": "Target and cursor start moving; the decoder is the calibration's.",
        "ramp_end": f"End of the {RAMP_S:g} s ramp: target and cursor at full speed.",
        "decoder_update": "A new decoder D = alpha D- + (1 - alpha) D* takes over (value: update "
                          "number, D* fitted on the previous 20 s, on the intended velocity "
                          "(target - cursor) / 0.5 s).",
        "probe_start": f"The decoder goes back to the calibration's for the last "
                       f"{coadapt.PROBE_S:g} s, unannounced.",
        "cursor_reset": "Cursor held at the screen edge for 3.33 s, moved to the centre.",
        "trial_end": "End of the trial."}},
    "learning_rate": {"Description": "SmoothBatch learning rate condition.",
                      "Levels": {"fixed": "alpha = 1: never updated (the control)",
                                 "slow": "alpha = 0.75", "fast": "alpha = 0.25"}},
    "alpha": {"Description": "Weight of the previous decoder in the SmoothBatch update."},
    "penalty": {"Description": "Decoder effort penalty condition.",
                "Levels": {"low": "lambda = 1", "high": "lambda = 10"}},
    "lambda": {"Description": "Penalty on ||D||_F^2 in the decoder cost (its constant's "
                              "column excepted), times the mean energy of the features of the "
                              "20 s fitted."},
}
BEH_COLUMNS = {
    "time": ("Time from the first EMG sample of the run's EMG file", "s"),
    "trial_time": ("Time from the start of the trial", "s"),
    "target_x": ("Target position, right of the screen centre", "cm"),
    "target_y": ("Target position, above the screen centre", "cm"),
    "cursor_x": ("Cursor position before this frame's move (on screen while u was measured), "
                 "right of the screen centre", "cm"),
    "cursor_y": ("As cursor_x, above the screen centre", "cm"),
    "velocity_x": ("Decoded cursor velocity D x (times the ramp speed) of this frame", "cm/s"),
    "velocity_y": ("As velocity_x, vertical", "cm/s"),
    "dt": ("Time since the previous frame: the cursor moved by velocity * dt", "s"),
    "decoder": ("Number of the decoder in use: 0 the initial one, k after the k-th update", None),
    "cursor_reset": ("1 when the cursor was moved back to the centre after this frame", None),
    "effort": ("Mean over the decoder's channels of the envelope above its resting floor, over "
               "its mean during the calibration's cues (1: contracting as in the calibration)", None),
    "probe": ("1 in the last seconds, on the calibration's decoder", None),
}


class Experiment(ui.TaskWindow):
    def __init__(self, wb, stream, plan, first_trial):
        super().__init__(wb, stream, args, "Cursor tracking", plan["channels"], tuple(plan["band"]))
        self.plan = plan
        self.n_trials = len(plan["conditions"]) if not args.trials else \
            min(len(plan["conditions"]), first_trial + args.trials)
        self.trial = first_trial  # index of the next or current trial
        self.wf = decoder.WienerFilter.load(plan["decoder"])  # its features, and the start
        self.D0 = self.wf.weights.T.copy()  # (2, features): the calibration's fit
        self.break_end = None

    def layout_task(self):
        self.target_radius = args.cursor_radius_cm * np.sqrt(3)  # 3 times the cursor's area
        return ""

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.state == "running":
            qp.fillRect(self.rect(), QtGui.QColor(ui.BACKGROUND_COLOR))
            ui.draw_tablet(qp, self.field, self.px_per_cm)
            # the target an app icon, its sides touching the target's circle
            ui.draw_icon(qp, self.to_px(self.tau), 2 * self.target_radius * self.px_per_cm,
                         ui.TARGET_COLORS[self.trial % len(ui.TARGET_COLORS)])
            self.draw_cursor(qp, self.cursor.y)
            return
        if self.state == "break":
            left = max(0.0, self.break_end - time.monotonic()) if self.break_end is not None else 0.0
            text = f"Pause : {int(left // 60)}:{int(left % 60):02d}\n\nEspace pour continuer"
        elif self.state == "done":
            text = "Terminé, merci !\n\nÉchap pour fermer"
        else:
            text = (f"Essai {self.trial + 1} sur {len(self.plan['conditions'])}\n\n"
                    "Suivez la cible colorée avec le curseur blanc.\n"
                    "Contractez comme pendant la calibration.\n\nEspace pour commencer")
        self.draw_message(qp, text, channel_info=self.state != "done")

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            if self.state == "running":
                path = save_interrupted(self)
                print(f"\nTrial {self.trial + 1} interrupted, run again next time; what it "
                      f"recorded: {path}")
            QtWidgets.QApplication.quit()
        elif key == QtCore.Qt.Key.Key_Space and self.state in ("ready", "break"):
            self.start_trial()

    def idle(self, now):
        if args.autostart and (self.state == "ready" or
                               (self.state == "break" and now >= self.break_end)):
            self.start_trial()
        else:
            super().idle(now)  # the text screens (break countdown)

    def start_trial(self):
        self.cond = self.plan["conditions"][self.trial]
        self.D = self.D0.copy()
        self.wf.reset()
        self.probing, self.probe_time, self.windows, self.window_start = False, None, [], 0
        rng = np.random.default_rng(self.plan["seed"] + 1000 + self.trial)  # new path each trial
        span = args.amplitude * (self.half - self.target_radius)
        self.traj = coadapt.Trajectory(rng, span, args.trial_seconds)
        self.cursor = coadapt.Cursor(self.half, args.cursor_radius_cm)
        self.tau = self.traj.position(0.0)
        self.decoders, self.optimal, self.update_times = [self.D.copy()], [], []
        self.resets = []
        self.next_update = UPDATE_S
        self.start_recording(entities(self.trial), TASK_DESCRIPTION)
        print(f"Trial {self.trial + 1}/{len(self.plan['conditions'])} started")

    def frame(self, now):
        trial_t = now - self.t_start
        e, dt = self.next_input(now)
        x = self.wf.push(e)
        self.tau = self.traj.position(trial_t)
        y = self.cursor.y.copy()  # the cursor the participant saw while making u
        v = self.D @ x * self.traj.speed(trial_t)  # the cursor ramps up with the target
        reset = self.cursor.step(v, dt, trial_t)  # y_t = y_t-1 + v_t dt
        effort = np.clip(e - self.wf.floor, 0, None).mean() / self.plan["effort_ref"]
        self.frames.append({"t": now, "trial_t": trial_t, "tau": self.tau, "y": y, "e": e, "x": x, "v": v,
                            "dt": dt, "decoder": 0 if self.probing else len(self.decoders) - 1,
                            "reset": int(reset), "effort": effort, "probe": int(self.probing)})
        if reset:
            self.resets.append(now)

        if not self.probing and trial_t >= args.trial_seconds - coadapt.PROBE_S:
            self.close_window(trial_t)
            self.probing, self.probe_time = True, now
            self.D = self.D0.copy()  # unannounced: the participant sees no change but the cursor's
            print(f"  {trial_t:5.1f} s: probe, the calibration's decoder")
        elif not self.probing and trial_t >= self.next_update:
            # D* on the last UPDATE_S s: intended velocity (tau - y) / H at each frame
            recent = [f for f in self.frames if f["trial_t"] > self.next_update - UPDATE_S]
            self.close_window(trial_t)
            X = np.array([f["x"] for f in recent]).T
            V = np.array([(f["tau"] - f["y"]) / coadapt.HORIZON_S for f in recent]).T
            D_star = coadapt.optimal_decoder(X, V, self.cond["lam"], self.D0[:, -1])
            self.D = coadapt.smoothbatch(self.D, D_star, self.cond["alpha"])
            self.optimal.append(D_star)
            self.decoders.append(self.D.copy())
            self.update_times.append(now)
            self.next_update += UPDATE_S
        if trial_t >= args.trial_seconds:
            self.close_window(trial_t)
            self.end_trial(now)

    def close_window(self, trial_t):
        """The metrics of the frames since the last window, printed and kept."""
        fr = [f for f in self.frames[self.window_start:] if f["trial_t"] > coadapt.RAMP_S]
        self.window_start = len(self.frames)
        if not fr:
            return
        m = coadapt.window_metrics([f["tau"] for f in fr], [f["y"] for f in fr],
                                   [f["effort"] for f in fr], np.array([f["x"] for f in fr]),
                                   self.D, self.D0)
        m.update(start=fr[0]["trial_t"], end=trial_t, probe=self.probing,
                 decoder=fr[-1]["decoder"])
        self.windows.append(m)
        print(f"  {m['start']:5.1f}-{m['end']:5.1f} s{' probe' if self.probing else ''}: error "
              f"{m['error']:4.1f} cm, effort {m['effort']:4.2f}, decoder {m['decoder']}: gain "
              f"{m['gain']:4.2f}, direction {m['direction']:+.2f}", flush=True)

    def end_trial(self, now):
        self.state = "saving"
        if save_trial(self, now):
            print(f"Trial {self.trial + 1} saved")
        else:  # nothing recorded: run it again
            self.state = "ready"
            return
        self.trial += 1
        if self.trial >= self.n_trials:
            self.state = "done"
            if args.autostart:
                QtWidgets.QApplication.quit()
        elif self.trial == len(self.plan["conditions"]) // 2:
            self.state, self.break_end = "break", time.monotonic() + args.break_seconds
        else:
            self.state = "ready"


def entities(trial):
    return bids.Entities(args.subject, args.task, args.session, run=f"{trial + 1:02d}")


def save_trial(exp, t_end):
    """The trial's BIDS files and derivatives; False when it recorded nothing."""
    root = args.bids_root
    ent = entities(exp.trial)
    written = exp.export()  # the EMG streamed to sourcedata/, block by block (then deleted from there)
    if written is None:
        return False
    t0 = exp.raw.t0
    fs = exp.stream.nominal_rate
    cond = exp.cond
    cond_cols = {"learning_rate": cond["rate"], "alpha": cond["alpha"], "penalty": cond["penalty"],
                 "lambda": cond["lam"]}

    def event(t, kind, value=None, **extra):
        return bids.event_row(t, t0, fs, **{"duration": 0, "trial_type": kind, "value": value, **extra})
    rows = [event(exp.t_start, "trial_start", duration=round(t_end - exp.t_start, 6), **cond_cols),
            event(exp.t_start + RAMP_S, "ramp_end"), event(t_end, "trial_end")]
    rows += [event(t, "decoder_update", k + 1) for k, t in enumerate(exp.update_times)]
    rows += [event(exp.probe_time, "probe_start")]
    rows += [event(t, "cursor_reset") for t in exp.resets]
    bids.write_events(written["emg"].with_name(f"{ent.label()}_events.tsv"), rows, EVENT_COLUMNS,
                      EVENT_DESCRIPTIONS)

    beh = bids.write_beh(
        root, ent, BEH_COLUMNS,
        [[fr["t"] - t0, fr["trial_t"], *fr["tau"], *fr["y"], *fr["v"], fr["dt"], fr["decoder"],
          fr["reset"], fr["effort"], fr["probe"]] for fr in exp.frames],
        Condition=cond_cols, **exp.beh_sidecar(), TargetRadiusCm=float(exp.target_radius),
        Decoder=exp.plan["decoder"], HorizonS=coadapt.HORIZON_S, ProbeS=coadapt.PROBE_S,
        EffortReferenceMv=exp.plan["effort_ref"])
    bids.write_scans(root, ent, [written["emg"], beh], written["acq_time"])
    save_derivative(exp, f"{ent.label()}_desc-coadapt.npz", t0)
    return True


def save_interrupted(exp):
    """What an interrupted trial recorded, to derivatives/ (its run is recorded again)."""
    exp.raw.flush()
    exp.close_window(exp.frames[-1]["trial_t"] if exp.frames else 0.0)
    stamp = time.strftime("%Y%m%d%H%M%S")
    return save_derivative(exp, f"{entities(exp.trial).label()}_desc-interrupted{stamp}.npz",
                           exp.raw.t0 if exp.raw.t0 is not None else exp.t_start)


def save_derivative(exp, name, t0):
    """The trial's 60 Hz frames, decoders and windows, as derivatives/coadapt/.../emg/``name``."""
    f, labels, w = exp.frames, exp.stream.channel_labels, exp.windows
    deriv = entities(exp.trial).folder(args.bids_root / "derivatives" / PIPELINE) / "emg"
    deriv.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "effort": np.array([fr["effort"] for fr in f]), "probe": np.array([fr["probe"] for fr in f]),
        **{f"window_{k}": np.array([m[k] for m in w]) for k in
           ("start", "end", "probe", "decoder", "error", "effort", "gain", "direction")},
        "probe_time": np.nan if exp.probe_time is None else exp.probe_time - t0,
        "effort_reference": exp.plan["effort_ref"],
        "time": np.array([fr["t"] - t0 for fr in f]), "trial_time": np.array([fr["trial_t"] for fr in f]),
        "target": np.array([fr["tau"] for fr in f]), "cursor": np.array([fr["y"] for fr in f]),
        "velocity": np.array([fr["v"] for fr in f]), "dt": np.array([fr["dt"] for fr in f]),
        "envelopes": np.array([fr["e"] for fr in f]),
        "decoder_index": np.array([fr["decoder"] for fr in f]),
        "decoders": np.array(exp.decoders), "optimal_decoders": np.array(exp.optimal),
        "update_time": np.array(exp.update_times) - t0,
        "decoder_channels": [labels[c] for c in exp.channels],  # the columns of the envelopes
        "condition": json.dumps(exp.cond), "traj_phases": np.array(exp.traj.phases),
        "traj_scale": exp.traj.scale, "horizon": coadapt.HORIZON_S, "decoder": exp.plan["decoder"],
        "envelope": f"envelopes in mV ({exp.plan['band'][0]:g}-{exp.plan['band'][1]:g} Hz band, mains "
                    "notch, rectified, 40 Hz low-pass, averaged over each 60 Hz frame); D's columns: "
                    "the decoder's features of them (clvhd.decoder.WienerFilter.push)"}
    np.savez_compressed(deriv / name, **payload)
    return deriv / name


def load_plan(stream):
    """The session plan (conditions, the decoder), created on the first run from the --decoder
    calibration, and reused to resume."""
    labels = list(stream.channel_labels)
    root = args.bids_root
    ent = bids.Entities(args.subject, args.task, args.session)
    deriv_root = root / "derivatives" / PIPELINE
    path = ent.folder(deriv_root) / f"{ent.session_prefix}_task-{args.task}_desc-plan.json"
    source = str(Path(args.decoder).resolve())
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["labels"] != labels:
            raise SystemExit(f"{path} was made for {len(plan['labels'])} channels, not {len(labels)}")
        if plan.get("decoder") != source or plan.get("horizon") != coadapt.HORIZON_S or \
                "effort_ref" not in plan:
            raise SystemExit(f"{path} was made with another decoder or version of the "
                             f"experiment ({plan.get('decoder')}): use another --session, or its "
                             "--decoder")
        return plan
    try:
        floor = decoder.WienerFilter.load(args.decoder).floor
        channels = decoder.channel_indices(args.decoder, labels)
    except ValueError as e:
        raise SystemExit(e) from None
    f = np.load(args.decoder)
    band = [float(b) for b in f["band"]] if "band" in f else list(coadapt.BAND)
    cues = f["fit"].astype(bool) & (f["target"] >= 0)
    effort_ref = float(np.clip(np.nan_to_num(f["u"][cues]) - floor, 0, None).mean())
    print(f"Decoder of {args.decoder}: {len(channels)} of {len(labels)} channels, "
          f"{band[0]:g}-{band[1]:g} Hz")
    entropy = np.random.SeedSequence().entropy
    seed = args.seed if args.seed is not None else \
        int(entropy % 2**31) if isinstance(entropy, int) else 0
    plan = {"seed": seed, "conditions": coadapt.session_plan(np.random.default_rng(seed)),
            "labels": labels, "channels": channels, "decoder": source, "band": band,
            "horizon": coadapt.HORIZON_S, "penalties": coadapt.PENALTIES,
            "effort_ref": effort_ref}  # mV: the calibration cues' mean envelope above rest
    bids.write_dataset(root)
    bids.write_derivative_description(root, PIPELINE, "clvhd 04_cursor_coadapt.py",
                                      ignore=("*.npz", "*_desc-plan.json"))
    bids.write_json(path, plan)
    return plan


with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
    wb.configure()
    with wb.stream() as stream:
        plan = load_plan(stream)
        done = [k for k in range(len(plan["conditions"])) if (entities(k).folder(args.bids_root)
                / "beh" / f"{entities(k).label()}_beh.tsv").exists()]
        first = next(k for k in range(len(plan["conditions"]) + 1) if k not in done)
        if first == len(plan["conditions"]):
            raise SystemExit("Every trial of this session is already saved")
        if first:
            print(f"Resuming at trial {first + 1}")
        bids.write_participant(args.bids_root, args.subject)
        app = ui.application()
        exp = Experiment(wb, stream, plan, first)
        exp.open()
        app.exec()
