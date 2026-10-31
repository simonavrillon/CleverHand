"""Co-adaptive myoelectric cursor tracking, after Madduri et al., Nat. Mach. Intell. 8, 372 (2026).

    python 04_cursor_coadapt.py --subject 01 --decoder CALIBRATION [--session 01]
        [--protocol coadapt|gain] [--bids-root data]
        [--port PORT] [--modules 13] [--screen-width-cm 52.7] [--windowed]

The participant moves a white cursor with their forearm EMG to follow a coloured target along
a pseudo-random sum-of-sines path; the cursor velocity is a linear decoder of the 60 Hz EMG
envelope (the calibration's, --decoder) that adapts every 20 s (SmoothBatch; pieces in
clvhd.coadapt). Six 5-minute trials in two blocks run the 3 learning-rate conditions once per
block, shuffled, each ending with 20 s on the calibration's decoder, unannounced (the probe).
--protocol gain instead keeps the calibration's decoder and scales its EMG weights by a gain
that a staircase moves every 20 s, towards the largest gain whose tracking error stays within
10 % of the reference trials' (at gain 1): 2-minute trials, two references, three pairs of a
low- and a high-start trial (each start's staircase carried over between its trials), a last
reference; no probe. Space starts a trial, Escape quits (the trial is run again; what it
recorded goes to derivatives/coadapt/ as _desc-interrupted...npz), and a rerun resumes at the
first unsaved trial. Per trial, the BIDS dataset gets emg/ + _events.tsv, beh/ and derivatives/coadapt/
(.npz with the envelopes, every decoder, the windows' metrics and the trajectory). Full
description: the driver README, "Examples"; how this deviates from the paper:
docs/experiments.md.
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
p.add_argument(
    "--protocol",
    choices=("coadapt", "gain"),
    default="coadapt",
    help="coadapt: the decoder adapts (SmoothBatch); gain: the calibration's decoder at a "
    "staircase's gain",
)
p.add_argument(
    "--trial-seconds",
    type=float,
    help=f"default {coadapt.TRIAL_S:g} (coadapt) or {coadapt.GAIN_TRIAL_S:g} (gain)",
)
p.add_argument(
    "--break-seconds",
    type=float,
    default=300,
    help="break between the blocks (coadapt protocol; the gain protocol has none)",
)
p.add_argument("--trials", type=int, default=0, help="stop after this many trials (0: all)")
p.add_argument(
    "--amplitude",
    type=float,
    default=0.7,
    help="target excursion, fraction of the half field on each axis",
)
p.add_argument(
    "--decoder",
    required=True,
    metavar="CALIBRATION",
    help="_desc-decoder.npz of a 03_cursor_centerout.py calibration of this session",
)
p.add_argument(
    "--autostart", action="store_true", help="start each trial without waiting for space"
)
args = p.parse_args()
GAIN_PROTOCOL = args.protocol == "gain"
if args.trial_seconds is None:
    args.trial_seconds = coadapt.GAIN_TRIAL_S if GAIN_PROTOCOL else coadapt.TRIAL_S

PIPELINE = "coadapt"
TASK_DESCRIPTION = (
    "Myoelectric cursor tracking at a staircase's decoder gain: follow a target with a "
    "cursor whose velocity is decoded from the EMG by the calibration's linear decoder, its "
    "EMG weights scaled towards the largest gain that keeps the tracking error."
    if GAIN_PROTOCOL
    else "Co-adaptive myoelectric cursor tracking (Madduri et al. 2026): follow a "
    "target with a cursor whose velocity is decoded from the EMG by an adaptive "
    "linear decoder."
)
# the cursor tasks' look (clvhd.ui), as in 03: the field a tablet's screen, the target an app
# icon in the next module colour each trial, the cursor white
EVENT_COLUMNS = [
    "onset",
    "duration",
    "sample",
    "trial_type",
    "value",
    "learning_rate",
    "alpha",
    "penalty",
    "lambda",
    "protocol",
    "start",
    "gain",
]
EVENT_DESCRIPTIONS = {
    "onset": {"Description": "Time from the first EMG sample of the file.", "Units": "s"},
    "duration": {"Units": "s"},
    "sample": {"Description": "EMG sample at the onset (zero-based)."},
    "trial_type": {
        "Levels": {
            "trial_start": "Target and cursor start moving; the decoder is the calibration's (gain "
            "protocol: at the trial's starting gain).",
            "ramp_end": f"End of the {RAMP_S:g} s ramp: target and cursor at full speed.",
            "decoder_update": "A new decoder D = alpha D- + (1 - alpha) D* takes over (value: update "
            "number, D* fitted on the previous 20 s, on the intended velocity "
            "(target - cursor) / 0.5 s).",
            "gain_update": "The staircase sets the gain of the decoder's EMG weights (value: "
            "the new gain), from the error of the 20 s since the previous update, its first "
            f"{coadapt.SETTLE_S:g} s left out. At the trial's end: the gain the start's "
            "next trial begins with.",
            "probe_start": f"The decoder goes back to the calibration's for the last "
            f"{coadapt.PROBE_S:g} s, unannounced.",
            "cursor_reset": "Cursor held at the screen edge for 3.33 s, moved to the centre.",
            "trial_end": "End of the trial.",
        }
    },
    "learning_rate": {
        "Description": "SmoothBatch learning rate condition.",
        "Levels": {
            "fixed": "alpha = 1: never updated (the control)",
            "slow": "alpha = 0.75",
            "fast": "alpha = 0.25",
        },
    },
    "alpha": {"Description": "Weight of the previous decoder in the SmoothBatch update."},
    "penalty": {
        "Description": "Decoder effort penalty condition.",
        "Levels": {"low": "lambda = 1", "high": "lambda = 10"},
    },
    "lambda": {
        "Description": "Penalty on ||D||_F^2 in the decoder cost (its constant's "
        "column excepted), times the mean energy of the features of the "
        "20 s fitted."
    },
    "protocol": {
        "Levels": {
            "coadapt": "the decoder adapts every 20 s (SmoothBatch)",
            "gain": "the calibration's decoder, its EMG weights times a staircase's gain",
        }
    },
    "start": {
        "Description": "Gain protocol: the trial's staircase.",
        "Levels": {
            "reference": "gain 1 throughout: the reference error",
            **{k: f"the staircase started at gain {g:g}" for k, g in coadapt.GAIN_STARTS.items()},
        },
    },
    "gain": {"Description": "Gain protocol: the decoder gain at the trial's start."},
}
BEH_COLUMNS = {
    "time": ("Time from the first EMG sample of the run's EMG file", "s"),
    "trial_time": ("Time from the start of the trial", "s"),
    "target_x": ("Target position, right of the screen centre", "cm"),
    "target_y": ("Target position, above the screen centre", "cm"),
    "cursor_x": (
        "Cursor position before this frame's move (on screen while u was measured), "
        "right of the screen centre",
        "cm",
    ),
    "cursor_y": ("As cursor_x, above the screen centre", "cm"),
    "velocity_x": ("Decoded cursor velocity D x (times the ramp speed) of this frame", "cm/s"),
    "velocity_y": ("As velocity_x, vertical", "cm/s"),
    "dt": ("Time since the previous frame: the cursor moved by velocity * dt", "s"),
    "decoder_index": (
        "Number of the decoder in use: 0 the initial one, k after the k-th update",
        None,
    ),
    "cursor_reset": ("1 when the cursor was moved back to the centre after this frame", None),
    "effort": (
        "Mean over the decoder's channels of the envelope above its resting floor, over "
        "its mean during the calibration's cues (1: contracting as in the calibration)",
        None,
    ),
    "probe": ("1 in the last seconds, on the calibration's decoder", None),
    "gain": ("Gain of the decoder's EMG weights (1 in the coadapt protocol)", None),
}


class Experiment(ui.TaskWindow):
    def __init__(self, wb, stream, plan, first_trial):
        super().__init__(wb, stream, args, "Cursor tracking", plan["channels"], tuple(plan["band"]))
        self.plan = plan
        self.n_trials = (
            len(plan["conditions"])
            if not args.trials
            else min(len(plan["conditions"]), first_trial + args.trials)
        )
        self.trial = first_trial  # index of the next or current trial
        self.wf = decoder.WienerFilter.load(plan["decoder_file"])  # its features, and the start
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
            ui.draw_icon(
                qp,
                self.to_px(self.tau),
                2 * self.target_radius * self.px_per_cm,
                ui.TARGET_COLORS[self.trial % len(ui.TARGET_COLORS)],
            )
            self.draw_cursor(qp, self.cursor.y)
            return
        if self.state == "break":
            left = (
                max(0.0, self.break_end - time.monotonic()) if self.break_end is not None else 0.0
            )
            text = f"Pause : {int(left // 60)}:{int(left % 60):02d}\n\nEspace pour continuer"
        elif self.state == "done":
            text = "Terminé, merci !\n\nÉchap pour fermer"
        else:
            text = (
                f"Essai {self.trial + 1} sur {len(self.plan['conditions'])}\n\n"
                "Suivez la cible colorée avec le curseur blanc.\n"
                "Contractez comme pendant la calibration.\n\nEspace pour commencer"
            )
        self.draw_message(qp, text, channel_info=self.state != "done")

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            if self.state == "running":
                path = save_interrupted(self)
                print(
                    f"\nTrial {self.trial + 1} interrupted, run again next time; what it "
                    f"recorded: {path}"
                )
            QtWidgets.QApplication.quit()
        elif key == QtCore.Qt.Key.Key_Space and self.state in ("ready", "break"):
            self.start_trial()

    def idle(self, now):
        if args.autostart and (
            self.state == "ready" or (self.state == "break" and now >= self.break_end)
        ):
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
        self.changed = -np.inf  # trial time of the last gain change
        self.staircase, self.e_ref = None, None
        if GAIN_PROTOCOL and self.cond["start"] != "reference":
            self.e_ref = reference_error(self.plan, self.trial)
            self.staircase = coadapt.GainStaircase(
                e_ref=self.e_ref, **staircase_state(self.plan, self.trial, self.cond["start"])
            )
            self.D[:, :-1] *= self.staircase.gain
            self.decoders = [self.D.copy()]
        self.gains = [1.0 if self.staircase is None else self.staircase.gain]
        self.start_recording(trial_entities(self.trial), TASK_DESCRIPTION)
        print(
            f"Trial {self.trial + 1}/{len(self.plan['conditions'])} started"
            + (f", {self.cond['start']}: gain {self.gains[0]:.2f}" if GAIN_PROTOCOL else "")
        )

    def set_gain(self, gain, now, trial_t):
        """The calibration's decoder, its EMG weights times ``gain``, from now on."""
        self.D = self.D0.copy()
        self.D[:, :-1] *= gain
        self.decoders.append(self.D.copy())
        self.gains.append(gain)
        self.update_times.append(now)
        self.changed = trial_t
        print(f"  {trial_t:5.1f} s: gain {gain:.2f}")

    def frame(self, now):
        trial_t = now - self.t_start
        u, dt = self.next_input(now)
        x = self.wf.push(u)
        self.tau = self.traj.position(trial_t)
        y = self.cursor.y.copy()  # the cursor the participant saw while making u
        v = self.D @ x * self.traj.speed(trial_t)  # the cursor ramps up with the target
        reset = self.cursor.step(v, dt, trial_t)  # y_t = y_t-1 + v_t dt
        effort = np.clip(u - self.wf.floor, 0, None).mean() / self.plan["effort_reference"]
        self.frames.append(
            {
                "t": now,
                "trial_t": trial_t,
                "tau": self.tau,
                "y": y,
                "u": u,
                "x": x,
                "v": v,
                "dt": dt,
                "decoder_index": 0 if self.probing else len(self.decoders) - 1,
                "reset": int(reset),
                "effort": effort,
                "probe": int(self.probing),
                "gain": self.gains[-1],
            }
        )
        if reset:
            self.resets.append(now)

        if GAIN_PROTOCOL:  # no probe; the last update is the start's next trial's gain
            if trial_t >= self.next_update:
                m = self.close_window(trial_t)
                if self.staircase is not None and m is not None:
                    self.set_gain(self.staircase.update(m["error"]), now, trial_t)
                self.next_update += UPDATE_S
        elif not self.probing and trial_t >= args.trial_seconds - coadapt.PROBE_S:
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
            D_star = coadapt.optimal_decoder(X, V, self.cond["lambda"], self.D0[:, -1])
            self.D = coadapt.smoothbatch(self.D, D_star, self.cond["alpha"])
            self.optimal.append(D_star)
            self.decoders.append(self.D.copy())
            self.update_times.append(now)
            self.next_update += UPDATE_S
        if trial_t >= args.trial_seconds:
            self.close_window(trial_t)
            self.end_trial(now)

    def close_window(self, trial_t):
        """The metrics of the frames since the last window (after the ramp, and ``SETTLE_S``
        after a gain change), printed, kept and returned (None without frames)."""
        settled = max(coadapt.RAMP_S, self.changed + coadapt.SETTLE_S)
        frames = [fr for fr in self.frames[self.window_start :] if fr["trial_t"] > settled]
        self.window_start = len(self.frames)
        if not frames:
            return None
        m = coadapt.window_metrics(
            [fr["tau"] for fr in frames],
            [fr["y"] for fr in frames],
            [fr["effort"] for fr in frames],
            np.array([fr["x"] for fr in frames]),
            self.D,
            self.D0,
            [fr["trial_t"] for fr in frames],
            [fr["v"] for fr in frames],
            self.wf.activity(np.array([fr["u"] for fr in frames])),
        )
        m.update(
            start=frames[0]["trial_t"],
            end=trial_t,
            probe=self.probing,
            decoder_index=frames[-1]["decoder_index"],
        )
        self.windows.append(m)
        print(
            f"  {m['start']:5.1f}-{m['end']:5.1f} s{' probe' if self.probing else ''}: error "
            f"{m['error']:4.1f} cm, effort {m['effort']:4.2f}, decoder {m['decoder_index']}: gain "
            f"{m['gain']:4.2f}, direction {m['direction']:+.2f}, remnant "
            f"{np.mean(m['remnant']):.2f}, null space {m['null_space']:.2f}",
            flush=True,
        )
        return m

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
        # the gain protocol rests the same between every trial: its starts see the same drift
        elif not GAIN_PROTOCOL and self.trial == len(self.plan["conditions"]) // 2:
            self.state, self.break_end = "break", time.monotonic() + args.break_seconds
        else:
            self.state = "ready"


def trial_entities(trial):
    return bids.Entities(args.subject, args.task, args.session, run=f"{trial + 1:02d}")


def derivative_path(trial):
    """The saved trial's derivatives/coadapt/.../emg/_desc-coadapt.npz."""
    ent = trial_entities(trial)
    deriv = ent.folder(args.bids_root / "derivatives" / PIPELINE) / "emg"
    return deriv / f"{ent.label()}_desc-coadapt.npz"


def reference_error(plan, trial):
    """Gain protocol: the mean window error (cm) of the reference trials before ``trial``."""
    errors: list[float] = []
    for k in range(trial):
        if plan["conditions"][k]["start"] == "reference":
            with np.load(derivative_path(k)) as f:
                errors += f["window_error"].tolist()
    if not errors:
        raise SystemExit("No reference trial is saved before this one: run the session in order")
    return float(np.mean(errors))


def staircase_state(plan, trial, start):
    """Gain protocol: the staircase state the last trial of ``start`` before ``trial`` ended
    with, or its starting gain."""
    for k in reversed(range(trial)):
        if plan["conditions"][k]["start"] == start:
            with np.load(derivative_path(k)) as f:
                return json.loads(str(f["staircase"]))
    return {"gain": coadapt.GAIN_STARTS[start]}


def save_trial(exp, t_end):
    """The trial's BIDS files and derivatives; False when it recorded nothing."""
    root = args.bids_root
    ent = trial_entities(exp.trial)
    # the EMG streamed to sourcedata/, block by block (then deleted from there)
    written = exp.export()
    if written is None:
        return False
    t0 = exp.raw.t0
    fs = exp.stream.nominal_rate

    def event(t, kind, value=None, **extra):
        return bids.event_row(
            t, t0, fs, **{"duration": 0, "trial_type": kind, "value": value, **extra}
        )

    start = {
        "protocol": args.protocol,
        **exp.cond,
        **({"gain": exp.gains[0]} if GAIN_PROTOCOL else {}),
    }
    rows = [
        event(exp.t_start, "trial_start", duration=round(t_end - exp.t_start, 6), **start),
        event(exp.t_start + RAMP_S, "ramp_end"),
        event(t_end, "trial_end"),
    ]
    if GAIN_PROTOCOL:
        rows += [
            event(t, "gain_update", round(g, 4))
            for t, g in zip(exp.update_times, exp.gains[1:], strict=True)
        ]
    else:
        rows += [event(t, "decoder_update", k + 1) for k, t in enumerate(exp.update_times)]
    if exp.probe_time is not None:
        rows += [event(exp.probe_time, "probe_start")]
    rows += [event(t, "cursor_reset") for t in exp.resets]
    bids.write_events(
        written["emg"].with_name(f"{ent.label()}_events.tsv"),
        rows,
        EVENT_COLUMNS,
        EVENT_DESCRIPTIONS,
    )

    beh = bids.write_beh(
        root,
        ent,
        BEH_COLUMNS,
        [
            [
                fr["t"] - t0,
                fr["trial_t"],
                *fr["tau"],
                *fr["y"],
                *fr["v"],
                fr["dt"],
                fr["decoder_index"],
                fr["reset"],
                fr["effort"],
                fr["probe"],
                fr["gain"],
            ]
            for fr in exp.frames
        ],
        Protocol=args.protocol,
        Condition=exp.cond,
        **exp.beh_sidecar(),
        TargetRadiusCm=float(exp.target_radius),
        DecoderFile=exp.plan["decoder_file"],
        EffortReferenceMv=exp.plan["effort_reference"],
        **(
            {
                "ReferenceErrorCm": exp.e_ref,
                "GainTolerance": coadapt.GAIN_TOL,
                "SettleS": coadapt.SETTLE_S,
            }
            if GAIN_PROTOCOL
            else {"HorizonS": coadapt.HORIZON_S, "ProbeS": coadapt.PROBE_S}
        ),
    )
    bids.write_scans(root, ent, [written["emg"], beh], written["acq_time"])
    save_derivative(exp, f"{ent.label()}_desc-coadapt.npz", t0)
    return True


def save_interrupted(exp):
    """What an interrupted trial recorded, to derivatives/ (its run is recorded again)."""
    exp.raw.flush()
    exp.close_window(exp.frames[-1]["trial_t"] if exp.frames else 0.0)
    stamp = time.strftime("%Y%m%d%H%M%S")
    return save_derivative(
        exp,
        f"{trial_entities(exp.trial).label()}_desc-interrupted{stamp}.npz",
        exp.raw.t0 if exp.raw.t0 is not None else exp.t_start,
    )


def save_derivative(exp, name, t0):
    """The trial's 60 Hz frames, decoders and windows, as derivatives/coadapt/.../emg/``name``."""
    frames, labels, windows = exp.frames, exp.stream.channel_labels, exp.windows
    deriv = derivative_path(exp.trial).parent
    deriv.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "effort": np.array([fr["effort"] for fr in frames]),
        "probe": np.array([fr["probe"] for fr in frames]),
        **{
            f"window_{k}": np.array([m[k] for m in windows])
            for k in (
                "start",
                "end",
                "probe",
                "decoder_index",
                "error",
                "effort",
                "gain",
                "direction",
                "remnant",
                "null_space",
            )
        },
        "protocol": args.protocol,
        "gain": np.array([fr["gain"] for fr in frames]),
        "gains": np.array(exp.gains),  # the trial's initial gain, then each update's
        "reference_error": np.nan if exp.e_ref is None else exp.e_ref,
        # what the start's next trial continues from ("" outside the gain staircases)
        "staircase": "" if exp.staircase is None else json.dumps(exp.staircase.state()),
        "probe_time": np.nan if exp.probe_time is None else exp.probe_time - t0,
        "effort_reference": exp.plan["effort_reference"],
        "time": np.array([fr["t"] - t0 for fr in frames]),
        "trial_time": np.array([fr["trial_t"] for fr in frames]),
        "target_position": np.array([fr["tau"] for fr in frames]),
        "cursor": np.array([fr["y"] for fr in frames]),
        "velocity": np.array([fr["v"] for fr in frames]),
        "dt": np.array([fr["dt"] for fr in frames]),
        "envelopes": np.array([fr["u"] for fr in frames]),
        "decoder_index": np.array([fr["decoder_index"] for fr in frames]),
        "decoders": np.array(exp.decoders),
        "optimal_decoders": np.array(exp.optimal),  # empty (shape (0,)) in the gain protocol
        "update_time": np.array(exp.update_times) - t0,
        "decoder_channels": [labels[c] for c in exp.channels],  # the columns of the envelopes
        "condition": json.dumps(exp.cond),
        "trajectory_phases": np.array(exp.traj.phases),
        "trajectory_scale": exp.traj.scale,
        "horizon": coadapt.HORIZON_S,
        "decoder_file": exp.plan["decoder_file"],
        "envelope": f"envelopes in mV ({exp.plan['band'][0]:g}-{exp.plan['band'][1]:g} Hz band, mains "
        "notch, rectified, 40 Hz low-pass, averaged over each 60 Hz frame); D's columns: "
        "the decoder's features of them (clvhd.decoder.WienerFilter.push)",
    }
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
        made_for = plan.get("protocol", "coadapt")
        if made_for != args.protocol:
            raise SystemExit(
                f"{path} was made for the {made_for} protocol: use another --session, or "
                f"--protocol {made_for}"
            )
        if plan["labels"] != labels:
            raise SystemExit(
                f"{path} was made for {len(plan['labels'])} channels, not {len(labels)}"
            )
        if plan["decoder_file"] != source or plan["horizon"] != coadapt.HORIZON_S:
            raise SystemExit(
                f"{path} was made with another decoder or version of the "
                f"experiment ({plan['decoder_file']}): use another --session, or its "
                "--decoder"
            )
        return plan
    try:
        floor = decoder.WienerFilter.load(args.decoder).floor
        channels = decoder.channel_indices(args.decoder, labels)
    except ValueError as e:
        raise SystemExit(e) from None
    with np.load(args.decoder) as f:
        band = [float(b) for b in f["band"]]
        cues = f["fit"].astype(bool) & (f["target"] >= 0)
        envelopes = np.nan_to_num(f["envelopes"][cues])
    effort_reference = float(np.clip(envelopes - floor, 0, None).mean())
    print(
        f"Decoder of {args.decoder}: {len(channels)} of {len(labels)} channels, "
        f"{band[0]:g}-{band[1]:g} Hz"
    )
    entropy = np.random.SeedSequence().entropy
    seed = (
        args.seed
        if args.seed is not None
        else int(entropy % 2**31)
        if isinstance(entropy, int)
        else 0
    )
    rng = np.random.default_rng(seed)
    plan = {
        "seed": seed,
        "protocol": args.protocol,
        "conditions": coadapt.gain_session_plan(rng)
        if GAIN_PROTOCOL
        else coadapt.session_plan(rng),
        "labels": labels,
        "channels": channels,
        "decoder_file": source,
        "band": band,
        "horizon": coadapt.HORIZON_S,
        "penalties": coadapt.PENALTIES,
        # mV: the calibration cues' mean envelope above rest
        "effort_reference": effort_reference,
    }
    bids.write_dataset(root)
    bids.write_derivative_description(
        root, PIPELINE, "clvhd 04_cursor_coadapt.py", ignore=("*.npz", "*_desc-plan.json")
    )
    bids.write_json(path, plan)
    return plan


if __name__ == "__main__":  # imported by the tests
    with clvhd.Wristband(args.port, expected_modules=args.modules) as wb:
        wb.configure()
        with wb.stream() as stream:
            plan = load_plan(stream)
            done = [
                k
                for k in range(len(plan["conditions"]))
                if (
                    trial_entities(k).folder(args.bids_root)
                    / "beh"
                    / f"{trial_entities(k).label()}_beh.tsv"
                ).exists()
            ]
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
