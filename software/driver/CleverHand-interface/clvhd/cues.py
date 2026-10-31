"""Cued finger gestures, to record labelled EMG for gesture classifiers.

The participant performs each gesture once, cued by a card (the hand, its moving fingertips
and their motion), starting from the position the card shows: open hand for a pinch, closed
hand for an extension. Each trial: the card in grey with a countdown
(prepare), then lit (go: do it once), then the relaxed hand (rest).
``examples/02_live_emg_hand.py --protocol FILE`` shows the cards and records the EMG, the
camera's hand and the cue events together (clvhd.bids).

- ``GESTURES``: thumb pinches (with each finger, with index and middle, with all fingers),
  extensions (of each finger, of thumb + index, thumb + index + middle, all fingers), rest.
- ``Protocol``: the gestures, repetitions and timings, from a JSON file; ``trial_order`` the
  trials, every gesture once per block in a random order.
- ``draw_card``: a gesture's card, the hand of ``assets/hand.svg`` drawn with a QPainter.
- ``movement_windows``: offline, each trial's real movement from the camera's fingertips
  (people react 200-500 ms after the cue, variably): the windows to train classifiers on.
  ``clvhd-movements`` runs it on a recording's BIDS files.
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bids
from .hand import FINGERS, HAND_LANDMARKS, TIPS

# a gesture: its moving fingers and their motion (pinch: their tips meeting the thumb's,
# from an open hand; extend: the fingers straightening from a closed hand; rest: none)
_PINCHES = {
    "index": ("Thumb-index", ["index"]),
    "middle": ("Thumb-middle", ["middle"]),
    "ring": ("Thumb-ring", ["ring"]),
    "pinky": ("Thumb-pinky", ["pinky"]),
    "index_middle": ("Thumb-index-middle", ["index", "middle"]),
    "all": ("All-finger", ["index", "middle", "ring", "pinky"]),
}
_EXTENSIONS = {
    "thumb": ("Thumb", ["thumb"]),
    "index": ("Index", ["index"]),
    "middle": ("Middle", ["middle"]),
    "ring": ("Ring", ["ring"]),
    "pinky": ("Pinky", ["pinky"]),
    "thumb_index": ("Thumb-index", ["thumb", "index"]),
    "thumb_index_middle": ("Thumb-index-middle", ["thumb", "index", "middle"]),
    "all": ("All-finger", list(FINGERS)),
}
GESTURES = {
    **{
        f"pinch_{k}": {"name": f"{name} pinch", "fingers": ["thumb", *fingers], "motion": "pinch"}
        for k, (name, fingers) in _PINCHES.items()
    },
    **{
        f"extend_{k}": {"name": f"{name} extension", "fingers": fingers, "motion": "extend"}
        for k, (name, fingers) in _EXTENSIONS.items()
    },
    "rest": {"name": "Rest", "fingers": [], "motion": "rest"},
}


@dataclass
class Protocol:
    """What a cued recording runs. ``gestures``: names in ``GESTURES``, or dicts defining new
    ones (with a ``"id"``); ``repetitions``: blocks, each gesture once per block; times in s:
    ``prepare`` (the card in grey, counting down the seconds), ``go`` (lit: do it once),
    ``rest`` (relaxed hand);
    a break, ended by the participant, every ``break_every`` blocks."""

    name: str = "finger gestures"
    task: str = "gestures"
    gestures: list = field(default_factory=lambda: list(GESTURES))
    repetitions: int = 10
    prepare: float = 3.0
    go: float = 1.5
    rest: float = 2.0
    break_every: int = 2

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def specs(self):
        """{gesture id: spec} of the protocol's gestures, in order."""
        out = {}
        for g in self.gestures:
            if isinstance(g, str):
                if g not in GESTURES:
                    raise ValueError(f"unknown gesture {g!r} (known: {', '.join(GESTURES)})")
                out[g] = GESTURES[g]
            else:
                out[g["id"]] = {k: v for k, v in g.items() if k != "id"}
        return out

    def trial_order(self, rng):
        """[(block, gesture id)]: every gesture once per block, in a random order."""
        ids = list(self.specs())
        return [(b, ids[i]) for b in range(self.repetitions) for i in rng.permutation(len(ids))]

    def duration(self):
        return len(self.specs()) * self.repetitions * (self.prepare + self.go + self.rest)


# ---------------------------------------------------------------- cards

HAND_SVG = Path(__file__).with_name("assets") / "hand.svg"  # an open right hand, palm side
# in its 512 x 512 view box: each fingertip, and the direction its finger points
_SVG_TIPS = {
    "thumb": (42, 238),
    "index": (192, 24),
    "middle": (287, 17),
    "ring": (380, 50),
    "pinky": (452, 100),
}
_SVG_DIRECTIONS = {
    "thumb": (-1.0, -0.17),
    "index": (-0.06, -1.0),
    "middle": (-0.06, -1.0),
    "ring": (-0.08, -1.0),
    "pinky": (0.09, -1.0),
}
_svg: dict = {}  # renderers by colour


def _hand(color):
    from pyqtgraph.Qt import QtCore, QtSvg

    if color not in _svg:
        data = HAND_SVG.read_bytes().replace(b'fill="#000000"', f'fill="{color}"'.encode())
        _svg[color] = QtSvg.QSvgRenderer(QtCore.QByteArray(data))
    return _svg[color]


def draw_card(qp, rect, spec, lit, accent="#ffffff", dim="#b7b7b7", marks="#ffffff"):
    """The card of gesture ``spec`` in ``rect`` (QRectF, px): the hand, in ``dim`` while
    preparing and ``accent`` when ``lit``, its moving fingertips as discs in ``marks`` and
    their motion (pinch: a dashed line from the thumb to each finger, arrows closing it;
    extend: an arrow along each finger)."""
    from pyqtgraph.Qt import QtCore, QtGui

    s = 0.74 * min(rect.width(), rect.height())
    box = QtCore.QRectF(rect.center().x() - s / 2, rect.center().y() - 0.44 * s, s, s)
    _hand(accent if lit else dim).render(qp, box)

    def P(p):  # a point of the 512 x 512 hand drawing, in the box
        return QtCore.QPointF(box.left() + p[0] / 512 * s, box.top() + p[1] / 512 * s)

    fingers = spec.get("fingers", [])
    if spec.get("motion") == "rest" or not fingers:
        return
    color = QtGui.QColor(marks)
    tips = {n: np.array(_SVG_TIPS[n], dtype=float) for n in fingers}
    pen = QtGui.QPen(color, max(2.0, s / 110))
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    if spec["motion"] == "pinch":
        thumb = tips["thumb"]
        for n, t in tips.items():
            if n == "thumb":
                continue
            v = (t - thumb) / np.linalg.norm(t - thumb)
            a, b = thumb + v * 0.07 * 512, t - v * 0.07 * 512
            dashed = QtGui.QPen(pen)
            dashed.setStyle(QtCore.Qt.PenStyle.DashLine)
            qp.setPen(dashed)
            qp.drawLine(P(a), P(b))
            qp.setPen(pen)
            for end, d in ((a, v), (b, -v)):  # arrowheads closing the line
                _head(qp, P(end + d * 0.04 * 512), np.arctan2(-d[1], d[0]), s)
    elif spec["motion"] == "extend":  # each finger straightening, along its direction
        qp.setPen(pen)
        for n, t in tips.items():
            v = np.array(_SVG_DIRECTIONS[n]) / np.linalg.norm(_SVG_DIRECTIONS[n])
            _arrow(qp, P(t + v * 0.07 * 512), P(t + v * 0.2 * 512), s)
    qp.setPen(QtCore.Qt.PenStyle.NoPen)
    qp.setBrush(color)
    for t in tips.values():
        qp.drawEllipse(P(t), 0.04 * s, 0.04 * s)


def _arrow(qp, a, b, s):
    qp.drawLine(a, b)
    _head(qp, b, np.arctan2(-(b.y() - a.y()), b.x() - a.x()), s)


def _head(qp, tip, angle, s):
    """An arrowhead at ``tip`` pointing at ``angle`` (radians, counter-clockwise from x)."""
    from pyqtgraph.Qt import QtCore

    L = 0.035 * s
    for side in (-1, 1):
        a = angle + np.pi - side * 0.45
        qp.drawLine(tip, QtCore.QPointF(tip.x() + L * np.cos(a), tip.y() - L * np.sin(a)))


# ---------------------------------------------------------------- movements, offline


def fingertip_speed(t, hand_world):
    """Speed (m/s) of the fastest fingertip at each camera frame: ``t`` (N,) s, ``hand_world``
    (N, 21, 3) m (NaN without a hand), from central differences; NaN at the ends."""
    t = np.asarray(t, dtype=float)
    tips = np.asarray(hand_world, dtype=float)[:, list(TIPS.values())]
    v = np.full(len(t), np.nan)
    if len(t) > 2:
        d = np.linalg.norm(tips[2:] - tips[:-2], axis=2) / (t[2:] - t[:-2])[:, None]
        v[1:-1] = np.max(d, axis=1)
    return v


def movement_windows(go, t, speed, window=(-0.2, 2.5), k=4.0, min_s=0.08, gap_s=0.25):
    """Each trial's movement: onset and offset (s) of the fingertips' speed above threshold
    after its ``go`` (s), within ``window`` of it; the threshold: the median plus ``k`` MADs
    of the speed outside every window. The onset: the first stretch above it lasting
    ``min_s``; the offset: the end of the last stretch following it by less than ``gap_s``.
    ``(onset, offset, status)`` per trial, status "ok", "no_movement" or "no_hand" (no camera
    frame with a hand in the window)."""
    t, speed = np.asarray(t, dtype=float), np.asarray(speed, dtype=float)
    inside = np.zeros(len(t), bool)
    for g in go:
        inside |= (t >= g + window[0]) & (t <= g + window[1])
    quiet = speed[~inside & np.isfinite(speed)]
    if len(quiet) < 10:
        quiet = speed[np.isfinite(speed)]
    med = np.median(quiet) if len(quiet) else 0.0
    thr = med + k * 1.4826 * np.median(np.abs(quiet - med)) if len(quiet) else np.inf
    out = []
    for g in go:
        w = (t >= g + window[0]) & (t <= g + window[1]) & np.isfinite(speed)
        if not w.any():
            out.append((np.nan, np.nan, "no_hand"))
            continue
        tw = t[w]
        idx = np.flatnonzero(speed[w] > thr)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
        first = next((i for i, r in enumerate(runs) if tw[r[-1]] - tw[r[0]] >= min_s), None)
        if first is None:
            out.append((np.nan, np.nan, "no_movement"))
            continue
        end = runs[first][-1]
        for r in runs[first + 1 :]:
            if tw[r[0]] - tw[end] > gap_s:
                break
            end = r[-1]
        out.append((float(tw[runs[first][0]]), float(tw[end]), "ok"))
    return out


def _num(v):
    return np.nan if v in ("n/a", "", None) else float(v)


def main(argv=None):
    """``clvhd-movements EVENTS.tsv``: each go's movement from the recording's camera (its
    motion/ file), to derivatives/movements/.../<label>_desc-movements_events.tsv."""
    p = argparse.ArgumentParser(description=(main.__doc__ or "").splitlines()[0])
    p.add_argument("events", help="the recording's emg/..._events.tsv")
    p.add_argument("--k", type=float, default=4.0, help="threshold: median + k MADs of the speed")
    a = p.parse_args(argv)
    events_path = Path(a.events).resolve()
    label = events_path.name[: -len("_events.tsv")]
    task_part = next(x for x in label.split("_") if x.startswith("task-"))
    motion_name = label.replace(task_part, f"{task_part}_tracksys-{bids.TRACKSYS}") + "_motion.tsv"
    motion_path = events_path.parent.parent / "motion" / motion_name
    if not motion_path.exists():
        raise SystemExit(f"no camera recording for it: {motion_path}")
    root = next(
        (
            p
            for p in events_path.parents
            if (p / "dataset_description.json").exists() and p.name != "derivatives"
        ),
        None,
    )
    if root is None:
        raise SystemExit(f"{events_path} is not in a BIDS dataset (no dataset_description.json)")
    events = bids.read_tsv(events_path)
    channels = [
        r["name"]
        for r in bids.read_tsv(
            motion_path.with_name(motion_name.replace("_motion.tsv", "_channels.tsv"))
        )
    ]
    with motion_path.open(encoding="utf-8") as f:
        motion = np.array([[_num(v) for v in row] for row in csv.reader(f, delimiter="\t")])
    col = {n: i for i, n in enumerate(channels)}
    t = motion[:, col["latency"]]
    world = np.stack(
        [motion[:, [col[f"hand_{p}_{c}_world"] for c in "xyz"]] for p in HAND_LANDMARKS], axis=1
    )
    gos = [e for e in events if e["trial_type"] == "go"]
    windows = movement_windows(
        [float(e["onset"]) for e in gos], t, fingertip_speed(t, world), k=a.k
    )
    out = (
        root
        / "derivatives"
        / "movements"
        / events_path.parent.relative_to(root)
        / f"{label}_desc-movements_events.tsv"
    )
    rows = []
    for e, (on, off, status) in zip(gos, windows, strict=True):
        rows.append(
            [
                f"{on:.4f}" if status == "ok" else e["onset"],
                f"{off - on:.4f}" if status == "ok" else "n/a",
                "movement",
                e.get("gesture", "n/a"),
                e.get("block", "n/a"),
                e.get("trial", "n/a"),
                e["onset"],
                f"{on - float(e['onset']):.4f}" if status == "ok" else "n/a",
                status,
            ]
        )
    bids.write_tsv(
        out,
        [
            "onset",
            "duration",
            "trial_type",
            "gesture",
            "block",
            "trial",
            "go_onset",
            "reaction_time",
            "status",
        ],
        rows,
    )
    bids.write_json(
        out.with_suffix(".json"),
        {
            "onset": {
                "Description": "Movement onset (fingertips' speed above threshold), from the "
                "first EMG sample; the go's onset when none was found",
                "Units": "s",
            },
            "reaction_time": {"Description": "Movement onset after the go", "Units": "s"},
            "status": {
                "Levels": {
                    "ok": "movement found",
                    "no_movement": "none above threshold",
                    "no_hand": "no hand in the camera",
                }
            },
        },
    )
    ok = sum(w[2] == "ok" for w in windows)
    print(f"{ok} of {len(windows)} trials with a movement -> {out}")


if __name__ == "__main__":
    main()
