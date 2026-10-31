"""Export wristband recordings to a BIDS dataset, with the conventions of MUedit2's exporter.

    sub-<s>/[ses-<x>/]emg/      <label>_emg.edf (or .bdf), _emg.json, _channels.tsv/.json, _events.tsv/.json,
                                <sub[_ses]>_electrodes.tsv, <sub[_ses]>_space-<board>_coordsystem.json
    sub-<s>/[ses-<x>/]motion/   <label with tracksys>_motion.tsv, _motion.json, _channels.tsv/.json
    sub-<s>/[ses-<x>/]beh/      <label>_beh.tsv, _beh.json
    sub-<s>/[ses-<x>/]<sub[_ses]>_scans.tsv   acq_time of every file, for synchronisation
    sourcedata/sub-<s>/[ses-<x>/]  the stream as recorded (RawRecording), until it is exported
    dataset_description.json, README, participants.tsv/.json

The EMG is written in EDF (16 bits, MUedit2's default) or BDF (24 bits). EDF spreads
its 65536 steps over each channel's range in the recording: the DC offset itself
costs nothing, but its drift does. A channel spanning 65 mV gets 1 µV steps, whose
0.3 µV rms rounding noise disappears under the ADC's 3.4 µV; one that hits the
ADC's limits (an electrode lifting) gets about 21 µV steps for the whole file.
``_channels.tsv`` gives each channel's step (resolution); BDF's are 256 times finer.
Missing samples (NaN in the stream) are linearly interpolated and counted per
channel in ``_channels.tsv``. The motion files hold one row per processed
camera frame; their LATENCY channel gives its time in s from the first EMG
sample, and both files share that ``acq_time``.

Needs ``pyedflib`` (``pip install -e ".[bids]"``).
"""

import csv
import dataclasses
import datetime as dt
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import __version__, ads1293
from .contact import check_contact
from .hand import FINGERS, HAND_LANDMARKS, finger_flexion, pinch_aperture

BIDS_VERSION = "1.11.1"
CODE_URL = "https://github.com/simonavrillon/CleverHand"
UNITS = "uV"
BOARD = "DRY_ELEC_6CH_THIN"
SPACE = "DRYELEC6CHTHIN"
# DRY_ELEC_6CH_THIN: the six pogo-pin electrodes in one row, 5.08 mm apart, in this order (KiCad)
ELECTRODE_ORDER = [4, 5, 6, 1, 2, 3]
ELECTRODE_PITCH_MM = 5.08
TRACKSYS = "mediapipe"
# where recordings go by default: data/ next to the package (git-ignored)
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Entities:
    """The entities naming a recording: ``sub-<subject>[_ses-<session>]_task-<task>[_acq-][_run-]``."""

    subject: str
    task: str
    session: str | None = None
    acquisition: str | None = None
    run: str | None = None

    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is not None and not (str(value).isascii() and str(value).isalnum()):
                raise ValueError(f"BIDS {f.name} must be letters and digits only, not {value!r}")

    def label(self, tracksys=None):
        pairs = [
            ("sub", self.subject),
            ("ses", self.session),
            ("task", self.task),
            ("tracksys", tracksys),
            ("acq", self.acquisition),
            ("run", self.run),
        ]
        return "_".join(f"{k}-{v}" for k, v in pairs if v)

    @property
    def session_prefix(self):
        return "_".join(f"{k}-{v}" for k, v in (("sub", self.subject), ("ses", self.session)) if v)

    def folder(self, root):
        folder = Path(root) / f"sub-{self.subject}"
        return folder / f"ses-{self.session}" if self.session else folder


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    return str(v)


def write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header is not None:
            f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(_cell(v) for v in row) + "\n")


def read_tsv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_dataset(root):
    """``dataset_description.json`` and ``README`` of the dataset, when missing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    desc = root / "dataset_description.json"
    if not desc.exists():
        write_json(
            desc,
            {
                "Name": root.name,
                "BIDSVersion": BIDS_VERSION,
                "DatasetType": "raw",
                "License": "n/a",
                "Authors": [],
                "GeneratedBy": [{"Name": "clvhd", "Version": __version__, "CodeURL": CODE_URL}],
            },
        )
    readme = root / "README"
    if not readme.exists():
        readme.write_text(
            f"# {root.name}\n\nSurface EMG recorded with the CleverHand wristband "
            "(ADS1293 modules, dry electrodes).\n",
            encoding="utf-8",
        )


def write_participant(root, subject, age=None, sex=None, handedness=None):
    """Add or update the subject's row of ``participants.tsv`` (only the fields given)."""
    root = Path(root)
    parts_json = root / "participants.json"
    if not parts_json.exists():
        write_json(
            parts_json,
            {
                "participant_id": {"Description": "Unique participant identifier."},
                "age": {"Description": "Age of the participant.", "Units": "years"},
                "sex": {
                    "Description": "Biological sex of the participant.",
                    "Levels": {"M": "male", "F": "female", "O": "other"},
                },
                "handedness": {
                    "Description": "Handedness of the participant.",
                    "Levels": {
                        "right": "right-handed",
                        "left": "left-handed",
                        "ambidextrous": "ambidextrous",
                    },
                },
            },
        )
    path = root / "participants.tsv"
    header = ["participant_id", "age", "sex", "handedness"]
    rows = read_tsv(path) if path.exists() else []
    pid = f"sub-{subject}"
    row = next((r for r in rows if r["participant_id"] == pid), None)
    if row is None:
        row = {"participant_id": pid}
        rows.append(row)
    for key, value in (("age", age), ("sex", sex), ("handedness", handedness)):
        if value not in (None, ""):
            row[key] = str(value)
    rows.sort(key=lambda r: r["participant_id"])
    write_tsv(path, header, [[r.get(h) or "n/a" for h in header] for r in rows])


def acq_time(t_monotonic):
    """ISO-8601 wall-clock time (local, with µs) of a ``time.monotonic()`` instant."""
    wall = time.time() - time.monotonic() + t_monotonic
    return dt.datetime.fromtimestamp(wall).isoformat(timespec="microseconds")


def write_scans(root, ent, files, acq):
    """Add or replace the rows of ``files`` (paths under the subject/session folder) in ``_scans.tsv``."""
    folder = ent.folder(root)
    path = folder / f"{ent.session_prefix}_scans.tsv"
    rows = {r["filename"]: r["acq_time"] for r in read_tsv(path)} if path.exists() else {}
    for f in files:
        rows[Path(f).relative_to(folder).as_posix()] = acq
    write_tsv(path, ["filename", "acq_time"], sorted(rows.items()))
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        write_json(
            sidecar,
            {
                "acq_time": {
                    "Description": "Wall-clock time of the first EMG sample of the recording; the motion "
                    "and beh files of a recording share it, their time columns counting from it."
                }
            },
        )


BLOCK_RECORDS = 16  # data records converted at a time (16 x 3 s at the default rate)
FORMATS = {"edf": (-32768, 32767), "bdf": (-8388608, 8388607)}  # digital range
COARSE_STEP_UV = 1.0  # EDF steps above this on a good channel are reported


def _scan(data, size):
    """Per channel, block by block: missing samples, min and max of the others, last valid index."""
    n, c = data.shape
    missing, last = np.zeros(c, dtype=np.int64), np.full(c, -1)
    lo, hi = np.full(c, np.inf), np.full(c, -np.inf)
    for a in range(0, n, size):
        x = np.asarray(data[a : a + size], dtype=np.float64)
        nan = np.isnan(x)
        missing += nan.sum(axis=0)
        lo = np.fmin(lo, np.nanmin(np.where(nan, np.inf, x), axis=0))
        hi = np.fmax(hi, np.nanmax(np.where(nan, -np.inf, x), axis=0))
        has = (~nan).any(axis=0)
        last[has] = a + len(x) - 1 - np.argmax(~nan[::-1], axis=0)[has]
    return missing, lo, hi, last


def _filled_blocks(t, data, size, last_valid):
    """``data`` block by block as float64, missing samples linearly interpolated in time from the
    nearest valid samples before and after, wherever they are (held at the ends; 0 for a channel
    with none). ``last_valid``: per channel, the index of its last valid sample (``_scan``)."""
    n, c = data.shape
    t_prev, v_prev = np.full(c, np.nan), np.full(c, np.nan)
    for a in range(0, n, size):
        x = np.array(data[a : a + size], dtype=np.float64)
        tb = np.asarray(t[a : a + size], dtype=np.float64)
        nan = np.isnan(x)
        for ch in np.flatnonzero(nan.any(axis=0)):
            ok = ~nan[:, ch]
            ts, vs = [tb[ok]], [x[ok, ch]]
            if not np.isnan(t_prev[ch]):
                ts.insert(0, t_prev[ch : ch + 1])
                vs.insert(0, v_prev[ch : ch + 1])
            if not ok[-1] and last_valid[ch] >= a + len(x):  # the next valid sample is further on
                j = a + len(x)
                while True:
                    seg = np.asarray(data[j : j + size, ch])
                    hit = np.flatnonzero(~np.isnan(seg))
                    if len(hit):
                        ts.append(np.asarray(t[j + hit[0] : j + hit[0] + 1], dtype=np.float64))
                        vs.append(seg[hit[:1]].astype(np.float64))
                        break
                    j += size
            ts_c, vs_c = np.concatenate(ts), np.concatenate(vs)
            x[:, ch] = np.interp(tb, ts_c, vs_c) if len(ts_c) else 0.0
        has = (~nan).any(axis=0)
        idx = len(x) - 1 - np.argmax(~nan[::-1], axis=0)
        t_prev[has], v_prev[has] = tb[idx[has]], x[idx[has], np.flatnonzero(has)]
        yield x


def _physical_range(lo, hi):
    """EDF/BDF physical min/max for a channel spanning [lo, hi] µV: whole µV, outward, never equal.

    Whole numbers fit the header's 8 characters exactly (up to 10 V); pyedflib reformats others,
    which can move a limit inside the data and clip it.
    """
    pmin, pmax = math.floor(lo), math.ceil(hi)
    if pmin == pmax:
        pmin, pmax = pmin - 1, pmax + 1
    if len(str(pmin)) > 8 or len(str(pmax)) > 8:
        raise ValueError(f"[{lo}, {hi}] uV does not fit an EDF header field")
    return pmin, pmax


def _record_duration(fs):
    """The shortest data record (s, at most 60) holding a whole number of samples at ``fs``."""
    frac = Fraction(fs).limit_denominator(1000)
    for k in range(1, 61):
        if (frac * k).denominator == 1:
            return k
    raise ValueError(f"No data record of at most 60 s holds a whole number of samples at {fs} Hz")


def channel_electrodes(cfg, nb_modules):
    """(signal electrode, reference electrode) of each channel, named M<module>E<input>."""
    return [
        (f"M{m}E{cfg.route[ch][0]}", f"M{m}E{cfg.route[ch][1]}")
        for m in range(nb_modules)
        for ch in range(3)
    ]


def write_emg(
    root,
    ent,
    t_host,
    data,
    cfg,
    labels,
    *,
    start=None,
    powerline_freq=None,
    muscles=None,
    placement_description=None,
    contact=None,
    task_description=None,
    firmware=None,
    stats=None,
    file_format="edf",
):
    """Write the EMG of one recording (``data``: (N, channels) in V, NaN where missing, at host
    times ``t_host``) as ``file_format`` "edf" or "bdf". ``start``: the ``acq_time`` of the first
    sample (default: from ``t_host[0]``, valid only on the computer's current boot).

    ``contact``: ``clvhd.check_contact`` status per channel; flagged channels are marked bad.
    Returns the paths written, and in ``coarse`` the good channels whose step (µV) exceeds
    ``COARSE_STEP_UV``.
    """
    import pyedflib

    if file_format not in FORMATS:
        raise ValueError(f"file_format must be one of {sorted(FORMATS)}, not {file_format!r}")
    nb_modules = len(labels) // 3
    fs = float(cfg.odr())
    emg_dir = ent.folder(root) / "emg"
    emg_dir.mkdir(parents=True, exist_ok=True)
    label = ent.label()
    n = len(data)
    spr_s = _record_duration(fs)
    spr = round(fs * spr_s)
    block = spr * BLOCK_RECORDS
    n_missing, lo, hi, last_valid = _scan(data, block)
    lo, hi = np.where(np.isfinite(lo), lo, 0) * 1e6, np.where(np.isfinite(hi), hi, 0) * 1e6
    dmin, dmax = FORMATS[file_format]
    signal = emg_dir / f"{label}_emg.{file_format}"
    for other in FORMATS:  # one signal file per recording
        if other != file_format:
            signal.with_suffix(f".{other}").unlink(missing_ok=True)
    tmp = signal.with_name(f".{signal.stem}.partial.{file_format}")
    file_type = pyedflib.FILETYPE_EDFPLUS if file_format == "edf" else pyedflib.FILETYPE_BDFPLUS
    writer = pyedflib.EdfWriter(str(tmp), len(labels), file_type=file_type)
    pmin, pmax = np.zeros(len(labels)), np.zeros(len(labels))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # forcing the record duration
            writer.setDatarecordDuration(spr_s)
        start = start or acq_time(float(t_host[0]))
        writer.setStartdatetime(dt.datetime.fromisoformat(start).replace(microsecond=0))
        headers = []
        for c, name in enumerate(labels):
            pmin[c], pmax[c] = _physical_range(float(lo[c]), float(hi[c]))
            headers.append(
                {
                    "label": name,
                    "dimension": UNITS,
                    "sample_frequency": fs,
                    "physical_min": int(pmin[c]),
                    "physical_max": int(pmax[c]),
                    "digital_min": dmin,
                    "digital_max": dmax,
                    "transducer": "",
                    "prefilter": "",
                }
            )
        writer.setSignalHeaders(headers)
        step = (pmax - pmin) / (dmax - dmin)  # µV per digital unit
        for chunk in _filled_blocks(t_host, data, block, last_valid):
            uv = chunk * 1e6
            if len(uv) % spr:  # the last data record: padded with the last value (no step)
                uv = np.vstack([uv, np.repeat(uv[-1:], -len(uv) % spr, axis=0)])
            # rounded to the nearest step here: pyedflib's physical writes truncate
            dig = np.clip(np.rint((uv - pmin) / step + dmin), dmin, dmax).astype(np.int32)
            for r in range(0, len(dig), spr):  # channel after channel within a record
                if (
                    writer.blockWriteDigitalSamples(
                        np.ascontiguousarray(dig[r : r + spr].T).ravel()
                    )
                    < 0
                ):
                    raise OSError("pyedflib could not write a data record")
    finally:
        writer.close()
    tmp.replace(signal)

    # channels
    muscles = muscles or {}
    pairs = channel_electrodes(cfg, nb_modules)
    contact = contact or [""] * len(labels)
    good = [cfg.enable[c % 3] and not contact[c] for c in range(len(labels))]
    rows = []
    for c, name in enumerate(labels):
        m, ch = divmod(c, 3)
        enabled = cfg.enable[ch]
        status = "good" if good[c] else "bad"
        status_desc = (
            "channel disabled"
            if not enabled
            else contact[c]
            or (f"{n_missing[c]} missing samples interpolated" if n_missing[c] else "n/a")
        )
        pos, neg = cfg.route[ch]
        ied = (
            ELECTRODE_PITCH_MM * abs(ELECTRODE_ORDER.index(pos) - ELECTRODE_ORDER.index(neg))
            if pos in ELECTRODE_ORDER and neg in ELECTRODE_ORDER
            else None
        )
        rows.append(
            [
                name,
                "EMG",
                UNITS,
                f"Bipolar channel {ch + 1} of module {m}",
                f"{fs:.10g}",
                pairs[c][0],
                pairs[c][1],
                f"M{m}",
                muscles.get(m, "n/a"),
                BOARD,
                ied,
                0,
                "n/a",
                "n/a",
                status,
                status_desc,
                int(n_missing[c]),
                f"{step[c]:.4g}",
            ]
        )
    channels = emg_dir / f"{label}_channels.tsv"
    write_tsv(
        channels,
        [
            "name",
            "type",
            "units",
            "description",
            "sampling_frequency",
            "signal_electrode",
            "reference",
            "group",
            "target_muscle",
            "grid_name",
            "interelectrode_distance",
            "low_cutoff",
            "high_cutoff",
            "notch",
            "status",
            "status_description",
            "n_interpolated",
            "resolution",
        ],
        rows,
    )
    write_json(
        channels.with_suffix(".json"),
        {
            "reference": {
                "Description": "The channel's negative input: every channel is bipolar, "
                "between two neighbouring electrodes of the same module."
            },
            "group": {
                "Description": "Module (M0, M1, ...) the channel belongs to, in chain order."
            },
            "target_muscle": {"Description": "Muscle under the module."},
            "grid_name": {"Description": "Electrode board of the module."},
            "interelectrode_distance": {
                "Description": "Distance between the channel's two electrodes.",
                "Units": "mm",
            },
            "low_cutoff": {"Description": "The ADS1293 is DC-coupled.", "Units": "Hz"},
            "n_interpolated": {
                "Description": "Samples missing from the stream (dropped or late "
                "frames), filled by linear interpolation.",
                "Units": "samples",
            },
            "resolution": {
                "Description": f"Step of the {file_format.upper()} encoding: the channel's "
                "range in the recording over the format's digital range "
                "(EDF 65536 steps, BDF 16777216).",
                "Units": UNITS,
            },
        },
    )

    # electrodes: module-local coordinates along each board's row of pads
    electrodes = emg_dir / f"{ent.session_prefix}_electrodes.tsv"
    write_tsv(
        electrodes,
        ["name", "x", "y", "z", "coordinate_system", "type", "material", "impedance", "group"],
        [
            [
                f"M{m}E{e}",
                ELECTRODE_PITCH_MM * k,
                0.0,
                "n/a",
                SPACE,
                "dry",
                "gold-plated",
                "n/a",
                f"M{m}",
            ]
            for m in range(nb_modules)
            for k, e in enumerate(ELECTRODE_ORDER)
        ],
    )
    coordsys = emg_dir / f"{ent.session_prefix}_space-{SPACE}_coordsystem.json"
    write_json(
        coordsys,
        {
            "EMGCoordinateSystem": "Other",
            "EMGCoordinateUnits": "mm",
            "EMGCoordinateSystemDescription": f"Local coordinates on each module's {BOARD} board: x along its row of six electrodes "
            f"({' '.join(f'E{e}' for e in ELECTRODE_ORDER)}, {ELECTRODE_PITCH_MM} mm apart) from "
            f"E{ELECTRODE_ORDER[0]}, y = 0. The modules (group) are chained "
            "around the forearm; each has its own copy of this frame.",
        },
    )

    # sidecar
    rld = (
        (
            f"Driven right-leg electrode M{cfg.rld_module}E{cfg.rld_input} (module "
            f"{cfg.rld_module}, input IN{cfg.rld_input})"
        )
        if cfg.rld_module is not None
        else None
    )
    bandwidth = (
        {
            "LowPassFilter": {
                "CutOffHz": 420,
                "Description": "ADS1293 sinc decimation filters (R1=4, R2=4, R3=6 at 200 kHz)",
            }
        }
        if (cfg.R1[0], cfg.R2, cfg.R3[0], cfg.clock_hz, cfg.high_freq[0]) == (4, 4, 6, 4e6, True)
        else {
            "Description": f"ADS1293 sinc decimation filters, R1={cfg.R1}, R2={cfg.R2}, "
            f"R3={cfg.R3} (see the ADS1293 datasheet for the bandwidth)"
        }
    )
    emg_json = {
        "TaskName": ent.task,
        "EMGReference": "Bipolar: each channel between two neighbouring electrodes of its module "
        "(see reference in _channels.tsv)",
        "SamplingFrequency": fs,
        "PowerLineFrequency": powerline_freq or "n/a",
        "RecordingType": "continuous",
        "EMGPlacementScheme": "Other",
        "EMGPlacementSchemeDescription": placement_description
        or f"Wristband of {nb_modules} modules chained around the forearm, each a {BOARD} board of "
        "six dry electrodes in a row across the band.",
        "HardwareFilters": {"DC-coupled": True, **bandwidth},
        "SoftwareFilters": {
            "MissingSampleInterpolation": {
                "Description": "Samples missing from the stream filled by linear interpolation "
                "(n_interpolated in _channels.tsv)",
                "Samples": int(n_missing.sum()),
            }
        },
        "Manufacturer": "CleverHand",
        "ManufacturersModelName": f"{nb_modules}-module EMG wristband (COM_MOD + EMG_DAQ_ADS1293 + "
        f"{BOARD}), Teensy 4.0 controller",
        "SoftwareVersions": f"clvhd {__version__}" + (f", firmware {firmware}" if firmware else ""),
        "ElectrodeType": "dry",
        "ElectrodeMaterial": "gold-plated",
        "InterelectrodeDistance": ELECTRODE_PITCH_MM,
        "EMGChannelCount": len(labels),
        "RecordingDuration": n / fs,
        "ADS1293Config": dataclasses.asdict(cfg),
    }
    if rld:
        emg_json["EMGGround"] = rld
    if task_description:
        emg_json["TaskDescription"] = task_description
    if muscles:
        emg_json["Muscles"] = sorted(set(muscles.values()))
    if stats:
        emg_json["StreamStatistics"] = stats
    sidecar = emg_dir / f"{label}_emg.json"
    write_json(sidecar, emg_json)
    coarse = {
        labels[c]: float(step[c])
        for c in range(len(labels))
        if good[c] and step[c] > COARSE_STEP_UV
    }
    return {
        "emg": signal,
        "emg_json": sidecar,
        "channels": channels,
        "electrodes": electrodes,
        "coordsystem": coordsys,
        "coarse": coarse,
    }


def _motion_channels():
    """(name, component, type, tracked_point, units, reference_frame, description) of each column."""
    rows = [
        (
            "latency",
            "n/a",
            "LATENCY",
            "n/a",
            "s",
            "n/a",
            "Time of the camera frame from the first EMG sample",
        )
    ]
    for frame, units, desc in (
        ("world", "m", "MediaPipe hand world landmarks (origin at the hand's centre)"),
        (
            "image",
            "n/a",
            "MediaPipe hand image landmarks (x, y "
            "fractions of the mirrored image; z depth relative to the wrist)",
        ),
    ):
        for point in HAND_LANDMARKS:
            rows.extend(
                (f"hand_{point}_{comp}_{frame}", comp, "POS", point, units, f"hand_{frame}", desc)
                for comp in "xyz"
            )
    rows += [
        (
            f"{finger}_flexion",
            "n/a",
            "JNTANG",
            finger,
            "deg",
            "n/a",
            f"Bend of the {finger}: the sum of its joint angles (0 straight; "
            "clvhd.hand.finger_flexion)",
        )
        for finger in FINGERS
    ]
    rows += [
        (
            "pinch_aperture",
            "n/a",
            "MISC",
            "thumb_tip",
            "cm",
            "n/a",
            "Distance between the thumb and index tips",
        ),
        (
            "hand_score",
            "n/a",
            "MISC",
            "n/a",
            "n/a",
            "n/a",
            "Handedness score of the detected hand (0-1); n/a when no hand was detected",
        ),
        ("hand_right", "n/a", "MISC", "n/a", "n/a", "n/a", "1: right hand, 0: left hand"),
    ]
    return rows


def write_motion(root, ent, hand, t0, camera_fps, task_description=None):
    """Write the hand frames (``clvhd.hand.FRAME_DTYPE`` records, e.g. ``clvhd.hand.stack``
    output or a ``RawRecording``'s) of one recording as BIDS motion, block by block.

    ``t0``: ``time.monotonic()`` of the first EMG sample; frames before it are left out.
    """
    motion_dir = ent.folder(root) / "motion"
    label = ent.label(TRACKSYS)
    channels = _motion_channels()
    motion = motion_dir / f"{label}_motion.tsv"
    motion_dir.mkdir(parents=True, exist_ok=True)
    n, first, latest = 0, None, None
    with motion.open("w", encoding="utf-8") as f:
        for a in range(0, len(hand), 5000):
            sub = hand[a : a + 5000]
            sub = sub[sub["t"] >= t0]
            if not len(sub):
                continue
            k = len(sub)
            table = np.column_stack(
                [sub["t"] - t0]
                + [
                    sub[key].reshape(k, -1).astype(np.float64)
                    for key in ("hand_world", "hand_image")
                ]
                + [finger_flexion(sub["hand_world"]), pinch_aperture(sub["hand_world"])[:, None]]
                + [sub["score"].astype(np.float64), sub["right"].astype(np.float64)]
            )
            assert table.shape[1] == len(channels)
            for row in table.tolist():
                f.write("\t".join([f"{row[0]:.6f}"] + [_cell(v) for v in row[1:]]) + "\n")
            first = table[0, 0] if first is None else first
            latest = table[-1, 0]
            n += k
    channels_tsv = motion_dir / f"{label}_channels.tsv"
    write_tsv(
        channels_tsv,
        [
            "name",
            "component",
            "type",
            "tracked_point",
            "units",
            "reference_frame",
            "description",
            "status",
        ],
        [[*row, "good"] for row in channels],
    )
    write_json(
        channels_tsv.with_suffix(".json"),
        {
            "reference_frame": {
                "Description": "Frame of the position channels; x right, y down, z away from the camera.",
                "Levels": {
                    "hand_world": {
                        "Description": "MediaPipe hand world frame, origin at the hand's centre",
                        "SpatialAxes": "RDB",
                        "RotationRule": "right-hand",
                        "RotationOrder": "n/a",
                    },
                    "hand_image": {
                        "Description": "Mirrored camera image: x, y fractions of its width "
                        "and height from the top left; z depth from the wrist"
                    },
                },
            }
        },
    )
    span = (latest - first) if n > 1 and first is not None and latest is not None else 0.0
    counts: dict[str, int] = {}
    for row in channels:
        counts[row[2]] = counts.get(row[2], 0) + 1
    sidecar = motion_dir / f"{label}_motion.json"
    write_json(
        sidecar,
        {
            "TaskName": ent.task,
            "SamplingFrequency": float(camera_fps),
            "SamplingFrequencyEffective": (n - 1) / span if span > 0 else "n/a",
            "TrackingSystemName": "MediaPipe from a single RGB camera",
            "ManufacturersModelName": "MediaPipe hand landmarker",
            "SoftwareVersions": f"clvhd {__version__}",
            "RecordingDuration": float(span),
            "MotionChannelCount": len(channels),
            "POSChannelCount": counts["POS"],
            "JNTANGChannelCount": counts["JNTANG"],
            "LATENCYChannelCount": counts["LATENCY"],
            "MiscChannelCount": counts["MISC"],
            "TrackedPointsCount": len(HAND_LANDMARKS),
            "MissingValues": "n/a",
            **({"TaskDescription": task_description} if task_description else {}),
            "SubjectArtefactDescription": "Frames are irregular (frames skipped while the models "
            "run): use the latency channel. Times are when the frame "
            "reached the host, about one frame period after exposure.",
        },
    )
    return {"motion": motion, "motion_json": sidecar, "channels": channels_tsv, "frames": n}


def event_row(t, t0, fs, **columns):
    """An events.tsv row for host time ``t``: its onset from ``t0`` (the first EMG sample) and
    the EMG sample there, at ``fs`` Hz."""
    onset = t - t0
    return {"onset": round(onset, 6), "sample": int(round(onset * fs)), **columns}


def write_events(path, rows, columns, descriptions):
    """Write ``rows`` (dicts) as events sorted by onset; ``descriptions``: the sidecar's columns."""
    rows = sorted(rows, key=lambda r: r["onset"])
    write_tsv(path, columns, [[r.get(c) for c in columns] for r in rows])
    write_json(path.with_suffix(".json"), descriptions)
    return path


def write_beh(root, ent, columns, table, **sidecar):
    """The 60 Hz task of a recording as ``beh/<label>_beh.tsv``: ``columns`` {name: (description,
    units or None)}, ``table`` its rows; ``sidecar``: more of the .json. Returns the .tsv path."""
    path = ent.folder(root) / "beh" / f"{ent.label()}_beh.tsv"
    write_tsv(path, list(columns), table)
    write_json(
        path.with_suffix(".json"),
        {
            "TaskName": ent.task,
            **{
                k: {"Description": d, **({"Units": u} if u else {})}
                for k, (d, u) in columns.items()
            },
            **sidecar,
        },
    )
    return path


def write_derivative_description(root, pipeline, generated_by, ignore=("*.npz",)):
    """``derivatives/<pipeline>/``'s ``dataset_description.json`` and ``.bidsignore`` (of the
    files ``ignore``); returns that folder."""
    folder = Path(root) / "derivatives" / pipeline
    write_json(
        folder / "dataset_description.json",
        {
            "Name": pipeline,
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": generated_by, "Version": __version__, "CodeURL": CODE_URL}],
        },
    )
    (folder / ".bidsignore").write_text("".join(f"{g}\n" for g in ignore))
    return folder


def next_run(root, ent):
    """The first run number (two digits) after every run of ``ent`` (``ent.run`` ignored) in the
    EMG files or in ``sourcedata/``, which also holds the recordings not exported yet."""
    ent = dataclasses.replace(ent, run=None)
    prefix = f"{ent.label()}_run-"
    used = set()
    for folder, pattern in (
        (ent.folder(root) / "emg", f"{prefix}*_emg.*"),
        (ent.folder(Path(root) / "sourcedata"), f"{prefix}*_clvhd.json"),
    ):
        for p in folder.glob(pattern):
            run = p.name[len(prefix) :].split("_")[0]
            if run.isdigit():
                used.add(int(run))
    return f"{max(used, default=0) + 1:02d}"


class RawRecording:
    """A recording streamed to ``sourcedata/`` as it arrives, so a long one never fills the RAM and
    a crash loses at most what came after the last ``flush()``.

        <label>_clvhd.json      what the files hold: entities, configuration, export settings
        <label>_clvhd-emg.dat   one record per sample: t_host, t_device (float64, s), data
                                (float32 per channel, V, NaN where missing)
        <label>_clvhd-hand.dat  one ``clvhd.hand.FRAME_DTYPE`` record per processed camera frame
        <label>_clvhd-events.tsv  one row per event (``append_event``): t_host (s), duration,
                                trial_type, then ``events`` columns

    ``export_recording`` turns them into the BIDS files; ``RawRecording.open`` maps them back
    as numpy record arrays (whole records only).
    """

    def __init__(
        self,
        root,
        ent,
        cfg,
        labels,
        *,
        hand=False,
        camera_fps=None,
        firmware=None,
        export=None,
        events=None,
    ):
        folder = ent.folder(Path(root) / "sourcedata")
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{ent.label()}_clvhd"
        self.header_path = folder / f"{stem}.json"
        self.emg_dtype = emg_dtype(len(labels))
        self.header = {
            "format": "clvhd raw recording 1",
            "entities": dataclasses.asdict(ent),
            "fs": cfg.odr(),
            "labels": list(labels),
            "config": dataclasses.asdict(cfg),
            "firmware": firmware,
            "camera_fps": camera_fps,
            "emg_file": f"{stem}-emg.dat",
            "hand_file": f"{stem}-hand.dat" if hand else None,
            "events_file": f"{stem}-events.tsv" if events is not None else None,
            "events_columns": list(events or []),
            "export": export or {},
            "acq_time": None,
            "complete": False,
            "stats": None,
        }
        self._write_header()
        self._emg = self._hand = self._events = None
        try:
            self._emg = (folder / self.header["emg_file"]).open("wb", buffering=1 << 20)
            if hand:
                self._hand = (folder / self.header["hand_file"]).open("wb", buffering=1 << 20)
            if events is not None:
                self._events = (folder / self.header["events_file"]).open("w", encoding="utf-8")
                self._events.write("\t".join(["t_host", "duration", "trial_type", *events]) + "\n")
        except Exception:
            for f in (self._emg, self._hand, self._events):
                if f:
                    f.close()
            raise
        self.samples = self.frames = 0
        self.t0 = None  # host time of the first sample

    def _write_header(self):
        write_json(self.header_path, self.header)

    def append(self, t_host, t_device, data):
        """Samples as ``Stream.read()`` returns them."""
        if not len(t_host):
            return
        rec = np.empty(len(t_host), self.emg_dtype)
        rec["t_host"], rec["t_device"], rec["data"] = t_host, t_device, data
        assert self._emg is not None  # opened in __init__
        self._emg.write(rec.tobytes())
        if not self.samples:
            self.t0 = float(t_host[0])
            self.header["acq_time"] = acq_time(self.t0)
            self._write_header()
        self.samples += len(rec)

    def append_frames(self, records):
        """Hand frames as ``clvhd.hand.FRAME_DTYPE`` records (``clvhd.hand.stack``)."""
        assert self._hand is not None  # opened in __init__ when hand=True
        self._hand.write(records.tobytes())
        self.frames += len(records)

    def append_event(self, t_host, trial_type, duration=0.0, **values):
        """An event at host time ``t_host`` (``time.monotonic()``, the samples' clock), with
        values of the ``events`` columns (n/a for those left out, and a ``duration`` None)."""
        cols = self.header["events_columns"]
        unknown = set(values) - set(cols)
        if unknown:
            raise ValueError(f"no event columns {sorted(unknown)}: {cols}")
        assert self._events is not None  # opened in __init__ when events is given
        self._events.write(
            "\t".join(
                [f"{t_host:.6f}", _cell(duration), trial_type]
                + [_cell(values.get(c)) for c in cols]
            )
            + "\n"
        )

    def flush(self):
        """Push what was appended to the disk."""
        for f in (self._emg, self._hand, self._events):
            if f:
                f.flush()
                os.fsync(f.fileno())

    def close(self, stats=None):
        self.flush()
        for f in (self._emg, self._hand, self._events):
            if f:
                f.close()
        self.header["complete"], self.header["stats"] = True, stats
        self._write_header()

    @staticmethod
    def open(header_path):
        """(header, entities, EMG records, hand records or None) of a streamed recording."""
        from .hand import FRAME_DTYPE

        header_path = Path(header_path)
        header = json.loads(header_path.read_text())
        emg = _records(header_path.with_name(header["emg_file"]), emg_dtype(len(header["labels"])))
        hand = (
            _records(header_path.with_name(header["hand_file"]), FRAME_DTYPE)
            if header["hand_file"]
            else None
        )
        return header, Entities(**header["entities"]), emg, hand


def emg_dtype(n_channels):
    return np.dtype([("t_host", "<f8"), ("t_device", "<f8"), ("data", "<f4", (n_channels,))])


def _records(path, dtype):
    """The whole records of a file (a partial last one, from a crash, left out), memory-mapped."""
    n = path.stat().st_size // dtype.itemsize if path.exists() else 0
    return np.memmap(path, dtype, "r", shape=(n,)) if n else np.zeros(0, dtype)


def export_recording(source, keep_source=False, file_format=None):
    """Write a recording streamed to ``sourcedata/`` (a ``RawRecording`` or its header .json) to
    the BIDS dataset holding it: dataset and participant files, the EMG, the hand frames as
    motion, and ``_scans.tsv``, block by block. Channels the contact check flags on the first
    second are marked bad. Then, unless ``keep_source``, deletes the recording from
    ``sourcedata/`` once the files written read back complete (``_check_export``). Returns the
    paths written (``sourcedata``: None once deleted), the contact status and the duration.
    ``file_format``: "edf" or "bdf" over the one chosen when recording (default EDF)."""
    path = Path(source.header_path if isinstance(source, RawRecording) else source).resolve()
    root = next(p for p in path.parents if p.name == "sourcedata").parent
    header, ent, emg, hand = RawRecording.open(path)
    if not len(emg):
        raise ValueError("no EMG sample was recorded")
    cfg = ads1293.Config(**header["config"])
    exp = header["export"]
    write_dataset(root)
    write_participant(root, ent.subject, **(exp.get("participant") or {}))
    contact = check_contact(np.asarray(emg["data"][: int(cfg.odr())], dtype=np.float64))
    # the wall-clock time of the first sample, taken while recording: time.monotonic() restarts
    # with the computer, so it cannot be converted after a reboot
    start = header.get("acq_time") or acq_time(float(emg["t_host"][0]))
    written = write_emg(
        root,
        ent,
        emg["t_host"],
        emg["data"],
        cfg,
        header["labels"],
        start=start,
        contact=contact,
        firmware=header["firmware"],
        stats=header["stats"],
        powerline_freq=exp.get("powerline_freq"),
        muscles={int(m): name for m, name in (exp.get("muscles") or {}).items()},
        placement_description=exp.get("placement_description"),
        task_description=exp.get("task_description"),
        file_format=file_format or exp.get("file_format", "edf"),
    )
    files = [written["emg"]]
    t0 = float(emg["t_host"][0])
    n_frames = None
    if hand is not None:
        motion = write_motion(
            root, ent, hand, t0, header["camera_fps"], task_description=exp.get("task_description")
        )
        written["motion"] = motion["motion"]
        files.append(motion["motion"])
        n_frames = motion["frames"]
    if header["events_file"] and path.with_name(header["events_file"]).exists():
        rows = read_tsv(path.with_name(header["events_file"]))
        cols = header["events_columns"]
        fs = cfg.odr()
        events = [
            event_row(
                float(r["t_host"]),
                t0,
                fs,
                duration=r["duration"],
                trial_type=r["trial_type"],
                **{c: r[c] for c in cols},
            )
            for r in rows
        ]
        written["events"] = write_events(
            written["emg"].with_name(f"{ent.label()}_events.tsv"),
            events,
            ["onset", "duration", "sample", "trial_type", *cols],
            {
                "onset": {
                    "Description": "Time from the first EMG sample of the file.",
                    "Units": "s",
                },
                "duration": {"Units": "s"},
                "sample": {"Description": "EMG sample at the onset (zero-based)."},
                **(exp.get("event_descriptions") or {}),
            },
        )
    write_scans(root, ent, files, start)
    written.update(sourcedata=path, contact=contact, duration=len(emg) / cfg.odr(), acq_time=start)
    if not keep_source:
        _check_export(written, len(emg), header["labels"], n_frames)
        written["sourcedata"] = None
        del emg, hand  # the memory maps, before their files go
        _delete_source(path, header, root)
    return written


def _check_export(written, n_samples, labels, n_frames):
    """Raise unless the EMG file opens with every channel and sample, and the motion file has
    every frame."""
    import pyedflib

    with pyedflib.EdfReader(str(written["emg"])) as r:
        if r.getSignalLabels() != list(labels) or min(r.getNSamples()) < n_samples:
            raise OSError(f"{written['emg'].name} does not read back complete: sourcedata kept")
    if n_frames is not None:
        with written["motion"].open(encoding="utf-8") as f:
            if sum(1 for _ in f) != n_frames:
                raise OSError(
                    f"{written['motion'].name} does not read back complete: sourcedata kept"
                )


def _delete_source(path, header, root):
    """Delete a streamed recording's files, then the folders it leaves empty up to sourcedata/."""
    for name in (header["emg_file"], header["hand_file"], header.get("events_file")):
        if name:
            path.with_name(name).unlink(missing_ok=True)
    path.unlink()
    folder = path.parent
    while folder != root and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def main():
    """``clvhd-export``: write recordings streamed to sourcedata/ (e.g. cut short by a crash) to BIDS."""
    import argparse

    p = argparse.ArgumentParser(description=(main.__doc__ or "").splitlines()[0])
    p.add_argument("headers", nargs="+", type=Path, help="<label>_clvhd.json files in sourcedata/")
    p.add_argument("--keep", action="store_true", help="keep them in sourcedata/ once exported")
    p.add_argument("--format", choices=sorted(FORMATS), help="over the one chosen when recording")
    args = p.parse_args()
    for path in args.headers:
        out = export_recording(path, keep_source=args.keep, file_format=args.format)
        print(
            f"{path.name}: {out['duration']:.1f} s -> {out['emg']}"
            + (f" and {out['motion'].name}" if "motion" in out else "")
        )
