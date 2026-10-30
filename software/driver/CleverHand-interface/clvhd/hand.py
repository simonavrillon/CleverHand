"""Hand landmarks from a camera with MediaPipe, on background threads.

One thread reads the camera as fast as it delivers frames and stamps each one
with ``time.monotonic()``, the clock of the EMG stream's ``t_host``, so both
streams line up by time. It keeps only the latest frame. A second thread runs
the MediaPipe hand landmarker on the latest frame: when the model is slower than the camera it skips frames
instead of falling behind, and every result keeps the time of its frame.
OpenCV and MediaPipe run in C++ without the GIL, so these threads do not slow
the EMG reader thread.

Needs ``mediapipe`` (``pip install -e ".[hand]"``). The model is downloaded
to ``~/.cache/clvhd`` on first use.
"""

import collections
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")
MODEL_DIR = Path.home() / ".cache" / "clvhd"

HAND_LANDMARKS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
FINGERS = {  # landmark chains, from the wrist (or the thumb's CMC) to the tip
    "thumb": (1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16), "pinky": (0, 17, 18, 19, 20)}
TIPS = {name: chain[-1] for name, chain in FINGERS.items()}  # each fingertip's landmark
KEEP_FRAMES = 3000  # results kept until read()


@dataclass
class HandFrame:
    """Landmarks of one camera frame; NaN where nothing was detected.

    ``hand_image``: (21, 3) x, y in [0, 1] of the (mirrored) image width and
    height, z the depth relative to the wrist, in about the same scale as x.
    ``hand_world``: (21, 3) in m, origin at the hand's centre, with the camera's
    axes (x right, y down, z away from the camera).
    """
    t: float
    index: int
    handedness: str
    score: float
    hand_image: np.ndarray
    hand_world: np.ndarray

    @classmethod
    def empty(cls, t, index):
        return cls(t, index, "", np.nan, np.full((21, 3), np.nan), np.full((21, 3), np.nan))


def finger_flexion(hand_world):
    """Each finger's bend (degrees, ``FINGERS`` order): the sum of its joints' angles between
    consecutive segments (MCP, PIP and DIP; the thumb's MCP and IP): near 0 with the hand
    flat (the metacarpals fan out a little from the wrist), about 250 in a fist.
    ``hand_world`` (..., 21, 3); NaN where no hand. Unsigned: a hyperextended joint also
    counts as bent. From a single camera, depth is the least accurate axis."""
    hand_world = np.asarray(hand_world, dtype=float)
    out = []
    for chain in FINGERS.values():
        seg = np.diff(hand_world[..., chain, :], axis=-2)
        seg = seg / np.linalg.norm(seg, axis=-1, keepdims=True)
        cos = np.clip((seg[..., :-1, :] * seg[..., 1:, :]).sum(axis=-1), -1, 1)
        out.append(np.degrees(np.arccos(cos)).sum(axis=-1))
    return np.stack(out, axis=-1)


def pinch_aperture(hand_world):
    """Distance between the thumb and index tips (cm), from ``hand_world`` (..., 21, 3) in m."""
    hand_world = np.asarray(hand_world, dtype=float)
    return 100 * np.linalg.norm(hand_world[..., TIPS["thumb"], :] - hand_world[..., TIPS["index"], :],
                                axis=-1)


def model_path():
    """Local path of the MediaPipe hand landmarker model, downloaded on first use."""
    path = MODEL_DIR / Path(MODEL_URL).name
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"Downloading the MediaPipe hand model to {path}")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class HandTracker:
    """Camera frames and their MediaPipe hand landmarks, on two background threads.

    >>> with HandTracker(camera=0) as tracker:
    ...     frames = tracker.read()          # HandFrames since the last read
    ...     image, frame = tracker.latest()  # for display: mirrored BGR image and its landmarks

    ``camera``: OpenCV camera index, or a video file (read at its frame rate). ``mirror``: flip the
    image horizontally, as MediaPipe expects for its Left/Right labels to name
    the real hand (and as a selfie view looks natural).
    """

    def __init__(self, camera=0, mirror=True, width=None, height=None, fps=None,
                 min_confidence=0.5):
        self.camera, self.mirror = camera, mirror
        self.size, self.fps_request = (width, height), fps
        self.min_confidence = min_confidence
        self.running = False

    def start(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._cv2, self._mp = cv2, mp
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera!r} (on macOS, allow camera access "
                               "for the terminal in System Settings > Privacy & Security > Camera)")
        if self.size[0]:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        if self.size[1]:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if self.fps_request:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_request)
        self.is_file = isinstance(self.camera, str)
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        conf = self.min_confidence
        self.hand_model = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path())),
            running_mode=vision.RunningMode.VIDEO, num_hands=1, min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf, min_tracking_confidence=conf))

        self._new = threading.Condition()
        self._frame = None  # (index, t, BGR image) of the latest camera frame
        self._lock = threading.Lock()
        self._results: collections.deque = collections.deque(maxlen=KEEP_FRAMES)
        self._latest = (None, None)
        self._stop = threading.Event()
        self._error = None
        self.captured = self.processed = 0
        self._last_ms = -1
        self._threads = [threading.Thread(target=self._guard, args=(fn,), name=name, daemon=True)
                         for fn, name in ((self._capture, "clvhd-camera"), (self._process, "clvhd-hand"))]
        for th in self._threads:
            th.start()
        self.running = True
        return self

    def stop(self):
        if not self.running:
            return
        self._stop.set()
        with self._new:
            self._new.notify_all()
        for th in self._threads:
            th.join(timeout=2)
        self.cap.release()
        self.hand_model.close()
        self.running = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def read(self):
        """The HandFrames processed since the last call, oldest first."""
        if self._error is not None:
            raise self._error
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    def latest(self):
        """(mirrored BGR image, HandFrame) of the last processed frame, (None, None) before the first."""
        with self._lock:
            return self._latest

    def stats(self):
        return {"captured": self.captured, "processed": self.processed,
                "nominal_fps": self.nominal_fps}

    def _guard(self, fn):
        try:
            fn()
        except Exception as e:  # raised to the caller by read()
            self._error = e
            self._stop.set()
            with self._new:
                self._new.notify_all()

    def _capture(self):
        period = 1 / self.nominal_fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self.is_file:  # a file plays at its frame rate, as a camera would deliver it
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
            ok, image = self.cap.read()
            t = time.monotonic()  # when the frame reached us: about one frame after its exposure
            if not ok:
                if self.is_file:
                    self._stop.set()
                    with self._new:
                        self._new.notify_all()
                    return
                time.sleep(0.005)
                continue
            with self._new:
                self._frame = (self.captured, t, image)
                self.captured += 1
                self._new.notify()

    def _process(self):
        while True:
            with self._new:
                while self._frame is None and not self._stop.is_set():
                    self._new.wait(0.5)
                if self._frame is None:
                    return
                index, t, image = self._frame
                self._frame = None
            ms = max(int(t * 1000), self._last_ms + 1)  # MediaPipe needs increasing timestamps
            self._last_ms = ms
            if self.mirror:
                image = self._cv2.flip(image, 1)
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            frame = HandFrame.empty(t, index)
            hand = self.hand_model.detect_for_video(mp_image, ms)
            if hand.hand_landmarks:
                frame.handedness = hand.handedness[0][0].category_name
                frame.score = hand.handedness[0][0].score
                frame.hand_image = np.array([(p.x, p.y, p.z) for p in hand.hand_landmarks[0]])
                frame.hand_world = np.array([(p.x, p.y, p.z) for p in hand.hand_world_landmarks[0]])
            with self._lock:
                self._results.append(frame)
                self._latest = (image, frame)
                self.processed += 1


# one camera frame as a fixed-size record, to stream to a file (clvhd.bids.RawRecording);
# right: 1 right hand, 0 left, NaN no hand
FRAME_DTYPE = np.dtype([("t", "<f8"), ("index", "<i8"), ("right", "<f4"), ("score", "<f4"),
                        ("hand_image", "<f4", (21, 3)), ("hand_world", "<f4", (21, 3))])


def stack(frames):
    """HandFrames as ``FRAME_DTYPE`` records."""
    out = np.zeros(len(frames), FRAME_DTYPE)
    for i, f in enumerate(frames):
        out[i] = (f.t, f.index, float(f.handedness == "Right") if f.handedness else np.nan, f.score,
                  f.hand_image, f.hand_world)
    return out
