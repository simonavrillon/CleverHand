"""Cued finger gestures (``clvhd.cues``)."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from clvhd import cues
from clvhd.hand import FINGERS


def test_protocol_runs_every_gesture_once_per_block(tmp_path):
    p = cues.Protocol(repetitions=3)
    order = p.trial_order(np.random.default_rng(0))
    assert len(order) == 3 * len(cues.GESTURES)
    for b in range(3):
        assert sorted(g for blk, g in order if blk == b) == sorted(cues.GESTURES)
    assert p.duration() == pytest.approx(len(order) * 6.5)
    # a file, with a gesture of one's own
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "name": "two",
                "gestures": [
                    "pinch_index",
                    {
                        "id": "flex_index",
                        "name": "Index flexion",
                        "fingers": ["index"],
                        "motion": "flex",
                    },
                ],
                "repetitions": 2,
            }
        )
    )
    q = cues.Protocol.load(path)
    assert list(q.specs()) == ["pinch_index", "flex_index"]
    assert q.specs()["flex_index"]["fingers"] == ["index"]
    with pytest.raises(ValueError, match="unknown gesture"):
        cues.Protocol(gestures=["wave"]).specs()


def test_every_gesture_has_a_valid_spec():
    for g, spec in cues.GESTURES.items():
        assert set(spec["fingers"]) <= set(FINGERS), g
        assert spec["motion"] in ("pinch", "extend", "rest"), g
        if spec["motion"] == "pinch":
            assert spec["fingers"][0] == "thumb" and len(spec["fingers"]) >= 2, g
    assert len(cues.GESTURES) == 15
    assert [f for f in FINGERS if f"extend_{f}" not in cues.GESTURES] == []
    assert cues.GESTURES["pinch_all"]["fingers"] == list(FINGERS)


def test_cards_draw():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore, QtGui

    from clvhd import ui

    app = ui.application()  # kept: drawing text needs it alive
    img = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB32)
    for spec in cues.GESTURES.values():
        for lit in (False, True):
            img.fill(QtGui.QColor("#000000"))
            qp = QtGui.QPainter(img)
            cues.draw_card(qp, QtCore.QRectF(0, 0, 200, 200), spec, lit)
            qp.end()
            arr = np.frombuffer(img.constBits().asstring(200 * 200 * 4), np.uint8)
            assert arr.max() > 0  # something drawn
    assert app is not None


def test_fingertip_speed_and_movement_windows():
    t = np.arange(0, 60, 1 / 30)
    hand = np.zeros((len(t), 21, 3))
    rng = np.random.default_rng(1)
    hand += rng.normal(scale=2e-4, size=hand.shape)  # landmark jitter at rest
    go = [5.0, 15.0, 25.0, 35.0]
    for g, rt in zip(go[:3], (0.3, 0.45, 0.25), strict=True):  # the index tip moves 3 cm in 0.5 s
        m = (t >= g + rt) & (t < g + rt + 0.5)
        hand[m, 8, 0] += 0.06 * (t[m] - g - rt)
        hand[t >= g + rt + 0.5, 8, 0] += 0.03
        hand[t >= g + rt + 0.5, 8, 0] -= 0.03  # back in place after
    speed = cues.fingertip_speed(t, hand)
    assert np.isnan(speed[0]) and np.isnan(speed[-1])
    hand[(t > 34) & (t < 38)] = np.nan  # the last trial: no hand in the camera
    speed = cues.fingertip_speed(t, hand)
    w = cues.movement_windows(go, t, speed)
    for (on, off, status), rt in zip(w[:3], (0.3, 0.45, 0.25), strict=True):
        assert status == "ok"
        assert on - go[w.index((on, off, status))] == pytest.approx(rt, abs=0.07)
        assert 0.3 < off - on < 0.65
    assert w[3][2] == "no_hand"
    quiet = cues.movement_windows([45.0], t, speed)  # nothing moves there
    assert quiet[0][2] == "no_movement"
