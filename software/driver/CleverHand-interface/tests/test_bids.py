"""The BIDS export (``clvhd.bids``): EMG files, motion, events, and ``sourcedata/`` recordings."""

import json

import numpy as np
import pytest

from clvhd import ads1293, bids, check_contact
from clvhd.bids import _filled_blocks, _scan
from clvhd.hand import HandFrame, stack
from tests._helpers import read_tsv


class TestWriteEmg:
    def test_writes_the_edf_sidecars_and_channels(self, tmp_path):
        pyedflib = pytest.importorskip("pyedflib")
        cfg = ads1293.Config()
        fs = cfg.odr()
        n_mod, n = 2, 7000  # not a whole number of 3 s data records
        labels = [f"m{m}_ch{c + 1}" for m in range(n_mod) for c in range(3)]
        t = 100 + np.arange(n) / fs
        rng = np.random.default_rng(5)
        data = rng.normal(size=(n, 3 * n_mod)) * 30e-6 + np.linspace(-0.2, 0.05, 3 * n_mod)
        data[10:13, 1] = np.nan
        data[:, 5] = rng.normal(size=n) * 1e-6  # flat
        ent = bids.Entities("01", "rest", session="01", run="02")
        root = tmp_path / "ds"
        bids.write_dataset(root)
        bids.write_participant(root, "01", age=30, handedness="right")
        out = bids.write_emg(root, ent, t, data, cfg, labels, powerline_freq=50,
                             muscles={0: "ECR"}, contact=check_contact(data[:2000]))

        with pyedflib.EdfReader(str(out["emg"])) as r:
            assert r.getSignalLabels() == labels
            assert r.getSampleFrequency(0) == pytest.approx(fs)
            x = np.column_stack([r.readSignal(c) for c in range(len(labels))])[:n] * 1e-6
        ok = ~np.isnan(data)
        steps = np.array([float(r["resolution"]) for r in read_tsv(out["channels"])]) * 1e-6
        assert out["emg"].suffix == ".edf"
        assert (np.abs(x - data) <= steps / 2 + 1e-12)[ok].all()  # half an EDF step at most
        assert np.allclose(x[10:13, 1], np.interp(t[10:13], t[ok[:, 1]], data[ok[:, 1], 1]), atol=2e-7)

        rows = read_tsv(out["channels"])
        assert [r["name"] for r in rows] == labels
        assert rows[0]["signal_electrode"] == "M0E4" and rows[0]["reference"] == "M0E5"
        assert rows[0]["target_muscle"] == "ECR" and rows[3]["target_muscle"] == "n/a"
        assert rows[1]["n_interpolated"] == "3"
        assert rows[5]["status"] == "bad" and rows[5]["status_description"] == "flat"
        assert rows[0]["interelectrode_distance"] == "5.08"
        sidecar = json.loads(out["emg_json"].read_text())
        assert sidecar["SamplingFrequency"] == pytest.approx(fs)
        assert sidecar["RecordingDuration"] == pytest.approx(n / fs)
        assert sidecar["EMGChannelCount"] == 6
        electrodes = read_tsv(out["electrodes"])
        assert len(electrodes) == 12 and electrodes[1]["name"] == "M0E5"
        assert float(electrodes[1]["x"]) == pytest.approx(5.08)

        # motion: frames before the first EMG sample are left out; latency from it
        frames = [HandFrame.empty(t[0] - 0.5, 0), HandFrame.empty(t[0] + 0.1, 1),
                  HandFrame.empty(t[0] + 0.2, 2)]
        frames[1].handedness, frames[1].score = "Right", 0.9
        frames[1].hand_world = rng.normal(size=(21, 3))
        motion = bids.write_motion(root, ent, stack(frames), t[0], 30)
        lines = motion["motion"].read_text().splitlines()
        channels = read_tsv(motion["channels"])
        assert len(lines) == 2 and all(len(l.split("\t")) == len(channels) for l in lines)
        first = dict(zip([c["name"] for c in channels], lines[0].split("\t"), strict=True))
        from clvhd.hand import finger_flexion, pinch_aperture
        assert first["latency"] == "0.100000"
        assert first["hand_right"] == "1"
        assert float(first["index_flexion"]) == pytest.approx(
            finger_flexion(frames[1].hand_world)[1], abs=1e-3)
        assert float(first["pinch_aperture"]) == pytest.approx(
            pinch_aperture(frames[1].hand_world), abs=1e-3)
        assert lines[1].split("\t")[1] == "n/a"
        assert "tracksys-mediapipe" in motion["motion"].name
        assert motion["motion"].name.index("task-") < motion["motion"].name.index("tracksys-")

        bids.write_scans(root, ent, [out["emg"], motion["motion"]], bids.acq_time(t[0]))
        scans = read_tsv(root / "sub-01" / "ses-01" / "sub-01_ses-01_scans.tsv")
        assert {s["filename"] for s in scans} == {f"emg/{out['emg'].name}",
                                                  f"motion/{motion['motion'].name}"}
        assert read_tsv(root / "participants.tsv") == [
            {"participant_id": "sub-01", "age": "30", "sex": "n/a", "handedness": "right"}]
        assert json.loads((root / "dataset_description.json").read_text())["DatasetType"] == "raw"

    def test_edf_reports_coarse_channels_and_the_bdf_option(self, tmp_path):
        pytest.importorskip("pyedflib")
        cfg = ads1293.Config()
        t = np.arange(6000) / cfg.odr()
        data = np.random.default_rng(9).normal(size=(6000, 3)) * 20e-6
        data[3000:3010, 1] = 0.68  # an electrode lifting: the channel hits the ADC's limit
        ent = bids.Entities("01", "rest")
        labels = ["m0_ch1", "m0_ch2", "m0_ch3"]
        out = bids.write_emg(tmp_path, ent, t, data, cfg, labels)
        assert list(out["coarse"]) == ["m0_ch2"] and 10 < out["coarse"]["m0_ch2"] < 12  # uV
        out = bids.write_emg(tmp_path, ent, t, data, cfg, labels, file_format="bdf")
        assert out["emg"].suffix == ".bdf" and not out["emg"].with_suffix(".edf").exists()
        assert out["coarse"] == {}  # 256 times finer
        assert bids.next_run(tmp_path, ent) == "01"  # no run entity: nothing counted


class TestFilledBlocks:
    def test_filled_blocks_match_whole_interpolation(self):
        rng = np.random.default_rng(7)
        t = np.arange(1000) / 100.0
        data = rng.normal(size=(1000, 4))
        data[5:300, 0] = np.nan    # a gap across several blocks
        data[:40, 1] = np.nan      # leading
        data[900:, 2] = np.nan     # trailing
        data[:, 3] = np.nan        # no valid sample
        missing, lo, hi, last = _scan(data, 64)
        got = np.concatenate(list(_filled_blocks(t, data, 64, last)))
        want = data.copy()
        for c in range(3):
            ok = ~np.isnan(data[:, c])
            want[:, c] = np.interp(t, t[ok], data[ok, c])
        want[:, 3] = 0
        assert np.allclose(got, want)
        assert list(missing) == [295, 40, 100, 1000]
        assert lo[0] == np.nanmin(data[:, 0]) and hi[1] == np.nanmax(data[:, 1])


class TestRawRecording:
    def test_export_and_next_run(self, tmp_path):
        pytest.importorskip("pyedflib")
        import pyedflib
        cfg = ads1293.Config()
        root = tmp_path / "ds"
        ent = bids.Entities("01", "rest", acquisition="dry")
        assert bids.next_run(root, ent) == "01"
        labels = [f"m0_ch{c + 1}" for c in range(3)]
        fs = cfg.odr()
        rng = np.random.default_rng(6)
        n = 40000  # several 16-record blocks
        t = 50 + np.arange(n) / fs
        data = rng.normal(size=(n, 3)) * 30e-6
        data[1000:1010, 0] = np.nan
        frames = [HandFrame.empty(50 + k / 30, k) for k in range(500)]
        for f in frames[::2]:
            f.handedness, f.score = "Left", 0.8
        raw = bids.RawRecording(root, bids.Entities("01", "rest", acquisition="dry", run="01"), cfg,
                                labels, hand=True, camera_fps=30,
                                export={"participant": {"age": 40}, "muscles": {"0": "ECR"}})
        for a in range(0, n, 700):  # as the GUI appends: chunk after chunk
            raw.append(t[a:a + 700], t[a:a + 700] - 50, data[a:a + 700])
            raw.append_frames(stack(frames[a // 80:(a + 700) // 80]))
            raw.flush()
        raw.close(stats=[{"samples": n}])
        out = bids.export_recording(raw)
        assert out["duration"] == pytest.approx(n / fs)
        assert out["sourcedata"] is None and not (root / "sourcedata").exists()  # exported: deleted
        assert bids.next_run(root, ent) == "02"

        with pyedflib.EdfReader(str(out["emg"])) as r:
            x = np.column_stack([r.readSignal(c) for c in range(3)])[:n] * 1e-6
        ok = ~np.isnan(data)
        assert np.abs(x[ok] - data[ok].astype(np.float32)).max() < 0.2e-6
        assert read_tsv(out["channels"])[0]["n_interpolated"] == "10"
        assert read_tsv(out["channels"])[0]["target_muscle"] == "ECR"
        lines = out["motion"].read_text().splitlines()
        assert len(lines) == len(frames)  # appended chunk by chunk, all after the first sample
        assert lines[0].split("\t")[-1] == "0" and lines[1].split("\t")[-1] == "n/a"
        assert read_tsv(root / "participants.tsv")[0]["age"] == "40"

    def test_export_after_a_crash_keeps_the_whole_records(self, tmp_path):
        """A recording never closed (and a half-written last record) still exports."""
        pytest.importorskip("pyedflib")
        cfg = ads1293.Config()
        raw = bids.RawRecording(tmp_path / "ds", bids.Entities("02", "rest", run="01"), cfg,
                                ["m0_ch1", "m0_ch2", "m0_ch3"])
        t = 10 + np.arange(5000) / cfg.odr()
        raw.append(t, t, np.random.default_rng(8).normal(size=(5000, 3)) * 1e-5)
        raw.flush()
        emg_file = raw._emg
        assert emg_file is not None
        emg_file.write(b"\x00" * 7)  # the crash: a partial record
        emg_file.flush()
        assert not json.loads(raw.header_path.read_text())["complete"]
        out = bids.export_recording(raw.header_path, keep_source=True)
        assert out["duration"] == pytest.approx(5000 / cfg.odr())
        assert raw.header_path.exists() and out["sourcedata"] == raw.header_path

    def test_export_dates_the_recording_from_its_header(self, tmp_path):
        """After a reboot time.monotonic() restarts: the export uses the time taken while recording."""
        pytest.importorskip("pyedflib")
        import pyedflib
        cfg = ads1293.Config()
        raw = bids.RawRecording(tmp_path / "ds", bids.Entities("04", "rest", run="01"), cfg,
                                ["m0_ch1", "m0_ch2", "m0_ch3"])
        t = 10 + np.arange(3000) / cfg.odr()
        raw.append(t, t, np.random.default_rng(9).normal(size=(3000, 3)) * 1e-5)
        raw.close()
        header = json.loads(raw.header_path.read_text())
        header["acq_time"] = "2020-01-02T03:04:05.000006"  # recorded on an earlier boot
        raw.header_path.write_text(json.dumps(header))
        out = bids.export_recording(raw.header_path)
        assert out["acq_time"] == header["acq_time"]
        scans = read_tsv(tmp_path / "ds" / "sub-04" / "sub-04_scans.tsv")
        assert {r["acq_time"] for r in scans} == {header["acq_time"]}
        with pyedflib.EdfReader(str(out["emg"])) as r:
            assert r.getStartdatetime().isoformat() == "2020-01-02T03:04:05"

    def test_sourcedata_is_kept_when_the_export_does_not_read_back(self, tmp_path, monkeypatch):
        pytest.importorskip("pyedflib")
        cfg = ads1293.Config()
        raw = bids.RawRecording(tmp_path / "ds", bids.Entities("03", "rest", session="1", run="01"),
                                cfg, ["m0_ch1", "m0_ch2", "m0_ch3"])
        t = np.arange(3000) / cfg.odr()
        raw.append(t, t, np.zeros((3000, 3)))
        raw.close()

        def broken(*args):
            raise OSError("does not read back complete")
        monkeypatch.setattr(bids, "_check_export", broken)
        with pytest.raises(OSError):
            bids.export_recording(raw)
        assert raw.header_path.exists() and raw.header_path.with_name(raw.header["emg_file"]).exists()
        monkeypatch.undo()
        bids.export_recording(raw)  # a retry succeeds and cleans up, empty folders included
        assert not (tmp_path / "ds" / "sourcedata").exists()

    def test_recorded_events_are_exported(self, tmp_path):
        pytest.importorskip("pyedflib")
        cfg = ads1293.Config()
        fs = cfg.odr()
        root = tmp_path / "ds"
        ent = bids.Entities("01", "gestures", run="01")
        raw = bids.RawRecording(root, ent, cfg, ["m0_ch1", "m0_ch2", "m0_ch3"],
                                events=["gesture", "block"],
                                export={"event_descriptions": {"gesture": {"Description": "cued"}}})
        n = 3000
        t = 100 + np.arange(n) / fs
        raw.append(t, t - 100, np.random.default_rng(2).normal(size=(n, 3)) * 30e-6)
        raw.append_event(100.5, "cue", 1.0, gesture="tap", block=1)
        raw.append_event(101.5, "go", 1.5, gesture="tap")
        raw.append_event(103.0, "break", None)
        with pytest.raises(ValueError, match="no event columns"):
            raw.append_event(104.0, "go", gesture="tap", finger="index")
        raw.close()
        out = bids.export_recording(raw)
        rows = read_tsv(out["events"])
        assert [r["trial_type"] for r in rows] == ["cue", "go", "break"]
        assert float(rows[0]["onset"]) == pytest.approx(0.5) and rows[0]["sample"] == str(round(0.5 * fs))
        assert rows[1]["block"] == "n/a" and rows[2]["duration"] == "n/a" and rows[0]["duration"] == "1"
        sidecar = json.loads(out["events"].with_suffix(".json").read_text())
        assert sidecar["gesture"]["Description"] == "cued" and sidecar["onset"]["Units"] == "s"
        assert not (root / "sourcedata").exists()  # the events file deleted with the rest
