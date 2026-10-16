"""Video/label alignment with deterministic unit-only counts; no flight model."""

import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from cnsbridge.config import default_config
from cnsbridge.learning.dataset import MotionDataset
from cnsbridge.learning.decoder import NeuralSchema, RidgeMotionDecoder


from cnsbridge.learning import prepare, train


class CountingReplay:
    """Known numerical rows isolate video alignment from neural dynamics."""

    def __init__(self, _cfg, _readout_level="neuron"):
        self.schema = NeuralSchema(("count_a", "count_b"), "unit_alignment_counts", 50.0)
        self.visual_names = tuple(f"pixel_feature_{index}" for index in range(2 * 7 * 6 * 8 + 3))
        self.provenance = {"kind": "unit_test_counts_not_neural_model"}
        self.frames = 0

    def step(self, _frame):
        self.frames += 1
        return np.array([self.frames, self.frames % 3], dtype=float), np.full(len(self.visual_names), self.frames)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(prepare, "RecordingBrain", CountingReplay))

    def labels(self, directory, rows):
        path = directory / "labels.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("start_s", "end_s", "vx_mps", "wz_rad_s", "reason"))
            writer.writerows(rows)
        return path

    def recording(self, name="run_a", scene="room_a", count=12):
        directory = self.directory / name
        directory.mkdir()
        metadata = {
            "format": "flydrones.ground_recording", "version": 1,
            "session_id": name, "scene_id": scene, "ground_only": True,
            "video": "video.avi", "frames": "frames.csv", "labels": "labels.csv",
            "nominal_fps": 20, "timing": "host_dequeue", "label_timebase": "video_playback",
        }
        (directory / "session.json").write_text(json.dumps(metadata))
        with (directory / "frames.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("frame_index", "t_s", "video_t_s"))
            # Host capture cadence deliberately differs from AVI presentation.
            writer.writerows((i, i * .1, i / 20) for i in range(count))
        writer = cv2.VideoWriter(str(directory / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20, (96, 72))
        self.assertTrue(writer.isOpened())
        y, x = np.indices((72, 96))
        for i in range(count):
            pattern = (((x + i * 2) // 8 + y // 8) % 2 * 200).astype(np.uint8)
            writer.write(np.stack((pattern, 255 - pattern, pattern), axis=-1))
        writer.release()
        self.labels(directory, [(.1, .3, .1, 0, "clear"), (.4, .6, 0, -.2, "left approach")])
        return directory

    def test_label_intervals_are_explicit_half_open_and_validate(self):
        path = self.labels(self.directory, [(0, .1, 0, .2, "right blocked"), (.2, .3, 0, 0, "stop")])
        labels = prepare.read_labels(path, .5)
        self.assertEqual(prepare.target_at(labels, 0)[5], .2)
        self.assertIsNone(prepare.target_at(labels, .1))
        np.testing.assert_array_equal(prepare.target_at(labels, .2), np.zeros(6))
        for rows in ([], [(0, .2, .1, 0, "a"), (.1, .3, 0, 0, "b")],
                     [(0, .6, 0, 0, "long")], [(0, .1, "nan", 0, "invalid")],
                     [(0, .1, 0, 0, "")], [(0, .1, "", 0, "incomplete")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prepare.read_labels(self.labels(self.directory, rows), .5)

    def test_recording_rejects_invalid_indices_times_and_label_timebase(self):
        directory = self.recording()
        metadata, times = prepare.read_recording(directory)
        self.assertAlmostEqual(times[1], .1)
        for rows in ([(0, 0), (2, .1)], [(0, .1), (1, .1)], [(0, 0), (1, "nan")]):
            with (directory / "frames.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("frame_index", "t_s"))
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                prepare.read_recording(directory)
        metadata["label_timebase"] = "host_dequeue"
        (directory / "session.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "video_playback"):
            prepare.read_recording(directory)

    def test_unlabelled_frames_advance_state_and_targets_use_playback_clock(self):
        directory = self.recording()
        engine = prepare.RecordingBrain(default_config())
        with patch.object(prepare, "RecordingBrain", return_value=engine), patch.object(engine, "step", wraps=engine.step) as step:
            dataset = prepare.extract_session(directory, default_config(), warmup_s=.15)
        self.assertEqual(step.call_count, 12)
        np.testing.assert_allclose(dataset.timestamps_s, [.15, .2, .25, .4, .45, .5, .55])
        np.testing.assert_allclose(dataset.targets[:, 0], [.1, .1, .1, 0, 0, 0, 0])
        np.testing.assert_allclose(dataset.targets[:, 5], [0, 0, 0, -.2, -.2, -.2, -.2])
        self.assertEqual(dataset.schema.rate_window_ms, 50)
        self.assertEqual(dataset.visual_features.shape[1], 2 * 7 * 6 * 8 + 3)
        self.assertTrue(np.isfinite(dataset.rates_hz).all())
        self.assertAlmostEqual(dataset.metadata["cadence"]["host_gap_p95_ms"], 100)
        # Rows retain their causal frame positions even when earlier frames are
        # unlabelled; this is an alignment assertion, not a neural response.
        np.testing.assert_array_equal(dataset.rates_hz[:, 0], [4, 5, 6, 9, 10, 11, 12])

    def test_missing_labels_fail_before_loading_brain_and_short_video_rejected(self):
        directory = self.recording()
        self.labels(directory, [])
        with patch.object(prepare, "RecordingBrain") as brain, self.assertRaisesRegex(ValueError, "No manual labels"):
            prepare.extract_session(directory, default_config())
        brain.assert_not_called()
        self.labels(directory, [(.1, .3, .1, 0, "clear")])
        with (directory / "frames.csv").open("a", newline="") as stream:
            csv.writer(stream).writerow((12, 1.2, .6))
        with self.assertRaisesRegex(ValueError, "Video ended"):
            prepare.extract_session(directory, default_config(), warmup_s=0)

    def test_recording_to_training_cli_saves_candidate_and_heldout_report(self):
        paths = []
        for name in ("a", "b", "c"):
            directory = self.recording(f"run_{name}", f"room_{name}")
            dataset = prepare.extract_session(directory, default_config(), warmup_s=0)
            path = self.directory / f"{name}.npz"
            dataset.save(path)
            paths.append(path)
        output = self.directory / "experiment"
        arguments = [*(str(path) for path in paths), "--test-scenes", "room_c", "--out-dir", str(output)]
        self.assertEqual(train.main(arguments), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["test_scene_ids"], ["room_c"])
        self.assertEqual(report["train_scene_ids"], ["room_a", "room_b"])
        self.assertEqual(report["status"], "offline_candidate_not_validated_for_actuation")
        model = RidgeMotionDecoder.load(output / "candidate.json")
        self.assertEqual(model.schema, MotionDataset.load(paths[0]).schema)
        with self.assertRaises(SystemExit):
            train.main(arguments)


if __name__ == "__main__":
    unittest.main()
