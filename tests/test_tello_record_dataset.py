"""Ground recording contract and failure checks using fake SDK/video objects."""

import csv
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

spec = importlib.util.spec_from_file_location("tello_record_dataset", Path(__file__).parents[1] / "tools" / "tello_record_dataset.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def now(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += max(seconds, 0.000001)


class Reader:
    def __init__(self, limit=None):
        self.count = 0
        self.limit = limit

    @property
    def frame(self):
        if self.limit is not None and self.count >= self.limit:
            return None
        self.count += 1
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        frame[..., 0] = 20
        frame[..., 2] = 100
        return frame


class GroundDrone:
    def __init__(self, clock, frame_limit=None, state_fn=None, stale=False):
        self.clock = clock
        self.reader = Reader(frame_limit)
        self.state_fn = state_fn
        self.stale = stale
        self.state = {"bat": 70, "h": 0, "yaw": 0}
        self.calls = []
        self.forbidden = []

    def __getattr__(self, name):
        self.forbidden.append(name)
        raise AssertionError(f"Unsupported ground SDK operation: {name}")

    def connect(self):
        self.calls.append("connect")

    def get_current_state(self):
        if self.state_fn:
            return self.state_fn(self.clock.elapsed)
        return self.state if self.stale else dict(self.state)

    def streamon(self):
        self.calls.append("streamon")

    def get_frame_read(self, *, with_queue, max_queue_len):
        if with_queue is not True or max_queue_len != 1:
            raise AssertionError("A one-frame consuming queue is required")
        self.calls.append("get_frame_read")
        return self.reader

    def end(self):
        self.calls.append("end")


class Writer:
    def __init__(self, path, size, fps, opened=True, fail_write=False):
        self.path, self.size, self.fps = path, size, fps
        self.opened = opened
        self.fail_write = fail_write
        self.frames = []
        self.released = False
        path.touch(exist_ok=False)

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.fail_write:
            raise RuntimeError("disk write failed")
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class GroundRecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.clock = Clock()
        self.writers = []

    def factory(self, path, size, fps):
        writer = Writer(path, size, fps)
        self.writers.append(writer)
        return writer

    def run_session(self, drone=None, name="session", **kwargs):
        drone = drone or GroundDrone(self.clock)
        result = recorder.run_recording(
            drone, 5, self.directory / name, "room_a",
            clock=self.clock.now, sleep=self.clock.sleep,
            writer_factory=kwargs.pop("writer_factory", self.factory), **kwargs,
        )
        self.assertFalse(drone.forbidden)
        self.assertEqual(drone.calls[-1], "end")
        return result, drone

    def test_records_matching_sequential_frames_timestamps_and_empty_labels(self):
        code, drone = self.run_session()
        self.assertEqual(code, 0, self.stderr.getvalue())
        session = self.directory / "session"
        metadata = json.loads((session / "session.json").read_text())
        self.assertEqual(metadata, {
            "format": "flydrones.ground_recording", "version": 1,
            "session_id": "session", "scene_id": "room_a",
            "video": "video.avi", "frames": "frames.csv", "labels": "labels.csv",
            "nominal_fps": 20, "timing": "host_dequeue", "ground_only": True,
            "label_timebase": "video_playback",
        })
        with (session / "frames.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["frame_index", "t_s", "video_t_s"])
            rows = list(reader)
        with (session / "labels.csv").open(newline="") as stream:
            self.assertEqual(list(csv.reader(stream)), [list(recorder.LABEL_FIELDS)])
        self.assertEqual([int(row["frame_index"]) for row in rows], list(range(len(rows))))
        self.assertEqual([float(row["video_t_s"]) for row in rows], [index / 20 for index in range(len(rows))])
        timestamps = [float(row["t_s"]) for row in rows]
        self.assertGreater(len(rows), 90)
        self.assertTrue(all(0 <= t < 5 for t in timestamps))
        self.assertTrue(all(a < b for a, b in zip(timestamps, timestamps[1:])))
        writer = self.writers[0]
        self.assertEqual(len(writer.frames), len(rows))
        self.assertEqual(writer.size, (12, 8))
        self.assertEqual(writer.fps, 20)
        self.assertEqual(writer.frames[0][0, 0].tolist(), [100, 0, 20])
        self.assertTrue(writer.released)
        self.assertEqual(drone.calls, ["connect", "streamon", "get_frame_read", "end"])

    def test_existing_session_is_preserved_without_connecting(self):
        session = self.directory / "session"
        session.mkdir()
        existing = session / "video.avi"
        existing.write_bytes(b"existing recording")
        code, drone = self.run_session()
        self.assertEqual(code, 1)
        self.assertNotIn("connect", drone.calls)
        self.assertEqual(existing.read_bytes(), b"existing recording")
        self.assertEqual(list(session.iterdir()), [existing])

    def test_playback_label_clock_remains_separate_when_recording_is_slow(self):
        clock = self.clock

        class SlowWriter(Writer):
            def write(self, frame):
                super().write(frame)
                clock.sleep(0.2)

        code, _ = self.run_session(writer_factory=SlowWriter)
        self.assertEqual(code, 0)
        with (self.directory / "session" / "frames.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(rows[1]["t_s"]), 0.2)
        self.assertEqual(float(rows[1]["video_t_s"]), 0.05)
        self.assertEqual([float(row["video_t_s"]) for row in rows], [index / 20 for index in range(len(rows))])

    def test_invalid_initial_telemetry_fails_before_streaming(self):
        for index, state in enumerate(({}, {"bat": 70, "h": 11, "yaw": 0},
                                       {"bat": 70, "h": float("nan"), "yaw": 0},
                                       {"bat": 101, "h": 0, "yaw": 0})):
            with self.subTest(state=state):
                drone = GroundDrone(self.clock, state_fn=lambda _, s=state: s)
                code, drone = self.run_session(drone, name=f"invalid_{index}")
                self.assertEqual(code, 1)
                self.assertNotIn("streamon", drone.calls)

    def test_airborne_height_or_invalid_telemetry_during_recording_stops(self):
        for index, bad_height in enumerate((11, float("inf"))):
            with self.subTest(height=bad_height):
                self.clock = Clock()
                drone = GroundDrone(self.clock, state_fn=lambda t, h=bad_height: {
                    "bat": 70, "h": h if t >= 0.4 else 0, "yaw": 0,
                })
                code, _ = self.run_session(drone, name=f"height_{index}")
                self.assertEqual(code, 1)
                self.assertLess(self.clock.elapsed, 0.5)
                self.assertTrue(self.writers[-1].released)

    def test_stale_telemetry_stops_even_with_fresh_video(self):
        code, _ = self.run_session(GroundDrone(self.clock, stale=True))
        self.assertEqual(code, 1)
        self.assertIn("Telemetry stopped", self.stderr.getvalue())
        self.assertLess(self.clock.elapsed, 1.1)
        self.assertTrue(self.writers[0].released)

    def test_missing_or_stopped_video_does_not_create_fake_frames(self):
        for limit in (0, 2):
            with self.subTest(limit=limit):
                self.clock = Clock()
                code, _ = self.run_session(GroundDrone(self.clock, frame_limit=limit), name=f"frames_{limit}")
                self.assertEqual(code, 1)
                with (self.directory / f"frames_{limit}" / "frames.csv").open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), limit)
                if limit:
                    self.assertTrue(self.writers[-1].released)

    def test_writer_open_and_write_failure_close_resources(self):
        for index, options in enumerate(({"opened": False}, {"fail_write": True})):
            with self.subTest(options=options):
                self.clock = Clock()
                made = []

                def factory(path, size, fps):
                    made.append(Writer(path, size, fps, **options))
                    return made[-1]

                code, _ = self.run_session(name=f"writer_{index}", writer_factory=factory)
                self.assertEqual(code, 1)
                self.assertTrue(made[0].released)

    def test_preview_stop_preserves_matching_data_and_interrupt_cleans_up(self):
        code, _ = self.run_session(preview=lambda frame, row: row["frame_index"] < 2)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.writers[0].frames), 3)

        def interrupt(frame, row):
            raise KeyboardInterrupt

        code, _ = self.run_session(name="interrupted", preview=interrupt)
        self.assertEqual(code, 130)
        self.assertTrue(self.writers[-1].released)

    def test_nonfinite_clock_fails_without_saving_timestamps(self):
        drone = GroundDrone(self.clock)
        code = recorder.run_recording(
            drone, 5, self.directory / "bad_clock", "room_a",
            clock=lambda: float("nan"), sleep=self.clock.sleep,
            writer_factory=self.factory,
        )
        self.assertEqual(code, 1)
        self.assertEqual(drone.calls[-1], "end")
        self.assertFalse(self.writers)

    def test_cli_rejects_invalid_args_before_loading_sdk(self):
        for args in (["--scene-id", ""], ["--scene-id", "room", "--seconds", "nan"],
                     ["--scene-id", "room", "--seconds", "4"], []):
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                recorder.main(args)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
