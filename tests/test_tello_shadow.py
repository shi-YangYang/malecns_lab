"""Real MiniFly processing and ground-only session failures; no aircraft or SDK."""

import csv
import importlib.util
import io
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cnsbridge.config import load_config

spec = importlib.util.spec_from_file_location("tello_shadow", Path(__file__).parents[1] / "tools" / "tello_shadow.py")
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)

AXES = ("throttle", "yaw", "forward", "lateral")


def changing_frame(index):
    y, x = np.indices((72, 96))
    pattern = ((x + 3 * index) // 6 + y // 6) % 2
    return np.stack((pattern * 180, (1 - pattern) * 120, pattern * 220), axis=-1).astype(np.uint8)


class FakeFrames:
    def __init__(self, limit=None):
        self.limit = limit
        self.count = 0

    @property
    def frame(self):
        if self.limit is not None and self.count >= self.limit:
            return None
        self.count += 1
        return changing_frame(self.count)


class GroundTello:
    """Expose only ground telemetry/video operations and reject every other call."""

    def __init__(self, frame_limit=None):
        self.calls = []
        self.forbidden = []
        self.frames = FakeFrames(frame_limit)
        self.is_flying = False

    def __getattr__(self, name):
        self.forbidden.append(name)
        raise AssertionError(f"Ground session attempted unsupported SDK operation: {name}")

    def connect(self):
        self.calls.append("connect")

    def get_current_state(self):
        return {"h": 0, "bat": 80, "yaw": 0}

    def streamon(self):
        self.calls.append("streamon")

    def get_frame_read(self, *, with_queue, max_queue_len):
        if not with_queue or max_queue_len != 1:
            raise AssertionError("Video must use a one-frame queue to detect stale input")
        self.calls.append("get_frame_read")
        return self.frames

    def end(self):
        self.calls.append("end")


@contextmanager
def session_clock():
    elapsed = [0.0]

    def monotonic():
        elapsed[0] += 0.01
        return elapsed[0]

    def sleep(seconds):
        elapsed[0] += max(0.0, seconds)

    with (
        patch.object(shadow.time, "monotonic", monotonic),
        patch.object(shadow.time, "sleep", sleep),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        yield


class ShadowTests(unittest.TestCase):
    def make_engine(self):
        cfg = load_config(
            Path(__file__).parents[1] / "configs" / "tello.yaml",
            overrides={"brain": {"source": "minifly"}, "decoder": {"settle_s": 0.1}},
        )
        return shadow.ShadowBrain(cfg), cfg

    def test_real_minifly_produces_finite_bounded_axes(self):
        engine, cfg = self.make_engine()
        rows = [engine.step(changing_frame(i), {"h": 0, "bat": 80, "yaw": i}, i * 0.2) for i in range(8)]

        self.assertGreater(engine.brain.n_neurons, 0)
        self.assertIn("settling", {row["phase"] for row in rows})
        self.assertIn("ready", {row["phase"] for row in rows})
        for row in rows:
            for axis in AXES:
                raw, cmd = row[f"raw_{axis}"], row[f"cmd_{axis}"]
                self.assertTrue(np.isfinite(raw))
                self.assertTrue(np.isfinite(cmd))
                self.assertLessEqual(abs(raw), 1.0)
                self.assertLessEqual(abs(cmd), cfg["safety"][f"max_{axis}"] + 1e-9)
            self.assertTrue(np.isfinite(row["processing_ms"]))
            self.assertGreaterEqual(row["processing_ms"], 0)
            for group in cfg["outputs"]:
                self.assertTrue(np.isfinite(row[f"rate_{group}"]))
        self.assertAlmostEqual(rows[-1]["model_t_s"] - rows[0]["model_t_s"], (len(rows) - 1) / engine.hz)

    def test_session_records_real_brain_without_motion(self):
        engine, _ = self.make_engine()
        drone = GroundTello()
        with TemporaryDirectory() as directory, session_clock():
            log_path = Path(directory) / "shadow.csv"
            result = shadow.run_shadow(drone, engine, seconds=1.0, log_path=log_path)
            with log_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(result, 0)
        self.assertTrue(rows)
        self.assertIn("ready", {row["phase"] for row in rows})
        for row in rows:
            for axis in AXES:
                self.assertTrue(np.isfinite(float(row[f"cmd_{axis}"])))
        self.assertEqual(drone.calls, ["connect", "streamon", "get_frame_read", "end"])
        self.assertEqual(drone.forbidden, [])

    def test_missing_or_stopped_frames_fail_and_close(self):
        for frame_limit in (0, 2):
            with self.subTest(frame_limit=frame_limit):
                engine, _ = self.make_engine()
                drone = GroundTello(frame_limit)
                with TemporaryDirectory() as directory, session_clock():
                    result = shadow.run_shadow(drone, engine, seconds=20, log_path=Path(directory) / "shadow.csv")
                self.assertNotEqual(result, 0)
                self.assertEqual(drone.calls[-1], "end")
                self.assertEqual(drone.forbidden, [])

    def test_brain_failure_closes_without_motion(self):
        engine, _ = self.make_engine()
        drone = GroundTello()
        with (
            TemporaryDirectory() as directory,
            session_clock(),
            patch.object(engine, "step", side_effect=RuntimeError("brain processing failed")),
        ):
            result = shadow.run_shadow(drone, engine, seconds=1, log_path=Path(directory) / "shadow.csv")
        self.assertNotEqual(result, 0)
        self.assertEqual(drone.calls[-1], "end")
        self.assertEqual(drone.forbidden, [])

    def test_existing_csv_is_preserved(self):
        engine, _ = self.make_engine()
        drone = GroundTello()
        with TemporaryDirectory() as directory, session_clock():
            log_path = Path(directory) / "shadow.csv"
            existing = "previous,experiment\n1,2\n"
            log_path.write_text(existing, encoding="utf-8")
            result = shadow.run_shadow(drone, engine, seconds=1, log_path=log_path)
            self.assertEqual(log_path.read_text(encoding="utf-8"), existing)
        self.assertNotEqual(result, 0)
        self.assertEqual(drone.forbidden, [])


if __name__ == "__main__":
    unittest.main()
