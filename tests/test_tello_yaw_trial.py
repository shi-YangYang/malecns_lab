"""Bounded first-flight control and failure paths, without an aircraft or SDK."""

import csv
import io
import math
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cnsbridge.experiments.tello import policy, trial
from cnsbridge.experiments.tello.worker import BrainProcess


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        self.elapsed += 0.001
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += max(0.0, seconds)


class FakeFrames:
    def __init__(self, drone):
        self.drone = drone
        self.image = np.zeros((24, 32, 3), dtype=np.uint8)

    @property
    def frame(self):
        if self.drone.flying and self.drone.lose_frames:
            return None
        return self.image.copy()


class FakeTello:
    def __init__(self, clock, *, battery=80, takeoff_error=None, rc_error=False,
                 lose_frames=False, lose_state=False, land_error=False, first_state_delay=0.0):
        self.clock = clock
        self.battery = battery
        self.takeoff_error = takeoff_error
        self.rc_error = rc_error
        self.lose_frames = lose_frames
        self.lose_state = lose_state
        self.land_error = land_error
        self.first_state_delay = first_state_delay
        self.takeoff_at = None
        self.flying = self.is_flying = False
        self.events = []
        self.state = {"bat": battery, "h": 0, "yaw": 0}
        self.reader = FakeFrames(self)

    def connect(self):
        self.events.append(("connect", self.clock.elapsed))

    def get_current_state(self):
        if self.flying and self.clock.elapsed - self.takeoff_at < self.first_state_delay:
            return self.state
        if not self.flying or not self.lose_state:
            self.state = {"bat": self.battery, "h": 80 if self.flying else 0, "yaw": 0}
        return self.state

    def streamon(self):
        self.events.append(("streamon", self.clock.elapsed))

    def get_frame_read(self, *, with_queue, max_queue_len):
        if not with_queue or max_queue_len != 1:
            raise AssertionError("Fresh-frame supervision requires a one-frame queue")
        return self.reader

    def takeoff(self):
        self.events.append(("takeoff", self.clock.elapsed))
        self.takeoff_at = self.clock.elapsed
        self.flying = True
        if self.takeoff_error is not None:
            raise self.takeoff_error
        self.is_flying = True

    def send_rc_control(self, *axes):
        self.events.append(("rc", self.clock.elapsed, axes))
        if self.rc_error:
            raise OSError("RC send failed")

    def land(self):
        self.events.append(("land", self.clock.elapsed))
        if self.land_error:
            raise TimeoutError("Land acknowledgment lost")
        self.flying = self.is_flying = False

    def end(self):
        self.events.append(("end", self.clock.elapsed))


class FakeWorker:
    def __init__(self, drone, *, stall=False, old_result=False, nonfinite=False):
        self.drone = drone
        self.stall = stall
        self.old_result = old_result
        self.nonfinite = nonfinite
        self.pending = None
        self.sequence = 0
        self.closed = False
        self.resets = []

    def alive(self):
        return not self.closed

    def submit(self, frame, state, t, reset=False):
        self.sequence += 1
        self.pending = self.sequence
        self.resets.append((self.drone.flying, reset))
        return self.sequence

    def poll(self):
        if self.pending is None or (self.drone.flying and self.stall):
            return None
        job, self.pending = self.pending, None
        if self.drone.flying and self.old_result:
            job -= 1
        proposal = 0.5 if self.drone.flying else 0.0
        if self.drone.flying and self.nonfinite:
            proposal = float("nan")
        return job, {"phase": "ready", "cmd_yaw": proposal,
                     "raw_yaw": proposal, "processing_ms": 8.0}, None

    def close(self):
        self.closed = True
        self.drone.events.append(("worker_close", self.drone.clock.elapsed))


class YawTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.enterContext(patch.object(trial.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(trial.time, "sleep", self.clock.sleep))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())
        self.log_path = Path(self.directory) / "yaw.csv"

    def run_session(self, drone=None, worker=None, seconds=1, *, skip_warmup=False):
        drone = drone or FakeTello(self.clock)
        worker = worker or FakeWorker(drone)
        if skip_warmup:
            with patch.object(trial, "warmup"):
                code = trial.run_trial(drone, worker, seconds, self.log_path)
        else:
            code = trial.run_trial(drone, worker, seconds, self.log_path)
        return code, drone, worker

    def assert_cleaned_up(self, drone, worker):
        names = [event[0] for event in drone.events]
        self.assertIn("land", names)
        self.assertLess(names.index("land"), names.index("end"))
        self.assertLess(names.index("end"), names.index("worker_close"))
        self.assertTrue(worker.closed)

    def test_yaw_is_finite_and_clamped(self):
        for proposal, expected in ((-2, -10), (-0.1, -6), (0, 0), (0.1, 6), (2, 10)):
            with self.subTest(proposal=proposal):
                self.assertEqual(policy.yaw_rc(proposal), expected)
        for proposal in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(proposal=proposal), self.assertRaises(RuntimeError):
                policy.yaw_rc(proposal)

    def test_no_takeoff_flag_does_not_load_sdk_or_start_brain(self):
        with (patch.dict(sys.modules, {"djitellopy": None}),
              patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start"))):
            self.assertEqual(trial.main([]), 0)

    def test_actual_warmup_yaw_only_deadline_and_log_after_landing(self):
        drone = FakeTello(self.clock, battery=40)
        worker = FakeWorker(drone)
        real_writer = csv.DictWriter

        def observe_writer(*args, **kwargs):
            drone.events.append(("write_csv", self.clock.elapsed))
            return real_writer(*args, **kwargs)

        with patch.object(trial.csv, "DictWriter", side_effect=observe_writer):
            code, _, _ = self.run_session(drone, worker, seconds=5)

        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_cleaned_up(drone, worker)
        names = [event[0] for event in drone.events]
        self.assertLess(names.index("takeoff"), names.index("rc"))
        self.assertLess(names.index("worker_close"), names.index("write_csv"))
        rc = [event for event in drone.events if event[0] == "rc"]
        self.assertTrue(any(event[2][3] == 10 for event in rc))
        self.assertEqual(rc[0][2], (0, 0, 0, 0))
        self.assertTrue(all(event[2][:3] == (0, 0, 0) and abs(event[2][3]) <= 10 for event in rc))
        took_off = next(event[1] for event in drone.events if event[0] == "takeoff")
        landed = next(event[1] for event in drone.events if event[0] == "land")
        self.assertLessEqual(landed - took_off, 5.1)
        self.assertEqual([reset for flying, reset in worker.resets if flying].count(True), 1)
        with self.log_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertTrue(all(float(row["t_s"]) <= 5.01 for row in rows))
        self.assertTrue(all(int(row[axis]) == 0 for row in rows
                            for axis in ("rc_lateral", "rc_forward", "rc_throttle")))

    def test_low_battery_and_existing_log_prevent_takeoff(self):
        drone = FakeTello(self.clock, battery=39)
        code, _, worker = self.run_session(drone, skip_warmup=True)
        self.assertEqual(code, 1)
        self.assertNotIn("takeoff", [event[0] for event in drone.events])
        self.assertTrue(worker.closed)
        previous = "previous,experiment\n1,2\n"
        self.log_path.write_text(previous, encoding="utf-8")
        code, drone, worker = self.run_session(skip_warmup=True)
        self.assertEqual(code, 1)
        self.assertNotIn("connect", [event[0] for event in drone.events])
        self.assertEqual(self.log_path.read_text(encoding="utf-8"), previous)
        self.assertTrue(worker.closed)

    def test_uncertain_takeoff_and_rc_failure_still_land(self):
        for label, options, expected in (
            ("lost_ack", {"takeoff_error": TimeoutError("Takeoff acknowledgment lost")}, 1),
            ("interrupted", {"takeoff_error": KeyboardInterrupt()}, 130),
            ("rc_failed", {"rc_error": True}, 1),
        ):
            with self.subTest(failure=label):
                self.log_path = Path(self.directory) / f"{label}.csv"
                drone = FakeTello(self.clock, **options)
                code, _, worker = self.run_session(drone, skip_warmup=True)
                self.assertEqual(code, expected, self.stderr.getvalue())
                self.assert_cleaned_up(drone, worker)

    def test_stalled_old_or_nonfinite_brain_output_cannot_continue_flight(self):
        for failure in ("stall", "old_result", "nonfinite"):
            with self.subTest(failure=failure):
                self.log_path = Path(self.directory) / f"{failure}.csv"
                drone = FakeTello(self.clock)
                worker = FakeWorker(drone, **{failure: True})
                code, _, _ = self.run_session(drone, worker, skip_warmup=True)
                self.assertEqual(code, 1)
                self.assert_cleaned_up(drone, worker)
                self.assertTrue(all(event[2] == (0, 0, 0, 0) for event in drone.events if event[0] == "rc"))

    def test_lost_video_or_telemetry_lands(self):
        for failure in ("lose_frames", "lose_state"):
            with self.subTest(failure=failure):
                self.log_path = Path(self.directory) / f"{failure}.csv"
                drone = FakeTello(self.clock, **{failure: True})
                code, _, worker = self.run_session(drone, skip_warmup=True)
                self.assertEqual(code, 1)
                self.assert_cleaned_up(drone, worker)

    def test_delayed_first_airborne_packet_keeps_zero_sticks_then_proceeds(self):
        drone = FakeTello(self.clock, first_state_delay=0.4)
        code, _, worker = self.run_session(drone, skip_warmup=True)
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_cleaned_up(drone, worker)
        rc = [event for event in drone.events if event[0] == "rc"]
        self.assertTrue(rc)
        self.assertEqual(rc[0][2], (0, 0, 0, 0))
        nonzero = [event for event in rc if event[2][3] != 0]
        self.assertTrue(nonzero)
        self.assertGreaterEqual(nonzero[0][1] - drone.takeoff_at, 0.4)

    def test_landing_ack_failure_is_reported(self):
        drone = FakeTello(self.clock, land_error=True)
        code, _, worker = self.run_session(drone, skip_warmup=True)
        self.assertEqual(code, 1)
        self.assertEqual([event[0] for event in drone.events].count("land"), 2)
        self.assert_cleaned_up(drone, worker)
        self.assertNotIn("TRIAL COMPLETE", self.stdout.getvalue())


class BrainProcessTests(unittest.TestCase):
    def test_real_spawn_returns_finite_result_and_stops(self):
        cfg = trial.load_config(trial.PROJECT / "configs" / "tello.yaml",
                                overrides={"brain": {"source": "minifly"}})
        worker = BrainProcess(cfg)
        try:
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            job = worker.submit(frame, {"bat": 80, "h": 0, "yaw": 0}, 0.0, reset=True)
            deadline = time.monotonic() + 10
            response = None
            while time.monotonic() < deadline and response is None:
                response = worker.poll()
                if response is None:
                    time.sleep(0.01)
            self.assertIsNotNone(response, "Spawned brain did not respond within 10 seconds")
            received, row, error = response
            self.assertIsNone(error)
            self.assertEqual(received, job)
            self.assertTrue(math.isfinite(row["cmd_yaw"]))
        finally:
            worker.close()
        self.assertFalse(worker.alive())


if __name__ == "__main__":
    unittest.main()
