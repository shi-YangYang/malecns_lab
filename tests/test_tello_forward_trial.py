"""Offline checks for a timed forward-command probe, without aircraft access."""

import csv
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_tello_cruise_trial import CruiseWorker
from test_tello_yaw_trial import FakeClock, FakeTello, trial


class ForwardTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.enterContext(patch.object(trial.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(trial.time, "sleep", self.clock.sleep))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())

    def run_forward(self, drone, worker, name, seconds=5):
        path = Path(self.directory) / f"{name}.csv"
        code = trial.run_trial(drone, worker, seconds, path, mode="forward-check")
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        return code, rows

    def assert_landed(self, drone, worker):
        names = [event[0] for event in drone.events]
        self.assertIn("land", names)
        self.assertLess(names.index("land"), names.index("end"))
        self.assertLess(names.index("end"), names.index("worker_close"))
        landed_at = next(event[1] for event in drone.events if event[0] == "land")
        self.assertLessEqual(landed_at - drone.takeoff_at, 5.1)
        self.assertTrue(worker.closed)

    def test_explicit_takeoff_and_fixed_duration_required_before_sdk(self):
        with (patch.dict(sys.modules, {"djitellopy": None}),
              patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start"))):
            self.assertEqual(trial.main(["--mode", "forward-check"]), 0)
            for duration in ("1", "4", "5.1"):
                with self.subTest(duration=duration), self.assertRaises(SystemExit) as raised:
                    trial.main(["--mode", "forward-check", "--seconds", duration, "--takeoff"])
                self.assertEqual(raised.exception.code, 2)
        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone)
        code, _ = self.run_forward(drone, worker, "short", seconds=4)
        self.assertEqual(code, 1)
        self.assertNotIn("connect", [event[0] for event in drone.events])

    def test_one_second_pulse_ignores_model_axes_and_escape_then_logs_after_landing(self):
        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone, pattern=lambda t: (-1.0, 1.0, -1.0, True),
                              ground_throttle=1.0, ground_escape=True)
        real_writer = csv.DictWriter

        def observe_writer(*args, **kwargs):
            drone.events.append(("write_csv", self.clock.elapsed))
            return real_writer(*args, **kwargs)

        with patch.object(trial.csv, "DictWriter", side_effect=observe_writer):
            code, rows = self.run_forward(drone, worker, "normal")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        names = [event[0] for event in drone.events]
        self.assertLess(names.index("worker_close"), names.index("write_csv"))
        forwards = [row for row in rows if int(row["rc_forward"]) > 0]
        self.assertTrue(forwards)
        self.assertGreaterEqual(float(forwards[0]["t_s"]), 2.0)
        self.assertLess(float(forwards[-1]["t_s"]), 3.0)
        self.assertTrue(all(row["escape"] == "True" for row in forwards))
        self.assertTrue(any(float(row["t_s"]) >= 4.5 for row in rows))
        for row in rows:
            t = float(row["t_s"])
            self.assertEqual(row["control_source"], "fixed_probe")
            axes = tuple(int(row[f"rc_{axis}"]) for axis in ("lateral", "forward", "throttle", "yaw"))
            self.assertIn(axes, ((0, 0, 0, 0), (0, 15, 0, 0)))
            if t < 2.0 or t >= 3.0:
                self.assertEqual(axes, (0, 0, 0, 0))
        last_forward = max(i for i, event in enumerate(drone.events)
                           if event[0] == "rc" and event[2][1] != 0)
        first_zero = next(event for event in drone.events[last_forward + 1:] if event[0] == "rc")
        self.assertEqual(first_zero[2], (0, 0, 0, 0))
        self.assertLess(first_zero[1] - drone.takeoff_at, 3.04)

    def test_delayed_fresh_results_cannot_start_early_or_extend_pulse(self):
        drone = FakeTello(self.clock)
        completed = []

        class RecordedWorker(CruiseWorker):
            def poll(self):
                response = super().poll()
                if response is not None and self.drone.flying:
                    completed.append((self.drone.clock.elapsed - self.drone.takeoff_at,
                                      self.frame_time, self.last_reset))
                return response

        worker = RecordedWorker(drone, result_delay=0.14,
                                pattern=lambda t: (1.0, 1.0, 1.0, True))
        code, rows = self.run_forward(drone, worker, "delayed")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        forwards = [row for row in rows if int(row["rc_forward"]) > 0]
        self.assertTrue(forwards)
        self.assertGreaterEqual(float(forwards[0]["t_s"]), 2.14)
        self.assertLess(float(forwards[-1]["t_s"]), 3.0)
        first_rc = next(event[1] - drone.takeoff_at for event in drone.events
                        if event[0] == "rc" and event[2][1] > 0)
        self.assertTrue(any(received <= first_rc and frame_at >= 2.0 and not priming
                            for received, frame_at, priming in completed))

    def test_no_post_window_frame_result_means_no_forward_command(self):
        drone = FakeTello(self.clock)

        class StopsAtTwoSeconds(CruiseWorker):
            def poll(self):
                if self.drone.flying and self.frame_time >= 2.0:
                    return None
                return super().poll()

        worker = StopsAtTwoSeconds(drone)
        code, rows = self.run_forward(drone, worker, "no_new_result")
        self.assertEqual(code, 1)
        self.assert_landed(drone, worker)
        self.assertTrue(rows)
        self.assertTrue(all(event[2] == (0, 0, 0, 0)
                            for event in drone.events if event[0] == "rc"))

    def test_raw_motion_telemetry_is_optional_and_preserved(self):
        telemetry = {"pitch": -4, "roll": 2, "vgx": 12, "vgy": -3, "vgz": 1}

        class TelemetryTello(FakeTello):
            def get_current_state(self):
                state = super().get_current_state()
                state.update(telemetry)
                return state

        columns = {"state_pitch_deg": "pitch", "state_roll_deg": "roll",
                   "state_vgx": "vgx", "state_vgy": "vgy", "state_vgz": "vgz"}
        for label, aircraft in (("present", TelemetryTello), ("missing", FakeTello)):
            with self.subTest(telemetry=label):
                drone = aircraft(self.clock)
                worker = CruiseWorker(drone)
                code, rows = self.run_forward(drone, worker, label)
                self.assertEqual(code, 0, self.stderr.getvalue())
                self.assert_landed(drone, worker)
                self.assertTrue(rows)
                for row in rows:
                    for column, key in columns.items():
                        self.assertIn(column, row)
                        if label == "present":
                            self.assertEqual(float(row[column]), telemetry[key])
                        else:
                            self.assertEqual(row[column], "")

    def test_timeouts_during_pulse_request_zero_rc_and_landing(self):
        for name in ("stall", "old_result", "video", "telemetry"):
            with self.subTest(fault=name):
                drone = FakeTello(self.clock)
                worker = CruiseWorker(drone, fault=name)
                if name in ("video", "telemetry"):
                    original_send = drone.send_rc_control

                    def fail_after_start(*axes, aircraft=drone, send=original_send, fault=name):
                        send(*axes)
                        if aircraft.flying and aircraft.clock.elapsed - aircraft.takeoff_at >= 2.5:
                            if fault == "video":
                                aircraft.lose_frames = True
                            else:
                                aircraft.lose_state = True

                    drone.send_rc_control = fail_after_start
                code, rows = self.run_forward(drone, worker, name)
                self.assertEqual(code, 1)
                self.assert_landed(drone, worker)
                self.assertTrue(any(int(row["rc_forward"]) == 15 for row in rows))
                rc = [event for event in drone.events if event[0] == "rc"]
                self.assertEqual(rc[-1][2], (0, 0, 0, 0))
                landed_at = next(event[1] for event in drone.events if event[0] == "land")
                self.assertLess(landed_at - drone.takeoff_at, 3.2)

    def test_interrupt_during_pulse_and_uncertain_takeoff_still_land(self):
        for name, expected in (("lost_ack", 1), ("interrupt", 130), ("rc_failed", 1)):
            with self.subTest(fault=name):
                drone = FakeTello(self.clock, takeoff_error=(
                    TimeoutError("Takeoff acknowledgment lost") if name == "lost_ack" else None))
                worker = CruiseWorker(drone)
                if name != "lost_ack":
                    original_send = drone.send_rc_control
                    triggered = [False]

                    def fail_once(*axes, send=original_send, fault=name, raised=triggered):
                        send(*axes)
                        if axes[1] and not raised[0]:
                            raised[0] = True
                            if fault == "interrupt":
                                raise KeyboardInterrupt()
                            raise OSError("RC send failed")

                    drone.send_rc_control = fail_once
                code, _ = self.run_forward(drone, worker, name)
                self.assertEqual(code, expected, self.stderr.getvalue())
                self.assert_landed(drone, worker)
                rc = [event for event in drone.events if event[0] == "rc"]
                self.assertTrue(rc)
                self.assertEqual(rc[-1][2], (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
