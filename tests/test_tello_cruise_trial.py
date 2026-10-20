"""Offline cruise admission, bounded actuation, and landing failure paths."""

import csv
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_tello_yaw_trial import FakeClock, FakeTello, FakeWorker, trial
from malecns_lab.control.legacy import MotorDecoder

from malecns_lab.experiments.tello import policy


class CruiseWorker(FakeWorker):
    """Produce a proposal for the submitted frame, not the later poll time."""

    def __init__(self, drone, *, pattern=None, priming_escape=True,
                 result_delay=0.0, fault=None, ground_throttle=0.0,
                 ground_escape=False):
        super().__init__(drone)
        self.pattern = pattern or self.normal
        self.priming_escape = priming_escape
        self.result_delay = result_delay
        self.fault = fault
        self.ground_throttle = ground_throttle
        self.ground_escape = ground_escape
        self.frame_time = 0.0
        self.ready_at = 0.0
        self.last_reset = False

    @staticmethod
    def normal(t):
        return (0.0, 0.0, 1.0, False) if t < 2.2 else (0.5, 0.5, 1.0, False)

    def submit(self, frame, state, t, reset=False):
        self.frame_time = t
        self.last_reset = reset
        self.ready_at = self.drone.clock.elapsed + (self.result_delay if self.drone.flying else 0.0)
        return super().submit(frame, state, t, reset=reset)

    def poll(self):
        if self.pending is None or self.drone.clock.elapsed < self.ready_at:
            return None
        airborne = self.drone.flying
        if airborne and self.frame_time >= 2.5 and self.fault == "stall":
            return None
        job, self.pending = self.pending, None
        if airborne:
            yaw, throttle, forward, escape = self.pattern(self.frame_time)
            if self.last_reset and self.priming_escape:
                yaw, throttle, forward, escape = 1.0, 1.0, 1.0, True
            if self.frame_time >= 2.5:
                if self.fault == "old_result":
                    job -= 1
                elif self.fault == "nan_forward":
                    forward = float("nan")
        else:
            yaw, throttle, forward, escape = 0.0, self.ground_throttle, 0.25, self.ground_escape
        return job, {
            "phase": "ready", "cmd_yaw": yaw, "raw_yaw": yaw,
            "cmd_throttle": throttle, "raw_throttle": throttle,
            "cmd_forward": forward, "raw_forward": forward,
            "escape": escape, "processing_ms": 8.0,
        }, None

class CruiseTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.enterContext(patch.object(trial.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(trial.time, "sleep", self.clock.sleep))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())

    def run_cruise(self, drone, worker, name, seconds=7):
        path = Path(self.directory) / f"{name}.csv"
        code = trial.run_trial(drone, worker, seconds, path, mode="cruise")
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
        self.assertLessEqual(landed_at - drone.takeoff_at, 7.1)
        self.assertTrue(worker.closed)

    def test_cli_requires_takeoff_and_fixed_seven_second_duration(self):
        with (patch.dict(sys.modules, {"djitellopy": None}),
              patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start"))):
            self.assertEqual(trial.main(["--mode", "cruise"]), 0)
            for duration in ("1", "4", "5", "5.1", "6", "7.1"):
                with self.subTest(duration=duration), self.assertRaises(SystemExit) as raised:
                    trial.main(["--mode", "cruise", "--seconds", duration, "--takeoff"])
                self.assertEqual(raised.exception.code, 2)
        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone)
        for duration in (4, 5, 6, 7.1):
            with self.subTest(duration=duration):
                code, _ = self.run_cruise(drone, worker, "short", seconds=duration)
                self.assertEqual(code, 1)
                self.assertNotIn("connect", [event[0] for event in drone.events])

    def test_cli_defaults_cruise_to_seven_seconds_with_sdk_and_flight_mocked(self):
        tello = Mock()
        with (patch.dict(sys.modules, {"djitellopy": SimpleNamespace(Tello=tello)}),
              patch.object(trial, "trial_config", return_value={}),
              patch.object(trial, "PROJECT", Path(self.directory)),
              patch.object(trial, "BrainProcess") as brain,
              patch.object(trial, "run_trial", return_value=0) as run):
            for duration_args in ([], ["--seconds", "7"]):
                with self.subTest(duration_args=duration_args):
                    self.assertEqual(trial.main(["--mode", "cruise", "--takeoff", *duration_args]), 0)
                    self.assertEqual(run.call_args.args[:3], (tello.return_value, brain.return_value, 7))
                    self.assertEqual(run.call_args.kwargs, {"mode": "cruise"})

    def test_final_axis_caps_height_limits_and_invalid_forward(self):
        for forward, escape, height, throttle, expected_forward, expected_throttle in (
            (2.0, False, 80, 2.0, 15, 10),
            (-2.0, False, 80, -2.0, 0, -10),
            (0.1, False, 80, 0.0, 6, 0),
            (0.2499999999, False, 80, 0.0, 15, 0),
            (0.2, False, 80, 0.0, 12, 0),
            (0.008, False, 80, 0.0, 0, 0),
            (0.012, False, 80, 0.0, 1, 0),
            (2.0, True, 80, 2.0, 0, 10),
            (2.0, False, 120, 2.0, 15, 0),
            (2.0, False, 50, -2.0, 15, 0),
        ):
            with self.subTest(forward=forward, escape=escape, height=height):
                row = {"cmd_yaw": -2.0, "cmd_throttle": throttle,
                       "cmd_forward": forward, "cmd_lateral": 1.0, "escape": escape}
                self.assertEqual(policy.flight_rc(row, height, "cruise"),
                                 (0, expected_forward, expected_throttle, -10))
        for forward in (float("nan"), float("inf"), -float("inf")):
            for escape in (False, True):
                with self.subTest(forward=forward, escape=escape), self.assertRaises(RuntimeError):
                    policy.flight_rc({"cmd_yaw": 0, "cmd_throttle": 0,
                                     "cmd_forward": forward, "escape": escape}, 80, "cruise")

    def test_actual_warmup_then_fixed_window_and_log_after_landing(self):
        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone)
        real_writer = csv.DictWriter

        def observe_writer(*args, **kwargs):
            drone.events.append(("write_csv", self.clock.elapsed))
            return real_writer(*args, **kwargs)

        with patch.object(trial.csv, "DictWriter", side_effect=observe_writer):
            code, rows = self.run_cruise(drone, worker, "normal")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        names = [event[0] for event in drone.events]
        self.assertLess(names.index("worker_close"), names.index("write_csv"))
        self.assertTrue(rows)
        forwards = [row for row in rows if int(row["rc_forward"]) > 0]
        self.assertTrue(forwards)
        self.assertGreaterEqual(float(forwards[0]["t_s"]), 2.0)
        self.assertGreater(float(forwards[-1]["t_s"]), 4.8)
        self.assertLess(float(forwards[-1]["t_s"]), 5.0)
        self.assertTrue(any(int(row["rc_throttle"]) == 10 and int(row["rc_yaw"]) == 10
                            for row in forwards))
        self.assertTrue(any(float(row["t_s"]) >= 6.9 for row in rows))
        stopping_events = [event for event in drone.events
                           if event[0] == "rc" and event[1] - drone.takeoff_at >= 5.0]
        self.assertTrue(stopping_events)
        self.assertLess(stopping_events[0][1] - drone.takeoff_at, 5.04)
        self.assertTrue(all(event[2] == (0, 0, 0, 0) for event in stopping_events))
        for row in rows:
            t = float(row["t_s"])
            axes = tuple(int(row[f"rc_{axis}"]) for axis in ("lateral", "forward", "throttle", "yaw"))
            self.assertEqual(axes[0], 0)
            self.assertLessEqual(axes[1], 15)
            self.assertGreaterEqual(axes[1], 0)
            self.assertTrue(all(abs(axis) <= 10 for axis in axes[2:]))
            if t < 2 or t >= 5:
                self.assertEqual(axes, (0, 0, 0, 0))
        for field in ("trial_phase", "cruise_armed", "forward_stop_reason", "raw_forward", "proposed_forward"):
            self.assertIn(field, rows[0])

    def test_escape_stops_forward_until_landing_while_reflex_remains_available(self):
        def pattern(t):
            if t < 2.2:
                return 0.0, 0.0, 1.0, False
            return 0.5, 0.5, 1.0, 2.6 <= t < 2.9

        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone, pattern=pattern)
        code, rows = self.run_cruise(drone, worker, "escape")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        self.assertTrue(any(int(row["rc_forward"]) > 0 for row in rows))
        escaped = next(i for i, row in enumerate(rows)
                       if float(row["t_s"]) >= 2.6 and row["escape"] == "True")
        after = rows[escaped:]
        self.assertTrue(all(int(row["rc_forward"]) == 0 for row in after))
        recovered = [row for row in after if row["escape"] == "False" and float(row["t_s"]) < 5]
        self.assertTrue(recovered)
        self.assertTrue(any(int(row["rc_yaw"]) != 0 and int(row["rc_throttle"]) != 0 for row in recovered))
        self.assertTrue(all(int(row[f"rc_{axis}"]) == 0
                            for row in rows if float(row["t_s"]) >= 5
                            for axis in ("lateral", "forward", "throttle", "yaw")))

    def test_admission_tolerates_isolated_small_yaw_but_blocks_persistent_or_strong_output(self):
        cases = (
            ("isolated_yaw", lambda i: (0.125 if i in (3, 9) else 0.0, 0.05, False), True),
            ("persistent_yaw", lambda i: (0.125, 0.0, False), False),
            ("opposite_yaw", lambda i: (0.125 if i % 2 else -0.125, 0.0, False), False),
            ("strong_yaw", lambda i: (0.151 if i == 9 else 0.0, 0.0, False), False),
            ("vertical", lambda i: (0.0, 0.101 if i == 9 else 0.0, False), False),
            ("escape", lambda i: (0.0, 0.0, i == 9), False),
        )
        for name, pattern, allowed in cases:
            with self.subTest(case=name):
                gate = policy.CruiseGate(10.0)
                for i in range(13):
                    yaw, throttle, escape = pattern(i)
                    gate.observe({"cmd_yaw": yaw, "cmd_throttle": throttle,
                                  "escape": escape}, 11.4 + i * 0.05)
                self.assertEqual(gate.apply((0, 15, 3, 0), 12.0, 12.05),
                                 (0, 15, 3, 0) if allowed else (0, 0, 0, 0))
                self.assertEqual(gate.armed, allowed)
                # The stronger cruise setting never extends the fixed deadline.
                self.assertEqual(gate.apply((0, 15, 10, 10), 15.0, 15.0), (0, 0, 0, 0))
                self.assertEqual(gate.phase, "stopping")

    def test_isolated_yaw_runs_model_forward_and_stops_at_five_seconds(self):
        def pattern(t):
            yaw = 0.125 if 1.75 <= t < 1.82 or 2.25 <= t < 2.32 else 0.0
            return yaw, 0.05, 0.25, False

        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone, pattern=pattern)
        code, rows = self.run_cruise(drone, worker, "isolated_yaw")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        forwards = [row for row in rows if int(row["rc_forward"]) > 0]
        self.assertTrue(forwards)
        self.assertTrue(all(int(row["rc_forward"]) == 15 for row in forwards))
        self.assertGreaterEqual(float(forwards[0]["t_s"]), 2.0)
        self.assertGreater(float(forwards[-1]["t_s"]), 4.8)
        self.assertLess(float(forwards[-1]["t_s"]), 5.0)
        self.assertTrue(any(int(row["rc_yaw"]) != 0 for row in forwards))
        self.assertTrue(all(int(row[f"rc_{axis}"]) == 0
                            for row in rows if float(row["t_s"]) >= 5
                            for axis in ("lateral", "forward", "throttle", "yaw")))

    def test_unsettled_or_insufficient_history_cannot_arm_and_is_unverified(self):
        cases = (
            ("noisy", {"pattern": lambda t: (0.5, 0.5, 1.0, False)}),
            ("late_escape", {"pattern": lambda t: (0.0, 0.0, 1.0, t < 2.25)}),
            ("sparse", {"pattern": lambda t: (0.0, 0.0, 1.0, False), "result_delay": 0.14}),
            ("zero_forward", {"pattern": lambda t: (0.0, 0.0, 0.0, False)}),
        )
        for name, options in cases:
            with self.subTest(case=name):
                drone = FakeTello(self.clock)
                worker = CruiseWorker(drone, **options)
                before = len(self.stdout.getvalue())
                code, rows = self.run_cruise(drone, worker, name)
                self.assertEqual(code, 1)
                self.assert_landed(drone, worker)
                self.assertTrue(rows)
                self.assertTrue(all(event[2] == (0, 0, 0, 0) for event in drone.events if event[0] == "rc"))
                self.assertIn("CRUISE UNVERIFIED", self.stdout.getvalue()[before:])
        for name, options in (("ground_throttle", {"ground_throttle": 0.2}),
                              ("ground_escape", {"ground_escape": True})):
            with self.subTest(case=name):
                drone = FakeTello(self.clock)
                worker = CruiseWorker(drone, **options)
                code, _ = self.run_cruise(drone, worker, name)
                self.assertEqual(code, 1)
                self.assertNotIn("takeoff", [event[0] for event in drone.events])

    def test_late_quiet_admission_does_not_extend_the_forward_window(self):
        def pattern(t):
            return (0.5, 0.5, 1.0, False) if t < 1.8 else (0.0, 0.0, 1.0, False)

        drone = FakeTello(self.clock)
        worker = CruiseWorker(drone, pattern=pattern)
        code, rows = self.run_cruise(drone, worker, "delayed")
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assert_landed(drone, worker)
        forwards = [row for row in rows if int(row["rc_forward"]) > 0]
        self.assertTrue(forwards)
        self.assertGreaterEqual(float(forwards[0]["t_s"]), 2.3)
        self.assertLess(float(forwards[-1]["t_s"]), 5.0)

    def test_faults_after_forward_starts_request_landing(self):
        for name in ("stall", "old_result", "nan_forward", "video", "telemetry"):
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
                code, rows = self.run_cruise(drone, worker, name)
                self.assertEqual(code, 1)
                self.assert_landed(drone, worker)
                self.assertTrue(any(int(row["rc_forward"]) > 0 for row in rows))
                rc = [event for event in drone.events if event[0] == "rc"]
                self.assertEqual(rc[-1][2], (0, 0, 0, 0))
                landed_at = next(event[1] for event in drone.events if event[0] == "land")
                self.assertLess(landed_at - drone.takeoff_at, 3.2)

    def test_decoder_rules_from_known_rates_preserve_cruise_and_brake_caps(self):
        with patch.object(Path, "is_file", return_value=True):
            for mode in ("yaw", "reflex"):
                self.assertEqual(trial.trial_config(mode)["decoder"]["cruise"], 0.0)
            cfg = trial.trial_config("cruise")
        self.assertEqual(Path(cfg["brain"]["source"]), trial.PROJECT / "data" / "malecns_brain.npz")
        self.assertEqual(cfg["decoder"]["cruise"], 0.25)
        # Isolate arithmetic rules with a known zero baseline; this does not
        # assert the real graph produces any chosen directional response.
        cfg["decoder"]["readout_file"] = None
        decoder = MotorDecoder(cfg)
        for _ in range(80):
            command = decoder.update({}, 0.05)
        self.assertGreater(command.forward, 0.24)
        self.assertFalse(command.escape)

        def rc_forward(command):
            return policy.flight_rc({"cmd_yaw": command.yaw, "cmd_throttle": command.throttle,
                                    "cmd_forward": command.forward, "escape": command.escape},
                                   80, "cruise")[1]

        self.assertEqual(rc_forward(command), 15)
        command = decoder.update({"DNp03_L": 10.0}, 0.05)
        self.assertGreater(rc_forward(command), 0)
        self.assertLess(rc_forward(command), 15)
        for _ in range(12):
            command = decoder.update({"DNp03_L": 50.0}, 0.05)
        self.assertEqual(command.forward, 0.0)
        self.assertEqual(rc_forward(command), 0)
        self.assertFalse(command.escape)
        for _ in range(3):
            command = decoder.update({"DNp01_L": 50.0}, 0.05)
        self.assertTrue(command.escape)
        self.assertEqual(command.forward, 0.0)
        self.assertEqual(rc_forward(command), 0)
        self.assertGreater(command.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
