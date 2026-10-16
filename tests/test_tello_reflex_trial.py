"""Offline boundaries for vertical reflex control using fake device proposals."""

import csv
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from test_tello_yaw_trial import FakeClock, FakeTello, FakeWorker, trial

from cnsbridge.experiments.tello import policy


class ReflexWorker(FakeWorker):
    def __init__(self, drone, *, throttle=0.5, ground_throttle=0.0,
                 ground_escape=False, land_requested=False, priming_only=False,
                 **kwargs):
        super().__init__(drone, **kwargs)
        self.throttle = throttle
        self.ground_throttle = ground_throttle
        self.ground_escape = ground_escape
        self.land_requested = land_requested
        self.priming_only = priming_only
        self.last_reset = False

    def submit(self, frame, state, t, reset=False):
        self.last_reset = reset
        return super().submit(frame, state, t, reset=reset)

    def poll(self):
        result = super().poll()
        if result is None:
            return None
        job, row, error = result
        airborne = self.drone.flying
        throttle = self.throttle if airborne else self.ground_throttle
        if airborne and self.priming_only and not self.last_reset:
            throttle = 0.0
            row["cmd_yaw"] = row["raw_yaw"] = 0.0
        row.update({
            "cmd_throttle": throttle,
            "raw_throttle": throttle,
            # Intentionally older than the aircraft's changing telemetry.
            "height_cm": 80,
            "escape": False if airborne else self.ground_escape,
            "land_requested": airborne and self.land_requested,
        })
        return job, row, error


class ChangingHeightTello(FakeTello):
    def __init__(self, clock, boundary_height):
        super().__init__(clock)
        self.boundary_height = boundary_height
        self.rc_heights = []

    def get_current_state(self):
        state = super().get_current_state()
        if self.flying and self.clock.elapsed - self.takeoff_at >= 0.4:
            state["h"] = self.boundary_height
        return state

    def send_rc_control(self, *axes):
        self.rc_heights.append((self.state["h"], axes))
        super().send_rc_control(*axes)


class ReflexTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.enterContext(patch.object(trial.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(trial.time, "sleep", self.clock.sleep))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())

    def run_reflex(self, drone, worker, name):
        path = Path(self.directory) / f"{name}.csv"
        code = trial.run_trial(drone, worker, 1, path, mode="reflex")
        return code, path

    def assert_landed(self, drone, worker):
        events = [event[0] for event in drone.events]
        self.assertIn("land", events)
        self.assertLess(events.index("land"), events.index("end"))
        self.assertLess(events.index("end"), events.index("worker_close"))
        self.assertTrue(worker.closed)

    def test_mode_selection_without_takeoff_never_loads_sdk(self):
        with (patch.dict(sys.modules, {"djitellopy": None}),
              patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start"))):
            for args in ([], ["--mode", "yaw"], ["--mode", "reflex"]):
                with self.subTest(args=args):
                    self.assertEqual(trial.main(args), 0)

    def test_final_command_caps_and_height_boundaries(self):
        for throttle, height, expected in (
            (2.0, 80, 10), (-2.0, 80, -10),
            (-0.5, 50, 0), (-0.5, 49, 0), (-0.5, 51, -10),
            (0.5, 120, 0), (0.5, 121, 0), (0.5, 119, 10),
            (0.5, 50, 10), (-0.5, 120, -10),
        ):
            with self.subTest(throttle=throttle, height=height):
                row = {"cmd_throttle": throttle, "cmd_yaw": -2.0,
                       "cmd_forward": 1.0, "cmd_lateral": 1.0}
                self.assertEqual(policy.flight_rc(row, height, "reflex"), (0, 0, expected, -10))
                self.assertEqual(policy.flight_rc(row, height, "yaw"), (0, 0, 0, -10))

    def test_reflex_reaches_sdk_but_priming_is_withheld_and_csv_follows_landing(self):
        for priming_only in (False, True):
            with self.subTest(priming_only=priming_only):
                drone = FakeTello(self.clock)
                worker = ReflexWorker(drone, priming_only=priming_only)
                real_writer = csv.DictWriter

                def observe_writer(*args, **kwargs):
                    drone.events.append(("write_csv", self.clock.elapsed))
                    return real_writer(*args, **kwargs)

                with patch.object(trial.csv, "DictWriter", side_effect=observe_writer):
                    code, path = self.run_reflex(drone, worker, f"normal_{priming_only}")
                self.assertEqual(code, 0, self.stderr.getvalue())
                self.assert_landed(drone, worker)
                names = [event[0] for event in drone.events]
                self.assertLess(names.index("land"), names.index("write_csv"))
                rc = [event[2] for event in drone.events if event[0] == "rc"]
                self.assertTrue(rc)
                self.assertTrue(all(axes[:2] == (0, 0) for axes in rc))
                self.assertTrue(all(abs(axis) <= 10 for axes in rc for axis in axes[2:]))
                if priming_only:
                    self.assertTrue(all(axes == (0, 0, 0, 0) for axes in rc))
                else:
                    self.assertIn((0, 0, 10, 10), rc)
                with path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertTrue(rows)
                self.assertTrue(any(row["control_ready"] in ("False", "0") for row in rows))
                self.assertTrue(all(int(row["rc_throttle"]) == 0 and int(row["rc_yaw"]) == 0
                                    for row in rows if row["control_ready"] in ("False", "0")))
                for field in ("raw_throttle", "proposed_throttle", "expansion", "escape",
                              "rate_DNp01_L", "rate_DNp01_R", "rate_DNp03_L", "rate_DNp03_R"):
                    self.assertIn(field, rows[0])

    def test_height_cutoff_uses_current_telemetry_despite_older_proposal(self):
        for height, throttle in ((50, -0.5), (120, 0.5)):
            with self.subTest(height=height):
                drone = ChangingHeightTello(self.clock, height)
                worker = ReflexWorker(drone, throttle=throttle)
                code, _ = self.run_reflex(drone, worker, f"height_{height}")
                self.assertEqual(code, 0, self.stderr.getvalue())
                self.assert_landed(drone, worker)
                self.assertTrue(any(axes[2] != 0 for h, axes in drone.rc_heights if h == 80))
                restricted = [axes for h, axes in drone.rc_heights if h == height]
                self.assertTrue(restricted)
                self.assertTrue(all(axes[2] == 0 for axes in restricted))

    def test_stale_invalid_or_landing_request_output_ends_with_landing(self):
        for label, options in (
            ("stale", {"stall": True}),
            ("nan", {"throttle": float("nan")}),
            ("infinite", {"throttle": float("inf")}),
            ("land_request", {"land_requested": True}),
        ):
            with self.subTest(failure=label):
                drone = FakeTello(self.clock)
                worker = ReflexWorker(drone, **options)
                code, _ = self.run_reflex(drone, worker, label)
                self.assertEqual(code, 1)
                self.assert_landed(drone, worker)
                self.assertTrue(all(event[2] == (0, 0, 0, 0)
                                    for event in drone.events if event[0] == "rc"))

    def test_unsettled_vertical_output_or_ground_escape_prevents_takeoff(self):
        for label, options in (
            ("up", {"ground_throttle": 0.2}),
            ("down", {"ground_throttle": -0.2}),
            ("escape", {"ground_escape": True}),
        ):
            with self.subTest(failure=label):
                drone = FakeTello(self.clock)
                worker = ReflexWorker(drone, **options)
                code, _ = self.run_reflex(drone, worker, label)
                self.assertEqual(code, 1)
                events = [event[0] for event in drone.events]
                self.assertNotIn("takeoff", events)
                self.assertNotIn("rc", events)
                self.assertTrue(worker.closed)


if __name__ == "__main__":
    unittest.main()
