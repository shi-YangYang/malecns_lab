"""Offline full-MaleCNS actuation boundaries and independent RC scheduling."""

import csv
import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_tello_cruise_trial import CruiseWorker
from test_tello_yaw_trial import FakeClock, FakeTello, trial

from cnsbridge.experiments.tello import policy


class FullWorker(CruiseWorker):
    """A slow model with four nonzero proposals and optional post-start stall."""

    def __init__(self, drone, **kwargs):
        kwargs.setdefault("pattern", lambda t: (0.5, -0.5, 0.5, False))
        kwargs.setdefault("result_delay", 0.1)
        super().__init__(drone, **kwargs)
        self.delivered = []

    def poll(self):
        result = super().poll()
        if result is None:
            return None
        job, row, error = result
        lateral = -0.5 if self.drone.flying else 0.0
        row.update(cmd_lateral=lateral, raw_lateral=lateral, processing_ms=100.0,
                   brain_source=str(trial.PROJECT / "data" / "malecns_brain.npz"),
                   brain_neurons=166700)
        if self.drone.flying:
            self.delivered.append((job, self.frame_time, self.drone.clock.elapsed))
        return job, row, error


class FullTrialTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.enterContext(patch.object(trial.time, "monotonic", self.clock.monotonic))
        self.enterContext(patch.object(trial.time, "sleep", self.clock.sleep))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))
        self.directory = self.enterContext(TemporaryDirectory())

    def run_full(self, drone, worker, name="full", seconds=12):
        path = Path(self.directory) / f"{name}.csv"
        code = trial.run_trial(drone, worker, seconds, path, mode="full")
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = []
        return code, rows

    def assert_landed(self, drone, worker):
        names = [event[0] for event in drone.events]
        self.assertIn("land", names)
        self.assertLess(names.index("land"), names.index("end"))
        self.assertLess(names.index("end"), names.index("worker_close"))
        self.assertTrue(worker.closed)
        landed_at = next(event[1] for event in drone.events if event[0] == "land")
        self.assertLessEqual(landed_at - drone.takeoff_at, 12.1)
        return landed_at

    def test_full_config_uses_absolute_malecns_assets_and_preserves_other_modes(self):
        with patch.object(Path, "is_file", return_value=True):
            cfg = trial.trial_config("full")
        self.assertEqual(Path(cfg["brain"]["source"]), trial.PROJECT / "data" / "malecns_brain.npz")
        self.assertEqual(Path(cfg["decoder"]["readout_file"]), trial.PROJECT / "readout_malecns.json")
        self.assertTrue(Path(cfg["brain"]["source"]).is_absolute())
        self.assertTrue(Path(cfg["decoder"]["readout_file"]).is_absolute())
        self.assertEqual(cfg["brain"]["lif"]["noise_mv"], 0.0)
        self.assertEqual(cfg["brain"]["bias"], {"DNg02_L": 8.0, "DNg02_R": 8.0})
        self.assertEqual(cfg["control"]["hz"], 20)
        self.assertEqual(cfg["decoder"]["cruise"], 0.25)
        for mode in ("yaw", "reflex", "cruise", "forward-check"):
            with self.subTest(mode=mode), patch.object(Path, "is_file", return_value=True):
                old_cfg = trial.trial_config(mode)
                self.assertEqual(Path(old_cfg["brain"]["source"]), trial.PROJECT / "data" / "malecns_brain.npz")
                self.assertEqual(Path(old_cfg["decoder"]["readout_file"]), trial.PROJECT / "readout_malecns.json")
                self.assertEqual(old_cfg["decoder"]["cruise"], 0.25 if mode == "cruise" else 0.0)

    def test_missing_assets_fail_before_worker_or_sdk(self):
        project = Path(self.directory)
        (project / "configs").mkdir()
        (project / "data").mkdir()
        (project / "configs" / "malecns.yaml").write_text(
            (trial.PROJECT / "configs" / "malecns.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        brain = project / "data" / "malecns_brain.npz"
        readout = project / "readout_malecns.json"
        readout_text = "{}"
        for missing in (brain, readout):
            with self.subTest(missing=missing.name):
                brain.touch()
                readout.write_text(readout_text, encoding="utf-8")
                missing.unlink()
                with (patch.object(trial, "PROJECT", project),
                      patch("cnsbridge.paths.source_project_root", return_value=project),
                      self.assertRaises((FileNotFoundError, ValueError))):
                    trial.trial_config("full")
                with (patch.object(trial, "PROJECT", project),
                      patch("cnsbridge.paths.source_project_root", return_value=project),
                      patch.dict(sys.modules, {"djitellopy": None}),
                      patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start")),
                      self.assertRaises(SystemExit) as raised):
                    trial.main(["--mode", "full", "--takeoff"])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_requires_takeoff_and_exact_twelve_seconds(self):
        with (patch.dict(sys.modules, {"djitellopy": None}),
              patch.object(trial, "BrainProcess", side_effect=AssertionError("Must not start"))):
            self.assertEqual(trial.main(["--mode", "full"]), 0)
            for duration in ("5", "7", "11.9", "12.1", "nan", "inf"):
                with self.subTest(duration=duration), self.assertRaises(SystemExit) as raised:
                    trial.main(["--mode", "full", "--seconds", duration, "--takeoff"])
                self.assertEqual(raised.exception.code, 2)
        drone = FakeTello(self.clock)
        worker = FullWorker(drone)
        code, _ = self.run_full(drone, worker, "bad_duration", seconds=7)
        self.assertEqual(code, 1)
        self.assertNotIn("connect", [event[0] for event in drone.events])
        policy.validate_duration(7, "cruise")
        with self.assertRaises(ValueError):
            policy.validate_duration(12, "cruise")

        tello = Mock()
        with (patch.dict(sys.modules, {"djitellopy": SimpleNamespace(Tello=tello)}),
              patch.object(trial, "trial_config", return_value={}),
              patch.object(trial, "PROJECT", Path(self.directory)),
              patch.object(trial, "BrainProcess") as brain,
              patch.object(trial, "run_trial", return_value=0) as run):
            for duration_args in ([], ["--seconds", "12"]):
                with self.subTest(duration_args=duration_args):
                    self.assertEqual(trial.main(["--mode", "full", "--takeoff", *duration_args]), 0)
                    self.assertEqual(run.call_args.args[:3], (tello.return_value, brain.return_value, 12))
                    self.assertEqual(run.call_args.kwargs, {"mode": "full"})

    def test_all_axes_are_limited_and_invalid_or_landing_proposals_rejected(self):
        row = {"cmd_yaw": -2.0, "cmd_throttle": 2.0,
               "cmd_forward": 2.0, "cmd_lateral": -2.0, "escape": False}
        self.assertEqual(policy.flight_rc(row, 80, "full"), (-10, 15, 10, -10))
        self.assertEqual(policy.flight_rc(row, 120, "full"), (-10, 15, 0, -10))
        self.assertEqual(policy.flight_rc(dict(row, cmd_throttle=-2.0), 50, "full"), (-10, 15, 0, -10))
        self.assertEqual(policy.flight_rc(dict(row, cmd_forward=-2.0), 80, "full"), (-10, 0, 10, -10))
        self.assertEqual(policy.flight_rc(dict(row, escape=True), 80, "full"), (-10, 0, 10, -10))
        with self.assertRaises(RuntimeError):
            policy.flight_rc(dict(row, land_requested=True), 80, "full")
        for axis in ("yaw", "throttle", "forward", "lateral"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(axis=axis, value=value), self.assertRaises(RuntimeError):
                    policy.flight_rc(dict(row, **{f"cmd_{axis}": value}), 80, "full")

    def test_full_gate_requires_active_window_frame_and_latches_escape(self):
        gate = policy.FullGate(10.0)
        axes = (-10, 15, 6, -6)
        self.assertEqual(gate.apply(axes, 11.8, 11.9), (0, 0, 0, 0))
        self.assertEqual(gate.apply(axes, 11.95, 12.05), (0, 0, 0, 0))
        self.assertEqual(gate.apply(axes, 12.0, 12.1), axes)
        gate.observe({"escape": True}, 12.2)
        self.assertEqual(gate.apply(axes, 12.2, 12.3), (-10, 0, 6, -6))
        gate.observe({"escape": False}, 12.4)
        self.assertEqual(gate.apply(axes, 12.4, 12.5), (-10, 0, 6, -6))
        self.assertEqual(gate.phase, "braked")
        self.assertEqual(gate.apply(axes, 15.0, 15.1), (-10, 0, 6, -6))
        self.assertEqual(gate.apply(axes, 17.0, 17.1), (-10, 0, 6, -6))
        self.assertEqual(gate.apply(axes, 19.9, 20.0), (0, 0, 0, 0))
        self.assertEqual(gate.phase, "stopping")
        # An escape in the first active frame must latch before first actuation.
        first_escape = policy.FullGate(10.0)
        first_escape.observe({"escape": True}, 12.0)
        self.assertEqual(first_escape.apply(axes, 12.0, 12.1), (-10, 0, 6, -6))
        first_escape.observe({"escape": False}, 12.2)
        self.assertEqual(first_escape.apply(axes, 12.2, 12.3), (-10, 0, 6, -6))

    def test_slow_model_reuses_fresh_results_without_slowing_rc_or_extending_window(self):
        drone = FakeTello(self.clock)
        worker = FullWorker(drone)
        code, rows = self.run_full(drone, worker)
        self.assertEqual(code, 0, self.stderr.getvalue())
        landed_at = self.assert_landed(drone, worker)
        self.assertGreaterEqual(landed_at - drone.takeoff_at, 12.0)
        self.assertGreaterEqual(len(rows), 210)
        jobs = [row["brain_job_id"] for row in rows if row["brain_job_id"]]
        self.assertGreater(len(set(jobs)), 70)
        self.assertLess(len(set(jobs)), len(rows) * 0.75)
        self.assertLess(len(set(jobs)), len(jobs))
        self.assertTrue(all(row["control_source"] == "malecns" for row in rows))
        nonzero = []
        for row in rows:
            elapsed = float(row["t_s"])
            axes = tuple(int(row[f"rc_{axis}"]) for axis in ("lateral", "forward", "throttle", "yaw"))
            if elapsed < 2 or elapsed >= 10:
                self.assertEqual(axes, (0, 0, 0, 0))
            elif any(axes):
                nonzero.append(row)
                self.assertEqual(axes, (-10, 15, -10, 10))
        self.assertTrue(nonzero)
        self.assertGreaterEqual(float(nonzero[0]["t_s"]), 2.0)
        self.assertTrue(any(5 <= float(row["t_s"]) < 7 for row in nonzero))
        self.assertTrue(any(7 <= float(row["t_s"]) < 10 for row in nonzero))
        self.assertGreater(float(nonzero[-1]["t_s"]), 9.8)
        self.assertTrue(any(float(row["t_s"]) >= 11.9 for row in rows))
        rc = [event for event in drone.events if event[0] == "rc"]
        self.assertLess(max(second[1] - first[1] for first, second in zip(rc, rc[1:])), 0.075)
        stopping = [event for event in rc if event[1] - drone.takeoff_at >= 10.0]
        self.assertTrue(stopping)
        self.assertLess(stopping[0][1] - drone.takeoff_at, 10.04)
        self.assertTrue(all(event[2] == (0, 0, 0, 0) for event in stopping))

    def test_expired_result_lands_even_while_sending_repeated_rc(self):
        drone = FakeTello(self.clock)
        worker = FullWorker(drone, fault="stall")
        code, rows = self.run_full(drone, worker, "stalled")
        self.assertEqual(code, 1)
        landed_at = self.assert_landed(drone, worker)
        self.assertIn("No fresh brain output for 0.35 seconds", self.stderr.getvalue())
        self.assertTrue(any(int(row["rc_forward"]) > 0 for row in rows))
        last_frame_time = worker.delivered[-1][1]
        age_at_landing = landed_at - drone.takeoff_at - last_frame_time
        self.assertGreaterEqual(age_at_landing, 0.35)
        self.assertLess(age_at_landing, 0.43)
        self.assertLess(landed_at - drone.takeoff_at, 3.2)

    def test_unsettled_lateral_proposal_prevents_takeoff(self):
        drone = FakeTello(self.clock)
        worker = FullWorker(drone)
        original_poll = worker.poll

        def moving_laterally():
            result = original_poll()
            if result is not None:
                result[1]["cmd_lateral"] = 0.3
            return result

        with patch.object(worker, "poll", side_effect=moving_laterally):
            code, _ = self.run_full(drone, worker, "unsettled")
        self.assertEqual(code, 1)
        self.assertNotIn("takeoff", [event[0] for event in drone.events])
        self.assertTrue(worker.closed)


if __name__ == "__main__":
    unittest.main()
