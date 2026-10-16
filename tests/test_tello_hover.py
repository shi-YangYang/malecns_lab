"""First-flight failure paths, using no SDK imports or aircraft connection."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cnsbridge.experiments.tello import hover as hover


class FakeTello:
    def __init__(self, *, battery=80, takeoff_error=None, rc_error=False, land_error=False, frozen=False):
        self.battery = battery
        self.takeoff_error = takeoff_error
        self.rc_error = rc_error
        self.land_error = land_error
        self.frozen = frozen
        self.flying = False
        self.is_flying = False
        self.calls = []
        self.state = {"bat": battery, "h": 0}

    def connect(self):
        self.calls.append("connect")

    def get_current_state(self):
        if not self.frozen:
            self.state = {"bat": self.battery, "h": 80 if self.flying else 0}
        return self.state

    def takeoff(self):
        self.calls.append("takeoff")
        self.flying = True
        if self.takeoff_error:
            raise self.takeoff_error
        self.is_flying = True

    def send_rc_control(self, *axes):
        assert axes == (0, 0, 0, 0)
        self.calls.append("zero")
        if self.rc_error:
            raise OSError("RC send failed")

    def land(self):
        self.calls.append("land")
        if self.land_error:
            raise TimeoutError("land reply lost")
        self.flying = self.is_flying = False

    def end(self):
        self.calls.append("end")


class HoverTests(unittest.TestCase):
    def setUp(self):
        elapsed = [0.0]

        def sleep(seconds):
            elapsed[0] += seconds

        self.enterContext(patch.object(hover.time, "sleep", sleep))
        self.enterContext(patch.object(hover.time, "monotonic", lambda: elapsed[0]))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))

    def test_no_flag_does_not_load_sdk(self):
        with patch.dict(hover.sys.modules, {"djitellopy": None}):
            self.assertEqual(hover.main([]), 0)

    def test_hover_lands_and_closes(self):
        drone = FakeTello()
        self.assertEqual(hover.run_hover(drone), 0)
        self.assertEqual(drone.calls.count("takeoff"), 1)
        self.assertEqual(drone.calls[-2:], ["land", "end"])
        self.assertFalse(drone.flying)

    def test_failed_preflight_never_sends_motion(self):
        for drone in (FakeTello(battery=49), FakeTello(frozen=True)):
            with self.subTest(battery=drone.battery, frozen=drone.frozen):
                self.assertEqual(hover.run_hover(drone), 1)
                self.assertEqual(drone.calls, ["connect", "end"])

    def test_uncertain_takeoff_still_lands(self):
        for failure, code in ((TimeoutError("takeoff reply lost"), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(failure=type(failure).__name__):
                drone = FakeTello(takeoff_error=failure)
                self.assertEqual(hover.run_hover(drone), code)
                self.assertEqual(drone.calls[-2:], ["land", "end"])
                self.assertFalse(drone.flying)

    def test_rc_failure_does_not_skip_landing(self):
        drone = FakeTello(rc_error=True)
        self.assertEqual(hover.run_hover(drone), 1)
        self.assertEqual(drone.calls[-2:], ["land", "end"])

    def test_landing_failure_is_not_reported_as_success(self):
        drone = FakeTello(land_error=True)
        self.assertEqual(hover.run_hover(drone), 1)
        self.assertEqual(drone.calls.count("land"), 2)
        self.assertEqual(drone.calls[-1], "end")
        self.assertNotIn("SDK TEST COMPLETE", self.output.getvalue())

    def test_telemetry_stops_during_flight(self):
        drone = FakeTello()
        original_takeoff = drone.takeoff

        def takeoff_then_lose_state():
            original_takeoff()
            drone.frozen = True

        drone.takeoff = takeoff_then_lose_state
        self.assertEqual(hover.run_hover(drone), 1)
        self.assertEqual(drone.calls[-2:], ["land", "end"])
