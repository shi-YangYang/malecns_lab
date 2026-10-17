"""Offline body-target and adapter contracts; not evidence of device transfer."""

import unittest

import numpy as np

from cnsbridge.control.body import BODY_AXES, BodyVelocity
from cnsbridge.control.adapters import DifferentialDriveAdapter, FlightCommandAdapter


class BodyAdapterTests(unittest.TestCase):
    def flight_adapter(self):
        return FlightCommandAdapter(
            forward_scale_mps=2.0,
            lateral_scale_mps=1.0,
            vertical_scale_mps=0.5,
            yaw_scale_rad_s=1.0,
        )

    def test_body_axes_units_and_finite_validation(self):
        self.assertEqual(BODY_AXES, ("vx", "vy", "vz", "wx", "wy", "wz"))
        np.testing.assert_array_equal(BodyVelocity().as_array(), np.zeros(6))
        values = [0.2, -0.3, 0.4, -0.1, 0.05, 0.8]
        np.testing.assert_array_equal(BodyVelocity.from_array(values).as_array(), values)
        for values in ([0.0] * 5, [0.0] * 7, [float("nan")] * 6, [float("inf")] * 6):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BodyVelocity.from_array(values)
        with self.assertRaises(ValueError):
            BodyVelocity(vx=float("nan"))

    def test_flight_adapter_preserves_physical_direction_with_explicit_sign_conversion(self):
        target = BodyVelocity(vx=0.4, vy=0.2, vz=0.1, wz=0.3)
        result = self.flight_adapter().adapt(target)
        self.assertAlmostEqual(result.command.forward, 0.2)
        self.assertAlmostEqual(result.command.lateral, -0.2)
        self.assertAlmostEqual(result.command.throttle, 0.2)
        self.assertAlmostEqual(result.command.yaw, -0.3)
        np.testing.assert_allclose(result.limited_target.as_array(), target.as_array())
        self.assertFalse(result.saturated_axes)

    def test_flight_saturation_reports_actual_limited_si_target(self):
        result = self.flight_adapter().adapt(BodyVelocity(vx=4.0, vy=-2.0, vz=-3.0, wz=2.0))
        self.assertEqual(result.command.forward, 1.0)
        self.assertEqual(result.command.lateral, 1.0)
        self.assertEqual(result.command.throttle, -1.0)
        self.assertEqual(result.command.yaw, -1.0)
        np.testing.assert_allclose(result.limited_target.as_array(), [2.0, -1.0, -0.5, 0.0, 0.0, 1.0])
        self.assertEqual(set(result.saturated_axes), {"vx", "vy", "vz", "wz"})

    def test_adapters_reject_unsupported_axes(self):
        flight = self.flight_adapter()
        ground = DifferentialDriveAdapter(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=10.0)
        for axis in ("wx", "wy"):
            with self.subTest(adapter="flight", axis=axis), self.assertRaises(ValueError):
                flight.adapt(BodyVelocity(**{axis: 0.1}))
        for axis in ("vy", "vz", "wx", "wy"):
            with self.subTest(adapter="ground", axis=axis), self.assertRaises(ValueError):
                ground.adapt(BodyVelocity(**{axis: 0.1}))

    def test_adapter_calibrations_require_positive_finite_values(self):
        with self.assertRaises(TypeError):
            FlightCommandAdapter()
        flight_values = dict(
            forward_scale_mps=1.0, lateral_scale_mps=1.0,
            vertical_scale_mps=1.0, yaw_scale_rad_s=1.0,
        )
        ground_values = dict(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=10.0)
        for adapter, values in ((FlightCommandAdapter, flight_values), (DifferentialDriveAdapter, ground_values)):
            for key in values:
                for value in (0.0, -1.0, float("nan"), float("inf")):
                    arguments = dict(values, **{key: value})
                    with self.subTest(adapter=adapter.__name__, key=key, value=value), self.assertRaises(ValueError):
                        adapter(**arguments)

    def test_differential_drive_geometry_and_curvature_preserving_saturation(self):
        free = DifferentialDriveAdapter(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=20.0)
        result = free.adapt(BodyVelocity(vx=1.0, wz=2.0))
        self.assertAlmostEqual(result.command.left_rad_s, 5.0)
        self.assertAlmostEqual(result.command.right_rad_s, 15.0)
        self.assertFalse(result.saturated_axes)
        limited = DifferentialDriveAdapter(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=5.0)
        result = limited.adapt(BodyVelocity(vx=1.0, wz=2.0))
        self.assertAlmostEqual(result.command.left_rad_s, 5.0 / 3.0)
        self.assertAlmostEqual(result.command.right_rad_s, 5.0)
        self.assertAlmostEqual(result.limited_target.vx, 1.0 / 3.0)
        self.assertAlmostEqual(result.limited_target.wz, 2.0 / 3.0)
        self.assertAlmostEqual(result.limited_target.wz / result.limited_target.vx, 2.0)
        self.assertTrue(result.saturated_axes)

    def test_same_planar_target_has_consistent_direction_in_both_adapters(self):
        ground = DifferentialDriveAdapter(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=10.0)
        flight = self.flight_adapter()
        for vx, wz in ((-0.08, 0.2), (0.11, -0.18)):
            with self.subTest(vx=vx, wz=wz):
                motion = BodyVelocity(vx=vx, wz=wz)
                drone = flight.adapt(motion)
                robot = ground.adapt(motion)
                wheel_vx = (robot.command.left_rad_s + robot.command.right_rad_s) * 0.1 / 2.0
                wheel_wz = (robot.command.right_rad_s - robot.command.left_rad_s) * 0.1 / 0.5
                self.assertAlmostEqual(drone.command.forward * 2.0, wheel_vx)
                self.assertAlmostEqual(-drone.command.yaw, wheel_wz)
                self.assertFalse(drone.saturated_axes)
                self.assertFalse(robot.saturated_axes)


if __name__ == "__main__":
    unittest.main()
