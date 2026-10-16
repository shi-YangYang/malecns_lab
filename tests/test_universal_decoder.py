"""Offline contracts for a learned neural readout and two physical adapters.

Synthetic rates have independently specified SI targets. These checks establish
software behavior, not that MaleCNS signals contain a useful control policy.
"""

import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cnsbridge.motor.body import BODY_AXES, BodyVelocity
from cnsbridge.motor.body_adapters import DifferentialDriveAdapter, FlightCommandAdapter
from cnsbridge.motor.neural import NeuralObservation, NeuralSchema, RidgeMotionDecoder


class NeuralDecoderTests(unittest.TestCase):
    def setUp(self):
        self.schema = NeuralSchema(tuple(f"neuron_{i}" for i in range(8)), "synthetic-v1", 50.0)
        generator = np.random.default_rng(711)
        self.train_rates = generator.uniform(2.0, 70.0, size=(256, 8))
        self.heldout_rates = generator.uniform(2.0, 70.0, size=(64, 8))
        # A constant feature must not cause a division by zero during scaling.
        self.train_rates[:, -1] = 12.5
        self.heldout_rates[:, -1] = 12.5
        self.weights = generator.normal(0.0, 0.004, size=(8, 6))
        self.bias = np.array([0.12, -0.05, 0.03, 0.01, -0.02, 0.04])
        self.train_targets = self.truth(self.train_rates)

    def truth(self, rates):
        return (rates - 30.0) @ self.weights + self.bias

    def fit_decoder(self):
        return RidgeMotionDecoder.fit(self.schema, self.train_rates, self.train_targets, ridge=1e-6)

    def test_learned_si_targets_generalize_to_independent_samples(self):
        decoder = self.fit_decoder()
        predictions = decoder.predict_batch(self.heldout_rates, schema=self.schema)
        self.assertEqual(predictions.shape, (64, 6))
        np.testing.assert_allclose(predictions, self.truth(self.heldout_rates), atol=1e-5, rtol=0)
        observation = NeuralObservation(
            self.schema, dict(zip(self.schema.feature_names, self.heldout_rates[0]))
        )
        motion = decoder.decode(observation)
        self.assertIsInstance(motion, BodyVelocity)
        np.testing.assert_allclose(motion.as_array(), predictions[0], atol=1e-12, rtol=0)

    def test_schema_validates_identity_and_is_immutable(self):
        invalid = (
            ((), "source", 50.0),
            (("a", "a"), "source", 50.0),
            (("",), "source", 50.0),
            (("a",), "", 50.0),
            (("a",), "source", 0.0),
            (("a",), "source", -1.0),
            (("a",), "source", float("nan")),
            (("a",), "source", float("inf")),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                NeuralSchema(*arguments)
        with self.assertRaises(FrozenInstanceError):
            self.schema.source_id = "changed"

    def test_observation_orders_features_and_rejects_missing_or_invalid_rates(self):
        schema = NeuralSchema(("left", "right"), "synthetic", 50.0)
        observation = NeuralObservation(schema, {"right": 12.0, "extra": 100.0, "left": 4.0})
        np.testing.assert_array_equal(observation.vector(), [4.0, 12.0])
        for rates in (
            {"left": 1.0},
            {"left": -1.0, "right": 1.0},
            {"left": float("nan"), "right": 1.0},
            {"left": 1.0, "right": float("inf")},
        ):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                NeuralObservation(schema, rates).vector()

    def test_decoder_rejects_changed_feature_order_source_or_rate_window(self):
        decoder = self.fit_decoder()
        changed_schemas = (
            NeuralSchema(tuple(reversed(self.schema.feature_names)), self.schema.source_id, 50.0),
            NeuralSchema(self.schema.feature_names, "another-connectome", 50.0),
            NeuralSchema(self.schema.feature_names, self.schema.source_id, 100.0),
        )
        rates = dict(zip(self.schema.feature_names, self.heldout_rates[0]))
        for schema in changed_schemas:
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError):
                    decoder.decode(NeuralObservation(schema, rates))
                with self.assertRaises(ValueError):
                    decoder.predict_batch(self.heldout_rates, schema=schema)

    def test_training_rejects_invalid_shapes_rates_targets_and_regularization(self):
        for ridge in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                RidgeMotionDecoder.fit(self.schema, self.train_rates, self.train_targets, ridge=ridge)
        invalid_data = [
            (self.train_rates[:, :-1], self.train_targets),
            (self.train_rates, self.train_targets[:-1]),
            (self.train_rates, self.train_targets[:, :-1]),
            (self.train_rates[0], self.train_targets),
            (self.train_rates[:0], self.train_targets[:0]),
        ]
        for value in (-1.0, float("nan"), float("inf")):
            rates = self.train_rates.copy()
            rates[0, 0] = value
            invalid_data.append((rates, self.train_targets))
        for value in (float("nan"), float("inf")):
            targets = self.train_targets.copy()
            targets[0, 0] = value
            invalid_data.append((self.train_rates, targets))
        for index, (rates, targets) in enumerate(invalid_data):
            with self.subTest(case=index), self.assertRaises(ValueError):
                RidgeMotionDecoder.fit(self.schema, rates, targets)

    def test_batch_prediction_rejects_bad_dimensions_and_nonfinite_rates(self):
        decoder = self.fit_decoder()
        for values in (
            self.heldout_rates[:, :-1],
            self.heldout_rates[0],
            np.full((1, 8), -1.0),
            np.full((1, 8), float("nan")),
            np.full((1, 8), float("inf")),
        ):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                decoder.predict_batch(values, schema=self.schema)

    def test_save_load_preserves_predictions_and_never_overwrites(self):
        decoder = self.fit_decoder()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readout.json"
            decoder.save(path)
            original = path.read_bytes()
            restored = RidgeMotionDecoder.load(path)
            np.testing.assert_allclose(
                restored.predict_batch(self.heldout_rates, schema=self.schema),
                decoder.predict_batch(self.heldout_rates, schema=self.schema),
                atol=1e-12,
                rtol=0,
            )
            with self.assertRaises(FileExistsError):
                decoder.save(path)
            self.assertEqual(path.read_bytes(), original)
            metadata = json.loads(original)
            self.assertEqual(metadata["schema"]["feature_names"], list(self.schema.feature_names))
            self.assertEqual(metadata["schema"]["source_id"], self.schema.source_id)
            self.assertEqual(metadata["schema"]["rate_window_ms"], 50.0)
            self.assertEqual(metadata["output"]["axes"], list(BODY_AXES))
            self.assertEqual(metadata["output"]["frame"], "body_flu")
            self.assertEqual(metadata["output"]["units"], ["m/s"] * 3 + ["rad/s"] * 3)

    def test_load_rejects_incompatible_metadata_and_malformed_parameters(self):
        decoder = self.fit_decoder()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readout.json"
            decoder.save(path)
            original = json.loads(path.read_text())
            changes = (
                ("format", "other-model"),
                ("version", 999),
                ("schema.source_id", ""),
                ("schema.rate_window_ms", 0.0),
                ("schema.feature_names", ["same"] * 8),
                ("output.axes", list(reversed(BODY_AXES))),
                ("output.frame", "world_enu"),
                ("output.units", ["cm/s"] * 3 + ["deg/s"] * 3),
                ("feature_mean_hz", [0.0] * 7),
                ("feature_scale_hz", [0.0] * 8),
                ("feature_scale_hz", [-1.0] * 8),
                ("weights", [[0.0] * 5] * 8),
                ("weights", [[float("nan")] * 6] * 8),
                ("bias", [float("inf")] * 6),
            )
            for key, value in changes:
                malformed = json.loads(json.dumps(original))
                section = malformed
                parts = key.split(".")
                for part in parts[:-1]:
                    section = section[part]
                section[parts[-1]] = value
                path.write_text(json.dumps(malformed))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RidgeMotionDecoder.load(path)
            malformed = dict(original)
            del malformed["output"]
            path.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError):
                RidgeMotionDecoder.load(path)


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

    def test_same_learned_planar_motion_transfers_without_refitting_decoder(self):
        schema = NeuralSchema(("cell_a", "cell_b"), "planar-fixture", 50.0)
        rates = np.array([(a, b) for a in (5.0, 15.0, 25.0, 35.0) for b in (5.0, 15.0, 25.0, 35.0)])
        targets = np.zeros((len(rates), 6))
        targets[:, 0] = 0.01 * (rates[:, 0] - 20.0)
        targets[:, 5] = 0.02 * (rates[:, 1] - 20.0)
        decoder = RidgeMotionDecoder.fit(schema, rates, targets, ridge=1e-6)
        ground = DifferentialDriveAdapter(wheel_radius_m=0.1, track_width_m=0.5, max_wheel_rad_s=10.0)
        flight = self.flight_adapter()
        for a, b in ((12.0, 30.0), (31.0, 11.0)):
            with self.subTest(a=a, b=b):
                motion = decoder.decode(NeuralObservation(schema, {"cell_a": a, "cell_b": b}))
                expected = [0.01 * (a - 20.0), 0.0, 0.0, 0.0, 0.0, 0.02 * (b - 20.0)]
                np.testing.assert_allclose(motion.as_array(), expected, atol=1e-6, rtol=0)
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
