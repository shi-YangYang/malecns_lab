"""Focused tests of dataset contracts and scene-held-out evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cnsbridge.learning.dataset import MotionDataset, concatenate_datasets, evaluate_split
from cnsbridge.learning.decoder import NeuralSchema


class MotionDatasetTests(unittest.TestCase):
    def make_dataset(self, **changes):
        rates = np.array([[1, 6], [3, 4], [5, 2], [7, 1], [2, 5], [6, 3]], dtype=float)
        targets = np.zeros((6, 6))
        targets[:, 0] = rates[:, 0] * .02
        targets[:, 5] = (rates[:, 0] - rates[:, 1]) * .03
        kwargs = dict(
            schema=NeuralSchema(("a", "b"), "fixture-v1", 50),
            rates_hz=rates, visual_features=rates - 4,
            visual_feature_names=("horizontal", "vertical"), targets=targets,
            session_ids=("run-a",) * 4 + ("run-b",) * 2,
            scene_ids=("room-a",) * 4 + ("room-b",) * 2,
            timestamps_s=np.array([0, .05, .1, .15, 0, .05]),
            metadata={"label_source": "human desired velocities"},
        )
        kwargs.update(changes)
        return MotionDataset(**kwargs)

    def test_pickle_free_roundtrip_and_no_overwrite(self):
        dataset = self.make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.npz"
            dataset.save(path)
            original = path.read_bytes()
            loaded = MotionDataset.load(path)
            self.assertEqual(loaded.schema, dataset.schema)
            self.assertEqual(loaded.metadata, dataset.metadata)
            self.assertEqual(loaded.session_ids, dataset.session_ids)
            np.testing.assert_array_equal(loaded.targets, dataset.targets)
            np.testing.assert_array_equal(loaded.visual_features, dataset.visual_features)
            with np.load(path, allow_pickle=False) as artifact:
                for name in artifact.files:
                    self.assertNotEqual(artifact[name].dtype.kind, "O")
            with self.assertRaises(FileExistsError):
                dataset.save(path)
            self.assertEqual(original, path.read_bytes())

    def test_single_sample_valid_and_arrays_are_independent(self):
        original = self.make_dataset()
        rates = original.rates_hz[:1].copy()
        dataset = self.make_dataset(rates_hz=rates, visual_features=original.visual_features[:1],
                                    targets=original.targets[:1], session_ids=("one",),
                                    scene_ids=("room",), timestamps_s=[0])
        rates[0, 0] = 999
        self.assertEqual(dataset.rates_hz[0, 0], 1)
        self.assertFalse(dataset.targets.flags.writeable)

    def test_rejects_malformed_examples(self):
        cases = [dict(rates_hz=np.zeros((5, 2))), dict(rates_hz=np.full((6, 2), -1)),
                 dict(visual_features=np.full((6, 2), np.inf)), dict(targets=np.zeros((6, 4))),
                 dict(targets=np.full((6, 6), np.nan)), dict(timestamps_s=[0, 1, 2, 3, 4, -1]),
                 dict(scene_ids=("",) * 6), dict(session_ids="run"),
                 dict(visual_feature_names=("same", "same")), dict(metadata={"bad": float("nan")})]
        for changes in cases:
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    self.make_dataset(**changes)

    def test_load_rejects_incompatible_output_and_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            self.make_dataset().save(path)
            with np.load(path, allow_pickle=False) as artifact:
                content = {name: artifact[name] for name in artifact.files}
            metadata = json.loads(content["metadata_json"].item())
            metadata["output"]["frame"] = "body_frd"
            bad_path = Path(directory) / "bad.npz"
            np.savez_compressed(bad_path, **{**content, "metadata_json": np.asarray(json.dumps(metadata))})
            with self.assertRaisesRegex(ValueError, "frame"):
                MotionDataset.load(bad_path)
            np.savez_compressed(bad_path, **{**content, "rates_hz": np.array([[object()]], dtype=object)})
            with self.assertRaises(ValueError):
                MotionDataset.load(bad_path)

    def test_concatenate_requires_matching_features_and_new_sessions(self):
        first = self.make_dataset()
        second = self.make_dataset(session_ids=("run-c",) * 4 + ("run-d",) * 2)
        joined = concatenate_datasets([first, second])
        self.assertEqual(joined.rates_hz.shape, (12, 2))
        self.assertEqual(joined.scene_ids.count("room-a"), 8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            concatenate_datasets([first, first])
        different = self.make_dataset(schema=NeuralSchema(("b", "a"), "fixture-v1", 50))
        with self.assertRaisesRegex(ValueError, "schemas"):
            concatenate_datasets([first, different])
        with self.assertRaises(ValueError):
            concatenate_datasets([])

    def test_scene_holdout_fits_only_training_data_and_compares_baselines(self):
        dataset = self.make_dataset()
        model, report = evaluate_split(dataset, ("room-b",), ridge=.001)
        self.assertEqual(report["train_samples"], 4)
        self.assertEqual(report["test_samples"], 2)
        self.assertEqual(report["train_scene_ids"], ["room-a"])
        self.assertEqual(report["test_session_ids"], ["run-b"])
        np.testing.assert_allclose(model.feature_mean_hz, dataset.rates_hz[:4].mean(axis=0))
        self.assertEqual(report["active_axes"], ["vx", "wz"])
        self.assertEqual(set(report["metrics"]), {"neural", "visual_ridge", "mean_target", "zero_motion"})
        self.assertEqual(report["feature_counts"], {"neural": 2, "visual": 2})
        self.assertEqual(report["action_coverage"]["train"], {"forward": 4, "left": 2, "right": 2, "stop": 0})
        self.assertEqual(report["action_coverage"]["test"], {"forward": 2, "left": 1, "right": 1, "stop": 0})
        self.assertEqual(report["per_session_metrics"]["run-b"]["samples"], 2)
        self.assertEqual(report["per_session_metrics"]["run-b"]["metrics"], report["metrics"])
        self.assertEqual(report["metrics"]["zero_motion"]["yaw_sign_accuracy"], 0)
        for name in ("neural", "visual_ridge"):
            metrics = report["metrics"][name]
            self.assertLess(metrics["mae_by_axis"]["vx"], .001)
            self.assertLess(metrics["mae_by_axis"]["wz"], .001)
            self.assertEqual(metrics["yaw_sign_accuracy"], 1)
        self.assertGreater(report["metrics"]["mean_target"]["mae_by_axis"]["wz"], .01)
        self.assertTrue(any("one held-out scene" in warning for warning in report["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_test_labels_do_not_change_fitted_weights(self):
        dataset = self.make_dataset()
        altered_targets = dataset.targets.copy()
        altered_targets[4:, :2] = 99
        altered = self.make_dataset(targets=altered_targets)
        model, _ = evaluate_split(dataset, ("room-b",))
        changed_model, changed_report = evaluate_split(altered, ("room-b",))
        np.testing.assert_array_equal(model.weights, changed_model.weights)
        np.testing.assert_array_equal(model.bias, changed_model.bias)
        self.assertGreater(changed_report["metrics"]["neural"]["mae_by_axis"]["vx"], 90)

    def test_rejects_session_leakage_and_invalid_splits(self):
        leaking = self.make_dataset(session_ids=("same-recording",) * 6)
        with self.assertRaisesRegex(ValueError, "session"):
            evaluate_split(leaking, ("room-b",))
        dataset = self.make_dataset()
        for held_out in ((), ("missing",), ("room-a", "room-b"), ("room-b", "room-b")):
            with self.subTest(held_out=held_out):
                with self.assertRaises(ValueError):
                    evaluate_split(dataset, held_out)

    def test_still_targets_do_not_claim_yaw_accuracy(self):
        dataset = self.make_dataset(targets=np.zeros((6, 6)))
        _, report = evaluate_split(dataset, ("room-b",))
        self.assertEqual(report["active_axes"], [])
        self.assertEqual(report["varying_axes"], [])
        self.assertEqual(report["action_coverage"]["train"]["stop"], 4)
        self.assertEqual(report["action_coverage"]["test"]["stop"], 2)
        for metrics in report["metrics"].values():
            self.assertIsNone(metrics["yaw_sign_accuracy"])
            self.assertEqual(metrics["yaw_sign_samples"], 0)
        self.assertTrue(any("constant" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
