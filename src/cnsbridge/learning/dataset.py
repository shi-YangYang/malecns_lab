"""Offline, aligned examples and scene-held-out evaluation of motion readouts.

Targets are independently labelled desired body velocities, not the existing
controller's RC output. Evaluation here measures offline imitation of labels;
it does not measure flight performance or establish a connectome advantage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..control.body import BODY_AXES, BODY_FRAME, BODY_UNITS
from .decoder import NeuralSchema, RidgeMotionDecoder


def _names(values, field: str, *, unique: bool = False) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field} must be a sequence of nonempty names")
    names = tuple(values)
    if not names or any(not isinstance(value, str) or not value.strip() for value in names):
        raise ValueError(f"{field} must contain nonempty strings")
    if unique and len(set(names)) != len(names):
        raise ValueError(f"{field} must be unique")
    return names


def _output_spec() -> dict:
    return {"axes": list(BODY_AXES), "frame": BODY_FRAME, "units": list(BODY_UNITS)}


@dataclass(frozen=True)
class MotionDataset:
    """One or more recording sessions with aligned rates, features and labels.

    timestamps_s are video presentation times in seconds within each session.
    scene_ids identify physical settings, not arbitrary clips: recordings from
    the same setting should retain the same scene ID for held-out evaluation.
    visual_features are the encoder's inputs before neural processing, retained
    for a direct-visual baseline. They may contain signed motion components.
    """

    schema: NeuralSchema
    rates_hz: np.ndarray
    visual_features: np.ndarray
    visual_feature_names: tuple[str, ...]
    targets: np.ndarray
    session_ids: tuple[str, ...]
    scene_ids: tuple[str, ...]
    timestamps_s: np.ndarray
    metadata: dict | None = None

    FORMAT = "flydrones.motion_dataset"
    VERSION = 1

    def __post_init__(self):
        if not isinstance(self.schema, NeuralSchema):
            raise ValueError("schema must be a NeuralSchema")
        names = _names(self.visual_feature_names, "visual_feature_names", unique=True)
        sessions = _names(self.session_ids, "session_ids")
        scenes = _names(self.scene_ids, "scene_ids")
        count = len(sessions)
        if len(scenes) != count:
            raise ValueError("scene_ids must align with session_ids")
        for field, shape in (
            ("rates_hz", (count, len(self.schema.feature_names))),
            ("visual_features", (count, len(names))),
            ("targets", (count, len(BODY_AXES))),
            ("timestamps_s", (count,)),
        ):
            values = np.array(getattr(self, field), dtype=np.float64, copy=True)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{field} must be finite with shape {shape}")
            if field in ("rates_hz", "timestamps_s") and (values < 0).any():
                raise ValueError(f"{field} must be nonnegative")
            values.setflags(write=False)
            object.__setattr__(self, field, values)
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        try:
            metadata = json.loads(json.dumps(metadata, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise ValueError("metadata must contain finite JSON-compatible values") from exc
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "visual_feature_names", names)
        object.__setattr__(self, "session_ids", sessions)
        object.__setattr__(self, "scene_ids", scenes)

    def save(self, path: str | Path) -> None:
        """Create a pickle-free NPZ artifact; never overwrite an existing file."""
        metadata = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "schema": {"feature_names": list(self.schema.feature_names),
                       "source_id": self.schema.source_id,
                       "rate_window_ms": self.schema.rate_window_ms},
            "visual_feature_names": list(self.visual_feature_names),
            "output": _output_spec(),
            "metadata": self.metadata,
        }
        encoded = json.dumps(metadata, allow_nan=False)
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream, metadata_json=np.asarray(encoded),
                rates_hz=self.rates_hz, visual_features=self.visual_features,
                targets=self.targets, timestamps_s=self.timestamps_s,
                session_ids=np.asarray(self.session_ids, dtype=np.str_),
                scene_ids=np.asarray(self.scene_ids, dtype=np.str_),
            )

    @classmethod
    def load(cls, path: str | Path) -> MotionDataset:
        """Read and validate an artifact without enabling NumPy pickle loading."""
        with np.load(path, allow_pickle=False) as artifact:
            try:
                encoded = artifact["metadata_json"]
                if encoded.shape != () or encoded.dtype.kind != "U":
                    raise ValueError("metadata_json must be a scalar Unicode string")
                metadata = json.loads(encoded.item())
                if (not isinstance(metadata, dict) or metadata.get("format") != cls.FORMAT
                        or metadata.get("version") != cls.VERSION):
                    raise ValueError("Unsupported motion dataset format/version")
                if metadata.get("output") != _output_spec():
                    raise ValueError("Dataset motion axes, frame or units do not match")
                return cls(
                    schema=NeuralSchema(**metadata["schema"]),
                    rates_hz=artifact["rates_hz"],
                    visual_features=artifact["visual_features"],
                    visual_feature_names=metadata["visual_feature_names"],
                    targets=artifact["targets"],
                    session_ids=artifact["session_ids"].tolist(),
                    scene_ids=artifact["scene_ids"].tolist(),
                    timestamps_s=artifact["timestamps_s"],
                    metadata=metadata["metadata"],
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Incomplete or invalid motion dataset") from exc


def concatenate_datasets(datasets: Iterable[MotionDataset]) -> MotionDataset:
    """Join compatible artifacts, rejecting repeated sessions/duplicate files."""
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("At least one dataset is required")
    first = datasets[0]
    seen_sessions: set[str] = set()
    for dataset in datasets:
        if dataset.schema != first.schema or dataset.visual_feature_names != first.visual_feature_names:
            raise ValueError("Datasets must share the exact neural and visual feature schemas")
        sessions = set(dataset.session_ids)
        if seen_sessions & sessions:
            raise ValueError("Dataset session_ids overlap; do not include a session twice")
        seen_sessions.update(sessions)
    return MotionDataset(
        schema=first.schema,
        rates_hz=np.concatenate([d.rates_hz for d in datasets]),
        visual_features=np.concatenate([d.visual_features for d in datasets]),
        visual_feature_names=first.visual_feature_names,
        targets=np.concatenate([d.targets for d in datasets]),
        session_ids=tuple(s for d in datasets for s in d.session_ids),
        scene_ids=tuple(s for d in datasets for s in d.scene_ids),
        timestamps_s=np.concatenate([d.timestamps_s for d in datasets]),
        metadata={"concatenated_sources": [d.metadata for d in datasets]},
    )


def _visual_ridge_predict(train_x, train_y, test_x, ridge: float):
    """Same standardized ridge estimator as the neural baseline; signed input."""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (train_x - mean) / scale
    bias = train_y.mean(axis=0)
    y = train_y - bias
    if x.shape[1] <= x.shape[0]:
        weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    else:
        weights = x.T @ np.linalg.solve(x @ x.T + ridge * np.eye(x.shape[0]), y)
    prediction = ((test_x - mean) / scale) @ weights + bias
    if not np.isfinite(prediction).all():
        raise ValueError("Visual baseline produced non-finite motion targets")
    return prediction


def _metrics(targets, predictions) -> dict:
    errors = np.abs(targets - predictions).mean(axis=0)
    moving_yaw = np.abs(targets[:, 5]) >= 1e-6
    count = int(moving_yaw.sum())
    accuracy = None
    if count:
        accuracy = float(np.mean(np.sign(targets[moving_yaw, 5]) == np.sign(predictions[moving_yaw, 5])))
    return {"mae_by_axis": dict(zip(BODY_AXES, map(float, errors))),
            "yaw_sign_accuracy": accuracy, "yaw_sign_samples": count}


def _action_coverage(targets) -> dict:
    """Sample counts; combined forward/turn labels can count in both groups."""
    return {"forward": int((targets[:, 0] > 1e-6).sum()),
            "left": int((targets[:, 5] > 1e-6).sum()),
            "right": int((targets[:, 5] < -1e-6).sum()),
            "stop": int(np.all(np.abs(targets) < 1e-6, axis=1).sum())}


def evaluate_split(dataset: MotionDataset, test_scene_ids: tuple[str, ...],
                   ridge: float = 1.0) -> tuple[RidgeMotionDecoder, dict]:
    """Train on remaining scenes and evaluate once on explicit held-out scenes.

    No frame-random split, fitting on held-out data, or test-driven parameter
    search occurs here. Do not repeatedly tune against the same final holdout.
    A session spanning both sides is rejected even if its scene IDs differ.
    """
    test_scenes = set(_names(test_scene_ids, "test_scene_ids", unique=True))
    scenes = set(dataset.scene_ids)
    if not test_scenes <= scenes:
        raise ValueError(f"Unknown held-out scenes: {sorted(test_scenes - scenes)}")
    is_test = np.array([scene in test_scenes for scene in dataset.scene_ids])
    is_train = ~is_test
    if is_train.sum() < 2 or not is_test.any():
        raise ValueError("Need at least two training samples and one held-out sample")
    train_sessions = {s for s, keep in zip(dataset.session_ids, is_train) if keep}
    test_sessions = {s for s, keep in zip(dataset.session_ids, is_test) if keep}
    if train_sessions & test_sessions:
        raise ValueError("A recording session cannot span training and held-out scenes")
    train_y, test_y = dataset.targets[is_train], dataset.targets[is_test]
    model = RidgeMotionDecoder.fit(dataset.schema, dataset.rates_hz[is_train], train_y, ridge=ridge)
    predictions = {
        "neural": model.predict_batch(dataset.rates_hz[is_test], schema=dataset.schema),
        "visual_ridge": _visual_ridge_predict(dataset.visual_features[is_train], train_y,
                                             dataset.visual_features[is_test], float(ridge)),
        "mean_target": np.broadcast_to(train_y.mean(axis=0), test_y.shape),
        "zero_motion": np.zeros_like(test_y),
    }
    held_out_sessions = np.asarray(dataset.session_ids)[is_test]
    held_out_scenes = np.asarray(dataset.scene_ids)[is_test]
    per_session_metrics = {}
    for session in sorted(test_sessions):
        selected = held_out_sessions == session
        per_session_metrics[session] = {
            "samples": int(selected.sum()),
            "scene_ids": sorted(set(held_out_scenes[selected].tolist())),
            "action_coverage": _action_coverage(test_y[selected]),
            "metrics": {name: _metrics(test_y[selected], prediction[selected])
                        for name, prediction in predictions.items()},
        }
    warnings = ["Offline label prediction is not evidence of flight performance or cross-device transfer."]
    train_scenes = scenes - test_scenes
    if len(train_scenes) < 2:
        warnings.append("Only one training scene: scene diversity is insufficient for a generalization claim.")
    if len(test_scenes) < 2:
        warnings.append("Only one held-out scene: this is a pipeline check, not robust generalization evidence.")
    if not (np.ptp(train_y, axis=0) > 1e-6).any():
        warnings.append("All training targets are constant; no action discrimination can be learned.")
    report = {
        "format": "flydrones.motion_evaluation", "version": 1,
        "output": _output_spec(), "ridge": float(ridge),
        "train_samples": int(is_train.sum()), "test_samples": int(is_test.sum()),
        "train_scene_ids": sorted(train_scenes), "test_scene_ids": sorted(test_scenes),
        "train_session_ids": sorted(train_sessions), "test_session_ids": sorted(test_sessions),
        "feature_counts": {"neural": len(dataset.schema.feature_names),
                           "visual": len(dataset.visual_feature_names)},
        "action_coverage": {"train": _action_coverage(train_y), "test": _action_coverage(test_y)},
        "active_axes": [axis for axis, active in zip(BODY_AXES, np.any(np.abs(dataset.targets) >= 1e-6, axis=0)) if active],
        "varying_axes": [axis for axis, varying in zip(BODY_AXES, np.ptp(dataset.targets, axis=0) > 1e-6) if varying],
        "metrics": {name: _metrics(test_y, prediction) for name, prediction in predictions.items()},
        "per_session_metrics": per_session_metrics,
        "warnings": warnings,
    }
    return model, report
