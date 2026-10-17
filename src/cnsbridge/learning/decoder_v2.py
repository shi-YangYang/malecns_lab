"""Stage A masked linear readout, with explicit unsupported axes and validity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..control.body import BODY_AXES, BodyVelocity
from .contracts_v2 import OUTPUT_SPEC, FeatureSchemaV2, write_json


@dataclass(frozen=True)
class MotionPredictionV2:
    target: BodyVelocity
    supported_axes: tuple[str, ...]
    prediction_valid: bool


class MaskedRidgeV2:
    FORMAT = "cnsbridge.masked_motion_ridge.v2"

    def __init__(self, schema: FeatureSchemaV2, mean, scale, weights, bias, supported, *, ridge: float):
        self.schema, self.ridge = schema, float(ridge)
        size = len(schema.feature_names)
        for name, value, shape in (("mean", mean, (size,)), ("scale", scale, (size,)),
                                   ("weights", weights, (size, 6)), ("bias", bias, (6,))):
            array = np.array(value, dtype=float, copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid model {name}")
            array.setflags(write=False)
            setattr(self, name, array)
        self.supported = np.array(supported, copy=True)
        if self.supported.dtype != np.bool_ or self.supported.shape != (6,) or self.supported[[1, 2, 3, 4]].any():
            raise ValueError("Supported mask must identify supervised vx/wz axes only")
        if np.any(self.scale <= 0) or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("Positive finite feature scales and ridge required")
        if np.any(self.weights[:, ~self.supported]) or np.any(self.bias[~self.supported]):
            raise ValueError("Unsupported axes must have zero placeholders, not learned parameters")
        self.supported.setflags(write=False)

    @classmethod
    def fit(cls, schema: FeatureSchemaV2, features, targets, mask, *, ridge: float):
        x, y, mask = np.asarray(features, float), np.asarray(targets, float), np.asarray(mask)
        if x.ndim != 2 or x.shape[1] != len(schema.feature_names) or y.shape != (len(x), 6):
            raise ValueError("Aligned N x features and N x 6 targets required")
        if mask.dtype != np.bool_ or mask.shape != y.shape or mask[:, [1, 2, 3, 4]].any():
            raise ValueError("Six-axis boolean mask required; only vx/wz may be supervised")
        selected = mask.any(axis=1)
        x, y, mask = x[selected], y[selected], mask[selected]
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
            raise ValueError("At least two finite training observations with independent labels required")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("Positive finite ridge required")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        xn = (x - mean) / scale
        weights, bias, supported = np.zeros((x.shape[1], 6)), np.zeros(6), np.zeros(6, dtype=bool)
        for axis in (0, 5):
            use = mask[:, axis]
            if not use.any():
                continue
            if use.sum() < 2:
                raise ValueError(f"At least two supervised examples required for {BODY_AXES[axis]}")
            xa, ya = xn[use], y[use, axis]
            center, ymean = xa.mean(axis=0), ya.mean()
            xc, yc = xa - center, ya - ymean
            if xc.shape[1] <= len(xc):
                weights[:, axis] = np.linalg.solve(xc.T @ xc + ridge * np.eye(xc.shape[1]), xc.T @ yc)
            else:
                weights[:, axis] = xc.T @ np.linalg.solve(xc @ xc.T + ridge * np.eye(len(xc)), yc)
            bias[axis] = ymean - center @ weights[:, axis]
            supported[axis] = True
        return cls(schema, mean, scale, weights, bias, supported, ridge=ridge)

    def predict_batch(self, features, *, schema: FeatureSchemaV2, prediction_valid=None) -> np.ndarray:
        if schema != self.schema:
            raise ValueError("Feature schema/source/order mismatch")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(schema.feature_names):
            raise ValueError("Wrong feature matrix shape")
        valid = np.ones(len(x), dtype=bool) if prediction_valid is None else np.asarray(prediction_valid)
        if valid.dtype != np.bool_ or valid.shape != (len(x),):
            raise ValueError("prediction_valid must be an N-vector of booleans")
        if not np.isfinite(x[valid]).all():
            raise ValueError("Valid prediction features are not finite")
        result = np.full((len(x), 6), np.nan)
        result[valid] = ((x[valid] - self.mean) / self.scale) @ self.weights + self.bias
        if not np.isfinite(result[valid]).all():
            raise ValueError("Nonfinite decoder output")
        return result

    def decode(self, features, *, schema: FeatureSchemaV2, prediction_valid: bool) -> MotionPredictionV2:
        if not isinstance(prediction_valid, (bool, np.bool_)):
            raise ValueError("Explicit boolean prediction validity required")
        values = self.predict_batch(np.asarray(features)[None, :], schema=schema, prediction_valid=np.array([prediction_valid]))
        return MotionPredictionV2(BodyVelocity.from_array(values[0]) if prediction_valid else BodyVelocity(),
                                  tuple(axis for axis, enabled in zip(BODY_AXES, self.supported) if enabled), bool(prediction_valid))

    def save(self, path: str | Path):
        write_json(Path(path), {"format": self.FORMAT, "schema": self.schema.as_dict(), "output": OUTPUT_SPEC,
                               "ridge": self.ridge, **{name: getattr(self, name).tolist() for name in
                               ("mean", "scale", "weights", "bias", "supported")}})

    @classmethod
    def load(cls, path: str | Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != cls.FORMAT or value.get("output") != OUTPUT_SPEC:
            raise ValueError("Not a compatible V2 masked motion model")
        return cls(FeatureSchemaV2(**value["schema"]), *(value[name] for name in ("mean", "scale", "weights", "bias", "supported")),
                   ridge=value["ridge"])
