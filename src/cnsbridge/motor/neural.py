"""Trainable neural readout, independent of aircraft or robot actuators.

This ridge model is a baseline, not a claim that a linear readout can solve
MaleCNS flight control. It learns from aligned neural rates and independently
provided target body velocities. It does not train the connectome itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .body import BODY_AXES, BODY_FRAME, BODY_UNITS, BodyVelocity


@dataclass(frozen=True)
class NeuralSchema:
    """Identity, order and rate window of a particular neural representation.

    source_id must identify the connectome version AND feature definitions,
    e.g. a graph/group-manifest digest. Matching type names alone is insufficient.
    rate_window_ms is the neural simulation window, not the camera frame period.
    """

    feature_names: tuple[str, ...]
    source_id: str
    rate_window_ms: float

    def __post_init__(self):
        if isinstance(self.feature_names, str):
            raise ValueError("feature_names must be a sequence of names")
        names = tuple(self.feature_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Neural features need nonempty names")
        if len(set(names)) != len(names):
            raise ValueError("Neural feature names must be unique")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("A neural source/feature-definition identity is required")
        window = float(self.rate_window_ms)
        if not np.isfinite(window) or window <= 0:
            raise ValueError("rate_window_ms must be positive and finite")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "rate_window_ms", window)


@dataclass(frozen=True)
class NeuralObservation:
    schema: NeuralSchema
    rates_hz: Mapping[str, float]

    def vector(self) -> np.ndarray:
        missing = [name for name in self.schema.feature_names if name not in self.rates_hz]
        if missing:
            raise ValueError(f"Missing neural features: {', '.join(missing)}")
        values = np.asarray([self.rates_hz[name] for name in self.schema.feature_names], dtype=np.float64)
        if values.shape != (len(self.schema.feature_names),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Neural firing rates must be finite nonnegative scalar Hz values")
        return values


class MotionDecoder(Protocol):
    """Other learned readouts can implement the same device-independent API."""

    schema: NeuralSchema

    def decode(self, observation: NeuralObservation) -> BodyVelocity: ...


class RidgeMotionDecoder:
    """Standardised neural rates -> six SI body-velocity targets.

    No cruise speed, escape action, actuator limits or missing-feature defaults
    are embedded here. Learned intercepts are training means, not resting rates.
    Device feasibility and safety belong to downstream adapters/controllers.
    """

    FORMAT = "flydrones.neural_motion.ridge"
    VERSION = 1

    def __init__(self, schema: NeuralSchema, feature_mean_hz, feature_scale_hz, weights, bias):
        self.schema = schema
        size = len(schema.feature_names)
        for name, value, shape in (
            ("feature_mean_hz", feature_mean_hz, (size,)),
            ("feature_scale_hz", feature_scale_hz, (size,)),
            ("weights", weights, (size, len(BODY_AXES))),
            ("bias", bias, (len(BODY_AXES),)),
        ):
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name}: expected finite array with shape {shape}")
            array.setflags(write=False)
            setattr(self, name, array)
        if (self.feature_mean_hz < 0).any() or (self.feature_scale_hz <= 0).any():
            raise ValueError("Feature means must be nonnegative and scales strictly positive")

    @staticmethod
    def _rates(rates_hz, size):
        values = np.asarray(rates_hz, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != size or values.shape[0] == 0:
            raise ValueError(f"Expected a nonempty N x {size} neural rate matrix")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Neural rates must be finite and nonnegative")
        return values

    @classmethod
    def fit(cls, schema: NeuralSchema, rates_hz, targets, ridge: float = 1.0):
        """Fit paired examples; hold-out splits must be made BEFORE calling this.

        targets is N x 6 in BODY_AXES order, metres/second and radians/second.
        It should contain desired actions from an independent teacher/task, not
        the old decoder's commands or unaligned measured aircraft velocities.
        """
        ridge = float(ridge)
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        x = cls._rates(rates_hz, len(schema.feature_names))
        y = np.asarray(targets, dtype=np.float64)
        if len(x) < 2 or y.shape != (len(x), len(BODY_AXES)) or not np.isfinite(y).all():
            raise ValueError("Training requires at least two aligned samples and finite N x 6 targets")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        xn = (x - mean) / scale
        bias = y.mean(axis=0)
        yc = y - bias
        # A per-neuron readout may have far more features than examples. Avoid
        # constructing a D x D matrix in that case; no full-brain dense inverse.
        if xn.shape[1] <= xn.shape[0]:
            system = xn.T @ xn + ridge * np.eye(xn.shape[1])
            weights = np.linalg.solve(system, xn.T @ yc)
        else:
            system = xn @ xn.T + ridge * np.eye(xn.shape[0])
            weights = xn.T @ np.linalg.solve(system, yc)
        return cls(schema, mean, scale, weights, bias)

    def predict_batch(self, rates_hz, *, schema: NeuralSchema) -> np.ndarray:
        if schema != self.schema:
            raise ValueError("Neural schema mismatch: source, feature order or rate window changed")
        values = self._rates(rates_hz, len(self.schema.feature_names))
        result = ((values - self.feature_mean_hz) / self.feature_scale_hz) @ self.weights + self.bias
        if not np.isfinite(result).all():
            raise ValueError("Decoder produced non-finite motion targets")
        return result

    def decode(self, observation: NeuralObservation) -> BodyVelocity:
        result = self.predict_batch(observation.vector()[None, :], schema=observation.schema)
        return BodyVelocity.from_array(result[0])

    @staticmethod
    def _output_spec():
        return {"axes": list(BODY_AXES), "frame": BODY_FRAME, "units": list(BODY_UNITS)}

    def save(self, path: str | Path) -> None:
        """Create an artifact without overwriting any existing trained model."""
        data = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "schema": {"feature_names": list(self.schema.feature_names), "source_id": self.schema.source_id,
                       "rate_window_ms": self.schema.rate_window_ms},
            "output": self._output_spec(),
            **{name: getattr(self, name).tolist() for name in
               ("feature_mean_hz", "feature_scale_hz", "weights", "bias")},
        }
        payload = json.dumps(data, indent=2, allow_nan=False)
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != cls.FORMAT or data.get("version") != cls.VERSION:
            raise ValueError("Unsupported neural motion model format/version")
        if data.get("output") != cls._output_spec():
            raise ValueError("Motion output order, coordinate frame or units do not match")
        try:
            schema = NeuralSchema(**data["schema"])
            return cls(schema, *(data[name] for name in ("feature_mean_hz", "feature_scale_hz", "weights", "bias")))
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete or invalid neural motion model") from exc
