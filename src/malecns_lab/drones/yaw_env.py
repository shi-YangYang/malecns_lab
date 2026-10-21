"""Fixed single-axis yaw plant and scoring; privileged state is not a policy input."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from ..control.body import BodyVelocity

DT_S = 0.05
RESPONSE_TAU_S = 0.2
PIXELS_PER_RAD = 60.0
TEXTURE_SEED = 7
SETTLING_THRESHOLD_RAD_S = 0.03
SETTLING_SUPPORT_S = 0.5


def _finite_scalar(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


class YawEnvironment:
    """FLU-positive left yaw with exact first-order response integration.

    A fixed world texture is sampled at x_pixel - 60 * yaw. Positive yaw
    therefore moves visible features right. The texture has no clock or motion
    independent of the body. This is a near-axis cylindrical camera model,
    not calibrated animal or robot dynamics.
    """

    dt_s = DT_S
    response_tau_s = RESPONSE_TAU_S
    pixels_per_rad = PIXELS_PER_RAD
    texture_seed = TEXTURE_SEED

    def __init__(self) -> None:
        self._yaw = 0.0
        self._actuator_wz = 0.0
        self._disturbance_wz = 0.0
        self._phase = np.random.default_rng(TEXTURE_SEED).uniform(-np.pi, np.pi, size=4)

    @property
    def yaw_rad(self) -> float:
        return self._yaw

    def frame(self) -> np.ndarray:
        """Render current pose only; calling this method never advances the plant."""
        y, x = np.mgrid[:72, :96].astype(np.float64)
        x -= PIXELS_PER_RAD * self._yaw
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite yaw projection")
        # Same deterministic two-dimensional texture as GeneratedSourceV2.
        gray = 127 + 34 * np.sin(x * 0.49 + self._phase[0]) + 31 * np.cos(y * 0.43 + self._phase[1])
        gray += 25 * np.sin(0.21 * x + 0.31 * y + self._phase[2]) + 19 * np.cos(0.37 * x - 0.27 * y + self._phase[3])
        return np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None], 3, axis=2)

    def step(self, action: BodyVelocity, disturbance_wz_rad_s: float) -> dict[str, float]:
        """Respond independently to held command and disturbance-rate targets.

        Each channel has tau=0.2 s. The logged actuator is the actuator's end
        state only; actual rate is the sum of both interval averages. Removing
        the disturbance target therefore leaves a physical decay tail.
        """
        if not isinstance(action, BodyVelocity):
            raise TypeError("Yaw environment requires BodyVelocity")
        if not np.isfinite(action.as_array()).all():
            raise ValueError("Action must be finite")
        if any(getattr(action, axis) != 0.0 for axis in ("vx", "vy", "vz", "wx", "wy")):
            raise ValueError("Yaw environment supports only wz")
        disturbance = _finite_scalar(disturbance_wz_rad_s, "disturbance_wz_rad_s")
        response = -math.expm1(-DT_S / RESPONSE_TAU_S)
        old_weight = RESPONSE_TAU_S * response / DT_S
        average_actuator = self._actuator_wz * old_weight + action.wz * (1.0 - old_weight)
        actuator_end = self._actuator_wz * (1.0 - response) + action.wz * response
        average_disturbance = self._disturbance_wz * old_weight + disturbance * (1.0 - old_weight)
        disturbance_end = self._disturbance_wz * (1.0 - response) + disturbance * response
        actual = average_actuator + average_disturbance
        yaw_after = self._yaw + DT_S * actual
        row = {"yaw_rad": self._yaw, "yaw_after_rad": yaw_after,
               "actuator_wz_rad_s": actuator_end, "actual_wz_rad_s": actual}
        if not all(math.isfinite(value) for value in (*row.values(), disturbance_end)):
            raise FloatingPointError("Nonfinite yaw dynamics; state was not advanced")
        self._yaw, self._actuator_wz, self._disturbance_wz = yaw_after, actuator_end, disturbance_end
        return row


def episode_metrics(actual_wz_rad_s, warmup_frames: int = 40, dt_s: float = DT_S,
                    disturbance_duration_s: float = 0.5) -> dict[str, float | None]:
    """Score interval-average rates after warmup, with no heading-return objective.

    Settling time is delay from disturbance end to the first interval whose
    entire remaining suffix meets the threshold and provides at least 0.5 s
    of support. A short or never-settled record reports None.
    """
    values = np.asarray(actual_wz_rad_s, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite one-dimensional angular-rate record")
    if isinstance(warmup_frames, (bool, np.bool_)) or not isinstance(warmup_frames, Integral) or warmup_frames < 0:
        raise ValueError("warmup_frames must be a nonnegative integer")
    dt = _finite_scalar(dt_s, "dt_s")
    disturbance_duration = _finite_scalar(disturbance_duration_s, "disturbance_duration_s")
    if dt <= 0 or disturbance_duration < 0:
        raise ValueError("dt_s must be positive and disturbance_duration_s nonnegative")
    observed = values[warmup_frames:]
    if not observed.size:
        raise ValueError("At least one observation interval is required")
    absolute = np.abs(observed)
    peak = float(absolute.max())
    rms = peak * float(np.sqrt(np.mean((observed / peak) ** 2))) if peak else 0.0
    integral = float(np.sum(absolute * dt))
    if not math.isfinite(integral):
        raise FloatingPointError("Nonfinite angular-rate integral")
    stable_suffix = np.logical_and.accumulate((absolute <= SETTLING_THRESHOLD_RAD_S)[::-1])[::-1]
    settling = None
    for index in range(len(observed)):
        start = index * dt
        if start + 1e-12 < disturbance_duration:
            continue
        if (len(observed) - index) * dt + 1e-12 < SETTLING_SUPPORT_S:
            break
        if stable_suffix[index]:
            settling = max(0.0, start - disturbance_duration)
            break
    return {"rms_rotation_rad_s": rms, "peak_rotation_rad_s": peak,
            "integral_abs_rotation_rad": integral, "settling_time_s": settling}
