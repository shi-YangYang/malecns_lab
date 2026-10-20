"""Device-independent target velocity in a right-handed body frame.

The FLU frame has x forward, y left and z up. Angular velocities follow the
right-hand rule; positive wz turns left. These are requested velocities, not
measurements or a guarantee that a device can execute them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


BODY_AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
BODY_FRAME = "body_flu"
BODY_UNITS = ("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s")


@dataclass(frozen=True)
class BodyVelocity:
    """Body-frame linear and angular target velocities, in SI units."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    def __post_init__(self) -> None:
        for axis in BODY_AXES:
            try:
                value = float(getattr(self, axis))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{axis} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{axis} must be a finite number")
            object.__setattr__(self, axis, value)

    def as_array(self) -> NDArray[np.float64]:
        """Return a new six-element array in BODY_AXES order."""
        return np.array([getattr(self, axis) for axis in BODY_AXES], dtype=np.float64)

    @classmethod
    def from_array(cls, vector: ArrayLike) -> BodyVelocity:
        """Build a target from an exact (6,) vector; do not flatten matrices."""
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(BODY_AXES),):
            raise ValueError(f"body velocity must have shape (6,), got {values.shape}")
        return cls(*values)
