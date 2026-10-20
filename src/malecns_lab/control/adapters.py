"""Pure conversions from body velocity targets to device commands.

Adapters report limits and reject unsupported motion. They perform no I/O,
state estimation, obstacle avoidance, arming or safety supervision. A caller
must provide those separately before any command is sent to hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Generic, TypeVar

from .body import BodyVelocity
from .command import FlightCommand


CommandT = TypeVar("CommandT")
_UNSUPPORTED_TOLERANCE = 1e-9


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _reject_unsupported(target: BodyVelocity, axes: tuple[str, ...]) -> None:
    unsupported = tuple(axis for axis in axes if abs(getattr(target, axis)) > _UNSUPPORTED_TOLERANCE)
    if unsupported:
        raise ValueError(f"unsupported body motion: {', '.join(unsupported)}")


@dataclass(frozen=True)
class Adaptation(Generic[CommandT]):
    """Converted command and the body target remaining after device limits.

    limited_target is a command-space target, not measured physical motion.
    saturated_axes uses body-axis names, independent of device conventions.
    """

    command: CommandT
    limited_target: BodyVelocity
    saturated_axes: tuple[str, ...]


@dataclass(frozen=True)
class FlightCommandAdapter:
    """Map SI body velocity to the existing normalized flight command.

    Each scale is the empirically calibrated speed at normalized magnitude
    one. RC inputs do not guarantee precise velocity. In particular, a Tello
    RC backend's scaling must match the setting used for this calibration.
    Pitch and roll angular-rate targets cannot be expressed by FlightCommand.
    """

    forward_scale_mps: float
    lateral_scale_mps: float
    vertical_scale_mps: float
    yaw_scale_rad_s: float

    def __post_init__(self) -> None:
        for name in ("forward_scale_mps", "lateral_scale_mps", "vertical_scale_mps", "yaw_scale_rad_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def adapt(self, target: BodyVelocity) -> Adaptation[FlightCommand]:
        _reject_unsupported(target, ("wx", "wy"))
        scales = {
            "vx": self.forward_scale_mps,
            "vy": self.lateral_scale_mps,
            "vz": self.vertical_scale_mps,
            "wz": self.yaw_scale_rad_s,
        }
        limited = {axis: max(-scale, min(scale, getattr(target, axis))) for axis, scale in scales.items()}
        saturated = tuple(axis for axis, scale in scales.items() if abs(getattr(target, axis)) > scale)
        command = FlightCommand(
            forward=limited["vx"] / self.forward_scale_mps,
            lateral=-limited["vy"] / self.lateral_scale_mps,
            throttle=limited["vz"] / self.vertical_scale_mps,
            yaw=-limited["wz"] / self.yaw_scale_rad_s,
        )
        return Adaptation(command, BodyVelocity(**limited), saturated)


@dataclass(frozen=True)
class WheelSpeeds:
    """Wheel angular-speed targets; positive values drive each wheel forward."""

    left_rad_s: float
    right_rad_s: float

    def __post_init__(self) -> None:
        for name in ("left_rad_s", "right_rad_s"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class DifferentialDriveAdapter:
    """Ideal differential-drive kinematics with curvature-preserving limits.

    The physical controller must track these wheel-speed targets. Wheel slip,
    acceleration limits and physical safety constraints are outside this
    conversion. track_width_m is the distance between the wheels' contact
    centerlines, and max_wheel_rad_s is the shared absolute wheel-speed limit.
    """

    wheel_radius_m: float
    track_width_m: float
    max_wheel_rad_s: float

    def __post_init__(self) -> None:
        for name in ("wheel_radius_m", "track_width_m", "max_wheel_rad_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def adapt(self, target: BodyVelocity) -> Adaptation[WheelSpeeds]:
        _reject_unsupported(target, ("vy", "vz", "wx", "wy"))
        turn_speed = target.wz * (self.track_width_m / 2.0)
        left = (target.vx - turn_speed) / self.wheel_radius_m
        right = (target.vx + turn_speed) / self.wheel_radius_m
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("target and geometry produce non-finite wheel speeds")

        peak = max(abs(left), abs(right))
        scale = self.max_wheel_rad_s / peak if peak > self.max_wheel_rad_s else 1.0
        command = WheelSpeeds(left * scale, right * scale)
        limited = BodyVelocity(vx=target.vx * scale, wz=target.wz * scale)
        saturated = tuple(axis for axis in ("vx", "wz") if scale < 1.0 and getattr(target, axis) != 0.0)
        return Adaptation(command, limited, saturated)
