"""Compatibility imports; use :mod:`cnsbridge.control.adapters` for new code."""

from cnsbridge.control.adapters import (
    CommandT,
    _UNSUPPORTED_TOLERANCE,
    _finite,
    _positive,
    _reject_unsupported,
    Adaptation,
    FlightCommandAdapter,
    WheelSpeeds,
    DifferentialDriveAdapter,
)

__all__ = [
    "CommandT",
    "Adaptation",
    "FlightCommandAdapter",
    "WheelSpeeds",
    "DifferentialDriveAdapter",
]
