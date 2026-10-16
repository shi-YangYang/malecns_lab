"""Compatibility exports; new code uses cnsbridge.control and cnsbridge.learning."""

from cnsbridge.control import (
    BodyVelocity, DifferentialDriveAdapter, FlightCommand, FlightCommandAdapter, MotorDecoder,
)
from cnsbridge.learning import MotionDecoder, NeuralObservation, NeuralSchema, RidgeMotionDecoder

__all__ = [
    "FlightCommand",
    "MotorDecoder",
    "BodyVelocity",
    "FlightCommandAdapter",
    "DifferentialDriveAdapter",
    "MotionDecoder",
    "NeuralObservation",
    "NeuralSchema",
    "RidgeMotionDecoder",
]
