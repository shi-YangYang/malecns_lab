"""Compatibility imports; use :mod:`cnsbridge.learning.decoder` for new code."""

from cnsbridge.learning.decoder import (
    NeuralSchema,
    NeuralObservation,
    MotionDecoder,
    RidgeMotionDecoder,
)

__all__ = [
    "NeuralSchema",
    "NeuralObservation",
    "MotionDecoder",
    "RidgeMotionDecoder",
]
