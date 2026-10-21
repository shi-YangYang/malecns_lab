"""Parameters for stimulus-response experiments."""

from .stimulus import validate_parameters


def total_frames(parameters: dict) -> int:
    return 40 + round(parameters["duration_s"] * 20)
