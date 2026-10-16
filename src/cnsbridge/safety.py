"""Compatibility imports; use :mod:`cnsbridge.control.safety` for new code."""

from cnsbridge.control.safety import (
    Telemetry,
    SafetyGovernor,
)

__all__ = [
    "Telemetry",
    "SafetyGovernor",
]
