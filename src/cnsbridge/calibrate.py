"""Compatibility imports; use :mod:`cnsbridge.control.calibration` for new code."""

from cnsbridge.control.calibration import (
    STIMULI,
    _frame,
    record_responses,
    fit_readout,
    calibrate,
)

__all__ = [
    "STIMULI",
    "record_responses",
    "fit_readout",
    "calibrate",
]
