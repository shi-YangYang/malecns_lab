"""Compatibility imports; use :mod:`cnsbridge.learning.dataset` for new code."""

from cnsbridge.learning.dataset import (
    _names,
    _output_spec,
    MotionDataset,
    concatenate_datasets,
    _visual_ridge_predict,
    _metrics,
    _action_coverage,
    evaluate_split,
)

__all__ = [
    "MotionDataset",
    "concatenate_datasets",
    "evaluate_split",
]
