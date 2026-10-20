"""Pixel-only marked-target observations; no simulator or controller dependency."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label, find_objects


@dataclass(frozen=True)
class TargetCamera:
    width: int = 160
    height: int = 120
    horizontal_fov_deg: float = 72.0
    marker_size_m: float = 0.2

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 16 for v in (self.width, self.height)):
            raise ValueError("camera dimensions must be integers >= 16")
        if not math.isfinite(self.horizontal_fov_deg) or not 1 < self.horizontal_fov_deg < 170:
            raise ValueError("horizontal FOV must be finite and between 1 and 170 degrees")
        if not math.isfinite(self.marker_size_m) or self.marker_size_m <= 0:
            raise ValueError("marker size must be finite and positive")

    @property
    def focal_px(self) -> float:
        """Square pixels, with image coordinates measured at pixel centers."""
        return self.width / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))

    @property
    def center(self) -> tuple[float, float]:
        return (self.width - 1) / 2, (self.height - 1) / 2


@dataclass(frozen=True)
class TargetObservation:
    detected: bool
    valid: bool
    reason: str
    # Normalized by image half-width/height. Positive x is image-right; y is up.
    x_error: float | None = None
    y_error: float | None = None
    apparent_height: float | None = None  # fraction of full image height
    distance_m: float | None = None  # known-size approximate Euclidean range
    coverage: float = 0.0  # red occupancy inside the bounding box
    confidence: float = 0.0  # engineering score, not a probability calibration
    bbox: tuple[int, int, int, int] | None = None  # left, top, right, bottom; exclusive end


def _marker_pixels(frame: object, camera: TargetCamera) -> tuple[NDArray[np.bool_] | None, str]:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        return None, "invalid_bgr_uint8"
    if frame.shape != (camera.height, camera.width, 3):
        return None, "invalid_image_shape"
    blue, green, red = frame.astype(np.int16).transpose(2, 0, 1)
    return (red >= 70) & (red > 1.65 * green) & (red > 1.65 * blue) & (red - green >= 35), "ok"


def detect_target(frame: object, camera: TargetCamera = TargetCamera()) -> TargetObservation:
    """Detect the largest connected red square from pixels only.

    A silhouette touching the image boundary, visibly incomplete, tiny or
    ambiguous is invalid. No estimated position/range is emitted on invalid
    input. Occlusions that preserve a plausible square cannot in general be
    disambiguated by this simple detector; this is not a general object tracker.
    """
    mask, reason = _marker_pixels(frame, camera)
    if mask is None:
        return TargetObservation(False, False, reason)
    components, count = label(mask)
    if count == 0:
        return TargetObservation(False, False, "not_detected")
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    sizes[0] = 0
    largest = int(sizes.argmax())
    slices = find_objects(components)[largest - 1]
    assert slices is not None
    yy, xx = slices
    box = (xx.start, yy.start, xx.stop, yy.stop)
    width, height = xx.stop - xx.start, yy.stop - yy.start
    coverage = float(sizes[largest] / (width * height))
    invalid = None
    if width < 4 or height < 4:
        invalid = "too_small"
    elif xx.start <= 0 or yy.start <= 0 or xx.stop >= camera.width or yy.stop >= camera.height:
        invalid = "edge_clipped"
    elif count > 1 and np.sum(sizes >= 0.5 * sizes[largest]) > 1:
        invalid = "ambiguous_markers"
    elif not 0.6 <= width / height <= 1.35 or coverage < 0.85:
        invalid = "incomplete_marker"
    if invalid:
        return TargetObservation(True, False, invalid, coverage=coverage, bbox=box)
    cx, cy = camera.center
    u, v = (xx.start + xx.stop - 1) / 2, (yy.start + yy.stop - 1) / 2
    depth = camera.focal_px * camera.marker_size_m / height
    distance = depth * math.sqrt(1 + ((u - cx) / camera.focal_px) ** 2 + ((v - cy) / camera.focal_px) ** 2)
    return TargetObservation(
        True, True, "ok", (u - cx) / (camera.width / 2), (cy - v) / (camera.height / 2),
        height / camera.height, distance, coverage, coverage, box,
    )


def target_mask(
    frame: object, camera: TargetCamera = TargetCamera(), shape: tuple[int, int] = (12, 16),
) -> NDArray[np.float32]:
    """Return image-aligned occupancy cells (row top-down, column left-right).

    Invalid frames raise instead of looking like valid blank observations.
    A mask alone does not establish detection validity; use detect_target too.
    """
    mask, reason = _marker_pixels(frame, camera)
    if mask is None:
        raise ValueError(reason)
    if len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape):
        raise ValueError("mask shape must contain two positive integers")
    if shape[0] > camera.height or shape[1] > camera.width:
        raise ValueError("mask shape cannot exceed image dimensions")
    rows = np.linspace(0, camera.height, shape[0] + 1, dtype=int)
    cols = np.linspace(0, camera.width, shape[1] + 1, dtype=int)
    result = np.empty(shape, dtype=np.float32)
    for row in range(shape[0]):
        for col in range(shape[1]):
            result[row, col] = mask[rows[row]:rows[row + 1], cols[col]:cols[col + 1]].mean()
    result.setflags(write=False)
    return result
