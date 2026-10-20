"""Causal V2 visual features, independent of a brain or a device.

The two image halves are engineering fields of view, not anatomical eyes.
All motion uses 20 Hz playback timestamps and resized-image pixels/second.
Quality thresholds are explicit engineering candidates, not biological values
or parameters shown to be optimal by an experiment. No ridge term, flow clip,
or texture multiplier is used to make an unidentifiable flow look valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from numbers import Real

import numpy as np

FEATURES = ("brightness", "ftb", "btf", "up", "down", "loom", "loom_speed")
CHANNELS = tuple(f"{eye}.{feature}" for eye in ("L", "R") for feature in FEATURES)
WIDTH, HEIGHT = 96, 72
ROWS, COLS = 6, 8
FRAME_INTERVAL_S = 0.05


@dataclass(frozen=True)
class VisionV2Config:
    """Untuned quality candidates; identities must include these values."""

    min_texture_eigenvalue: float = 1e-6
    max_texture_condition: float = 1000.0
    time_tolerance_s: float = 1e-9

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_texture_condition < 1:
            raise ValueError("max_texture_condition must be at least 1")
        if self.time_tolerance_s >= FRAME_INTERVAL_S / 2:
            raise ValueError("time_tolerance_s must not hide a missing playback frame")

    def identity(self) -> dict:
        """Configuration identity; the integrating model also hashes source code."""
        return {
            "protocol": "cnsbridge.vision.v2",
            "channels": list(CHANNELS),
            "color": "BGR_uint8",
            "gray_weights": [0.114, 0.587, 0.299],
            "resize": "separable_exact_area_mean",
            "width": WIDTH,
            "height": HEIGHT,
            "eye_grid": [ROWS, COLS],
            "timebase": "video_playback",
            "frame_interval_s": FRAME_INTERVAL_S,
            "flow": "local_lucas_kanade_mean_spatial_gradient_no_regularization",
            "flow_units": "resized_pixel/s",
            "quality_parameters": asdict(self),
            "quality_parameter_evidence": "engineering_candidate_not_optimized",
            "direction_scale_px_s": 24.0,
            "expansion_fit": "unweighted_valid_cells_independent_axis_with_intercept",
            "expansion_ema_tau_s": 0.1,
            "loom_offset_s_inv": 0.1,
            "loom_scale_s_inv": 0.5,
            "loom_speed_scale_s_inv": 5.0,
            "initialization": "real_brightness_zero_motion_and_ema_warmup_only",
            "failure": "no_features_no_state_commit_stop_segment_until_reset",
        }


@dataclass(frozen=True)
class EyeMotion:
    """Per-cell diagnostics; invalid/unevaluated flow is NaN, never zero-filled.

    Texture is the trace of the mean gradient outer product, in normalized
    luminance squared / pixel squared. It is diagnostic only, not a flow gain.
    Centers are actual resized pixel-center coordinates within this eye.
    """

    brightness_grid: np.ndarray
    centers_px: np.ndarray
    flow_px_s: np.ndarray
    texture: np.ndarray
    min_eigenvalue: np.ndarray
    condition_number: np.ndarray
    flow_valid: np.ndarray
    expansion_x_s_inv: float | None = None
    expansion_y_s_inv: float | None = None
    expansion_s_inv: float | None = None
    expansion_ema_s_inv: float | None = None
    loom: float | None = None
    loom_speed_s_inv: float | None = None

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.flow_valid))

    @property
    def valid_fraction(self) -> float:
        return self.valid_count / (ROWS * COLS)


@dataclass(frozen=True)
class VisionV2Observation:
    """A valid first frame is initialization only, never prediction evidence.

    ``valid`` means features may drive the next processing stage, including
    its warmup. It does not certify neural history or the required 2 s warmup.
    The replay layer owns that readiness. Invalid observations have no features.
    """

    input_cutoff_s: float | None
    features: np.ndarray | None
    eyes: dict[str, EyeMotion]
    valid: bool
    initialization: bool
    segment_stopped: bool
    invalid_reason: str | None = None
    invalid_eye: str | None = None
    segment_stop_reason: str | None = None


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _area_axis(image: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Integrate piecewise-constant pixels, including fractional bin edges."""
    data = np.moveaxis(image, axis, -1)
    edges = np.linspace(0, data.shape[-1], size + 1)
    left = np.floor(edges).astype(np.intp)
    fraction = edges - left
    integral = np.concatenate((np.zeros((*data.shape[:-1], 1)), np.cumsum(data, axis=-1)), axis=-1)
    at_edges = integral[..., left] + data[..., np.minimum(left, data.shape[-1] - 1)] * fraction
    return np.moveaxis(np.diff(at_edges, axis=-1) / np.diff(edges), -1, axis)


def _image_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an H x W x 3 uint8 BGR array")
    if image.shape[0] < HEIGHT or image.shape[1] < WIDTH:
        raise ValueError(f"image must be at least {WIDTH} x {HEIGHT}; upsampling is not in this protocol")
    gray = image.astype(np.float64) @ np.array([0.114, 0.587, 0.299]) / 255.0
    return _area_axis(_area_axis(gray, WIDTH, 1), HEIGHT, 0)


def _cells(image: np.ndarray) -> np.ndarray:
    return image.reshape(ROWS, HEIGHT // ROWS, COLS, (WIDTH // 2) // COLS).transpose(0, 2, 1, 3).reshape(ROWS, COLS, -1)


def _centers() -> np.ndarray:
    x = (np.arange(COLS) + 0.5) * (WIDTH // 2) / COLS - 0.5
    y = (np.arange(ROWS) + 0.5) * HEIGHT / ROWS - 0.5
    xx, yy = np.meshgrid(x, y)
    return _readonly(np.stack((xx, yy), axis=-1))


def _eye_flow(current: np.ndarray, previous: np.ndarray | None, dt: float, config: VisionV2Config) -> EyeMotion:
    shape = (ROWS, COLS)
    brightness = _readonly(_cells(current).mean(axis=-1))
    if previous is None:
        return EyeMotion(
            brightness, _centers(), _readonly(np.full((*shape, 2), np.nan)),
            _readonly(np.full(shape, np.nan)), _readonly(np.full(shape, np.nan)),
            _readonly(np.full(shape, np.nan)), _readonly(np.zeros(shape, dtype=bool)),
            expansion_x_s_inv=0.0, expansion_y_s_inv=0.0, expansion_s_inv=0.0,
            expansion_ema_s_inv=0.0, loom=0.0, loom_speed_s_inv=0.0,
        )
    gy, gx = np.gradient((current + previous) * 0.5)
    ix, iy, it = (_cells(value) for value in (gx, gy, current - previous))
    sxx, syy, sxy = (ix * ix).mean(-1), (iy * iy).mean(-1), (ix * iy).mean(-1)
    sxt, syt = (ix * it).mean(-1), (iy * it).mean(-1)
    tensor = np.empty((*shape, 2, 2), dtype=np.float64)
    tensor[..., 0, 0], tensor[..., 1, 1] = sxx, syy
    tensor[..., 0, 1] = tensor[..., 1, 0] = sxy
    eigenvalues = np.linalg.eigvalsh(tensor)
    smallest, largest = eigenvalues[..., 0], eigenvalues[..., 1]
    condition = np.full(shape, np.inf)
    np.divide(largest, smallest, out=condition, where=smallest > 0)
    valid = (smallest >= config.min_texture_eigenvalue) & (condition <= config.max_texture_condition)
    determinant = sxx * syy - sxy * sxy
    valid &= determinant > 0
    flow = np.full((*shape, 2), np.nan)
    flow[..., 0][valid] = ((-syy * sxt + sxy * syt)[valid] / determinant[valid]) / dt
    flow[..., 1][valid] = ((sxy * sxt - sxx * syt)[valid] / determinant[valid]) / dt
    valid &= np.isfinite(flow).all(axis=-1)
    flow[~valid] = np.nan
    return EyeMotion(brightness, _centers(), _readonly(flow), _readonly(sxx + syy),
                     _readonly(smallest), _readonly(condition), _readonly(valid))


class _MotionFailure(Exception):
    pass


def _fit_expansion(eye: EyeMotion) -> tuple[float, float]:
    if not eye.valid_count:
        raise _MotionFailure("no_valid_flow")
    centers, flow = eye.centers_px[eye.flow_valid], eye.flow_px_s[eye.flow_valid]
    if not np.isfinite(flow).all() or not np.isfinite(centers).all():
        raise _MotionFailure("nonfinite_expansion")
    slopes = []
    for axis in (0, 1):
        design = np.column_stack((np.ones(len(centers)), centers[:, axis]))
        if np.linalg.matrix_rank(design) != 2:
            raise _MotionFailure("unidentifiable_expansion")
        try:
            slope = float(np.linalg.lstsq(design, flow[:, axis], rcond=None)[0][1])
        except np.linalg.LinAlgError as exc:
            raise _MotionFailure("unidentifiable_expansion") from exc
        if not np.isfinite(slope):
            raise _MotionFailure("nonfinite_expansion")
        slopes.append(slope)
    return slopes[0], slopes[1]


class VisionV2Encoder:
    """Transactional fixed-20-Hz visual encoder; call reset for a new segment.

    Images smaller than the target, wrong color shape/dtype, non-finite or
    non-increasing times, and gaps all stop the segment. A failed frame never
    commits the previous image, timestamp, expansion EMA or previous loom.
    It also prevents processing later frames until an explicit reset.
    """

    def __init__(self, config: VisionV2Config | None = None):
        self.config = config or VisionV2Config()
        self.reset()

    def reset(self) -> None:
        self._previous_image: np.ndarray | None = None
        self._previous_time_s: float | None = None
        self._expansion_ema = {eye: 0.0 for eye in ("L", "R")}
        self._previous_loom = {eye: 0.0 for eye in ("L", "R")}
        self._stop_reason: str | None = None

    @property
    def last_committed_time_s(self) -> float | None:
        return self._previous_time_s

    @property
    def stopped(self) -> bool:
        return self._stop_reason is not None

    def _fail(self, time_s: float | None, reason: str, eyes: dict[str, EyeMotion] | None = None,
              eye: str | None = None) -> VisionV2Observation:
        self._stop_reason = self._stop_reason or reason
        return VisionV2Observation(time_s, None, eyes or {}, False, False, True, reason, eye, self._stop_reason)

    def encode(self, image_bgr: np.ndarray, video_time_s: float) -> VisionV2Observation:
        time_s = None
        if isinstance(video_time_s, Real) and not isinstance(video_time_s, bool) and np.isfinite(video_time_s):
            time_s = float(video_time_s)
        if self.stopped:
            return self._fail(time_s, "segment_stopped")
        if time_s is None or time_s < 0:
            return self._fail(time_s, "invalid_time")
        dt = FRAME_INTERVAL_S if self._previous_time_s is None else time_s - self._previous_time_s
        if dt <= 0:
            return self._fail(time_s, "nonmonotonic_time")
        if abs(dt - FRAME_INTERVAL_S) > self.config.time_tolerance_s:
            return self._fail(time_s, "time_step_mismatch")
        try:
            gray = _image_gray(image_bgr)
        except ValueError:
            return self._fail(time_s, "invalid_image")
        initial = self._previous_image is None
        eyes = {}
        for side, column in (("L", slice(0, WIDTH // 2)), ("R", slice(WIDTH // 2, WIDTH))):
            previous = None if initial else self._previous_image[:, column]
            eyes[side] = _eye_flow(gray[:, column], previous, dt, self.config)

        # Compute both halves before committing *any* time-dependent state.
        slopes = {}
        if not initial:
            for side, eye in eyes.items():
                try:
                    slopes[side] = _fit_expansion(eye)
                except _MotionFailure as exc:
                    return self._fail(time_s, str(exc), eyes, side)

        values = []
        next_ema, next_loom = {}, {}
        for side, eye in eyes.items():
            brightness = float(eye.brightness_grid.mean())
            if initial:
                values.extend((brightness, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                next_ema[side] = next_loom[side] = 0.0
                continue
            ex, ey = slopes[side]
            expansion = min(ex, ey)
            ema = self._expansion_ema[side] + (-np.expm1(-dt / 0.1)) * (expansion - self._expansion_ema[side])
            loom = float(np.clip((ema - 0.1) / 0.5, 0.0, 1.0))
            loom_speed = max(0.0, (loom - self._previous_loom[side]) / dt)
            flow = eye.flow_px_s[eye.flow_valid]
            forward_sign = -1.0 if side == "L" else 1.0
            directions = (forward_sign * flow[:, 0], -forward_sign * flow[:, 0], -flow[:, 1], flow[:, 1])
            values.extend((brightness, *(float(np.clip(direction / 24.0, 0.0, 1.0).mean()) for direction in directions),
                           loom, float(np.clip(loom_speed / 5.0, 0.0, 1.0))))
            eyes[side] = replace(eye, expansion_x_s_inv=ex, expansion_y_s_inv=ey, expansion_s_inv=expansion,
                                 expansion_ema_s_inv=float(ema), loom=loom, loom_speed_s_inv=loom_speed)
            next_ema[side], next_loom[side] = float(ema), loom
        features = np.asarray(values, dtype=np.float64)
        if not np.isfinite(features).all():
            return self._fail(time_s, "nonfinite_features", eyes)
        self._previous_image, self._previous_time_s = gray, time_s
        self._expansion_ema, self._previous_loom = next_ema, next_loom
        return VisionV2Observation(time_s, _readonly(features), eyes, True, initial, False)
