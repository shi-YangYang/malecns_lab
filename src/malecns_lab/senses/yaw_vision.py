"""Causal engineering motion features on MPS; no environment state enters here."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from ..brain.mps_trainable import require_mps
from .vision_v2 import CHANNELS, VisionV2Config


@dataclass(frozen=True)
class YawVisionState:
    gray: torch.Tensor
    expansion_ema: torch.Tensor
    previous_loom: torch.Tensor
    frame: int


class YawVision:
    """Float32 counterpart of VisionV2, with explicit immutable temporal state.

    The camera contract is exactly 96x72 BGR; resizing is deliberately left to
    a future registered camera adapter. A rejected frame never advances state.
    """

    def __init__(self):
        require_mps()
        self.config = VisionV2Config()
        yy, xx = torch.meshgrid(torch.arange(6, device="mps", dtype=torch.float32) * 12 + 5.5,
                                torch.arange(8, device="mps", dtype=torch.float32) * 6 + 2.5, indexing="ij")
        self.centers = torch.stack((xx, yy), -1)
        self.identity = {"schema": "malecns_lab.yaw_vision.v1", "backend": "mps_float32",
                         "reference_math": self.config.identity(), "shape": [72, 96, 3],
                         "channels": list(CHANNELS), "resize": "none_exact_shape",
                         "linear_algebra": "analytic_2x2_eigenvalues_and_inverse; scalar_centered_expansion_fit"}

    def initial_state(self):
        return YawVisionState(torch.zeros((72, 96), device="mps"), torch.zeros(2, device="mps"),
                              torch.zeros(2, device="mps"), 0)

    def validate(self, state):
        if not isinstance(state, YawVisionState) or type(state.frame) is not int or state.frame < 0:
            raise ValueError("Invalid visual state clock")
        for value, shape in ((state.gray, (72, 96)), (state.expansion_ema, (2,)), (state.previous_loom, (2,))):
            if (value.device.type != "mps" or value.dtype != torch.float32 or tuple(value.shape) != shape
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError("Invalid MPS visual state")
        if (state.gray < 0).any() or (state.gray > 1).any() or (state.previous_loom < 0).any() or (state.previous_loom > 1).any():
            raise ValueError("Visual state outside intensity/loom range")

    @staticmethod
    def _cells(value):
        return value.reshape(6, 12, 8, 6).permute(0, 2, 1, 3).reshape(6, 8, 72)

    def _eye(self, current, previous):
        brightness = self._cells(current).mean()
        gy, gx = torch.gradient((current + previous) * .5)
        ix, iy, it = (self._cells(v) for v in (gx, gy, current - previous))
        xx, yy, xy = (ix * ix).mean(-1), (iy * iy).mean(-1), (ix * iy).mean(-1)
        xt, yt = (ix * it).mean(-1), (iy * it).mean(-1)
        discriminant = ((xx - yy).square() + 4 * xy.square()).clamp_min(0).sqrt()
        small, large = (xx + yy - discriminant) * .5, (xx + yy + discriminant) * .5
        determinant = xx * yy - xy.square()
        valid = ((small >= self.config.min_texture_eigenvalue)
                 & (large <= self.config.max_texture_condition * small) & (determinant > 0))
        denominator = torch.where(valid, determinant, torch.ones_like(determinant))
        flow = torch.stack(((-yy * xt + xy * yt) / denominator / .05,
                            (xy * xt - xx * yt) / denominator / .05), -1)
        valid = valid & torch.isfinite(flow).all(-1)
        if not valid.any():
            raise ValueError("no_valid_flow")
        weights = valid.to(torch.float32)
        count = weights.sum()
        centers_mean = (self.centers * weights[..., None]).sum((0, 1)) / count
        flow_mean = (torch.where(valid[..., None], flow, 0.)).sum((0, 1)) / count
        centered = self.centers - centers_mean
        variance = (centered.square() * weights[..., None]).sum((0, 1))
        if (variance <= 0).any():
            raise ValueError("unidentifiable_expansion")
        slopes = (centered * torch.where(valid[..., None], flow - flow_mean, 0.)).sum((0, 1)) / variance
        if not torch.isfinite(slopes).all():
            raise ValueError("nonfinite_expansion")
        return brightness, flow, weights, count, torch.minimum(slopes[0], slopes[1])

    @torch.no_grad()
    def encode(self, image_bgr, state):
        self.validate(state)
        if isinstance(image_bgr, np.ndarray):
            if image_bgr.dtype != np.uint8 or image_bgr.shape != (72, 96, 3):
                raise ValueError("Expected 96x72 uint8 BGR camera frame")
            image = torch.from_numpy(np.ascontiguousarray(image_bgr)).to("mps")
        elif isinstance(image_bgr, torch.Tensor):
            if image_bgr.device.type != "mps" or image_bgr.dtype != torch.uint8 or tuple(image_bgr.shape) != (72, 96, 3):
                raise ValueError("Expected 96x72 MPS uint8 BGR camera frame")
            image = image_bgr
        else:
            raise ValueError("Unsupported camera frame")
        channels = image.to(torch.float32).unbind(-1)
        gray = (channels[0] * .114 + channels[1] * .587 + channels[2] * .299) / 255.
        features, emas, looms = [], [], []
        for side, columns in enumerate((slice(0, 48), slice(48, 96))):
            current, previous = gray[:, columns], state.gray[:, columns]
            if state.frame == 0:
                features.extend((current.mean(), *(gray.new_zeros(()) for _ in range(6))))
                emas.append(gray.new_zeros(()))
                looms.append(gray.new_zeros(()))
                continue
            brightness, flow, weights, count, expansion = self._eye(current, previous)
            ema = state.expansion_ema[side] + (-math.expm1(-.05 / .1)) * (expansion - state.expansion_ema[side])
            loom = ((ema - .1) / .5).clamp(0., 1.)
            speed = ((loom - state.previous_loom[side]) / .05).clamp_min(0.)
            sign = -1. if side == 0 else 1.
            directions = (sign * flow[..., 0], -sign * flow[..., 0], -flow[..., 1], flow[..., 1])
            features.extend((brightness, *((torch.where(weights.bool(), (d / 24.).clamp(0., 1.), 0.)).sum() / count
                                           for d in directions), loom, (speed / 5.).clamp(0., 1.)))
            emas.append(ema)
            looms.append(loom)
        result = torch.stack(features)
        if not torch.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise ValueError("nonfinite_features")
        return result, YawVisionState(gray, torch.stack(emas), torch.stack(looms), state.frame + 1)
