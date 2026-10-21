"""Bounded visual stimuli for independent atlas experiments."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash, file_hashes
from ..diagnostics.sources_v2 import GeneratedSourceV2, STIMULUS_KINDS

DEFAULT_PARAMETERS = {"stimulus": "expansion", "speed": 1.0, "contrast": 1.0, "duration_s": 0.5}


def validate_parameters(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != set(DEFAULT_PARAMETERS):
        raise ValueError("请提供刺激、速度、对比度和观察时长，不接受其他参数。")
    if not isinstance(value["stimulus"], str) or value["stimulus"] not in STIMULUS_KINDS:
        raise ValueError("请选择有效的刺激类型。")
    result = {"stimulus": value["stimulus"]}
    for key, lower, upper, label in (("speed", 0.25, 2.0, "速度"), ("contrast", 0.25, 1.0, "对比度"),
                                     ("duration_s", 0.1, 3.0, "观察时长")):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{label}须在 {lower} 至 {upper} 之间。")
        result[key] = float(number)
    if abs(result["duration_s"] * 20 - round(result["duration_s"] * 20)) > 1e-8:
        raise ValueError("观察时长须为 0.05 秒的整数倍。")
    return result


class AtlasStimulus(GeneratedSourceV2):
    """Same default pixels as GeneratedSourceV2; controls modify pixels only."""

    def __init__(self, parameters: dict):
        self.parameters = validate_parameters(parameters)
        super().__init__(self.parameters["stimulus"], observation_seconds=self.parameters["duration_s"], texture_seed=7)
        self.metadata.pop("source_id")
        self.metadata.update(kind="generated_atlas_stimulus", speed=self.parameters["speed"], contrast=self.parameters["contrast"],
                             translation_px_s=10.0 * self.parameters["speed"], expansion_s_inv=0.35 * self.parameters["speed"],
                             generator_sha256=file_hashes(Path(__file__))["sha256"],
                             base_generator_sha256=file_hashes(Path(__file__).parents[1] / "diagnostics/sources_v2.py")["sha256"])
        self.metadata["source_id"] = canonical_hash(self.metadata)

    def frame(self, index: int) -> np.ndarray:
        y, x = np.mgrid[:72, :96].astype(np.float64)
        t = max(0.0, index / 20 - 2.0) * self.parameters["speed"]
        if self.stimulus in ("left", "right"):
            x -= (-1 if self.stimulus == "left" else 1) * 10 * t
        if self.stimulus in ("up", "down"):
            y -= (-1 if self.stimulus == "up" else 1) * 10 * t
        if self.stimulus == "expansion":
            scale = np.exp(0.35 * t)
            x, y = 47.5 + (x - 47.5) / scale, 35.5 + (y - 35.5) / scale
        gray = 127 + 34 * np.sin(x * 0.49 + self.phase[0]) + 31 * np.cos(y * 0.43 + self.phase[1])
        gray += 25 * np.sin(0.21 * x + 0.31 * y + self.phase[2]) + 19 * np.cos(0.37 * x - 0.27 * y + self.phase[3])
        if self.parameters["contrast"] != 1.0:
            gray = 127 + (gray - 127) * self.parameters["contrast"]
        return np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
