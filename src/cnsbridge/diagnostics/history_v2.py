"""Causal neural and direct-visual features sharing one reset/validity boundary."""

from __future__ import annotations

import numpy as np

from ..senses.vision_v2 import CHANNELS, FRAME_INTERVAL_S

WARMUP_STEPS = 40
EMA_TAUS = (0.05, 0.2, 1.0)


class CausalHistoryV2:
    def __init__(self, output_body_ids):
        self.output_body_ids = tuple(int(value) for value in output_body_ids)
        if len(self.output_body_ids) != 33 or list(self.output_body_ids) != sorted(set(self.output_body_ids)):
            raise ValueError("Expected 33 unique ascending output body IDs")
        self.neural_names = tuple(f"lag_{lag}.body_{body}.hz" for lag in (3, 2, 1, 0) for body in self.output_body_ids)
        self.visual_names = tuple(f"lag_{lag}.{name}" for lag in (3, 2, 1, 0) for name in CHANNELS) + tuple(
            f"ema_{tau:g}s.{name}" for tau in EMA_TAUS for name in CHANNELS)
        self.reset()

    def reset(self):
        self.neural = np.zeros((4, 33), dtype=np.float64)
        self.visual = np.zeros((4, 14), dtype=np.float64)
        self.ema = np.zeros((3, 14), dtype=np.float64)
        self.steps = 0

    def identity(self):
        return {"neural_names": list(self.neural_names), "visual_names": list(self.visual_names),
                "neural_dimensions": 132, "visual_dimensions": 98, "history_order": "oldest_to_newest",
                "ema_tau_s": list(EMA_TAUS), "initial_state": "all histories and visual EMA zero",
                "warmup_steps_excluded": WARMUP_STEPS, "warmup_seconds": 2.0,
                "visual_ema": "ema += (1-exp(-0.05/tau))*(current-ema)",
                "failure": "no history or EMA advance; explicit new segment resets all"}

    def append(self, neural_hz, visual_features):
        neural, visual = np.asarray(neural_hz), np.asarray(visual_features)
        if neural.shape != (33,) or visual.shape != (14,) or not np.isfinite(neural).all() or not np.isfinite(visual).all():
            raise ValueError("Finite 33-cell and 14-channel observations required")
        if np.any(neural < 0) or np.any((visual < 0) | (visual > 1)):
            raise ValueError("Invalid neural rates or visual feature range")
        self.neural[:-1] = self.neural[1:]
        self.neural[-1] = neural
        self.visual[:-1] = self.visual[1:]
        self.visual[-1] = visual
        alpha = -np.expm1(-FRAME_INTERVAL_S / np.asarray(EMA_TAUS))
        self.ema += alpha[:, None] * (visual - self.ema)
        self.steps += 1
        return self.neural.ravel().copy(), np.concatenate((self.visual.ravel(), self.ema.ravel()))

    @property
    def ready(self):
        return self.steps > WARMUP_STEPS
