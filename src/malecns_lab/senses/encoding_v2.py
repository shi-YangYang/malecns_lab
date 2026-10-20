"""Group-broadcast V2 drive, with no inferred neuron-index retinotopy."""

from __future__ import annotations

import numpy as np

from ..brain.connectome import Connectome
from .vision_v2 import CHANNELS, VisionV2Observation

GROUP_STEMS = ("R16", "T4a", "T4b", "T4c", "T4d", "LPLC2", "LC4")
INPUT_GROUPS = tuple(f"{stem}_{eye}" for eye in "LR" for stem in GROUP_STEMS)
MAX_RATES = np.tile(np.array([40, 120, 120, 120, 120, 200, 150], dtype=np.float64), 2)


class GroupInputV2:
    def __init__(self, connectome: Connectome):
        self.groups = {name: connectome.group(name).copy() for name in INPUT_GROUPS}
        if any(not len(idx) for idx in self.groups.values()):
            raise ValueError("Every required V2 visual input group must contain cells")
        indices = np.concatenate(list(self.groups.values()))
        if len(np.unique(indices)) != len(indices):
            raise ValueError("V2 input groups must not overlap")
        self.indices = np.sort(indices)
        self.body_ids = connectome.body_ids[self.indices].copy()
        self.group_positions = {name: np.searchsorted(self.indices, idx) for name, idx in self.groups.items()}
        self.neuron_counts = np.array([len(idx) for idx in self.groups.values()], dtype=np.int64)

    def identity(self) -> dict:
        return {
            "protocol": "cnsbridge.group_broadcast.v2", "channels": list(CHANNELS), "groups": list(INPUT_GROUPS),
            "max_hz": MAX_RATES.tolist(), "mapping": "each matching neuron gets the same channel rate",
            "haltere": "disabled; no telemetry supplied and no zero-measurement claim",
            "index_to_image_mapping": None,
        }

    def rates(self, observation: VisionV2Observation) -> tuple[np.ndarray, np.ndarray]:
        if not observation.valid or observation.features is None:
            raise ValueError("Invalid visual observation must not generate rates")
        features = np.asarray(observation.features)
        if features.shape != (14,) or not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
            raise ValueError("Expected 14 finite visual features in [0, 1]")
        group_rates = MAX_RATES * features
        rates = np.empty(len(self.indices), dtype=np.float64)
        for name, rate in zip(INPUT_GROUPS, group_rates):
            rates[self.group_positions[name]] = rate
        return rates, group_rates

    def count_events(self, events: np.ndarray) -> np.ndarray:
        return np.array([events[:, self.group_positions[name]].sum(dtype=np.int64) for name in INPUT_GROUPS])
