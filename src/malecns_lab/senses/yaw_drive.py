"""MPS phase events broadcast over explicitly registered visual groups."""
from dataclasses import dataclass

import numpy as np
import torch

from ..brain.mps_trainable import require_mps
from .encoding_v2 import INPUT_GROUPS, MAX_RATES


@dataclass(frozen=True)
class YawDriveState:
    phase: torch.Tensor
    event_counts: torch.Tensor
    step: int


class YawDrive:
    def __init__(self, n_nodes, input_indices, channel_indices):
        require_mps()
        self.n_nodes = n_nodes
        indices, channels = np.asarray(input_indices), np.asarray(channel_indices)
        if (type(n_nodes) is not int or n_nodes <= 0 or indices.ndim != 1 or not len(indices)
                or indices.dtype != np.int64 or channels.dtype != np.int64 or channels.shape != indices.shape
                or np.any(indices < 0) or np.any(indices >= n_nodes) or np.any(indices[1:] <= indices[:-1])
                or np.any(channels < 0) or np.any(channels >= 14) or len(np.unique(channels)) != 14):
            raise ValueError("Invalid disjoint complete visual group mapping")
        self.input_indices = torch.tensor(indices.copy(), device="mps")
        self.channel_indices = torch.tensor(channels.copy(), device="mps")
        self.max_rates = torch.tensor(MAX_RATES.astype(np.float32), device="mps")
        self.identity = {"schema": "malecns_lab.yaw_drive.v1", "groups": list(INPUT_GROUPS),
                         "max_hz": MAX_RATES.tolist(), "dt_ms": .1, "event_voltage_mv": 68.75, "phase_seed": 1702,
                         "phase": "PCG64 seed1702 float64 initialization cast float32; phase+=rate*.0001; event>=1",
                         "input_indices": indices.tolist(), "channel_indices": channels.tolist(),
                         "non_input_increment": "zero", "mapping": "group broadcast; no inferred retinotopy"}

    @torch.no_grad()
    def initial_state(self):
        phase = np.random.default_rng(1702).random(len(self.input_indices)).astype(np.float32)
        return YawDriveState(torch.tensor(phase, device="mps").clamp_max(.9999999403953552),
                             torch.zeros_like(self.input_indices), 0)

    def validate(self, state):
        if not isinstance(state, YawDriveState) or type(state.step) is not int or state.step < 0:
            raise ValueError("Invalid sensory state clock")
        for value, dtype in ((state.phase, torch.float32), (state.event_counts, torch.int64)):
            if (value.device.type != "mps" or value.dtype != dtype or value.shape != self.input_indices.shape
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError("Invalid sensory tensor")
        if ((state.phase < 0).any() or (state.phase >= 1).any()
                or (state.event_counts < 0).any() or (state.event_counts > state.step).any()):
            raise ValueError("Invalid sensory phase or event count")

    @torch.no_grad()
    def rates(self, features):
        if (features.device.type != "mps" or features.dtype != torch.float32 or features.shape != (14,)
                or not torch.isfinite(features).all() or (features < 0).any() or (features > 1).any()):
            raise ValueError("Expected finite fourteen-channel MPS visual features")
        return features * self.max_rates

    @torch.no_grad()
    def step(self, group_rates, state):
        # The policy validates owned state/rates once at each frame boundary;
        # this pure arithmetic step is also replayed inside checkpoint blocks.
        phase = state.phase + group_rates[self.channel_indices] * .0001
        fired = phase >= 1
        next_state = YawDriveState((phase - fired.to(torch.float32)).clamp_max(.9999999403953552),
                                  state.event_counts + fired.to(torch.int64), state.step + 1)
        increment = torch.zeros(self.n_nodes, device="mps")
        increment.index_copy_(0, self.input_indices, fired.to(torch.float32) * 68.75)
        return increment, next_state
