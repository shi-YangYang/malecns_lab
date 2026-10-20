"""Explicit MPS pixel encoding and pure device-resident sensory state.

The float32 phase is a new numerical contract, not an exact migration of the
historical NumPy float64 event stream. Only initialization and archive I/O use
NumPy. Every step returns fresh tensors; callers must not mutate owned states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..brain.metadata import canonical_hash, file_hashes
from ..brain.mps_trainable import require_mps
from .spatial_drive import MASK_SHAPE, SpatialDriveConfig, SpatialInputMap
from .target import TargetCamera


@dataclass(frozen=True)
class GPUDriveState:
    phase: torch.Tensor
    event_counts: torch.Tensor
    step: int
    drive_id: str
    spatial_map_id: str


class GPUDrive:
    """A camera mask and event encoder with no CPU numerical step fallback."""

    def __init__(self, mapping: SpatialInputMap, config: SpatialDriveConfig | None = None,
                 camera: TargetCamera | None = None):
        require_mps()
        if not isinstance(mapping, SpatialInputMap):
            raise ValueError("A validated spatial input map is required")
        self.mapping = mapping
        self.config, self.camera = config or SpatialDriveConfig(), camera or TargetCamera()
        if not isinstance(self.config, SpatialDriveConfig) or not isinstance(self.camera, TargetCamera):
            raise ValueError("Invalid sensory or camera configuration")
        self.input_indices = torch.tensor(mapping.input_indices.copy(), dtype=torch.int64, device="mps")
        self.pixel_cells = torch.tensor(mapping.pixel_cells.copy(), dtype=torch.int64, device="mps")
        # Geometry is fixed initialization metadata, not per-frame CPU encoding.
        rows = np.linspace(0, self.camera.height, MASK_SHAPE[0] + 1, dtype=int)
        cols = np.linspace(0, self.camera.width, MASK_SHAPE[1] + 1, dtype=int)
        cells = np.empty((self.camera.height, self.camera.width), np.int64)
        sizes = []
        for row in range(MASK_SHAPE[0]):
            for col in range(MASK_SHAPE[1]):
                cells[rows[row]:rows[row + 1], cols[col]:cols[col + 1]] = row * MASK_SHAPE[1] + col
                sizes.append((rows[row + 1] - rows[row]) * (cols[col + 1] - cols[col]))
        self.image_cells = torch.tensor(cells.ravel(), dtype=torch.int64, device="mps")
        self.cell_sizes = torch.tensor(sizes, dtype=torch.float32, device="mps")
        identity = {
            "schema": "cnsbridge.gpu_spatial_drive.v1", "backend": "torch_mps_float32",
            "spatial_map_id": mapping.map_id, "config": asdict(self.config), "camera": asdict(self.camera),
            "phase_dtype": "float32", "event_count_dtype": "int64", "shape": list(MASK_SHAPE),
            "initialization": "PCG64(phase_seed) float64 once, cast float32, clamp maximum to nextbelow(1)",
            "pixel_rule": "uint8 BGR; red>=70; 100*red>165*green; 100*red>165*blue; red-green>=35",
            "pooling": "floor linspace nonoverlapping cells; float32 red count divided by cell pixel count",
            "event_rule": ("float32 phase + (occupancy*max_rate_hz)*(dt_ms/1000); emit>=1; subtract event; "
                           "clamp rounded residual to nextbelow(1)"),
            "non_input_drive": "exact zero", "background_drive": "none",
            "numpy_initialization_version": np.__version__, "torch_version": torch.__version__,
            "implementation_sha256": file_hashes(Path(__file__))["sha256"],
        }
        self._identity = {**identity, "drive_id": canonical_hash(identity)}

    @property
    def identity(self):
        from copy import deepcopy
        return deepcopy(self._identity)

    @property
    def drive_id(self):
        return self._identity["drive_id"]

    def validate(self, state):
        if (not isinstance(state, GPUDriveState) or state.drive_id != self.drive_id
                or state.spatial_map_id != self.mapping.map_id or type(state.step) is not int or state.step < 0):
            raise ValueError("GPU drive state identity/clock differs")
        shape = (len(self.mapping.input_indices),)
        if (state.phase.device.type != "mps" or state.phase.dtype != torch.float32 or state.phase.shape != shape
                or state.phase.requires_grad or not torch.isfinite(state.phase).all()
                or (state.phase < 0).any() or (state.phase >= 1).any()):
            raise ValueError("Invalid device phase state")
        if (state.event_counts.device.type != "mps" or state.event_counts.dtype != torch.int64
                or state.event_counts.shape != shape or state.event_counts.requires_grad
                or (state.event_counts < 0).any() or (state.event_counts > state.step).any()):
            raise ValueError("Invalid device event counts")

    @torch.no_grad()
    def initial_state(self):
        values = np.random.Generator(np.random.PCG64(self.config.phase_seed)).random(len(self.mapping.input_indices))
        phase = torch.tensor(values.astype(np.float32), device="mps").clamp_max(0.9999999403953552)
        return GPUDriveState(phase, torch.zeros_like(self.input_indices), 0, self.drive_id, self.mapping.map_id)

    @torch.no_grad()
    def mask(self, frame):
        expected = (self.camera.height, self.camera.width, 3)
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8 or frame.shape != expected:
                raise ValueError("Frame must retain registered uint8 BGR shape")
            value = torch.from_numpy(np.ascontiguousarray(frame)).to("mps")
        elif isinstance(frame, torch.Tensor) and frame.device.type == "mps":
            if frame.dtype != torch.uint8 or tuple(frame.shape) != expected:
                raise ValueError("Frame must retain registered uint8 BGR shape")
            value = frame
        else:
            raise ValueError("Frame must be a uint8 NumPy archive value or MPS tensor")
        blue, green, red = value.to(torch.int32).unbind(-1)
        marked = (red >= 70) & (red * 100 > 165 * green) & (red * 100 > 165 * blue) & (red - green >= 35)
        counts = torch.zeros(MASK_SHAPE[0] * MASK_SHAPE[1], dtype=torch.float32, device="mps")
        counts.scatter_add_(0, self.image_cells, marked.flatten().to(torch.float32))
        return (counts / self.cell_sizes).reshape(MASK_SHAPE)

    @torch.no_grad()
    def step(self, mask, state):
        self.validate(state)
        if (not isinstance(mask, torch.Tensor) or mask.device.type != "mps" or mask.dtype != torch.float32
                or tuple(mask.shape) != MASK_SHAPE or not torch.isfinite(mask).all()
                or (mask < 0).any() or (mask > 1).any()):
            raise ValueError("Mask must be finite MPS float32 occupancy in [0,1]")
        phase = state.phase + (mask.flatten()[self.pixel_cells] * self.config.max_rate_hz) * (self.config.dt_ms / 1000)
        fired = phase >= 1
        # At the legal full increment, nextbelow(1) + 1 rounds to 2 in float32.
        # Preserve one event and the nearest legal residual, instead of phase=1.
        residual = (phase - fired.to(torch.float32)).clamp_max(0.9999999403953552)
        next_state = GPUDriveState(residual, state.event_counts + fired.to(torch.int64),
                                   state.step + 1, self.drive_id, self.mapping.map_id)
        increment = torch.zeros(self.mapping.n_nodes, dtype=torch.float32, device="mps")
        increment.index_copy_(0, self.input_indices, fired.to(torch.float32) * self.config.event_voltage_mv)
        return increment, next_state

    def snapshot(self, state):
        self.validate(state)
        return {"drive_id": self.drive_id, "spatial_map_id": self.mapping.map_id, "step": state.step,
                "phase": state.phase.detach().cpu().numpy().copy(),
                "event_counts": state.event_counts.detach().cpu().numpy().copy()}

    def restore(self, saved):
        if not isinstance(saved, dict) or set(saved) != {"drive_id", "spatial_map_id", "step", "phase", "event_counts"}:
            raise ValueError("Unexpected GPU drive archive fields")
        for name, dtype in (("phase", np.float32), ("event_counts", np.int64)):
            if not isinstance(saved[name], np.ndarray) or saved[name].dtype != dtype:
                raise ValueError("GPU drive archive dtype differs")
        state = GPUDriveState(torch.from_numpy(saved["phase"].copy()).to("mps"),
                              torch.from_numpy(saved["event_counts"].copy()).to("mps"),
                              saved["step"], saved["drive_id"], saved["spatial_map_id"])
        self.validate(state)
        return state
