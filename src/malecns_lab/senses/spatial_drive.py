"""Reproducible engineering pixel-to-neuron mapping and continuous input events.

No retinotopy, camera-eye calibration, action decoder or spontaneous background
activity is inferred. This module only supplies the registered input neurons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

import numpy as np

from ..brain.metadata import canonical_hash
from ..brain.policy_groups import PolicyNeuronMap, _immutable

MASK_SHAPE = (12, 16)
MAP_SEED = 1701


@dataclass(frozen=True)
class SpatialInputMap:
    n_nodes: int
    input_indices: np.ndarray
    input_body_ids: np.ndarray
    pixel_cells: np.ndarray
    _identity_json: str
    _map_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("input_indices", "input_body_ids", "pixel_cells"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "_map_id", json.loads(self._identity_json)["spatial_map_id"])

    @property
    def identity(self) -> dict:
        return json.loads(self._identity_json)

    @property
    def map_id(self) -> str:
        return self._map_id


def build_spatial_input_map(policy: PolicyNeuronMap) -> SpatialInputMap:
    """Balance each exact rootSide group over the same whole camera image.

    Group order: missing first, then raw strings sorted by Unicode. Within each
    group, permute ascending body IDs with one PCG64(1701) stream, then assign
    row-major cell number j % 192 to permutation position j.
    """
    if not isinstance(policy, PolicyNeuronMap):
        raise ValueError("A validated policy neuron map is required")
    inputs = policy.inputs
    cells = np.empty(len(inputs.indices), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(MAP_SEED))
    sides = sorted(set(inputs.root_side[~inputs.root_side_missing].tolist()))
    groups = ([None] if inputs.root_side_missing.any() else []) + sides
    coverage = []
    for side in groups:
        members = np.flatnonzero(inputs.root_side_missing if side is None else
                                 (~inputs.root_side_missing & (inputs.root_side == side)))
        members = members[np.argsort(inputs.body_ids[members], kind="stable")]
        permuted = members[rng.permutation(len(members))]
        cells[permuted] = np.arange(len(members), dtype=np.int64) % math.prod(MASK_SHAPE)
        counts = np.bincount(cells[members], minlength=math.prod(MASK_SHAPE))
        coverage.append({"root_side": side, "missing": side is None, "neurons": len(members),
                         "cell_counts": counts.tolist(), "uncovered_cells": np.flatnonzero(counts == 0).tolist(),
                         "uncovered_count": int(np.count_nonzero(counts == 0))})
    identity = {
        "schema": "cnsbridge.spatial_input_map.v1", "role": policy.identity["role"],
        "policy_map_id": policy.map_id, "training_graph_id": policy.identity["training_graph_id"],
        "nodes": policy.n_nodes, "shape": list(MASK_SHAPE), "seed": MAP_SEED,
        "rng": "numpy.random.Generator(PCG64)", "numpy_version": np.__version__,
        "algorithm": "missing then Unicode rootSide; ascending body IDs; one shared RNG stream; permute members; cell=j%192",
        "coordinates": "row-major; row top-to-bottom, column left-to-right; full image for every side group",
        "input_indices": inputs.indices.tolist(), "input_body_ids": inputs.body_ids.tolist(),
        "pixel_cells": cells.tolist(), "coverage": coverage,
        "boundary": "engineered spatial code, not anatomical retinotopy or binocular calibration",
    }
    identity["spatial_map_id"] = canonical_hash(identity)
    return SpatialInputMap(policy.n_nodes, inputs.indices, inputs.body_ids, cells, json.dumps(identity, sort_keys=True))


@dataclass(frozen=True)
class SpatialDriveConfig:
    dt_ms: float = 0.1
    max_rate_hz: float = 150.0
    event_voltage_mv: float = 68.75
    phase_seed: int = 1702

    def __post_init__(self) -> None:
        for name in ("dt_ms", "max_rate_hz", "event_voltage_mv"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.max_rate_hz * self.dt_ms / 1000 > 1:
            raise ValueError("rate * dt must not exceed one event per step")
        if type(self.phase_seed) is not int or self.phase_seed < 0:
            raise ValueError("phase_seed must be a nonnegative integer")


@dataclass(frozen=True)
class SpatialDriveState:
    drive_id: str
    spatial_map_id: str
    dt_ms: float
    step: int
    phase: np.ndarray
    event_counts: np.ndarray

    def __post_init__(self) -> None:
        for name in ("phase", "event_counts"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


@dataclass(frozen=True)
class DriveEvents:
    """One neural step's sparse voltage increments, with its source identity."""

    n_nodes: int
    input_indices: np.ndarray
    increments_mv: np.ndarray
    drive_id: str
    spatial_map_id: str
    step: int

    def __post_init__(self) -> None:
        for name in ("input_indices", "increments_mv"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))

    def to_dense(self, dtype=np.float32) -> np.ndarray:
        """Fresh full-node vector; all non-input cells are exactly zero."""
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Dense voltage increments require float32 or float64")
        result = np.zeros(self.n_nodes, dtype=dtype)
        result[self.input_indices] = self.increments_mv
        return result

    def to_torch(self, *, dtype=None):
        """Allocate the core's CPU input tensor only when requested."""
        import torch

        dtype = torch.float32 if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CPU voltage tensor requires torch.float32 or torch.float64")
        numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
        return torch.from_numpy(self.to_dense(numpy_dtype))


class SpatialDrive:
    """Mutable clock with immutable exported state; phase persists across frames."""

    def __init__(self, mapping: SpatialInputMap, config: SpatialDriveConfig | None = None):
        if not isinstance(mapping, SpatialInputMap):
            raise ValueError("A validated spatial input map is required")
        self.mapping, self.config = mapping, config or SpatialDriveConfig()
        if not isinstance(self.config, SpatialDriveConfig):
            raise ValueError("A SpatialDriveConfig is required")
        identity = {
            "schema": "cnsbridge.spatial_drive.v1", "spatial_map_id": mapping.map_id,
            "config": asdict(self.config), "phase_dtype": "float64", "numpy_version": np.__version__,
            "phase_initialization": "Generator(PCG64(phase_seed)).random, ascending input body ID order",
            "event_rule": "phase += occupancy*max_rate_hz*dt_ms/1000; emit iff phase>=1; subtract 1",
            "non_input_drive": "exact zero", "background_drive": "none",
        }
        identity["drive_id"] = canonical_hash(identity)
        self._drive_id = identity["drive_id"]
        self._identity_json = json.dumps(identity, sort_keys=True)
        self.reset()

    @property
    def identity(self) -> dict:
        return json.loads(self._identity_json)

    @property
    def drive_id(self) -> str:
        return self._drive_id

    def reset(self) -> SpatialDriveState:
        self._phase = np.random.Generator(np.random.PCG64(self.config.phase_seed)).random(len(self.mapping.input_indices))
        self._event_counts = np.zeros(len(self._phase), dtype=np.int64)
        self._step = 0
        return self.export_state()

    def export_state(self) -> SpatialDriveState:
        return SpatialDriveState(self.drive_id, self.mapping.map_id, self.config.dt_ms,
                                 self._step, self._phase, self._event_counts)

    def restore_state(self, state: SpatialDriveState) -> None:
        """Validate everything before changing this encoder's current state."""
        if (not isinstance(state, SpatialDriveState) or state.drive_id != self.drive_id
                or state.spatial_map_id != self.mapping.map_id or state.dt_ms != self.config.dt_ms
                or type(state.step) is not int or state.step < 0):
            raise ValueError("Spatial drive state identity, time step or clock mismatch")
        shape = (len(self.mapping.input_indices),)
        if (state.phase.shape != shape or state.phase.dtype != np.float64 or not np.isfinite(state.phase).all()
                or np.any(state.phase < 0) or np.any(state.phase >= 1)):
            raise ValueError("Invalid spatial phase state")
        if (state.event_counts.shape != shape or state.event_counts.dtype != np.int64
                or np.any(state.event_counts < 0) or np.any(state.event_counts > state.step)):
            raise ValueError("Invalid spatial event counts")
        self._phase, self._event_counts, self._step = state.phase.copy(), state.event_counts.copy(), state.step

    def step(self, mask: np.ndarray) -> DriveEvents:
        """Advance one neural dt; a new image does not reset accumulated phase."""
        if (not isinstance(mask, np.ndarray) or mask.shape != MASK_SHAPE or mask.dtype.kind not in "biuf"
                or not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1)):
            raise ValueError("Spatial input must be a finite 12x16 occupancy array in [0, 1]")
        rates = mask.ravel()[self.mapping.pixel_cells].astype(np.float64) * self.config.max_rate_hz
        phase = self._phase + rates * (self.config.dt_ms / 1000)
        fired = phase >= 1.0
        phase -= fired
        self._phase = phase
        self._event_counts += fired
        self._step += 1
        indices = self.mapping.input_indices[fired]
        return DriveEvents(self.mapping.n_nodes, indices, np.full(len(indices), self.config.event_voltage_mv),
                           self.drive_id, self.mapping.map_id, self._step)
