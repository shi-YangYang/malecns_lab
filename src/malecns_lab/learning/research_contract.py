"""Explicit common numeric contract, independent of graph-derived identity IDs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import platform
import math

import numpy as np
import torch

from ..brain.metadata import canonical_hash
from ..brain.tonic import LaminaTonicLIF
from ..senses.spatial_drive import SpatialDrive
from .neural_policy import NeuralPolicy
from .rewiring_contract import require


def _array(value):
    value = np.ascontiguousarray(value.detach().numpy() if isinstance(value, torch.Tensor) else value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}


def common_numeric_contract(policy):
    """Initial untrained values only. No recursive deletion/ignored unknown fields.

    The listed fields describe the shared numerical experiment. Actual graph,
    admission and graph-derived IDs remain in each policy's separate identity.
    """
    require(type(policy) is NeuralPolicy and type(policy.brain) is LaminaTonicLIF, "Exact existing policy/core required")
    policy._validate_parameters()
    brain, mapping, identity = policy.brain, policy.mapping, policy.identity
    require(brain._dtype == torch.float32 and brain.tonic_mode == "fixed" and not torch.count_nonzero(brain.theta),
            "Common initial contract requires unchanged CPU float32 theta=0")
    # Reconstruct expected initialization with the same local generator and order.
    generator = torch.Generator(device="cpu").manual_seed(policy.config.readout_seed)
    expected = torch.empty_like(policy.readout.weight).normal_(std=.01 / math.sqrt(policy.readout.in_features), generator=generator)
    require(torch.equal(expected, policy.readout.weight) and not torch.count_nonzero(policy.readout.bias),
            "Common initial contract requires the registered initial readout")
    model = identity["model"]
    require(model["config"] == asdict(brain.config) and model["dtype"] == str(brain._dtype)
            and model["input_body_ids"] == list(brain.input_body_ids), "Live neural configuration differs from identity")
    require(identity["config"] == asdict(policy.config) and identity["camera"] == asdict(policy.camera)
            and identity["drive"]["config"] == asdict(policy.drive_config)
            and identity["background"] == policy.background
            and (identity["frame_steps"], identity["precondition_steps"])
                == (policy.frame_steps, policy.precondition_steps) == policy.config.steps(brain.config.dt_ms),
            "Live policy clock/configuration differs from identity")
    require(identity["neurons"] == mapping.identity and identity["spatial"] == policy.spatial_map.identity,
            "Live input/output or spatial mapping differs from identity")
    allowed = {"schema", "base_dynamics_schema", "graph", "config", "input_body_ids", "dtype", "backend", "gain", "gradient",
               "event_order", "reference", "implementation_sha256", "tonic", "model_id"}
    require(set(model) in (allowed, allowed | {"research_factory"}), "Unknown dynamics fields cannot be ignored")
    drive = SpatialDrive(policy.spatial_map, policy.drive_config)
    spatial = policy.spatial_map.identity
    contract = {
        "schema": "cnsbridge.common_initial_neural_contract.v1",
        "dynamics": {key: model[key] for key in ("schema", "base_dynamics_schema", "config", "dtype", "backend", "gain",
                     "gradient", "event_order", "reference", "tonic")},
        "numeric_runtime": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "numpy", "scipy")}},
        "numeric_sources": {**{"brain/" + k: v for k, v in model["implementation_sha256"].items()},
                            **identity["implementation_sha256"]},
        "body_ids": _array(brain.graph.body_ids), "group_names": list(brain.graph.group_names),
        "group_index": _array(brain.group_index), "input_mask": _array(brain.input_mask),
        "tonic_group_index": _array(brain.tonic_group_index), "rho": _array(brain.rho),
        "inputs": mapping.inputs.to_dict(), "outputs": mapping.outputs.to_dict(),
        "spatial": {key: spatial[key] for key in ("nodes", "shape", "seed", "rng", "numpy_version", "algorithm", "coordinates",
                                                   "input_indices", "input_body_ids", "pixel_cells", "coverage", "boundary")},
        "drive": {"config": asdict(policy.drive_config), "initial_phase": _array(drive.export_state().phase),
                  "initial_events": _array(drive.export_state().event_counts),
                  **{key: drive.identity[key] for key in ("schema", "phase_dtype", "numpy_version", "phase_initialization",
                                                          "event_rule", "non_input_drive", "background_drive")}},
        "camera": asdict(policy.camera), "policy_config": asdict(policy.config), "frame_steps": policy.frame_steps,
        "precondition_steps": policy.precondition_steps, "background": policy.background,
        "features": identity["features"], "readout": identity["readout"],
        "initial_parameters": {name: _array(value) for name, value in policy.named_parameters()},
    }
    return {**contract, "common_contract_id": canonical_hash(contract)}


def require_common_numeric_contract(left, right):
    """Compare every field of the explicit schema, including initial arrays."""
    for value in (left, right):
        require(value.get("schema") == "cnsbridge.common_initial_neural_contract.v1"
                and value.get("common_contract_id") == canonical_hash({k: v for k, v in value.items() if k != "common_contract_id"}),
                "Unbound common numeric contract")
    require(left == right, "Paired numeric contracts differ")
