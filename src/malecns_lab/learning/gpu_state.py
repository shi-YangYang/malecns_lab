"""Boundary transfers for the new float32 sensory and device optimizer state."""
import numpy as np
import torch
from ..brain.trainable import LIFState
from .neural_policy import PolicyState
from .gpu_optimizer import GPUAdam
from .training_contract import PARAMETERS, require
from .training_archive import checkpoint_state_digest


def state_metadata(state, dt_ms):
    return {"policy_id": state.policy_id, "frame_index": state.frame_index,
            "brain": {"model_id": state.brain.model_id, "step": state.brain.step},
            "drive": {"drive_id": state.drive.drive_id, "spatial_map_id": state.drive.spatial_map_id,
                      "dt_ms": dt_ms, "step": state.drive.step}}


def state_arrays(state, *, detached=False):
    tensors = (state.brain.v, state.brain.g, state.brain.refractory_until, *state.brain.pending,
               state.drive.phase, state.drive.event_counts)
    require(all(v.device.type == "mps" for v in tensors), "State left MPS")
    if detached:
        require(all(not v.requires_grad and v.grad_fn is None for v in tensors), "Boundary retains autograd graph")
    result = {"brain.v": state.brain.v.detach().cpu().numpy().copy(), "brain.g": state.brain.g.detach().cpu().numpy().copy(),
              "brain.refractory_until": state.brain.refractory_until.detach().cpu().numpy().copy(),
              "brain.pending": np.stack([v.detach().cpu().numpy() for v in state.brain.pending]),
              "drive.phase": state.drive.phase.detach().cpu().numpy().copy(),
              "drive.event_counts": state.drive.event_counts.detach().cpu().numpy().copy()}
    require(all(np.isfinite(v).all() for v in result.values()), "Nonfinite GPU state")
    return result


def state_digest(policy, state):
    return checkpoint_state_digest(state_metadata(state, policy.drive_config.dt_ms), state_arrays(state))


def boundary_arrays(policy, optimizer, state):
    require(all(p.grad is None for p in policy.parameters()), "Boundary parameters retain gradients")
    policy.validate_state(state)
    saved = optimizer.snapshot()
    return {**state_arrays(state, detached=True),
            **{f"parameter.{k}": saved["parameters"][n] for k, n in PARAMETERS.items()},
            **{f"adam.{k}": v for k, v in saved["moments"].items()}}


def restore(policy, previous):
    arrays, metadata = previous["arrays"], previous["metadata"]["state"]
    policy.restore_parameters(previous["parameters"])
    steps = previous["context"]["optimizer_steps"]
    if steps == 0:
        require(all(np.count_nonzero(v) == 0 for v in previous["moments"].values()), "Genesis Adam is not zero")
    optimizer = GPUAdam(policy.named_parameters(), moments=previous["moments"] if steps else None, steps=steps)
    brain = LIFState(torch.from_numpy(arrays["brain.v"].copy()).to("mps"),
                     torch.from_numpy(arrays["brain.g"].copy()).to("mps"),
                     torch.from_numpy(arrays["brain.refractory_until"].copy()).to("mps"),
                     tuple(torch.from_numpy(v.copy()).to("mps") for v in arrays["brain.pending"]),
                     metadata["brain"]["step"], policy.brain.model_id)
    drive = policy.drive.restore({k: metadata["drive"][k] for k in ("drive_id", "spatial_map_id", "step")} |
                                 {"phase": arrays["drive.phase"], "event_counts": arrays["drive.event_counts"]})
    state = PolicyState(brain, drive, metadata["frame_index"], policy.policy_id)
    actual = boundary_arrays(policy, optimizer, state)
    require(state_metadata(state, policy.drive_config.dt_ms) == metadata and set(actual) == set(arrays)
            and all(v.dtype == actual[k].dtype and np.array_equal(v, actual[k]) for k, v in arrays.items()),
            "GPU restore changed archived values or metadata")
    return optimizer, state, actual
