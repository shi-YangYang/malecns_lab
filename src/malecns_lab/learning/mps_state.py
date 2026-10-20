"""Device transfers and complete boundary snapshots, with exact roundtrip checks."""
import numpy as np
import torch
from ..brain.trainable import LIFState
from ..senses.spatial_drive import SpatialDriveState
from .neural_policy import PolicyState
from .training_archive import checkpoint_state_digest
from .training_checkpoint import _state_metadata, _validate_state
from .training_contract import PARAMETERS, require
from .mps_optimizer import CPUAdamBridge, snapshot


def state_arrays(state, *, detached=False):
    tensors = (state.brain.v, state.brain.g, state.brain.refractory_until, *state.brain.pending)
    if detached:
        require(all(not t.requires_grad and t.grad_fn is None for t in tensors), "Checkpoint state retains an autograd graph")
    return {"brain.v": state.brain.v.detach().cpu().numpy().copy(), "brain.g": state.brain.g.detach().cpu().numpy().copy(),
            "brain.refractory_until": state.brain.refractory_until.detach().cpu().numpy().copy(),
            "brain.pending": np.stack([t.detach().cpu().numpy() for t in state.brain.pending]),
            "drive.phase": state.drive.phase.copy(), "drive.event_counts": state.drive.event_counts.copy()}


def state_digest(state):
    values = state_arrays(state)
    require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite recurrent state")
    return checkpoint_state_digest(_state_metadata(state), values)


def boundary_arrays(policy, bridge, state):
    require(all(p.grad is None for p in policy.parameters()) and all(p.grad is None for p in bridge.masters.values()),
            "Checkpoint parameters retain gradients")
    return {**state_arrays(state, detached=True),
            **{f"parameter.{k}": v for k, v in snapshot(policy).items() if k in PARAMETERS},
            **{f"adam.{k}": v for k, v in bridge.moments().items()}}


def restore(policy, previous):
    arrays, info = previous["arrays"], previous["metadata"]["state"]
    bridge = CPUAdamBridge(policy, previous["parameters"], previous["moments"], previous["context"]["optimizer_steps"])
    brain = LIFState(torch.from_numpy(arrays["brain.v"].copy()).to("mps"), torch.from_numpy(arrays["brain.g"].copy()).to("mps"),
                     torch.from_numpy(arrays["brain.refractory_until"].copy()).to("mps"),
                     tuple(torch.from_numpy(v.copy()).to("mps") for v in arrays["brain.pending"]),
                     info["brain"]["step"], policy.brain.model_id)
    drive = SpatialDriveState(**info["drive"], phase=arrays["drive.phase"].copy(), event_counts=arrays["drive.event_counts"].copy())
    state = PolicyState(brain, drive, info["frame_index"], policy.policy_id)
    policy._drive_for(state)
    _validate_state(_state_metadata(state), state_arrays(state), policy.identity, previous["context"])
    restored = boundary_arrays(policy, bridge, state)
    require(set(restored) == set(arrays) and all(restored[k].dtype == v.dtype and np.array_equal(restored[k], v)
            for k, v in arrays.items()), "Device transfer changed saved parameters/Adam/recurrent state")
    return bridge, state, restored
