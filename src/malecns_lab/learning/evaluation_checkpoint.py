"""No-Adam evaluation boundaries, including frame zero and odd frame indices.

Reading and validating is Torch-free. Only restore_state imports tensor types.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..brain.metadata import canonical_hash
from .evaluation_contract import parameter_identity
from .resource_guard import write_json
from .training_archive import checkpoint_state_digest, validate_state_digest
from .training_contract import bound, read_arrays, read_json, require, sha256

SCHEMA = "cnsbridge.evaluation_checkpoint.v1"
PURPOSE = "no_grad_closed_loop_continuation"
STATE_KEYS = {"brain.v", "brain.g", "brain.refractory_until", "brain.pending", "drive.phase", "drive.event_counts"}
CONTEXT_KEYS = {"evaluation_run_id", "protocol_id", "condition_id", "parameter_set_id", "scene_id", "scene_spec_id",
                "next_frame", "planned_frames", "terminal", "termination_reason", "parent_checkpoint_id",
                "log_prefix_sha256", "environment"}


def state_payload(state):
    tensors = (state.brain.v, state.brain.g, state.brain.refractory_until, *state.brain.pending)
    require(all(not t.requires_grad and t.grad_fn is None for t in tensors), "Evaluation state must be no-grad")
    info = {"policy_id": state.policy_id, "frame_index": state.frame_index,
            "brain": {"model_id": state.brain.model_id, "step": state.brain.step},
            "drive": {"drive_id": state.drive.drive_id, "spatial_map_id": state.drive.spatial_map_id,
                      "dt_ms": state.drive.dt_ms, "step": state.drive.step}}
    arrays = {"brain.v": state.brain.v.numpy().copy(), "brain.g": state.brain.g.numpy().copy(),
              "brain.refractory_until": state.brain.refractory_until.numpy().copy(),
              "brain.pending": np.stack([t.numpy() for t in state.brain.pending]),
              "drive.phase": state.drive.phase.copy(), "drive.event_counts": state.drive.event_counts.copy()}
    return info, arrays


def validate_arrays(info, arrays, identity, frame):
    require(type(frame) is int and frame >= 0, "Frame must be a non-boolean nonnegative integer")
    require(set(info) == {"policy_id", "frame_index", "brain", "drive"}
            and set(info["brain"]) == {"model_id", "step"}
            and set(info["drive"]) == {"drive_id", "spatial_map_id", "dt_ms", "step"}, "State metadata differs")
    require(type(info["frame_index"]) is int and all(type(info[key]["step"]) is int for key in ("brain", "drive")),
            "State clocks must be non-boolean integers")
    nodes, inputs = identity["model"]["graph"]["nodes"], len(identity["neurons"]["inputs"]["body_ids"])
    config = identity["model"]["config"]
    delay = round(config["delay_ms"] / config["dt_ms"])
    expected = {"brain.v": ((nodes,), np.float32), "brain.g": ((nodes,), np.float32),
                "brain.refractory_until": ((nodes,), np.int64), "brain.pending": ((delay, nodes), np.float32),
                "drive.phase": ((inputs,), np.float64), "drive.event_counts": ((inputs,), np.int64)}
    require(set(arrays) == STATE_KEYS, "State array inventory differs")
    for key, (shape, dtype) in expected.items():
        require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(),
                f"Invalid state array: {key}")
    digest = checkpoint_state_digest(info, arrays)
    validate_state_digest(digest, identity, frame)
    require(np.all(arrays["brain.refractory_until"] >= 0) and np.all(arrays["drive.event_counts"] >= 0)
            and np.all(arrays["drive.event_counts"] <= digest["drive_step"])
            and np.all((arrays["drive.phase"] >= 0) & (arrays["drive.phase"] < 1)), "State range invalid")
    return digest


def context_for(run, episode, frame, environment, parent, prefix_sha):
    return {**{key: episode[key] for key in ("evaluation_run_id", "protocol_id", "condition_id", "parameter_set_id",
                                            "scene_id", "scene_spec_id", "planned_frames")},
            "next_frame": frame, "terminal": environment["done"],
            "termination_reason": environment["truth"]["failure"] or ("duration_complete" if environment["done"] else None),
            "parent_checkpoint_id": parent, "log_prefix_sha256": prefix_sha, "environment": environment}


def validate_context(context, episode):
    require(set(context) == CONTEXT_KEYS, "Evaluation context fields differ")
    require(all(context[k] == episode[k] for k in ("evaluation_run_id", "protocol_id", "condition_id", "parameter_set_id",
                                                   "scene_id", "scene_spec_id", "planned_frames")), "Context identity differs")
    frame = context["next_frame"]
    require(type(frame) is int and 0 <= frame <= episode["planned_frames"] and type(context["planned_frames"]) is int,
            "Invalid evaluation frame cursor")
    environment = context["environment"]
    require(type(context["terminal"]) is bool and context["terminal"] == environment["done"]
            and environment["next_frame"] == frame and type(environment["next_frame"]) is int
            and environment["truth"]["time_s"] == min(frame * .1, 8.), "Environment cursor/terminal differs")
    expected_reason = environment["truth"]["failure"] or ("duration_complete" if context["terminal"] else None)
    require(context["termination_reason"] == expected_reason
            and context["terminal"] == (frame == episode["planned_frames"] or environment["truth"]["failure"] is not None),
            "Terminal reason is inconsistent")
    for key in ("log_prefix_sha256", "parent_checkpoint_id"):
        value = context[key]
        require((key == "parent_checkpoint_id" and value is None) or
                (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)), "Invalid prefix/parent ID")


def save_checkpoint(directory, run, episode, context, *, state=None, actual_parameters=None):
    validate_context(context, episode)
    neural = episode["parameter_set_id"] is not None
    require(neural == (state is not None) == (actual_parameters is not None), "Neural/baseline state inventory differs")
    info, arrays, digest = None, {}, None
    if neural:
        require(parameter_identity(run["policy"], actual_parameters)["parameter_set_id"] == episode["parameter_set_id"],
                "Parameters changed before checkpoint")
        info, arrays = state_payload(state)
        digest = validate_arrays(info, arrays, run["policy"], context["next_frame"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if neural:
        np.savez(directory / "state.npz", **arrays)
    metadata = {"schema": SCHEMA, "purpose": PURPOSE, "context": context, "state": info, "state_digest": digest,
                "parameter_identity": parameter_identity(run["policy"], actual_parameters) if neural else None,
                "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in arrays.items()},
                "files_sha256": {"state.npz": sha256(directory / "state.npz")} if neural else {},
                "implementation_sha256": run["implementation_sha256"], "runtime": run["runtime"]}
    metadata["checkpoint_id"] = canonical_hash(metadata)
    write_json(directory / "checkpoint.json", metadata)
    return metadata


def inspect_checkpoint(directory, run, episode, context, *, keep_arrays=False):
    validate_context(context, episode)
    directory = Path(directory)
    metadata = read_json(directory / "checkpoint.json")
    bound(metadata, "checkpoint_id")
    require(set(metadata) == {"schema", "purpose", "context", "state", "state_digest", "parameter_identity", "arrays",
                              "files_sha256", "implementation_sha256", "runtime", "checkpoint_id"}
            and metadata["schema"] == SCHEMA and metadata["purpose"] == PURPOSE and metadata["context"] == context,
            "Evaluation checkpoint identity/context differs")
    require(metadata["implementation_sha256"] == run["implementation_sha256"] and metadata["runtime"] == run["runtime"],
            "Evaluation checkpoint source/runtime changed")
    neural = episode["parameter_set_id"] is not None
    require({p.name for p in directory.iterdir()} == ({"checkpoint.json", "state.npz"} if neural else {"checkpoint.json"}),
            "Checkpoint contains unexpected files")
    arrays = {}
    if neural:
        require(metadata["parameter_identity"] == run["parameters"][episode["condition"]], "Checkpoint parameter values differ")
        require(metadata["files_sha256"] == {"state.npz": sha256(directory / "state.npz")}, "State file SHA differs")
        arrays = read_arrays(directory / "state.npz")
        digest = validate_arrays(metadata["state"], arrays, run["policy"], context["next_frame"])
        require(metadata["state_digest"] == digest, "Actual state differs from its digest")
    else:
        require(metadata["state"] is None and metadata["state_digest"] is None and metadata["parameter_identity"] is None
                and metadata["files_sha256"] == {} and not (directory / "state.npz").exists(), "Baseline fabricated neural state")
    require(metadata["arrays"] == {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in arrays.items()},
            "Checkpoint array declarations differ")
    return {"checkpoint_id": metadata["checkpoint_id"], "state": metadata["state_digest"], "context": context,
            "path": str(directory), **({"metadata": metadata, "arrays": arrays} if keep_arrays else {})}


def restore_state(archive, policy):
    """Worker-only tensor construction; does not initialize or replay the brain."""
    import torch
    from ..brain.trainable import LIFState
    from ..senses.spatial_drive import SpatialDriveState
    from .neural_policy import PolicyState
    metadata, arrays = archive["metadata"], archive["arrays"]
    actual = {"theta": policy.brain.theta.detach().numpy(), "readout_weight": policy.readout.weight.detach().numpy(),
              "readout_bias": policy.readout.bias.detach().numpy()}
    require(parameter_identity(policy.identity, actual) == metadata["parameter_identity"], "Wrong actual parameters at restore")
    info = metadata["state"]
    validate_arrays(info, arrays, policy.identity, metadata["context"]["next_frame"])
    brain = LIFState(torch.from_numpy(arrays["brain.v"]), torch.from_numpy(arrays["brain.g"]),
                     torch.from_numpy(arrays["brain.refractory_until"]),
                     tuple(torch.from_numpy(value) for value in arrays["brain.pending"]), **info["brain"])
    drive = SpatialDriveState(**info["drive"], phase=arrays["drive.phase"], event_counts=arrays["drive.event_counts"])
    state = PolicyState(brain, drive, info["frame_index"], info["policy_id"])
    policy._drive_for(state)
    return state
