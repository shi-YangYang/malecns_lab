"""Torch-free complete checkpoint inspection for the supervised training parent.

The immutable P3c1 reader constructs tensors. This companion checks its persisted
schema against an externally validated run, retaining only small arrays/digests.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..brain.metadata import canonical_hash
from .training_contract import (PARAMETERS, ADAM_FIELDS, active_keys, array_digest, bound, expected_ids,
                                optimizer_contract, read_arrays, read_json, require, sha256)

STATE_KEYS = {"brain.v", "brain.g", "brain.refractory_until", "brain.pending", "drive.phase", "drive.event_counts"}
DETERMINISM = {
    "sample_order": "declared_scene_order; ascending_frame_index; fixed_two_frame_windows",
    "stochastic_training_operators": "none; no sampling/noise/dropout",
    "rng_resume": "no global RNG consumed; initialization is identity-bound; sensory phase is saved in full",
    "state_boundary": "state computed before Adam; retain values after update and detach; no prefix replay",
}


def checkpoint_state_digest(info, arrays):
    values = {key: array_digest(arrays["brain." + key]) for key in ("v", "g", "refractory_until")}
    values.update({f"pending_{i}": array_digest(value) for i, value in enumerate(arrays["brain.pending"])})
    values.update(drive_phase=array_digest(arrays["drive.phase"]), drive_event_counts=array_digest(arrays["drive.event_counts"]))
    record = {"policy_id": info["policy_id"], "model_id": info["brain"]["model_id"],
              "drive_id": info["drive"]["drive_id"], "spatial_map_id": info["drive"]["spatial_map_id"],
              "brain_step": info["brain"]["step"], "drive_step": info["drive"]["step"], "frame_index": info["frame_index"],
              "drive_dt_ms": info["drive"]["dt_ms"], "values": values}
    return {**record, "state_id": canonical_hash(record)}


def validate_state_digest(value, identity, frame):
    bound(value, "state_id")
    step = identity["precondition_steps"] + frame * identity["frame_steps"]
    require(value["policy_id"] == identity["policy_id"] and value["model_id"] == identity["model"]["model_id"]
            and value["drive_id"] == identity["drive"]["drive_id"] and value["spatial_map_id"] == identity["spatial"]["spatial_map_id"]
            and value["frame_index"] == frame and value["brain_step"] == value["drive_step"] == step
            and value["drive_dt_ms"] == identity["drive"]["config"]["dt_ms"], "State identity/clock mismatch")
    nodes = identity["model"]["graph"]["nodes"]
    inputs = len(identity["neurons"]["inputs"]["body_ids"])
    lif = identity["model"]["config"]
    delay = round(lif["delay_ms"] / lif["dt_ms"])
    specs = {"v": ((nodes,), np.float32), "g": ((nodes,), np.float32), "refractory_until": ((nodes,), np.int64),
             "drive_phase": ((inputs,), np.float64), "drive_event_counts": ((inputs,), np.int64),
             **{f"pending_{i}": ((nodes,), np.float32) for i in range(delay)}}
    require(set(value["values"]) == set(specs), "Incomplete state digests")
    for key, (shape, dtype) in specs.items():
        item = value["values"][key]
        require(set(item) == {"shape", "dtype", "sha256"} and item["shape"] == list(shape) and item["dtype"] == np.dtype(dtype).str
                and isinstance(item["sha256"], str) and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]), "State digest declaration differs")


def inspect_checkpoint(directory, run, context):
    directory = Path(directory)
    meta = read_json(directory / "checkpoint.json")
    bound(meta, "checkpoint_id")
    require(set(meta) == {"schema", "purpose", "context", "optimizer", "state", "arrays", "policy_checkpoint_id",
                          "files_sha256", "implementation_sha256", "runtime", "determinism", "checkpoint_id"}, "Training manifest fields differ")
    require(meta["schema"] == "cnsbridge.training_checkpoint.v1" and meta["purpose"] == "detached_two_frame_tbptt_continuation",
            "Not a complete training checkpoint")
    require(meta["context"] == context and meta["optimizer"] == optimizer_contract(run["mode"]), "Checkpoint cursor/optimizer mismatch")
    require(meta["runtime"] == run["runtime"] and meta["determinism"] == DETERMINISM, "Checkpoint runtime/determinism mismatch")
    source_names = ("learning/training_checkpoint.py", "learning/training_optimizer.py", "learning/policy_checkpoint.py")
    require(meta["implementation_sha256"] == {k: run["implementation_sha256"][k] for k in source_names}, "Checkpoint sources differ")
    files = ("policy/checkpoint.json", "policy/parameters.npz", "training.npz")
    require(set(meta["files_sha256"]) == set(files), "Checkpoint file inventory differs")
    for name in files:
        require(sha256(directory / name) == meta["files_sha256"][name], "Checkpoint file SHA mismatch")
    policy = read_json(directory / "policy/checkpoint.json")
    bound(policy, "checkpoint_id")
    require(set(policy) == {"schema", "purpose", "policy", "parameters", "arrays_sha256", "run_identity", "checkpoint_id"}
            and policy["schema"] == "cnsbridge.neural_policy_checkpoint.v1" and policy["purpose"] == "deployment_parameters_only"
            and policy["policy"] == run["policy"] and policy["run_identity"] == expected_ids(run)
            and policy["checkpoint_id"] == meta["policy_checkpoint_id"]
            and policy["arrays_sha256"] == sha256(directory / "policy/parameters.npz"), "Nested policy checkpoint mismatch")
    identity = run["policy"]
    shapes = {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "readout_weight": (3, identity["features"]["width"]), "readout_bias": (3,)}
    params = read_arrays(directory / "policy/parameters.npz")
    require(set(params) == set(PARAMETERS), "Parameter inventory mismatch")
    for key, value in params.items():
        require(value.shape == shapes[key] and value.dtype == np.float32 and np.isfinite(value).all(), "Parameter values invalid")
    require(policy["parameters"] == {key: {"parameter": name, "shape": list(shapes[key]), "dtype": np.dtype(np.float32).str}
                                     for key, name in PARAMETERS.items()}, "Parameter declaration mismatch")
    data = read_arrays(directory / "training.npz")
    moments_keys = {f"{PARAMETERS[key]}.{field}" for key in active_keys(run["mode"]) for field in ADAM_FIELDS}
    require(set(data) == STATE_KEYS | moments_keys and meta["arrays"] == {
        key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in data.items()}, "Training array inventory differs")
    nodes, inputs = identity["model"]["graph"]["nodes"], len(identity["neurons"]["inputs"]["body_ids"])
    lif = identity["model"]["config"]
    delay = round(lif["delay_ms"] / lif["dt_ms"])
    specs = {"brain.v": ((nodes,), np.float32), "brain.g": ((nodes,), np.float32),
             "brain.refractory_until": ((nodes,), np.int64), "brain.pending": ((delay, nodes), np.float32),
             "drive.phase": ((inputs,), np.float64), "drive.event_counts": ((inputs,), np.int64)}
    steps = context["optimizer_steps"]
    require(type(steps) is int and 0 < steps <= context["completed_windows"], "Invalid Adam steps")
    for key in active_keys(run["mode"]):
        for field in ADAM_FIELDS:
            specs[f"{PARAMETERS[key]}.{field}"] = (() if field == "step" else shapes[key], np.float32)
    for key, (shape, dtype) in specs.items():
        require(data[key].shape == shape and data[key].dtype == dtype and np.isfinite(data[key]).all(), "State/moment values invalid")
    for key in active_keys(run["mode"]):
        require(float(data[f"{PARAMETERS[key]}.step"]) == steps and np.all(data[f"{PARAMETERS[key]}.exp_avg_sq"] >= 0), "Adam states invalid")
    info = meta["state"]
    require(set(info) == {"policy_id", "frame_index", "brain", "drive"}
            and set(info["brain"]) == {"model_id", "step"}
            and set(info["drive"]) == {"drive_id", "spatial_map_id", "dt_ms", "step"}, "State metadata fields differ")
    digest = checkpoint_state_digest(info, data)
    validate_state_digest(digest, identity, context["next_frame"])
    require(np.all(data["brain.refractory_until"] >= 0) and np.all(data["drive.event_counts"] >= 0)
            and np.all(data["drive.event_counts"] <= digest["drive_step"])
            and np.all((data["drive.phase"] >= 0) & (data["drive.phase"] < 1)), "State range invalid")
    moments = {key: data[key] for key in moments_keys}
    return {"checkpoint_id": meta["checkpoint_id"], "context": context, "state": digest,
            "parameters": params, "moments": moments}
