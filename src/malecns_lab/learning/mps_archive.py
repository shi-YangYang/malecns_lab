"""Strict, Torch-free complete MPS boundary archives and restoration evidence."""
from pathlib import Path
import numpy as np
from ..brain.metadata import canonical_hash
from .resource_guard import write_json
from .training_contract import (PARAMETERS, ADAM_FIELDS, bound, cursor, expected_ids, optimizer_contract,
                               read_arrays, read_json, require, sha256)
from .training_archive import STATE_KEYS, checkpoint_state_digest, validate_state_digest

SCHEMA = "cnsbridge.mps_training_checkpoint.v1"


def declarations(values):
    return {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in values.items()}


def validate_payload(info, arrays, run):
    require(set(info) == {"schema", "training_run_id", "policy_id", "context", "optimizer", "state", "arrays",
                          "implementation_sha256", "runtime", "arrays_sha256", "checkpoint_id"}, "Checkpoint fields differ")
    bound(info, "checkpoint_id")
    require(info["schema"] == SCHEMA and info["training_run_id"] == run["training_run_id"]
            and info["policy_id"] == run["policy"]["policy_id"] and info["optimizer"] == optimizer_contract("joint")
            and info["implementation_sha256"] == run["implementation_sha256"] and info["runtime"] == run["runtime"],
            "Checkpoint source/runtime/backend/optimizer differs")
    context = info["context"]
    require(set(context) == {*expected_ids(run), "scenes", "epoch", "scene_index", "scene_id", "next_frame", "completed_windows",
                             "optimizer_steps", "parent_checkpoint_id", "log_prefix_sha256"}, "Context fields differ")
    for key, value in expected_ids(run).items():
        require(context[key] == value, "Foreign run/dataset/protocol")
    for key in ("parent_checkpoint_id", "log_prefix_sha256"):
        value = context[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Invalid parent/prefix")
    steps, completed = context["optimizer_steps"], context["completed_windows"]
    require(type(steps) is int and type(completed) is int and 0 < steps == completed <= 160, "Invalid update/window cursor")
    require(context["scenes"] == run["scenes"] and all(context[k] == v for k, v in cursor(run["scenes"], completed, steps).items()), "Scene cursor differs")
    identity = run["policy"]
    nodes, inputs = identity["model"]["graph"]["nodes"], len(identity["neurons"]["inputs"]["body_ids"])
    cfg = identity["model"]["config"]
    shapes = {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "readout_weight": (3, identity["features"]["width"]), "readout_bias": (3,)}
    specs = {"brain.v": ((nodes,), np.float32), "brain.g": ((nodes,), np.float32),
             "brain.refractory_until": ((nodes,), np.int64),
             "brain.pending": ((round(cfg["delay_ms"] / cfg["dt_ms"]), nodes), np.float32),
             "drive.phase": ((inputs,), np.float64), "drive.event_counts": ((inputs,), np.int64)}
    specs.update({f"parameter.{k}": (shape, np.float32) for k, shape in shapes.items()})
    specs.update({f"adam.{PARAMETERS[k]}.{f}": (() if f == "step" else shape, np.float32)
                  for k, shape in shapes.items() for f in ADAM_FIELDS})
    require(set(arrays) == set(specs) and info["arrays"] == declarations(arrays), "Archive inventory/declarations differ")
    for key, (shape, dtype) in specs.items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), f"Invalid array {key}")
    for name in PARAMETERS.values():
        require(float(arrays[f"adam.{name}.step"]) == steps and np.all(arrays[f"adam.{name}.exp_avg_sq"] >= 0), "Adam state invalid")
    state = info["state"]
    require(set(state) == {"policy_id", "frame_index", "brain", "drive"} and set(state["brain"]) == {"model_id", "step"}
            and set(state["drive"]) == {"drive_id", "spatial_map_id", "dt_ms", "step"}, "State metadata fields differ")
    require(type(state["frame_index"]) is int and type(state["brain"]["step"]) is int and type(state["drive"]["step"]) is int,
            "State clocks must be integers")
    digest = checkpoint_state_digest(state, arrays)
    validate_state_digest(digest, identity, context["next_frame"])
    require(np.all(arrays["brain.refractory_until"] >= 0) and np.all(arrays["drive.event_counts"] >= 0)
            and np.all(arrays["drive.event_counts"] <= digest["drive_step"])
            and np.all((arrays["drive.phase"] >= 0) & (arrays["drive.phase"] < 1)), "State values outside contract")
    return {"checkpoint_id": info["checkpoint_id"], "context": context, "state": digest,
            "parameters": {k: arrays[f"parameter.{k}"] for k in PARAMETERS},
            "moments": {f"{n}.{f}": arrays[f"adam.{n}.{f}"] for n in PARAMETERS.values() for f in ADAM_FIELDS}}


def write_checkpoint(directory, run, context, state_metadata, arrays):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "arrays.npz", **arrays)
    info = {"schema": SCHEMA, "training_run_id": run["training_run_id"], "policy_id": run["policy"]["policy_id"],
            "context": context, "optimizer": optimizer_contract("joint"), "state": state_metadata, "arrays": declarations(arrays),
            "implementation_sha256": run["implementation_sha256"], "runtime": run["runtime"],
            "arrays_sha256": sha256(directory / "arrays.npz")}
    info["checkpoint_id"] = canonical_hash(info)
    validate_payload(info, arrays, run)
    write_json(directory / "checkpoint.json", info)
    return info


def inspect_checkpoint(directory, run):
    directory = Path(directory)
    require({p.name for p in directory.iterdir()} == {"arrays.npz", "checkpoint.json"}, "Checkpoint file inventory differs")
    info = read_json(directory / "checkpoint.json")
    require(sha256(directory / "arrays.npz") == info["arrays_sha256"], "Checkpoint SHA mismatch")
    arrays = read_arrays(directory / "arrays.npz")
    saved = validate_payload(info, arrays, run)
    return {**saved, "path": str(directory), "metadata": info, "arrays": arrays}


def verify_resume(segment, registration, previous, run):
    info = read_json(segment / "resume.json")
    arrays = read_arrays(segment / "resume.npz")
    require(set(info) == {"schema", "segment_id", "parent_checkpoint_id", "source_state", "restored_state", "arrays_sha256", "exact"}
            and info["schema"] == "cnsbridge.mps_restore.v1" and info["segment_id"] == registration["segment_id"]
            and info["parent_checkpoint_id"] == previous["checkpoint_id"]
            and info["arrays_sha256"] == sha256(segment / "resume.npz") and info["exact"] is True, "Invalid resume identity")
    require(set(arrays) == set(previous["arrays"]) and all(arrays[k].dtype == v.dtype and np.array_equal(arrays[k], v)
            for k, v in previous["arrays"].items()), "Restoration changed array values")
    require(info["source_state"] == previous["metadata"]["state"], "Restoration source state differs")
    expected = {**info["source_state"], "policy_id": run["policy"]["policy_id"],
                "brain": {**info["source_state"]["brain"], "model_id": run["policy"]["model"]["model_id"]}}
    require(info["restored_state"] == expected, "Restoration changed clocks/drive or failed to bind new backend")
    return checkpoint_state_digest(expected, arrays)
