"""Torch-free validation of complete all-GPU boundary archives."""
from pathlib import Path
import numpy as np
from ..brain.metadata import canonical_hash
from .resource_guard import write_json
from .training_contract import PARAMETERS, ADAM_FIELDS, bound, cursor, expected_ids, read_arrays, read_json, require, sha256
from .training_archive import checkpoint_state_digest

SCHEMA = "cnsbridge.gpu_recovery_checkpoint.v1"


def declarations(values):
    return {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in values.items()}


def state_specs(identity):
    nodes, cfg = identity["model"]["graph"]["nodes"], identity["model"]["config"]
    return {"brain.v": ((nodes,), np.float32), "brain.g": ((nodes,), np.float32),
            "brain.refractory_until": ((nodes,), np.int64),
            "brain.pending": ((round(cfg["delay_ms"] / cfg["dt_ms"]), nodes), np.float32),
            "drive.phase": ((len(identity["neurons"]["inputs"]["body_ids"]),), np.float32),
            "drive.event_counts": ((len(identity["neurons"]["inputs"]["body_ids"]),), np.int64)}


def validate_state(metadata, arrays, identity, frame):
    require(set(metadata) == {"policy_id", "frame_index", "brain", "drive"}
            and set(metadata["brain"]) == {"model_id", "step"}
            and set(metadata["drive"]) == {"drive_id", "spatial_map_id", "dt_ms", "step"}, "State metadata fields differ")
    step = identity["precondition_steps"] + frame * identity["frame_steps"]
    require(type(frame) is int and frame >= 0 and type(metadata["frame_index"]) is int
            and type(metadata["brain"]["step"]) is int and type(metadata["drive"]["step"]) is int
            and metadata["frame_index"] == frame and metadata["brain"]["step"] == metadata["drive"]["step"] == step,
            "State clocks differ")
    require(metadata["policy_id"] == identity["policy_id"] and metadata["brain"]["model_id"] == identity["model"]["model_id"]
            and metadata["drive"]["drive_id"] == identity["drive"]["drive_id"]
            and metadata["drive"]["spatial_map_id"] == identity["spatial"]["spatial_map_id"]
            and metadata["drive"]["dt_ms"] == identity["drive"]["config"]["dt_ms"], "State identities differ")
    for key, (shape, dtype) in state_specs(identity).items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), f"Invalid {key}")
    require(np.all(arrays["brain.refractory_until"] >= 0) and np.all(arrays["drive.event_counts"] >= 0)
            and np.all(arrays["drive.event_counts"] <= step) and np.all((arrays["drive.phase"] >= 0) & (arrays["drive.phase"] < 1)),
            "State ranges differ")
    return checkpoint_state_digest(metadata, arrays)


def validate_state_digest(value, identity, frame):
    bound(value, "state_id")
    step = identity["precondition_steps"] + frame * identity["frame_steps"]
    require(value["policy_id"] == identity["policy_id"] and value["model_id"] == identity["model"]["model_id"]
            and value["drive_id"] == identity["drive"]["drive_id"] and value["spatial_map_id"] == identity["spatial"]["spatial_map_id"]
            and value["brain_step"] == value["drive_step"] == step and value["frame_index"] == frame
            and value["drive_dt_ms"] == identity["drive"]["config"]["dt_ms"], "State digest clocks/identity differ")
    specs = state_specs(identity)
    expected = {k.split(".")[1]: v for k, v in specs.items() if k.startswith("brain.") and k != "brain.pending"}
    expected.update(drive_phase=specs["drive.phase"], drive_event_counts=specs["drive.event_counts"])
    expected.update({f"pending_{i}": (specs["brain.v"][0], np.float32) for i in range(specs["brain.pending"][0][0])})
    require(set(value["values"]) == set(expected), "State digest inventory differs")
    for key, (shape, dtype) in expected.items():
        item = value["values"][key]
        require(set(item) == {"shape", "dtype", "sha256"} and item["shape"] == list(shape) and item["dtype"] == np.dtype(dtype).str
                and isinstance(item["sha256"], str) and len(item["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in item["sha256"]), "State digest declaration differs")


def parameter_shapes(identity):
    return {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
            "readout_weight": (3, identity["features"]["width"]), "readout_bias": (3,)}


def validate_payload(info, arrays, run):
    bound(info, "checkpoint_id")
    require(set(info) == {"schema", "training_run_id", "policy_id", "context", "optimizer", "state", "arrays",
                          "implementation_sha256", "runtime", "arrays_sha256", "checkpoint_id"}, "Checkpoint fields differ")
    require(info["schema"] == SCHEMA and info["training_run_id"] == run["training_run_id"]
            and info["policy_id"] == run["policy"]["policy_id"] and info["optimizer"] == run["optimizer"]
            and info["implementation_sha256"] == run["implementation_sha256"] and info["runtime"] == run["runtime"],
            "Checkpoint sources/runtime/optimizer differ")
    context = info["context"]
    require(set(context) == {*expected_ids(run), "scenes", "epoch", "scene_index", "scene_id", "next_frame", "completed_windows",
                             "optimizer_steps", "parent_checkpoint_id", "log_prefix_sha256"}, "Checkpoint context differs")
    completed, steps = context["completed_windows"], context["optimizer_steps"]
    require(type(steps) is int and type(completed) is int and 0 <= completed == steps <= run["protocol"]["windows"], "Invalid cursor")
    require(context["scenes"] == run["scenes"] and all(context[k] == v for k, v in expected_ids(run).items())
            and all(context[k] == v for k, v in cursor(run["scenes"], completed, steps).items()), "Checkpoint run/cursor differs")
    for key in ("parent_checkpoint_id", "log_prefix_sha256"):
        require(isinstance(context[key], str) and len(context[key]) == 64 and all(c in "0123456789abcdef" for c in context[key]),
                "Invalid parent or prefix")
    shapes = parameter_shapes(run["policy"])
    specs = state_specs(run["policy"])
    specs.update({f"parameter.{k}": (shape, np.float32) for k, shape in shapes.items()})
    specs.update({f"adam.{PARAMETERS[k]}.{f}": (() if f == "step" else shape, np.float32)
                  for k, shape in shapes.items() for f in ADAM_FIELDS})
    require(set(arrays) == set(specs) and info["arrays"] == declarations(arrays), "Archive inventory differs")
    for key, (shape, dtype) in specs.items():
        require(arrays[key].shape == shape and arrays[key].dtype == dtype and np.isfinite(arrays[key]).all(), f"Invalid array {key}")
    for name in PARAMETERS.values():
        require(float(arrays[f"adam.{name}.step"]) == steps and np.all(arrays[f"adam.{name}.exp_avg_sq"] >= 0), "Adam state differs")
    if steps == 0:
        require(all(np.count_nonzero(v) == 0 for k, v in arrays.items() if k.startswith("adam.")), "Genesis moments must reset")
    digest = validate_state(info["state"], arrays, run["policy"], context["next_frame"])
    return {"checkpoint_id": info["checkpoint_id"], "context": context, "state": digest,
            "parameters": {k: arrays[f"parameter.{k}"] for k in PARAMETERS},
            "moments": {f"{n}.{f}": arrays[f"adam.{n}.{f}"] for n in PARAMETERS.values() for f in ADAM_FIELDS}}


def write_checkpoint(directory, run, context, metadata, arrays):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "arrays.npz", **arrays)
    info = {"schema": SCHEMA, "training_run_id": run["training_run_id"], "policy_id": run["policy"]["policy_id"],
            "context": context, "optimizer": run["optimizer"], "state": metadata, "arrays": declarations(arrays),
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
    require(sha256(directory / "arrays.npz") == info["arrays_sha256"], "Checkpoint checksum differs")
    arrays = read_arrays(directory / "arrays.npz")
    return {**validate_payload(info, arrays, run), "path": str(directory), "metadata": info, "arrays": arrays}


def verify_resume(directory, registration, previous, run):
    info, arrays = read_json(directory / "resume.json"), read_arrays(directory / "resume.npz")
    require(set(info) == {"job_id", "parent_checkpoint_id", "state", "arrays_sha256"}
            and info["job_id"] == registration["job_id"] and info["parent_checkpoint_id"] == previous["checkpoint_id"]
            and info["state"] == previous["metadata"]["state"] and sha256(directory / "resume.npz") == info["arrays_sha256"],
            "Restore identity differs")
    require(set(arrays) == set(previous["arrays"]) and all(v.dtype == arrays[k].dtype and np.array_equal(v, arrays[k])
            for k, v in previous["arrays"].items()), "Actual GPU restoration changed array values")
    return validate_state(info["state"], arrays, run["policy"], previous["context"]["next_frame"])
