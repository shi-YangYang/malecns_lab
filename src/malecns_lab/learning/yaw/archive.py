"""Hash-bound NPZ checkpoints; no pickle, source relabeling, or partial success."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from ...brain.metadata import canonical_hash
from ..resource_guard import write_json
from .protocol import PARAMETERS, read_json, require, sha256


def write_arrays(path, arrays):
    values = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        require(value.dtype.kind in "biufUS" and (value.dtype.kind not in "f" or np.isfinite(value).all()), "Invalid archive array")
        values[name] = value
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    require(all(v.dtype.kind in "biufUS" and (v.dtype.kind != "f" or np.isfinite(v).all()) for v in result.values()), "Invalid archive array")
    return result


def write_checkpoint(path, *, run, policy, optimizer, state, environment, completed_windows, scene_index, parent_id):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    parameters = policy.parameters_numpy()
    require(set(parameters) == set(PARAMETERS), "Parameter inventory differs")
    snapshot = optimizer.snapshot()
    arrays = {**{"parameter."+k: v for k,v in parameters.items()},
              **{"state."+k: v for k,v in policy.state_arrays(state).items()},
              **{"adam."+k: v for k,v in snapshot["moments"].items()}}
    write_arrays(path / "arrays.npz", arrays)
    metadata = {"schema": "malecns_lab.yaw_checkpoint.v1", "run_id": run["run_id"], "family_id": run["family_id"],
                "mode": run["mode"], "completed_windows": completed_windows, "scene_index": scene_index,
                "parent_checkpoint_id": parent_id, "policy": policy.identity, "environment": environment.snapshot(),
                "optimizer": {k:v for k,v in snapshot.items() if k != "moments"}, "arrays_sha256": sha256(path / "arrays.npz")}
    metadata["checkpoint_id"] = canonical_hash(metadata)
    write_json(path / "checkpoint.json", metadata)
    return metadata


def read_checkpoint(path, run, *, load_arrays=True):
    path = Path(path)
    meta = read_json(path / "checkpoint.json")
    claimed = dict(meta)
    require(claimed.pop("checkpoint_id", None) == canonical_hash(claimed), "Checkpoint metadata changed")
    require(meta["schema"] == "malecns_lab.yaw_checkpoint.v1" and meta["run_id"] == run["run_id"]
            and meta["family_id"] == run["family_id"] and meta["mode"] == run["mode"], "Foreign checkpoint")
    require(meta["arrays_sha256"] == sha256(path / "arrays.npz"), "Checkpoint arrays changed")
    if not load_arrays:
        return {"path": str(path.resolve()), "metadata": meta}
    arrays = read_arrays(path / "arrays.npz")
    parts = {}
    for prefix in ("parameter", "state", "adam"):
        parts[prefix] = {k[len(prefix)+1:]: v for k,v in arrays.items() if k.startswith(prefix+".")}
    require(sum(map(len, parts.values())) == len(arrays) and set(parts["parameter"]) == set(PARAMETERS), "Checkpoint array inventory differs")
    return {"path": str(path.resolve()), "metadata": meta, **parts}


def scan_checkpoints(directory, run):
    paths = sorted((Path(directory) / "checkpoints").glob("[0-9]*"))
    previous, records = None, []
    for index, path in enumerate(paths):
        require(path.name == f"{index:06d}", "Checkpoint sequence has a gap")
        record = read_checkpoint(path, run, load_arrays=False)
        meta = record["metadata"]
        require(meta["completed_windows"] == index and meta["optimizer"]["steps"] == index,
                "Window/optimizer cursor differs")
        require(meta["parent_checkpoint_id"] == previous, "Checkpoint parent chain differs")
        require(index <= run["protocol"]["windows"], "Checkpoint beyond registered budget")
        expected_scene = (index - 1) // 20 if index else 0
        expected_frame = 40 + ((index - 1) % 20 + 1)*2 if index else 0
        require(meta["scene_index"] == expected_scene and meta["environment"]["frame_index"] == expected_frame,
                "Checkpoint scene/environment cursor differs")
        if index:
            event_path = Path(directory) / "windows" / f"{index:06d}"
            event = read_json(event_path / "evidence.json")
            payload = dict(event)
            require(payload.pop("evidence_id", None) == canonical_hash(payload), "Window seal changed")
            require(event["window"] == index and event["checkpoint_id"] == meta["checkpoint_id"] and
                    event["parent_checkpoint_id"] == previous and event["run_id"] == run["run_id"], "Window/checkpoint chain differs")
            require(set(event["artifacts"]) == {"frames.npz"} and sha256(event_path / "frames.npz") == event["artifacts"]["frames.npz"],
                    "Window evidence changed")
        previous = meta["checkpoint_id"]
        records.append(record)
    if records:
        records[-1] = read_checkpoint(records[-1]["path"], run)
    return records


def restore_checkpoint(record, policy, optimizer_class, environment):
    meta = record["metadata"]
    require(policy.identity == meta["policy"], "Policy identity differs on restore")
    policy.load_parameters(record["parameter"])
    optimizer = optimizer_class(policy, mode=meta["mode"], snapshot={**meta["optimizer"], "moments": record["adam"]})
    state = policy.restore_state(record["state"])
    environment.restore(meta["environment"])
    require(state.frame_index == environment.frame_index, "Neural and environment frame clocks differ")
    require(all(np.array_equal(v, record["parameter"][k]) for k,v in policy.parameters_numpy().items()), "Parameter restore changed bytes")
    require(set(policy.state_arrays(state)) == set(record["state"]) and
            all(np.array_equal(v, record["state"][k]) for k,v in policy.state_arrays(state).items()), "State restore changed bytes")
    restored = optimizer.snapshot()
    require(restored["steps"] == meta["optimizer"]["steps"] and set(restored["moments"]) == set(record["adam"]) and
            all(np.array_equal(v, record["adam"][k]) for k,v in restored["moments"].items()), "Optimizer restore changed bytes")
    return state, optimizer
