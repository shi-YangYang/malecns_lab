"""Frozen continuous-training registration; never imports Torch or a full graph."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from ..brain.metadata import canonical_hash
from .joint_probe_contract import (PROTOCOL as PILOT, PARAMETERS, ADAM_FIELDS, array_digest, bound,
                                   graph_registration, require, sha256)
from .joint_probe_identity import validate_policy_identity

SCHEMA = "cnsbridge.continuous_training.v1"
MODES = ("joint", "frozen_brain")
IDENTITIES = ("training_run_id", "protocol_id", "dataset_id", "dataset_manifest_sha256")
SCENES = [{"scene_id": f"train-sine-{name}", "frame_count": 80} for name in ("combined", "forward", "vertical", "yaw")]
POLICY_ID = "2e2945c5c9925e7d05698912a034bb34a815ae2e52e295317ad7732d1d910f09"
PROTOCOL = {
    "schema": SCHEMA, "epochs": 1, "windows": 160, "frames_per_window": 2, "scenes": SCENES,
    "dataset_id": PILOT["dataset_id"], "dataset_manifest_sha256": PILOT["dataset_manifest_sha256"],
    "policy_id": POLICY_ID, "order": "declared_scenes_ascending_frames_no_shuffle",
    "state": "two_frame_tbptt; retain_state_after_update; reset_only_at_scene_boundary",
    "target": "same_frame_teacher_target_float32; valid_mask; all_invalid_skips_entire_update",
    "loss": "mean(((actions-targets)/[.3,.2,.5])**2)+1e-4*mean((gain-1)**2)",
    "clip": PILOT["clipping"], "learning_rates": {"theta": .001, "readout": .0001},
    "adam": {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0, "amsgrad": False,
             "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
             "fused": False, "decoupled_weight_decay": False},
    "adam_check": {"moment_rtol": 1e-6, "moment_atol": float(np.nextafter(np.float32(0), np.float32(1))),
                   "parameter_rtol": 5e-6, "parameter_atol": 1e-9,
                   "lerp": "float32_delta; float64_multiply_add_then_float32; actual_old_moments_each_window"},
    "training_operators": "deterministic; no_dropout_noise_scheduler", "role": "offline_imitation_not_closed_loop",
}
R2_SHA = {
    "registration.json": "e54f7c95a270ff2577448cceb3e9abef122dbf8d9821dc2e3e6091e673674582",
    "result.json": "a62ca9c55323e3f30e02a43f55d6663f16784477485b820bc84237121d4775a1",
    "policy_identity.json": "573e3e5f96f76b11e290629924026d5dd54f594a460020d90df9deccc34c65d8",
    "initial_parameters.npz": "bf36ff1ccd15918b076481f7e2935b7c875df304719378361ceef11d0d12ef75",
    "baseline.npz": "2133633c08800152e6a20dc3ebb9ac32b8946276b3d293854eec1b9248eb4240",
    "gradients_before_clip.npz": "26b2eb26f2a02c398dc7bc4cc61529e6b5904839f25d2d58b29f92fb631540e6",
    "update.npz": "fb3905ca1e1a24c75efd328a64039c6ffeeba1fa1a68a780ec7f91555b2f17fb",
    "adam_state.npz": "ed7ad30b8fb8866354e2b29e7e5da6d833a1d8d7e31bad8c44836440486b1d95",
    "selection.npz": "1c4f14ba2a98be54ad19200dac2364df043972076e44b966e043f48e5ef6cf11",
}
SOURCE_FILES = (
    "learning/train_snn.py", "learning/training_contract.py", "learning/training_math.py",
    "learning/training_archive.py", "learning/training_evidence.py", "learning/training_window.py", "learning/training_worker.py",
    "learning/training_checkpoint.py", "learning/training_optimizer.py", "learning/policy_checkpoint.py",
    "learning/joint_probe_contract.py", "learning/joint_probe_identity.py", "learning/joint_probe_worker.py",
    "learning/resource_guard.py", "learning/neural_policy.py", "learning/neural_data.py", "learning/follow_task.py",
    "brain/trainable.py", "brain/trainable_graph.py", "brain/tonic.py", "brain/policy_groups.py",
    "brain/reference.py", "brain/connectome.py", "brain/metadata.py", "senses/spatial_drive.py", "senses/target.py",
    "drones/follow_env.py", "drones/follow_scenes.py", "control/body.py",
)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)
    require(isinstance(value, dict), "JSON object required")
    return value


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), "Duplicate NPZ fields")
        return {key: data[key].copy() for key in data.files}


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {key: sha256(package / key) for key in SOURCE_FILES}


def runtime():
    # importlib metadata does not import Torch in the parent.
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": version("scipy"),
            "torch": version("torch"), "machine": platform.machine(), "system": platform.system(),
            "num_threads": 4, "num_interop_threads": 1, "device": "cpu", "dtype": "float32"}


def expected_ids(run):
    return {key: run[key] for key in IDENTITIES}


def optimizer_contract(mode):
    require(mode in MODES, "Unknown mode")
    groups = []
    if mode == "joint":
        groups.append({"parameters": ["brain.theta"], "lr": .001, **PROTOCOL["adam"]})
    groups.append({"parameters": ["readout.weight", "readout.bias"], "lr": .0001, **PROTOCOL["adam"]})
    return {"type": "torch.optim.Adam", "mode": mode, "groups": groups}


def active_keys(mode):
    require(mode in MODES, "Unknown mode")
    return tuple(PARAMETERS) if mode == "joint" else ("readout_weight", "readout_bias")


def dataset_for_run(path):
    from .neural_data import read_neural_data
    path = Path(path)
    require(sha256(path / "manifest.json") == PROTOCOL["dataset_manifest_sha256"], "Sealed dataset changed")
    data = read_neural_data(path)
    require(data.dataset_id == PROTOCOL["dataset_id"], "Dataset identity mismatch")
    scenes = [{"scene_id": seq.scene["scene_id"], "frame_count": len(seq.frames)} for seq in data.sequences]
    require(scenes == SCENES and all(seq.valid.all() for seq in data.sequences), "Registered 320 valid train frames required")
    require(data.identity["camera"] == PILOT["camera"] and data.identity["dt_s"] == .1, "Camera/time mismatch")
    return data


def history(path, implementation, graph):
    path = Path(path)
    for name, digest in R2_SHA.items():
        require(sha256(path / name) == digest, f"r2 anchor changed: {name}")
    old, final = read_json(path / "registration.json"), read_json(path / "result.json")
    bound(old, "registration_id")
    require(old["registration_id"] == "7ed634ceda4f6e495848fcf6ef7b1567691de3c14ef5aad29a2ca6945152be07"
            and old["protocol_id"] == "b4c292aa34d649c3d2d8d1cbe68154400ac0953239cfd64872feb795a2f85c9b",
            "Wrong r2 identities")
    require(final["status"] == "completed" and final["protocol_complete"] is True, "r2 was not completed")
    for name, digest in final["terminal_artifacts_sha256"].items():
        require(name in {"registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json", "supervisor_result.json"},
                "Unexpected historical terminal path")
        require(sha256(path / name) == digest, "r2 terminal artifact changed")
    worker = read_json(path / "worker_result.json")
    for name, digest in worker["artifacts_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid r2 path")
        require(sha256(path / name) == digest, "r2 worker artifact changed")
    require(all(graph[k] == old["graph"][k] for k in ("file_sha256", "source_id", "provenance")), "Graph differs from r2")
    for name, digest in old["implementation_sha256"].items():
        if name in implementation:
            require(implementation[name] == digest, f"Bound r2 source changed: {name}")
    policy = read_json(path / "policy_identity.json")
    require(policy["policy_id"] == POLICY_ID, "Wrong r2 policy")
    validate_policy_identity(policy, {"graph": graph, "implementation_sha256": implementation})
    return {"path": str(path.resolve()), "files_sha256": dict(R2_SHA)}, policy


def register_run(graph, dataset, mode):
    require(mode in MODES, "Unknown training mode")
    data = dataset_for_run(dataset)
    graph_info, sources = graph_registration(graph), source_hashes()
    prior, policy = history(Path(dataset).resolve().parent / "joint_update_20261019_r2", sources, graph_info)
    run = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
           "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "scenes": SCENES,
           "graph": graph_info, "dataset_path": str(Path(dataset).resolve()), "dataset_id": data.dataset_id,
           "dataset_manifest_sha256": PROTOCOL["dataset_manifest_sha256"], "policy": policy,
           "implementation_sha256": sources, "runtime": runtime(), "prior_run": prior}
    run["training_run_id"] = canonical_hash(run)
    return run


def validate_run(run):
    bound(run, "training_run_id")
    require(run["schema"] == SCHEMA and run["mode"] in MODES and run["protocol"] == PROTOCOL
            and run["protocol_id"] == canonical_hash(PROTOCOL) and run["scenes"] == SCENES,
            "Training run protocol differs")
    require(run["implementation_sha256"] == source_hashes() and run["runtime"] == runtime(), "Run source/runtime changed")
    require(run["dataset_id"] == PROTOCOL["dataset_id"]
            and run["dataset_manifest_sha256"] == PROTOCOL["dataset_manifest_sha256"], "Run dataset differs")
    require(graph_registration(run["graph"]["path"]) == run["graph"], "Run graph changed")
    prior, policy = history(Path(run["dataset_path"]).parent / "joint_update_20261019_r2", run["implementation_sha256"], run["graph"])
    require(run["prior_run"] == prior and run["policy"] == policy, "Run history/policy differs")
    return dataset_for_run(run["dataset_path"])


def cursor(scenes, completed, steps):
    require(type(completed) is int and 0 <= completed <= sum(s["frame_count"] for s in scenes) // 2, "Window count invalid")
    remaining = completed * 2
    for index, scene in enumerate(scenes):
        if remaining <= scene["frame_count"]:
            return {"epoch": 0, "scene_index": index, "scene_id": scene["scene_id"], "next_frame": remaining,
                    "completed_windows": completed, "optimizer_steps": steps}
        remaining -= scene["frame_count"]
    raise ValueError("Cursor outside epoch")


def next_selection(data, completed):
    remaining = completed * 2
    for index, seq in enumerate(data.sequences):
        if remaining < len(seq.frames):
            chosen = {key: getattr(seq, key)[remaining:remaining + 2].copy()
                      for key in ("frames", "time_s", "valid", "teacher_target")}
            chosen["indices"] = np.arange(remaining, remaining + 2, dtype=np.int64)
            return index, chosen
        remaining -= len(seq.frames)
    raise ValueError("Epoch already complete")


def prefix_line(segment, window, digest):
    return (json.dumps({"window": window, "event": f"segments/{segment:06d}/windows/{window:06d}/event.json",
                        "sha256": digest}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()
