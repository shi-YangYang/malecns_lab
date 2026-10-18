"""Frozen final-epoch admission and value identities for closed-loop evaluation.

The public evaluator has no numerical-fixture or partial-training fallback.
This module and every parent-side dependency are Torch-free.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from ..brain.metadata import canonical_hash
from ..drones.follow_scenes import scene_registry
from ..senses.target import TargetCamera
from .resource_guard import write_json
from . import training_contract as training
from .training_archive import inspect_checkpoint
from .training_evidence import scan_run, validate_segment_completion
from .training_contract import PARAMETERS, array_digest, bound, read_arrays, read_json, require, runtime, sha256

SCHEMA = "cnsbridge.closed_loop_evaluation.v1"
CONDITIONS = ("joint", "frozen_brain", "acute_restore", "direct_visual", "zero")
NEURAL = CONDITIONS[:3]
VAL = tuple(f"val-triangle-{axis}" for axis in ("forward", "yaw", "vertical", "combined"))
TEST = tuple(f"test-chirp-{axis}" for axis in ("forward", "yaw", "vertical", "combined", "occluded")) + ("test-excursion-out-of-view",)
SOURCES = (*training.SOURCE_FILES, "learning/evaluate_snn.py", "learning/evaluation_contract.py",
           "learning/evaluation_checkpoint.py", "learning/evaluation_worker.py", "learning/evaluation_evidence.py",
           "learning/evaluation_rollout.py")
PROTOCOL = {"schema": SCHEMA, "conditions": list(CONDITIONS), "val": list(VAL), "test": list(TEST),
            "scenes": [asdict(s) for s in scene_registry() if s.scene_id in VAL + TEST],
            "camera": asdict(TargetCamera()), "dt_s": .1, "response_tau_s": .25, "planned_frames": 80,
            "chunk_frames": 10, "maximum_segment_frames": 40, "final_training_windows": 160,
            "first_actual_gate": {"condition": "joint", "scene": VAL[0], "segments_frames": [1, 1]},
            "thresholds": {"position": .15, "distance_m": .25, "success_fraction": .8},
            "readout_check": {"rtol": 1e-6, "atol": 1e-7},
            "selection": "final_epoch_only; all_conditions_val_then_test; no_parameter_updates",
            "scope": "marked_target_velocity_simulation_not_real_flight_or_topology_contribution"}

PROTOCOL = json.loads(json.dumps(PROTOCOL, allow_nan=False))
TRAINING_TERMINAL_FILES = ("registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json",
                           "supervisor_result.json", "resume.json", "resume.npz")


def sources():
    package = Path(__file__).resolve().parents[1]
    return {name: sha256(package / name) for name in SOURCES}


def normalized(value):
    import json
    return json.loads(json.dumps(value, allow_nan=False))


def path_inside(root, relative):
    root, part = Path(root).resolve(), Path(relative)
    require(not part.is_absolute() and ".." not in part.parts, "Path escapes evidence directory")
    path = root / part
    require(path.resolve().is_relative_to(root), "Symlink escapes evidence directory")
    return path


def parameter_identity(identity, values):
    shapes = {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "readout_weight": (3, identity["features"]["width"]), "readout_bias": (3,)}
    require(set(values) == set(PARAMETERS), "Parameter inventory differs")
    for key, shape in shapes.items():
        value = values[key]
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), "Invalid effective parameters")
    payload = {"policy_id": identity["policy_id"], **{key + "_digest": array_digest(values[key]) for key in PARAMETERS}}
    return {**payload, "parameter_set_id": canonical_hash(payload)}


def training_terminals(directory, run, chain):
    """Admit finalized training evidence, not merely a recoverable window prefix."""
    origins = []
    require(chain["segments"], "Training has no finalized segments")
    for index, item in enumerate(chain["segments"]):
        segment = path_inside(directory, f"segments/{item['segment']:06d}")
        reg = read_json(path_inside(segment, "registration.json"))
        result = read_json(path_inside(segment, "result.json"))
        supervisor = read_json(path_inside(segment, "supervisor_result.json"))
        require(result.get("schema") == "cnsbridge.training_segment_result.v1"
                and result.get("training_run_id") == run["training_run_id"]
                and result.get("segment_id") == reg["segment_id"]
                and result.get("scope") == "full_reference_training", "Training parent identity differs")
        require(result.get("supervisor") == supervisor and supervisor.get("schema") == "cnsbridge.resource_run.v1",
                "Training raw supervisor differs")
        require(type(result.get("verified_end_windows")) is int and result["verified_end_windows"] == item["end_windows"]
                and not result.get("continuation_blocked_reason")
                and result.get("reason") != "invalid_segment_evidence", "Training parent did not verify its actual prefix")
        expected_prefix = None
        if item["end_windows"]:
            last = chain["windows"][item["end_windows"] - 1]
            expected_prefix = {"checkpoint_id": last["checkpoint_id"], "cursor": last["cursor"],
                "path": str(path_inside(directory, f"segments/{last['segment']:06d}/windows/{last['window']:06d}/checkpoint"))}
        require(result.get("recoverable_prefix") == expected_prefix, "Training parent prefix identity differs")
        files = {name for name in TRAINING_TERMINAL_FILES if path_inside(segment, name).exists()}
        require(set(result.get("terminal_artifacts_sha256", {})) == files, "Training terminal inventory differs")
        for name in files:
            require(sha256(path_inside(segment, name)) == result["terminal_artifacts_sha256"][name],
                    "Training terminal artifact changed")
        worker = read_json(segment / "worker_result.json") if "worker_result.json" in files else None
        if worker is not None:
            require(worker.get("schema") == "cnsbridge.training_worker.v1"
                    and worker.get("training_run_id") == run["training_run_id"] and worker.get("segment_id") == reg["segment_id"]
                    and worker.get("failure_kind") != "numerical_or_protocol", "Training worker identity or failure invalid")
            require((worker.get("status") == "completed" and not worker.get("failure_kind"))
                    or (worker.get("status") == "failed" and worker.get("failure_kind") == "io_or_interrupt"),
                    "Training worker has an unsupported terminal status")
            worker_files = {name for name in ("phase.json", "resume.json", "resume.npz") if name in files}
            require("phase.json" in worker_files and set(worker.get("artifacts_sha256", {})) == worker_files,
                    "Training worker terminal inventory differs")
            for name in worker_files:
                require(worker["artifacts_sha256"][name] == result["terminal_artifacts_sha256"][name],
                        "Training worker terminal hash differs")
        if result.get("status") == "completed":
            expected_files = set(TRAINING_TERMINAL_FILES) - ({"resume.json", "resume.npz"} if reg["parent_checkpoint_id"] is None else set())
            require(files == expected_files and result.get("reason") == "registered_windows_verified" and not item["unsealed_tail"],
                    "Completed training terminal is incomplete")
            validate_segment_completion(segment, reg, item["end_windows"] - item["start_windows"])
        else:
            reason = supervisor.get("reason", "")
            resource_failure = reason in ("wall_limit", "rss_limit", "system_reserve", "resource_preflight_failed")
            io_failure = (reason == "worker_exited" and worker is not None and worker.get("status") == "failed"
                          and worker.get("failure_kind") == "io_or_interrupt")
            supervisor_io = any(reason.startswith(f"supervisor_exception:{kind}:") for kind in
                                ("OSError", "IOError", "FileNotFoundError", "PermissionError", "InterruptedError", "KeyboardInterrupt"))
            require(result.get("status") == supervisor.get("status") == "failed" and result.get("reason") == reason
                    and (resource_failure or io_failure or supervisor_io), "Training failure is not a verified resource/IO prefix")
            require(index < len(chain["segments"]) - 1, "Final training segment must have a completed parent terminal")
        origins.append({"segment": item["segment"], "segment_id": reg["segment_id"], "status": result["status"],
                        "verified_end_windows": item["end_windows"], "result_sha256": sha256(segment / "result.json"),
                        "terminal_artifacts_sha256": result["terminal_artifacts_sha256"]})
    return origins


def final_training_source(directory, mode):
    directory = Path(directory).resolve()
    run = read_json(directory / "run.json")
    require(run["mode"] == mode, "Training mode mismatch")
    data = training.validate_run(run)
    chain = scan_run(directory, run, data)
    expected = training.cursor(run["scenes"], 160, 160)
    require(chain["completed_windows"] == chain["optimizer_steps"] == 160 and chain["last"] is not None,
            "Both independently completed 160-window training chains are required")
    last = chain["last"]
    require(all(last["context"][key] == value for key, value in expected.items()), "Final training cursor differs")
    terminals = training_terminals(directory, run, chain)
    # inspect_checkpoint is deliberately the Torch-free archive reader.
    saved = inspect_checkpoint(last["path"], run, last["context"])
    original = read_arrays(Path(run["prior_run"]["path"]) / "initial_parameters.npz")
    if mode == "frozen_brain":
        require(np.array_equal(saved["parameters"]["theta"], original["theta"]), "Frozen brain changed its original theta")
    nested = read_json(Path(last["path"]) / "policy/checkpoint.json")
    source = {"path": str(directory), "run_sha256": sha256(directory / "run.json"),
              "training_run_id": run["training_run_id"], "protocol_id": run["protocol_id"],
              "dataset_id": run["dataset_id"], "dataset_manifest_sha256": run["dataset_manifest_sha256"],
              "window_id": last["window_id"], "checkpoint_id": saved["checkpoint_id"],
              "policy_checkpoint_id": nested["checkpoint_id"], "checkpoint_path": last["path"],
              "finalized_segments": terminals,
              "files_sha256": {name: sha256(Path(last["path"]) / name) for name in
                                 ("checkpoint.json", "training.npz", "policy/checkpoint.json", "policy/parameters.npz")},
              "initial_parameters_sha256": sha256(Path(run["prior_run"]["path"]) / "initial_parameters.npz")}
    return source, run, saved["parameters"], {key: original[key] for key in PARAMETERS}


def admit_sources(joint_path, frozen_path):
    joint, jr, jp, initial = final_training_source(joint_path, "joint")
    frozen, fr, fp, fi = final_training_source(frozen_path, "frozen_brain")
    require(joint["training_run_id"] != frozen["training_run_id"] and Path(joint_path).resolve() != Path(frozen_path).resolve(),
            "Frozen training must be an independent run")
    for key in ("protocol_id", "graph", "dataset_id", "dataset_manifest_sha256", "policy", "runtime", "implementation_sha256", "prior_run"):
        require(jr[key] == fr[key], f"Training sources differ: {key}")
    require(all(np.array_equal(initial[key], fi[key]) for key in PARAMETERS), "Initial training arrays differ")
    values = {"joint": jp, "frozen_brain": fp,
              "acute_restore": {**jp, "theta": initial["theta"].copy()}}
    return {"joint": joint, "frozen_brain": frozen}, jr, values


def register_suite(joint_path, frozen_path, directory):
    directory = Path(directory).resolve()
    require(not directory.exists(), "Evaluation suite directory already exists")
    origins, training_run, values = admit_sources(joint_path, frozen_path)
    identities = {name: parameter_identity(training_run["policy"], value) for name, value in values.items()}
    run = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(),
           "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "training": origins,
           "graph": training_run["graph"], "policy": training_run["policy"],
           "parameters": identities, "implementation_sha256": sources(), "runtime": runtime()}
    run["evaluation_run_id"] = canonical_hash(run)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "run.json", run)
    for name in CONDITIONS:
        target = directory / "conditions" / name
        target.mkdir(parents=True)
        if name in NEURAL:
            np.savez(target / "parameters.npz", **values[name])
        payload = condition_payload(run, name, sha256(target / "parameters.npz") if name in NEURAL else None)
        write_json(target / "condition.json", payload)
    return run


def condition_payload(run, name, parameters_sha):
    require(name in CONDITIONS, "Unknown condition")
    origin = "frozen_brain" if name == "frozen_brain" else "joint"
    payload = {"schema": "cnsbridge.evaluation_condition.v1", "evaluation_run_id": run["evaluation_run_id"],
               "name": name, "parameter_identity": run["parameters"].get(name), "parameters_sha256": parameters_sha,
               "source": run["training"][origin] if name in NEURAL else
                         {"implementation_sha256": {key: run["implementation_sha256"][key] for key in
                                                   ("learning/follow_task.py", "senses/target.py", "control/body.py")}},
               "intervention": "restore_original_theta_before_initial_state" if name == "acute_restore" else None}
    return {**payload, "condition_id": canonical_hash(payload)}


def load_condition(directory, run, name):
    target = path_inside(directory, f"conditions/{name}")
    metadata = read_json(target / "condition.json")
    values = read_arrays(target / "parameters.npz") if name in NEURAL else None
    require(metadata == condition_payload(run, name, sha256(target / "parameters.npz") if values is not None else None),
            "Condition identity/files changed")
    if values is not None:
        require(parameter_identity(run["policy"], values) == metadata["parameter_identity"], "Effective parameter values changed")
    return metadata, values


def validate_suite(directory):
    directory = Path(directory).resolve()
    run = read_json(directory / "run.json")
    bound(run, "evaluation_run_id")
    require(run["schema"] == SCHEMA and run["protocol"] == PROTOCOL and run["protocol_id"] == canonical_hash(PROTOCOL),
            "Evaluation protocol differs")
    require(run["implementation_sha256"] == sources() and run["runtime"] == runtime(), "Evaluation source/runtime changed")
    origins, training_run, values = admit_sources(run["training"]["joint"]["path"], run["training"]["frozen_brain"]["path"])
    require(origins == run["training"] and training_run["policy"] == run["policy"] and training_run["graph"] == run["graph"],
            "Admitted training provenance changed")
    require(run["parameters"] == {name: parameter_identity(run["policy"], value) for name, value in values.items()},
            "Effective parameter derivation differs")
    for name in CONDITIONS:
        load_condition(directory, run, name)
    return run


def episode_payload(run, condition, scene_id):
    require(scene_id in VAL + TEST, "Unknown evaluation scene")
    scene = next(scene for scene in PROTOCOL["scenes"] if scene["scene_id"] == scene_id)
    payload = {"schema": "cnsbridge.evaluation_episode.v1", "evaluation_run_id": run["evaluation_run_id"],
               "protocol_id": run["protocol_id"], "condition_id": condition["condition_id"], "condition": condition["name"],
               "parameter_set_id": condition["parameter_identity"]["parameter_set_id"] if condition["parameter_identity"] else None,
               "scene_id": scene_id, "scene": normalized(scene), "scene_spec_id": canonical_hash(scene), "planned_frames": 80}
    return {**payload, "episode_id": canonical_hash(payload)}
