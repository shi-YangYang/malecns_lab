"""Torch-free registration for CPU-prefix / MPS continuation runs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ..brain.metadata import canonical_hash
from ..brain.mps_identity import ACTIVATION_CHECKPOINT, policy_identity
from .training_contract import (SOURCE_FILES, PROTOCOL, PILOT, SCENES, bound, bytes_sha, read_arrays, read_json,
                               require, runtime as cpu_runtime, sha256, validate_run as validate_cpu_run)
from .training_evidence import scan_run as scan_cpu_run
from .training_archive import STATE_KEYS, checkpoint_state_digest

SCHEMA = "cnsbridge.mps_continuation.v1"
NEW_SOURCES = ("brain/mps_identity.py", "brain/mps_trainable.py", "learning/mps_policy.py", "learning/mps_optimizer.py",
               "learning/mps_archive.py", "learning/mps_contract.py", "learning/mps_evidence.py", "learning/mps_state.py",
               "learning/mps_window.py", "learning/mps_memory.py", "learning/mps_worker.py", "learning/train_mps.py",
               "learning/mps_recompute.py")
PROTOCOL_MPS = {**PROTOCOL, "schema": SCHEMA, "backend": "mps_forward_backward_cpu_adam",
                "policy_id": "derived_from_actual_cpu_source_and_mps_sources",
                "activation_checkpoint": ACTIVATION_CHECKPOINT,
                "numerics": "Decision014: accept disclosed CPU/MPS differences; retain own finite/clip/Adam checks",
                "migration": "exact parameter/Adam/recurrent state transfer; inherit CPU completed-window budget"}
NUMERIC_PRECEDENT = {"decision": ".ai/decisions/014-resume-on-mps-with-recorded-numerics.md",
                     "report": "specs/spec-008-trainable-snn/mps-fullgraph-run-20261019-r1.md", "old_numeric_pass": False}


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {name: sha256(package / name) for name in (*SOURCE_FILES, *NEW_SOURCES)}


def runtime():
    return {**cpu_runtime(), "device": "mps", "optimizer_device": "cpu", "mps_fallback": "0"}


def read_parent(path):
    directory = Path(path).resolve()
    run = read_json(directory / "run.json")
    require(run["mode"] == "joint" and run["policy"]["background"] == "fixed_lamina_8mv", "Only existing joint/fixed_lamina_8mv may migrate")
    data = validate_cpu_run(run)
    chain = scan_cpu_run(directory, run, data)
    require(chain["last"] is not None and 0 < chain["completed_windows"] < 160, "A completed, nonterminal CPU checkpoint is required")
    last = chain["last"]
    checkpoint = Path(last["path"])
    info = read_json(checkpoint / "checkpoint.json")
    states = read_arrays(checkpoint / "training.npz")
    arrays = {**{k: states[k] for k in STATE_KEYS}, **{f"parameter.{k}": v for k, v in last["parameters"].items()},
              **{f"adam.{k}": v for k, v in last["moments"].items()}}
    anchor = {"path": str(directory), "run_sha256": sha256(directory / "run.json"), "training_run_id": run["training_run_id"],
              "checkpoint_id": last["checkpoint_id"], "checkpoint_path": str(checkpoint),
              "checkpoint_files_sha256": {name: sha256(checkpoint / name) for name in
                                         ("checkpoint.json", "training.npz", "policy/checkpoint.json", "policy/parameters.npz")},
              "prefix_sha256": bytes_sha(chain["prefix"]), "context": last["context"]}
    return run, data, anchor, {**last, "metadata": info, "arrays": arrays}


def register_run(cpu_path):
    old, _, anchor, _ = read_parent(cpu_path)
    value = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(), "mode": "joint",
             "protocol": PROTOCOL_MPS, "protocol_id": canonical_hash(PROTOCOL_MPS), "scenes": SCENES,
             "graph": old["graph"], "dataset_path": old["dataset_path"], "dataset_id": old["dataset_id"],
             "dataset_manifest_sha256": old["dataset_manifest_sha256"], "cpu_parent": anchor,
             "policy": policy_identity(old["policy"]), "implementation_sha256": source_hashes(), "runtime": runtime(),
             "numeric_precedent": NUMERIC_PRECEDENT}
    value["training_run_id"] = canonical_hash(value)
    return value


def initial_prefix(run):
    return (json.dumps({"cpu_parent_run_id": run["cpu_parent"]["training_run_id"],
                        "cpu_prefix_sha256": run["cpu_parent"]["prefix_sha256"],
                        "cpu_checkpoint_id": run["cpu_parent"]["checkpoint_id"]}, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_run(run):
    bound(run, "training_run_id")
    require(set(run) == {"schema", "registered_utc", "mode", "protocol", "protocol_id", "scenes", "graph", "dataset_path",
                         "dataset_id", "dataset_manifest_sha256", "cpu_parent", "policy", "implementation_sha256", "runtime",
                         "numeric_precedent", "training_run_id"}, "Run fields differ")
    require(run["schema"] == SCHEMA and run["mode"] == "joint" and run["protocol"] == PROTOCOL_MPS
            and run["protocol_id"] == canonical_hash(PROTOCOL_MPS) and run["scenes"] == SCENES
            and run["numeric_precedent"] == NUMERIC_PRECEDENT, "MPS protocol/mode differs")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0" and run["runtime"] == runtime()
            and run["implementation_sha256"] == source_hashes(), "Runtime/fallback/sources differ")
    old, data, anchor, previous = read_parent(run["cpu_parent"]["path"])
    require(anchor == run["cpu_parent"] and run["policy"] == policy_identity(old["policy"]), "CPU parent or derived MPS identity differs")
    require(all(run[k] == old[k] for k in ("graph", "dataset_path", "dataset_id", "dataset_manifest_sha256")), "Graph/dataset differs")
    require(run["policy"]["model"]["graph"]["role"] == "full_reference", "Numerical fixture excluded")
    # Only the identity labels change; all inherited array values are preserved.
    adapted = {**previous["metadata"]["state"], "policy_id": run["policy"]["policy_id"],
               "brain": {**previous["metadata"]["state"]["brain"], "model_id": run["policy"]["model"]["model_id"]}}
    previous["state"] = checkpoint_state_digest(adapted, previous["arrays"])
    return data, previous
