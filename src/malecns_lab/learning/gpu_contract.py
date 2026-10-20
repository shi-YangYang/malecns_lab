"""Torch-free identities and immutable parent admission for all-GPU recovery."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from ..brain.metadata import canonical_hash
from ..brain.mps_identity import policy_identity as prior_policy_identity
from .training_contract import (SCENES, PARAMETERS, ADAM_FIELDS, bound, bytes_sha, read_json, require,
                                runtime as cpu_runtime, sha256)
from .mps_contract import source_hashes as prior_sources, validate_run as validate_prior
from .mps_evidence import scan_run as scan_prior

SCHEMA = "cnsbridge.gpu_recovery.v1"
PARENT_RUN_ID = "b7807f8e1997da44e46e83f39155e21bdce84aba020314a1ed68a145851413b7"
PARENT_CHECKPOINT_ID = "fd3019989892f86bac44faefd5230ffff91891dbde9e813956c0945eb7b1e8b1"
PARENT_PREFIX_SHA = "1964b500ead62bceec06bf763b75c3e4992433bebb8b66f0d2fdea75d00c13a0"
NEW_SOURCES = ("senses/gpu_drive.py", "learning/gpu_optimizer.py", "learning/gpu_policy.py",
               "learning/gpu_recompute.py", "learning/gpu_state.py", "learning/gpu_contract.py",
               "learning/gpu_archive.py", "learning/gpu_numbers.py", "learning/gpu_evidence.py",
               "learning/gpu_window.py", "learning/gpu_worker.py", "learning/train_gpu.py")
POLICY_SOURCES = ("senses/gpu_drive.py", "learning/gpu_policy.py", "learning/gpu_recompute.py", "learning/gpu_contract.py")
RTOL, ATOL = 2e-5, 1e-8


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**prior_sources(), **{name: sha256(root / name) for name in NEW_SOURCES}}


def runtime():
    return {**cpu_runtime(), "device": "mps", "backend": "all_gpu_sensory_brain_loss_clip_fused_adam", "fallback": "0"}


def optimizer_identity():
    value = {"schema": "cnsbridge.gpu_adam.v1", "backend": "torch_mps_fused_adam",
             "groups": [{"parameters": ["brain.theta"], "lr": .001},
                        {"parameters": ["readout.weight", "readout.bias"], "lr": .0001}],
             "options": {"betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
                         "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
                         "fused": True, "decoupled_weight_decay": False},
             "max_group_norm": 1., "clip_eps": 1e-6,
             "clipping": "scale=max(maxabs,1); u=g/scale; q=sqrt(sum(u^2)); post=u*min(scale,1/(q+eps/scale))",
             "device": "mps", "dtype": "float32", "torch_version": runtime()["torch"],
             "implementation_sha256": sha256(Path(__file__).with_name("gpu_optimizer.py"))}
    return {**value, "optimizer_id": canonical_hash(value)}


def policy_identity(cpu, horizon):
    require(type(horizon) is int and horizon in (100, 250, 500, 1000), "Explicit supported credit horizon required")
    value = prior_policy_identity(cpu)
    value.pop("policy_id")
    drive = {"schema": "cnsbridge.gpu_spatial_drive.v1", "backend": "torch_mps_float32",
             "spatial_map_id": cpu["spatial"]["spatial_map_id"], "config": deepcopy(cpu["drive"]["config"]),
             "camera": deepcopy(cpu["camera"]), "phase_dtype": "float32", "event_count_dtype": "int64", "shape": [12, 16],
             "initialization": "PCG64(phase_seed) float64 once, cast float32, clamp maximum to nextbelow(1)",
             "pixel_rule": "uint8 BGR; red>=70; 100*red>165*green; 100*red>165*blue; red-green>=35",
             "pooling": "floor linspace nonoverlapping cells; float32 red count divided by cell pixel count",
             "event_rule": ("float32 phase + (occupancy*max_rate_hz)*(dt_ms/1000); emit>=1; subtract event; "
                            "clamp rounded residual to nextbelow(1)"),
             "non_input_drive": "exact zero", "background_drive": "none",
             "numpy_initialization_version": runtime()["numpy"], "torch_version": runtime()["torch"],
             "implementation_sha256": sha256(Path(__file__).resolve().parents[1] / "senses/gpu_drive.py")}
    value.update(schema="cnsbridge.gpu_recovery_policy.v1", backend=runtime()["backend"],
                 drive={**drive, "drive_id": canonical_hash(drive)}, credit_horizon_steps=horizon,
                 activation_checkpoint={"neural_steps_per_block": 50, "use_reentrant": False, "preserve_rng_state": False,
                    "determinism_check": "default", "early_stop": True,
                    "sensory_replay": "pure MPS float32 phase/int64 counts passed as block tensors",
                    "gradient_boundary": "detach v/g/refractory/all pending at credit horizon; retain DN running sums",
                    "dn_accumulation": "original ordered per-step running sums"})
    root = Path(__file__).resolve().parents[1]
    value["implementation_sha256"].update({name: sha256(root / name) for name in POLICY_SOURCES})
    return {**value, "policy_id": canonical_hash(value)}


def protocol(horizon, theta_norm_limit):
    require(type(horizon) is int and horizon in (100, 250, 500, 1000), "Explicit credit horizon required")
    require(type(theta_norm_limit) in (float, int) and 0 < theta_norm_limit < 1e30, "Explicit finite theta norm gate required")
    return {"schema": SCHEMA, "windows": 160, "frames_per_window": 2, "scenes": SCENES,
            "credit_horizon_steps": horizon, "theta_norm_limit": float(theta_norm_limit),
            "loss": "mean(softplus(2*z)-(1+y)*z)+1e-4*mean((gain-1)^2); y=teacher/scales",
            "report_action_mse": True, "migration": "inherit old160 theta/readout exactly; reset Adam m/v/step to zero",
            "scene": "new epoch; initial 600 blank steps; scene reset; keep detached post-forward state after update",
            "gate": "old160 negative frames from new genesis state; discard update; fixed-feature vx and total logit loss improve",
            "adam_oracle": {"rtol": RTOL, "atol": ATOL}, "maximum_segment_windows": 4,
            "warm_start": "first train segment one window; second new-process segment one window",
            "role": "offline_imitation_not_closed_loop"}


def read_parent(path):
    path = Path(path).resolve()
    old = read_json(path / "run.json")
    require(old["training_run_id"] == PARENT_RUN_ID, "Only the completed registered GPU160 parent is admitted")
    data, initial = validate_prior(old)
    chain = scan_prior(path, old, data, initial)
    last = chain["last"]
    require(chain["completed_windows"] == 160 and last["checkpoint_id"] == PARENT_CHECKPOINT_ID
            and bytes_sha(chain["prefix"]) == PARENT_PREFIX_SHA, "Parent160 chain changed")
    checkpoint = Path(last["path"])
    anchor = {"path": str(path), "training_run_id": old["training_run_id"], "checkpoint_id": last["checkpoint_id"],
              "checkpoint_path": str(checkpoint), "prefix_sha256": bytes_sha(chain["prefix"]),
              "checkpoint_files_sha256": {name: sha256(checkpoint / name) for name in ("checkpoint.json", "arrays.npz")}}
    cpu = read_json(Path(old["cpu_parent"]["path"]) / "run.json")["policy"]
    return old, data, anchor, last, cpu


def register_run(parent, horizon, theta_norm_limit):
    require(horizon == 100 and theta_norm_limit == 1000, "Frozen recovery mathematics differ")
    old, _, anchor, _, cpu = read_parent(parent)
    math = protocol(horizon, theta_norm_limit)
    value = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(), "mode": "joint",
             "protocol": math, "protocol_id": canonical_hash(math), "scenes": SCENES,
             **{k: old[k] for k in ("graph", "dataset_path", "dataset_id", "dataset_manifest_sha256")},
             "parent": anchor, "policy": policy_identity(cpu, horizon), "optimizer": optimizer_identity(),
             "implementation_sha256": source_hashes(), "runtime": runtime()}
    return {**value, "training_run_id": canonical_hash(value)}


def validate_run(run):
    bound(run, "training_run_id")
    require(set(run) == {"schema", "registered_utc", "mode", "protocol", "protocol_id", "scenes", "graph", "dataset_path",
                         "dataset_id", "dataset_manifest_sha256", "parent", "policy", "optimizer", "implementation_sha256",
                         "runtime", "training_run_id"}, "Run fields differ")
    p = run["protocol"]
    require(p["credit_horizon_steps"] == 100 and p["theta_norm_limit"] == 1000, "Frozen recovery mathematics differ")
    require(run["schema"] == SCHEMA and run["mode"] == "joint" and p == protocol(p["credit_horizon_steps"], p["theta_norm_limit"])
            and run["protocol_id"] == canonical_hash(p) and run["scenes"] == SCENES, "GPU recovery protocol differs")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0" and run["runtime"] == runtime()
            and run["implementation_sha256"] == source_hashes(), "GPU source/runtime/fallback differs")
    old, data, anchor, last, cpu = read_parent(run["parent"]["path"])
    require(anchor == run["parent"] and run["policy"] == policy_identity(cpu, p["credit_horizon_steps"])
            and run["optimizer"] == optimizer_identity(), "Parent or declared device identities differ")
    require(all(run[k] == old[k] for k in ("graph", "dataset_path", "dataset_id", "dataset_manifest_sha256")), "Graph/data differs")
    require(run["policy"]["model"]["graph"]["role"] == "full_reference", "Numerical fixtures excluded")
    return data, last, cpu


def initial_prefix(run, genesis):
    return (json.dumps({"parent_checkpoint_id": run["parent"]["checkpoint_id"], "parent_prefix_sha256": run["parent"]["prefix_sha256"],
                       "genesis_checkpoint_id": genesis["checkpoint_id"]}, sort_keys=True, separators=(",", ":")) + "\n").encode()
