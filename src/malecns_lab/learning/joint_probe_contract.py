"""Frozen P3b registration and small, Torch-free input bindings."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash
from ..senses.target import TargetCamera
from .resource_guard import GIB, resource_budget

SCHEMA = "cnsbridge.joint_probe_registration.v2"
RESULT_SCHEMA = "cnsbridge.joint_probe_worker.v2"
FINAL_SCHEMA = "cnsbridge.joint_probe_result.v2"
BRANCHES = ("baseline", "updated", "acute_restore", "full_restore", "checkpoint_reload")
PARAMETERS = {"theta": "brain.theta", "readout_weight": "readout.weight", "readout_bias": "readout.bias"}
CLIP_GROUPS = {"theta": ("theta",), "readout": ("readout_weight", "readout_bias")}
ADAM_FIELDS = ("step", "exp_avg", "exp_avg_sq")
SCALES = [0.3, 0.2, 0.5]
LIF_CONFIG = {"dt_ms": 0.1, "v_rest_mv": -52.0, "v_reset_mv": -52.0, "v_threshold_mv": -45.0,
              "tau_m_ms": 20.0, "tau_syn_ms": 5.0, "refractory_ms": 2.2, "delay_ms": 1.8,
              "weight_mv": 0.275, "surrogate_scale_mv": 1.0}
PROTOCOL = {
    "schema": "cnsbridge.joint_action_update.v2", "role": "offline_two_frame_feasibility_only",
    "dataset_id": "f0fe9735abbffff6135a8a5d7c8f3943d638da90bfe6a40eb21ba53583137076",
    "dataset_manifest_sha256": "2418141cc7711c64cfc27f7bd8056150b8a170dd0ebed3a3079c25bffecb1104",
    "scene_id": "train-sine-combined", "indices": [0, 1], "times_s": [0.0, 0.1],
    "target": "same_frame_teacher_target", "camera": asdict(TargetCamera()),
    "config": {"frame_ms": 50, "precondition_ms": 60, "control_dt_s": 0.1, "readout_seed": 1703},
    "lif_config": LIF_CONFIG, "drive_config": {"dt_ms": 0.1, "max_rate_hz": 150.0,
                                                  "event_voltage_mv": 68.75, "phase_seed": 1702},
    "spatial_seed": 1701, "background": "fixed_lamina_8mv", "rho": 0, "dtype": "torch.float32",
    "device": "cpu", "torch_threads": 4, "torch_interop_threads": 1,
    "nodes": 166700, "edges": 25582938, "synapse_count": 124177617, "inputs": 3377, "outputs": 1314,
    "raw_ordered_counts_sha256": "4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565",
    "gradient_window": "two_contiguous_frames_no_detach; blank_precondition_no_grad",
    "optimizer": {"name": "Adam", "groups": [{"parameters": ["brain.theta"], "lr": 0.001},
                    {"parameters": ["readout.weight", "readout.bias"], "lr": 0.0001}],
                  "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "amsgrad": False,
                  "foreach": False, "fused": False, "updates": 1},
    "objective": "mean(((actions-targets)/axis_scales)**2)", "axis_scales": SCALES,
    "regularization": 1e-4,
    "clipping": {"rule": "per_optimizer_group_l2", "groups": {k: [PARAMETERS[p] for p in v] for k, v in CLIP_GROUPS.items()},
                 "max_norm_per_group": 1.0, "norm_dtype": "float64", "coefficient_epsilon": 1e-6,
                 "coefficient": "float64 min(1,1/(norm+1e-6)); cast float32 then multiply float32 gradient",
                 "nonfinite": "reject_before_mutating_any_gradient", "post_norm_tolerance": 1e-6},
    "adam_evidence": {"fields": list(ADAM_FIELDS), "dtype": "float32", "step": 1,
                      "moment_rtol": 1e-6, "moment_atol": float(np.nextafter(np.float32(0), np.float32(1))),
                      "update_rtol": 5e-6, "update_atol": 1e-9,
                      "recompute": "zero initial moments; float32 lerp/addcmul, sqrt/div/add, addcdiv"},
    "branches": list(BRANCHES), "rtol": 1e-6, "atol": 1e-7,
}
SOURCE_FILES = (
    "learning/joint_probe.py", "learning/joint_probe_contract.py", "learning/joint_probe_worker.py",
    "learning/joint_probe_evidence.py", "learning/joint_probe_identity.py", "learning/joint_probe_history.py",
    "learning/joint_probe_optimizer.py", "learning/resource_guard.py",
    "learning/neural_policy.py", "learning/neural_data.py", "learning/policy_checkpoint.py", "learning/follow_task.py",
    "brain/trainable.py", "brain/trainable_graph.py", "brain/tonic.py", "brain/policy_groups.py",
    "brain/reference.py", "brain/connectome.py", "brain/metadata.py", "senses/spatial_drive.py",
    "senses/target.py", "drones/follow_env.py", "drones/follow_scenes.py", "control/body.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    result = json.loads(Path(path).read_text(), parse_constant=reject)
    require(isinstance(result, dict), "Expected a JSON object")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sources():
    package = Path(__file__).resolve().parents[1]
    return {name: sha256(package / name) for name in SOURCE_FILES}


def bound(payload, name):
    require(isinstance(payload, dict) and payload.get(name) == canonical_hash({k: v for k, v in payload.items() if k != name}),
            f"Invalid {name}")


def array_digest(value):
    value = np.asarray(value)
    require(value.dtype.kind in "bifu" and np.isfinite(value).all(), "Cannot digest nonfinite/object state")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def graph_registration(path):
    """Only the small metadata ZIP member is decoded; never load graph arrays."""
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta"]))
    require(meta.get("asset_type") == "full_reference" and meta.get("format_version") == 2, "Full reference required")
    require(meta["source_id"] == canonical_hash(meta["provenance"]), "Reference provenance binding mismatch")
    return {"path": str(Path(path).resolve()), "file_sha256": sha256(path), "source_id": meta["source_id"],
            "provenance": meta["provenance"]}


def select_data(directory):
    from .neural_data import read_neural_data
    directory = Path(directory)
    require(sha256(directory / "manifest.json") == PROTOCOL["dataset_manifest_sha256"], "Wrong sealed dataset manifest")
    data = read_neural_data(directory)
    require(data.dataset_id == PROTOCOL["dataset_id"], "Wrong dataset identity")
    require(data.identity["camera"] == PROTOCOL["camera"] and data.identity["dt_s"] == 0.1, "Dataset camera/time mismatch")
    sequence = next(item for item in data.sequences if item.scene["scene_id"] == PROTOCOL["scene_id"])
    selection = {name: getattr(sequence, name)[[0, 1]].copy()
                 for name in ("frames", "time_s", "valid", "teacher_target")}
    require(selection["valid"].all() and np.array_equal(selection["time_s"], PROTOCOL["times_s"]), "Selected frames invalid")
    selection["indices"] = np.array([0, 1], dtype=np.int64)
    return selection, sequence.scene


def validate_registration(registration):
    require(registration.get("schema") == SCHEMA, "Wrong joint registration schema")
    bound(registration, "registration_id")
    require(registration["protocol"] == PROTOCOL and registration["protocol_id"] == canonical_hash(PROTOCOL),
            "Fixed P3b protocol changed")
    require(registration["implementation_sha256"] == sources(), "Registered sources changed")
    for path in (registration["graph"]["path"], registration["dataset_path"]):
        require(Path(path).is_absolute(), "Registered paths must be absolute")
    resources = registration["resources"]
    expected = resource_budget(resources["available_bytes_before_launch"], rss_bytes=6 * GIB, reserve_bytes=GIB)
    require(resources == {**expected, "wall_limit_seconds": 600, "sampling_seconds": 0.1}, "Resource budget changed")
    require(registration["dataset_id"] == PROTOCOL["dataset_id"]
            and registration["dataset_manifest_sha256"] == PROTOCOL["dataset_manifest_sha256"], "Registered dataset changed")
    require(registration["graph"]["source_id"] == canonical_hash(registration["graph"]["provenance"]),
            "Registered graph provenance mismatch")
    from .joint_probe_history import history_binding
    require(registration["prior_run"] == history_binding(registration["prior_run"]["path"],
            registration["implementation_sha256"], registration["graph"]), "r1 history binding changed")
    return {"registration_id": registration["registration_id"], "protocol_id": registration["protocol_id"]}
