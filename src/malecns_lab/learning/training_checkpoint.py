"""Versioned complete TBPTT boundary snapshots, distinct from deployment weights.

No executable serialization or graph duplication. The reader validates numeric
state without constructing a model; production loading verifies the full graph.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import re

import numpy as np
import scipy
import torch

from ..brain.metadata import canonical_hash, file_hashes
from ..brain.trainable import LIFState, TrainableLIFConfig
from ..senses.spatial_drive import SpatialDriveConfig, SpatialDriveState
from ..senses.target import TargetCamera
from .neural_policy import NeuralPolicy, NeuralPolicyConfig, PolicyState
from . import policy_checkpoint as parameters
from .training_optimizer import make_training_optimizer, optimizer_contract, snapshot_optimizer, validate_moments, validate_optimizer

SCHEMA = "cnsbridge.training_checkpoint.v1"
PURPOSE = "detached_two_frame_tbptt_continuation"
IDENTITIES = ("training_run_id", "protocol_id", "dataset_id", "dataset_manifest_sha256")
CONTEXT_FIELDS = {*IDENTITIES, "scenes", "epoch", "scene_index", "scene_id", "next_frame", "completed_windows",
                  "optimizer_steps", "parent_checkpoint_id", "log_prefix_sha256"}
FILES = ("policy/checkpoint.json", "policy/parameters.npz", "training.npz")
SOURCES = ("learning/training_checkpoint.py", "learning/training_optimizer.py", "learning/policy_checkpoint.py")
DETERMINISM = {
    "sample_order": "declared_scene_order; ascending_frame_index; fixed_two_frame_windows",
    "stochastic_training_operators": "none; no sampling/noise/dropout",
    "rng_resume": "no global RNG consumed; initialization is identity-bound; sensory phase is saved in full",
    "state_boundary": "state computed before Adam; retain values after update and detach; no prefix replay",
}
STATE_KEYS = {"brain.v", "brain.g", "brain.refractory_until", "brain.pending", "drive.phase", "drive.event_counts"}


@dataclass(frozen=True)
class TrainingArchive:
    metadata: dict
    parameter_metadata: dict
    parameters: dict[str, np.ndarray]
    optimizer_arrays: dict[str, np.ndarray]
    state: PolicyState

    @property
    def context(self):
        return parameters._json(self.metadata["context"])


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": str(torch.__version__), "machine": platform.machine(), "system": platform.system(),
            "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "dtype": "float32"}


def _sources():
    package = Path(__file__).resolve().parents[1]
    return {name: file_hashes(package / name)["sha256"] for name in SOURCES}


def _validate_context(context, expected):
    context = parameters._json(context)
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise ValueError("Unknown or missing training context fields")
    if (not isinstance(expected, dict) or set(expected) != set(IDENTITIES)
            or any(not _digest(expected[key]) for key in IDENTITIES)):
        raise ValueError("Expected run/protocol/dataset identities must be explicit SHA256 digests")
    if any(not _digest(context[key]) or context[key] != expected[key] for key in IDENTITIES):
        raise ValueError("Training context differs from expected run/protocol/dataset identity")
    if not _digest(context["log_prefix_sha256"]) or (context["parent_checkpoint_id"] is not None
                                                    and not _digest(context["parent_checkpoint_id"])):
        raise ValueError("Log prefix and parent checkpoint identities must be SHA256 digests")
    scenes = context["scenes"]
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Ordered scenes are required")
    ids = []
    for scene in scenes:
        if (not isinstance(scene, dict) or set(scene) != {"scene_id", "frame_count"}
                or not isinstance(scene["scene_id"], str) or not scene["scene_id"].strip()
                or type(scene["frame_count"]) is not int or scene["frame_count"] <= 0 or scene["frame_count"] % 2):
            raise ValueError("Scenes require distinct explicit names and positive even frame counts")
        ids.append(scene["scene_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Scene IDs must be unique and ordered")
    for key in ("epoch", "scene_index", "next_frame", "completed_windows", "optimizer_steps"):
        if type(context[key]) is not int or context[key] < 0:
            raise ValueError("Training cursor counts must be non-boolean nonnegative integers")
    index, frame = context["scene_index"], context["next_frame"]
    if (index >= len(scenes) or context["scene_id"] != scenes[index]["scene_id"]
            or frame < 2 or frame > scenes[index]["frame_count"] or frame % 2):
        raise ValueError("Scene/next_frame cursor must identify a completed two-frame boundary")
    count = (context["epoch"] * sum(item["frame_count"] for item in scenes)
             + sum(item["frame_count"] for item in scenes[:index]) + frame) // 2
    if context["completed_windows"] != count or not 0 < context["optimizer_steps"] <= count:
        raise ValueError("Completed windows/optimizer steps disagree with the independent scene cursor")
    return context


def _state_metadata(state):
    return {"policy_id": state.policy_id, "frame_index": state.frame_index,
            "brain": {"model_id": state.brain.model_id, "step": state.brain.step},
            "drive": {"drive_id": state.drive.drive_id, "spatial_map_id": state.drive.spatial_map_id,
                      "dt_ms": state.drive.dt_ms, "step": state.drive.step}}


def _state_arrays(state):
    tensors = (state.brain.v, state.brain.g, state.brain.refractory_until, *state.brain.pending)
    if any(value.requires_grad or value.grad_fn is not None for value in tensors):
        raise ValueError("Checkpoint state must be detached at the completed update boundary")
    return {"brain.v": state.brain.v.numpy().copy(), "brain.g": state.brain.g.numpy().copy(),
            "brain.refractory_until": state.brain.refractory_until.numpy().copy(),
            "brain.pending": np.stack([value.numpy() for value in state.brain.pending]),
            "drive.phase": state.drive.phase.copy(), "drive.event_counts": state.drive.event_counts.copy()}


def _validate_state(info, arrays, identity, context):
    """Array checks mirror State contracts; no model/graph is allocated here."""
    if (not isinstance(info, dict) or set(info) != {"policy_id", "frame_index", "brain", "drive"}
            or not isinstance(info["brain"], dict) or set(info["brain"]) != {"model_id", "step"}
            or not isinstance(info["drive"], dict)
            or set(info["drive"]) != {"drive_id", "spatial_map_id", "dt_ms", "step"}):
        raise ValueError("Unknown or missing state identity fields")
    step = identity["precondition_steps"] + context["next_frame"] * identity["frame_steps"]
    if (info["policy_id"] != identity["policy_id"] or type(info["frame_index"]) is not int
            or info["frame_index"] != context["next_frame"] or info["brain"]["model_id"] != identity["model"]["model_id"]
            or info["drive"]["drive_id"] != identity["drive"]["drive_id"]
            or info["drive"]["spatial_map_id"] != identity["spatial"]["spatial_map_id"]
            or type(info["drive"]["dt_ms"]) not in (int, float)
            or info["drive"]["dt_ms"] != identity["drive"]["config"]["dt_ms"]
            or any(type(info[part]["step"]) is not int or info[part]["step"] != step for part in ("brain", "drive"))):
        raise ValueError("State neural/drive/frame clocks or identities differ from the cursor/policy")
    nodes = identity["model"]["graph"]["nodes"]
    config = TrainableLIFConfig(**identity["model"]["config"])
    delay = config.integer_steps(config.delay_ms)
    inputs = len(identity["neurons"]["inputs"]["body_ids"])
    declarations = {"brain.v": ((nodes,), np.float32), "brain.g": ((nodes,), np.float32),
                    "brain.refractory_until": ((nodes,), np.int64), "brain.pending": ((delay, nodes), np.float32),
                    "drive.phase": ((inputs,), np.float64), "drive.event_counts": ((inputs,), np.int64)}
    if set(arrays) != STATE_KEYS:
        raise ValueError("State archive must retain every declared state array")
    for key, (shape, dtype) in declarations.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid state shape/dtype/values: {key}")
    if (np.any(arrays["brain.refractory_until"] < 0) or np.any(arrays["drive.event_counts"] < 0)
            or np.any(arrays["drive.event_counts"] > step)
            or np.any(arrays["drive.phase"] < 0) or np.any(arrays["drive.phase"] >= 1)):
        raise ValueError("Invalid refractory clock, sensory phase or event count")


def _build_state(info, arrays):
    brain = LIFState(torch.from_numpy(arrays["brain.v"].copy()), torch.from_numpy(arrays["brain.g"].copy()),
                     torch.from_numpy(arrays["brain.refractory_until"].copy()),
                     tuple(torch.from_numpy(value.copy()) for value in arrays["brain.pending"]),
                     info["brain"]["step"], info["brain"]["model_id"])
    drive = SpatialDriveState(**info["drive"], phase=arrays["drive.phase"], event_counts=arrays["drive.event_counts"])
    return PolicyState(brain, drive, info["frame_index"], info["policy_id"])


def _declarations(arrays):
    return {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in arrays.items()}


def _no_gradients(policy):
    if any(value.grad is not None for value in policy.parameters()):
        raise ValueError("All parameter gradients must be None at the update boundary")


def save_training_checkpoint(policy, optimizer, state, directory, *, mode, context):
    """Seal one completed boundary in a new directory; final manifest is last."""
    identity, parameter_arrays = parameters._arrays(policy)
    expected = {key: context.get(key) for key in IDENTITIES} if isinstance(context, dict) else {}
    context = _validate_context(context, expected)
    _no_gradients(policy)
    policy._drive_for(state)  # Reuse existing complete state checks before serialization.
    arrays = _state_arrays(state)
    info = _state_metadata(state)
    _validate_state(info, arrays, identity, context)
    contract, moments = snapshot_optimizer(policy, optimizer, mode, parameter_arrays, context["optimizer_steps"])
    arrays.update(moments)
    sources, runtime = _sources(), _runtime()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parameter_metadata = parameters.save_policy_checkpoint(policy, directory / "policy", run_identity=expected)
    np.savez(directory / "training.npz", **arrays)
    metadata = {"schema": SCHEMA, "purpose": PURPOSE, "context": context, "optimizer": contract,
                "state": info, "arrays": _declarations(arrays), "policy_checkpoint_id": parameter_metadata["checkpoint_id"],
                "files_sha256": {name: file_hashes(directory / name)["sha256"] for name in FILES},
                "implementation_sha256": sources, "runtime": runtime, "determinism": DETERMINISM}
    metadata["checkpoint_id"] = canonical_hash(metadata)
    (directory / "checkpoint.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metadata


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def read_training_checkpoint(directory, *, expected):
    """Parse/validate without graph construction; expected identities are required."""
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "checkpoint.json").read_text(), object_pairs_hook=_unique_object)
        parameters._json(metadata)
        keys = {"schema", "purpose", "context", "optimizer", "state", "arrays", "policy_checkpoint_id",
                "files_sha256", "implementation_sha256", "runtime", "determinism", "checkpoint_id"}
        if not isinstance(metadata, dict) or set(metadata) != keys or metadata["schema"] != SCHEMA or metadata["purpose"] != PURPOSE:
            raise ValueError("Unsupported complete training checkpoint schema/purpose")
        parameters._bound(metadata, "checkpoint_id")
        context = _validate_context(metadata["context"], expected)
        if metadata["implementation_sha256"] != _sources() or metadata["runtime"] != _runtime():
            raise ValueError("Training source/runtime/thread configuration mismatch")
        if metadata["determinism"] != DETERMINISM:
            raise ValueError("Unsupported deterministic continuation contract")
        if not isinstance(metadata["files_sha256"], dict) or set(metadata["files_sha256"]) != set(FILES):
            raise ValueError("Unexpected checkpoint file names")
        for name in FILES:
            if file_hashes(directory / name)["sha256"] != metadata["files_sha256"][name]:
                raise ValueError("Training checkpoint file hash mismatch")
        # Reject duplicate keys in the nested JSON before delegating all parameter semantics.
        json.loads((directory / "policy/checkpoint.json").read_text(), object_pairs_hook=_unique_object)
        parameter_metadata, parameter_arrays = parameters.read_policy_checkpoint(directory / "policy")
        identity = parameter_metadata["policy"]
        if (identity["dtype"] != "torch.float32" or parameter_metadata["checkpoint_id"] != metadata["policy_checkpoint_id"]
                or parameter_metadata["run_identity"] != expected):
            raise ValueError("Nested policy checkpoint identity/dtype/run mismatch")
        contract = optimizer_contract(metadata["optimizer"]["mode"])
        if metadata["optimizer"] != contract:
            raise ValueError("Optimizer configuration differs from the declared mode")
        with np.load(directory / "training.npz", allow_pickle=False) as payload:
            if len(payload.files) != len(set(payload.files)):
                raise ValueError("Duplicate NPZ field")
            arrays = {key: payload[key].copy() for key in payload.files}
        if metadata["arrays"] != _declarations(arrays):
            raise ValueError("Training array declarations mismatch")
        states = {key: value for key, value in arrays.items() if key in STATE_KEYS}
        moments = {key: value for key, value in arrays.items() if key not in STATE_KEYS}
        _validate_state(metadata["state"], states, identity, context)
        validate_moments(moments, contract, parameter_arrays, context["optimizer_steps"])
        return TrainingArchive(metadata, parameter_metadata, parameter_arrays, moments, _build_state(metadata["state"], states))
    except (KeyError, TypeError, OverflowError, OSError, AttributeError) as exc:
        raise ValueError(f"Invalid or incomplete training checkpoint: {exc}") from exc


def _restore(policy, optimizer, archive):
    mode = archive.metadata["optimizer"]["mode"]
    contract, active = validate_optimizer(policy, optimizer, mode)
    _no_gradients(policy)
    identity, _ = parameters._arrays(policy)
    if identity != archive.parameter_metadata["policy"] or contract != archive.metadata["optimizer"]:
        raise ValueError("Target policy/optimizer identity differs from the checkpoint")
    policy._drive_for(archive.state)
    # Prepare every allocation and validate every target before the first mutation.
    values = {parameters.PARAMETERS[key]: torch.from_numpy(value.copy()) for key, value in archive.parameters.items()}
    moments = defaultdict(dict, {
        parameter: {key: torch.from_numpy(archive.optimizer_arrays[f"{name}.{key}"].copy())
                    for key in ("step", "exp_avg", "exp_avg_sq")}
        for name, parameter in active.items()
    })
    named = dict(policy.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            named[name].copy_(value)
    optimizer.state = moments
    return archive.state, archive.context


def restore_training_checkpoint(policy, optimizer, directory, *, expected):
    """Restore atomically after complete validation; never reset/precondition."""
    return _restore(policy, optimizer, read_training_checkpoint(directory, expected=expected))


def load_training_checkpoint(directory, reference_path, *, expected):
    """Production construction verifies full_reference, and returns policy/Adam/state/context."""
    archive = read_training_checkpoint(directory, expected=expected)
    identity = archive.parameter_metadata["policy"]
    if identity["role"] != "full_reference":
        raise ValueError("Production training loading requires full_reference; numerical fixtures are excluded")
    policy = NeuralPolicy.from_reference(reference_path, background=identity["background"], dtype=torch.float32,
                                         config=NeuralPolicyConfig(**identity["config"]), camera=TargetCamera(**identity["camera"]),
                                         drive_config=SpatialDriveConfig(**identity["drive"]["config"]),
                                         lif_config=TrainableLIFConfig(**identity["model"]["config"]))
    optimizer = make_training_optimizer(policy, archive.metadata["optimizer"]["mode"])
    state, context = _restore(policy, optimizer, archive)
    return policy, optimizer, state, context
