"""Identity-bound internal-gain checkpoints: JSON and non-pickled numeric NPZ."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from malecns_lab.brain.metadata import canonical_hash, file_hashes
from malecns_lab.brain.trainable import MODEL_SCHEMA, TrainableLIF, TrainableLIFConfig
from malecns_lab.brain.trainable_graph import TrainingGraph, load_training_graph

SCHEMA = "cnsbridge.internal_gain_checkpoint.v1"


def _supported_model(model: TrainableLIF) -> dict:
    """Reject unrepresented dynamics/parameters before creating or mutating anything."""
    if not isinstance(model, TrainableLIF) or model.identity.get("schema") != MODEL_SCHEMA:
        raise ValueError("Gain-only checkpoints support only the original TrainableLIF dynamics schema")
    if set(dict(model.named_parameters(remove_duplicate=False))) != {"theta"}:
        raise ValueError("Gain-only checkpoints require the exact parameter set {theta}")
    theta = model.theta
    if (theta.shape != (len(model.graph.group_names),) or theta.dtype != model._dtype
            or theta.device.type != "cpu" or not torch.isfinite(theta).all()):
        raise ValueError("Cannot checkpoint invalid internal parameter shape, dtype, device or values")
    return model.identity


def save_checkpoint(model: TrainableLIF, directory: str | Path, *, run_identity: dict | None = None) -> dict:
    """Create a new checkpoint directory; never replace an earlier artifact."""
    identity = _supported_model(model)
    theta = model.theta.detach().cpu().numpy().copy()
    if not np.isfinite(theta).all():
        raise ValueError("Cannot save nonfinite internal parameters")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = directory / "parameters.npz"
    np.savez(arrays, theta=theta)
    metadata = {"schema": SCHEMA, "model": identity,
                "parameters": {"theta": {"shape": list(theta.shape), "dtype": theta.dtype.str}},
                "arrays_sha256": file_hashes(arrays)["sha256"]}
    if run_identity is not None:
        metadata["run_identity"] = dict(run_identity)
    metadata["checkpoint_id"] = canonical_hash(metadata)
    (directory / "checkpoint.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def read_checkpoint(directory: str | Path) -> tuple[dict, np.ndarray]:
    """Read and verify bytes/schema before any model parameter is mutated."""
    directory = Path(directory)
    metadata = json.loads((directory / "checkpoint.json").read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unknown internal checkpoint schema")
    if not isinstance(metadata.get("model"), dict) or metadata["model"].get("schema") != MODEL_SCHEMA:
        raise ValueError("Unsupported checkpoint dynamics schema")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"theta"}:
        raise ValueError("Checkpoint metadata must declare exactly theta parameters")
    if not isinstance(parameters["theta"], dict) or set(parameters["theta"]) != {"shape", "dtype"}:
        raise ValueError("Invalid theta parameter metadata")
    content = {key: value for key, value in metadata.items() if key != "checkpoint_id"}
    if metadata.get("checkpoint_id") != canonical_hash(content):
        raise ValueError("Checkpoint metadata identity mismatch")
    arrays = directory / "parameters.npz"
    if file_hashes(arrays)["sha256"] != metadata["arrays_sha256"]:
        raise ValueError("Checkpoint parameter bytes changed")
    with np.load(arrays, allow_pickle=False) as payload:
        if set(payload.files) != {"theta"}:
            raise ValueError("Only internal theta parameters may be restored")
        theta = payload["theta"].copy()
    expected = metadata["parameters"]["theta"]
    if (theta.dtype.kind != "f" or list(theta.shape) != expected["shape"]
            or theta.dtype.str != expected["dtype"] or not np.isfinite(theta).all()):
        raise ValueError("Invalid checkpoint parameter shape, dtype, or finite values")
    return metadata, theta


def restore_checkpoint(model: TrainableLIF, directory: str | Path) -> dict:
    """Restore to an already validated model, including explicit numerical fixtures."""
    identity = _supported_model(model)
    metadata, theta = read_checkpoint(directory)
    if metadata["model"] != identity:
        raise ValueError("Checkpoint model, graph, grouping, input or numeric identity differs")
    expected = model.theta.detach().cpu().numpy()
    if theta.shape != expected.shape or theta.dtype != expected.dtype:
        raise ValueError("Checkpoint theta does not match model parameters")
    with torch.no_grad():
        model.theta.copy_(torch.from_numpy(theta))
    return metadata


def load_checkpoint(directory: str | Path, *, reference: str | Path | TrainingGraph) -> TrainableLIF:
    """Production reload needs a separately verified complete reference graph."""
    metadata, _ = read_checkpoint(directory)
    identity = metadata["model"]
    if identity["graph"].get("role") != "full_reference":
        raise ValueError("Production reload requires full_reference; numerical fixtures are excluded")
    graph = reference if isinstance(reference, TrainingGraph) else load_training_graph(reference)
    if graph.identity != identity["graph"] or graph.identity.get("role") != "full_reference":
        raise ValueError("Checkpoint reference graph or type grouping differs")
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(identity.get("dtype"))
    if dtype is None:
        raise ValueError("Unsupported checkpoint dtype")
    model = TrainableLIF(graph, identity["input_body_ids"], config=TrainableLIFConfig(**identity["config"]), dtype=dtype)
    restore_checkpoint(model, directory)
    return model
