"""Complete policy *parameters*, not optimizer/hidden-state training resumes.

JSON plus numeric NPZ avoids pickle and makes graph, sensory mapping, clocks and
readout semantics explicit. Reading never loads the full reference graph.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import torch

from ..brain.metadata import canonical_hash, file_hashes
from ..brain.policy_groups import NeuronSelection, PolicyNeuronMap, POLICY_MAP_SCHEMA
from ..brain.tonic import TONIC_SCHEMA, LAMINA_TYPES
from ..brain.trainable import MODEL_SCHEMA, SHIU_REFERENCE, TrainableLIFConfig
from ..senses.spatial_drive import SpatialDrive, SpatialDriveConfig, build_spatial_input_map
from ..senses.target import TargetCamera
from .neural_policy import NeuralPolicy, NeuralPolicyConfig, POLICY_SCHEMA, ACTION_AXES, ACTION_SCALES

SCHEMA = "cnsbridge.neural_policy_checkpoint.v1"
PURPOSE = "deployment_parameters_only"
PARAMETERS = {"theta": "brain.theta", "readout_weight": "readout.weight", "readout_bias": "readout.bias"}
SOURCES = ("learning/neural_policy.py", "learning/follow_task.py", "senses/target.py", "senses/spatial_drive.py",
           "brain/policy_groups.py", "brain/metadata.py", "control/body.py")


def _json(value):
    """Validate finite JSON and own a normalized copy before touching disk/model."""
    return json.loads(json.dumps(value, allow_nan=False))


def _bound(identity, key):
    if not isinstance(identity, dict) or key not in identity:
        raise ValueError(f"Missing {key} identity")
    if identity[key] != canonical_hash({k: v for k, v in identity.items() if k != key}):
        raise ValueError(f"Invalid {key} canonical identity")


def _source_hashes(names, *, brain=False):
    package = Path(__file__).resolve().parents[1]
    return {name: file_hashes(package / ("brain" if brain else "") / name)["sha256"] for name in names}


def _selection(payload, nodes):
    keys = {"body_ids", "indices", "root_side", "root_side_missing", "soma_side", "soma_side_missing"}
    if set(payload) != keys:
        raise ValueError("Invalid neuron selection declaration")
    count = len(payload["body_ids"])
    if not count or any(not isinstance(value, list) or len(value) != count for value in payload.values()):
        raise ValueError("Neuron selection fields must have the same positive length")
    for name in ("body_ids", "indices"):
        if any(type(value) is not int for value in payload[name]):
            raise ValueError("Neuron IDs/indices must be integers")
    ids, indices = np.array(payload["body_ids"], dtype=np.int64), np.array(payload["indices"], dtype=np.int64)
    if np.any(np.diff(ids) <= 0) or np.any(np.diff(indices) <= 0) or np.any(indices < 0) or np.any(indices >= nodes):
        raise ValueError("Neuron selections must retain ascending unique IDs/indices")
    for name in ("root_side", "soma_side"):
        if any(not isinstance(value, str) for value in payload[name]):
            raise ValueError("Neuron sides must retain raw strings")
        if any(type(value) is not bool for value in payload[name + "_missing"]):
            raise ValueError("Neuron side missing masks must be boolean")
    return NeuronSelection(ids, indices, np.array(payload["root_side"]), np.array(payload["root_side_missing"]),
                           np.array(payload["soma_side"]), np.array(payload["soma_side_missing"]))


def _validate_model_semantics(model, background, role, nodes, group_names, input_count):
    """Check the versioned dynamics contract, not merely its self-declared hash.

    Membership digests are format/relationship checked here. The production
    reference loader still independently verifies actual graph membership.
    """
    fixed = {
        "backend": "scipy_cpu_csr_forward_torch_csr_adjoint",
        "gain": "1 + 0.5*tanh(theta[type(pre)]); initial theta=0; nextafter guards strict floating bounds",
        "gradient": "fast_sigmoid; reset/refractory gates detached; all-cell sparse adjoint",
        "event_order": "arrivals; exact active integration; explicit voltage input; strict threshold; reset; delayed enqueue",
        "reference": SHIU_REFERENCE,
    }
    keys = {"schema", "graph", "config", "input_body_ids", "dtype", "implementation_sha256", "model_id", *fixed}
    if background == "fixed_lamina_8mv":
        keys |= {"tonic", "base_dynamics_schema"}
    if set(model) != keys or any(model[key] != value for key, value in fixed.items()):
        raise ValueError("Unsupported model fields or fixed dynamics semantics")
    if background == "zero_background":
        if model["schema"] != MODEL_SCHEMA:
            raise ValueError("Zero background must retain original dynamics")
        return ("trainable.py", "trainable_graph.py")
    if background != "fixed_lamina_8mv":
        raise ValueError("Unsupported background")
    tonic = model["tonic"]
    fixed_tonic = {
        "mode": "fixed", "initial_rho": 0, "initial_offset_mv": 8,
        "formula": "16*sigmoid(rho[type]); nextafter guards strict floating bounds; nonmembers=0",
        "range_mv": {"lower": 0, "upper": 16, "inclusive": False},
        "integration": "active only: integrated_v += (1-exp(-dt/tau_m))*offset; g unchanged",
        "hypothesis": "engineering equilibrium offset, not a measured physiological constant",
    }
    tonic_keys = {"types", "members", "member_count", "mapping_sha256", *fixed_tonic}
    if (model["schema"] != TONIC_SCHEMA or model["base_dynamics_schema"] != MODEL_SCHEMA
            or not isinstance(tonic, dict) or set(tonic) != tonic_keys
            or any(canonical_hash(tonic[key]) != canonical_hash(value) for key, value in fixed_tonic.items())):
        raise ValueError("Unsupported fixed tonic dynamics semantics")
    types, members = tonic["types"], tonic["members"]
    if (not isinstance(types, list) or not types or types != [name for name in LAMINA_TYPES if name in types]
            or any(name not in group_names for name in types)
            or (role == "full_reference" and types != list(LAMINA_TYPES))
            or not isinstance(members, list) or len(members) != len(types)):
        raise ValueError("Invalid tonic type/member declaration")
    def digest(value):
        return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None
    for name, member in zip(types, members):
        if (not isinstance(member, dict) or set(member) != {"type", "count", "body_ids_sha256"}
                or member["type"] != name or type(member["count"]) is not int or member["count"] <= 0
                or not digest(member["body_ids_sha256"])):
            raise ValueError("Invalid tonic member count/type/digest")
    if (type(tonic["member_count"]) is not int
            or tonic["member_count"] != sum(member["count"] for member in members)
            or tonic["member_count"] > nodes - input_count or not digest(tonic["mapping_sha256"])):
        raise ValueError("Tonic members must be disjoint from inputs and agree with their declared total")
    return ("trainable.py", "trainable_graph.py", "tonic.py")


def _validate_identity(identity):
    """Check nested hashes AND their semantic relationships without graph allocation."""
    _json(identity)
    _bound(identity, "policy_id")
    expected_keys = {"schema", "role", "model", "neurons", "spatial", "drive", "camera", "config", "background", "dtype",
                     "frame_steps", "precondition_steps", "features", "readout", "training_parameters", "boundary",
                     "implementation_sha256", "policy_id"}
    if set(identity) != expected_keys or identity["schema"] != POLICY_SCHEMA:
        raise ValueError("Unsupported policy identity schema/fields")
    role = identity["role"]
    if role not in ("full_reference", "numerical_test_only"):
        raise ValueError("Unsupported policy role")
    model, neurons = identity["model"], identity["neurons"]
    graph = model["graph"]
    for obj, key in ((model, "model_id"), (graph, "training_graph_id"), (neurons, "policy_map_id"),
                     (identity["spatial"], "spatial_map_id"), (identity["drive"], "drive_id")):
        _bound(obj, key)
    nodes, names = graph["nodes"], graph["grouping"]["names"]
    if (type(nodes) is not int or nodes <= 0 or not isinstance(names, list) or len(names) < 2 or names[0] is not None
            or any(not isinstance(name, str) or not name.strip() for name in names[1:])
            or names[1:] != sorted(set(names[1:])) or "R1-R6" not in names
            or graph["role"] != role or neurons["role"] != role or neurons["nodes"] != nodes
            or neurons["schema"] != POLICY_MAP_SCHEMA or neurons["training_graph_id"] != graph["training_graph_id"]
            or neurons["reference_source_id"] != graph["reference_source_id"]
            or neurons["reference_graph_sha256"] != graph.get("reference_graph_sha256")):
        raise ValueError("Graph role, grouping or neuron-map identity mismatch")
    if role == "full_reference" and (not graph.get("reference_source_id") or not graph.get("reference_graph_sha256")
                                      or graph["reference_provenance"]["graph_sha256"] != graph["reference_graph_sha256"]):
        raise ValueError("Full reference provenance required")
    if role == "numerical_test_only" and graph["reference_source_id"] is not None:
        raise ValueError("Numerical fixtures cannot claim a reference source")
    inputs, outputs = (_selection(neurons[name], nodes) for name in ("inputs", "outputs"))
    if model["input_body_ids"] != inputs.body_ids.tolist():
        raise ValueError("Model input registration differs from sensory selection")
    mapping = PolicyNeuronMap(nodes, inputs, outputs, json.dumps(neurons, sort_keys=True))
    spatial = build_spatial_input_map(mapping)
    if identity["spatial"] != spatial.identity:
        raise ValueError("Spatial map does not match the complete registered input selection")
    drive_config = SpatialDriveConfig(**identity["drive"]["config"])
    if identity["drive"] != SpatialDrive(spatial, drive_config).identity:
        raise ValueError("Drive semantics/configuration mismatch")
    config = NeuralPolicyConfig(**identity["config"])
    camera = TargetCamera(**identity["camera"])
    lif = TrainableLIFConfig(**model["config"])
    frame_steps, precondition_steps = config.steps(lif.dt_ms)
    if (asdict(config) != identity["config"] or asdict(camera) != identity["camera"] or asdict(lif) != model["config"]
            or lif.dt_ms != drive_config.dt_ms or identity["frame_steps"] != frame_steps
            or identity["precondition_steps"] != precondition_steps):
        raise ValueError("Policy camera or clock declaration mismatch")
    dtype = {"torch.float32": np.dtype("float32"), "torch.float64": np.dtype("float64")}.get(identity["dtype"])
    if dtype is None or model["dtype"] != identity["dtype"]:
        raise ValueError("Policy/model dtype mismatch")
    source_names = _validate_model_semantics(model, identity["background"], role, nodes, names, len(inputs.indices))
    if (identity["implementation_sha256"] != _source_hashes(SOURCES)
            or model["implementation_sha256"] != _source_hashes(source_names, brain=True)):
        raise ValueError("Policy/model source version mismatch")
    width = 2 * len(outputs.indices)
    expected_features = {"ordering": "first_half_DN_then_second_half_DN; ascending body ID in each half",
                         "body_ids": outputs.body_ids.tolist(), "width": width,
                         "sample": "post_reset_v_each_neural_step", "reduction": "arithmetic_mean_per_half",
                         "raw_units": "mV", "normalization": "(mean_v-v_rest)/(v_threshold-v_rest)",
                         "offset_mv": lif.v_rest_mv, "scale_mv": lif.v_threshold_mv - lif.v_rest_mv}
    expected_readout = {"architecture": "linear_bias_then_tanh_then_axis_scales", "axes": list(ACTION_AXES),
                        "scales": list(ACTION_SCALES), "units": ["m/s", "m/s", "rad/s"], "frame": "body_flu",
                        "weight_initialization": "local_CPU_Generator.normal(std=0.01/sqrt(feature_width)); bias=0"}
    if (identity["features"] != expected_features or identity["readout"] != expected_readout
            or identity["training_parameters"] != list(PARAMETERS.values())):
        raise ValueError("Readout feature/axis/parameter declaration mismatch")
    return {"theta": (len(names),), "readout_weight": (3, width), "readout_bias": (3,)}, dtype


def _arrays(policy):
    if type(policy) is not NeuralPolicy:
        raise ValueError("An existing NeuralPolicy is required")
    policy._validate_parameters()
    parameters = dict(policy.named_parameters(remove_duplicate=False))
    if set(parameters) != set(PARAMETERS.values()):
        raise ValueError("Policy must have exactly theta and linear readout parameters")
    identity = policy.identity
    shapes, dtype = _validate_identity(identity)
    arrays = {key: parameters[name].detach().numpy().copy() for key, name in PARAMETERS.items()}
    _validate_arrays(arrays, shapes, dtype)
    return identity, arrays


def _validate_arrays(arrays, shapes, dtype):
    if set(arrays) != set(PARAMETERS):
        raise ValueError("Policy archive must contain exactly the three declared parameters")
    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid parameter shape/dtype/values: {name}")


def _declarations(arrays):
    return {key: {"parameter": PARAMETERS[key], "shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()}


def save_policy_checkpoint(policy, directory, *, run_identity=None):
    identity, arrays = _arrays(policy)
    run_identity = _json(run_identity)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "parameters.npz", **arrays)
    metadata = {"schema": SCHEMA, "purpose": PURPOSE, "policy": identity, "parameters": _declarations(arrays),
                "arrays_sha256": file_hashes(directory / "parameters.npz")["sha256"], "run_identity": run_identity}
    metadata["checkpoint_id"] = canonical_hash(metadata)
    (directory / "checkpoint.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metadata


def read_policy_checkpoint(directory):
    """Return (metadata, arrays) without constructing a graph, model or controller."""
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "checkpoint.json").read_text())
        _json(metadata)
        if (set(metadata) != {"schema", "purpose", "policy", "parameters", "arrays_sha256", "run_identity", "checkpoint_id"}
                or metadata["schema"] != SCHEMA or metadata["purpose"] != PURPOSE):
            raise ValueError("Unsupported complete policy checkpoint schema/purpose")
        _bound(metadata, "checkpoint_id")
        shapes, dtype = _validate_identity(metadata["policy"])
        if file_hashes(directory / "parameters.npz")["sha256"] != metadata["arrays_sha256"]:
            raise ValueError("Policy parameter file hash mismatch")
        with np.load(directory / "parameters.npz", allow_pickle=False) as payload:
            if len(payload.files) != len(PARAMETERS) or set(payload.files) != set(PARAMETERS):
                raise ValueError("Unexpected policy parameter fields")
            arrays = {name: payload[name].copy() for name in PARAMETERS}
        _validate_arrays(arrays, shapes, dtype)
        if metadata["parameters"] != _declarations(arrays):
            raise ValueError("Parameter metadata does not match declared policy/arrays")
        return metadata, arrays
    except (KeyError, TypeError, OverflowError, OSError) as exc:
        raise ValueError(f"Invalid or incomplete policy checkpoint: {exc}") from exc


def _restore_validated(policy, metadata, arrays):
    identity, expected = _arrays(policy)
    if identity != metadata["policy"]:
        raise ValueError("Policy identity differs: graph/role/background/camera/time/order/dtype/source must all match")
    # Validate every target and prepare every tensor before the first mutation.
    for name, value in arrays.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError("Checkpoint parameters differ from the target policy")
    tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name, value in tensors.items():
            parameters[PARAMETERS[name]].copy_(value)
    return metadata


def restore_policy_checkpoint(policy, directory):
    metadata, arrays = read_policy_checkpoint(directory)
    return _restore_validated(policy, metadata, arrays)


def load_policy_checkpoint(directory, reference_path):
    """Production entry: independently verify the reference before restoring.

    This constructs parameters only; the caller must explicitly initialize a new
    scene. It is not an optimizer or recurrent-state resume operation.
    """
    metadata, arrays = read_policy_checkpoint(directory)
    identity = metadata["policy"]
    if identity["role"] != "full_reference":
        raise ValueError("Production loading requires full_reference; numerical fixtures are excluded")
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}[identity["dtype"]]
    policy = NeuralPolicy.from_reference(reference_path, background=identity["background"], dtype=dtype,
                                         config=NeuralPolicyConfig(**identity["config"]), camera=TargetCamera(**identity["camera"]),
                                         drive_config=SpatialDriveConfig(**identity["drive"]["config"]),
                                         lif_config=TrainableLIFConfig(**identity["model"]["config"]))
    _restore_validated(policy, metadata, arrays)
    return policy
