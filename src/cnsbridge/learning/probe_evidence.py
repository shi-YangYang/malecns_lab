"""Lightweight P1 evidence validation; no Torch, full graph load, or simulation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .resource_guard import DEFAULT_SYSTEM_RESERVE_BYTES, GIB

REGISTRATION_SCHEMA = "cnsbridge.probe_registration.v2"
RESULT_SCHEMA = "cnsbridge.fullgraph_probe_result.v2"
ARTIFACTS = ("model_identity.json", "baseline.npz", "update.npz", "responses.npz",
             "checkpoint/checkpoint.json", "checkpoint/parameters.npz")
PHASES = ("imports", "load_graph", "construct_model", "construct_optimizer", "baseline_forward", "backward",
          "optimizer_step", "updated_forward", "checkpoint_save", "restored_zero_forward", "checkpoint_load",
          "checkpoint_forward", "verify_raw_graph")


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes() -> dict:
    paths = {name: Path(__file__).with_name(name)
             for name in ("probe.py", "checkpoint.py", "resource_guard.py", "probe_evidence.py")}
    paths.update({f"brain/{name}": Path(__file__).parent.parent / "brain" / name
                  for name in ("trainable.py", "trainable_graph.py")})
    return {name: file_sha256(path) for name, path in paths.items()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    _require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    return value


def _identity(value: dict, key: str) -> None:
    _require(isinstance(value, dict) and value.get(key) == canonical_hash({k: v for k, v in value.items() if k != key}),
             f"Invalid {key}")


def _finite(value, name: str, *, minimum: float = 0.0) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, f"Invalid finite metric: {name}")
    return value


def validate_registration(registration: dict, protocol: dict) -> dict:
    """Bind a newly registered run to this implementation and fixed protocol."""
    _require(registration.get("schema") == REGISTRATION_SCHEMA, "Wrong registration schema")
    _identity(registration, "registration_id")
    _require(registration.get("protocol") == protocol, "Registered protocol differs from fixed P1 protocol")
    _require(registration.get("protocol_id") == canonical_hash(protocol), "Protocol identity mismatch")
    _require(registration.get("implementation_sha256") == implementation_hashes(), "Registered implementation changed")
    _require(isinstance(registration.get("graph_path"), str) and Path(registration["graph_path"]).is_absolute(),
             "Registered graph path must be absolute")
    resources = registration["resources"]
    _require(resources["wall_limit_seconds"] == 600 and resources["sampling_seconds"] == 0.1
             and resources["reserve_bytes"] == DEFAULT_SYSTEM_RESERVE_BYTES and resources["requested_rss_limit_bytes"] == 6 * GIB,
             "Registered resource protocol differs")
    available = _finite(resources["available_bytes_before_launch"], "available_bytes_before_launch")
    expected_limit = min(resources["requested_rss_limit_bytes"], max(0, available - resources["reserve_bytes"]))
    _require(resources["effective_rss_limit_bytes"] == expected_limit and resources["can_launch"] is (expected_limit > 0),
             "Registered resource budget is inconsistent")
    return {"registration_id": registration["registration_id"], "protocol_id": registration["protocol_id"]}


def _arrays(path: Path, keys: set[str], run_identity: dict | None = None) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        _require(set(payload.files) == keys | (set(run_identity) if run_identity else set()), f"Wrong arrays in {path.name}")
        values = {key: payload[key].copy() for key in payload.files}
    if run_identity:
        for key, expected in run_identity.items():
            _require(values[key].shape == () and values[key].dtype.kind == "U" and values[key].item() == expected,
                     f"Artifact run identity mismatch: {path.name}/{key}")
    return values


def _vector(value, shape: tuple[int, ...], name: str, *, dtype="float32") -> None:
    import numpy as np
    _require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
             f"Invalid evidence vector: {name}")


def _model(result: dict, protocol: dict) -> tuple[dict, int, int]:
    graph, model = result["graph"], result["model"]
    _identity(graph, "training_graph_id")
    _identity(model, "model_id")
    _require(graph.get("role") == "full_reference" and graph.get("reference_source_id"), "P1 requires a full_reference identity")
    _require(model.get("schema") == "cnsbridge.trainable_lif.v1" and model.get("graph") == graph,
             "Model and graph identities differ")
    for name in ("nodes", "edges", "synapse_count"):
        _require(type(graph[name]) is int and graph[name] > 0, f"Invalid graph count: {name}")
    _require(model.get("dtype") == protocol["dtype"] and model.get("backend") == "torch_cpu_csr", "Model backend/dtype mismatch")
    expected_config = {"dt_ms": protocol["dt_ms"], "v_rest_mv": -52.0, "v_reset_mv": -52.0, "v_threshold_mv": -45.0,
                       "tau_m_ms": 20.0, "tau_syn_ms": 5.0, "refractory_ms": 2.2, "delay_ms": 1.8,
                       "weight_mv": 0.275, "surrogate_scale_mv": 1.0}
    _require(model.get("config") == expected_config, "Model numerical configuration differs from P1")
    expected_core = {name: file_sha256(Path(__file__).parent.parent / "brain" / name)
                     for name in ("trainable.py", "trainable_graph.py")}
    _require(model.get("implementation_sha256") == expected_core, "Model implementation identity changed")
    inputs = model["input_body_ids"]
    _require(isinstance(inputs, list) and inputs and all(type(value) is int for value in inputs)
             and inputs == sorted(set(inputs)), "Invalid model input IDs")
    groups = graph["grouping"]
    names = groups["names"]
    _require(groups.get("schema") == "cnsbridge.presynaptic_raw_type.v1" and groups.get("missing_index") == 0
             and isinstance(names, list) and names and names[0] is None
             and all(isinstance(name, str) and name.strip() for name in names[1:])
             and names[1:] == sorted(set(names[1:])) and protocol["input_raw_type"] in names
             and protocol["observation_raw_type"] in names, "Invalid type grouping identity")
    for name in ("input_neurons", "observation_neurons"):
        _require(type(result[name]) is int and 0 < result[name] <= graph["nodes"], f"Invalid group size: {name}")
    _require(result["input_neurons"] == len(inputs), "Input count differs from model")
    for value in (graph["raw_ordered_counts_sha256"], groups["mapping_sha256"], graph["reference_graph_sha256"]):
        _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                 "Invalid graph digest")
    return model, len(names), result["observation_neurons"]


def validate_probe_evidence(directory: Path, protocol: dict) -> dict:
    """Recompute P1 conclusions from identity-bound small artifacts.

    This validates persisted evidence, not the truth of fabricated files. Only
    the registered worker's production loader establishes the full graph. No
    fixture role or alternate protocol can enter this production entry point.
    """
    import numpy as np
    registration = _json(directory / "registration.json")
    run_identity = validate_registration(registration, protocol)
    _require(registration["resources"]["can_launch"], "Registration did not allow computation")
    result = _json(directory / "worker_result.json")
    _require(result.get("schema") == RESULT_SCHEMA, "Wrong worker result schema")
    _require(all(result.get(key) == value for key, value in run_identity.items()), "Worker registration/protocol identity mismatch")
    _require(result.get("implementation_sha256") == registration["implementation_sha256"], "Worker implementation identity mismatch")
    _require(result.get("graph_path") == registration["graph_path"], "Worker graph path mismatch")
    _require(result.get("status") == "passed" and result.get("reason") == "numeric_internal_update_verified",
             "Worker did not report a completed numerical update")
    environment = result["environment"]
    _require(environment.get("device") == "cpu" and environment.get("torch_threads") == protocol["torch_threads"],
             "Worker execution configuration mismatch")
    for name in ("python", "numpy", "scipy", "torch", "psutil"):
        _require(isinstance(environment.get(name), str) and bool(environment[name]), f"Missing environment version: {name}")
    phases = result["phases_seconds"]
    _require(set(phases) == set(PHASES), "Missing or unexpected numerical phases")
    for name, duration in phases.items():
        _finite(duration, name)
    wall = _finite(result["wall_seconds"], "wall_seconds")
    _require(wall + 1e-6 >= sum(phases.values()) and wall <= registration["resources"]["wall_limit_seconds"],
             "Worker phase/wall times are inconsistent")
    _require(result.get("simulated_ms_per_forward") == protocol["simulated_ms"], "Wrong simulated duration")

    model, groups, observations = _model(result, protocol)
    artifact_hashes = result["artifacts_sha256"]
    _require(isinstance(artifact_hashes, dict) and set(artifact_hashes) == set(ARTIFACTS), "Incomplete artifact manifest")
    for name in ARTIFACTS:
        _require(file_sha256(directory / name) == artifact_hashes[name], f"Artifact digest mismatch: {name}")
    _require(_json(directory / "model_identity.json") == model, "Model identity artifact differs")

    baseline = _arrays(directory / "baseline.npz", {"observation_body_ids", "voltage_mv"}, run_identity)
    update = _arrays(directory / "update.npz", {"theta_before", "gradient", "theta", "gain"}, run_identity)
    responses = _arrays(directory / "responses.npz", {"observation_body_ids", "baseline_mv", "updated_mv",
                                                    "restored_zero_mv", "checkpoint_mv"}, run_identity)
    ids = responses["observation_body_ids"]
    _vector(ids, (observations,), "observation_body_ids", dtype="int64")
    _require(np.all(ids[1:] > ids[:-1]) and not np.intersect1d(ids, model["input_body_ids"]).size,
             "Observation IDs must be unique, ordered, and disjoint from inputs")
    _require(np.array_equal(ids, baseline["observation_body_ids"]), "Baseline/response observation IDs differ")
    _vector(baseline["observation_body_ids"], (observations,), "baseline.observation_body_ids", dtype="int64")
    _vector(baseline["voltage_mv"], (observations,), "baseline.voltage_mv")
    for name in ("baseline_mv", "updated_mv", "restored_zero_mv", "checkpoint_mv"):
        _vector(responses[name], (observations,), name)
    _require(np.array_equal(baseline["voltage_mv"], responses["baseline_mv"]), "Baseline response artifacts differ")
    for name in ("theta_before", "gradient", "theta", "gain"):
        _vector(update[name], (groups,), name)
    _require(np.all(update["theta_before"] == protocol["theta_initial"]), "Theta was not initially zero")
    expected_theta = -protocol["learning_rate"] * update["gradient"].astype(np.float64) / (np.abs(update["gradient"]) + 1e-8)
    _require(np.allclose(update["theta"], expected_theta, rtol=2e-6, atol=1e-9), "Theta does not match one registered Adam update")
    expected_gain = np.clip(1 + np.float32(0.5) * np.tanh(update["theta"]),
                            np.nextafter(np.float32(0.5), np.float32(1)), np.nextafter(np.float32(1.5), np.float32(1)))
    _require(np.allclose(update["gain"], expected_gain, rtol=0, atol=2 * np.finfo(np.float32).eps)
             and np.all((update["gain"] > 0.5) & (update["gain"] < 1.5)), "Gain does not match bounded internal theta")

    checkpoint = _json(directory / "checkpoint/checkpoint.json")
    _require(checkpoint.get("schema") == "cnsbridge.internal_gain_checkpoint.v1", "Wrong checkpoint schema")
    _identity(checkpoint, "checkpoint_id")
    _require(checkpoint.get("checkpoint_id") == result.get("checkpoint_id") and checkpoint.get("model") == model
             and checkpoint.get("run_identity") == run_identity, "Checkpoint model/run identity mismatch")
    _require(checkpoint.get("arrays_sha256") == artifact_hashes["checkpoint/parameters.npz"], "Checkpoint parameter digest differs")
    theta = _arrays(directory / "checkpoint/parameters.npz", {"theta"})["theta"]
    _vector(theta, (groups,), "checkpoint.theta")
    _require(checkpoint.get("parameters") == {"theta": {"shape": [groups], "dtype": theta.dtype.str}}
             and np.array_equal(theta, update["theta"]), "Checkpoint and updated theta differ")

    initial, updated = responses["baseline_mv"], responses["updated_mv"]
    atol, rtol = protocol["restoration_atol_mv"], protocol["restoration_rtol"]
    metrics = {"gradient_l2": float(np.linalg.norm(update["gradient"].astype(np.float64))),
               "gradient_nonzero_groups": int(np.count_nonzero(update["gradient"])),
               "theta_nonzero_groups": int(np.count_nonzero(update["theta"])),
               "theta_max_abs_change": float(np.max(np.abs(update["theta"]))),
               "gain_nonzero_groups": int(np.count_nonzero(update["gain"] - 1)),
               "gain_max_abs_change": float(np.max(np.abs(update["gain"] - 1))),
               "response_changed_neurons": int(np.count_nonzero(updated != initial)),
               "response_max_abs_change_mv": float(np.max(np.abs(updated - initial))),
               "loss_before": float(np.mean((initial.astype(np.float64) - protocol["target_final_voltage_mv"]) ** 2)),
               "loss_after": float(np.mean((updated.astype(np.float64) - protocol["target_final_voltage_mv"]) ** 2))}
    for name, value in metrics.items():
        reported = _finite(result[name], name)
        if type(value) is int:
            _require(type(reported) is int and reported == value, f"Inconsistent metric: {name}")
        else:
            _require(math.isclose(reported, value, rel_tol=1e-6, abs_tol=1e-10), f"Inconsistent metric: {name}")
    criteria = {"zero_restoration_matches": bool(np.allclose(initial, responses["restored_zero_mv"], atol=atol, rtol=rtol)),
                "checkpoint_response_matches": bool(np.allclose(updated, responses["checkpoint_mv"], atol=atol, rtol=rtol)),
                "raw_graph_unchanged": result["raw_counts_sha256_after"] == result["graph"]["raw_ordered_counts_sha256"],
                "loss_decreased": metrics["loss_after"] < metrics["loss_before"]}
    for name, value in criteria.items():
        _require(type(result[name]) is bool and result[name] is value, f"Inconsistent criterion: {name}")
    _require(all(criteria[name] for name in ("zero_restoration_matches", "checkpoint_response_matches", "raw_graph_unchanged"))
             and all(metrics[name] > 0 for name in ("gradient_nonzero_groups", "theta_nonzero_groups", "gain_nonzero_groups",
                                                    "response_changed_neurons")), "Numerical success criteria failed")
    return {"schema": "cnsbridge.probe_evidence_validation.v1", **run_identity,
            "status": "validated", "checkpoint_id": checkpoint["checkpoint_id"], "metrics": metrics, "criteria": criteria}
