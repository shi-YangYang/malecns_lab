"""Independent persisted-evidence checks. NumPy only; no model or Torch imports."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import re

import numpy as np

from .joint_probe_contract import (BRANCHES, PARAMETERS, PROTOCOL, RESULT_SCHEMA, array_digest, bound,
                                   read_json, require, select_data, sha256, validate_registration)
from .joint_probe_identity import validate_policy_identity

ARTIFACTS = ("selection.npz", "dn_ids.npz", "policy_identity.json", "initial_parameters.npz", "update.npz",
             "gradients_before_clip.npz", "checkpoint/checkpoint.json", "checkpoint/parameters.npz",
             *(name + suffix for name in BRANCHES for suffix in (".npz", ".json")))


def arrays(path, keys, run_identity=None):
    with np.load(path, allow_pickle=False) as payload:
        expected = set(keys) | set(run_identity or {})
        require(len(payload.files) == len(expected) and set(payload.files) == expected, f"Unexpected arrays: {path}")
        result = {key: payload[key].copy() for key in payload.files}
    for key, value in (run_identity or {}).items():
        require(result[key].shape == () and result[key].dtype.kind == "U" and result.pop(key).item() == value,
                "Artifact run identity mismatch")
    return result


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"Invalid metric: {name}")
    return value


def check_array(value, shape, dtype, name):
    require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
            f"Invalid array shape/dtype/finite values: {name}")


def comparison(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    changed = ~np.isclose(a, b, rtol=PROTOCOL["rtol"], atol=PROTOCOL["atol"])
    return {"exact": bool(np.array_equal(a, b)), "allclose": bool(not np.any(changed)),
            "changed_elements": int(np.count_nonzero(changed)), "max_abs_difference": float(np.max(delta))}


def losses(actions, targets, gains):
    # These are reporting reductions of saved numbers, not a second neural run.
    target = targets.astype(np.float32).astype(np.float64)
    scale = np.array(PROTOCOL["axis_scales"], dtype=np.float32).astype(np.float64)
    per_axis = np.mean(((actions.astype(np.float64) - target) / scale) ** 2, axis=0)
    reg = float(PROTOCOL["regularization"] * np.mean((gains.astype(np.float64) - 1) ** 2))
    return {"action_mse_per_axis": per_axis.tolist(), "action_loss": float(per_axis.mean()),
            "regularization_loss": reg, "total_loss": float(per_axis.mean()) + reg}


def _same_metrics(actual, expected):
    require(set(actual) == set(expected), "Metric fields differ")
    for key, value in expected.items():
        require(np.allclose(actual[key], value, rtol=1e-12, atol=1e-14), f"Recomputed {key} differs")


def _state(record, policy, frame):
    bound(record, "state_id")
    expected_step = policy["precondition_steps"] + frame * policy["frame_steps"]
    require(record["policy_id"] == policy["policy_id"] and record["model_id"] == policy["model"]["model_id"]
            and record["drive_id"] == policy["drive"]["drive_id"]
            and record["spatial_map_id"] == policy["spatial"]["spatial_map_id"]
            and record["brain_step"] == record["drive_step"] == expected_step and record["frame_index"] == frame
            and record["drive_dt_ms"] == policy["model"]["config"]["dt_ms"], "State identity or clock mismatch")
    delay = round(policy["model"]["config"]["delay_ms"] / policy["model"]["config"]["dt_ms"])
    expected = {"v": ((policy["model"]["graph"]["nodes"],), "float32"),
                "g": ((policy["model"]["graph"]["nodes"],), "float32"),
                "refractory_until": ((policy["model"]["graph"]["nodes"],), "int64"),
                "drive_phase": ((len(policy["neurons"]["inputs"]["body_ids"]),), "float64"),
                "drive_event_counts": ((len(policy["neurons"]["inputs"]["body_ids"]),), "int64")}
    expected.update({f"pending_{i}": expected["v"] for i in range(delay)})
    require(set(record["values"]) == set(expected), "Incomplete state digest inventory")
    for name, (shape, dtype) in expected.items():
        item = record["values"][name]
        require(set(item) == {"shape", "dtype", "sha256"} and item["shape"] == list(shape)
                and item["dtype"] == np.dtype(dtype).str and re.fullmatch("[0-9a-f]{64}", item["sha256"]) is not None,
                f"Malformed full-state digest: {name}")


def _parameter_check(values, shapes):
    for key, shape in shapes.items():
        check_array(values[key], shape, "float32", key)
    expected = np.clip(1 + np.float32(0.5) * np.tanh(values["theta"]),
                       np.nextafter(np.float32(0.5), np.float32(1)), np.nextafter(np.float32(1.5), np.float32(1)))
    require(np.allclose(values["gains"], expected, rtol=0, atol=1.2e-7), "Saved gains disagree with theta")
    require(np.count_nonzero(values["rho"]) == 0, "Fixed background changed")


def validate_numeric_evidence(directory, policy, shapes, run_identity, worker):
    """Recompute the numeric experiment. Production wrapper separately requires the full reference.

    The numerical-test suite exercises this reduction using its explicitly test-only
    circuit; this function never grants a production protocol_complete result.
    """
    directory = Path(directory)
    selection = arrays(directory / "selection.npz", {"frames", "time_s", "valid", "teacher_target", "indices"}, run_identity)
    targets = selection["teacher_target"]
    check_array(targets, (2, 3), "float64", "teacher_target")
    require(np.all(np.abs(targets) <= PROTOCOL["axis_scales"]), "Target exceeds registered action bounds")
    initial = arrays(directory / "initial_parameters.npz", shapes, run_identity)
    _parameter_check(initial, shapes)
    require(not np.count_nonzero(initial["theta"]) and np.all(initial["gains"] == 1)
            and not np.count_nonzero(initial["readout_bias"]), "Initial theta/gain/bias differs")
    keys = {f"{when}_{key}" for when in ("before", "after") for key in shapes}
    keys |= {f"{when}_{key}" for when in ("pre_clip", "post_clip") for key in PARAMETERS} | {"reg_gradient"}
    update = arrays(directory / "update.npz", keys, run_identity)
    before = {key: update[f"before_{key}"] for key in shapes}
    after = {key: update[f"after_{key}"] for key in shapes}
    _parameter_check(before, shapes)
    _parameter_check(after, shapes)
    require(all(np.array_equal(before[k], initial[k]) for k in shapes), "Before parameters differ from initialization")
    check_array(update["reg_gradient"], shapes["theta"], "float32", "reg_gradient")
    require(not np.count_nonzero(update["reg_gradient"]), "Regularizer contributed an initial gradient")
    gradient_file = arrays(directory / "gradients_before_clip.npz", set(PARAMETERS) | {"reg_gradient"}, run_identity)
    require(np.array_equal(gradient_file["reg_gradient"], update["reg_gradient"]), "Regularizer gradient files disagree")
    norm_squared = 0.0
    for key in PARAMETERS:
        for prefix in ("pre_clip", "post_clip"):
            check_array(update[f"{prefix}_{key}"], shapes[key], "float32", f"{prefix}_{key}")
        require(np.array_equal(update[f"pre_clip_{key}"], gradient_file[key]), "Pre-clip gradient files disagree")
        norm_squared += np.square(update[f"pre_clip_{key}"].astype(np.float64)).sum()
    norm = math.sqrt(norm_squared)
    recorded_norm = finite(worker["gradient_total_norm"], "gradient_total_norm")
    require(math.isclose(recorded_norm, norm, rel_tol=2e-5, abs_tol=1e-12), "Global gradient norm differs")
    coefficient = float(min(np.float32(1), np.float32(1) / (np.float32(recorded_norm) + np.float32(1e-6))))
    require(worker["clip_coefficient"] == coefficient, "Clip coefficient differs")
    require(worker["optimizer"] == PROTOCOL["optimizer"] and worker["optimizer_steps"] == dict.fromkeys(PARAMETERS, 1.0),
            "Optimizer declaration or one-step count differs")
    for key in PARAMETERS:
        grad = update[f"post_clip_{key}"]
        require(np.array_equal(grad, update[f"pre_clip_{key}"] * np.float32(coefficient)), "Post-clip gradient differs")
        # First Adam step from zero moment state, independently in NumPy.
        lr = 0.001 if key == "theta" else 0.0001
        expected = before[key].astype(np.float64) - lr * grad.astype(np.float64) / (np.abs(grad.astype(np.float64)) + 1e-8)
        require(np.allclose(after[key], expected, rtol=5e-6, atol=1e-9), "Parameters do not match the registered Adam step")
    records, outputs, computed_losses = {}, {}, {}
    for branch in BRANCHES:
        values = arrays(directory / f"{branch}.npz", set(shapes) | {"actions", "features_mv"}, run_identity)
        _parameter_check(values, shapes)
        check_array(values["actions"], (2, 3), "float32", branch + ".actions")
        check_array(values["features_mv"], (2, shapes["readout_weight"][1]), "float32", branch + ".features")
        expected_params = before if branch in ("baseline", "full_restore") else after.copy()
        if branch == "acute_restore":
            expected_params = {**after, "theta": before["theta"], "gains": before["gains"]}
        require(all(np.array_equal(values[k], expected_params[k]) for k in shapes), f"Wrong branch parameters: {branch}")
        features = policy["features"]
        normalized = (values["features_mv"] - features["offset_mv"]) / features["scale_mv"]
        expected_action = np.tanh(normalized @ values["readout_weight"].T + values["readout_bias"])
        expected_action *= np.array(PROTOCOL["axis_scales"], dtype=np.float32)
        require(np.allclose(values["actions"], expected_action, rtol=1e-6, atol=1e-7), "Action does not match neural readout")
        record = read_json(directory / f"{branch}.json")
        require(record == worker["branches"][branch] and record["completed_frames"] == 2, "Branch completion evidence differs")
        _state(record["initial_state"], policy, 0)
        _state(record["end_state"], policy, 2)
        computed_losses[branch] = losses(values["actions"], targets, values["gains"])
        _same_metrics(record["metrics"], computed_losses[branch])
        records[branch], outputs[branch] = record, values
    for key in ("action", "regularization", "total"):
        require(np.isclose(worker["training_losses"][key], computed_losses["baseline"][key + "_loss"], rtol=1e-6, atol=1e-7),
                "Training loss differs from stored baseline")
    require(worker["training_losses"]["regularization"] == 0, "Initial regularizer is nonzero")
    checks = {}
    for label, left, right in (("full_restore", "baseline", "full_restore"), ("checkpoint_reload", "updated", "checkpoint_reload"),
                               ("acute_restore", "updated", "acute_restore")):
        checks[label] = {key: comparison(outputs[left][key], outputs[right][key]) for key in ("actions", "features_mv")}
    for label in ("full_restore", "checkpoint_reload"):
        require(all(v["allclose"] for v in checks[label].values()), "Restoration/reload numerical comparison failed")
    # Branches with theta0 have identical precondition; readout cannot affect state.
    require(records["baseline"]["initial_state"] == records["full_restore"]["initial_state"]
            == records["acute_restore"]["initial_state"], "Fresh theta0 precondition differs")
    require(records["updated"]["initial_state"] == records["checkpoint_reload"]["initial_state"], "Reload precondition differs")
    checkpoint = read_json(directory / "checkpoint/checkpoint.json")
    bound(checkpoint, "checkpoint_id")
    require(checkpoint["schema"] == "cnsbridge.neural_policy_checkpoint.v1" and checkpoint["purpose"] == "deployment_parameters_only"
            and checkpoint["policy"] == policy and checkpoint["run_identity"] == run_identity
            and checkpoint["checkpoint_id"] == worker["checkpoint_id"], "Checkpoint identity differs")
    require(checkpoint["arrays_sha256"] == sha256(directory / "checkpoint/parameters.npz"), "Checkpoint file hash differs")
    saved = arrays(directory / "checkpoint/parameters.npz", PARAMETERS)
    declarations = {key: {"parameter": name, "shape": list(shapes[key]), "dtype": np.dtype("float32").str}
                    for key, name in PARAMETERS.items()}
    require(checkpoint["parameters"] == declarations and all(np.array_equal(saved[k], after[k]) for k in PARAMETERS),
            "Checkpoint parameters differ from after-update arrays")
    criteria = {"theta_task_gradient_nonzero": bool(np.any(update["pre_clip_theta"])),
                "theta_changed": bool(np.any(after["theta"] != before["theta"])),
                "gain_changed": bool(np.any(after["gains"] != before["gains"])),
                "readout_weight_changed": bool(np.any(after["readout_weight"] != before["readout_weight"])),
                "readout_bias_changed": bool(np.any(after["readout_bias"] != before["readout_bias"])),
                "background_fixed": bool(not np.any(after["rho"]))}
    difference = computed_losses["updated"]["action_loss"] - computed_losses["baseline"]["action_loss"]
    acute_difference = computed_losses["updated"]["action_loss"] - computed_losses["acute_restore"]["action_loss"]
    return {"joint_numeric_update": all(criteria.values()), "joint_numeric_update_details": criteria,
            "internal_action_effect": not checks["acute_restore"]["actions"]["allclose"], "comparisons": checks,
            "losses": computed_losses, "action_loss_change": difference,
            "action_loss_direction": "decreased" if difference < 0 else "increased" if difference > 0 else "unchanged",
            "internal_action_loss_change_vs_acute_restore": acute_difference,
            "internal_loss_effect": "improves" if acute_difference < 0 else "worsens" if acute_difference > 0 else "unchanged"}


def validate_resources(directory, registration, supervisor):
    budget = registration["resources"]
    require(supervisor["schema"] == "cnsbridge.resource_run.v1" and supervisor["status"] == "completed"
            and supervisor["returncode"] == 0 and supervisor["reason"] == "worker_exited"
            and supervisor["worker_status"] == "completed", "Supervisor did not collect a clean worker exit")
    wall = finite(supervisor["wall_seconds"], "supervisor.wall_seconds")
    require(0 < wall <= 600, "Supervisor wall budget exceeded")
    with (directory / "resources.csv").open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["wall_seconds", "tree_rss_bytes", "system_available_bytes"], "Resource columns differ")
        rows = [(float(r["wall_seconds"]), int(r["tree_rss_bytes"]), int(r["system_available_bytes"])) for r in reader]
    require(len(rows) >= 2 and all(math.isfinite(t) and 0 <= t <= wall and 0 <= rss <= budget["effective_rss_limit_bytes"]
            and available >= budget["reserve_bytes"] for t, rss, available in rows), "Invalid/breached resource samples")
    require(all(a[0] < b[0] for a, b in zip(rows, rows[1:])), "Resource time did not advance")
    require(supervisor["peak_observed_rss_bytes"] == max(r[1] for r in rows), "Sampled peak differs")
    return {"samples": len(rows), "peak_observed_rss_bytes": max(r[1] for r in rows),
            "minimum_system_available_bytes": min(r[2] for r in rows), "wall_seconds": wall}


def validate_joint_evidence(directory, supervisor):
    directory = Path(directory)
    registration = read_json(directory / "registration.json")
    identity = validate_registration(registration)
    require(registration["resources"]["can_launch"], "Registration did not permit launch")
    worker = read_json(directory / "worker_result.json")
    require(worker["schema"] == RESULT_SCHEMA and worker["status"] == "completed"
            and worker["reason"] == "five_branches_recorded", "Worker did not finish five branches")
    require(all(worker[key] == value for key, value in identity.items())
            and worker["implementation_sha256"] == registration["implementation_sha256"], "Worker run binding mismatch")
    for key in ("device", "torch_threads", "torch_interop_threads"):
        require(worker["environment"][key] == PROTOCOL[key], "Worker execution environment differs")
    for key in ("python", "numpy", "scipy", "torch", "psutil"):
        require(type(worker["environment"][key]) is str and bool(worker["environment"][key]), "Missing worker version")
    require(set(worker["artifacts_sha256"]) == set(ARTIFACTS), "Artifact inventory incomplete")
    for name in ARTIFACTS:
        require(sha256(directory / name) == worker["artifacts_sha256"][name], f"Artifact hash differs: {name}")
    require(sha256(registration["graph"]["path"]) == registration["graph"]["file_sha256"], "Reference file changed")
    policy = read_json(directory / "policy_identity.json")
    shapes = validate_policy_identity(policy, registration)
    for key in ("raw_counts_sha256_before", "raw_counts_sha256_after", "raw_counts_sha256_reload"):
        require(worker[key] == PROTOCOL["raw_ordered_counts_sha256"], "Original graph changed")
    expected_selection, scene = select_data(registration["dataset_path"])
    selection = arrays(directory / "selection.npz", expected_selection, identity)
    require(worker["scene"] == scene and all(np.array_equal(selection[k], v) for k, v in expected_selection.items()),
            "Selected inputs/targets differ from registered same-frame teacher data")
    for key, value in expected_selection.items():
        require(selection[key].dtype == value.dtype and selection[key].shape == value.shape, "Selection dtype/shape differs")
    ids = arrays(directory / "dn_ids.npz", {"body_ids"}, identity)["body_ids"]
    check_array(ids, (PROTOCOL["outputs"],), "int64", "DN IDs")
    require(np.array_equal(ids, policy["features"]["body_ids"]), "DN order differs")
    phases = worker["phases_seconds"]
    expected_phases = {"validate_registration", "load_policy", "backward", "optimizer_step", "checkpoint_save",
                       "verify_raw_graph", "checkpoint_load", "seal_artifacts"}
    expected_phases |= {name + suffix for name in BRANCHES for suffix in ("_precondition", "_forward")}
    require(set(phases) == expected_phases, "Worker phase inventory differs")
    wall = finite(worker["wall_seconds"], "worker wall")
    require(sum(finite(v, k) for k, v in phases.items()) <= wall + 1e-6 and wall <= 600, "Worker phases/wall inconsistent")
    numeric = validate_numeric_evidence(directory, policy, shapes, identity, worker)
    resources = validate_resources(directory, registration, supervisor)
    return {"protocol_complete": True, **numeric, "resources": resources,
            "scope": "one offline two-frame update; not closed-loop learning, generalization or AC3-AC6 completion"}
