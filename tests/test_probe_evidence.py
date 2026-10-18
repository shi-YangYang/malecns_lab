"""Fabricated artifact bundles test validation only, never a brain/full-graph run.

All arrays below are hand-authored software fixtures. A full-reference-shaped
identity permits exercising the normal validator, but it is not a graph asset
and cannot be loaded by the production full-reference loader.
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cnsbridge.learning.probe import PROTOCOL, finalize_run, register
from cnsbridge.learning.probe_evidence import (ARTIFACTS, PHASES, RESULT_SCHEMA, canonical_hash, file_sha256,
                                              validate_probe_evidence, validate_registration)
from cnsbridge.learning.resource_guard import GIB, supervise, write_json


def rehash(value, key):
    value.pop(key, None)
    value[key] = canonical_hash(value)
    return value


def software_evidence(tmp_path, monkeypatch):
    import cnsbridge.learning.probe as probe
    monkeypatch.setattr(probe.psutil, "virtual_memory", lambda: SimpleNamespace(available=12 * GIB))
    directory = tmp_path / "software_fixture_only"
    registration = register(tmp_path / "NOT_A_REAL_GRAPH.npz", directory)
    run_id = {key: registration[key] for key in ("registration_id", "protocol_id")}
    graph = rehash({"role": "full_reference", "reference_source_id": "fabricated_software_fixture_not_a_brain",
                    "reference_graph_sha256": "a" * 64, "reference_provenance": {"purpose": "software_fixture_only"},
                    "raw_ordered_counts_sha256": "b" * 64, "nodes": 2, "edges": 1, "synapse_count": 3,
                    "grouping": {"schema": "cnsbridge.presynaptic_raw_type.v1", "names": [None, "L1", "R1-R6"],
                                 "missing_index": 0, "mapping_sha256": "c" * 64}}, "training_graph_id")
    model = rehash({"schema": "cnsbridge.trainable_lif.v1", "graph": graph, "input_body_ids": [101],
                    "dtype": "torch.float32", "backend": "torch_cpu_csr",
                    "config": {"dt_ms": 0.1, "v_rest_mv": -52.0, "v_reset_mv": -52.0, "v_threshold_mv": -45.0,
                               "tau_m_ms": 20.0, "tau_syn_ms": 5.0, "refractory_ms": 2.2, "delay_ms": 1.8,
                               "weight_mv": 0.275, "surrogate_scale_mv": 1.0},
                    "implementation_sha256": {name: file_sha256(Path(probe.__file__).parent.parent / "brain" / name)
                                              for name in ("trainable.py", "trainable_graph.py")}}, "model_id")
    write_json(directory / "model_identity.json", model)
    gradient = np.array([0, 0, 0.5], dtype=np.float32)
    theta = (-PROTOCOL["learning_rate"] * gradient / (np.abs(gradient) + 1e-8)).astype(np.float32)
    gain = 1 + np.float32(0.5) * np.tanh(theta)
    ids = np.array([202], dtype=np.int64)
    baseline, updated = np.array([-52], dtype=np.float32), np.array([-51], dtype=np.float32)
    np.savez(directory / "baseline.npz", observation_body_ids=ids, voltage_mv=baseline, **run_id)
    np.savez(directory / "update.npz", theta_before=np.zeros(3, dtype=np.float32), gradient=gradient, theta=theta,
             gain=gain, **run_id)
    np.savez(directory / "responses.npz", observation_body_ids=ids, baseline_mv=baseline, updated_mv=updated,
             restored_zero_mv=baseline, checkpoint_mv=updated, **run_id)
    checkpoint_dir = directory / "checkpoint"
    checkpoint_dir.mkdir()
    np.savez(checkpoint_dir / "parameters.npz", theta=theta)
    checkpoint = rehash({"schema": "cnsbridge.internal_gain_checkpoint.v1", "model": model, "run_identity": run_id,
                         "parameters": {"theta": {"shape": [3], "dtype": theta.dtype.str}},
                         "arrays_sha256": file_sha256(checkpoint_dir / "parameters.npz")}, "checkpoint_id")
    write_json(checkpoint_dir / "checkpoint.json", checkpoint)
    result = {"schema": RESULT_SCHEMA, **run_id, "implementation_sha256": registration["implementation_sha256"],
              "graph_path": registration["graph_path"], "status": "passed", "reason": "numeric_internal_update_verified",
              "environment": {"python": sys.version, "numpy": np.__version__, "scipy": "software-fixture", "torch": "software-fixture",
                              "psutil": "software-fixture", "device": "cpu", "torch_threads": 4,
                              "evidence_purpose": "fabricated artifact validation fixture, no brain computation"},
              "phases_seconds": {name: 0.001 for name in PHASES}, "wall_seconds": 0.05,
              "simulated_ms_per_forward": 20.0, "graph": graph, "model": model,
              "input_neurons": 1, "observation_neurons": 1, "gradient_l2": 0.5, "gradient_nonzero_groups": 1,
              "theta_nonzero_groups": 1, "theta_max_abs_change": float(np.max(np.abs(theta))),
              "gain_nonzero_groups": 1, "gain_max_abs_change": float(np.max(np.abs(gain - 1))),
              "response_changed_neurons": 1, "response_max_abs_change_mv": 1.0, "loss_before": 4.0, "loss_after": 1.0,
              "loss_decreased": True, "zero_restoration_matches": True, "checkpoint_response_matches": True,
              "raw_graph_unchanged": True, "raw_counts_sha256_after": graph["raw_ordered_counts_sha256"],
              "checkpoint_id": checkpoint["checkpoint_id"]}
    write_result(directory, result)
    return directory, registration, result


def write_result(directory, result):
    result["artifacts_sha256"] = {name: file_sha256(directory / name) for name in ARTIFACTS}
    write_json(directory / "worker_result.json", result)


def rewrite_array(path, mutate):
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key].copy() for key in payload.files}
    mutate(values)
    np.savez(path, **values)


def test_complete_software_fixture_needs_all_evidence_before_passed(tmp_path, monkeypatch):
    directory, registration, result = software_evidence(tmp_path, monkeypatch)
    outcome = supervise([sys.executable, "-c", "pass"], directory, rss_limit_bytes=GIB,
                        reserve_bytes=0, wall_seconds=10, poll_seconds=0.02)
    assert outcome["status"] == "completed"
    final = finalize_run(directory, outcome)
    assert final["status"] == "passed"
    assert final["evidence_validation"]["registration_id"] == registration["registration_id"]
    assert final["evidence_validation"]["metrics"]["loss_before"] == 4.0
    assert final["evidence_validation"]["checkpoint_id"] == result["checkpoint_id"]
    assert json.loads((directory / "result.json").read_text()) == final


@pytest.mark.parametrize("defect,reason", [
    ("schema", "schema"), ("false_restoration", "criterion"), ("raw_graph_changed", "criterion"),
    ("missing_checkpoint", "FileNotFoundError"), ("missing_responses", "FileNotFoundError"),
    ("run_identity", "identity mismatch"), ("artifact_identity", "run identity"),
    ("checkpoint_identity", "Checkpoint model/run"), ("wrong_metrics", "Inconsistent metric"),
    ("nonfinite", "Invalid evidence vector"), ("wrong_dtype", "Invalid evidence vector"),
    ("wrong_shape", "Invalid evidence vector"), ("stale_digest", "Artifact digest"),
    ("restore_array", "Inconsistent criterion"), ("checkpoint_array", "Inconsistent criterion"),
    ("adam_update", "Adam update"), ("missing_phase", "phases"),
    ("numerical_role", "full_reference"), ("no_update", "Numerical success"),
])
def test_incomplete_or_inconsistent_evidence_never_passes(tmp_path, monkeypatch, defect, reason):
    directory, _, result = software_evidence(tmp_path, monkeypatch)
    if defect == "schema":
        result["schema"] = "wrong-schema"
    elif defect == "false_restoration":
        result["zero_restoration_matches"] = False
    elif defect == "raw_graph_changed":
        result["raw_counts_sha256_after"] = "d" * 64
    elif defect.startswith("missing_") and defect != "missing_phase":
        (directory / ("checkpoint/checkpoint.json" if defect == "missing_checkpoint" else "responses.npz")).unlink()
    elif defect == "run_identity":
        result["registration_id"] = "other-run"
    elif defect == "artifact_identity":
        rewrite_array(directory / "responses.npz", lambda values: values.update(registration_id=np.array("other-run")))
    elif defect == "checkpoint_identity":
        path = directory / "checkpoint/checkpoint.json"
        value = json.loads(path.read_text())
        value["run_identity"]["registration_id"] = "other-run"
        rehash(value, "checkpoint_id")
        result["checkpoint_id"] = value["checkpoint_id"]
        write_json(path, value)
    elif defect == "wrong_metrics":
        result["gradient_l2"] = 1.0
    elif defect in ("nonfinite", "wrong_dtype", "wrong_shape", "restore_array", "checkpoint_array", "stale_digest"):
        key = "restored_zero_mv" if defect == "restore_array" else "checkpoint_mv"
        value = np.array([np.nan] if defect == "nonfinite" else [-48, -49] if defect == "wrong_shape" else [-48],
                         dtype=np.float64 if defect == "wrong_dtype" else np.float32)
        rewrite_array(directory / "responses.npz", lambda values: values.update({key: value}))
    elif defect == "adam_update":
        rewrite_array(directory / "update.npz", lambda values: values.update(theta=np.array([0, 0, 0.1], dtype=np.float32)))
    elif defect == "missing_phase":
        result["phases_seconds"].pop("backward")
    elif defect == "numerical_role":
        result["graph"]["role"] = "numerical_test_only"
        rehash(result["graph"], "training_graph_id")
        rehash(result["model"], "model_id")
    elif defect == "no_update":
        # An internally consistent zero-gradient/no-change run is still not success.
        rewrite_array(directory / "update.npz", lambda values: values.update(gradient=np.zeros(3, np.float32),
                       theta=np.zeros(3, np.float32), gain=np.ones(3, np.float32)))
        rewrite_array(directory / "checkpoint/parameters.npz", lambda values: values.update(theta=np.zeros(3, np.float32)))
        path = directory / "checkpoint/checkpoint.json"
        value = json.loads(path.read_text())
        value["arrays_sha256"] = file_sha256(directory / "checkpoint/parameters.npz")
        rehash(value, "checkpoint_id")
        result["checkpoint_id"] = value["checkpoint_id"]
        write_json(path, value)
        for name in ("gradient_nonzero_groups", "theta_nonzero_groups", "gain_nonzero_groups"):
            result[name] = 0
        for name in ("gradient_l2", "theta_max_abs_change", "gain_max_abs_change"):
            result[name] = 0.0
    if defect not in ("missing_checkpoint", "missing_responses", "stale_digest"):
        write_result(directory, result)
    else:
        write_json(directory / "worker_result.json", result)
    outcome = {"status": "completed", "reason": "worker_exited", "returncode": 0}
    final = finalize_run(directory, outcome)
    assert final["status"] == "failed" and final["reason"] == "invalid_probe_evidence"
    assert reason in final["detail"]


@pytest.mark.parametrize("defect,reason", [("hash", "registration_id"), ("protocol", "protocol differs"),
                                         ("implementation", "implementation changed"), ("budget", "resource protocol"),
                                         ("reserve_below", "resource protocol"), ("reserve_above", "resource protocol")])
def test_registration_binds_protocol_implementation_and_budget(tmp_path, monkeypatch, defect, reason):
    _, registration, _ = software_evidence(tmp_path, monkeypatch)
    value = deepcopy(registration)
    if defect in ("hash", "protocol"):
        value["protocol"]["steps"] = 201
    elif defect == "implementation":
        value["implementation_sha256"]["probe.py"] = "wrong"
    elif defect.startswith("reserve_"):
        value["resources"]["reserve_bytes"] = GIB - 1 if defect == "reserve_below" else 2 * GIB
    else:
        value["resources"]["wall_limit_seconds"] = 601
    if defect != "hash":
        rehash(value, "registration_id")
    with pytest.raises(ValueError, match=reason):
        validate_registration(value, PROTOCOL)


def test_status_only_report_and_process_failure_cannot_be_promoted(tmp_path):
    write_json(tmp_path / "worker_result.json", {"status": "passed"})
    failed = finalize_run(tmp_path, {"status": "completed", "reason": "worker_exited", "returncode": 0})
    assert failed["reason"] == "invalid_probe_evidence"
    original = {"status": "failed", "reason": "rss_limit", "returncode": -15}
    assert finalize_run(tmp_path, original) == original


def test_evidence_module_does_not_import_torch_or_brain_loader():
    code = ("import sys; from cnsbridge.learning import probe_evidence; "
            "assert 'torch' not in sys.modules; assert 'cnsbridge.brain.reference' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
