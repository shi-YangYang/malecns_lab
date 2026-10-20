"""Explicit numerical-only P3b fixtures; no full graph or device is loaded."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_neural_policy import policy as circuit_policy, frame  # noqa: E402
from malecns_lab.brain.metadata import canonical_hash  # noqa: E402
from malecns_lab.learning.joint_probe import finalize_run, run_registered  # noqa: E402
from malecns_lab.learning.joint_probe_contract import (  # noqa: E402
    ADAM_FIELDS, BRANCHES, CLIP_GROUPS, FINAL_SCHEMA, PARAMETERS, PROTOCOL, RESULT_SCHEMA, SCHEMA,
    array_digest, select_data, sha256, sources, validate_registration,
)
from malecns_lab.learning.joint_probe_evidence import (  # noqa: E402
    arrays, validate_joint_evidence, validate_numeric_evidence, validate_resources,
)
from malecns_lab.learning.joint_probe_identity import validate_policy_identity  # noqa: E402
from malecns_lab.learning.joint_probe_worker import (  # noqa: E402
    clip_gradients, optimizer_for, response, snapshot, state_digest, update_and_intervene,
)
from malecns_lab.learning.joint_probe_optimizer import adam_first_step, validate_optimizer_update  # noqa: E402
from malecns_lab.learning.policy_checkpoint import restore_policy_checkpoint  # noqa: E402
from malecns_lab.learning.resource_guard import GIB, resource_budget, write_json  # noqa: E402


def make_policy():
    return circuit_policy(dtype=torch.float32, background="fixed_lamina_8mv")


def exercise(directory, *, zero_loss=False, small_error=False):
    directory.mkdir()
    model = make_policy()
    identity = {"registration_id": "numerical-test-only", "protocol_id": "not-a-production-protocol"}
    frames = np.stack((frame(), frame(red=False)))
    targets = np.array([[0.3, 0.1, 0.2], [0.29, 0.09, 0.19]], dtype=np.float64)
    if zero_loss or small_error:
        with torch.no_grad():
            state = model.initial_state()
            values = []
            for image in frames:
                out = model.forward_tensor(image, state)
                state = out.state
                values.append(out.action.numpy().copy())
            targets = np.stack(values).astype(np.float64)
            if small_error:
                targets += 1e-6
    np.savez(directory / "selection.npz", frames=frames, time_s=np.array([0.0, 0.1]), valid=np.ones(2, dtype=bool),
             indices=np.array([0, 1], dtype=np.int64), teacher_target=targets, **identity)
    records = []
    original_initial, original_forward = model.initial_state, model.forward_tensor
    previous = None
    initial_calls = 0
    def initial():
        nonlocal initial_calls, previous
        initial_calls += 1
        previous = original_initial()
        assert previous.brain.v.grad_fn is None
        return previous
    def forward(image, state):
        nonlocal previous
        assert state is previous  # Includes exact frame-1 -> frame-2 state: no reset or detach.
        if state.frame_index == 1 and torch.is_grad_enabled():
            assert state.brain.v.grad_fn is not None
        output = original_forward(image, state)
        previous = output.state
        records.append((state.frame_index, output.state.frame_index, torch.is_grad_enabled()))
        return output
    model.initial_state, model.forward_tensor = initial, forward
    result = update_and_intervene(model, frames, targets, directory, identity, lambda *_a, **_kw: None)
    assert initial_calls == 4
    assert records == [(0, 1, True), (1, 2, True)] + [(0, 1, False), (1, 2, False)] * 3
    model_identity = model.identity
    del model
    reloaded = make_policy()
    restore_policy_checkpoint(reloaded, directory / "checkpoint")
    with torch.no_grad():
        action, result["branches"]["checkpoint_reload"] = response(reloaded, frames, targets, "checkpoint_reload", directory,
                                                                   identity, lambda *_a, **_kw: None)
        del action
    shapes = {key: value.shape for key, value in snapshot(reloaded).items()}
    return directory, model_identity, shapes, identity, result


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return exercise(tmp_path_factory.mktemp("joint") / "numerical_only")


def test_real_two_frame_joint_step_continuity_and_five_branches(evidence):
    result = validate_numeric_evidence(*evidence)
    assert result["joint_numeric_update"]
    assert result["joint_numeric_update_details"]["theta_task_gradient_nonzero"]
    assert result["comparisons"]["full_restore"]["actions"]["exact"]
    assert result["comparisons"]["checkpoint_reload"]["features_mv"]["exact"]
    directory, policy, shapes, identity, worker = evidence
    assert set(worker["branches"]) == set(BRANCHES)
    assert all(v == 1 for v in worker["optimizer_steps"].values())
    assert set(result["clipping_groups"]) == {"theta", "readout"}
    assert set(result["adam_states"]) == set(PARAMETERS.values())
    assert worker["training_losses"]["regularization"] == 0
    saved = arrays(directory / "update.npz", {f"{when}_{key}" for when in ("before", "after") for key in shapes} |
                   {f"{when}_{key}" for when in ("pre_clip", "post_clip") for key in PARAMETERS} | {"reg_gradient"}, identity)
    assert np.count_nonzero(saved["pre_clip_theta"])
    assert not np.count_nonzero(saved["reg_gradient"])
    assert policy["role"] == "numerical_test_only"


def test_zero_gradient_is_complete_numeric_evidence_but_not_learning(tmp_path):
    fixture = exercise(tmp_path / "zero_loss", zero_loss=True)
    result = validate_numeric_evidence(*fixture)
    assert result["joint_numeric_update"] is False and result["internal_action_effect"] is False
    assert result["action_loss_direction"] == "unchanged"


def test_loss_increase_is_preserved_not_a_protocol_rejection(tmp_path):
    fixture = exercise(tmp_path / "small_error", small_error=True)
    result = validate_numeric_evidence(*fixture)
    assert result["action_loss_direction"] == "increased"
    assert result["action_loss_change"] > 0


@pytest.mark.parametrize("mutation", ["missing", "target", "gradient", "clip", "theta", "acute_readout", "restore", "state",
                                     "checkpoint", "loss", "nonfinite", "dtype", "extra"])
def test_independent_reduction_rejects_corrupt_evidence(evidence, tmp_path, mutation):
    source, policy, shapes, identity, original_worker = evidence
    directory = tmp_path / mutation
    shutil.copytree(source, directory)
    worker = deepcopy(original_worker)
    if mutation == "missing":
        (directory / "updated.npz").unlink()
    elif mutation == "clip":
        worker["gradient_clipping"]["groups"]["readout"]["coefficient_float32"] *= 0.5
    elif mutation == "loss":
        worker["training_losses"]["action"] += 1
    elif mutation == "state":
        path = directory / "updated.json"
        record = json.loads(path.read_text())
        record["end_state"]["brain_step"] -= 1
        record["end_state"]["state_id"] = canonical_hash({k: v for k, v in record["end_state"].items() if k != "state_id"})
        write_json(path, record)
        worker["branches"]["updated"] = record
    else:
        name = {"target": "selection.npz", "gradient": "update.npz", "theta": "update.npz", "acute_readout": "acute_restore.npz",
                "restore": "full_restore.npz", "checkpoint": "checkpoint/parameters.npz", "nonfinite": "baseline.npz",
                "dtype": "updated.npz", "extra": "update.npz"}[mutation]
        path = directory / name
        with np.load(path, allow_pickle=False) as payload:
            values = {key: payload[key].copy() for key in payload.files}
        if mutation == "target":
            values["teacher_target"] += 0.001
        elif mutation == "gradient":
            values["pre_clip_theta"][0] += 0.5
        elif mutation == "theta":
            values["after_theta"][0] += 0.01
        elif mutation == "acute_readout":
            values["readout_bias"][0] += 0.01
        elif mutation == "restore":
            values["actions"][0, 0] += 0.01
        elif mutation == "checkpoint":
            values["theta"][0] += 0.01
        elif mutation == "nonfinite":
            values["features_mv"][0, 0] = np.nan
        elif mutation == "dtype":
            values["actions"] = values["actions"].astype(np.float64)
        else:
            values["ignored"] = np.array([1])
        np.savez(path, **values)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_numeric_evidence(directory, policy, shapes, identity, worker)


def test_gradient_clipping_is_per_group_with_float32_coefficient():
    model = make_policy()
    for value in model.parameters():
        value.grad = torch.full_like(value, 20)
    before, after, record = clip_gradients(model)
    expected = np.sqrt(sum(np.square(v.astype(float)).sum() for v in before.values()))
    assert record["total_norm_before"] == pytest.approx(expected, rel=1e-12)
    assert record["total_norm_after"] == pytest.approx(np.sqrt(2), rel=2e-6)
    for group, keys in CLIP_GROUPS.items():
        item = record["groups"][group]
        assert item["norm_after"] == pytest.approx(1, rel=1e-6)
        coefficient = min(1, 1 / (item["norm_before"] + 1e-6))
        assert item["coefficient_float64"] == coefficient
        assert item["coefficient_float32"] == float(np.float32(coefficient)) != coefficient
        for key in keys:
            np.testing.assert_array_equal(after[key], before[key] * np.float32(coefficient))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_any_group_is_refused_before_any_mutation(value):
    model = make_policy()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 20)
    original = model.brain.theta.grad.clone()
    model.readout.bias.grad[0] = value  # Last group must be checked before theta is clipped.
    with pytest.raises(ValueError, match="Nonfinite"):
        clip_gradients(model)
    torch.testing.assert_close(model.brain.theta.grad, original, rtol=0, atol=0)


def test_large_finite_gradient_preserves_readout_and_checks_actual_float32_adam():
    # Explicit optimizer arithmetic fixture, no neural graph or production inputs.
    model = torch.nn.Module()
    model.brain = torch.nn.Module()
    model.brain.theta = torch.nn.Parameter(torch.zeros(129))
    model.readout = torch.nn.Linear(32, 3)
    model.brain.theta.grad = torch.logspace(-13, 31, 129)
    model.readout.weight.grad = torch.full_like(model.readout.weight, 20)
    model.readout.bias.grad = torch.full_like(model.readout.bias, -0.25)
    before = {key: dict(model.named_parameters())[name].detach().numpy().copy() for key, name in PARAMETERS.items()}
    pre, post, clipping = clip_gradients(model)
    assert np.isfinite(clipping["total_norm_before"]) and clipping["total_norm_before"] > 1e30
    assert clipping["groups"]["theta"]["coefficient_float64"] < 1e-30
    assert clipping["groups"]["readout"]["coefficient_float64"] > 1e-3
    optimizer = optimizer_for(model)
    optimizer.step()
    params = dict(model.named_parameters())
    after = {key: params[name].detach().numpy().copy() for key, name in PARAMETERS.items()}
    states = {name + "." + field: optimizer.state[params[name]][field].numpy().copy()
              for name in PARAMETERS.values() for field in ADAM_FIELDS}
    changes = {}
    for group, keys in CLIP_GROUPS.items():
        deltas = [after[key].astype(float) - before[key].astype(float) for key in keys]
        changes[group] = {"parameters": [PARAMETERS[key] for key in keys],
                          "changed_elements": sum(np.count_nonzero(v) for v in deltas),
                          "l2": float(np.sqrt(sum(np.square(v).sum() for v in deltas))),
                          "max_abs": max(float(np.abs(v).max()) for v in deltas)}
    worker = {"gradient_clipping": clipping, "parameter_updates": changes, "optimizer": PROTOCOL["optimizer"],
              "optimizer_steps": dict.fromkeys(PARAMETERS, 1.0)}
    update = {**{f"pre_clip_{k}": v for k, v in pre.items()}, **{f"post_clip_{k}": v for k, v in post.items()}}
    checked = validate_optimizer_update(update, states, before, after, {k: v.shape for k, v in before.items()}, worker)
    assert checked["adam_states"]["brain.theta"]["zero_second_moment_with_nonzero_gradient"] > 0
    assert np.all(after["readout_weight"] != before["readout_weight"])
    first, second, predicted = adam_first_step(before["theta"], post["theta"], 0.001)
    assert first.dtype == second.dtype == predicted.dtype == np.float32
    assert np.any((post["theta"] != 0) & (second == 0))


@pytest.mark.parametrize("mutation", ["moment", "second_moment", "step", "step_shape", "dtype", "shape", "name", "nonfinite"])
def test_named_actual_adam_states_are_strict_and_independently_checked(evidence, tmp_path, mutation):
    source, policy, shapes, identity, worker = evidence
    directory = tmp_path / mutation
    shutil.copytree(source, directory)
    path = directory / "adam_state.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {k: payload[k].copy() for k in payload.files}
    if mutation == "moment": values["brain.theta.exp_avg"][0] += 0.01
    elif mutation == "second_moment": values["readout.weight.exp_avg_sq"][0, 0] += 0.01
    elif mutation == "step": values["readout.bias.step"] = np.array(2, dtype=np.float32)
    elif mutation == "step_shape": values["readout.bias.step"] = np.array([1], dtype=np.float32)
    elif mutation == "dtype": values["brain.theta.exp_avg_sq"] = values["brain.theta.exp_avg_sq"].astype(np.float64)
    elif mutation == "shape": values["brain.theta.exp_avg"] = values["brain.theta.exp_avg"][:-1]
    elif mutation == "name": values["arbitrary.exp_avg"] = values.pop("brain.theta.exp_avg")
    else: values["readout.bias.exp_avg"][0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError):
        validate_numeric_evidence(directory, policy, shapes, identity, worker)


def test_state_digest_covers_every_value_and_clock():
    model = make_policy()
    state = model.initial_state()
    original = state_digest(state)
    assert set(original["values"]) == {"v", "g", "refractory_until", "pending_0", "drive_phase", "drive_event_counts"}
    state.brain.pending[0][0] += 1
    assert state_digest(state)["state_id"] != original["state_id"]
    with pytest.raises(ValueError, match="nonfinite"):
        array_digest(np.array([float("nan")]))


def test_production_identity_rejects_numerical_fixture(evidence):
    with pytest.raises(ValueError, match="full_reference"):
        validate_policy_identity(evidence[1], {})


def test_parent_import_and_help_never_import_torch_or_worker():
    code = """
import sys
import malecns_lab.learning.joint_probe as module
assert 'torch' not in sys.modules
assert 'malecns_lab.learning.joint_probe_worker' not in sys.modules
sys.argv = ['joint_probe', '--help']
try: module.main()
except SystemExit as exc: assert exc.code == 0
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def registration_fixture():
    graph = {"path": "/explicit/test-only.npz", "source_id": canonical_hash({}), "provenance": {}, "file_sha256": "0" * 64}
    value = {"schema": SCHEMA, "graph": graph, "dataset_path": "/explicit/test-only-data",
             "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "implementation_sha256": sources(),
             "resources": {**resource_budget(5 * GIB), "wall_limit_seconds": 600, "sampling_seconds": 0.1},
             "dataset_id": PROTOCOL["dataset_id"], "dataset_manifest_sha256": PROTOCOL["dataset_manifest_sha256"]}
    value["registration_id"] = canonical_hash(value)
    return value


def history_fixture(evidence, tmp_path, monkeypatch):
    """Explicit fake history anchors for tests, never a production r1 fallback."""
    from malecns_lab.learning import joint_probe_history as history
    source, policy, _shapes, identity, _worker = evidence
    current, previous = tmp_path / "current", tmp_path / "previous"
    shutil.copytree(source, current)
    write_json(current / "policy_identity.json", policy)
    np.savez(current / "dn_ids.npz", body_ids=np.array(policy["features"]["body_ids"], dtype=np.int64), **identity)
    shutil.copytree(current, previous)
    registration = registration_fixture()
    old = deepcopy(registration)
    old["schema"] = "cnsbridge.joint_probe_registration.v1"
    old["protocol"]["schema"] = "cnsbridge.joint_action_update.v1"
    old["protocol_id"] = canonical_hash(old["protocol"])
    old["registration_id"] = canonical_hash({k: v for k, v in old.items() if k != "registration_id"})
    old_identity = {k: old[k] for k in identity}
    monkeypatch.setattr(history, "R1_ID", old_identity)
    for path in previous.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            values = {k: payload[k].copy() for k in payload.files}
        values.update(old_identity)
        np.savez(path, **values)
    write_json(previous / "registration.json", old)
    write_json(previous / "worker_result.json", {"schema": "cnsbridge.joint_probe_worker.v1", "status": "failed",
                **old_identity, "implementation_sha256": old["implementation_sha256"]})
    for name in ("phase.json", "supervisor_result.json"):
        write_json(previous / name, {"fixture": True})
    for name in ("resources.csv", "worker.log"):
        (previous / name).write_text("explicit test fixture")
    terminal_files = ("registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json",
                      "supervisor_result.json")
    write_json(previous / "result.json", {"schema": "cnsbridge.joint_probe_result.v1", "status": "failed",
                "protocol_complete": False, "joint_numeric_update": None, "internal_action_effect": None,
                "terminal_artifacts_sha256": {name: sha256(previous / name) for name in terminal_files}})
    monkeypatch.setattr(history, "R1_SHA256", {name: sha256(previous / name) for name in history.R1_SHA256})
    binding = history.history_binding(previous, registration["implementation_sha256"], registration["graph"])
    registration["prior_run"] = binding
    registration["registration_id"] = canonical_hash({k: v for k, v in registration.items() if k != "registration_id"})
    return current, previous, identity, registration


def test_r2_registration_binds_old_version_without_requiring_old_optimization_sources(evidence, tmp_path, monkeypatch):
    from malecns_lab.learning import joint_probe_history as history
    current, previous, identity, registration = history_fixture(evidence, tmp_path, monkeypatch)
    assert validate_registration(registration)["protocol_id"] == canonical_hash(PROTOCOL)
    assert history.compare_r1(current, registration["prior_run"], identity)["passed"]
    new_sources = deepcopy(registration["implementation_sha256"])
    new_sources["learning/joint_probe_worker.py"] = "new r2 source"
    assert history.history_binding(previous, new_sources, registration["graph"])
    new_sources["brain/trainable.py"] = "unapproved brain change"
    with pytest.raises(ValueError, match="outside"):
        history.history_binding(previous, new_sources, registration["graph"])


@pytest.mark.parametrize("mutation", ["frozen_npz", "old_registration", "old_terminal", "terminal_file", "id", "graph"])
def test_r1_frozen_anchors_and_terminal_sources_cannot_be_rebound(evidence, tmp_path, monkeypatch, mutation):
    from malecns_lab.learning import joint_probe_history as history
    _current, previous, _identity, registration = history_fixture(evidence, tmp_path, monkeypatch)
    if mutation in ("frozen_npz", "old_registration", "old_terminal", "terminal_file"):
        filename = {"frozen_npz": "baseline.npz", "old_registration": "registration.json", "old_terminal": "result.json",
                    "terminal_file": "worker.log"}[mutation]
        with (previous / filename).open("ab") as stream:
            stream.write(b"tampered")
    elif mutation == "id":
        monkeypatch.setattr(history, "R1_ID", {**history.R1_ID, "protocol_id": "wrong"})
    else:
        registration["graph"]["file_sha256"] = "wrong"
    with pytest.raises(ValueError):
        history.history_binding(previous, registration["implementation_sha256"], registration["graph"])


@pytest.mark.parametrize("mutation", ["initial", "gradient", "features", "actions", "selection", "policy"])
def test_r1_baseline_mismatch_is_saved_and_invalidates_protocol(evidence, tmp_path, monkeypatch, mutation):
    from malecns_lab.learning import joint_probe_history as history
    current, _previous, identity, registration = history_fixture(evidence, tmp_path, monkeypatch)
    if mutation == "policy":
        value = json.loads((current / "policy_identity.json").read_text())
        value["role"] = "different"
        write_json(current / "policy_identity.json", value)
    else:
        filename, field = {"initial": ("initial_parameters.npz", "theta"),
                           "gradient": ("gradients_before_clip.npz", "theta"),
                           "features": ("baseline.npz", "features_mv"), "actions": ("baseline.npz", "actions"),
                           "selection": ("selection.npz", "teacher_target")}[mutation]
        with np.load(current / filename, allow_pickle=False) as payload:
            values = {k: payload[k].copy() for k in payload.files}
        values[field].flat[0] += 1e-8 if mutation == "initial" else 0.5
        np.savez(current / filename, **values)
    report = history.compare_r1(current, registration["prior_run"], identity)
    assert report["passed"] is False
    assert "comparisons" in report and report["rtol"] == 1e-6 and report["atol"] == 1e-7
    # Production worker records the failed comparison before stopping; parent
    # independently recomputes this gate even when worker claims completion.
    write_json(current / "r1_comparison.json", report)
    assert json.loads((current / "r1_comparison.json").read_text())["passed"] is False


def test_r2_versions_and_negative_classification_are_explicit(evidence, tmp_path, monkeypatch):
    import malecns_lab.learning.joint_probe as module
    assert SCHEMA.endswith(".v2") and RESULT_SCHEMA.endswith(".v2") and PROTOCOL["schema"].endswith(".v2")
    monkeypatch.setattr(module, "validate_joint_evidence", lambda *_: {
        "protocol_complete": True, "joint_numeric_update": False, "internal_action_effect": False})
    result = finalize_run(tmp_path, {"status": "completed", "reason": "worker_exited", "returncode": 0})
    assert result["schema"] == FINAL_SCHEMA and result["status"] == "completed"
    assert result["joint_numeric_update"] is False and result["internal_action_effect"] is False


@pytest.mark.parametrize("mutation", ["protocol", "sources", "reserve", "dataset", "relative_path"])
def test_registration_rejects_rehashed_changes(mutation):
    value = deepcopy(registration_fixture())
    if mutation == "protocol":
        value["protocol"]["config"]["frame_ms"] = 20
        value["protocol_id"] = canonical_hash(value["protocol"])
    elif mutation == "sources":
        value["implementation_sha256"]["learning/joint_probe_worker.py"] = "0" * 64
    elif mutation == "reserve":
        value["resources"]["reserve_bytes"] = 0
    elif mutation == "dataset":
        value["dataset_id"] = "new"
    else:
        value["graph"]["path"] = "relative.npz"
    value["registration_id"] = canonical_hash({k: v for k, v in value.items() if k != "registration_id"})
    with pytest.raises(ValueError):
        validate_registration(value)


def test_invalid_dataset_never_launches_worker_and_retains_failure(tmp_path, monkeypatch):
    import malecns_lab.learning.joint_probe as module
    def forbidden(*_a, **_k):
        raise AssertionError("worker launched")
    monkeypatch.setattr(module, "supervise", forbidden)
    result = run_registered(tmp_path / "absent-graph", tmp_path / "absent-data", tmp_path / "run")
    assert result["reason"] == "registration_failed" and result["protocol_complete"] is False
    assert (tmp_path / "run/result.json").exists()
    with pytest.raises(ValueError, match="new directory"):
        run_registered(tmp_path / "absent", tmp_path / "absent", tmp_path / "run")


def test_exact_teacher_selection_does_not_take_executed_or_next_frame(monkeypatch, tmp_path):
    from malecns_lab.learning import joint_probe_contract as contract
    from malecns_lab.learning import neural_data
    from types import SimpleNamespace
    manifest = tmp_path / "manifest.json"
    manifest.write_text("fixture")
    sequence = SimpleNamespace(scene={"scene_id": "train-sine-combined"}, frames=np.zeros((3, 120, 160, 3), dtype=np.uint8),
        time_s=np.array([0, 0.1, 0.2]), valid=np.ones(3, dtype=bool),
        teacher_target=np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0, 0.5]]),
        executed_action=np.ones((3, 3)) * -10)
    data = SimpleNamespace(dataset_id=PROTOCOL["dataset_id"], identity={"camera": PROTOCOL["camera"], "dt_s": 0.1},
                           sequences=(sequence,))
    monkeypatch.setattr(contract, "sha256", lambda _p: PROTOCOL["dataset_manifest_sha256"])
    monkeypatch.setattr(neural_data, "read_neural_data", lambda _p: data)
    selected, _ = select_data(tmp_path)
    np.testing.assert_array_equal(selected["teacher_target"], sequence.teacher_target[:2])
    assert not np.array_equal(selected["teacher_target"], sequence.executed_action[:2])
    sequence.valid[0] = False
    with pytest.raises(ValueError, match="invalid"):
        select_data(tmp_path)


def test_supervision_failure_and_invalid_worker_artifacts_are_never_promoted(tmp_path):
    outcome = {"status": "failed", "reason": "rss_limit", "returncode": -15}
    result = finalize_run(tmp_path, outcome)
    assert result["protocol_complete"] is False and result["joint_numeric_update"] is None
    assert json.loads((tmp_path / "supervisor_result.json").read_text()) == outcome
    result = finalize_run(tmp_path, {"status": "completed", "reason": "worker_exited", "returncode": 0})
    assert result["reason"] == "invalid_joint_evidence" and result["protocol_complete"] is False


def test_parent_launch_binds_exact_guarded_worker_arguments(tmp_path, monkeypatch):
    import malecns_lab.learning.joint_probe as module
    registration = registration_fixture()
    calls = []
    def register(_g, _d, directory):
        directory.mkdir()
        write_json(directory / "registration.json", registration)
        return registration
    def supervise(command, directory, **kwargs):
        calls.append((command, kwargs))
        return {"status": "failed", "reason": "wall_limit"}
    monkeypatch.setattr(module, "register", register)
    monkeypatch.setattr(module, "supervise", supervise)
    result = run_registered(tmp_path / "graph", tmp_path / "data", tmp_path / "run")
    assert result["reason"] == "wall_limit"
    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "malecns_lab.learning.joint_probe", "--worker", str(tmp_path / "run")]
    assert kwargs == {"rss_limit_bytes": 4 * GIB, "wall_seconds": 600, "reserve_bytes": GIB, "poll_seconds": 0.1}


@pytest.mark.parametrize("mutation", [None, "rss", "reserve", "wall", "peak", "returncode", "backwards"])
def test_resource_evidence_is_independently_checked(tmp_path, mutation):
    reg = registration_fixture()
    result = {"schema": "cnsbridge.resource_run.v1", "status": "completed", "returncode": 0,
              "reason": "worker_exited", "worker_status": "completed", "wall_seconds": 0.2, "peak_observed_rss_bytes": 200}
    rows = [[0.01, 100, 2 * GIB], [0.11, 200, 2 * GIB]]
    if mutation == "rss": rows[1][1] = 7 * GIB
    elif mutation == "reserve": rows[1][2] = GIB - 1
    elif mutation == "wall": result["wall_seconds"] = 601
    elif mutation == "peak": result["peak_observed_rss_bytes"] = 199
    elif mutation == "returncode": result["returncode"] = 1
    elif mutation == "backwards": rows[1][0] = 0
    (tmp_path / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n" +
                                           "\n".join(",".join(map(str, row)) for row in rows))
    if mutation:
        with pytest.raises(ValueError):
            validate_resources(tmp_path, reg, result)
    else:
        assert validate_resources(tmp_path, reg, result)["samples"] == 2
