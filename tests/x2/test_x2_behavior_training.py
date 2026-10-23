"""Small CPU protocol/archive checks; none constructs a brain or touches MPS."""
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from malecns_lab.learning import x2_behavior_training as training
from malecns_lab.learning.x2_behavior_policy import BehaviorOutput, X2BehaviorPolicy, array_identity


def test_preview_has_no_heavy_import_or_output(tmp_path):
    script = """
import sys
from malecns_lab.experiments.x2_behavior_training import main
assert main(['--output', sys.argv[1]]) == 0
assert not any(x in sys.modules for x in ('torch','numpy','scipy'))
"""
    target = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-c", script, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["brain_loaded"] is False
    assert not target.exists()


def test_probe_registration_cannot_silently_reduce_or_expand():
    cfg = training.contract()
    assert (cfg["precondition_steps"], cfg["decision_steps"], cfg["credit_steps"], cfg["max_updates"], cfg["wall_seconds"]) == (600, 500, 1000, 1, 600)
    for overrides in ({"max_updates": 2}, {"wall_seconds": 601}, {"seed": 1}, {"initialization": "neutral"}):
        with pytest.raises(ValueError):
            training.contract("probe", **overrides)
    with pytest.raises(ValueError):
        training.contract("joint", max_updates=0)
    with pytest.raises(ValueError):
        training.contract("evaluate", max_updates=1)


def test_episode_selection_balances_without_frame_splitting():
    tasks = ("GREET", "FACE_LEFT", "FACE_RIGHT", "SHOW_FORWARD", "END_INTERACTION")
    episodes = [{"task": task, "episode_id": f"{task}{i}", "frames": [1, 2]} for task in tasks for i in range(2)]
    selected = training.select_episodes(episodes, 5)
    assert [e["task"] for e in selected] == list(tasks)
    assert all(any(e is source for source in episodes) for e in selected)


def test_source_inventory_covers_transitive_package_entrypoints():
    inventory = training.source_inventory(Path(__file__).resolve().parents[2])
    assert {"brain/mps_trainable.py", "brain/trainable.py", "control/adapters.py", "brain/connectome.py",
            "senses/x2_behavior.py", "learning/x2_behavior_data.py", "experiments/x2_behavior_training.py"} <= set(inventory)
    assert all(len(value) == 64 for value in inventory.values())


def test_masked_parameter_head_has_expected_zero_gradient():
    behavior = torch.zeros(5, requires_grad=True)
    phase = torch.zeros(3, requires_grad=True)
    raw = torch.zeros(1, requires_grad=True)
    output = BehaviorOutput(behavior, phase, raw, raw.sigmoid(), torch.zeros(2628), None)
    loss = training.loss_for(output, {"behavior": "WAVE_RIGHT", "phase": "START", "intensity": .7})
    loss.backward()
    assert torch.count_nonzero(behavior.grad) and torch.count_nonzero(phase.grad)
    assert raw.grad is not None and torch.count_nonzero(raw.grad) == 0
    raw.grad = None
    output = BehaviorOutput(behavior, phase, raw, raw.sigmoid(), torch.zeros(2628), None)
    training.loss_for(output, {"behavior": "ADVANCE", "phase": "START", "intensity": .7}).backward()
    assert torch.count_nonzero(raw.grad) == 1


def test_read_heads_only_consume_full_neural_feature_vector():
    # A head-only CPU fixture, not a substitute graph or production policy.
    fixture = X2BehaviorPolicy.__new__(X2BehaviorPolicy)
    nn.Module.__init__(fixture)
    fixture.brain = SimpleNamespace(config=SimpleNamespace(v_rest_mv=-52., v_threshold_mv=-45.))
    fixture.behavior_head, fixture.phase_head, fixture.parameter_head = nn.Linear(2628, 5), nn.Linear(2628, 3), nn.Linear(2628, 1)
    feature = torch.full((2628,), -50.)
    before = fixture.read_features(feature)
    fixture.task = "arbitrary_external_task_not_an_input"
    after = fixture.read_features(feature)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    with pytest.raises(ValueError):
        fixture.read_features(torch.zeros(2629))


class ArchiveFixture(nn.Module):
    """Only exercises serialization, never neural dynamics."""
    def __init__(self):
        super().__init__()
        self.brain = nn.Module()
        self.brain.theta = nn.Parameter(torch.zeros(2))
        self.behavior_head, self.phase_head, self.parameter_head = nn.Linear(4, 5), nn.Linear(4, 3), nn.Linear(4, 1)
        identity = {"schema": "explicit_archive_test_fixture", "role": "not_a_brain"}
        self.identity = {**identity, "policy_id": training.frozen.canonical_hash(identity)}

    def validate_parameters(self):
        assert all(torch.isfinite(p).all() for p in self.parameters())


def test_new_checkpoint_roundtrip_and_tamper_rejection(tmp_path):
    policy = ArchiveFixture()
    original = {k: v.detach().clone() for k, v in policy.named_parameters()}
    meta = training.save_checkpoint(tmp_path / "checkpoint", policy, {"role": "archive_fixture"})
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(10)
    restored = training.restore_checkpoint(tmp_path / "checkpoint", policy)
    assert restored["checkpoint_id"] == meta["checkpoint_id"]
    assert all(torch.equal(v, original[k]) for k, v in policy.named_parameters())
    with pytest.raises(FileExistsError):
        training.save_checkpoint(tmp_path / "checkpoint", policy, {})
    path = tmp_path / "checkpoint" / "arrays.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        training.inspect_checkpoint(tmp_path / "checkpoint")


def test_scalar_optimizer_array_keeps_scalar_declaration():
    assert array_identity(np.asarray(1., dtype=np.float32))["shape"] == []


@pytest.mark.parametrize("fault", ["missing_parameter", "nonfinite_parameter", "zero_parameter",
                                  "missing_boundary", "zero_boundary", "nonfinite_boundary", "nonzero_masked"])
def test_rejected_gradient_evidence_exists_before_admission(tmp_path, monkeypatch, fault):
    policy = ArchiveFixture()
    boundary = torch.ones(3, requires_grad=True)
    active = tuple(name for name, _ in policy.named_parameters() if not name.startswith("parameter_head."))
    masked = ("parameter_head.weight", "parameter_head.bias")
    terms = []
    for name, parameter in policy.named_parameters():
        if name == "brain.theta" and fault == "missing_parameter":
            continue
        coefficient = 0 if name in masked or (name == "brain.theta" and fault == "zero_parameter") else 1
        if fault == "nonzero_masked" and name in masked:
            coefficient = 1
        terms.append(parameter.sum() * coefficient)
    if fault == "nonfinite_parameter":
        policy.brain.theta.register_hook(lambda grad: torch.tensor([float("nan"), float("inf")]))
    if fault != "missing_boundary":
        terms.append(boundary.sum() * (0 if fault == "zero_boundary" else 1))
    if fault == "nonfinite_boundary":
        boundary.register_hook(lambda grad: torch.tensor([float("nan"), float("inf"), -float("inf")]))
    folder = tmp_path / "audit"
    original_require, admissions = training.frozen.require, []

    def require_after_persist(condition, message):
        if "gradient" in message:
            saved = json.loads((folder / "audit.json").read_text())
            assert saved["stage"] == "gradient_admission" and saved["backward_completed"]
            assert saved["gradient_arrays"]["sha256"] == training.frozen.sha256(folder / "gradients.npz")
            admissions.append(message)
        original_require(condition, message)

    monkeypatch.setattr(training.frozen, "require", require_after_persist)
    result = {"optimizer_steps": 0}
    optimizer = SimpleNamespace(step=lambda: pytest.fail("Rejected gradient reached optimizer"))
    with pytest.raises(ValueError, match="gradient"):
        training._audited_gradient_update(policy, optimizer, sum(terms), directory=folder,
            result=result, max_updates=1, boundary=boundary, required_nonzero=active, expected_zero=masked)
    audit = json.loads((folder / "audit.json").read_text())
    assert admissions and audit["status"] == "failed" and audit["stage"] == "gradient_admission"
    assert audit["optimizer_steps_before"] == audit["optimizer_steps_after"] == result["optimizer_steps"] == 0
    assert not audit["optimizer_step_attempted"] and not audit["optimizer_step_completed"]
    assert not any(audit["parameter_changed_counts"].values())
    json.dumps(audit, allow_nan=False)
    with np.load(folder / "gradients.npz", allow_pickle=False) as archive:
        if fault == "missing_parameter":
            assert not audit["parameter_gradients"]["brain.theta"]["present"]
            assert "gradient.brain.theta" not in archive.files
        elif fault == "nonfinite_parameter":
            stats = audit["parameter_gradients"]["brain.theta"]
            assert stats["nonfinite_counts"] == {"nan": 1, "positive_inf": 1, "negative_inf": 0}
            assert stats["l2"] is None and not stats["finite"]
            assert np.isnan(archive["gradient.brain.theta"][0]) and np.isposinf(archive["gradient.brain.theta"][1])
        elif fault == "nonfinite_boundary":
            assert audit["first_boundary_v_gradient"]["nonfinite_counts"] == {"nan": 1, "positive_inf": 1, "negative_inf": 1}
            assert np.isneginf(archive["first_boundary_v_gradient"][2])
        elif fault == "missing_boundary":
            assert not audit["first_boundary_v_gradient"]["present"]
            assert "first_boundary_v_gradient" not in archive.files
        elif fault == "zero_boundary":
            assert audit["first_boundary_v_gradient"]["finite"] and audit["first_boundary_v_gradient"]["nonzero"] == 0
    with np.load(folder / "parameters-after.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(dict(policy.named_parameters()))


@pytest.mark.parametrize("outcome", ["no_change", "partial_failure", "completed"])
def test_update_outcome_retains_actual_changes_and_phase(tmp_path, outcome):
    policy = ArchiveFixture()
    active = tuple(name for name, _ in policy.named_parameters() if not name.startswith("parameter_head."))
    masked = ("parameter_head.weight", "parameter_head.bias")
    boundary = torch.ones(3, requires_grad=True)
    loss = sum(p.sum() * (0 if name in masked else 1) for name, p in policy.named_parameters()) + boundary.sum()
    folder = tmp_path / "audit"
    adam = torch.optim.Adam(policy.parameters(), lr=.001)

    def step():
        persisted = json.loads((folder / "audit.json").read_text())
        assert persisted["stage"] == "optimizer_step" and persisted["optimizer_step_attempted"]
        assert persisted["gradient_arrays"]["sha256"] == training.frozen.sha256(folder / "gradients.npz")
        if outcome == "partial_failure":
            with torch.no_grad():
                policy.brain.theta.add_(1)
            raise RuntimeError("step failed after partial change")
        if outcome == "completed":
            adam.step()

    result = {"optimizer_steps": 0}
    arguments = dict(directory=folder, result=result, max_updates=1, boundary=boundary,
                     required_nonzero=active, expected_zero=masked)
    if outcome == "completed":
        training._audited_gradient_update(policy, SimpleNamespace(step=step), loss, **arguments)
    else:
        with pytest.raises((ValueError, RuntimeError), match="expected update|partial change"):
            training._audited_gradient_update(policy, SimpleNamespace(step=step), loss, **arguments)
    audit = json.loads((folder / "audit.json").read_text())
    assert audit["optimizer_steps_after"] == result["optimizer_steps"] == (0 if outcome == "partial_failure" else 1)
    assert audit["optimizer_step_completed"] == (outcome != "partial_failure")
    assert audit["status"] == ("completed" if outcome == "completed" else "failed")
    assert audit["stage"] == {"completed": "completed", "no_change": "parameter_validation", "partial_failure": "optimizer_step"}[outcome]
    assert audit["parameter_changed_counts"]["brain.theta"] == (0 if outcome == "no_change" else 2)
    assert all(audit["parameter_changed_counts"][name] == 0 for name in masked)
    assert all(audit["parameter_gradients"][name]["present"] and audit["parameter_gradients"][name]["nonzero"] == 0 for name in masked)
    assert audit["parameters_after"]["sha256"] == training.frozen.sha256(folder / "parameters-after.npz")


def test_stabilized_spike_is_exact_binary_forward_and_tenth_surrogate_gradient():
    from malecns_lab.brain.mps_trainable import scale_spike_backward
    from malecns_lab.brain.trainable import surrogate_spike
    distance = torch.tensor([-5., -1., -.01, 0., .01, 1., 5.], requires_grad=True)
    spikes = surrogate_spike(distance, 1.)
    assert scale_spike_backward(spikes, 1.) is spikes
    stable = scale_spike_backward(spikes, .1)
    assert torch.equal(stable, spikes) and set(stable.detach().tolist()) == {0., 1.}
    original_gradient, = torch.autograd.grad(spikes.sum(), distance, retain_graph=True)
    stable_gradient, = torch.autograd.grad(stable.sum(), distance)
    assert torch.allclose(stable_gradient, original_gradient * .1, rtol=1e-6, atol=0)
    for invalid in (0., -1., 2., float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            scale_spike_backward(spikes, invalid)


def test_stabilized_contract_and_backward_identity_are_explicit_and_fixed():
    from malecns_lab.learning.x2_behavior_policy import backward_identity
    cfg, original = training.contract("probe-stabilized"), training.contract("probe")
    assert cfg["spike_backward_gain"] == .1 and original["spike_backward_gain"] == 1.
    assert cfg["loss_backward_scale"] == original["loss_backward_scale"] == 1.
    assert (cfg["precondition_steps"], cfg["decision_steps"], cfg["credit_steps"], cfg["checkpoint_block_steps"], cfg["max_updates"], cfg["wall_seconds"]) == (600, 500, 1000, 50, 1, 600)
    assert cfg["paired_reference"] == training.PAIR_DIRECTORY and original["paired_reference"] is None
    assert backward_identity(.1)["backward_id"] != backward_identity(1.)["backward_id"]
    for overrides in ({"spike_backward_gain": 1.}, {"seed": 1}, {"max_updates": 2},
                      {"wall_seconds": 601}, {"initialization": "neutral"}):
        with pytest.raises(ValueError):
            training.contract("probe-stabilized", **overrides)
    with pytest.raises(ValueError):
        training.contract("probe", spike_backward_gain=.1)


def test_stabilized_preview_stays_inert(tmp_path):
    script = """
import sys,json
from malecns_lab.experiments.x2_behavior_training import main
assert main(['--mode','probe-stabilized','--output',sys.argv[1]]) == 0
assert not any(x in sys.modules for x in ('torch','numpy','scipy'))
"""
    target = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "-c", script, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["configuration"]["spike_backward_gain"] == .1
    assert not target.exists()


@pytest.mark.parametrize("kind", ["state", "frame"])
@pytest.mark.parametrize("change", ["none", "one_bit", "same_policy"])
def test_paired_artifacts_require_exact_arrays_and_keep_distinct_identities(tmp_path, kind, change):
    def artifact(name, policy_id, value):
        path = tmp_path / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez(stream, sample=value)
        if kind == "state":
            metadata = {"schema": "state_fixture", "state_id": name, "policy_id": policy_id,
                "session_id": "session", "frame_index": 1, "observation_seq": 0, "neural_step": 1100,
                "arrays": {"sample": array_identity(value)}, "file_sha256": training.frozen.sha256(path)}
        else:
            metadata = {"decision": {"policy_id": policy_id}, "observation": {"fixed": "input"},
                        "arrays_sha256": training.frozen.sha256(path)}
        path.with_suffix(".json").write_text(json.dumps(metadata))
        return path
    original = np.asarray([1., 0., -1.], dtype=np.float32)
    altered = original.copy()
    if change == "one_bit":
        altered[0] = np.nextafter(altered[0], np.float32(2.))
    old = artifact("old", "old-policy", original)
    new_id = "old-policy" if change == "same_policy" else "new-policy"
    new = artifact("new", new_id, altered)
    report, output = {}, tmp_path / "comparison.json"
    if change == "none":
        training.compare_paired_artifact(old, new, report, policy_id=new_id, report_path=output)
    else:
        with pytest.raises(ValueError, match="Paired forward artifact differs"):
            training.compare_paired_artifact(old, new, report, policy_id=new_id, report_path=output)
    saved = json.loads(output.read_text())
    assert saved["all_exact"] is (change == "none")
    assert saved["artifacts"][0]["reference_policy_id"] == "old-policy"
    assert saved["artifacts"][0]["actual_policy_id"] == new_id


def test_paired_initialization_copies_archive_parameters_without_restoring_old_identity(tmp_path):
    from malecns_lab.learning.x2_behavior_policy import backward_identity
    policy = ArchiveFixture()
    base = {"schema": "explicit_archive_test_fixture", "role": "not_a_brain",
        "model": {"model_id": "old-model", "graph": "fixture-only", "implementation_sha256":
                  {"mps_trainable.py": "old-source", "trainable.py": "unchanged"}},
        "implementation_sha256": "old-policy-source"}
    policy.identity = {**base, "policy_id": training.frozen.canonical_hash(base)}
    meta = training.save_checkpoint(tmp_path / "initial-checkpoint", policy, {})
    original = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    base = json.loads(json.dumps(base))
    base["model"]["model_id"] = "new-model"
    base["model"]["implementation_sha256"]["mps_trainable.py"] = "new-source"
    base.update(implementation_sha256="new-policy-source", mps_step_sha256="new-source", backward=backward_identity(.1))
    policy.identity = {**base, "policy_id": training.frozen.canonical_hash(base)}
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(3.)
    paired = {"path": str(tmp_path), "files": {}, "policy_id": meta["policy"]["policy_id"],
              "checkpoint_id": meta["checkpoint_id"]}
    report = training._paired_initialization(policy, paired)
    assert all(torch.equal(parameter, original[name]) for name, parameter in policy.named_parameters())
    assert report["source_policy_id"] != report["new_policy_id"] == policy.identity["policy_id"]
    policy.identity["model"]["graph"] = "different-fixture"
    with pytest.raises(ValueError, match="forward graph/configuration differs"):
        training._paired_initialization(policy, paired)


def test_boundary_hooks_stream_exact_adjoint_without_altering_gradients(tmp_path):
    value = torch.tensor([1., 2.], requires_grad=True)
    voltage, conductance, pending = value * 2, value * 3, value * 4
    audit = training.BoundaryGradientAudit(tmp_path / "boundaries")
    audit.observe(SimpleNamespace(step=650, v=voltage, g=conductance, pending=(pending,)))
    (voltage.sum() * 5 + conductance.sum() * 7).backward()
    summary = audit.finish()
    assert torch.equal(value.grad, torch.full_like(value, 31.))
    assert summary["observed_finite"]
    assert np.array_equal(np.load(tmp_path / "boundaries/000650.v.npy", allow_pickle=False), [5., 5.])
    assert np.array_equal(np.load(tmp_path / "boundaries/000650.g.npy", allow_pickle=False), [7., 7.])
    assert not summary["records"]["000650.pending.00"]["gradient"]["present"]


def test_adam_rejects_finite_but_unsquareable_gradient_before_step(tmp_path):
    policy = ArchiveFixture()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    loss = sum(parameter.sum() * 1e20 for parameter in policy.parameters())
    result = {"optimizer_steps": 0}
    with pytest.raises(ValueError, match="Unsafe float32 Adam gradient scale"):
        training._audited_gradient_update(policy, optimizer, loss, directory=tmp_path / "audit",
            result=result, max_updates=1, check_adam_scale=True)
    audit = json.loads((tmp_path / "audit/audit.json").read_text())
    assert not audit["optimizer_step_attempted"] and result["optimizer_steps"] == 0
    assert not audit["adam_scale"]["safe"] and audit["adam_scale"]["fresh_optimizer"]
    assert all(value["finite"] for value in audit["parameter_gradients"].values())
