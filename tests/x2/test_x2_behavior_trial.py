"""Small offline checks for bounded trial selection and coordinate migration."""
import json
import os

import numpy as np
import pytest
import torch

from malecns_lab.brain.mps_trainable import scale_spike_backward
from malecns_lab.learning.x2_behavior_data import generate_dataset
from malecns_lab.learning.x2_behavior_trial import TASKS, configuration, select_prefixes, output_directory
from malecns_lab.learning.x2_behavior_trial_policy import affine_head, normalization


def test_prefixes_are_balanced_normal_and_family_disjoint():
    data = generate_dataset()["splits"]
    train = select_prefixes(data["train"], "family-007")
    validation = select_prefixes(data["validation"], "family-107")
    assert [e["task"] for e in train] == list(TASKS)
    assert not {e["family_id"] for e in train} & {e["family_id"] for e in validation}
    assert all(len(e["frames"]) == 2 and e["context"] == "normal" for e in train+validation)
    assert [e["frames"][0]["target"]["behavior"] for e in train] == ["WAVE_RIGHT", "TURN_LEFT", "TURN_RIGHT", "ADVANCE", "HOLD"]
    with pytest.raises(ValueError):
        select_prefixes(data["train"], "family-107")


def test_affine_normalization_preserves_existing_head_not_teacher_answers():
    rng = np.random.default_rng(2401)
    train = rng.normal(-80, 50, (12, 2628)).astype(np.float32)
    center, scale = normalization(train)
    weight = rng.normal(0, .01, (5, 2628)).astype(np.float32)
    bias = rng.normal(0, .01, 5).astype(np.float32)
    new_weight, new_bias = affine_head(weight, bias, center, scale)
    unseen = rng.normal(-75, 40, (3, 2628))
    expected = ((unseen+52)/7)@weight.T+bias
    actual = ((unseen-center)/scale)@new_weight.T+new_bias
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    _, constant_scale = normalization(np.full((2, 2628), -52., dtype=np.float32))
    assert (constant_scale == np.float32(.1)).all()


def test_new_surrogate_gain_changes_only_backward():
    source = torch.tensor([0., 1., 1., 0.], requires_grad=True)
    actual = scale_spike_backward(source, .01)
    assert torch.equal(actual.detach(), source.detach())
    actual.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, .01))


def test_trial_budget_and_credit_are_fixed():
    config = configuration()
    assert config["wall_seconds"] == 600
    assert config["credit_steps"] == config["decision_steps"]*config["prefix_frames"] == 1000
    assert config["max_updates"] == 1
    assert config["system_reserve_gib"] == 1
    assert config["locked_test_opened"] is False


def test_trial_outputs_are_new_direct_children_separate_from_archives(tmp_path, monkeypatch):
    from malecns_lab.learning import x2_behavior_trial as trial
    data = tmp_path / "data"
    data.mkdir()
    cache, checkpoint = data / "old-cache", data / "old-checkpoint"
    cache.mkdir()
    checkpoint.mkdir()
    existing = data / "x2_behavior_short_trial_existing"
    existing.mkdir()
    linked = data / "x2_behavior_short_trial_link"
    linked.symlink_to(cache, target_is_directory=True)
    dangling = data / "x2_behavior_short_trial_dangling"
    dangling.symlink_to(data / "missing", target_is_directory=True)
    invalid = (cache / "x2_behavior_short_trial_new", checkpoint / "x2_behavior_short_trial_new",
               existing, linked, dangling, data / "unprefixed", data / "x2_behavior_short_trial_")
    monkeypatch.setattr(trial.frozen, "numerical_lock", lambda *_: pytest.fail("Invalid output reached registration"))
    for output in invalid:
        with pytest.raises(ValueError):
            trial.execute(output, checkpoint, cache, root=tmp_path)
    fresh = data / "x2_behavior_short_trial_new"
    assert output_directory(fresh, tmp_path, inputs=(cache, checkpoint)) == fresh
    for overlapping in (fresh, fresh / "checkpoint", data):
        with pytest.raises(ValueError, match="archived input"):
            output_directory(fresh, tmp_path, inputs=(overlapping,))
    assert not fresh.exists() and not list(cache.iterdir()) and not list(checkpoint.iterdir())


def test_worker_rechecks_output_boundary_before_any_writes(tmp_path):
    from malecns_lab.learning import x2_behavior_trial as trial
    directory = tmp_path / "data/old-cache/x2_behavior_short_trial_nested"
    directory.mkdir(parents=True)
    reg = {"parent_pid": os.getppid(), "output": str(directory), "root": str(tmp_path),
           "checkpoint": str(tmp_path / "data/checkpoint"), "cache": str(directory.parent)}
    reg["registration_id"] = trial.frozen.canonical_hash(reg)
    (directory / "registration.json").write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="direct data/ child"):
        trial.run_worker(directory)
    assert [p.name for p in directory.iterdir()] == ["registration.json"]
    assert not (tmp_path / "data/.continuous-training.lock").exists()


def test_boundary_gradient_claim_does_not_isolate_final_frame_credit():
    assert "combined two-frame loss" in configuration()["boundary_gradient_claim"]
    assert "final-frame contribution not isolated" in configuration()["boundary_gradient_claim"]
