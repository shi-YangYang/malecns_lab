"""CPU contracts only; no complete graph, MPS computation or simulator."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from malecns_lab.learning.x2_interaction_data import generate_dataset
from malecns_lab.learning.x2_theta_trial import (
    VALIDATION_FAMILY, VALIDATION_ORDER, VALIDATION_SEED, configuration, intensity_report,
    learning_screen, output_directory, row_ce, train_ce, training_branches, trial_dataset, validation_score,
)


def test_preview_is_lazy_and_fixed_to_registered_budget():
    script = "import sys;from malecns_lab.learning.x2_theta_trial import main;main([]);assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable,"-c",script],check=True,capture_output=True,text=True)
    value = json.loads(result.stdout)
    assert value["brain_loaded"] is False
    config = value["configuration"]
    assert config["total_neural_frames"] == sum(config[k] for k in
        ("prefix_replay_frames","gradient_frames","train_after_frames","validation_before_frames","validation_after_frames")) == 48
    assert config["wall_seconds"] == 600 and config["theta_updates"] == 1
    assert config["validation_seed"] == VALIDATION_SEED == 24102401
    assert config["validation_family"] == VALIDATION_FAMILY
    assert config["automatic_deployment"] is False


def test_real_train_cache_provenance_and_new_continuous_validation():
    source = Path(__file__).resolve().parents[2]/"data/x2_behavior_short_trial_interaction_paired_20261024_r1"
    original = json.loads((source/"dataset.json").read_text()) if source.is_dir() else generate_dataset()
    before = deepcopy(original)
    dataset = trial_dataset(original)
    assert original == before and dataset["splits"]["train"] == original["splits"]["train"]
    val = dataset["splits"]["validation"][0]
    assert len(val["frames"]) == 12 and len(val["requests"]) == 8
    assert [r["task"] for r in val["requests"]] == [task for task,_ in VALIDATION_ORDER]
    assert val["episode_id"] not in {e["episode_id"] for group in original["splits"].values() for e in group}
    assert sum(r["task"] == "SHOW_FORWARD" for r in val["requests"]) == 3
    assert sum(r["task"] == "END_INTERACTION" for r in val["requests"]) == 2
    all_branches = []
    for episode in dataset["splits"]["train"]:
        branches = training_branches(episode)
        all_branches += branches
        assert [len(b["frames"]) for b in branches] == [2,1,1]
        first,second = branches[0]["frames"]
        assert first["observation"]["request_id"] == second["observation"]["request_id"] == second["observation"]["feedback"]["request_id"]
        assert second["observation"]["task"] == "SHOW_FORWARD" and second["target"]["phase"] == "FINISH"
        assert first["target"]["behavior"] == "ADVANCE" and first["target"]["phase"] == "START"
        assert all(b["frames"][0]["target"]["behavior"] == "HOLD" and b["frames"][0]["target"]["phase"] == "FINISH" for b in branches[1:])
        assert all("WAVE_RIGHT" not in f["observation"]["capabilities"] for b in branches for f in b["frames"])
        if source.is_dir():
            contexts = json.loads((source/"paired-observations.json").read_text())["contexts"]
            context = next(c for c in contexts if c["family"] == episode["family_id"] and c["frame_index"] == 4)
            assert [b["frames"][0]["observation"] for b in branches] == [b["observation"] for b in context["branches"]]
    assert len(all_branches) == 6 and sum(len(b["frames"]) for b in all_branches) == 8


def test_new_identity_freezes_six_heads_and_adam_only_mutates_theta():
    import torch
    from malecns_lab.learning.x2_theta_trial_policy import bind_theta_trial_policy, SCHEMA
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.brain = torch.nn.Module()
            self.brain.theta = torch.nn.Parameter(torch.tensor([.2,.4]))
            for name,n in (("behavior_head",5),("phase_head",3),("parameter_head",1)):
                setattr(self,name,torch.nn.Linear(2,n))
            self._identity = {"schema":"malecns_lab.x2_interaction_paired_policy.v1","policy_id":"source",
                "readout_fit":{"theta":"frozen"},"input_map":{"input_id":"unchanged"},"model":{"graph":"unchanged"}}
        @property
        def identity(self):
            return deepcopy(self._identity)
    policy = Toy()
    old = policy.identity
    before = {n:p.detach().clone() for n,p in policy.named_parameters()}
    bind_theta_trial_policy(policy,"checkpoint-source")
    assert policy.identity["schema"] == SCHEMA and policy.identity["parent_policy_id"] == "source"
    assert policy.identity["input_map"] == old["input_map"] and policy.identity["model"] == old["model"]
    assert policy.spike_backward_gain == .01 and policy.identity["readout_fit"]["theta"] != "frozen"
    assert [n for n,p in policy.named_parameters() if p.requires_grad] == ["brain.theta"]
    optimizer = torch.optim.Adam([policy.brain.theta],lr=1e-4)
    # Two connected states expose a gradient to the first boundary; the real
    # runner still reports combined credit, not an isolated final-frame claim.
    first = policy.brain.theta*2
    first.retain_grad()
    second = first+policy.brain.theta
    loss = torch.nn.functional.cross_entropy(policy.behavior_head(first)[None],torch.tensor([1]))
    loss = loss+torch.nn.functional.cross_entropy(policy.phase_head(second)[None],torch.tensor([2]))
    loss.backward()
    assert first.grad is not None and torch.isfinite(policy.brain.theta.grad).all()
    assert all(p.grad is None for n,p in policy.named_parameters() if n != "brain.theta")
    optimizer.step()
    assert not torch.equal(policy.brain.theta,before["brain.theta"])
    assert all(torch.equal(p,before[n]) for n,p in policy.named_parameters() if n != "brain.theta")


def test_screen_requires_heldout_requests_and_all_greet_motion_nonregression():
    before = {"ce":2.,"request_sequence_matches":2,"forward_starts":2,"forward_sequences":1,
        "end_sequences":0,"greet_sequences":0,"greet_false_motion_requests":0}
    after = {**before,"ce":1.9,"request_sequence_matches":3}
    assert learning_screen(2.,1.8,before,after)["supports_further_internal_learning"]
    for change in ({"ce":2.1},{"request_sequence_matches":2},{"forward_starts":1},{"greet_false_motion_requests":1}):
        assert not learning_screen(2.,1.8,before,{**after,**change})["supports_further_internal_learning"]
    assert not learning_screen(2.,2.1,before,after)["supports_further_internal_learning"]
    assert not learning_screen(2.,1.8,before,after)["deployment_qualified"]
    val = trial_dataset(generate_dataset())["splits"]["validation"][0]
    from malecns_lab.control.x2_behavior import BEHAVIORS,PHASES
    rows = []
    for f in val["frames"]:
        t=f["target"]
        rows.append({"behavior":t["behavior"],"phase":t["phase"],"intensity":.5,
            "behavior_logits":[float(str(x)==t["behavior"]) for x in BEHAVIORS],
            "phase_logits":[float(str(x)==t["phase"]) for x in PHASES],"parameter_raw":[0.]})
    rows[0]["behavior"] = "TURN_LEFT"
    score = validation_score(val,rows)
    assert score["greet_false_motion_requests"] == 1 and score["greet_false_advance_requests"] == 0


def test_cpu_ce_recomputation_and_output_guard(tmp_path):
    import torch
    row={"behavior_logits":[1001.,1002.,1003.,1004.,1005.],"phase_logits":[-999.,-1001.,-1000.],"intensity":.6}
    target={"behavior":"ADVANCE","phase":"FINISH","intensity":.5}
    expected = (torch.nn.functional.cross_entropy(torch.tensor(row["behavior_logits"],dtype=torch.float64)[None],torch.tensor([1]))
        +torch.nn.functional.cross_entropy(torch.tensor(row["phase_logits"],dtype=torch.float64)[None],torch.tensor([2]))).item()
    assert row_ce(row,target) == pytest.approx(expected)
    branches = [{"frames":[{"target":target}]*(2 if i%3 == 0 else 1)} for i in range(6)]
    assert train_ce([row]*8,branches) == pytest.approx(expected)
    assert intensity_report([row],[target])["active_target_mse"] is None
    (tmp_path/"data").mkdir()
    with pytest.raises(ValueError):
        output_directory(tmp_path/"data/nested/x2_behavior_short_trial_theta_bad",tmp_path)
    assert not (tmp_path/"data/nested").exists()
