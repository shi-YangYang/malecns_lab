"""Paired-training contracts on CPU only; no graph, GPU, VM or deployment."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from malecns_lab.control.x2_behavior import Observation
from malecns_lab.learning.x2_interaction_data import generate_dataset
from malecns_lab.learning.x2_interaction_paired import (
    PAIR_FRAMES, PAIR_TASKS, VALIDATION_ORDER, branch_observations, configuration,
    fit_readout, output_directory, paired_dataset, pair_targets,
)
from malecns_lab.learning.x2_interaction_paired_policy import ALPHA, SCHEMA, bind_paired_policy


def test_default_generator_preserves_archived_observations_and_labels():
    original=Path(__file__).resolve().parents[2]/"data/x2_behavior_short_trial_interaction_20261023_r1/dataset.json"
    if not original.is_file():
        pytest.skip("Archived local scientific asset unavailable")
    old=json.loads(original.read_text())["splits"]["train"]
    current=json.loads(json.dumps(generate_dataset()))["splits"]["train"]
    assert current==old


def test_new_validation_is_fixed_twelve_frames_and_pair_sources_are_train_only():
    original=generate_dataset()
    dataset=paired_dataset(original)
    assert dataset["splits"]["train"]==original["splits"]["train"]
    episode=dataset["splits"]["validation"][0]
    assert len(episode["frames"])==12 and len(episode["requests"])==8
    assert [r["task"] for r in episode["requests"]]==[task for task,_ in VALIDATION_ORDER]
    assert sum(r["task"]=="SHOW_FORWARD" for r in episode["requests"])==3
    assert sum(r["task"]=="END_INTERACTION" for r in episode["requests"])==2
    assert episode["episode_id"] not in {e["episode_id"] for v in original["splits"].values() for e in v}
    counts=0
    for train in dataset["splits"]["train"]:
        for index in PAIR_FRAMES:
            obs=Observation.from_dict(train["frames"][index]["observation"])
            branches=branch_observations(obs)
            assert tuple(str(o.task) for o in branches)==PAIR_TASKS
            original_branch=branches[0].to_dict()
            assert all({**o.to_dict(),"task":original_branch["task"]}==original_branch for o in branches)
            assert all(o.feedback.heading_error_rad==0. and o.request_changed for o in branches)
            assert [(t["behavior"],t["phase"]) for t in pair_targets(branches)]==[("ADVANCE","START"),("HOLD","FINISH"),("HOLD","FINISH")]
            counts+=len(branches)
    assert counts==18 and configuration()["total_neural_frames"]==24+counts+12==54


def test_fixed_alpha_fit_uses_exactly_42_train_rows():
    rng=np.random.default_rng(24)
    features=rng.normal(-65.,4.,(42,2628))
    targets=[{"behavior":"ADVANCE" if i%3==0 else "HOLD","phase":"START" if i%3==0 else "FINISH","intensity":.5} for i in range(42)]
    fitted=fit_readout(features,targets)
    assert ALPHA==1e-4
    np.testing.assert_allclose(fitted["center"],features.mean(0))
    np.testing.assert_allclose(fitted["scale"],np.maximum(features.std(0),.1))
    scores=((features-fitted["center"])/fitted["scale"])@fitted["weight"]+fitted["bias"]
    assert (scores[:,:5].argmax(1)==np.array([1 if i%3==0 else 0 for i in range(42)])).all()
    with pytest.raises(ValueError,match="Exactly 42"):
        fit_readout(features[:24],targets[:24])


def test_paired_identity_has_separate_stable_file_and_shared_strict_loader(tmp_path,monkeypatch):
    from .test_x2_interaction_trial import PolicyFixture
    from malecns_lab.learning import x2_inference as frozen,x2_behavior_training as training
    from malecns_lab.learning.x2_behavior_policy import X2BehaviorPolicy,backward_identity
    from malecns_lab.learning.x2_interaction_policy import bind_interaction_policy
    from malecns_lab.learning.x2_behavior_session import checkpoint_registration_metadata,load_candidate_policy,MAX_PACKET_BYTES
    base={"schema":"malecns_lab.x2_behavior_policy.v1","policy_id":"base",
        "backward":backward_identity(.1),"initialization":{"theta":"migrated","head_seed":2401}}
    policy=PolicyFixture(base)
    bind_interaction_policy(policy,source_checkpoint_id="a"*64)
    previous=policy.identity
    bind_paired_policy(policy)
    assert policy.identity["schema"]==SCHEMA and policy.identity["readout_fit"]["alpha"]==ALPHA
    assert policy.identity["input_map"]==previous["input_map"]
    assert policy.identity["parent_interaction_policy_id"]==previous["policy_id"]
    metadata={"checkpoint_id":"candidate","policy":policy.identity}
    compact=checkpoint_registration_metadata(metadata)
    assert len(json.dumps(compact).encode())<MAX_PACKET_BYTES and "input_map" not in compact["policy"]
    monkeypatch.setattr(frozen,"read_anchors",lambda *_:{"run":{"policy":{}},"checkpoint":{}})
    monkeypatch.setattr(frozen,"load_parameters",lambda *_:{"theta":None})
    monkeypatch.setattr(X2BehaviorPolicy,"from_reference",lambda *a,**k:PolicyFixture(base))
    def strict(path,actual):
        assert actual.identity==metadata["policy"]
        return metadata
    monkeypatch.setattr(training,"restore_checkpoint",strict)
    assert load_candidate_policy(tmp_path,compact,tmp_path/"checkpoint").identity==policy.identity


def test_preview_and_output_boundary_remain_bounded(tmp_path):
    code=("import sys; from malecns_lab.learning.x2_interaction_paired import main; main([]); "
          "assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    value=json.loads(subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout)
    assert not value["brain_loaded"] and value["configuration"]["wall_seconds"]==600
    (tmp_path/"data").mkdir()
    output=tmp_path/"data/x2_behavior_short_trial_interaction_paired_new"
    assert output_directory(output,tmp_path)==output
    with pytest.raises(ValueError,match="dedicated"):
        output_directory(tmp_path/"data/x2_behavior_short_trial_other",tmp_path)
