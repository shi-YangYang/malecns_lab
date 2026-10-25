"""CPU-only observation/continuation contracts; never load or run a full brain."""
from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning import x2_inference as frozen
from malecns_lab.learning import x2_task_path_probe as probe
from malecns_lab.learning import x2_task_rate_causal as runner


def test_new_mode_is_bounded_lazy_and_old_default_configuration_unchanged(tmp_path):
    code = ("import sys; from malecns_lab.learning.x2_task_rate_causal import main; "
            "main(['--mode','task-path']); "
            "assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True)
    value = json.loads(result.stdout)
    assert value["brain_loaded"] is False and value["configuration"] == probe.configuration()
    assert value["configuration"]["wall_seconds"] == 180
    assert value["configuration"]["total_frames"] == 8
    assert value["configuration"]["branch_steps"] == 1000
    old = frozen.ROOT/"data/x2_behavior_short_trial_task_rate_20261024_r1/registration.json"
    if old.exists():
        assert runner.configuration() == frozen.read_json(old)["configuration"]
    data = tmp_path/"data"
    source = data/"archive"
    source.mkdir(parents=True)
    target = data/(probe.PREFIX+"test")
    assert runner.output_directory(target,tmp_path,mode="task-path",inputs=(source,)) == target
    with pytest.raises(ValueError,match="dedicated"):
        runner.output_directory(target,tmp_path)
    with pytest.raises(ValueError,match="dedicated"):
        runner.output_directory(data/(runner.PREFIX+"test"),tmp_path,mode="task-path")
    nested = source/target.name
    nested.mkdir()
    registration = {"root":str(tmp_path),"source_run":str(source),"causal_run":str(source),
        "output":str(nested),"parent_pid":os.getppid(),"configuration":probe.configuration()}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (nested/"registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError,match="direct data"):
        runner.run_worker(nested)
    assert list(nested.iterdir()) == [nested/"registration.json"]
    assert not (data/".continuous-training.lock").exists()


def test_lamina_selection_is_topology_only_with_untouched_fixed_controls():
    from scipy.sparse import csc_matrix
    # Five task inputs; three contacted and three untouched cells per type.
    edges = [(5,0),(11,0),(6,1),(12,1),(7,2),(13,2),(7,3),(13,4)]
    graph = SimpleNamespace(counts=csc_matrix(([-1]*len(edges),tuple(zip(*edges))),shape=(17,17)),
        body_ids=np.arange(100,117),group_names=(None,"L1","L2"),
        group_index=np.array([0]*5+[1]*6+[2]*6))
    encoder = SimpleNamespace(identity={"channels":["task."+x for x in
        ("GREET","SHOW_FORWARD","LEFT","RIGHT","END")]},channel_indices=np.arange(5),
        input_indices=np.arange(5),input_body_ids=tuple(range(100,105)))
    manifest = probe.selection(graph,encoder)
    assert manifest["groups"]["GREET.L1.targets"]["graph_indices"] == [5]
    assert manifest["groups"]["SHOW_FORWARD.L2.targets"]["graph_indices"] == [12]
    assert manifest["groups"]["GREET.L1.controls"]["graph_indices"] == [8]
    assert manifest["groups"]["SHOW_FORWARD.L2.controls"]["graph_indices"] == [14]
    assert manifest["task_input_ports"] == {"GREET":[0],"SHOW_FORWARD":[1]}
    for name,group in manifest["groups"].items():
        assert [manifest["selected_graph_indices"][i] for i in group["columns"]] == group["graph_indices"]
        if name.endswith("controls"):
            assert not set(group["graph_indices"]) & {post for post,_ in edges}


def test_observation_collector_separates_input_events_spikes_and_one_ms_means():
    import torch
    policy = SimpleNamespace(input_indices=torch.tensor([0,1]),dn_indices=torch.tensor([2]))
    state = SimpleNamespace(brain=SimpleNamespace(step=600),event_counts=torch.zeros(2,dtype=torch.int64))
    capture = probe.PathCollector(policy,{"selected_graph_indices":[3]},state)
    spikes = torch.tensor([0.,1.,1.,1.])
    with pytest.raises(ValueError,match="Incomplete"):
        capture.arrays()
    for step in range(1000):
        brain = SimpleNamespace(step=601+step,v=torch.tensor([0.,0.,0.,float(step)]),
                                g=torch.tensor([0.,0.,0.,-float(step)]))
        capture.capture(brain,torch.tensor([step+1,0]),spikes)
    arrays = capture.arrays()
    np.testing.assert_array_equal(arrays["input_spikes"],np.tile([0,10],(100,1)))
    np.testing.assert_array_equal(arrays["input_events"],np.tile([10,0],(100,1)))
    np.testing.assert_array_equal(arrays["dn_spikes"],np.full((100,1),10))
    np.testing.assert_array_equal(arrays["post_spikes"],np.full((100,1),10))
    np.testing.assert_array_equal(arrays["post_v"][:,0],np.arange(100)*10+4.5)
    np.testing.assert_array_equal(arrays["post_g"],-arrays["post_v"])
    assert torch.equal(state.event_counts,torch.zeros(2,dtype=torch.int64))
    assert torch.equal(spikes,torch.tensor([0.,1.,1.,1.]))
    with pytest.raises(ValueError,match="clock"):
        capture.capture(brain,torch.tensor([1001,0]),spikes)


def test_second_chunk_keeps_observation_and_state_continuous_without_reset():
    import torch
    from malecns_lab.brain.trainable import LIFState
    from malecns_lab.learning.x2_behavior_policy import BehaviorState
    initial = BehaviorState(LIFState(torch.zeros(2),torch.zeros(2),torch.zeros(2,dtype=torch.int64),
        (torch.zeros(2),),1100,"fixture"),torch.zeros(1),torch.zeros(1,dtype=torch.int64),1,7,"session","policy")
    observation = object()
    encoded = []
    class Policy:
        dn_indices = torch.tensor([0,1])
        encoder = SimpleNamespace(encode=lambda obs: encoded.append(obs) or np.ones(1,np.float32))
        def validate_state(self,state):
            assert state.brain.step == 600+500*state.frame_index
            assert state.observation_seq == 7 and state.session_id == "session"
        def _step(self,occupancy,brain,phase,counts):
            assert occupancy.tolist() == [1.]
            return replace(brain,v=brain.v+1,step=brain.step+1),phase,counts+1,torch.zeros(2)
        def read_features(self,features):
            assert features.shape == (4,)
            return torch.zeros(5),torch.zeros(3),torch.zeros(1),torch.ones(1)*.5
    output = probe.continue_observation(Policy(),observation,initial)
    np.testing.assert_array_equal(output.features_mv.numpy(),[125.5,125.5,375.5,375.5])
    assert encoded == [observation] and output.state.frame_index == 2
    assert output.state.observation_seq == 7 and output.state.event_counts.item() == 500
    assert initial.brain.step == 1100 and torch.count_nonzero(initial.brain.v) == 0
    assert initial.event_counts.item() == 0


def test_read_only_step_hooks_profile_fixed_offsets_and_restore_after_error(monkeypatch):
    import torch
    from malecns_lab.brain import mps_trainable
    monkeypatch.setattr(torch.mps,"synchronize",lambda: None)
    matvec = lambda *args: args[-1]
    monkeypatch.setattr(mps_trainable,"fixed_mps_matvec",matvec)
    class Brain:
        def _validate_state(self,state):
            assert state.step >= 0
    class Policy:
        def __init__(self):
            self.brain = Brain()
        def _step(self,state):
            self.brain._validate_state(state)
            spikes = mps_trainable.fixed_mps_matvec(None,None,torch.ones(1))
            return SimpleNamespace(step=state.step+1),None,None,spikes
    class Collector:
        steps = 0
        def capture(self,brain,events,spikes):
            assert brain.step == self.steps+1
            self.steps += 1
    policy, collector, timings = Policy(),Collector(),[]
    with pytest.raises(RuntimeError,match="stop"):
        with probe.observed_steps(policy,collector,timings):
            for i in range(501):
                policy._step(SimpleNamespace(step=i))
            raise RuntimeError("stop")
    assert "_step" not in policy.__dict__ and "_validate_state" not in policy.brain.__dict__
    assert mps_trainable.fixed_mps_matvec is matvec
    assert [row["step_offset"] for row in timings] == [0,500]
    for row in timings:
        for segment in probe.configuration()["performance"]["segments"]:
            assert row[segment+"_s"] >= 0
    assert collector.steps == 501


def test_fixed_whole_window_metrics_never_use_a_favorable_single_bin_as_success():
    arrays = {name:np.zeros((2,2,100,2)) for name in
        ("input_events","input_spikes","dn_spikes","post_spikes","post_v","post_g")}
    arrays["features_mv"] = np.zeros((2,2,2,2628))
    # Task responses exist in both states but directions differ; size is no rescue.
    arrays["dn_spikes"][0,1,0,0] = 1
    arrays["dn_spikes"][1,1,99,1] = 100
    arrays["features_mv"][0,1,0,0] = 1
    arrays["features_mv"][1,1,1,1] = 100
    manifest = {"groups":{"fixture":{"columns":[0]}}}
    result = probe.measurements(arrays,manifest,np.zeros(2628),np.ones(2628))
    assert result["first_dn_task_difference_bin_ms"] == [0,99]
    assert result["dn_spikes"]["task_cosine_between_contexts"] == 0
    assert result["route_evidence"]["stop_current_readout_route"] is True
    assert result["route_evidence"]["action_learned"] is False
    assert result["input_events"]["task_cosine_between_contexts"] is None
    arrays["dn_spikes"][1,1,0,0] = 1000
    result = probe.measurements(arrays,manifest,np.zeros(2628),np.ones(2628))
    assert result["route_evidence"]["stop_current_readout_route"] is False
    assert result["route_evidence"]["automatic_followup_training"] is False
    assert result["route_evidence"]["deployment_qualified"] is False
