"""CPU contracts for the one bounded polarity diagnostic; no real neural steps."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning import x2_inference as frozen
from malecns_lab.learning import x2_task_path_probe as path
from malecns_lab.learning import x2_task_rate_causal as runner
from malecns_lab.learning.x2_task_polarity_policy import TaskPolarityEncoder, bind_task_polarity_policy, SCHEMA
from malecns_lab.senses.x2_behavior import CHANNELS


def original_encoder():
    class Encoder:
        channel_indices = tuple(c for c in range(57) for _ in range(60 if c < 14 else 59))
        input_indices = tuple(range(3377))
        input_body_ids = tuple(range(3377))
        @property
        def identity(self):
            payload = {"schema":"malecns_lab.x2_behavior_input.no_absolute_yaw.v1",
                "channels":list(CHANNELS),"channel_indices":list(self.channel_indices),
                "max_rate_hz":150.,"event_voltage_mv":68.75}
            return {**payload,"input_map_id":frozen.canonical_hash(payload)}
        def encode(self,task):
            values = np.linspace(0,1,3377,dtype=np.float32)
            for i,channel in enumerate(self.channel_indices):
                if CHANNELS[channel].startswith("task."):
                    values[i] = float(CHANNELS[channel] == "task."+task)
            return values
    return Encoder()


def test_preview_and_prior_default_configs_are_unchanged_and_guard_precedes_writes(tmp_path):
    code = ("import sys; from malecns_lab.learning.x2_task_rate_causal import main; "
        "main(['--mode','task-polarity']); assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    value = json.loads(subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout)
    config = value["configuration"]
    assert not value["brain_loaded"] and config == path.configuration("task-polarity")
    assert config["total_frames"] == 8 and config["baseline_replay_frames"] == 0
    assert config["wall_seconds"] == 180 and config["actual_event_equality_assumed"] is False
    assert config["interpretation"]["minimum_cross_context_cosine"] == .2
    for mode,name in (("rate","task_rate"),("task-path","task_path")):
        old = frozen.ROOT/f"data/x2_behavior_short_trial_{name}_20261024_r1/registration.json"
        if old.exists():
            assert runner.configuration(mode) == frozen.read_json(old)["configuration"]
    source = tmp_path/"data/archive"
    source.mkdir(parents=True)
    output = source.parent/(path.POLARITY_PREFIX+"test")
    assert runner.output_directory(output,tmp_path,mode="task-polarity",inputs=(source,)) == output
    with pytest.raises(ValueError,match="dedicated"):
        runner.output_directory(output,tmp_path,mode="task-path")
    nested = source/output.name
    nested.mkdir()
    registration = {"root":str(tmp_path),"source_run":str(source),"causal_run":str(source),
        "output":str(nested),"parent_pid":os.getppid(),"configuration":config}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (nested/"registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError,match="direct data"):
        runner.run_worker(nested)
    assert list(nested.iterdir()) == [nested/"registration.json"]
    assert not (source.parent/".continuous-training.lock").exists()


def test_polarity_changes_only_five_task_channels_and_preserves_nominal_sum():
    original = original_encoder()
    encoder = TaskPolarityEncoder(original)
    ports = list(encoder.task_ports)
    other = np.setdiff1d(np.arange(3377),ports)
    assert encoder.input_map_id != original.identity["input_map_id"]
    assert encoder.input_indices == original.input_indices and encoder.input_body_ids == original.input_body_ids
    assert encoder.identity["max_rate_hz"] == 150 and encoder.identity["event_voltage_mv"] == 68.75
    for task in ("GREET","FACE_LEFT","FACE_RIGHT","SHOW_FORWARD","END_INTERACTION"):
        before, after = original.encode(task),encoder.encode(task)
        np.testing.assert_array_equal(after[other],before[other])
        np.testing.assert_array_equal(after[ports],(1-before[ports])/4)
        assert before[ports].sum() == after[ports].sum() == 60
        assert np.count_nonzero(after[ports] == .25) == 240 and np.count_nonzero(after[ports] == 0) == 60
    observations = ["GREET","SHOW_FORWARD","GREET"]
    audit, encoded = runner.polarity_input_audit(encoder,observations,np.asarray([original.encode(o) for o in observations]))
    assert audit["between_task_changed_ports"] == 120 and not audit["actual_event_equality_assumed"]
    assert np.array_equal(encoded[0],encoded[2])
    with pytest.raises(ValueError,match="one-hot"):
        encoder.encode("INVALID")


def test_binding_changes_identity_without_graph_parameters_or_step_replacement():
    class Policy:
        def __init__(self):
            self.encoder,self.brain,self.parameters = original_encoder(),object(),object()
            identity = {"schema":"malecns_lab.x2_interaction_policy.v1","input_map":self.encoder.identity}
            self.policy_id = frozen.canonical_hash(identity)
            self._identity = {**identity,"policy_id":self.policy_id}
        @property
        def identity(self):
            return deepcopy(self._identity)
        def _step(self):
            raise AssertionError("No neural step authorized")
    policy = Policy()
    before,brain,parameters,step = policy.identity,policy.brain,policy.parameters,policy._step.__func__
    bind_task_polarity_policy(policy)
    assert policy.identity["schema"] == SCHEMA and policy.policy_id != before["policy_id"]
    assert policy.identity["parent_interaction_policy_id"] == before["policy_id"]
    assert policy.brain is brain and policy.parameters is parameters and policy._step.__func__ is step
    assert "_step" not in policy.__dict__
    assert "x2_task_path_probe" not in json.dumps(policy.identity)
    with pytest.raises(ValueError,match="original interaction"):
        bind_task_polarity_policy(policy)


def test_completed_baseline_binding_reads_exact_assets_without_replaying():
    causal = frozen.ROOT/"data/x2_behavior_short_trial_interaction_causal_20261024_r1"
    baseline = frozen.ROOT/"data/x2_behavior_short_trial_task_path_20261024_r1"
    if not causal.exists() or not baseline.exists():
        pytest.skip("Sealed local scientific assets unavailable")
    _,metadata,comparison,observations,_,archived = runner.source_inputs(causal,frozen.ROOT)
    evidence = runner.path_baseline(frozen.ROOT,causal,metadata,comparison,observations,archived)
    assert evidence["registration_id"] == runner.PATH_BASELINE_ID
    assert evidence["neural_frames_replayed"] == 0
    assert evidence["four_first_50ms_responses_bound_to_causal_r1"]
    bad = dict(archived,features_mv=archived["features_mv"].copy())
    bad["features_mv"][0,0,0] += 1
    with pytest.raises(ValueError,match="features/normalization"):
        runner.path_baseline(frozen.ROOT,causal,metadata,comparison,observations,bad)


def test_eight_new_frames_use_rebound_states_and_measure_events_without_baseline_steps(tmp_path,monkeypatch):
    import torch
    from malecns_lab.learning.x2_behavior_policy import BehaviorOutput
    @dataclass(frozen=True)
    class State:
        brain: object
        event_counts: object
        observation_seq: int
        policy_id: str
    def fixture_identity(state):
        result = {"policy_id":state.policy_id,"neural_step":state.brain.step,"observation_seq":state.observation_seq,
                  "arrays":{"fixture":"unmodified"}}
        return {**result,"state_id":frozen.canonical_hash(result)}
    calls = []
    class Policy:
        policy_id = "new-polarity"
        identity = {"policy_id":"new-polarity"}
        brain = SimpleNamespace(graph=None)
        encoder = SimpleNamespace(task_ports=(0,1),encode=lambda o: np.array([0.,.25],np.float32))
        state_identity = staticmethod(fixture_identity)
        def __call__(self,observation,state,**kwargs):
            calls.append((observation,state.brain.step))
            next_state = replace(state,brain=SimpleNamespace(step=state.brain.step+500),
                observation_seq=observation.observation_seq,event_counts=state.event_counts+torch.tensor([0,2]))
            return BehaviorOutput(torch.zeros(5),torch.zeros(3),torch.zeros(1),torch.ones(1)*.5,
                                  torch.zeros(2628),next_state)
    class Collector:
        def __init__(self,*args): pass
        def arrays(self):
            values = {name:np.zeros((100,2)) for name in
                ("input_events","input_spikes","dn_spikes","post_spikes","post_v","post_g")}
            # Deliberately differ: scientific evidence must not assume events==spikes.
            values["input_events"][[0,50],1] = 2
            values["input_spikes"][[0,50],1] = 1
            return values
    policy = Policy()
    old_states = [State(SimpleNamespace(step=step),torch.zeros(2,dtype=torch.int64),-1,"original") for step in (600,1600)]
    states = [replace(s,policy_id=policy.policy_id) for s in old_states]
    comparison = {"contexts":[{"prestate":fixture_identity(s)} for s in old_states]}
    observations = [SimpleNamespace(task=task,observation_seq=3) for task in path.TASKS]
    archived = {"features_mv":np.ones((2,3,2628))*999,"fixed_train_center":np.zeros(2628),"fixed_train_scale":np.ones(2628)}
    monkeypatch.setattr(path,"selection",lambda *args: {"groups":{},"selected_graph_indices":[]})
    monkeypatch.setattr(path,"PathCollector",Collector)
    monkeypatch.setattr(path,"observed_steps",lambda *args: nullcontext())
    monkeypatch.setattr(path,"continue_observation",lambda p,o,s: p(o,s))
    monkeypatch.setattr(torch.mps,"synchronize",lambda: None)
    completed = []
    result = path.run_probe(policy,states,observations,archived,comparison,tmp_path,time.monotonic()+30,
        lambda: completed.append(True),lambda *a,**k: None,mode="task-polarity",baseline_evidence={"neural_frames_replayed":0})
    assert len(calls) == len(completed) == 8
    assert [step for _,step in calls] == [600,1100,600,1100,1600,2100,1600,2100]
    assert calls[0][0] is calls[1][0] and calls[2][0] is calls[3][0]
    assert result["baseline_replayed"] is False and "baseline_four_frames_exact" not in result
    actual = result["measurements"]["actual_task_input"]
    assert actual["input_events"]["all_100ms"] == [[4,4],[4,4]]
    assert actual["input_spikes"]["all_100ms"] == [[2,2],[2,2]]
    assert actual["input_events"]["first_50ms"] == [[2,2],[2,2]]
    assert result["measurements"]["polarity_screen"]["stop_polarity_branch"] is True
    assert all(torch.count_nonzero(s.event_counts) == 0 for s in old_states)
    with np.load(tmp_path/"responses.npz",allow_pickle=False) as saved:
        assert "encoded_occupancy" in saved and not np.array_equal(saved["input_events"],saved["input_spikes"])
