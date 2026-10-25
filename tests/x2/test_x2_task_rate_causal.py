"""CPU-only contracts for fixed-state input-rate intervention; no full brain/VM."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning import x2_inference as frozen
from malecns_lab.learning.x2_task_rate_causal import (
    DNSpikeCollector, compare_rates, configuration, output_directory, run_worker,
    source_inputs, validate_state_arrays,
)
from malecns_lab.learning.x2_task_rate_policy import (
    SCHEMA, TASK_CHANNELS, TaskRateEncoder, bind_task_rate_policy,
)
from malecns_lab.senses.x2_behavior import CHANNELS


def encoder_fixture():
    class Encoder:
        channel_indices = tuple(c for c in range(57) for _ in range(60 if c < 14 else 59))
        input_indices = tuple(range(3377))
        input_body_ids = tuple(range(3377))
        @property
        def identity(self):
            payload = {"schema": "malecns_lab.x2_behavior_input.no_absolute_yaw.v1",
                "channels": list(CHANNELS), "channel_indices": list(self.channel_indices),
                "max_rate_hz": 150., "event_voltage_mv": 68.75}
            return {**payload, "input_map_id": frozen.canonical_hash(payload)}
        def encode(self, observation):
            return observation.copy()
    return Encoder()


def test_preview_has_only_eight_frames_and_preregistered_screen_without_heavy_imports():
    code = ("import sys; from malecns_lab.learning.x2_task_rate_causal import main; main([]); "
            "assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    value = json.loads(subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout)
    config = value["configuration"]
    assert value["brain_loaded"] is False
    assert config["total_frames"] == 8 and config["context_replay_frames"] == 0 and config["wall_seconds"] == 180
    assert config["conditions_hz"] == [150,300]
    assert config["continuation_screen"]["standardized_task_cosine_min"] == .2
    assert config["automatic_deployment"] is False


def test_rate_identity_changes_only_existing_task_rates_and_cpu_event_scheduler():
    import torch
    original = encoder_fixture()
    encoder = TaskRateEncoder(original)
    assert encoder.input_map_id != original.identity["input_map_id"]
    assert encoder.input_indices == original.input_indices and encoder.input_body_ids == original.input_body_ids
    assert len(encoder.task_ports) == 300
    task_mask = np.isin(original.channel_indices, [CHANNELS.index(name) for name in TASK_CHANNELS])
    np.testing.assert_array_equal(np.array(encoder.rates_hz), np.where(task_mask,300.,150.))
    occupancy = np.linspace(0,1,3377,dtype=np.float32)
    np.testing.assert_array_equal(encoder.encode(occupancy),occupancy)

    class Recorder:
        graph = SimpleNamespace(n=3380)
        def step(self, increment, state, *, spike_backward_gain):
            assert not torch.count_nonzero(increment[3377:])
            assert torch.all((increment == 0) | (increment == 68.75))
            return increment != 0, state
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.brain, self.encoder = Recorder(), original
            self.register_buffer("input_indices",torch.arange(3377))
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.spike_backward_gain = .1
            self._identity = {"schema":"malecns_lab.x2_interaction_policy.v1", "input_map":original.identity}
            self.policy_id = frozen.canonical_hash(self._identity)
            self._identity["policy_id"] = self.policy_id
        @property
        def identity(self):
            return deepcopy(self._identity)
    fixture = Fixture()
    old_identity = fixture.identity
    old_parameters = tuple(fixture.named_parameters())
    bind_task_rate_policy(fixture)
    assert fixture.identity["schema"] == SCHEMA and fixture.policy_id != old_identity["policy_id"]
    assert fixture.identity["parent_interaction_policy_id"] == old_identity["policy_id"]
    assert tuple(fixture.named_parameters()) == old_parameters
    phase = torch.linspace(0,.9,3377)
    counts = torch.zeros(3377,dtype=torch.int64)
    occupancy = torch.ones(3377)
    a_phase, b_phase, a_count, b_count = phase.clone(),phase.clone(),counts.clone(),counts.clone()
    for _ in range(500):
        # Literal existing scalar scheduler, kept on CPU (the production
        # original _step allocates its full increment on MPS).
        advanced = a_phase+(occupancy*150.)*(.1/1000)
        fired = advanced >= 1
        a_phase = (advanced-fired.to(torch.float32)).clamp_max(.9999999403953552)
        a_count = a_count+fired.to(torch.int64)
        _, b_phase, b_count, _ = fixture._step(occupancy,None,b_phase,b_count)
    mask = torch.from_numpy(task_mask)
    assert torch.equal(a_phase[~mask],b_phase[~mask]) and torch.equal(a_count[~mask],b_count[~mask])
    assert torch.all((b_count[mask]-a_count[mask] >= 7) & (b_count[mask]-a_count[mask] <= 8))
    assert torch.all(occupancy == 1)
    with pytest.raises(ValueError,match="unmodified"):
        bind_task_rate_policy(fixture)


def test_screen_rejects_larger_but_orthogonal_task_effect_and_spikes_are_observational():
    import torch
    features = np.zeros((2,2,2,2628))
    features[:,1] = 10.
    features[0,:,1,:] += .1
    features[1,0,1,:1314] += 1.
    features[1,1,1,1314:] += 1.
    _, _, screen = compare_rates(features,np.zeros(2628),np.ones(2628))
    assert all(all(item.values()) for item in screen["task_rms_increased"].values())
    assert screen["standardized_cosine"] == 0.
    assert not screen["supports_considering_54_frame_trial"] and not screen["deployment_qualified"]
    features[1,1,1] = features[1,0,1]+10.
    assert compare_rates(features,np.zeros(2628),np.ones(2628))[2]["supports_considering_54_frame_trial"]
    capture = DNSpikeCollector(torch.tensor([1,3]),600)
    spikes = torch.tensor([1.,1.,0.,0.])
    for i in range(500):
        capture.capture(spikes,600+i)
    np.testing.assert_array_equal(capture.counts.numpy(), [[250,0],[250,0]])
    assert capture.steps == 500 and torch.equal(spikes,torch.tensor([1.,1.,0.,0.]))
    with pytest.raises(ValueError,match="clock"):
        capture.capture(spikes,1100)


def test_complete_archived_arrays_and_source_authentication_without_loading_brain():
    causal = frozen.ROOT/"data/x2_behavior_short_trial_interaction_causal_20261024_r1"
    if not causal.exists():
        pytest.skip("Local archived scientific asset unavailable")
    source, metadata, comparison, observations, states, responses = source_inputs(causal,frozen.ROOT)
    assert source.name == "x2_behavior_short_trial_interaction_20261023_r1"
    assert len(states) == 2 and len(observations) == 3
    assert responses["features_mv"].shape == (2,3,2628)
    assert metadata["policy"]["policy_id"] == states[0][0]["policy_id"]
    identity, arrays = states[1]
    incomplete = dict(arrays)
    incomplete.pop("pending.17")
    with pytest.raises(ValueError,match="Missing"):
        validate_state_arrays(incomplete,identity)
    altered = dict(arrays)
    altered["phase"] = arrays["phase"].copy()
    altered["phase"][0] += np.float32(.001)
    with pytest.raises(ValueError,match="phase"):
        validate_state_arrays(altered,identity)


def test_output_guard_and_worker_boundary_precede_writes(tmp_path):
    data = tmp_path/"data"
    source = data/"archive"
    source.mkdir(parents=True)
    output = data/"x2_behavior_short_trial_task_rate_test"
    assert output_directory(output,tmp_path,inputs=(source,)) == output
    with pytest.raises(ValueError,match="dedicated"):
        output_directory(data/"x2_behavior_short_trial_interaction_other",tmp_path)
    nested = source/output.name
    nested.mkdir()
    registration = {"root":str(tmp_path),"source_run":str(source),"causal_run":str(source),"output":str(nested),
                    "parent_pid":os.getppid(),"configuration":configuration()}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (nested/"registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError,match="direct data"):
        run_worker(nested)
    assert list(nested.iterdir()) == [nested/"registration.json"]
    assert not (data/".continuous-training.lock").exists()
