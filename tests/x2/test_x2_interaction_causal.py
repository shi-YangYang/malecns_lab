"""Small CPU contracts for the eight-frame intervention; no model or VM."""
from copy import deepcopy
from dataclasses import replace
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from malecns_lab.control.x2_behavior import Observation
from malecns_lab.learning.x2_interaction_causal import (
    branch_observations, configuration, delta_measurements, input_audit,
    output_directory, run_worker,
)
from malecns_lab.learning.x2_interaction_data import generate_dataset
from malecns_lab.senses.x2_behavior import CHANNELS, channel_values


def test_preview_has_eight_fixed_frames_no_heavy_imports_or_execution():
    code = ("import sys; from malecns_lab.learning.x2_interaction_causal import main; main([]); "
            "assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    value = json.loads(subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True).stdout)
    assert value["brain_loaded"] is False
    config = value["configuration"]
    assert config["context_replay_frames"]+config["branch_frames"] == config["total_frames"] == 8
    assert config["wall_seconds"] == 180 and config["automatic_deployment"] is False


def test_only_task_changes_and_repetition_is_exact():
    observation = Observation.from_dict(generate_dataset()["splits"]["validation"][0]["frames"][2]["observation"])
    branches = branch_observations(observation)
    assert branches[0] == branches[2] == observation
    class Encoder:
        channel_indices = tuple(c for c in range(57) for _ in range(60 if c < 14 else 59))
        identity = {"channels":list(CHANNELS),"channel_indices":channel_indices}
        def encode(self, value):
            channels = list(channel_values(value).values())
            return np.array([channels[i] for i in self.channel_indices],dtype=np.float32)
    audit, inputs = input_audit(Encoder(),branches)
    assert audit["changed_ports"] == 120 and audit["only_task_channels_changed"]
    assert np.array_equal(inputs[0],inputs[2])
    contaminated = [branches[0], replace(branches[1],feedback=replace(branches[1].feedback,age_s=.23)),branches[2]]
    with pytest.raises(ValueError,match="exactly the two"):
        input_audit(Encoder(),contaminated)


def test_task_and_history_effects_are_separate_not_refitted():
    values = np.empty((2,3,2628))
    values[0] = np.array([-52.,-51.,-52.])[:,None]
    values[1] = np.array([-42.,-40.,-42.])[:,None]
    center, scale = np.full(2628,-50.), np.full(2628,2.)
    original = values.copy()
    result, arrays = delta_measurements(values,center,scale)
    raw, standardized = result["raw_mv"],result["train_standardized"]
    assert raw["task_delta"]["initial"]["rms"] == 1.
    assert raw["task_delta"]["after_two_frames"]["rms"] == 2.
    assert raw["state_delta"]["GREET"]["rms"] == 10.
    assert raw["state_delta"]["SHOW_FORWARD"]["rms"] == 11.
    assert raw["pooled_task_to_state_rms_ratio"] == pytest.approx(np.sqrt(5/221))
    assert raw["task_delta_cosine_between_contexts"] == pytest.approx(1.)
    assert standardized["task_delta"]["initial"]["rms"] == .5
    np.testing.assert_array_equal(values,original)
    assert set(arrays) == {"raw_mv.task_delta","raw_mv.state_delta","train_standardized.task_delta","train_standardized.state_delta"}
    zero = delta_measurements(np.full((2,3,2628),-52.),center,scale)[0]
    assert zero["raw_mv"]["task_delta_cosine_between_contexts"] is None


def test_fresh_output_and_worker_boundary_precede_writes(tmp_path):
    from malecns_lab.learning import x2_inference as frozen
    data = tmp_path/"data"
    archive = data/"old-run"
    archive.mkdir(parents=True)
    output = data/"x2_behavior_short_trial_interaction_causal_new"
    assert output_directory(output,tmp_path,inputs=(archive,)) == output
    with pytest.raises(ValueError,match="dedicated"):
        output_directory(data/"x2_behavior_short_trial_interaction_other",tmp_path)
    nested = archive/output.name
    nested.mkdir()
    registration = {"root":str(tmp_path),"source_run":str(archive),"output":str(nested),
                    "parent_pid":os.getppid(),"configuration":configuration()}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (nested/"registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError,match="direct data"):
        run_worker(nested)
    assert list(nested.iterdir()) == [nested/"registration.json"]
    assert not (data/".continuous-training.lock").exists()
