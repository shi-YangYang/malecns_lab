"""Local derivative contracts on CPU; no graph, MPS, fitting or simulator."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from malecns_lab.learning.x2_theta_local_derivative import (
    EPSILON, block_spike_gradient, configuration, output_directory, source_inputs, spike_difference,
)


def test_lazy_fixed_six_frame_protocol_and_new_output_guard(tmp_path):
    code = "import sys;from malecns_lab.learning.x2_theta_local_derivative import main;main([]);assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    run = subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)
    config = json.loads(run.stdout)["configuration"]
    assert config == configuration() and config["total_frames"] == 6 and config["wall_seconds"] == 180
    assert config["conditions"] == [0.,1/16,-1/16] and config["optimizer_steps"] == 0
    assert config["new_preconditioning_steps"] == 0 and not config["validation_opened"]
    (tmp_path/"data").mkdir()
    with pytest.raises(ValueError):
        output_directory(tmp_path/"data/old/x2_behavior_short_trial_theta_derivative_bad",tmp_path)
    assert not (tmp_path/"data/old").exists()


def test_hook_preserves_binary_forward_and_direct_gain_derivative_with_checkpoint():
    import torch
    from torch.utils.checkpoint import checkpoint
    from malecns_lab.brain.trainable import surrogate_spike
    from malecns_lab.brain.mps_trainable import scale_spike_backward
    def step(theta, previous, block):
        spikes = scale_spike_backward(surrogate_spike(theta+previous),.01)
        transmitted = theta*spikes
        if block:
            block_spike_gradient(spikes)  # Same timing: after the matvec graph is built.
        return transmitted, spikes
    def run(theta, block):
        first,_ = checkpoint(lambda t:step(t,t*0,block),theta,use_reentrant=False)
        second,_ = checkpoint(lambda t,v:step(t,v,block),theta,first,use_reentrant=False)
        return first+second
    theta = torch.tensor(.3,requires_grad=True)
    original = run(theta,False)
    original.backward()
    sg = theta.grad.item()
    theta.grad = None
    local = run(theta,True)
    assert torch.equal(original,local)
    local.backward()
    assert theta.grad.item() == 2. and sg > 2.
    with torch.no_grad():
        finite = (run(theta+EPSILON*.001,False)-run(theta-EPSILON*.001,False))/(2*EPSILON*.001)
    assert finite.item() == pytest.approx(2.,rel=2e-3)


def test_packed_trace_reports_first_event_difference_without_dense_expansion():
    baseline = np.zeros((1000,2),np.uint8)
    changed = baseline.copy()
    changed[19,0] = 0b00000101
    changed[23,1] = 1
    result = spike_difference(baseline,changed,start_step=2600,nodes=9)
    assert result["first_different_neural_step"] == 2619
    assert result["changed_spike_bits"] == 3 and result["steps_with_differences"] == 2
    assert result["different_bits_per_step"][19] == 2
    assert spike_difference(baseline,baseline,start_step=2600,nodes=9)["spike_bits_equal"]


def test_sealed_train_prestate_and_direction_are_sufficient_without_validation(monkeypatch):
    source = Path(__file__).resolve().parents[2]/"data/x2_behavior_short_trial_theta_20261024_r1"
    if not source.is_dir():
        pytest.skip("Sealed local theta scientific asset unavailable")
    from malecns_lab.learning import x2_inference as frozen
    original_read, original_load = frozen.read_json,np.load
    paths = []
    def read(path):
        paths.append(str(path))
        assert "validation" not in str(path) and Path(path).name != "dataset.json"
        return original_read(path)
    def load(path,*args,**kwargs):
        paths.append(str(path))
        assert "validation" not in str(path)
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(frozen,"read_json",read)
    monkeypatch.setattr(np,"load",load)
    assets = source_inputs(source)
    assert assets["identity"]["neural_step"] == 2600 and len(assets["arrays"]) == 23
    assert len(assets["branch"]["frames"]) == 2 and assets["cached"]["features_mv"].shape == (2,2628)
    assert assets["sg"].shape == assets["direction"].shape == (11752,)
    assert np.any(assets["direction"] != 0)
    with original_load(source/"theta-raw-gradient.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(assets["sg"],z["branches"][0].astype(np.float64)*6)
