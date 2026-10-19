"""Small explicit numerical fixtures, never complete-connectome evidence."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from cnsbridge.learning.gradient_probe_math import compute_branch, stable_metrics, logit_loss_check
from cnsbridge.learning.gradient_probe_evidence import compare_arrays, PROTOCOL, source_hashes
from cnsbridge.learning.mps_state import restore, state_digest
from cnsbridge.learning.mps_window import compute_window
from cnsbridge.learning.neural_policy import NeuralPolicy
from cnsbridge.learning.train_snn import workspace_lock
from cnsbridge.learning.training_contract import PARAMETERS
from test_training_checkpoint import policy
from test_mps_continuation import _migrated, selection


@pytest.fixture(scope="module", autouse=True)
def numerical_slot():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; GPU diagnostic fixture not executed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with workspace_lock():
        yield


def test_parent_imports_lightweight_and_direct_worker_rejected(tmp_path):
    subprocess.run([sys.executable, "-c", "import sys; import cnsbridge.learning.gradient_probe; "
                    "import cnsbridge.learning.gradient_probe_worker; assert 'torch' not in sys.modules"], check=True)
    (tmp_path / "registration.json").write_text(json.dumps({"parent_pid": -1}))
    script = ("import sys; from pathlib import Path; from cnsbridge.learning.gradient_probe_worker import claim_parent; "
              "\ntry: claim_parent(Path(sys.argv[1]))\nexcept ValueError as e: "
              "assert 'supervising parent' in str(e); assert 'torch' not in sys.modules\nelse: raise AssertionError('accepted')")
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    assert not (tmp_path / "worker.claim").exists()
    assert set(source_hashes()) >= {"learning/gradient_probe_math.py", "learning/mps_recompute.py"}


def test_stable_mps_norm_does_not_square_huge_raw_values():
    value = torch.tensor([3e30, -4e30, 0], device="mps")
    measured = stable_metrics(value)
    assert measured["finite"] and measured["nonzero"] == 2
    assert measured["l2"] == pytest.approx(5e30, rel=1e-6)
    assert stable_metrics(torch.tensor([float("inf")], device="mps"))["finite"] is False
    assert stable_metrics(torch.zeros(3, device="mps"))["l2"] == 0


def test_full_and_short_credit_keep_forward_exact_recompute_state_and_change_gradient():
    source = policy()
    cpu = NeuralPolicy.for_numerical_test(source.brain, source.mapping, background=source.background,
                camera=source.camera, config=replace(source.config, frame_ms=50), drive_config=source.drive_config)
    _, gpu, bridge, initial, previous = _migrated(cpu)
    chosen = selection(2)
    before = {key: value.copy() for key, value in previous["arrays"].items()}
    _, original, original_record = compute_window(gpu, bridge, initial, chosen)
    results = []
    for horizon in PROTOCOL["horizons"]:
        _, state, restored = restore(gpu, previous)
        assert all(np.array_equal(restored[k], value) for k, value in before.items())
        initial_digest = state_digest(state)
        values, record = compute_branch(gpu, state, chosen, horizon)
        assert state_digest(state) == initial_digest
        assert record["forward_steps"] == 1000
        assert [b["window_step"] for b in record["boundaries"]] == list(range(50, 1001, 50))
        assert all(b["cut_after_boundary"] == (b["window_step"] % horizon == 0 and b["window_step"] < 1000)
                   for b in record["boundaries"])
        assert any(item["observed"] for b in record["boundaries"] for item in b["adjoints"].values())
        assert all(item.get("finite", True) for b in record["boundaries"] for item in b["adjoints"].values())
        for key in ("actions", "features_mv", "pre_clip_readout_weight", "pre_clip_readout_bias"):
            np.testing.assert_array_equal(values[key], original[key])
        assert record["end_state"] == original_record["end_state"]
        assert all(np.array_equal(dict(gpu.named_parameters())[name].detach().cpu().numpy(),
                                  before[f"parameter.{key}"]) for key, name in PARAMETERS.items())
        results.append(values)
    np.testing.assert_array_equal(results[0]["pre_clip_theta"], original["pre_clip_theta"])
    assert any(not np.array_equal(value["pre_clip_theta"], results[0]["pre_clip_theta"]) for value in results[1:])
    for values in results[1:]:
        for key in results[0]:
            if key != "pre_clip_theta":
                np.testing.assert_array_equal(values[key], results[0][key])
    print({"horizons": PROTOCOL["horizons"], "all_forward_states_and_readout_gradients_exact": True,
           "theta_gradients": [value["pre_clip_theta"].tolist() for value in results]})


def test_logit_loss_saturated_both_directions_and_no_model_update():
    _, gpu, _, state, _ = _migrated(policy())
    features = np.full((2, 2 * len(gpu.mapping.outputs.indices)), -52, np.float32)
    with torch.no_grad():
        gpu.readout.weight.zero_()
        gpu.readout.bias.copy_(torch.tensor([10., -10., 10.], device="mps"))
    selected = selection(2)
    selected["teacher_target"][:] = [-.3, .2, -.5]
    before = {name: p.detach().cpu().numpy().copy() for name, p in gpu.named_parameters()}
    arrays, report = logit_loss_check(gpu, features, selected)
    assert report["device"].startswith("mps") and not report["model_updated"]
    assert report["trial_logit_loss"] < report["loss"]
    assert np.all(arrays["gradient"][:, [0, 2]] > 0) and np.all(arrays["gradient"][:, 1] < 0)
    np.testing.assert_allclose(arrays["gradient"], arrays["expected_gradient"], atol=1e-7, rtol=1e-5)
    assert all(np.array_equal(p.detach().cpu().numpy(), before[name]) for name, p in gpu.named_parameters())
    assert all(p.grad is None for p in gpu.parameters())


def test_bitwise_comparison_does_not_relax_a_small_mismatch():
    a = np.array([1.], np.float32)
    b = np.nextafter(a, np.float32(2))
    check = compare_arrays({"x": a}, {"x": b}, ["x"])
    assert not check["x"]["exact"] and check["x"]["max_abs_difference"] > 0
