"""R2 replay contract and production-path checks; no full-graph experiment."""
from copy import deepcopy
import os

import numpy as np
import pytest

from cnsbridge.learning.gradient_probe_evidence import (
    BRANCHES, THETA_TOLERANCE, theta_comparison, replay_comparison, classify_repeatability,
)


def comparison_fixture():
    values = {"actions": np.array([[.1, .2, .3]], np.float32),
              "features_mv": np.array([[-52., -51.]], np.float32),
              "pre_clip_readout_weight": np.array([[1., 2.]], np.float32),
              "pre_clip_readout_bias": np.array([1.], np.float32),
              "pre_clip_theta": np.array([0., 1e30, -2e30], np.float32),
              "end.brain.v": np.array([-52.], np.float32)}
    return values, {"start_state": "unchanged_start", "end_state": "unchanged_end"}


def test_r2_theta_tolerance_support_sign_and_exactness_are_separate():
    reference = np.array([0., 1e30, -2e30], np.float32)
    actual = reference.copy()
    actual[1] = np.nextafter(actual[1], np.float32(np.inf))
    check = theta_comparison(actual, reference)
    assert check["passed"] and not check["exact"] and check["difference_count"] == 1
    assert check["max_ulp_difference"] == 1 and check["relative_l2"] > 0
    assert check["max_abs_difference"] > 0 and check["max_relative_difference_on_nonzero_reference"] > 0
    assert check["tolerance"] == THETA_TOLERANCE
    actual[1] = reference[1] * np.float32(1.001)
    assert not theta_comparison(actual, reference)["passed"]
    assert not theta_comparison(np.array([1e-9], np.float32), np.array([0.], np.float32))["passed"]
    sign = theta_comparison(np.array([-1e-9], np.float32), np.array([1e-9], np.float32))
    assert sign["within_tolerance"] and sign["nonzero_sign_difference_count"] == 1 and not sign["passed"]
    assert not theta_comparison(np.array([np.inf], np.float32), np.array([1.], np.float32))["passed"]
    assert not theta_comparison(reference.astype(np.float64), reference)["passed"]


def test_r2_forward_readout_exactness_and_repeatability_classification():
    assert BRANCHES == (("reference-1", 1000, False), ("reference-2", 1000, False),
                        ("horizon-500", 500, True), ("horizon-250", 250, True), ("horizon-100", 100, True))
    values, record = comparison_fixture()
    changed = deepcopy(values)
    changed["pre_clip_theta"][1] = np.nextafter(changed["pre_clip_theta"][1], np.float32(np.inf))
    passed = replay_comparison(changed, record, values, record, "reference", full_theta=True)
    assert passed["passed"] and not passed["theta"]["exact"]
    for key in ("actions", "features_mv", "pre_clip_readout_weight", "pre_clip_readout_bias", "end.brain.v"):
        bad = deepcopy(changed)
        bad[key].flat[0] = np.nextafter(bad[key].flat[0], np.float32(np.inf))
        assert not replay_comparison(bad, record, values, record, "reference", full_theta=True)["passed"]
    bad = deepcopy(values)
    bad["pre_clip_theta"] *= -100
    assert replay_comparison(bad, record, values, record, "reference", full_theta=False)["passed"]
    assert not replay_comparison(bad, record, values, record, "reference", full_theta=True)["passed"]
    exact = replay_comparison(values, record, values, record, "reference", full_theta=True)
    first = {"comparisons": {"saved160": exact, "observed_r1": exact}}
    second = {"comparisons": {"saved160": exact, "observed_r1": exact, "repeat": passed}}
    assert classify_repeatability(first, second)["classification"] == "ordinary_replay_last_bit_variation_observed"
    second["comparisons"] = {"saved160": exact, "observed_r1": passed, "repeat": exact}
    assert classify_repeatability(first, second)["classification"] == "observer_version_or_environment_related_difference"
    second["comparisons"] = {"saved160": passed, "observed_r1": exact, "repeat": exact}
    assert classify_repeatability(first, second)["classification"] == "saved_window_difference_unresolved"
    second["comparisons"]["repeat"] = {"passed": False}
    with pytest.raises(ValueError, match="Cannot classify"):
        classify_repeatability(first, second)


def test_r2_unobserved_replay_calls_real_policy_forward_without_observer(monkeypatch):
    torch = pytest.importorskip("torch")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; original-path fixture not executed")
    from cnsbridge.learning import gradient_probe_math as probe
    from cnsbridge.learning.mps_state import boundary_arrays
    from cnsbridge.learning.train_snn import workspace_lock
    from test_mps_continuation import _migrated, selection
    from test_training_checkpoint import policy
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with workspace_lock():
        _, gpu, bridge, state, previous = _migrated(policy())
        original = gpu.forward_tensor
        calls = []
        def production(frame, current):
            calls.append(current.frame_index)
            return original(frame, current)
        def forbidden(*args, **kwargs):
            raise AssertionError("Unobserved reference took the observer implementation")
        monkeypatch.setattr(gpu, "forward_tensor", production)
        monkeypatch.setattr(probe, "observed_frame", forbidden)
        monkeypatch.setattr(probe, "BoundaryObserver", forbidden)
        values, report = probe.compute_branch(gpu, state, selection(2), 2 * gpu.frame_steps, observed=False)
        assert calls == [state.frame_index, state.frame_index + 1]
        assert report["forward_path"] == "policy.forward_tensor" and report["observed"] is False
        assert report["boundaries"] == [] and report["forward_steps"] == 2 * gpu.frame_steps
        assert all(np.isfinite(array).all() for array in values.values())
        unchanged = boundary_arrays(gpu, bridge, state)
        assert all(np.array_equal(unchanged[key], value) for key, value in previous["arrays"].items())
