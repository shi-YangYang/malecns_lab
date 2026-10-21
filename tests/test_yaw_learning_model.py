"""Actual MPS small-array/numerical-fixture evidence, never a full-brain claim."""
import os
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
import torch

from malecns_lab.brain.mps_trainable import MPSLIF
from malecns_lab.learning.train_snn import workspace_lock
from malecns_lab.learning.yaw.optimizer import YawAdam
from malecns_lab.learning.yaw.policy import YawVisualPolicy
from malecns_lab.senses.encoding_v2 import INPUT_GROUPS
from malecns_lab.senses.vision_v2 import VisionV2Encoder
from malecns_lab.senses.yaw_drive import YawDrive
from malecns_lab.senses.yaw_vision import YawVision


@pytest.fixture(scope="module", autouse=True)
def actual_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("Actual MPS is required; no CPU fallback")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    with workspace_lock():
        yield


def frame(shift=0.):
    y, x = np.mgrid[:72, :96].astype(float)
    x -= shift
    gray = 127 + 34 * np.sin(.49*x+.3) + 31 * np.cos(.43*y+.5)
    gray += 25 * np.sin(.21*x+.31*y-.2) + 19 * np.cos(.37*x-.27*y+.7)
    return np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None], 3, axis=2)


@pytest.mark.parametrize("direction", [-1, 1])
def test_mps_visual_features_match_independent_numpy_math(direction):
    gpu, reference = YawVision(), VisionV2Encoder()
    state = gpu.initial_state()
    for index in range(4):
        image = frame(direction * index * .35)
        before = state.gray.clone()
        actual, after = gpu.encode(image, state)
        expected = reference.encode(image, index * .05)
        assert expected.valid
        np.testing.assert_allclose(actual.cpu().numpy(), expected.features, rtol=2e-4, atol=2e-5)
        assert torch.equal(state.gray, before) and after.frame == index + 1
        if index:
            # Image-right flow excites left BTF / right FTB; no body truth provided.
            high, low = ((2, 8), (1, 9)) if direction == 1 else ((1, 9), (2, 8))
            assert all(actual[h] > .2 for h in high)
            assert all(actual[l] < .01 for l in low)
        state = after


def test_mps_invalid_vision_is_transactional():
    vision = YawVision()
    state = vision.initial_state()
    blank = np.zeros((72, 96, 3), np.uint8)
    _, state = vision.encode(blank, state)
    with pytest.raises(ValueError, match="no_valid_flow"):
        vision.encode(blank, state)
    assert state.frame == 1 and torch.count_nonzero(state.gray) == 0
    with pytest.raises(ValueError, match="96x72"):
        vision.encode(blank[:, :-1], state)
    with pytest.raises(ValueError, match="96x72"):
        vision.encode(blank.astype(np.float32), state)


def test_group_drive_event_order_and_noninput_zero():
    drive = YawDrive(16, np.arange(14, dtype=np.int64), np.arange(14, dtype=np.int64))
    state = drive.initial_state()
    rates = drive.rates(torch.linspace(0, 1, 14, device="mps"))
    phase, counts = state.phase.cpu().numpy().copy(), np.zeros(14, np.int64)
    for _ in range(13):
        phase = phase + rates.cpu().numpy() * np.float32(.0001)
        fired = phase >= 1
        phase = np.minimum(phase - fired.astype(np.float32), np.float32(.9999999403953552))
        counts += fired
        increment, state = drive.step(rates, state)
        np.testing.assert_array_equal(increment[:14].cpu().numpy(), fired * np.float32(68.75))
        assert torch.count_nonzero(increment[14:]) == 0
    np.testing.assert_allclose(state.phase.cpu().numpy(), phase, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(state.event_counts.cpu().numpy(), counts)
    drive.validate(state)


def fixture_policy():
    # 14 labelled input cells, one L1 tonic cell and two artificial output cells.
    # This intentionally tiny circuit is only numerical interface evidence.
    matrix = np.zeros((17, 17), np.int64)
    matrix[15, :14:2] = 30
    matrix[16, 1:14:2] = 25
    matrix[15, 14] = 7
    matrix[16, 14] = 5
    types = np.array([f"input{i}" for i in range(14)] + ["L1", "outputL", "outputR"])
    brain = MPSLIF.for_numerical_test(sparse.csc_matrix(matrix), np.arange(14), types=types, tonic_types=("L1",))
    groups = {name: np.array([index], np.int64) for index, name in enumerate(INPUT_GROUPS)}
    groups.update(output_L=np.array([15], np.int64), output_R=np.array([16], np.int64))
    return YawVisualPolicy.for_numerical_test(brain, groups, np.array([15, 16], np.int64))


def test_policy_numerical_checkpoint_gradient_state_restore_and_baseline():
    policy = fixture_policy()
    state = policy.initial_state()
    first = policy.forward_tensor(frame(), state)
    second = policy.forward_tensor(frame(.35), first.state, frame_offset=500)
    assert first.state.brain.step == 500 and second.state.brain.step == 1000
    assert first.action == 0 and second.action == 0
    assert second.features_mv.shape == (4,) and second.network_spikes > 0
    assert second.group_rates_hz.shape == (16,)
    loss = torch.nn.functional.softplus(2 * second.logit) - 1.4 * second.logit
    loss.backward()
    assert policy.brain.theta.grad is not None and torch.isfinite(policy.brain.theta.grad).all()
    assert torch.count_nonzero(policy.brain.theta.grad) > 0
    arrays = policy.state_arrays(second.state.detach())
    restored = policy.restore_state(arrays)
    for name, values in policy.state_arrays(restored).items():
        np.testing.assert_array_equal(values, arrays[name])
    with pytest.raises(ValueError, match="different policy"):
        policy.restore_state({**arrays, "policy_id": np.array("foreign")})
    with pytest.raises(ValueError, match="clocks"):
        policy.restore_state({**arrays, "brain_step": np.array(1001, np.int64)})
    with pytest.raises(ValueError, match="tensor"):
        policy.restore_state({**arrays, "phase": arrays["phase"].astype(np.float64)})
    # Direct fixture replay verifies checkpoint numerical values without any
    # production graph bypass. No additional brain forward is needed for grads.
    with torch.no_grad():
        direct = policy.forward_tensor(frame(), state, direct_numerical_fixture=True)
    torch.testing.assert_close(direct.features_mv, first.features_mv, rtol=0, atol=0)
    torch.testing.assert_close(direct.state.brain.v, first.state.brain.v, rtol=0, atol=0)
    assert state.frame_index == 0 and state.brain.step == 0
    # Explicit state-construction fixture isolates the frame39 -> frozen40 boundary.
    # It is not claimed to be a real 40-frame neural warmup.
    boundary = replace(restored, frame_index=39, baseline_count=19,
                       baseline_sum=torch.full((4,), -52.*19, device="mps"),
                       brain=replace(restored.brain, step=19500),
                       drive=replace(restored.drive, step=19500), vision=replace(restored.vision, frame=39))
    with torch.no_grad():
        frozen = policy.forward_tensor(frame(.35), boundary)
    assert frozen.state.baseline_count == 20 and frozen.state.frame_index == 40 and frozen.action == 0
    torch.testing.assert_close(frozen.state.baseline, frozen.state.baseline_sum / 20, rtol=0, atol=0)
    policy.validate_state(frozen.state)


@pytest.mark.parametrize("mode", ["joint", "frozen"])
def test_yaw_adam_gpu_update_and_exact_restore(mode):
    policy = fixture_policy()
    optimizer = YawAdam(policy, mode=mode)
    before = policy.parameters_numpy()
    # Small manufactured gradients isolate optimizer mathematics from neurons.
    for index, parameter in enumerate(optimizer.parameters.values(), 1):
        parameter.grad = torch.full_like(parameter, .1 * index)
    report = optimizer.step()
    assert report["steps"] == 1 and all(norm <= 1.000001 for norm in report["post_clip_norms"].values())
    assert report["changed"]["readout.weight"] > 0
    assert (report["changed"]["brain.theta"] > 0) == (mode == "joint")
    after = policy.parameters_numpy()
    for name in optimizer.parameters:
        # Adam's first step for these positive gradients is lr, up to epsilon.
        np.testing.assert_allclose(before[name] - after[name], .001, rtol=2e-5, atol=2e-8)
    saved = optimizer.snapshot()
    optimizer = YawAdam(policy, mode=mode, snapshot=saved)
    assert optimizer.steps == 1 and ("brain.theta" in optimizer.parameters) == (mode == "joint")
    for name, value in optimizer.snapshot()["moments"].items():
        np.testing.assert_array_equal(value, saved["moments"][name])
    if mode == "frozen":
        assert not policy.brain.theta.requires_grad and np.count_nonzero(after["brain.theta"]) == 0
    else:
        with pytest.raises(ValueError, match="unmodified"):
            YawAdam(policy, mode="frozen")
    for parameter in optimizer.parameters.values():
        parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(ValueError, match="gradient"):
        optimizer.step()
