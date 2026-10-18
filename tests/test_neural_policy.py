"""Small explicit numerical circuits; these checks are not full-brain learning."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from cnsbridge.brain.metadata import canonical_hash  # noqa: E402
from cnsbridge.brain.policy_groups import numerical_test_policy_map  # noqa: E402
from cnsbridge.brain.tonic import LaminaTonicLIF  # noqa: E402
from cnsbridge.brain.trainable import TrainableLIF, TrainableLIFConfig  # noqa: E402
from cnsbridge.learning.follow_task import FollowPolicyInput  # noqa: E402
from cnsbridge.learning.neural_policy import (  # noqa: E402
    ACTION_SCALES, NeuralController, NeuralPolicy, NeuralPolicyConfig,
)
from cnsbridge.senses.spatial_drive import SpatialDrive, SpatialDriveConfig  # noqa: E402
from cnsbridge.senses.target import TargetCamera, TargetObservation, target_mask  # noqa: E402


def circuit(*, dtype=torch.float64, tonic_mode=None):
    counts = np.zeros((5, 5), dtype=np.int64)
    counts[2, 0], counts[4, 1], counts[3, 2], counts[4, 3] = 20, -10, 15, 10
    types = np.array(["R1-R6", "R1-R6", "DNtest", "L1", ""])
    kwargs = {"types": types, "missing": np.array([False] * 4 + [True]), "dtype": dtype,
              "config": TrainableLIFConfig(dt_ms=1, delay_ms=1, refractory_ms=2)}
    if tonic_mode is None:
        brain = TrainableLIF.for_numerical_test(sparse.csc_matrix(counts), (0, 1), **kwargs)
    else:
        brain = LaminaTonicLIF.for_numerical_test(sparse.csc_matrix(counts), (0, 1), tonic_types=("L1",),
                                               mode=tonic_mode, **kwargs)
    annotations = {"type": types, "superclass": np.array(["optic", "optic", "descending_neuron", "central",
                                                         "descending_neuron"]),
                   "rootSide": np.array(["L", "L", "L", "M", ""]), "somaSide": np.array(["M"] * 5)}
    missing = {name: np.zeros(5, dtype=bool) for name in annotations}
    missing["type"][4] = missing["rootSide"][4] = True
    return brain, numerical_test_policy_map(brain.graph, annotations, missing)


def policy(*, dtype=torch.float64, background="zero_background", config=None, camera=None):
    brain, mapping = circuit(dtype=dtype, tonic_mode="fixed" if background == "fixed_lamina_8mv" else None)
    return NeuralPolicy.for_numerical_test(
        brain, mapping, background=background,
        config=config or NeuralPolicyConfig(frame_ms=8, precondition_ms=2, control_dt_s=0.1, readout_seed=123),
        camera=camera or TargetCamera(), drive_config=SpatialDriveConfig(dt_ms=1, max_rate_hz=1000),
    )


def frame(camera=TargetCamera(), *, red=True):
    value = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    if red:
        value[:, :, 2] = 255
    return value


def assert_same_state(a, b):
    assert a.frame_index == b.frame_index and a.policy_id == b.policy_id
    assert a.brain.step == b.brain.step and a.brain.model_id == b.brain.model_id
    for left, right in zip((a.brain.v, a.brain.g, a.brain.refractory_until, *a.brain.pending),
                           (b.brain.v, b.brain.g, b.brain.refractory_until, *b.brain.pending)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert a.drive.step == b.drive.step and a.drive.drive_id == b.drive.drive_id
    np.testing.assert_array_equal(a.drive.phase, b.drive.phase)
    np.testing.assert_array_equal(a.drive.event_counts, b.drive.event_counts)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_manual_temporal_voltage_means_and_fixed_linear_algebra(dtype):
    model = policy(dtype=dtype)
    initial = model.initial_state()
    with torch.no_grad():
        model.readout.weight.copy_(torch.tensor([[1, 2, 3, 4], [-1, 0, 2, 0], [0, -2, 1, 1]], dtype=dtype) * 0.1)
        model.readout.bias.copy_(torch.tensor([0.02, -0.01, 0.03], dtype=dtype))
    image = frame()
    out = model.forward_tensor(image, initial)
    drive = SpatialDrive(model.spatial_map, model.drive_config)
    drive.restore_state(initial.drive)
    state, values = initial.brain, []
    with torch.no_grad():
        for _ in range(model.frame_steps):
            _, state = model.brain.step(drive.step(target_mask(image)).to_torch(dtype=dtype), state)
            values.append(state.v[torch.tensor([2, 4])].numpy().copy())
    expected = np.concatenate((np.mean(values[:4], axis=0), np.mean(values[4:], axis=0)))
    np.testing.assert_allclose(out.features_mv.detach().numpy(), expected, rtol=0, atol=1e-5 if dtype == torch.float32 else 1e-13)
    normalized = (expected + 52) / 7
    expected_action = np.tanh(model.readout.weight.detach().numpy() @ normalized + model.readout.bias.detach().numpy())
    expected_action *= np.asarray(ACTION_SCALES)
    np.testing.assert_allclose(out.action.detach().numpy(), expected_action, rtol=1e-5, atol=1e-7)
    assert out.features_mv.grad_fn is not None and out.action.grad_fn is not None
    assert out.features_mv.shape == (4,) and model.mapping.outputs.body_ids.tolist() == [2, 4]


def test_split_frames_preserve_full_state_and_drive_like_contiguous_core_steps():
    model = policy()
    initial = model.initial_state()
    images = (frame(), frame(red=False))
    a = model.forward_tensor(images[0], initial)
    b = model.forward_tensor(images[1], a.state)
    drive = SpatialDrive(model.spatial_map, model.drive_config)
    drive.restore_state(initial.drive)
    state = initial.brain
    for image in images:
        for _ in range(model.frame_steps):
            events = drive.step(target_mask(image)).to_torch(dtype=torch.float64)
            assert torch.count_nonzero(events[2:]) == 0
            _, state = model.brain.step(events, state)
    expected = replace(b.state, brain=state, drive=drive.export_state())
    assert_same_state(b.state, expected)
    assert b.state.brain.step == b.state.drive.step == 18 and b.state.frame_index == 2
    assert b.state.drive.event_counts.tolist() == [8, 8]
    # A branch from the same immutable input gives exactly the same result.
    repeated = model.forward_tensor(images[0], initial)
    assert_same_state(a.state, repeated.state)


def test_detach_retains_every_state_component_and_next_frame_values():
    model = policy()
    first = model.forward_tensor(frame(), model.initial_state())
    detached = first.state.detach()
    assert_same_state(first.state, detached)
    assert first.state.brain.v.grad_fn is not None
    assert all(value.grad_fn is None for value in (detached.brain.v, detached.brain.g, *detached.brain.pending))
    assert detached.drive is first.state.drive
    with pytest.raises(ValueError):
        detached.drive.phase.setflags(write=True)
    with torch.no_grad():
        continued = model.forward_tensor(frame(red=False), first.state)
        truncated = model.forward_tensor(frame(red=False), detached)
    assert_same_state(continued.state, truncated.state)


def test_action_loss_updates_internal_gain_and_readout_and_supports_tbptt():
    model = policy()
    original_id = model.policy_id
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    first = model.forward_tensor(frame(), model.initial_state())
    target = first.action.new_tensor([0.1, -0.1, 0.2])
    loss = ((first.action - target) / first.action.new_tensor(ACTION_SCALES)).square().mean()
    loss.backward()
    gradient = model.brain.theta.grad
    assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    assert abs(gradient[model.brain.graph.group_names.index("R1-R6")]) > 0
    optimizer.step()
    assert set(before) == {"brain.theta", "readout.weight", "readout.bias"}
    assert all(not torch.equal(before[name], value) for name, value in model.named_parameters())
    assert model.policy_id == original_id  # Learned values are not structural identity.
    optimizer.zero_grad()
    second = model.forward_tensor(frame(), first.state.detach())
    second.action.square().sum().backward()
    assert torch.isfinite(model.brain.theta.grad).all() and torch.count_nonzero(model.brain.theta.grad)


def test_constructor_does_not_precondition_or_consume_global_random_state(monkeypatch):
    brain, mapping = circuit()
    def forbidden_step(*_args, **_kwargs):
        raise AssertionError("construction ran neural dynamics")
    monkeypatch.setattr(brain, "step", forbidden_step)
    before = torch.random.get_rng_state().clone()
    model = NeuralPolicy.for_numerical_test(brain, mapping, background="zero_background",
                                           config=NeuralPolicyConfig(8, 2, 0.1, 123), camera=TargetCamera(),
                                           drive_config=SpatialDriveConfig(dt_ms=1))
    assert torch.equal(before, torch.random.get_rng_state())
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected = torch.empty((3, 4), dtype=torch.float64).normal_(std=0.01 / math.sqrt(4), generator=generator)
    assert torch.equal(model.readout.weight, expected) and not model.readout.bias.any()
    assert model.readout.weight.count_nonzero() == 12


def test_blank_precondition_is_explicit_no_grad_and_fixed_background_stays_fixed():
    model = policy(background="fixed_lamina_8mv")
    state = model.initial_state()
    assert state.brain.step == state.drive.step == 2 and state.frame_index == 0
    assert all(value.grad_fn is None for value in (state.brain.v, state.brain.g, *state.brain.pending))
    assert state.brain.v[3] > -52 and not state.drive.event_counts.any()
    assert set(dict(model.named_parameters())) == {"brain.theta", "readout.weight", "readout.bias"}
    assert model.brain.rho.tolist() == [0] and not model.brain.rho.requires_grad
    assert torch.equal(model.brain.tonic_offsets_mv, torch.tensor([8.0], dtype=torch.float64))
    with torch.no_grad():
        model.brain.rho[0] = 0.01
    with pytest.raises(ValueError, match="rho=0"):
        model.forward_tensor(frame(), state)


@pytest.mark.parametrize("config", [
    {"frame_ms": 0}, {"frame_ms": float("nan")}, {"precondition_ms": -1}, {"control_dt_s": 0},
    {"readout_seed": True}, {"readout_seed": -1}, {"frame_ms": 3}, {"precondition_ms": 0.5},
])
def test_invalid_explicit_times_and_seed_rejected(config):
    values = {"frame_ms": 8, "precondition_ms": 2, "control_dt_s": 0.1, "readout_seed": 123}
    with pytest.raises(ValueError):
        policy(config=NeuralPolicyConfig(**(values | config)))


@pytest.mark.parametrize("mutation", ["policy", "frame_clock", "brain_clock", "drive_clock", "dtype", "drive_phase"])
def test_foreign_or_inconsistent_state_rejected_without_modifying_input(mutation):
    model = policy()
    good = model.forward_tensor(frame(), model.initial_state()).state
    if mutation == "policy":
        bad = replace(good, policy_id="different")
    elif mutation == "frame_clock":
        bad = replace(good, frame_index=0)
    elif mutation == "brain_clock":
        bad = replace(good, brain=replace(good.brain, step=9))
    elif mutation == "drive_clock":
        bad = replace(good, drive=replace(good.drive, step=9))
    elif mutation == "dtype":
        bad = replace(good, brain=replace(good.brain, v=good.brain.v.float()))
    else:
        bad = replace(good, drive=replace(good.drive, phase=np.full(2, np.nan)))
    saved = good.detach()
    with pytest.raises(ValueError):
        model.forward_tensor(frame(), bad)
    assert_same_state(good, saved)


@pytest.mark.parametrize("invalid", [np.zeros((120, 160, 3)), np.zeros((160, 120, 3), dtype=np.uint8), None])
def test_invalid_frame_rejected_without_advancing_state(invalid):
    model = policy()
    state = model.initial_state()
    with pytest.raises(ValueError):
        model.forward_tensor(invalid, state)
    assert_same_state(state, model.initial_state())


def test_role_mapping_background_and_dtype_boundaries_are_strict(monkeypatch, tmp_path):
    import cnsbridge.learning.neural_policy as module

    brain, mapping = circuit()
    kwargs = dict(background="zero_background", config=NeuralPolicyConfig(8, 2, 0.1, 123),
                  camera=TargetCamera(), drive_config=SpatialDriveConfig(dt_ms=1))
    with pytest.raises(TypeError, match="from_reference"):
        NeuralPolicy()
    # The production entry cannot accept the numerical role even if its graph
    # loader were replaced by a fixture; metadata binding still rejects it.
    monkeypatch.setattr(module, "load_training_graph", lambda _path: brain.graph)
    with pytest.raises(ValueError, match="full_reference"):
        NeuralPolicy.from_reference(tmp_path / "not-a-reference", **kwargs)
    with pytest.raises(ValueError, match="dt"):
        NeuralPolicy.for_numerical_test(brain, mapping, **(kwargs | {"drive_config": SpatialDriveConfig(dt_ms=0.1)}))
    forged = mapping.identity
    forged["outputs"]["body_ids"] = [4, 2]
    with pytest.raises(ValueError, match="identity payload"):
        NeuralPolicy.for_numerical_test(brain, replace(mapping, _identity_json=json.dumps(forged)), **kwargs)
    learned, learned_map = circuit(tonic_mode="learnable")
    with pytest.raises(ValueError, match="fixed tonic"):
        NeuralPolicy.for_numerical_test(learned, learned_map, **(kwargs | {"background": "fixed_lamina_8mv"}))
    model = policy()
    state = model.initial_state()
    model.readout.float()
    with pytest.raises(ValueError, match="CPU dtype"):
        model.forward_tensor(frame(), state)


def test_identity_binds_all_dn_camera_timing_mapping_dtype_and_source():
    model = policy()
    identity = model.identity
    identity_id = identity.pop("policy_id")
    assert canonical_hash(identity) == identity_id
    assert identity["role"] == "numerical_test_only" and identity["features"]["body_ids"] == [2, 4]
    assert identity["features"]["width"] == 4 and identity["features"]["sample"] == "post_reset_v_each_neural_step"
    assert "learning/neural_policy.py" in identity["implementation_sha256"]
    assert identity["spatial"]["policy_map_id"] == identity["neurons"]["policy_map_id"]
    variants = [policy(dtype=torch.float32), policy(camera=TargetCamera(horizontal_fov_deg=80)),
                policy(config=NeuralPolicyConfig(10, 2, 0.1, 123)), policy(background="fixed_lamina_8mv")]
    for other in variants:
        assert other.policy_id != model.policy_id
        with pytest.raises(ValueError, match="identity"):
            other.forward_tensor(frame(other.camera), model.initial_state())
    identity["features"]["body_ids"].clear()
    assert model.identity["features"]["body_ids"] == [2, 4]


def test_controller_uses_pixels_only_requires_reset_and_exposes_separate_clocks():
    model = policy()
    controller = NeuralController(model)
    a = FollowPolicyInput(frame(), TargetObservation(True, True, "test", 0.9, 0.8, 0.1, 9.0), model.camera, 0.0)
    b = replace(a, observation=TargetObservation(False, False, "irrelevant", -0.9, -0.8, 0.9, 0.01))
    with pytest.raises(ValueError, match="Reset"):
        controller(a)
    assert controller.neural_time_ms is None and controller.control_time_s is None
    controller.reset()
    first = controller(a)
    assert first.vy == first.wx == first.wy == 0
    assert controller.neural_time_ms == 10 and controller.control_time_s == 0.1
    assert controller.state.brain.v.grad_fn is None
    controller.reset()
    assert controller(b) == first  # Even invalid observations cannot inject a hidden teacher/zero rule.
    retained = controller.state
    for bad in (replace(a, time_s=0), replace(a, time_s=float("nan")),
                replace(a, time_s=0.1, camera=TargetCamera(horizontal_fov_deg=80))):
        with pytest.raises(ValueError):
            controller(bad)
        assert controller.state is retained
    controller(replace(a, time_s=0.1))
    assert controller.neural_time_ms == 18 and controller.control_time_s == 0.2
    assert abs(first.vx) < 0.3 and abs(first.vz) < 0.2 and abs(first.wz) < 0.5


def test_no_visual_bypass_when_neural_readout_is_constant():
    model = policy()
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    state = model.initial_state()
    visible = model.forward_tensor(frame(), state)
    blank = model.forward_tensor(frame(red=False), state)
    assert not torch.equal(visible.features_mv, blank.features_mv)
    assert torch.equal(visible.action, blank.action)
