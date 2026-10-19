"""Actual-MPS foundation checks; tiny fixtures never stand in for the full graph."""
import json
import os

import numpy as np
import pytest
import torch

from cnsbridge.brain.metadata import canonical_hash
from cnsbridge.learning.gpu_optimizer import GPUAdam, NAMES
from cnsbridge.learning.train_snn import workspace_lock
from cnsbridge.senses.gpu_drive import GPUDrive
from cnsbridge.senses.spatial_drive import MASK_SHAPE, SpatialDrive, SpatialDriveConfig, SpatialInputMap
from cnsbridge.senses.target import TargetCamera, target_mask


@pytest.fixture
def mps_lock():
    if not torch.backends.mps.is_available():
        pytest.skip("Actual MPS required")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    with workspace_lock():
        yield


def mapping():
    value = {"schema": "explicit_numerical_fixture", "nodes": 209, "inputs": 192}
    value["spatial_map_id"] = canonical_hash(value)
    return SpatialInputMap(209, np.arange(5, 197, dtype=np.int64), np.arange(1000, 1192, dtype=np.int64),
                           np.arange(192, dtype=np.int64), json.dumps(value))


def test_gpu_drive_pixels_state_replay_and_float64_difference(mps_lock):
    camera = TargetCamera(width=256, height=256)
    drive = GPUDrive(mapping(), camera=camera)
    red, green = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    for channel in (0, 1):
        frame = np.zeros((256, 256, 3), np.uint8)
        frame[:, :, 2] = red
        frame[:, :, channel] = green
        actual = drive.mask(frame)
        assert actual.device.type == "mps" and actual.dtype == torch.float32
        np.testing.assert_allclose(actual.cpu().numpy(), target_mask(frame, camera), rtol=0, atol=1e-7)
    rng = np.random.default_rng(5)
    uneven = GPUDrive(mapping(), camera=TargetCamera(width=163, height=127))
    frame = rng.integers(0, 256, (127, 163, 3), dtype=np.uint8)
    np.testing.assert_allclose(uneven.mask(frame).cpu().numpy(), target_mask(frame, uneven.camera), rtol=0, atol=1e-7)

    state, old = drive.initial_state(), SpatialDrive(drive.mapping)
    masks = (np.linspace(0, 1, 192, dtype=np.float32).reshape(MASK_SHAPE),
             np.linspace(1, 0, 192, dtype=np.float32).reshape(MASK_SHAPE))
    event_mismatches = 0
    for frame_mask in masks:
        tensor = torch.tensor(frame_mask, device="mps")
        initial = drive.snapshot(state)
        replay = drive.restore(initial)
        for _ in range(100):
            increment, state = drive.step(tensor, state)
            expected, replay = drive.step(tensor, replay)
            assert torch.equal(increment, expected) and torch.equal(state.phase, replay.phase)
            assert torch.equal(state.event_counts, replay.event_counts)
            original = old.step(frame_mask).to_dense()
            event_mismatches += int(np.count_nonzero(increment.cpu().numpy() != original))
        assert state.step == initial["step"] + 100
        # Neither restoring nor advancing changed an owned earlier state/archive.
        np.testing.assert_array_equal(initial["phase"], drive.snapshot(drive.restore(initial))["phase"])
    saved = drive.snapshot(state)
    assert saved["phase"].dtype == np.float32 and state.phase.device.type == "mps"
    assert state.event_counts.device.type == "mps"
    phase_error = float(np.max(np.abs(saved["phase"].astype(np.float64) - old.export_state().phase)))
    print(json.dumps({"gpu_drive_regular_steps": state.step, "float64_max_phase_error": phase_error,
                      "float64_step_event_mismatches": event_mismatches}))

    # Explicit near-threshold counterexample: the new float32 phase has a new ID.
    near = drive.snapshot(drive.initial_state())
    near["phase"].fill(np.float32(.985 - 1e-9))
    gpu_increment, _ = drive.step(torch.ones(MASK_SHAPE, device="mps"), drive.restore(near))
    old_state = old.reset()
    from cnsbridge.senses.spatial_drive import SpatialDriveState
    old.restore_state(SpatialDriveState(old.drive_id, old.mapping.map_id, old.config.dt_ms, 0,
                                       np.full(192, .985 - 1e-9), old_state.event_counts))
    cpu_increment = old.step(np.ones(MASK_SHAPE, np.float32)).to_dense()
    boundary_differences = int(np.count_nonzero(gpu_increment.cpu().numpy() != cpu_increment))
    assert boundary_differences == 192 and drive.drive_id != old.drive_id
    print(json.dumps({"near_threshold_float64_event_mismatches": boundary_differences}))

    invalid = {**saved, "phase": saved["phase"].astype(np.float64)}
    with pytest.raises(ValueError, match="dtype"):
        drive.restore(invalid)
    with pytest.raises(ValueError, match="identity"):
        drive.restore({**saved, "drive_id": "foreign"})
    with pytest.raises(ValueError, match="Mask"):
        drive.step(torch.full(MASK_SHAPE, float("nan"), device="mps"), state)


def test_gpu_drive_full_increment_phase_boundary_continues_and_restores(mps_lock):
    drive = GPUDrive(mapping(), config=SpatialDriveConfig(dt_ms=1.0, max_rate_hz=1000.0))
    saved = drive.snapshot(drive.initial_state())
    upper = np.nextafter(np.float32(1), np.float32(0))
    saved["phase"] = np.resize(np.array([0, .25, .5, .75, upper], dtype=np.float32), 192)
    state = drive.restore(saved)
    mask = torch.ones(MASK_SHAPE, dtype=torch.float32, device="mps")
    expected_increment = np.zeros(drive.mapping.n_nodes, np.float32)
    expected_increment[drive.mapping.input_indices] = drive.config.event_voltage_mv

    increment, advanced = drive.step(mask, state)
    np.testing.assert_array_equal(increment.cpu().numpy(), expected_increment)
    first = drive.snapshot(advanced)
    np.testing.assert_array_equal(first["phase"], saved["phase"])
    np.testing.assert_array_equal(first["event_counts"], np.ones(192, np.int64))
    assert first["step"] == 1
    # Advancing preserves the owned input state and returns a restorable state.
    np.testing.assert_array_equal(drive.snapshot(state)["phase"], saved["phase"])
    np.testing.assert_array_equal(drive.snapshot(state)["event_counts"], saved["event_counts"])
    replay = drive.restore(first)
    for expected_step in (2, 3):
        increment, advanced = drive.step(mask, advanced)
        replay_increment, replay = drive.step(mask, replay)
        assert torch.equal(increment, replay_increment)
        assert torch.equal(advanced.phase, replay.phase)
        assert torch.equal(advanced.event_counts, replay.event_counts)
        current = drive.snapshot(advanced)
        assert current["step"] == expected_step
        np.testing.assert_array_equal(current["phase"], saved["phase"])
        np.testing.assert_array_equal(current["event_counts"], np.full(192, expected_step, np.int64))
        np.testing.assert_array_equal(increment.cpu().numpy(), expected_increment)
    np.testing.assert_array_equal(first["phase"], saved["phase"])
    np.testing.assert_array_equal(first["event_counts"], np.ones(192, np.int64))
    assert "clamp rounded residual to nextbelow(1)" in drive.identity["event_rule"]
    print(json.dumps({"full_increment_boundary_steps": 3, "max_phase": float(current["phase"].max()),
                      "events_per_input": 3, "restore_after_step": 1, "drive_id": drive.drive_id}))


def parameters(values=None):
    if values is None:
        values = {NAMES[0]: np.linspace(-.1, .1, 7, dtype=np.float32),
                  NAMES[1]: np.zeros((3, 4), np.float32), NAMES[2]: np.zeros(3, np.float32)}
    return {name: torch.nn.Parameter(torch.tensor(value.copy(), device="mps")) for name, value in values.items()}


def oracle_step(p, m, v, raw, step):
    clipped = {}
    for group in ((NAMES[0],), NAMES[1:]):
        norm = np.sqrt(sum(np.square(raw[n].astype(np.float64)).sum() for n in group))
        factor = min(1., 1. / (norm + 1e-6))
        clipped.update({n: raw[n].astype(np.float64) * factor for n in group})
    for name in NAMES:
        g = clipped[name]
        m[name] = .9 * m[name] + .1 * g
        v[name] = .999 * v[name] + .001 * g * g
        lr = .001 if name == NAMES[0] else .0001
        p[name] -= lr * (m[name] / (1 - .9 ** step)) / (np.sqrt(v[name] / (1 - .999 ** step)) + 1e-8)
    return clipped


def test_gpu_fused_adam_stable_clipping_multistep_restore(mps_lock):
    params = parameters()
    optimizer = GPUAdam(params)
    initial = optimizer.snapshot()
    reference = {n: initial["parameters"][n].astype(np.float64) for n in NAMES}
    first = {n: np.zeros_like(v) for n, v in reference.items()}
    second = {n: np.zeros_like(v) for n, v in reference.items()}
    errors = {"parameters": 0., "first_moment": 0., "second_moment": 0.}
    for step, magnitude in enumerate((1e30, 0., 1e-8, .25), 1):
        raw = {n: (np.linspace(-1, 1, p.numel(), dtype=np.float32).reshape(tuple(p.shape)) * np.float32(magnitude))
               for n, p in params.items()}
        for name, parameter in params.items():
            parameter.grad = torch.tensor(raw[name], device="mps")
        expected_clip = oracle_step(reference, first, second, raw, step)
        evidence = optimizer.step()
        saved = optimizer.snapshot()
        assert evidence["steps"] == step
        for name in NAMES:
            clipped = evidence["clipped"][name].cpu().numpy()
            np.testing.assert_allclose(clipped, expected_clip[name], rtol=2e-5, atol=1e-8)
            if magnitude == 1e-8:
                assert np.array_equal(clipped != 0, raw[name] != 0)
                np.testing.assert_array_equal(np.sign(clipped), np.sign(raw[name]))
            for label, actual, expected in (
                ("parameters", saved["parameters"][name], reference[name]),
                ("first_moment", saved["moments"][f"{name}.exp_avg"], first[name]),
                ("second_moment", saved["moments"][f"{name}.exp_avg_sq"], second[name]),
            ):
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-8)
                errors[label] = max(errors[label], float(np.max(np.abs(actual.astype(np.float64) - expected))))
            assert float(saved["moments"][f"{name}.step"]) == step
        assert all(t.device.type == "mps" for state in optimizer.optimizer.state.values() for t in state.values())
        if step == 3:
            params = parameters(saved["parameters"])
            optimizer = GPUAdam(params, moments=saved["moments"], steps=saved["steps"])
            restored = optimizer.snapshot()
            for kind in ("parameters", "moments"):
                for name in saved[kind]:
                    np.testing.assert_array_equal(restored[kind][name], saved[kind][name])
    print(json.dumps({"gpu_fused_adam_steps": 4, "restore_after_step": 3, "max_abs_oracle_error": errors}))

    before = optimizer.snapshot()
    params[NAMES[0]].grad = torch.full_like(params[NAMES[0]], float("inf"))
    with pytest.raises(ValueError, match="raw gradients"):
        optimizer.step()
    after = optimizer.snapshot()
    assert before["steps"] == after["steps"]
    for kind in ("parameters", "moments"):
        for name in before[kind]:
            np.testing.assert_array_equal(before[kind][name], after[kind][name])

    # Tiny first-step moments and correction cannot disappear behind an atol.
    tiny_params = parameters({n: np.zeros_like(v) for n, v in initial["parameters"].items()})
    tiny = GPUAdam(tiny_params)
    for p in tiny_params.values():
        p.grad = torch.full_like(p, 1e-8)
    tiny.step()
    saved = tiny.snapshot()
    for name in NAMES:
        assert (saved["parameters"][name] < 0).all()
        assert (saved["moments"][f"{name}.exp_avg"] > 0).all()
        assert (saved["moments"][f"{name}.exp_avg_sq"] > 0).all()
