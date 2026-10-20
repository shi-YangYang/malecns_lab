"""New GPU training interfaces only; explicit small circuits are not brain evidence."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from malecns_lab.brain.metadata import canonical_hash
from malecns_lab.brain.mps_trainable import MPSLIF
from malecns_lab.learning.neural_policy import NeuralPolicy
from malecns_lab.learning.gpu_policy import GPUPolicy
from malecns_lab.learning.gpu_optimizer import GPUAdam
from malecns_lab.learning.gpu_window import compute_window, logit_loss
from malecns_lab.learning.gpu_state import boundary_arrays, restore, state_arrays, state_digest, state_metadata
from malecns_lab.learning.gpu_archive import write_checkpoint, inspect_checkpoint, validate_payload, verify_resume
from malecns_lab.learning.gpu_contract import source_hashes, runtime, protocol, optimizer_identity
from malecns_lab.learning.gpu_numbers import validate_numbers, validate_correction
from malecns_lab.learning.gpu_evidence import validate_registration, validate_residency
from malecns_lab.learning.train_snn import workspace_lock
from malecns_lab.learning.train_gpu import run_job
from malecns_lab.learning.training_contract import PARAMETERS, IDENTITIES, cursor, bytes_sha
from malecns_lab.learning.resource_guard import resource_budget, GIB, write_json
from test_training_checkpoint import policy as cpu_policy
from test_neural_policy import frame


@pytest.fixture(scope="module", autouse=True)
def mps_slot():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_available():
        pytest.skip("Actual MPS required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with workspace_lock():
        yield


def make_policy(horizon=100, frame_ms=12):
    old = cpu_policy()
    cpu = NeuralPolicy.for_numerical_test(old.brain, old.mapping, background=old.background,
        camera=old.camera, config=replace(old.config, frame_ms=frame_ms), drive_config=old.drive_config)
    brain = MPSLIF(cpu.brain.graph, cpu.brain.input_body_ids, config=cpu.brain.config, tonic_types=("L1",))
    policy = GPUPolicy.from_brain(brain, cpu.mapping, cpu.identity, credit_horizon=horizon)
    values = {k: dict(cpu.named_parameters())[n].detach().numpy().copy() for k, n in PARAMETERS.items()}
    values["theta"] += np.float32(.1)
    policy.restore_parameters(values)
    return policy, cpu, values


def selection(start=0):
    return {"frames": np.stack([frame(), frame(red=False)]), "time_s": np.arange(start, start + 2, dtype=np.float64) * .1,
            "valid": np.ones(2, bool), "teacher_target": np.array([[-.2, -.1, .3], [-.1, .1, -.3]], np.float64),
            "indices": np.arange(start, start + 2, dtype=np.int64)}


def run_identity(policy):
    return {**{k: canonical_hash({"fixture": k}) for k in IDENTITIES}, "mode": "joint", "policy": policy.identity,
            "runtime": runtime(), "implementation_sha256": source_hashes(), "optimizer": optimizer_identity(),
            "protocol": protocol(100, 1000), "scenes": [{"scene_id": "fixture", "frame_count": 4}]}


def context(run, completed, parent="1" * 64):
    return {**{k: run[k] for k in IDENTITIES}, "scenes": run["scenes"], **cursor(run["scenes"], completed, completed),
            "parent_checkpoint_id": parent, "log_prefix_sha256": bytes_sha(b"fixture")}


def test_light_parent_and_job_budget_guards(tmp_path):
    code = "import sys; import malecns_lab.learning.train_gpu; import malecns_lab.learning.gpu_worker; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    assert not any("gradient_probe" in k for k in source_hashes())
    with pytest.raises(ValueError, match="Invalid window"):
        run_job(directory=tmp_path / "unused", kind="train", windows=5)
    assert not (tmp_path / "unused").exists()
    run = {"training_run_id": "a" * 64, "scenes": [{"scene_id": "fixture", "frame_count": 320}]}
    previous = {"checkpoint_id": "b" * 64}
    reg = {"schema": "cnsbridge.gpu_recovery_job.v1", "kind": "train", "training_run_id": run["training_run_id"],
           "segment": 1, "start_cursor": cursor(run["scenes"], 0, 0), "parent_checkpoint_id": previous["checkpoint_id"],
           "prefix_sha256": bytes_sha(b""), "windows": 2,
           "resources": {**resource_budget(8 * GIB, rss_bytes=6 * GIB, reserve_bytes=GIB), "wall_limit_seconds": 600, "sampling_seconds": .1},
           "clock_monotonic": 1., "parent_pid": os.getpid()}
    reg["job_id"] = canonical_hash(reg)
    with pytest.raises(ValueError, match="First two"):
        validate_registration(reg, run, "train", previous, segment=1)
    p = subprocess.run([sys.executable, "-m", "malecns_lab.learning.train_gpu", "--worker", str(tmp_path)], capture_output=True)
    assert p.returncode != 0 and not (tmp_path / "worker.claim").exists()


def test_gpu_recompute_preserves_forward_and_pure_sensory_with_credit_cuts():
    outcomes = []
    for horizon, direct in ((100, True), (100, False), (1000, False)):
        policy, _, _ = make_policy(horizon)
        initial = policy.initial_state()
        brain = replace(initial.brain, v=torch.tensor([-47.1, -46.8, -45.5, -48.2, -49.], device="mps"),
                        g=torch.tensor([.5, .4, 1.5, 2., .2], device="mps"),
                        pending=tuple(torch.full((5,), .03 + i * .01, device="mps") for i in range(policy.brain.delay_steps)))
        initial = replace(initial, brain=brain)
        original = state_arrays(initial)
        first = policy.forward_tensor(frame(), initial, frame_offset=0, direct_numerical_fixture=direct)
        second = policy.forward_tensor(frame(red=False), first.state, frame_offset=policy.frame_steps, direct_numerical_fixture=direct)
        last = state_arrays(second.state)
        logits = torch.stack([first.logits, second.logits])
        target = torch.tensor(selection()["teacher_target"], dtype=torch.float32, device="mps")
        loss = logit_loss(logits, target, policy.action_scales) + 1e-4 * (policy.brain.gains - 1).square().mean()
        loss.backward()
        raw = {k: dict(policy.named_parameters())[n].grad.cpu().numpy().copy() for k, n in PARAMETERS.items()}
        assert all(np.isfinite(v).all() for v in raw.values())
        assert all(np.array_equal(v, state_arrays(second.state)[k]) for k, v in last.items())
        assert all(np.array_equal(v, state_arrays(initial)[k]) for k, v in original.items())
        outcomes.append((logits.detach().cpu().numpy(), torch.stack([first.features_mv, second.features_mv]).detach().cpu().numpy(), last, raw))
    for other in outcomes[1:]:
        np.testing.assert_array_equal(outcomes[0][0], other[0])
        np.testing.assert_array_equal(outcomes[0][1], other[1])
        for k in outcomes[0][2]:
            np.testing.assert_array_equal(outcomes[0][2][k], other[2][k])
    for k in PARAMETERS:
        np.testing.assert_allclose(outcomes[0][3][k], outcomes[1][3][k], rtol=2e-5, atol=1e-7)
    assert not np.array_equal(outcomes[1][3]["theta"], outcomes[2][3]["theta"])
    np.testing.assert_array_equal(outcomes[1][3]["readout_weight"], outcomes[2][3]["readout_weight"])
    print(json.dumps({"pure_checkpoint_forward_exact": True, "credit_cut_changes_theta": True,
                      "theta_norms": [float(np.linalg.norm(o[3]["theta"].astype(float))) for o in outcomes]}))


def test_correction_update_checkpoint_roundtrip_and_reject_forged_state(tmp_path):
    policy, cpu, inherited = make_policy(frame_ms=4)
    with torch.no_grad():
        policy.readout.bias[0] = 10
    optimizer = GPUAdam(policy.named_parameters())
    run = run_identity(policy)
    initial = policy.initial_state()
    assert optimizer.identity == run["optimizer"]
    write_checkpoint(tmp_path / "genesis", run, context(run, 0), state_metadata(initial, policy.drive_config.dt_ms),
                     boundary_arrays(policy, optimizer, initial))
    genesis = inspect_checkpoint(tmp_path / "genesis", run)
    assert all(not np.count_nonzero(v) for v in genesis["moments"].values())
    original = state_digest(policy, initial)
    state, values, record = compute_window(policy, optimizer, initial, selection(), theta_norm_limit=1000)
    correction = validate_correction(values, record["numeric"])
    validate_residency(record["residency"])
    assert correction["fixed_feature_update_corrects_vx"] and not correction["actual_action_sign_reversed"]
    assert record["start_state"] == original
    write_checkpoint(tmp_path / "updated", run, context(run, 1, genesis["checkpoint_id"]),
                     state_metadata(state, policy.drive_config.dt_ms), boundary_arrays(policy, optimizer, state))
    saved = inspect_checkpoint(tmp_path / "updated", run)
    second, _, _ = make_policy(frame_ms=4)
    other_optimizer, other_state, actual = restore(second, saved)
    assert all(np.array_equal(v, actual[k]) for k, v in saved["arrays"].items())
    np.savez(tmp_path / "resume.npz", **actual)
    write_json(tmp_path / "resume.json", {"job_id": "c" * 64, "parent_checkpoint_id": saved["checkpoint_id"],
        "state": state_metadata(other_state, second.drive_config.dt_ms),
        "arrays_sha256": __import__("malecns_lab.learning.training_contract", fromlist=["sha256"]).sha256(tmp_path / "resume.npz")})
    verify_resume(tmp_path, {"job_id": "c" * 64}, saved, run)
    continued = []
    for model, opt, current in ((policy, optimizer, state), (second, other_optimizer, other_state)):
        end, evidence, result = compute_window(model, opt, current, selection(2), theta_norm_limit=1000)
        continued.append((boundary_arrays(model, opt, end), result))
    for key in continued[0][0]:
        np.testing.assert_array_equal(continued[0][0][key], continued[1][0][key])
    # A self-consistent hash cannot make float64 sensory state a valid new archive.
    forged = {k: v.copy() for k, v in saved["arrays"].items()}
    forged["drive.phase"] = forged["drive.phase"].astype(np.float64)
    info = deepcopy(saved["metadata"])
    info["arrays"]["drive.phase"]["dtype"] = forged["drive.phase"].dtype.str
    info.pop("checkpoint_id")
    info["checkpoint_id"] = canonical_hash(info)
    with pytest.raises(ValueError, match="Invalid array drive.phase"):
        validate_payload(info, forged, run)
    tampered = {k: v.copy() for k, v in values.items()}
    tampered["pre_clip_readout_bias"][0] *= -1
    with pytest.raises(ValueError, match="gradient"):
        validate_numbers(tampered, policy.identity, 0, 1000)
    with pytest.raises(ValueError, match="gate"):
        validate_numbers(values, policy.identity, 0, 1e-30)
    print(json.dumps({"negative_target_correction": correction, "raw_theta_norm": record["numeric"]["clipping"]["theta"]["norm_before"],
                      "fixed_vx_loss_before": record["numeric"]["fixed_vx_loss_before"],
                      "fixed_vx_loss_after": record["numeric"]["fixed_vx_loss_after"], "restored_next_update_exact": True}))
