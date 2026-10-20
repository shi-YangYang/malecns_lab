"""Explicit tiny numerical fixtures; never complete-connectome training evidence."""
from copy import deepcopy
from dataclasses import replace
import os
import subprocess
import sys
import time
import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")
from malecns_lab.brain.metadata import canonical_hash
from malecns_lab.brain.mps_trainable import MPSLIF, fixed_mps_matvec, checked_coo
from malecns_lab.brain.trainable import TrainableLIFConfig
from malecns_lab.learning.mps_policy import MPSPolicy
from malecns_lab.learning.mps_optimizer import snapshot
from malecns_lab.learning.mps_state import restore, boundary_arrays, state_arrays, state_digest
from malecns_lab.learning.mps_archive import write_checkpoint, inspect_checkpoint, verify_resume
from malecns_lab.learning.mps_contract import source_hashes, runtime, validate_run, initial_prefix
from malecns_lab.learning.mps_evidence import scan_run
from malecns_lab.learning.mps_window import compute_window as mps_window
from malecns_lab.learning.training_checkpoint import _state_metadata
from malecns_lab.learning.training_contract import PARAMETERS, IDENTITIES, cursor, bytes_sha
from malecns_lab.learning.training_optimizer import make_training_optimizer
from malecns_lab.learning.training_window import compute_window, optimizer_snapshot
from malecns_lab.learning.resource_guard import GIB, resource_budget, write_json
from malecns_lab.learning.train_snn import workspace_lock
from malecns_lab.learning.train_mps import run_segment
from test_training_checkpoint import policy
from test_neural_policy import frame


@pytest.fixture(scope="module", autouse=True)
def numerical_slot():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; actual GPU checks not executed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with workspace_lock():
        yield


def selection(start):
    return {"frames": np.stack([frame(), frame(red=False)]), "time_s": np.arange(start, start + 2, dtype=np.float64) * .1,
            "valid": np.ones(2, bool), "teacher_target": np.array([[.2, -.1, .3], [-.2, .1, -.3]], np.float64),
            "indices": np.arange(start, start + 2, dtype=np.int64)}


@pytest.fixture(scope="module")
def resumed():
    return _migrated(policy())


def _migrated(cpu):
    optimizer = make_training_optimizer(cpu, "joint")
    state, _, _ = compute_window(cpu, optimizer, cpu.initial_state(), selection(0), mode="joint", before_steps=0)
    params = {k: dict(cpu.named_parameters())[n].detach().numpy().copy() for k, n in PARAMETERS.items()}
    moments = optimizer_snapshot(cpu, optimizer, "joint", 1)
    arrays = {**state_arrays(state), **{f"parameter.{k}": v for k, v in params.items()}, **{f"adam.{k}": v for k, v in moments.items()}}
    previous = {"checkpoint_id": "1" * 64, "path": "explicit_numerical_fixture", "parameters": params, "moments": moments,
                "metadata": {"state": _state_metadata(state)}, "arrays": arrays,
                "context": {"optimizer_steps": 1, "completed_windows": 1, "next_frame": 2}}
    brain = MPSLIF(cpu.brain.graph, cpu.brain.input_body_ids, config=cpu.brain.config, tonic_types=("L1",))
    gpu = MPSPolicy.from_brain(brain, cpu.mapping, cpu.identity)
    bridge, migrated, actual = restore(gpu, previous)
    assert all(np.array_equal(actual[k], v) for k, v in arrays.items())
    previous["state"] = state_digest(migrated)
    return cpu, gpu, bridge, migrated, previous


def test_checkpoint_matches_direct_multiple_blocks_two_frames_and_raw_gradients():
    from malecns_lab.brain.mps_identity import ACTIVATION_CHECKPOINT
    from malecns_lab.learning.neural_policy import NeuralPolicy
    from malecns_lab.learning.mps_contract import PROTOCOL_MPS
    source = policy()
    cpu = NeuralPolicy.for_numerical_test(
        source.brain, source.mapping, background=source.background, camera=source.camera,
        config=replace(source.config, frame_ms=12), drive_config=source.drive_config)
    _, gpu, _, migrated, _ = _migrated(cpu)
    assert gpu.frame_steps == 120  # 50 + 50 + 20; the half-frame boundary is inside a block.
    assert gpu.identity["activation_checkpoint"] == PROTOCOL_MPS["activation_checkpoint"] == ACTIVATION_CHECKPOINT
    assert "learning/mps_recompute.py" in gpu.identity["implementation_sha256"]
    assert "learning/mps_recompute.py" in source_hashes()
    brain = replace(migrated.brain,
                    v=torch.tensor([-47.1, -46.8, -45.5, -48.2, -49.], device="mps"),
                    g=torch.tensor([.5, .4, 1.5, 2., .2], device="mps"),
                    pending=tuple(torch.full((5,), .03 + i * .01, device="mps") for i in range(gpu.brain.delay_steps)),
                    refractory_until=torch.tensor([0, 0, migrated.brain.step + 15, 0, 0], device="mps"))
    initial = replace(migrated, brain=brain)
    original = state_arrays(initial)
    assert np.count_nonzero(original["brain.pending"]) and np.count_nonzero(original["brain.g"])
    assert np.count_nonzero(original["drive.event_counts"]) and np.count_nonzero(original["drive.phase"])
    results = []
    for forward in (gpu.forward_direct_for_numerical_test, gpu.forward_tensor):
        gpu.zero_grad(set_to_none=True)
        leaves = [v.detach().clone().requires_grad_() for v in (brain.v, brain.g, *brain.pending)]
        state = replace(initial, brain=replace(brain, v=leaves[0], g=leaves[1], pending=tuple(leaves[2:])))
        first = forward(frame(), state)
        first.state.brain.v.retain_grad()
        second = forward(frame(red=False), first.state)
        before_backward = state_arrays(second.state)
        actions = torch.stack([first.action, second.action])
        target = torch.tensor([[.2, -.1, .3], [-.2, .1, -.3]], device="mps")
        loss = ((actions - target) / gpu.action_scales).square().mean() + 1e-4 * (gpu.brain.gains - 1).square().mean()
        loss.backward()
        assert first.state.brain.v.grad is not None and torch.count_nonzero(first.state.brain.v.grad)
        raw = {name: p.grad.detach().cpu().numpy().copy() for name, p in gpu.named_parameters()}
        assert np.count_nonzero(raw["brain.theta"]) and all(np.isfinite(v).all() for v in raw.values())
        for index, value in enumerate(leaves):
            assert value.grad is not None and torch.isfinite(value.grad).all()
            raw[f"initial.{index}"] = value.grad.detach().cpu().numpy().copy()
        assert np.count_nonzero(raw["initial.2"])  # Pending-arrival gradients remain connected.
        after_backward = state_arrays(second.state)
        assert all(np.array_equal(after_backward[k], v) for k, v in before_backward.items())
        assert all(np.array_equal(state_arrays(initial)[k], v) for k, v in original.items())
        assert second.state.brain.step == second.state.drive.step == brain.step + 240
        results.append((actions.detach().cpu().numpy().copy(),
                        torch.stack([first.features_mv, second.features_mv]).detach().cpu().numpy().copy(),
                        after_backward, raw))
    a, b = results
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    for key in a[2]:
        np.testing.assert_array_equal(a[2][key], b[2][key])
    for key in a[3]:
        np.testing.assert_allclose(a[3][key], b[3][key], rtol=1e-5, atol=1e-7, err_msg=key)
    print({"direct_checkpoint_actions_dn_state_exact": True,
           "raw_gradient_max_abs_difference": {key: float(np.max(np.abs(a[3][key] - b[3][key]))) for key in a[3]}})


def numeric_run(gpu):
    return {"schema": "explicit_numerical_fixture_only", "mode": "joint", "policy": gpu.identity,
            **{k: canonical_hash({"numerical": k}) for k in IDENTITIES}, "runtime": runtime(),
            "implementation_sha256": source_hashes(), "scenes": [{"scene_id": "numeric", "frame_count": 8}],
            "cpu_parent": {"training_run_id": "2" * 64, "prefix_sha256": "3" * 64, "checkpoint_id": "1" * 64}}


def test_parent_is_lightweight_and_source_mode_guards():
    subprocess.run([sys.executable, "-c", "import sys; import malecns_lab.learning.train_mps; import malecns_lab.learning.mps_worker; assert 'torch' not in sys.modules"], check=True)
    import malecns_lab.learning.mps_contract as contract
    paths = contract.source_hashes()
    assert set(contract.NEW_SOURCES) <= set(paths)
    for field, bad in (("mode", "frozen_brain"), ("implementation_sha256", {})):
        fake = {"schema": contract.SCHEMA, "registered_utc": "fixture", "mode": "joint", "protocol": contract.PROTOCOL_MPS,
                "protocol_id": canonical_hash(contract.PROTOCOL_MPS), "scenes": contract.SCENES, "graph": {}, "dataset_path": "bad",
                "dataset_id": "0" * 64, "dataset_manifest_sha256": "0" * 64, "cpu_parent": {}, "policy": {},
                "implementation_sha256": paths, "runtime": runtime(), "numeric_precedent": contract.NUMERIC_PRECEDENT}
        fake[field] = bad
        fake["training_run_id"] = canonical_hash(fake)
        with pytest.raises(ValueError, match="protocol/mode" if field == "mode" else "sources"):
            validate_run(fake)


def test_mps_signed_adjoint_silent_cell_gradient_and_state_guards():
    counts = sparse.csc_array([[0, -2, 0], [3, 0, 1], [0, 0, 0]])
    for matrix in (counts, counts.T):
        coo = matrix.tocoo()
        reference = torch.sparse_coo_tensor(torch.tensor(np.stack((coo.row, coo.col)), dtype=torch.int64),
                                            torch.tensor(coo.data, dtype=torch.float32) * .275,
                                            size=coo.shape).coalesce()
        cached = checked_coo(matrix, .275)
        assert cached.is_coalesced() and not cached.requires_grad and cached.device.type == "mps"
        assert torch.equal(cached.indices().cpu(), reference.indices())
        assert torch.equal(cached.values().cpu(), reference.values())
    duplicate = sparse.csr_array((np.array([1, 2]), np.array([1, 1]), np.array([0, 2, 2])), shape=(2, 2))
    with pytest.raises(ValueError, match="canonical"):
        checked_coo(duplicate, .275)
    w, wt = checked_coo(counts, 1.), checked_coo(counts.T, 1.)
    x = torch.tensor([0., .5, 0.], device="mps", requires_grad=True)
    y = fixed_mps_matvec(w, wt, x)
    np.testing.assert_array_equal(y.detach().cpu().numpy(), counts @ x.detach().cpu().numpy())
    gradient = torch.tensor([.7, -1.2, 2.], device="mps")
    y.backward(gradient)
    np.testing.assert_allclose(x.grad.cpu().numpy(), counts.T @ gradient.cpu().numpy(), rtol=1e-6)
    assert x.grad[0] != 0 and x.grad[2] != 0
    cfg = TrainableLIFConfig(v_rest_mv=0, v_reset_mv=0, v_threshold_mv=1, delay_ms=.2, refractory_ms=.3, weight_mv=1)
    brain = MPSLIF.for_numerical_test(sparse.csc_array([[0, 0, 0], [3, 0, 0], [0, 0, 0]]), [0],
                                      types=np.array(["R1-R6", "DN", "L1"]), tonic_types=("L1",), config=cfg)
    first = torch.tensor([.5, 0., 0.], device="mps", requires_grad=True)
    spikes, state = brain.step(first, brain.reset())
    assert not spikes.any()
    for _ in range(2):
        _, state = brain.step(torch.zeros(3, device="mps"), state)
    state.g[1].backward()
    assert first.grad[0].item() == pytest.approx(3 * np.exp(-.1 / 5) / 1.5 ** 2, rel=1e-5)
    with pytest.raises(ValueError, match="identity"):
        brain.step(torch.zeros(3, device="mps"), replace(state, model_id="foreign"))
    with pytest.raises(ValueError, match="input"):
        brain.step(torch.tensor([0., 1., 0.], device="mps"), state)


def test_actual_mps_window_cpu_adam_exact_archive_restore_and_scene_reset(resumed, tmp_path):
    cpu, gpu, bridge, state, previous = resumed
    run = numeric_run(gpu)
    reg = {"segment_id": "4" * 64}
    np.savez(tmp_path / "resume.npz", **boundary_arrays(gpu, bridge, state))
    from malecns_lab.learning.training_contract import sha256
    write_json(tmp_path / "resume.json", {"schema": "cnsbridge.mps_restore.v1", "segment_id": reg["segment_id"],
        "parent_checkpoint_id": previous["checkpoint_id"], "source_state": previous["metadata"]["state"],
        "restored_state": _state_metadata(state), "arrays_sha256": sha256(tmp_path / "resume.npz"), "exact": True})
    assert verify_resume(tmp_path, reg, previous, run) == state_digest(state)
    state, values, record = mps_window(gpu, bridge, state, selection(2))
    assert record["numeric"]["parameter_updates"]["theta"]["changed"] > 0 and bridge.steps == 2
    assert record["numeric"]["clipping"]["theta"]["nonzero_before"] > 0
    context = {**{k: run[k] for k in IDENTITIES}, "scenes": run["scenes"], **cursor(run["scenes"], 2, 2),
               "parent_checkpoint_id": previous["checkpoint_id"], "log_prefix_sha256": "5" * 64}
    arrays = boundary_arrays(gpu, bridge, state)
    write_checkpoint(tmp_path / "checkpoint", run, context, _state_metadata(state), arrays)
    saved = inspect_checkpoint(tmp_path / "checkpoint", run)
    for bad_name in ("extra", "nonfinite", "shape", "moment", "clock"):
        bad_arrays = {k: v.copy() for k, v in arrays.items()}
        bad_info = deepcopy(_state_metadata(state))
        if bad_name == "extra":
            bad_arrays["unexpected"] = np.zeros(1)
        elif bad_name == "nonfinite":
            bad_arrays["brain.v"][0] = np.nan
        elif bad_name == "shape":
            bad_arrays["parameter.theta"] = bad_arrays["parameter.theta"][:-1]
        elif bad_name == "moment":
            bad_arrays["adam.brain.theta.step"] = np.array(1, np.float32)
        else:
            bad_info["brain"]["step"] += 1
        with pytest.raises(ValueError):
            write_checkpoint(tmp_path / bad_name, run, context, bad_info, bad_arrays)
    second = MPSPolicy.from_brain(MPSLIF(cpu.brain.graph, cpu.brain.input_body_ids, config=cpu.brain.config, tonic_types=("L1",)), cpu.mapping, cpu.identity)
    bridge2, state2, again = restore(second, saved)
    assert all(np.array_equal(again[k], v) for k, v in arrays.items())
    assert state_digest(state2) == state_digest(state)
    next_state, _, next_record = mps_window(second, bridge2, state2, selection(4))
    assert bridge2.steps == 3 and next_state.frame_index == 6 and next_record["start_state"] == state_digest(state2)
    before, moments = snapshot(second), bridge2.moments()
    reset = second.initial_state()
    assert reset.frame_index == 0 and reset.brain.step == reset.drive.step == second.precondition_steps
    assert all(np.array_equal(snapshot(second)[k], v) for k, v in before.items())
    assert all(np.array_equal(bridge2.moments()[k], v) for k, v in moments.items())
    # Same bytes with wrong run, extra fields, changed dtype or nonfinite state are rejected.
    foreign = deepcopy(run)
    foreign["training_run_id"] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        inspect_checkpoint(tmp_path / "checkpoint", foreign)
    np.savez(tmp_path / "checkpoint/arrays.npz", **{**arrays, "unexpected": np.zeros(1)})
    with pytest.raises(ValueError, match="SHA"):
        inspect_checkpoint(tmp_path / "checkpoint", run)


def test_unsealed_window_not_progress_and_bad_seal_rejected(resumed, tmp_path):
    _, gpu, _, _, initial = resumed
    run = numeric_run(gpu)
    root = tmp_path / "numeric"
    segment = root / "segments/000001"
    window = segment / "windows/000002"
    window.mkdir(parents=True)
    resources = {**resource_budget(2 * GIB, reserve_bytes=GIB, rss_bytes=6 * GIB), "wall_limit_seconds": 600, "sampling_seconds": .1}
    reg = {"schema": "cnsbridge.mps_segment.v1", "training_run_id": run["training_run_id"], "segment": 1,
           "start_cursor": cursor(run["scenes"], 1, 1), "parent_checkpoint_id": initial["checkpoint_id"],
           "prefix_sha256": bytes_sha(initial_prefix(run)), "windows": 1, "resources": resources,
           "clock_monotonic": time.monotonic(), "parent_pid": os.getpid()}
    reg["segment_id"] = canonical_hash(reg)
    write_json(segment / "registration.json", reg)
    chain = scan_run(root, run, None, initial, current_segment=1)
    assert chain["completed_windows"] == 1 and chain["last"] is initial
    write_json(window / "window.json", {})
    with pytest.raises((ValueError, FileNotFoundError)):
        scan_run(root, run, None, initial, current_segment=1)
    (window / "window.json").unlink()
    write_json(segment / "worker_result.json", {"failure_kind": "numerical_or_protocol"})
    with pytest.raises(ValueError, match="blocks continuation"):
        scan_run(root, run, None, initial)
    write_json(segment / "worker_result.json", {"failure_kind": "resource_or_interrupt"})
    write_json(segment / "result.json", {"status": "failed", "verified_end_windows": 1, "terminal_artifacts_sha256": {}})
    assert scan_run(root, run, None, initial)["completed_windows"] == 1


def test_cli_rejects_invalid_budget_and_existing_output_without_worker(tmp_path, monkeypatch):
    # Run in a separate process: the fixture owns the numerical lock, so this
    # additionally verifies cross-process exclusion without touching a model.
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="between"):
        run_segment(directory=existing, windows=0, cpu_run=tmp_path / "absent")
    result = subprocess.run([sys.executable, "-m", "malecns_lab.learning.train_mps", "--cpu-run", str(tmp_path / "absent"),
                             "--out", str(existing), "--windows", "1"], text=True, capture_output=True)
    assert result.returncode == 1 and "owns the workspace lock" in result.stdout and not list(existing.iterdir())
    from contextlib import nullcontext
    import malecns_lab.learning.train_mps as runner
    # Isolate the subsequent preflight branch; the module fixture still owns the real numerical lock.
    monkeypatch.setattr(runner, "workspace_lock", nullcontext)
    with pytest.raises(ValueError, match="already exists"):
        runner.run_segment(directory=existing, windows=1, cpu_run=tmp_path / "absent")
    failed = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        runner.run_segment(directory=failed, windows=1, cpu_run=tmp_path / "absent")
    assert (failed / "registration_failure.json").exists() and not (failed / "segments").exists()
