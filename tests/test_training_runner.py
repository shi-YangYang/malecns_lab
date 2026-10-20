"""P3c2 bounded training, using explicitly numerical-only circuits and process fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from malecns_lab.brain.metadata import canonical_hash
from malecns_lab.learning import training_checkpoint as checkpoint
from malecns_lab.learning import training_contract as contract
from malecns_lab.learning import training_evidence as evidence
from malecns_lab.learning import train_snn
from malecns_lab.learning.training_math import adam_moments, adam_parameters, clipping, compare_r2, moment_check
from malecns_lab.learning.training_optimizer import make_training_optimizer
from malecns_lab.learning.training_window import compute_window
from malecns_lab.learning.training_worker import execute_windows
from malecns_lab.learning.resource_guard import GIB, write_json
from test_training_checkpoint import policy
from test_neural_policy import frame, assert_same_state


def numeric_data():
    sequences = []
    for name in ("numeric-a", "numeric-b"):
        sequences.append(SimpleNamespace(scene={"scene_id": name}, frames=np.stack([frame()] * 4),
            time_s=np.arange(4, dtype=np.float64) * .1, valid=np.ones(4, bool),
            teacher_target=np.array([[.2, -.1, .3], [-.2, .1, -.3], [.1, .15, -.2], [-.1, -.15, .2]], np.float64)))
    return SimpleNamespace(sequences=sequences)


def numeric_run(model, mode):
    return {"schema": "numerical_test_only", "mode": mode, "policy": model.identity,
            **{key: canonical_hash({"numeric": key, "mode": mode}) for key in contract.IDENTITIES},
            "runtime": checkpoint._runtime(), "implementation_sha256": contract.source_hashes(),
            "protocol": {"windows": 4}, "scenes": [{"scene_id": s.scene["scene_id"], "frame_count": len(s.frames)}
                                                     for s in numeric_data().sequences]}


def sample_resources(segment, registration):
    # Explicit fixture, never production resource evidence.
    elapsed = time.monotonic() - registration["clock_monotonic"] + .1
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n"
        f"0.0,1000000,{2*GIB}\n{elapsed},1000000,{2*GIB}\n")


def numeric_segment(root, run, model, optimizer, state, windows):
    data = numeric_data()
    chain = evidence.scan_run(root, run, data, production=False)
    segment, registration = train_snn.register_segment(root, run, chain, windows)
    state, _ = execute_windows(segment, run, data, registration, model, optimizer, state, chain["last"], chain["prefix"], production=False)
    sample_resources(segment, registration)
    chain = evidence.scan_run(root, run, data, production=False)
    return state, chain, segment


def numeric_worker(path):
    """Separate-process fixture endpoint, absent from the production CLI."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path(path)
    run = contract.read_json(root / "numeric_run.json")
    model = policy()
    optimizer = make_training_optimizer(model, run["mode"])
    chain = evidence.scan_run(root, run, numeric_data(), production=False)
    state = None
    if chain["last"]:
        state, _ = checkpoint.restore_training_checkpoint(model, optimizer, chain["last"]["path"], expected=contract.expected_ids(run))
    numeric_segment(root, run, model, optimizer, state, 1)


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


@pytest.mark.parametrize("mode", ("joint", "frozen_brain"))
def test_actual_multistep_adam_and_cancellation_underflow(mode):
    rng = np.random.default_rng(12)
    sizes = (11752, 7884, 3) if mode == "joint" else (7884, 3)
    parameters = [torch.tensor(rng.normal(0, .01, n).astype(np.float32), requires_grad=True) for n in sizes]
    groups = [{"params": [p], "lr": .001 if mode == "joint" and i == 0 else .0001} for i, p in enumerate(parameters)]
    optimizer = torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    for step in range(1, 161):
        for index, p in enumerate(parameters):
            before = p.detach().numpy().copy()
            old = optimizer.state.get(p, {})
            m = old["exp_avg"].numpy().copy() if old else np.zeros(p.shape, np.float32)
            v = old["exp_avg_sq"].numpy().copy() if old else np.zeros(p.shape, np.float32)
            g = rng.normal(0, .2, len(before)).astype(np.float32)
            g[::11], g[1::11], g[2::11] = 1e-30, -1e-30, 0
            if step % 4 == 0:
                g[3::11] = (-9 * m[3::11]).astype(np.float32)
            if index == 0 and step == 8:
                g = clipping({"readout_weight": g, "readout_bias": np.zeros(3, np.float32)}, "frozen_brain")[0]["readout_weight"]
            p.grad = torch.from_numpy(g.copy())
            # Each optimizer.step updates all parameters; isolate per-array state checks after one shared step below.
            p._check = before, m, v, g
        optimizer.step()
        for index, p in enumerate(parameters):
            before, old_m, old_v, g = p._check
            actual = optimizer.state[p]
            m, v = adam_moments(g, old_m, old_v)
            moment_check(actual["exp_avg"].numpy(), m)
            moment_check(actual["exp_avg_sq"].numpy(), v)
            expected = adam_parameters(before, actual["exp_avg"].numpy(), actual["exp_avg_sq"].numpy(), step,
                                       optimizer.param_groups[index]["lr"])
            np.testing.assert_allclose(p.detach().numpy(), expected, rtol=5e-6, atol=1e-9)
    old = np.array([-.0019786758348345757], np.float32)
    grad = np.array([.017808081582188606], np.float32)
    actual = torch.from_numpy(old.copy()).lerp_(torch.from_numpy(grad), .1).numpy()
    m, _ = adam_moments(grad, old, np.zeros_like(old))
    np.testing.assert_array_equal(m, actual)
    with pytest.raises(ValueError, match="moment"):
        moment_check(actual, old + np.float32(.1) * (grad - old))
    huge = np.array([1e31, -1e31, 1e-30], np.float32)
    clipped, report = clipping({"theta": huge, "readout_weight": np.ones(3, np.float32), "readout_bias": np.zeros(3, np.float32)}, "joint")
    assert np.isfinite(report["theta"]["norm_before"]) and np.linalg.norm(clipped["theta"].astype(float)) <= 1 + 1e-6


@pytest.mark.parametrize("mode", ("joint", "frozen_brain"))
def test_separate_process_resume_and_scene_boundary_equal_continuous(mode, tmp_path):
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    continuous.mkdir()
    split.mkdir()
    model = policy()
    optimizer = make_training_optimizer(model, mode)
    run = numeric_run(model, mode)
    initial_theta = model.brain.theta.detach().clone()
    resets = []
    original = model.initial_state
    def counted():
        resets.append(1)
        return original()
    model.initial_state = counted
    state, chain, segment = numeric_segment(continuous, run, model, optimizer, None, 4)
    assert len(resets) == 2 and chain["completed_windows"] == chain["optimizer_steps"] == 4
    assert [r["cursor"]["next_frame"] for r in chain["windows"]] == [2, 4, 2, 4]
    write_json(split / "numeric_run.json", run)
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve()) + os.pathsep + str(Path("tests").resolve())}
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", "from test_training_runner import numeric_worker; import sys; numeric_worker(sys.argv[1])", str(split)],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    split_chain = evidence.scan_run(split, run, numeric_data(), production=False)
    second_model = policy()
    second_optimizer = make_training_optimizer(second_model, mode)
    second_state, _ = checkpoint.restore_training_checkpoint(second_model, second_optimizer, split_chain["last"]["path"], expected=contract.expected_ids(run))
    second_state, split_chain, _ = numeric_segment(split, run, second_model, second_optimizer, second_state, 2)
    assert_same_state(state, second_state)
    for key, value in chain["last"]["parameters"].items():
        np.testing.assert_array_equal(value, split_chain["last"]["parameters"][key])
    for key, value in chain["last"]["moments"].items():
        np.testing.assert_array_equal(value, split_chain["last"]["moments"][key])
    for index in range(4):
        left = continuous / "segments/000001/windows" / f"{index+1:06d}" / "evidence.npz"
        s = 1 if index == 0 else 2 if index == 1 else 3
        right = split / "segments" / f"{s:06d}" / "windows" / f"{index+1:06d}" / "evidence.npz"
        a, b = contract.read_arrays(left), contract.read_arrays(right)
        assert set(a) == set(b)
        assert all(np.array_equal(a[k], b[k]) for k in a)
    if mode == "frozen_brain":
        assert torch.equal(initial_theta, model.brain.theta)
    assert any(torch.count_nonzero(value) for value in state.brain.pending)
    assert np.count_nonzero(state.drive.phase)


def test_invalid_window_advances_state_without_any_update(tmp_path):
    model = policy()
    optimizer = make_training_optimizer(model, "joint")
    _, selected = contract.next_selection(numeric_data(), 0)
    state, first, _ = compute_window(model, optimizer, model.initial_state(), selected, mode="joint", before_steps=0)
    selected["valid"][:] = False
    state, skipped, report = compute_window(model, optimizer, state, selected, mode="joint", before_steps=1)
    assert state.frame_index == 4 and report["numeric"]["updated"] is False
    for key in (*contract.PARAMETERS, "gains", "rho"):
        np.testing.assert_array_equal(skipped[f"after_{key}"], first[f"after_{key}"])
    for key in skipped:
        if key.startswith("adam_after."):
            np.testing.assert_array_equal(skipped[key], first[key])
    fresh = policy()
    with pytest.raises(ValueError, match="zero-step"):
        compute_window(fresh, make_training_optimizer(fresh, "joint"), fresh.initial_state(), selected,
                       mode="joint", before_steps=0, failure_directory=tmp_path)
    assert (tmp_path / "failed_window.npz").exists()


@pytest.fixture
def sealed(tmp_path):
    model = policy()
    optimizer = make_training_optimizer(model, "joint")
    run = numeric_run(model, "joint")
    _, chain, segment = numeric_segment(tmp_path, run, model, optimizer, None, 1)
    return tmp_path, run, numeric_data(), chain, segment


def reseal_json(path, key, mutate):
    value = contract.read_json(path)
    mutate(value)
    if key:
        value[key] = canonical_hash({k: v for k, v in value.items() if k != key})
    write_json(path, value)


@pytest.mark.parametrize("damage", ["parent", "prefix_external", "prefix_duplicate", "event_missing", "checkpoint_state", "dataset",
                                    "window_gap", "window_duplicate", "resume", "checkpoint_missing", "moment", "cursor"])
def test_bad_sealed_chain_never_falls_back(sealed, damage):
    root, run, data, chain, segment = sealed
    window = segment / "windows/000001"
    if damage == "parent":
        reseal_json(window / "window.json", "window_id", lambda value: value.update(parent_checkpoint_id="a" * 64))
    elif damage.startswith("prefix"):
        p = window / "prefix.jsonl"
        p.write_bytes(p.read_bytes() * 2 if damage.endswith("duplicate") else b'{"event":"../../external.json"}\n')
        reseal_json(window / "window.json", "window_id", lambda value: value["files_sha256"].update({"prefix.jsonl": contract.sha256(p)}))
    elif damage == "event_missing":
        (window / "event.json").unlink()
    elif damage == "checkpoint_missing":
        (window / "checkpoint/checkpoint.json").unlink()
    elif damage == "dataset":
        data.sequences[0].teacher_target[0, 0] += .01
    elif damage == "window_gap":
        window.rename(segment / "windows/000002")
    elif damage == "window_duplicate":
        (segment / "windows/000000").mkdir()
    elif damage == "resume":
        model = policy()
        optimizer = make_training_optimizer(model, "joint")
        state, _ = checkpoint.restore_training_checkpoint(model, optimizer, chain["last"]["path"], expected=contract.expected_ids(run))
        _, _, second = numeric_segment(root, run, model, optimizer, state, 1)
        reseal_json(second / "resume.json", None, lambda value: value.update(checkpoint_id="b" * 64))
    else:
        archive = window / "checkpoint/training.npz"
        arrays = contract.read_arrays(archive)
        key = "brain.v" if damage == "checkpoint_state" else "brain.theta.exp_avg"
        arrays[key].flat[0] += 1
        np.savez(archive, **arrays)
        if damage == "cursor":
            reseal_json(window / "checkpoint/checkpoint.json", "checkpoint_id", lambda value: value["context"].update(next_frame=4))
    with pytest.raises((ValueError, FileNotFoundError)):
        evidence.scan_run(root, run, data, production=False)


def test_failed_segment_keeps_verified_prefix_and_ignores_unsealed_tail(sealed, monkeypatch):
    root, run, data, chain, segment = sealed
    registration = contract.read_json(segment / "registration.json")
    # A two-window registration that was interrupted after sealing the first.
    registration["windows"] = 2
    registration["segment_id"] = canonical_hash({k: v for k, v in registration.items() if k != "segment_id"})
    # Rebuilding a fresh run is clearer than mutating identity-bound event files.
    fresh = root / "interrupted"
    fresh.mkdir()
    model = policy()
    optimizer = make_training_optimizer(model, "joint")
    empty = evidence.scan_run(fresh, run, data, production=False)
    child, reg = train_snn.register_segment(fresh, run, empty, 2)
    real_save = checkpoint.save_training_checkpoint
    calls = []
    import malecns_lab.learning.training_worker as worker
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("simulated storage interruption")
        return real_save(*args, **kwargs)
    monkeypatch.setattr(worker, "save_training_checkpoint", interrupted)
    with pytest.raises(OSError):
        execute_windows(child, run, data, reg, model, optimizer, None, None, b"", production=False)
    sample_resources(child, reg)
    recovered = evidence.scan_run(fresh, run, data, production=False)
    assert recovered["completed_windows"] == 1 and recovered["segments"][0]["unsealed_tail"]
    assert (child / "windows/000002/event.json").exists() and not (child / "windows/000002/window.json").exists()
    # Resource failure remains failed even with a valid boundary; fake exit 0 cannot supply missing windows.
    outcome = {"status": "failed", "reason": "rss_limit", "returncode": -15}
    result = train_snn.finalize_segment(fresh, child, run, data, reg, outcome, production=False)
    assert result["status"] == "failed" and result["verified_end_windows"] == 1 and result["recoverable_prefix"]
    result = train_snn.finalize_segment(fresh, child, run, data, reg, {"status": "completed", "reason": "worker_exited", "returncode": 0}, production=False)
    assert result["status"] == "failed" and result["reason"] == "invalid_segment_evidence"


def test_sample_boundary_rejects_uncovered_or_out_of_budget(sealed):
    root, run, data, chain, segment = sealed
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n0,1,1\n")
    with pytest.raises(ValueError, match="budget"):
        evidence.scan_run(root, run, data, production=False)
    (segment / "resources.csv").write_text(f"wall_seconds,tree_rss_bytes,system_available_bytes\n0,1,{2*GIB}\n")
    with pytest.raises(ValueError, match="resource sample"):
        evidence.scan_run(root, run, data, production=False)


def test_r2_first_window_mismatch_is_not_relaxed(tmp_path):
    model = policy()
    optimizer = make_training_optimizer(model, "joint")
    _, selection = contract.next_selection(numeric_data(), 0)
    _, values, _ = compute_window(model, optimizer, model.initial_state(), selection, mode="joint", before_steps=0)
    maps = {"initial_parameters.npz": {key: values[f"before_{key}"] for key in (*contract.PARAMETERS, "gains", "rho")},
            "baseline.npz": {key: values[key] for key in ("actions", "features_mv")},
            "gradients_before_clip.npz": {key: values[f"pre_clip_{key}"] for key in contract.PARAMETERS},
            "update.npz": {f"after_{key}": values[f"after_{key}"] for key in (*contract.PARAMETERS, "gains", "rho")},
            "selection.npz": selection}
    for name, arrays in maps.items():
        np.savez(tmp_path / name, **arrays)
    assert compare_r2(values, tmp_path)["passed"]
    values["actions"][0, 0] += .01
    assert not compare_r2(values, tmp_path)["passed"]


def test_parent_import_and_cli_validation_are_torch_free():
    script = "import sys; from malecns_lab.learning import train_snn; assert 'torch' not in sys.modules; assert 'malecns_lab.learning.training_checkpoint' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for args in (["--run", "/does-not-exist", "--out", "/tmp/unwanted", "--windows", "1"],
                 ["--run", "/does-not-exist", "--windows", "0"], ["--out", "/tmp/unwanted", "--windows", "1"]):
        with pytest.raises(SystemExit) as error:
            train_snn.main(args)
        assert error.value.code == 2


def test_window_budget_and_workspace_lock(sealed):
    root, run, data, chain, _ = sealed
    with pytest.raises(ValueError):
        train_snn.register_segment(root, run, chain, 4)
    with train_snn.workspace_lock():
        with pytest.raises(ValueError, match="lock"):
            with train_snn.workspace_lock():
                pass


def test_clock_origin_uncertainty_is_conservative(sealed):
    root, run, data, chain, segment = sealed
    registration = contract.read_json(segment / "registration.json")
    seal = contract.read_json(segment / "windows/000001/window.json")
    elapsed = seal["sealed_monotonic"] - registration["clock_monotonic"]
    # CSV origin may have started later. A last sample just short of the
    # conservative elapsed bound cannot establish a post-seal observation.
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n"
        f"0,1,{2*GIB}\n{elapsed - .000001},1,{2*GIB}\n")
    with pytest.raises(ValueError, match="clock offset unresolved"):
        evidence.sampled_boundary(segment, registration, seal["sealed_monotonic"])
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n"
        f"0,1,{2*GIB}\n{elapsed + .000001},1,{2*GIB}\n")
    assert evidence.sampled_boundary(segment, registration, seal["sealed_monotonic"]) == 2


def test_numerical_failure_is_not_labeled_resumable(sealed):
    root, run, data, chain, segment = sealed
    registration = contract.read_json(segment / "registration.json")
    write_json(segment / "worker_result.json", {"status": "failed", "failure_kind": "numerical_or_protocol", "reason": "nonfinite"})
    result = train_snn.finalize_segment(root, segment, run, data, registration,
                                       {"status": "failed", "reason": "worker_exited", "returncode": 1}, production=False)
    assert result["status"] == "failed" and result["verified_end_windows"] == 1
    assert result["recoverable_prefix"] is None and result["continuation_blocked_reason"] == "numerical_or_protocol"
    with pytest.raises(ValueError, match="numerical/protocol"):
        evidence.scan_run(root, run, data, production=False)


@pytest.mark.parametrize("damage", ("late_state", "late_moment", "nested_identity"))
def test_rehashed_checkpoint_still_requires_actual_semantics(sealed, damage):
    root, run, data, chain, segment = sealed
    window = segment / "windows/000001"
    path = window / "checkpoint"
    if damage == "nested_identity":
        reseal_json(path / "policy/checkpoint.json", "checkpoint_id", lambda value: value["run_identity"].update(dataset_id="b"*64))
    else:
        values = contract.read_arrays(path / "training.npz")
        key = "brain.v" if damage == "late_state" else "brain.theta.exp_avg"
        values[key].flat[0] += 1
        np.savez(path / "training.npz", **values)
    meta = contract.read_json(path / "checkpoint.json")
    meta["files_sha256"] = {key: contract.sha256(path / key) for key in meta["files_sha256"]}
    meta["policy_checkpoint_id"] = contract.read_json(path / "policy/checkpoint.json")["checkpoint_id"]
    meta["checkpoint_id"] = canonical_hash({k: v for k, v in meta.items() if k != "checkpoint_id"})
    write_json(path / "checkpoint.json", meta)
    seal = contract.read_json(window / "window.json")
    seal["files_sha256"]["checkpoint/checkpoint.json"] = contract.sha256(path / "checkpoint.json")
    seal["checkpoint_id"] = meta["checkpoint_id"]
    seal["window_id"] = canonical_hash({k: v for k, v in seal.items() if k != "window_id"})
    write_json(window / "window.json", seal)
    with pytest.raises(ValueError, match="state/ID|actual Adam|Nested policy"):
        evidence.scan_run(root, run, data, production=False)
