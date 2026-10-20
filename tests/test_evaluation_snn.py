"""P3d persistence and closed-loop checks on explicit tiny numerical circuits.

No production graph, training, hardware, Docker or service operations occur.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from malecns_lab.brain.metadata import canonical_hash
from malecns_lab.learning import evaluation_contract as contract
from malecns_lab.learning import evaluation_checkpoint as checkpoint
from malecns_lab.learning import evaluation_evidence as evidence
from malecns_lab.learning import evaluate_snn as cli
from malecns_lab.learning.evaluation_rollout import environment_for, environment_summary, metrics, readout_actions
from malecns_lab.learning.evaluation_worker import apply_effective_parameters, execute_segment, parameters
from malecns_lab.learning.resource_guard import GIB, write_json
from malecns_lab.learning.training_contract import bytes_sha, read_arrays, read_json, sha256
from test_training_checkpoint import policy
from test_neural_policy import assert_same_state


@pytest.fixture(scope="module", autouse=True)
def threads():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def numeric_suite(root, model=None):
    model = model or policy()
    values = parameters(model)
    run = {"schema": "numerical_test_only", "protocol": contract.PROTOCOL,
           "protocol_id": canonical_hash(contract.PROTOCOL), "policy": model.identity,
           "training": {name: {"fixture": name} for name in ("joint", "frozen_brain")},
           "implementation_sha256": contract.sources(), "runtime": contract.runtime(),
           "parameters": {name: contract.parameter_identity(model.identity, values) for name in contract.NEURAL}}
    run["evaluation_run_id"] = canonical_hash(run)
    root.mkdir(parents=True)
    write_json(root / "run.json", run)
    for name in contract.CONDITIONS:
        path = root / "conditions" / name
        path.mkdir(parents=True)
        if name in contract.NEURAL:
            np.savez(path / "parameters.npz", **values)
        condition = contract.condition_payload(run, name, sha256(path / "parameters.npz") if name in contract.NEURAL else None)
        write_json(path / "condition.json", condition)
    return run, model


def numeric_episode(root, run, name="joint", scene=contract.VAL[0], override=None):
    condition, params = contract.load_condition(root, run, name)
    episode = contract.episode_payload(run, condition, scene)
    if override:
        episode["scene"].update(override)
        episode["scene_spec_id"] = canonical_hash(episode["scene"])
        episode["episode_id"] = canonical_hash({k: v for k, v in episode.items() if k != "episode_id"})
    path = root / "episodes" / name / scene
    if not path.exists():
        path.mkdir(parents=True)
        write_json(path / "episode.json", episode)
    return path, episode, condition, params


def numeric_segment(root, run, frames, *, name="joint", model=None, override=None, outcome=None, failure_kind=None):
    directory, episode, condition, params = numeric_episode(root, run, name=name, override=override)
    chain = evidence.scan_episode(directory, run, episode, params, production=False)
    segment, reg = cli.register_segment(directory, run, episode, chain, frames)
    state = None
    if name in contract.NEURAL:
        model = model or policy()
        apply_effective_parameters(model, condition, params)
        if chain["last"]:
            archive = checkpoint.inspect_checkpoint(chain["last"]["path"], run, episode, chain["last"]["context"], keep_arrays=True)
            state = checkpoint.restore_state(archive, model)
    state, end, terminal = execute_segment(segment, run, episode, condition, reg, chain, model if name in contract.NEURAL else None, state)
    write_json(segment / "worker_result.json", {"schema": "cnsbridge.evaluation_worker.v1", "status": "completed" if not failure_kind else "failed",
        "segment_id": reg["segment_id"], "processed_frames": end-reg["start_frame"], "terminal": terminal,
        "parameters_unchanged": True, **({"failure_kind": failure_kind} if failure_kind else {})})
    write_json(segment / "phase.json", {"phase": "finished"})
    (segment / "worker.log").write_text("numerical_test_only\n")
    elapsed = time.monotonic() - reg["clock_monotonic"] + .1
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n"
        f"0,1000000,{3 * GIB}\n{elapsed},1000000,{3 * GIB}\n")
    outcome = outcome or {"status": "completed", "reason": "worker_exited", "returncode": 0, "wall_seconds": elapsed,
                          "peak_observed_rss_bytes": 1000000}
    result = cli.finalize_segment(root, directory, segment, run, episode, params, reg, outcome, production=False)
    return state, result, segment


def numeric_worker(root, frames):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run = read_json(Path(root) / "run.json")
    _, result, _ = numeric_segment(Path(root), run, frames)
    if result["status"] != "completed":
        raise ValueError(result)


def test_parent_imports_no_torch():
    command = "import sys; import malecns_lab.learning.evaluate_snn; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True, timeout=20)


def test_two_real_processes_resume_odd_frame_matches_continuous(tmp_path):
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    run, model = numeric_suite(continuous)
    split_run, _ = numeric_suite(split)
    expected, result, _ = numeric_segment(continuous, run, 4, model=model)
    assert result["status"] == "completed", result
    script = "import sys;sys.path.insert(0,'tests');from test_evaluation_snn import numeric_worker;numeric_worker(sys.argv[1],int(sys.argv[2]))"
    for frames in (1, 3):
        subprocess.run([sys.executable, "-c", script, str(split), str(frames)], check=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": "src"})
    directory, episode, _, params = numeric_episode(split, split_run)
    chain = evidence.scan_episode(directory, split_run, episode, params, production=False)
    assert chain["next_frame"] == 4 and len(chain["segments"]) == 2
    archive = checkpoint.inspect_checkpoint(chain["last"]["path"], split_run, episode, chain["last"]["context"], keep_arrays=True)
    restored = checkpoint.restore_state(archive, policy())
    assert_same_state(expected, restored)
    assert chain["last"]["state"]["frame_index"] == 4
    reference = evidence.scan_episode(continuous / "episodes/joint" / contract.VAL[0], run,
                                      numeric_episode(continuous, run)[1], params, production=False)
    assert chain["records"] == reference["records"]
    assert environment_summary(chain["environment"], 4) == environment_summary(reference["environment"], 4)
    assert chain["metrics"] == reference["metrics"]
    resumed = read_json(directory / "segments/000002/resume.json")
    assert resumed["next_frame"] == 1 and resumed["state_digest"]["frame_index"] == 1
    assert chain["last"]["state"]["brain_step"] == model.precondition_steps + 4 * model.frame_steps


@pytest.mark.parametrize("frame", (0, 1, 80))
def test_checkpoint_arbitrary_frame_and_terminal(tmp_path, frame):
    run, model = numeric_suite(tmp_path / "suite")
    _, episode, _, _ = numeric_episode(tmp_path / "suite", run)
    env = environment_for(run, episode)
    state = model.initial_state()
    from malecns_lab.control.body import BodyVelocity
    with torch.no_grad():
        for _ in range(frame):
            state = model.forward_tensor(env.frame(), state).state
            env.step(BodyVelocity())
    context = checkpoint.context_for(run, episode, frame, environment_summary(env, frame), None, bytes_sha(b""))
    path = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(path, run, episode, context, state=state, actual_parameters=parameters(model))
    archive = checkpoint.inspect_checkpoint(path, run, episode, context, keep_arrays=True)
    assert_same_state(state, checkpoint.restore_state(archive, model))
    assert context["terminal"] == (frame == 80)
    assert set(archive["arrays"]) == checkpoint.STATE_KEYS


@pytest.mark.parametrize("mutation", ("phase", "events", "refractory", "nan", "shape", "bool_frame", "bool_step", "missing"))
def test_invalid_state_arrays_and_strict_clocks(mutation):
    model = policy()
    info, values = checkpoint.state_payload(model.initial_state())
    if mutation == "phase": values["drive.phase"][0] = 1
    if mutation == "events": values["drive.event_counts"][0] = -1
    if mutation == "refractory": values["brain.refractory_until"][0] = -1
    if mutation == "nan": values["brain.v"][0] = np.nan
    if mutation == "shape": values["brain.pending"] = values["brain.pending"][:-1]
    if mutation == "bool_frame": info["frame_index"] = False
    if mutation == "bool_step": info["brain"]["step"] = True
    if mutation == "missing": values.pop("drive.phase")
    with pytest.raises(ValueError):
        checkpoint.validate_arrays(info, values, model.identity, 0)


def test_same_policy_wrong_parameters_and_acute_before_initialization(tmp_path):
    run, model = numeric_suite(tmp_path / "suite")
    directory, episode, condition, initial = numeric_episode(tmp_path / "suite", run)
    _, result, _ = numeric_segment(tmp_path / "suite", run, 1, model=model)
    assert result["status"] == "completed", result
    chain = evidence.scan_episode(directory, run, episode, initial, production=False)
    archive = checkpoint.inspect_checkpoint(chain["last"]["path"], run, episode, chain["last"]["context"], keep_arrays=True)
    other = policy()
    with torch.no_grad(): other.brain.theta[0] += .2
    assert other.identity == model.identity
    with pytest.raises(ValueError, match="parameters"):
        checkpoint.restore_state(archive, other)
    changed = parameters(other)
    changed["theta"][:] = .17  # Actual supplied original array, not an assumption that originals are zero.
    acute = {"name": "acute_restore", "parameter_identity": contract.parameter_identity(other.identity, changed)}
    resets = []
    old = other.initial_state
    def observe():
        resets.append(parameters(other)["theta"].copy())
        return old()
    other.initial_state = observe
    apply_effective_parameters(other, acute, changed)
    assert not resets
    other.initial_state()
    np.testing.assert_array_equal(resets[0], changed["theta"])


def test_pixel_only_forward_action_changes_observation_and_lost_target_advances(tmp_path):
    root = tmp_path / "suite"
    model = policy()
    with torch.no_grad():
        model.readout.bias.fill_(.1)  # Make supervisor suppression observable even with no visible marker.
    run, model = numeric_suite(root, model)
    calls = []
    original = model.forward_tensor
    def record(*args, **kwargs):
        assert len(args) == 2 and not kwargs and isinstance(args[0], np.ndarray) and not args[0].flags.writeable
        calls.append(args[1].frame_index)
        return original(*args)
    model.forward_tensor = record
    state, result, _ = numeric_segment(root, run, 3, model=model, override={"occlusion_s": [0., .3], "occlusion_fraction": 1.})
    assert result["status"] == "completed", result
    assert calls == [0, 1, 2] and state.frame_index == 3
    assert result["episode"]["metrics"]["intervention_samples"] == 3
    path = root / "episodes/joint" / contract.VAL[0] / "segments/000001/chunks/000001/evidence.npz"
    values = read_arrays(path)
    assert np.all(values["executed_actions"] == 0) and np.any(values["raw_actions"] != 0)
    from malecns_lab.control.body import BodyVelocity
    _, episode, _, _ = numeric_episode(root, run, override={"occlusion_s": [0., .3], "occlusion_fraction": 1.})
    episode["scene"]["occlusion_s"] = None
    a, b = environment_for(run, episode), environment_for(run, episode)
    for _ in range(20):
        a.step(BodyVelocity(vx=.3, wz=.3))
        b.step(BodyVelocity())
    assert not np.array_equal(a.frame(), b.frame())


@pytest.mark.parametrize("name", ("direct_visual", "zero"))
def test_baseline_parent_replay_no_fake_neural_arrays(tmp_path, name):
    run, _ = numeric_suite(tmp_path / "suite")
    _, result, segment = numeric_segment(tmp_path / "suite", run, 10, name=name)
    assert result["status"] == "completed", result
    values = read_arrays(segment / "chunks/000001/evidence.npz")
    assert "features_mv" not in values and read_arrays(segment / "resume.npz") == {}
    metadata = read_json(segment / "chunks/000001/checkpoint/checkpoint.json")
    assert metadata["state"] is None and metadata["parameter_identity"] is None
    assert result["episode"]["metrics"]["samples"] == 10
    assert result["episode"]["metrics"]["observed_fraction"] == 10 / 80
    assert not result["episode"]["metrics"]["success"]


def test_readout_and_environment_reject_resealed_wrong_values(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    _, result, segment = numeric_segment(root, run, 1)
    assert result["status"] == "completed", result
    path = segment / "chunks/000001"
    values = read_arrays(path / "evidence.npz")
    records = evidence.jsonl(path / "trajectory.jsonl")
    directory, episode, _, params = numeric_episode(root, run)
    from malecns_lab.learning.evaluation_rollout import replay_chunk
    for key in ("raw_actions", "frames", "executed_actions"):
        changed = {k: v.copy() for k, v in values.items()}
        changed[key].flat[0] += 1
        with pytest.raises(ValueError):
            replay_chunk(environment_for(run, episode), changed, records, run, episode, 0, params)
    changed_records = deepcopy(records)
    changed_records[0]["qualified"] = not changed_records[0]["qualified"]
    with pytest.raises(ValueError):
        replay_chunk(environment_for(run, episode), values, changed_records, run, episode, 0, params)
    seal = read_json(path / "chunk.json")
    seal["parent_checkpoint_id"] = "a" * 64
    seal["chunk_id"] = canonical_hash({k: v for k, v in seal.items() if k != "chunk_id"})
    write_json(path / "chunk.json", seal)
    with pytest.raises(ValueError, match="parent"):
        evidence.scan_episode(directory, run, episode, params, production=False)


def test_resource_failed_prefix_and_numerical_failure_not_success(tmp_path):
    root = tmp_path / "resource"
    run, _ = numeric_suite(root)
    outcome = {"status": "failed", "reason": "wall_limit", "returncode": -15, "wall_seconds": 600.1}
    _, result, _ = numeric_segment(root, run, 1, outcome=outcome)
    assert result["status"] == "failed" and result["supervisor"] == outcome and result["recoverable_prefix"]
    _, resumed, _ = numeric_segment(root, run, 1)
    assert resumed["status"] == "completed" and resumed["verified_end_frame"] == 2
    bad = tmp_path / "numeric_failure"
    br, _ = numeric_suite(bad)
    _, result, _ = numeric_segment(bad, br, 1, outcome=outcome, failure_kind="numerical_or_protocol")
    assert result["status"] == "failed" and result["recoverable_prefix"] is None
    with pytest.raises(ValueError, match="failure"):
        numeric_segment(bad, br, 1)
    summary = evidence.scan_suite(bad, br, production=False)
    assert not summary["evidence_valid"] and summary["episodes"][0]["status"] == "numerical_or_protocol_failure"
    assert summary["episodes"][0]["metrics"] is None


def test_all_validation_order_and_summary_seal(tmp_path):
    empty = {"episodes": [], "val_complete": False}
    evidence.check_order(tmp_path, empty, "joint", contract.VAL[0])
    with pytest.raises(ValueError, match="order"):
        evidence.check_order(tmp_path, empty, "joint", contract.TEST[0])
    episodes = [{"condition": name, "scene_id": scene, "terminal": True, "evidence_complete": True}
                for name in contract.CONDITIONS for scene in contract.VAL]
    summary = {"evaluation_run_id": "a" * 64, "episodes": episodes, "val_complete": True, "comparisons": {"val": {}}}
    with pytest.raises(ValueError, match="sealed validation"):
        evidence.check_order(tmp_path, summary, "joint", contract.TEST[0])
    evidence.seal_validation(tmp_path, summary)
    evidence.check_order(tmp_path, summary, "joint", contract.TEST[0])
    summary["episodes"][0]["terminal"] = False
    summary["episodes"][0]["evidence_complete"] = False
    with pytest.raises(ValueError):
        evidence.check_order(tmp_path, summary, "joint", contract.TEST[0])


def test_production_admission_has_no_fixture_or_partial_fallback(tmp_path, monkeypatch):
    run, _ = numeric_suite(tmp_path / "suite")
    with pytest.raises(ValueError):
        contract.validate_suite(tmp_path / "suite")
    with pytest.raises((ValueError, KeyError)):
        contract.final_training_source(tmp_path / "suite", "joint")
    with pytest.raises(ValueError):
        contract.path_inside(tmp_path, "../outside")
    with pytest.raises(ValueError):
        contract.path_inside(tmp_path, "/tmp")


def test_parameter_mutation_fails_before_seal(tmp_path):
    run, model = numeric_suite(tmp_path / "suite")
    original = model.forward_tensor
    def mutated(frame, state):
        output = original(frame, state)
        model.readout.bias[0] += .1
        return output
    model.forward_tensor = mutated
    with pytest.raises(ValueError, match="Parameters changed"):
        numeric_segment(tmp_path / "suite", run, 1, model=model)
    assert not list((tmp_path / "suite").rglob("chunk.json"))


def test_physical_failure_is_terminal_with_full_denominator(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    _, result, _ = numeric_segment(root, run, 10, name="zero",
        override={"family": "static", "initial_offset_m": [-1.4, 0., 0.], "amplitude_m": [0., 0., 0.]})
    assert result["status"] == "completed", result
    episode_result = result["episode"]
    assert episode_result["terminal"] and episode_result["next_frame"] == 1
    assert episode_result["metrics"]["failure"] and not episode_result["metrics"]["success"]
    assert episode_result["metrics"]["observed_fraction"] == 1 / 80
    assert episode_result["metrics"]["planned_samples"] == 80


def test_empty_failure_has_no_success_and_fake_completion_rejected(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    directory, episode, _, params = numeric_episode(root, run)
    chain = evidence.scan_episode(directory, run, episode, params, production=False)
    segment, reg = cli.register_segment(directory, run, episode, chain, 1)
    outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None}
    result = cli.finalize_segment(root, directory, segment, run, episode, params, reg, outcome, production=False)
    assert result["status"] == "failed" and result["verified_end_frame"] == 0
    assert result["episode"]["metrics"] is None and result["recoverable_prefix"] is None
    write_json(segment / "worker_result.json", {"schema": "cnsbridge.evaluation_worker.v1", "status": "completed",
        "segment_id": reg["segment_id"], "processed_frames": 1, "terminal": False, "parameters_unchanged": True})
    outcome = {"status": "completed", "reason": "worker_exited", "returncode": 0}
    result = cli.finalize_segment(root, directory, segment, run, episode, params, reg, outcome, production=False)
    assert result["status"] == "failed" and result["reason"] == "invalid_segment_evidence"


def test_explicit_segment_budget_and_tampered_resume(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    directory, episode, _, params = numeric_episode(root, run)
    chain = evidence.scan_episode(directory, run, episode, params, production=False)
    for frames in (True, 0, 41):
        with pytest.raises(ValueError):
            cli.register_segment(directory, run, episode, chain, frames)
    _, result, first = numeric_segment(root, run, 1)
    assert result["status"] == "completed", result
    reg = read_json(first / "registration.json")
    assert reg["resources"]["reserve_bytes"] == GIB and reg["resources"]["requested_rss_limit_bytes"] == 6 * GIB
    _, result, second = numeric_segment(root, run, 1)
    assert result["status"] == "completed", result
    report = read_json(second / "resume.json")
    arrays = read_arrays(second / "resume.npz")
    arrays["drive.phase"][0] = .5 if arrays["drive.phase"][0] != .5 else .25
    np.savez(second / "resume.npz", **arrays)
    report["arrays_sha256"] = sha256(second / "resume.npz")
    write_json(second / "resume.json", report)
    with pytest.raises(ValueError, match="state"):
        evidence.scan_episode(directory, run, episode, params, production=False)


def finalized_training_fixture(root, monkeypatch):
    """A previously validated 160-window chain stub; terminal admission is real.

    No numerical training is fabricated as production evidence: only the scanner
    and final array reader are isolated, so these tests target their caller's
    additional parent-terminal requirements.
    """
    root.mkdir()
    initial = root / "initial"
    initial.mkdir()
    values = parameters(policy())
    np.savez(initial / "initial_parameters.npz", **values)
    scenes = [{"scene_id": "train-sine-" + axis, "frame_count": 80} for axis in ("combined", "forward", "vertical", "yaw")]
    run = {"mode": "joint", "training_run_id": "a" * 64, "protocol_id": "b" * 64,
           "dataset_id": "c" * 64, "dataset_manifest_sha256": "d" * 64,
           "scenes": scenes, "prior_run": {"path": str(initial)}}
    write_json(root / "run.json", run)
    chain = {"completed_windows": 160, "optimizer_steps": 160, "segments": [], "windows": []}
    for number in range(1, 41):
        segment = root / "segments" / f"{number:06d}"
        segment.mkdir(parents=True)
        reg = {"training_run_id": run["training_run_id"], "segment_id": f"segment-{number}", "windows": 4,
               "parent_checkpoint_id": None if number == 1 else f"checkpoint-{(number-1)*4}"}
        write_json(segment / "registration.json", reg)
        for window in range((number - 1) * 4 + 1, number * 4 + 1):
            chain["windows"].append({"segment": number, "window": window, "checkpoint_id": f"checkpoint-{window}",
                                     "cursor": contract.training.cursor(scenes, window, window)})
        last = chain["windows"][-1]
        prefix = {"checkpoint_id": last["checkpoint_id"], "cursor": last["cursor"],
                  "path": str(segment / "windows" / f"{number*4:06d}" / "checkpoint")}
        for name in ("phase.json", "resources.csv", "worker.log"):
            (segment / name).write_text("terminal fixture\n")
        worker_files = ["phase.json"]
        if number > 1:
            for name in ("resume.json", "resume.npz"):
                (segment / name).write_text("validated scanner prefix fixture\n")
            worker_files += ["resume.json", "resume.npz"]
        write_json(segment / "worker_result.json", {"schema": "cnsbridge.training_worker.v1", "status": "completed",
            "segment_id": reg["segment_id"], "training_run_id": run["training_run_id"], "completed_windows": 4,
            "raw_counts_sha256_before": contract.training.PILOT["raw_ordered_counts_sha256"],
            "raw_counts_sha256_after": contract.training.PILOT["raw_ordered_counts_sha256"],
            "artifacts_sha256": {name: sha256(segment / name) for name in worker_files}})
        supervisor = {"schema": "cnsbridge.resource_run.v1", "status": "completed", "reason": "worker_exited", "returncode": 0}
        write_json(segment / "supervisor_result.json", supervisor)
        write_json(segment / "result.json", {"schema": "cnsbridge.training_segment_result.v1", "status": "completed",
            "reason": "registered_windows_verified", "training_run_id": run["training_run_id"], "segment_id": reg["segment_id"],
            "scope": "full_reference_training", "verified_end_windows": number * 4, "recoverable_prefix": prefix,
            "supervisor": supervisor, "terminal_artifacts_sha256": {name: sha256(segment / name)
                for name in contract.TRAINING_TERMINAL_FILES if (segment / name).exists()}})
        chain["segments"].append({"segment": number, "start_windows": (number - 1) * 4,
                                  "end_windows": number * 4, "unsealed_tail": False})
    final = Path(prefix["path"])
    (final / "policy").mkdir(parents=True)
    for name in ("checkpoint.json", "training.npz", "policy/parameters.npz"):
        (final / name).write_text("validated checkpoint fixture\n")
    write_json(final / "policy/checkpoint.json", {"checkpoint_id": "parameter-checkpoint"})
    chain["last"] = {"context": last["cursor"], "path": str(final), "window_id": "last-window"}
    monkeypatch.setattr(contract.training, "validate_run", lambda value: object())
    monkeypatch.setattr(contract, "scan_run", lambda *args: chain)
    monkeypatch.setattr(contract, "inspect_checkpoint", lambda *args: {"checkpoint_id": "checkpoint-160", "parameters": values})
    return chain


def fail_training_terminal(segment, kind):
    result = read_json(segment / "result.json")
    if kind == "io":
        worker = read_json(segment / "worker_result.json")
        worker.update(status="failed", failure_kind="io_or_interrupt", reason="OSError:fixture")
        write_json(segment / "worker_result.json", worker)
    supervisor = {"schema": "cnsbridge.resource_run.v1", "status": "failed",
                  "reason": "worker_exited" if kind == "io" else "rss_limit", "returncode": 1 if kind == "io" else -15}
    write_json(segment / "supervisor_result.json", supervisor)
    result.update(status="failed", reason=supervisor["reason"], supervisor=supervisor,
        terminal_artifacts_sha256={name: sha256(segment / name) for name in contract.TRAINING_TERMINAL_FILES if (segment / name).exists()})
    write_json(segment / "result.json", result)


@pytest.mark.parametrize("history", ("completed", "resource", "io"))
def test_final_training_admission_binds_all_parent_terminals(tmp_path, monkeypatch, history):
    root = tmp_path / "training"
    finalized_training_fixture(root, monkeypatch)
    if history != "completed":
        fail_training_terminal(root / "segments/000020", history)
    source, _, _, _ = contract.final_training_source(root, "joint")
    assert len(source["finalized_segments"]) == 40
    assert source["finalized_segments"][-1]["status"] == "completed"
    for item in source["finalized_segments"]:
        segment = root / "segments" / f"{item['segment']:06d}"
        assert item["result_sha256"] == sha256(segment / "result.json")
        assert set(item["terminal_artifacts_sha256"]) == {name for name in contract.TRAINING_TERMINAL_FILES if (segment / name).exists()}


@pytest.mark.parametrize("defect", ("missing_parent", "invalid_evidence", "missing_hash", "wrong_parent_identity",
                                   "wrong_worker_identity", "numerical_failure", "supervisor_changed", "final_failed", "wrong_prefix"))
def test_final_training_admission_rejects_unfinalized_or_invalid_terminal(tmp_path, monkeypatch, defect):
    root = tmp_path / "training"
    finalized_training_fixture(root, monkeypatch)
    segment = root / "segments/000040"
    result = read_json(segment / "result.json")
    if defect == "missing_parent":
        (segment / "result.json").unlink()
    elif defect == "final_failed":
        fail_training_terminal(segment, "resource")
    else:
        if defect == "invalid_evidence": result.update(status="failed", reason="invalid_segment_evidence")
        if defect == "missing_hash": result["terminal_artifacts_sha256"].pop("worker.log")
        if defect == "wrong_parent_identity": result["segment_id"] = "wrong"
        if defect == "wrong_prefix": result["recoverable_prefix"]["checkpoint_id"] = "wrong"
        if defect == "supervisor_changed": result["supervisor"]["reason"] = "rss_limit"
        if defect in ("wrong_worker_identity", "numerical_failure"):
            worker = read_json(segment / "worker_result.json")
            if defect == "wrong_worker_identity":
                worker["training_run_id"] = "wrong"
            else:
                worker.update(status="failed", failure_kind="numerical_or_protocol")
            write_json(segment / "worker_result.json", worker)
            result["terminal_artifacts_sha256"]["worker_result.json"] = sha256(segment / "worker_result.json")
        write_json(segment / "result.json", result)
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.final_training_source(root, "joint")


@pytest.mark.parametrize("ending", ("duration", "physical"))
@pytest.mark.parametrize("outcome_kind", ("completed", "resource", "io"))
def test_environment_terminal_requires_completed_parent_evidence(tmp_path, ending, outcome_kind):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    override = ({"family": "static", "initial_offset_m": [-1.4, 0., 0.], "amplitude_m": [0., 0., 0.]}
                if ending == "physical" else None)
    if ending == "duration":
        numeric_segment(root, run, 40, name="zero")
    outcome = None if outcome_kind == "completed" else {"status": "failed", "reason": "rss_limit" if outcome_kind == "resource" else "worker_exited",
                                                        "returncode": -15 if outcome_kind == "resource" else 1}
    _, result, segment = numeric_segment(root, run, 40, name="zero", override=override, outcome=outcome,
                                         failure_kind="io_or_interrupt" if outcome_kind == "io" else None)
    original_result = (segment / "result.json").read_bytes()
    directory, episode, _, params = numeric_episode(root, run, name="zero", override=override)
    chain = evidence.scan_episode(directory, run, episode, params, production=False)
    report = evidence.episode_report(chain, run, episode)
    assert report["terminal"] and chain["terminal"]
    assert report["evidence_complete"] == chain["evidence_complete"] == (outcome_kind == "completed")
    assert result["episode"] == report
    assert report["next_frame"] == (1 if ending == "physical" else 80)
    if outcome_kind != "completed":
        assert report["status"] in ("resource_failure", "io_or_worker_failure")
        assert report["metrics"]["success"] is False and report["diagnostic_metrics"] == chain["metrics"]
        present = {(name, scene): {**report, "condition": name, "scene_id": scene}
                   for name in contract.CONDITIONS for scene in contract.VAL}
        compared = evidence.comparisons(present)["val"]
        assert all(not entry["complete"] and entry["means"] is None for entry in compared["equal_scene_weight"].values())
        assert all(row["joint_minus_other"] is None for rows in compared["paired"].values() for row in rows)
        summary = {"episodes": list(present.values()), "val_complete": False}
        for name, scene in (("joint", contract.VAL[0]), ("joint", contract.VAL[1]), ("joint", contract.TEST[0])):
            with pytest.raises(ValueError):
                evidence.check_order(root, summary, name, scene)
        with pytest.raises(ValueError):
            evidence.seal_validation(root, summary)
        with pytest.raises(ValueError, match="scene end"):
            cli.register_segment(directory, run, episode, chain, 1)
        assert (segment / "result.json").read_bytes() == original_result
    else:
        assert report["status"] == ("physical_failure" if ending == "physical" else "duration_complete")


def test_terminal_resource_failure_stops_suite_order_and_formal_success(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    numeric_segment(root, run, 40)
    _, result, _ = numeric_segment(root, run, 40, outcome={"status": "failed", "reason": "wall_limit", "returncode": -15})
    assert result["episode"]["terminal"] and not result["episode"]["evidence_complete"]
    summary = evidence.scan_suite(root, run, production=False)
    assert not summary["val_complete"] and not summary["test_complete"]
    assert summary["episodes"][0]["status"] == "resource_failure"
    assert not summary["comparisons"]["val"]["equal_scene_weight"]["joint"]["complete"]
    directory, episode, _, params = numeric_episode(root, run)
    chain = evidence.scan_episode(directory, run, episode, params, production=False)
    chain["metrics"]["success"] = True  # Even a hypothetically qualifying trajectory is not formal success.
    report = evidence.episode_report(chain, run, episode)
    assert report["diagnostic_metrics"]["success"] and not report["metrics"]["success"]
    numeric_episode(root, run, scene=contract.VAL[1])
    with pytest.raises(ValueError, match="order"):
        evidence.scan_suite(root, run, production=False)


def test_nonterminal_io_prefix_explicit_continuation_can_finish(tmp_path):
    root = tmp_path / "suite"
    run, _ = numeric_suite(root)
    _, failed, segment = numeric_segment(root, run, 40, name="zero",
        outcome={"status": "failed", "reason": "worker_exited", "returncode": 1}, failure_kind="io_or_interrupt")
    original = (segment / "result.json").read_bytes()
    assert failed["status"] == "failed" and not failed["episode"]["terminal"] and not failed["episode"]["evidence_complete"]
    _, completed, _ = numeric_segment(root, run, 40, name="zero")
    assert completed["status"] == "completed" and completed["episode"]["evidence_complete"]
    assert completed["episode"]["segments"][0]["status"] == "failed"
    assert (segment / "result.json").read_bytes() == original
