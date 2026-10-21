"""Serialized supervised GPU jobs for the independently registered yaw task."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import traceback

from ...brain.metadata import canonical_hash
from ...drones.yaw_task import YawTaskEnvironment, scenes, control_gate, score_episode
from ..resource_guard import GIB, resource_budget, supervise, write_json
from ..train_snn import workspace_lock
from .protocol import MODES, read_json, require, sha256, validate_run
from .archive import (read_checkpoint, scan_checkpoints, write_checkpoint, restore_checkpoint,
                      write_arrays, read_arrays)


def _seal(path, payload, artifacts):
    payload = {**payload, "artifacts": {name: sha256(path / name) for name in artifacts}}
    payload["evidence_id"] = canonical_hash(payload)
    write_json(path / "evidence.json", payload)
    return payload


def read_evidence(path):
    path = Path(path)
    value = read_json(path / "evidence.json")
    payload = dict(value)
    require(payload.pop("evidence_id", None) == canonical_hash(payload), "Evidence seal changed")
    for name, digest in value["artifacts"].items():
        require(Path(name).name == name and sha256(path / name) == digest, "Evidence artifact changed")
    return value


def _gate_verified(directory, run):
    result = read_json(Path(directory) / "gate/result.json")
    require(result["status"] == "completed" and result["run_id"] == run["run_id"], "Disposable gate has not completed")
    evidence = read_evidence(Path(directory) / "gate")
    require(evidence["kind"] == "gate" and evidence["run_id"] == run["run_id"] and evidence["passed"], "Gate evidence failed")
    return evidence


def _job(directory, run, kind, *, windows=0, extra=None):
    import psutil
    directory = Path(directory)
    if kind == "gate":
        path = directory / "gate"
    elif kind == "train":
        jobs = sorted((directory / "jobs").glob("[0-9]*"))
        for previous in jobs:
            if (previous / "worker_result.json").exists():
                worker = read_json(previous / "worker_result.json")
                require(worker.get("failure_kind") != "numerical_or_protocol", "Previous numerical/protocol failure blocks continuation")
        path = directory / "jobs" / f"{len(jobs)+1:06d}"
    else:
        path = directory / "evaluation" / extra["split"] / extra["mode"] / extra["scene_name"]
    require(not path.exists(), "Job already exists; do not silently repeat failed or completed work")
    path.mkdir(parents=True)
    budget = resource_budget(psutil.virtual_memory().available, rss_bytes=6*GIB, reserve_bytes=GIB)
    registration = {"kind": kind, "run_id": run["run_id"], "directory": str(directory.resolve()),
                    "windows": windows, "parent_pid": os.getpid(), "resources": budget,
                    "started_monotonic": time.monotonic(), **(extra or {})}
    registration["job_id"] = canonical_hash(registration)
    write_json(path / "registration.json", registration)
    if not budget["can_launch"]:
        outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None}
    else:
        # Disable unsupported-op fallback before the child imports torch.
        previous = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        try:
            outcome = supervise([sys.executable, "-m", "malecns_lab.learning.yaw", "worker", "--job", str(path)], path,
                                rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                wall_seconds=600, poll_seconds=.1)
        finally:
            if previous is None:
                os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            else:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = previous
    write_json(path / "supervisor_result.json", outcome)
    result = {"status": "failed", "run_id": run["run_id"], "kind": kind, "reason": outcome["reason"], "job": str(path)}
    try:
        validate_run(directory)
        require(outcome["status"] == "completed" and outcome["returncode"] == 0, "Supervised child did not complete")
        worker = read_json(path / "worker_result.json")
        require(worker["status"] == "completed" and worker["job_id"] == registration["job_id"], "Worker did not complete registered job")
        evidence = read_evidence(path)
        require(evidence["job_id"] == registration["job_id"] and evidence["run_id"] == run["run_id"], "Job evidence identity differs")
        if kind == "train":
            chain = scan_checkpoints(directory, run)
            require(chain[-1]["metadata"]["completed_windows"] == evidence["end_window"] and
                    evidence["end_window"] - evidence["start_window"] == windows, "Incomplete window segment")
            result["completed_windows"] = evidence["end_window"]
            write_json(directory / "progress.json", {"run_id": run["run_id"], "completed_windows": evidence["end_window"],
                       "optimizer_steps": evidence["end_window"], "epoch_complete": evidence["end_window"] == 80,
                       "checkpoint": chain[-1]["path"], "scientific_success": False})
        result.update(status="completed", reason="verified_registered_job", evidence_id=evidence["evidence_id"])
    except Exception as error:
        result["detail"] = f"{type(error).__name__}: {error}"
    write_json(path / "result.json", result)
    return result


def run_gate(directory):
    with workspace_lock():
        run = validate_run(directory)
        require(not (Path(directory) / "checkpoints").exists(), "Gate must precede formal training")
        return _job(Path(directory), run, "gate")


def train_segment(directory, windows):
    require(type(windows) is int and 1 <= windows <= 4, "A segment must contain 1 to 4 windows")
    with workspace_lock():
        run = validate_run(directory)
        _gate_verified(directory, run)
        chain = scan_checkpoints(directory, run)
        start = chain[-1]["metadata"]["completed_windows"] if chain else 0
        require(start + windows <= 80, "Requested windows exceed registered epoch")
        require(not (Path(directory) / "test_lock.json").exists(), "Test-opened model is immutable")
        return _job(Path(directory), run, "train", windows=windows, extra={"start_window": start})


def _final_checkpoint(directory, run):
    chain = scan_checkpoints(directory, run)
    require(bool(chain) and chain[-1]["metadata"]["completed_windows"] == 80, "Both branches must finish 80 windows before evaluation")
    return chain[-1]


def _evaluation_records(directory, split, selection):
    import numpy as np
    summaries = []
    for scene in scenes(split):
        for mode in MODES:
            path = Path(directory) / "evaluation" / split / mode / scene.name
            result = read_json(path / "result.json")
            evidence = read_evidence(path)
            require(result["status"] == "completed" and result["evidence_id"] == evidence["evidence_id"], "Evaluation job is incomplete")
            require(evidence["selection"] == selection and evidence["mode"] == mode and evidence["scene"] == scene.__dict__,
                    "Evaluation selection or scene differs")
            arrays = read_arrays(path / "episode.npz")
            expected = score_episode(scene, arrays["actual_wz_rad_s"], arrays["command_wz_rad_s"])
            require(set(expected) == set(evidence["metrics"]), "Evaluation metric fields differ")
            for key, value in expected.items():
                saved = evidence["metrics"][key]
                require(saved is None if value is None else saved is not None and np.isclose(saved, value, rtol=1e-10, atol=1e-12),
                        "Evaluation metrics differ from saved trajectory")
            require(np.array_equal(arrays["frame_index"], np.arange(scene.total_frames)), "Incomplete evaluation frame sequence")
            if mode == "zero":
                require(np.count_nonzero(arrays["command_wz_rad_s"]) == 0, "Zero control was not zero")
            summaries.append(evidence)
    return summaries


def evaluate(directory, frozen_directory, split):
    directory, frozen_directory = Path(directory).resolve(), Path(frozen_directory).resolve()
    require(split in ("validation", "test"), "Evaluation split must be validation or test")
    with workspace_lock():
        run, frozen = validate_run(directory), validate_run(frozen_directory)
        require(run["mode"] == "joint" and frozen["mode"] == "frozen" and run["family_id"] == frozen["family_id"],
                "Evaluation requires matched joint/frozen families")
        joint_ckpt, frozen_ckpt = _final_checkpoint(directory, run), _final_checkpoint(frozen_directory, frozen)
        selection = {"family_id": run["family_id"], "joint_checkpoint_id": joint_ckpt["metadata"]["checkpoint_id"],
                     "frozen_checkpoint_id": frozen_ckpt["metadata"]["checkpoint_id"], "frozen_directory": str(frozen_directory)}
        if split == "test":
            validation = read_json(directory / "evaluation/validation/summary.json")
            require(validation["selection"] == selection and validation["status"] == "completed" and validation["control_gate"]["passed"],
                    "Validation must pass before opening locked test")
            validated = _evaluation_records(directory, "validation", selection)
            require(control_gate(validated, require_static=False)["passed"], "Saved validation trajectories do not pass")
            lock = directory / "test_lock.json"
            if lock.exists():
                require(read_json(lock)["selection"] == selection, "Test checkpoint selection changed")
            else:
                write_json(lock, {"selection": selection, "opened_unix_s": time.time(), "protocol_id": canonical_hash(run["protocol"])})
        summaries = []
        for scene in scenes(split):
            for mode in MODES:
                path = directory / "evaluation" / split / mode / scene.name
                if not path.exists():
                    result = _job(directory, run, "evaluate", extra={"split": split, "scene_name": scene.name, "mode": mode,
                        "selection": selection, "joint_checkpoint": joint_ckpt["path"], "frozen_checkpoint": frozen_ckpt["path"]})
                    require(result["status"] == "completed", f"Evaluation failed: {result}")
                result = read_json(path / "result.json")
                require(result["status"] == "completed", "An existing failed evaluation cannot be silently repeated")
                evidence = read_evidence(path)
                require(evidence["selection"] == selection and evidence["mode"] == mode and evidence["scene"]["name"] == scene.name,
                        "Evaluation selection or scene differs")
                summaries.append(evidence)
        summaries = _evaluation_records(directory, split, selection)
        gate = control_gate(summaries, require_static=split == "test")
        result = {"status": "completed", "selection": selection, "split": split, "control_gate": gate,
                  "rows": summaries, "simulation_ready": False,
                  "interpretation": "Control evidence only; package/interface/web acceptance is separate"}
        write_json(directory / "evaluation" / split / "summary.json", result)
        return result


def run_worker(job):
    job = Path(job).resolve()
    registration = read_json(job / "registration.json")
    expected = dict(registration)
    require(expected.pop("job_id", None) == canonical_hash(expected), "Job registration changed")
    parent_pid = registration["parent_pid"]
    require(os.getppid() == parent_pid, "Worker must remain attached to its supervisor")
    def watch_parent():
        while True:
            if os.getppid() != parent_pid:
                os._exit(3)
            time.sleep(.2)
    threading.Thread(target=watch_parent, daemon=True).start()
    # A direct worker invocation must not bypass shared workspace serialization.
    import fcntl
    from .protocol import ROOT
    with (ROOT / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must own workspace lock")
    with (job / "worker.claim").open("x") as claim:
        claim.write(str(os.getpid()))
    result = {"status": "failed", "job_id": registration["job_id"], "reason": "unfinished"}
    started, sampler = time.monotonic(), None
    def progress(**details):
        write_json(job / "phase.json", {"wall_seconds": time.monotonic()-started, **details})
    try:
        progress(phase="validate")
        directory = Path(registration["directory"])
        run = validate_run(directory)
        require(run["run_id"] == registration["run_id"], "Worker run differs")
        mode, kind = registration.get("mode", run["mode"]), registration["kind"]
        neural = kind != "evaluate" or mode in ("joint", "frozen", "untrained")
        policy = optimizer = None
        if neural:
            import torch
            from ..mps_memory import MPSMemorySampler
            from .policy import YawVisualPolicy
            from .optimizer import YawAdam
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            sampler = MPSMemorySampler(job / "mps_memory.csv", registration["resources"]["effective_rss_limit_bytes"])
            progress(phase="load_full_gpu_policy")
            policy = YawVisualPolicy.from_reference(run["graph"]["path"])
            graph = policy.identity["graph"]
            require(graph["role"] == "full_reference" and graph["nodes"] == 166700 and graph["edges"] == 25582938,
                    "Complete registered graph required")
        from .rollout import warmup, compute_window, pack_rows, evaluate_episode
        if kind in ("gate", "train"):
            optimizer = YawAdam(policy, mode=run["mode"])
            all_scenes = scenes("train")
            if kind == "gate":
                environment = YawTaskEnvironment(all_scenes[0])
                state, warm_rows = warmup(policy, policy.initial_state(), environment, progress=progress)
                write_arrays(job / "warmup.npz", pack_rows(warm_rows))
                state, arrays, report = compute_window(policy, optimizer, state, environment, progress=progress)
                write_arrays(job / "window.npz", arrays)
                write_arrays(job / "state.npz", policy.state_arrays(state))
                changed = report["optimizer"]["changed"]
                require(changed["readout.weight"] + changed["readout.bias"] > 0, "No readout update in disposable gate")
                if run["mode"] == "joint":
                    require(changed["brain.theta"] > 0 and report["optimizer"]["raw_nonzero"]["brain.theta"] > 0,
                            "No internal neural update in disposable gate")
                _seal(job, {"run_id": run["run_id"], "job_id": registration["job_id"], "kind": "gate", "passed": True,
                           "policy": policy.identity, "report": report, "environment": environment.snapshot()},
                      ("warmup.npz", "window.npz", "state.npz"))
            else:
                _gate_verified(directory, run)
                chain = scan_checkpoints(directory, run)
                start = chain[-1]["metadata"]["completed_windows"] if chain else 0
                require(start == registration["start_window"], "Registered training cursor changed")
                if chain:
                    previous = chain[-1]
                    environment = YawTaskEnvironment(all_scenes[previous["metadata"]["scene_index"]])
                    state, optimizer = restore_checkpoint(previous, policy, YawAdam, environment)
                    write_arrays(job / "restored_state.npz", policy.state_arrays(state))
                    write_json(job / "restore.json", {"checkpoint_id": previous["metadata"]["checkpoint_id"],
                               "state_sha256": sha256(job / "restored_state.npz"), "exact_state_optimizer_parameters": True})
                else:
                    environment = YawTaskEnvironment(all_scenes[0])
                    state = policy.initial_state()
                    write_checkpoint(directory / "checkpoints/000000", run=run, policy=policy, optimizer=optimizer,
                                     state=state, environment=environment, completed_windows=0, scene_index=0, parent_id=None)
                    previous = read_checkpoint(directory / "checkpoints/000000", run)
                events = []
                for number in range(start + 1, start + registration["windows"] + 1):
                    scene_index = (number - 1) // 20
                    if environment.scene != all_scenes[scene_index]:
                        environment = YawTaskEnvironment(all_scenes[scene_index])
                        state = policy.initial_state()
                    if environment.frame_index < 40:
                        state, warm_rows = warmup(policy, state, environment, progress=progress)
                        warm_path = directory / "warmup" / f"{scene_index:02d}.npz"
                        warm_path.parent.mkdir(exist_ok=True)
                        require(not warm_path.exists(), "Scene warmup cannot be overwritten")
                        write_arrays(warm_path, pack_rows(warm_rows))
                    require(environment.frame_index == 40 + ((number-1) % 20)*2, "Scene training cursor differs")
                    state, arrays, report = compute_window(policy, optimizer, state, environment,
                        theta_raw_norm_limit=run["protocol"]["theta_raw_norm_limit"], progress=progress)
                    path = directory / "windows" / f"{number:06d}"
                    path.mkdir(parents=True, exist_ok=False)
                    write_arrays(path / "frames.npz", arrays)
                    checkpoint = write_checkpoint(directory / "checkpoints" / f"{number:06d}", run=run, policy=policy,
                        optimizer=optimizer, state=state, environment=environment, completed_windows=number,
                        scene_index=scene_index, parent_id=previous["metadata"]["checkpoint_id"])
                    event = _seal(path, {"run_id": run["run_id"], "job_id": registration["job_id"], "kind": "window",
                        "window": number, "scene_index": scene_index, "checkpoint_id": checkpoint["checkpoint_id"],
                        "parent_checkpoint_id": previous["metadata"]["checkpoint_id"], "report": report}, ("frames.npz",))
                    events.append(event)
                    previous = {"metadata": checkpoint}
                    progress(phase="window_sealed", completed_windows=number)
                write_json(job / "windows.json", {"events": events})
                _seal(job, {"run_id": run["run_id"], "job_id": registration["job_id"], "kind": "train",
                           "start_window": start, "end_window": number, "checkpoint_id": checkpoint["checkpoint_id"]}, ("windows.json",))
        else:
            require(kind == "evaluate", "Unknown worker operation")
            selected = [s for s in scenes(registration["split"]) if s.name == registration["scene_name"]]
            require(len(selected) == 1, "Unknown registered scene")
            if mode in ("joint", "frozen"):
                source_directory = directory if mode == "joint" else Path(registration["selection"]["frozen_directory"])
                source_run = validate_run(source_directory)
                checkpoint = read_checkpoint(registration[f"{mode}_checkpoint"], source_run)
                require(checkpoint["metadata"]["checkpoint_id"] == registration["selection"][f"{mode}_checkpoint_id"], "Selected checkpoint changed")
                require(checkpoint["metadata"]["policy"] == policy.identity, "Evaluation model identity differs")
                policy.load_parameters(checkpoint["parameter"])
            summary, arrays = evaluate_episode(selected[0], mode, policy, progress=progress)
            write_arrays(job / "episode.npz", arrays)
            _seal(job, {"run_id": run["run_id"], "job_id": registration["job_id"], "kind": "evaluate",
                       "selection": registration["selection"], **summary}, ("episode.npz",))
        result.update(status="completed", reason="registered_job_sealed")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        result["failure_kind"] = ("resource_or_interrupt" if isinstance(error, (MemoryError, KeyboardInterrupt, OSError))
                                  or "out of memory" in str(error).lower() else "numerical_or_protocol")
        traceback.print_exc()
    finally:
        if sampler is not None:
            result["mps_memory"] = sampler.close()
            if result["mps_memory"]["error"] is not None:
                result.update(status="failed", failure_kind="numerical_or_protocol", reason="MPS memory sampler failed")
        result["wall_seconds"] = time.monotonic() - started
        write_json(job / "worker_result.json", result)
    return result
