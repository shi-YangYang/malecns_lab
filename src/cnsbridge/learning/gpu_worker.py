"""Claimed all-GPU worker; heavy imports follow PID and shared-lock checks."""
import fcntl
import os
from pathlib import Path
import time
import traceback


def run_worker(job):
    from .resource_guard import write_json
    from .training_contract import read_json, require
    job = Path(job).resolve()
    reg = read_json(job / "registration.json")
    require(os.getppid() == reg["parent_pid"], "Worker must belong to its supervising parent")
    lock_path = Path(__file__).resolve().parents[3] / "data/.continuous-training.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must own the shared numerical lock")
    with (job / "worker.claim").open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    started = time.monotonic()
    result = {"schema": "cnsbridge.gpu_recovery_worker.v1", "status": "failed", "kind": reg["kind"],
              "training_run_id": reg["training_run_id"], "job_id": reg["job_id"], "completed_windows": 0}
    sampler, durations = None, {}
    def phase(name, **details):
        write_json(job / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - started, **details})
        if sampler is not None:
            sampler.sample(name)
    try:
        phase("imports")
        import numpy as np
        import torch
        from ..brain.metadata import canonical_hash
        from ..brain.mps_trainable import require_mps
        from .joint_probe_worker import raw_counts_digest
        from .mps_memory import MPSMemorySampler
        from .training_contract import bytes_sha, cursor, expected_ids, next_selection, prefix_line, sha256
        from .gpu_contract import validate_run, initial_prefix
        from .gpu_evidence import read_genesis, read_gate, scan_run, validate_registration
        from .gpu_archive import write_checkpoint
        from .gpu_policy import GPUPolicy
        from .gpu_optimizer import GPUAdam
        from .gpu_state import restore, boundary_arrays, state_metadata, state_arrays
        from .gpu_window import compute_window, clock
        from .gpu_numbers import validate_correction
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require_mps()
        directory = job.parents[1] if reg["kind"] == "train" else job.parent
        run = read_json(directory / "run.json")
        phase("validate_registration")
        data, parent, cpu = validate_run(run)
        if reg["kind"] == "initialize":
            previous = parent
            validate_registration(reg, run, "initialize", parent)
        else:
            genesis = read_genesis(directory, run, parent)
            if reg["kind"] == "gate":
                previous = genesis
                validate_registration(reg, run, "gate", previous, prefix=initial_prefix(run, genesis))
            else:
                read_gate(directory, run, data, genesis)
                chain = scan_run(directory, run, data, genesis, current_segment=reg["segment"])
                previous = chain["last"]
                validate_registration(reg, run, "train", previous, segment=reg["segment"],
                                      completed=chain["completed_windows"], prefix=chain["prefix"])
        sampler = MPSMemorySampler(job / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(job / "mps_budget.json", sampler.budget)
        phase("load_full_gpu_policy")
        began = clock()
        policy = GPUPolicy.from_reference(run["graph"]["path"], cpu, credit_horizon=run["protocol"]["credit_horizon_steps"])
        require(policy.identity == run["policy"], "Actual GPU policy identity differs")
        durations["construction_seconds"] = clock() - began
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == run["policy"]["model"]["graph"]["raw_ordered_counts_sha256"], "Full graph differs")
        if reg["kind"] == "initialize":
            phase("inherit_parameters_reset_optimizer")
            policy.restore_parameters(parent["parameters"])
            optimizer = GPUAdam(policy.named_parameters())
            phase("new_scene_precondition")
            state = policy.initial_state()
            require(optimizer.identity == run["optimizer"], "Actual GPU optimizer differs")
            context = {**expected_ids(run), "scenes": run["scenes"], **cursor(run["scenes"], 0, 0),
                       "parent_checkpoint_id": parent["checkpoint_id"], "log_prefix_sha256": bytes_sha(b"")}
            write_checkpoint(job / "checkpoint", run, context, state_metadata(state, policy.drive_config.dt_ms),
                             boundary_arrays(policy, optimizer, state))
        else:
            phase("restore_all_gpu_state")
            optimizer, state, restored = restore(policy, previous)
            require(optimizer.identity == run["optimizer"], "Actual GPU optimizer differs")
            np.savez(job / "resume.npz", **restored)
            write_json(job / "resume.json", {"job_id": reg["job_id"], "parent_checkpoint_id": previous["checkpoint_id"],
                       "state": state_metadata(state, policy.drive_config.dt_ms), "arrays_sha256": sha256(job / "resume.npz")})
            del restored
            if reg["kind"] == "gate":
                _, selection = next_selection(data, 159)
                state, values, record = compute_window(policy, optimizer, state, selection,
                    theta_norm_limit=run["protocol"]["theta_norm_limit"], phase=phase, failure_directory=job)
                np.savez(job / "evidence.npz", **values)
                np.savez(job / "end_state.npz", **state_arrays(state, detached=True))
                correction = validate_correction(values, record["numeric"])
                write_json(job / "gate.json", {"job_id": reg["job_id"], "evidence_sha256": sha256(job / "evidence.npz"),
                    "end_arrays_sha256": sha256(job / "end_state.npz"), "end_metadata": state_metadata(state, policy.drive_config.dt_ms),
                    **record, "correction": correction})
                seal = {"job_id": reg["job_id"], "files_sha256": {name: sha256(job / name) for name in ("gate.json", "evidence.npz", "end_state.npz")}}
                write_json(job / "seal.json", {**seal, "gate_id": canonical_hash(seal)})
            else:
                completed, prefix = chain["completed_windows"], chain["prefix"]
                (job / "windows").mkdir(exist_ok=False)
                for _ in range(reg["windows"]):
                    number = completed + 1
                    window = job / "windows" / f"{number:06d}"
                    window.mkdir(exist_ok=False)
                    scene, selection = next_selection(data, completed)
                    if selection["indices"][0] == 0 and completed != 0:
                        phase("scene_precondition", scene_index=scene)
                        state = policy.initial_state()
                    require(state.frame_index == selection["indices"][0], "State and selected frame differ")
                    state, values, record = compute_window(policy, optimizer, state, selection,
                        theta_norm_limit=run["protocol"]["theta_norm_limit"], phase=phase, failure_directory=window)
                    phase("save_window", window=number)
                    np.savez(window / "evidence.npz", **values)
                    event = {"training_run_id": run["training_run_id"], "job_id": reg["job_id"], "window": number,
                             "scene_index": scene, "scene_id": data.sequences[scene].scene["scene_id"],
                             "parent_checkpoint_id": previous["checkpoint_id"], "evidence_sha256": sha256(window / "evidence.npz"), **record}
                    write_json(window / "event.json", event)
                    prefix += prefix_line(reg["segment"], number, sha256(window / "event.json"))
                    (window / "prefix.jsonl").write_bytes(prefix)
                    context = {**expected_ids(run), "scenes": run["scenes"], **cursor(run["scenes"], number, number),
                               "parent_checkpoint_id": previous["checkpoint_id"], "log_prefix_sha256": bytes_sha(prefix)}
                    metadata = write_checkpoint(window / "checkpoint", run, context, state_metadata(state, policy.drive_config.dt_ms),
                                                boundary_arrays(policy, optimizer, state))
                    seal = {"job_id": reg["job_id"], "window": number, "parent_checkpoint_id": previous["checkpoint_id"],
                            "checkpoint_id": metadata["checkpoint_id"], "files_sha256": {name: sha256(window / name)
                            for name in ("evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")},
                            "sealed_monotonic": time.monotonic()}
                    write_json(window / "window.json", {**seal, "window_id": canonical_hash(seal)})
                    previous = {"checkpoint_id": metadata["checkpoint_id"]}
                    completed = number
                    result["completed_windows"] += 1
                    phase("window_sealed", completed_windows=completed)
        result.update(actual_policy=policy.identity, actual_optimizer=optimizer.identity)
        phase("verify_complete_graph")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"], "Graph changed during job")
        result.update(status="completed", reason="registered_all_gpu_job_sealed", sealed_monotonic=time.monotonic())
        phase("job_sealed")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        resource = isinstance(error, MemoryError) or "out of memory" in str(error).lower()
        result["failure_kind"] = "resource_or_interrupt" if resource or isinstance(error, (OSError, KeyboardInterrupt)) else "numerical_or_protocol"
        traceback.print_exc()
    finally:
        if sampler is not None:
            result["mps_memory"] = sampler.close()
            if result["mps_memory"]["error"] is not None:
                result.update(status="failed", failure_kind="numerical_or_protocol", reason="mps_memory_sampling_failed")
        durations["worker_seconds"] = time.monotonic() - started
        write_json(job / "timing.json", durations)
        write_json(job / "phase.json", {"phase": "finished", "wall_seconds": durations["worker_seconds"]})
        result["wall_seconds"] = durations["worker_seconds"]
        write_json(job / "worker_result.json", result)
    return result
