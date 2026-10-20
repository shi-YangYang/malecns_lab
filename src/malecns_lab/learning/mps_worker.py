"""Claimed child entry. Torch/full-graph imports occur only after parent checks."""
import fcntl
import os
from pathlib import Path
import time
import traceback


def run_worker(segment):
    from .resource_guard import write_json
    from .training_contract import read_json, require
    segment = Path(segment).resolve()
    reg = read_json(segment / "registration.json")
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
    with (segment / "worker.claim").open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    started = time.monotonic()
    result = {"schema": "cnsbridge.mps_worker.v1", "status": "failed", "training_run_id": reg["training_run_id"],
              "segment_id": reg["segment_id"], "completed_windows": 0}
    sampler = None
    durations = {}

    def phase(name, **details):
        write_json(segment / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - started, **details})
        if sampler is not None:
            sampler.sample(name)

    try:
        phase("imports")
        import numpy as np
        import torch
        from ..brain.metadata import canonical_hash
        from .training_contract import (PARAMETERS, bytes_sha, cursor, expected_ids, next_selection, prefix_line, sha256)
        from .training_checkpoint import read_training_checkpoint, _state_metadata
        from .joint_probe_worker import raw_counts_digest
        from .mps_contract import validate_run
        from .mps_evidence import scan_run
        from .mps_archive import verify_resume, write_checkpoint
        from .mps_memory import MPSMemorySampler
        from .mps_policy import MPSPolicy
        from .mps_state import restore, boundary_arrays
        from .mps_window import compute_window, clock

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from ..brain.mps_trainable import require_mps
        require_mps()
        phase("validate_registration")
        directory = segment.parents[1]
        run = read_json(directory / "run.json")
        data, initial = validate_run(run)
        chain = scan_run(directory, run, data, initial, current_segment=reg["segment"])
        require(chain["completed_windows"] == reg["start_cursor"]["completed_windows"], "Worker prefix changed")
        previous = chain["last"]
        old = read_json(Path(run["cpu_parent"]["path"]) / "run.json")
        if not chain["windows"]:
            # Invoke the original strict reader, without constructing a second full CPU brain.
            archive = read_training_checkpoint(previous["path"], expected=expected_ids(old))
            require(archive.context == previous["context"] and archive.parameter_metadata["policy"] == old["policy"], "CPU archive differs")
            require(all(np.array_equal(archive.parameters[k], v) for k, v in previous["parameters"].items())
                    and all(np.array_equal(archive.optimizer_arrays[k], v) for k, v in previous["moments"].items()), "CPU archive values differ")
            del archive
        sampler = MPSMemorySampler(segment / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(segment / "mps_budget.json", sampler.budget)
        phase("load_full_policy")
        then = clock()
        policy = MPSPolicy.from_reference(run["graph"]["path"], old["policy"])
        require(policy.identity == run["policy"], "Actual policy does not match registered identity")
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == policy.brain.graph.identity["raw_ordered_counts_sha256"], "Actual graph digest differs")
        durations["construction_seconds"] = clock() - then
        phase("restore_all_state")
        then = clock()
        bridge, state, restored = restore(policy, previous)
        np.savez(segment / "resume.npz", **restored)
        write_json(segment / "resume.json", {"schema": "cnsbridge.mps_restore.v1", "segment_id": reg["segment_id"],
            "parent_checkpoint_id": previous["checkpoint_id"], "source_state": previous["metadata"]["state"],
            "restored_state": _state_metadata(state), "arrays_sha256": sha256(segment / "resume.npz"), "exact": True})
        verify_resume(segment, reg, previous, run)
        durations["restore_and_verify_seconds"] = clock() - then
        del restored
        completed, steps, parent, prefix = chain["completed_windows"], chain["optimizer_steps"], previous["checkpoint_id"], chain["prefix"]
        (segment / "windows").mkdir(exist_ok=False)
        for _ in range(reg["windows"]):
            number = completed + 1
            window = segment / "windows" / f"{number:06d}"
            window.mkdir(exist_ok=False)
            scene, selected = next_selection(data, completed)
            if selected["indices"][0] == 0:
                phase("scene_precondition", scene_index=scene, window=number)
                state = policy.initial_state()
            require(state.frame_index == selected["indices"][0], "State and selected frame differ")
            state, values, record = compute_window(policy, bridge, state, selected, phase=phase, failure_directory=window)
            phase("save_boundary", window=number)
            then = clock()
            np.savez(window / "evidence.npz", **values)
            after = cursor(run["scenes"], number, steps + 1)
            event = {"schema": "cnsbridge.mps_event.v1", "training_run_id": run["training_run_id"], "segment_id": reg["segment_id"],
                     "window": number, "scene_index": scene, "scene_id": data.sequences[scene].scene["scene_id"],
                     "before_cursor": cursor(run["scenes"], completed, steps), "after_cursor": after,
                     "parent_checkpoint_id": parent, "evidence_sha256": sha256(window / "evidence.npz"), **record}
            write_json(window / "event.json", event)
            prefix += prefix_line(reg["segment"], number, sha256(window / "event.json"))
            (window / "prefix.jsonl").write_bytes(prefix)
            context = {**expected_ids(run), "scenes": run["scenes"], **after,
                       "parent_checkpoint_id": parent, "log_prefix_sha256": bytes_sha(prefix)}
            metadata = write_checkpoint(window / "checkpoint", run, context, _state_metadata(state), boundary_arrays(policy, bridge, state))
            seal = {"schema": "cnsbridge.mps_window.v1", "training_run_id": run["training_run_id"], "segment_id": reg["segment_id"],
                    "window": number, "parent_checkpoint_id": parent, "checkpoint_id": metadata["checkpoint_id"], "cursor": after,
                    "files_sha256": {name: sha256(window / name) for name in ("evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")},
                    "sealed_monotonic": time.monotonic()}
            seal["window_id"] = canonical_hash(seal)
            write_json(window / "window.json", seal)
            durations[f"window_{number}_save_seconds"] = clock() - then
            completed, steps, parent = number, steps + 1, metadata["checkpoint_id"]
            result["completed_windows"] += 1
            phase("window_sealed", completed_windows=completed)
        phase("verify_raw_graph")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"], "Graph mutated during training")
        result.update(status="completed", reason="registered_mps_windows_sealed")
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
        write_json(segment / "timing.json", durations)
        write_json(segment / "phase.json", {"phase": "finished", "wall_seconds": durations["worker_seconds"]})
        result["wall_seconds"] = durations["worker_seconds"]
        write_json(segment / "worker_result.json", result)
    return result
