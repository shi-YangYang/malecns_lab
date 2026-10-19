"""Independent saved-array and chain checks; no Torch or model allocation."""
import csv
from pathlib import Path
import re
import numpy as np
from .resource_guard import GIB, resource_budget
from .training_contract import (PARAMETERS, bound, bytes_sha, cursor, expected_ids, next_selection, prefix_line,
                               read_arrays, read_json, require, sha256)
from .training_archive import validate_state_digest
from .training_evidence import sampled_boundary
from .training_math import validate_window_numbers
from .mps_contract import initial_prefix
from .mps_archive import inspect_checkpoint, verify_resume

TERMINAL_FILES = ("registration.json", "worker.claim", "worker_result.json", "resources.csv", "worker.log", "phase.json",
                  "supervisor_result.json", "resume.json", "resume.npz", "mps_memory.csv", "mps_budget.json", "timing.json")


def validate_registration(reg, run, number, completed, steps, previous, prefix):
    bound(reg, "segment_id")
    require(set(reg) == {"schema", "training_run_id", "segment", "start_cursor", "parent_checkpoint_id", "prefix_sha256",
                         "windows", "resources", "clock_monotonic", "parent_pid", "segment_id"}, "Segment fields differ")
    require(reg["schema"] == "cnsbridge.mps_segment.v1" and reg["training_run_id"] == run["training_run_id"]
            and reg["segment"] == number and reg["start_cursor"] == cursor(run["scenes"], completed, steps)
            and reg["parent_checkpoint_id"] == previous["checkpoint_id"] and reg["prefix_sha256"] == bytes_sha(prefix), "Segment lineage differs")
    require(type(reg["windows"]) is int and 1 <= reg["windows"] <= 4 and completed + reg["windows"] <= 160, "Window budget differs")
    resources = reg["resources"]
    budget = resource_budget(resources["available_bytes_before_launch"], reserve_bytes=GIB, rss_bytes=6 * GIB)
    require(resources == {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}
            and type(reg["parent_pid"]) is int and reg["parent_pid"] > 0
            and type(reg["clock_monotonic"]) is float and np.isfinite(reg["clock_monotonic"]), "Invalid resource/parent registration")


def memory_boundary(segment, reg, seal):
    sampled_boundary(segment, reg, seal)
    budget = read_json(segment / "mps_budget.json")
    effective = reg["resources"]["effective_rss_limit_bytes"]
    require(budget["effective_rss_budget_bytes"] == effective and 0 < budget["allocator_cap_bytes"] <= effective
            and 0 < budget["fraction"] <= 1 and budget["allocator_cap_bytes"] == budget["fraction"] * budget["recommended_max_memory_bytes"],
            "MPS allocator cap differs")
    with (segment / "mps_memory.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require(rows, "Missing MPS samples")
    times = [float(r["monotonic"]) for r in rows]
    require(all(np.isfinite(t) for t in times) and all(a <= b for a, b in zip(times, times[1:]))
            and times[-1] >= seal, "Missing post-seal MPS sample")
    for row in rows:
        require(int(row["current_allocated_bytes"]) >= 0 and int(row["driver_allocated_bytes"]) >= 0, "Invalid MPS usage")
        if float(row["monotonic"]) <= seal:
            require(int(row["current_allocated_bytes"]) <= effective, "MPS active allocation exceeded registered budget")


def validate_window(path, run, data, reg, completed, steps, previous, prefix):
    number = completed + 1
    seal, event = read_json(path / "window.json"), read_json(path / "event.json")
    bound(seal, "window_id")
    require(set(seal) == {"schema", "training_run_id", "segment_id", "window", "files_sha256", "checkpoint_id",
                          "parent_checkpoint_id", "cursor", "sealed_monotonic", "window_id"}, "Seal fields differ")
    require(seal["schema"] == "cnsbridge.mps_window.v1" and seal["training_run_id"] == run["training_run_id"]
            and seal["segment_id"] == reg["segment_id"] and seal["window"] == number
            and seal["parent_checkpoint_id"] == previous["checkpoint_id"], "Seal lineage differs")
    require(set(seal["files_sha256"]) == {"evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json"}, "Seal file inventory differs")
    for name, digest in seal["files_sha256"].items():
        require(sha256(path / name) == digest, "Sealed file changed")
    values = read_arrays(path / "evidence.npz")
    scene, selected = next_selection(data, completed)
    require(all(values[k].dtype == v.dtype and np.array_equal(values[k], v) for k, v in selected.items()), "Dataset selection differs")
    numeric = validate_window_numbers(values, run["policy"], "joint", steps)
    require(numeric["clipping"]["theta"]["nonzero_before"] > 0 and numeric["parameter_updates"]["theta"]["changed"] > 0, "Missing internal update")
    after = cursor(run["scenes"], number, steps + 1)
    expected = {"schema": "cnsbridge.mps_event.v1", "training_run_id": run["training_run_id"], "segment_id": reg["segment_id"],
                "window": number, "scene_index": scene, "scene_id": data.sequences[scene].scene["scene_id"],
                "before_cursor": cursor(run["scenes"], completed, steps), "after_cursor": after,
                "parent_checkpoint_id": previous["checkpoint_id"], "evidence_sha256": sha256(path / "evidence.npz"), "numeric": numeric}
    require(set(event) == {*expected, "start_state", "end_state", "training_losses", "durations"}
            and all(event[k] == v for k, v in expected.items()) and seal["cursor"] == after, "Event differs from independently checked window")
    require(all(type(v) is float and np.isfinite(v) and v >= 0 for v in event["durations"].values()), "Invalid durations")
    require(set(event["training_losses"]) == {"action", "regularization", "total"} and all(np.isclose(v, numeric["metrics"][k], rtol=1e-6, atol=1e-7)
            for k, v in event["training_losses"].items()), "Training loss differs")
    for key, value in previous["parameters"].items():
        require(np.array_equal(value, values[f"before_{key}"]), "Parameter continuation changed")
    for key, value in previous["moments"].items():
        require(np.array_equal(value, values[f"adam_before.{key}"]), "Adam continuation changed")
    validate_state_digest(event["start_state"], run["policy"], int(selected["indices"][0]))
    validate_state_digest(event["end_state"], run["policy"], after["next_frame"])
    if selected["indices"][0] != 0:
        require(event["start_state"] == previous["state"], "Recurrent state reset inside scene")
    prefix += prefix_line(reg["segment"], number, sha256(path / "event.json"))
    require((path / "prefix.jsonl").read_bytes() == prefix, "Prefix changed")
    saved = inspect_checkpoint(path / "checkpoint", run)
    context = {**expected_ids(run), "scenes": run["scenes"], **after,
               "parent_checkpoint_id": previous["checkpoint_id"], "log_prefix_sha256": bytes_sha(prefix)}
    require(saved["context"] == context and saved["checkpoint_id"] == seal["checkpoint_id"] and saved["state"] == event["end_state"], "Checkpoint differs")
    require(all(np.array_equal(v, values[f"after_{k}"]) for k, v in saved["parameters"].items())
            and all(np.array_equal(v, values[f"adam_after.{k}"]) for k, v in saved["moments"].items()), "Saved update differs")
    memory_boundary(path.parents[1], reg, seal["sealed_monotonic"])
    return saved, prefix, event


def scan_run(directory, run, data, initial, *, current_segment=None):
    completed, steps = initial["context"]["completed_windows"], initial["context"]["optimizer_steps"]
    prefix, previous = initial_prefix(run), initial
    segments, windows = [], []
    paths = sorted((directory / "segments").iterdir()) if (directory / "segments").exists() else []
    require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in paths)
            and [int(p.name) for p in paths] == list(range(1, len(paths) + 1)), "Invalid segment sequence")
    for segment in paths:
        number = int(segment.name)
        reg = read_json(segment / "registration.json")
        validate_registration(reg, run, number, completed, steps, previous, prefix)
        start = completed
        worker = read_json(segment / "worker_result.json") if (segment / "worker_result.json").exists() else {}
        if number != current_segment:
            require(worker.get("failure_kind") != "numerical_or_protocol", "Prior numerical/protocol failure blocks continuation")
            require((segment / "result.json").exists(), "Previous segment has no supervised terminal result")
        children = sorted((segment / "windows").iterdir()) if (segment / "windows").exists() else []
        require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in children)
                and [int(p.name) for p in children] == list(range(completed + 1, completed + 1 + len(children)))
                and len(children) <= reg["windows"], "Unexpected window sequence")
        if children and (children[0] / "window.json").exists():
            restored = verify_resume(segment, reg, previous, run)
            require(restored == previous["state"], "Actual restored state differs")
        tail = False
        for path in children:
            if not (path / "window.json").exists():
                tail = True
                continue
            require(not tail, "Seal after incomplete window")
            previous, prefix, event = validate_window(path, run, data, reg, completed, steps, previous, prefix)
            completed += 1
            steps += 1
            windows.append({"window": completed, "checkpoint_id": previous["checkpoint_id"], "numeric": event["numeric"], "durations": event["durations"]})
        if number != current_segment:
            result = read_json(segment / "result.json")
            require(result.get("verified_end_windows") == completed and not result.get("continuation_blocked_reason"), "Prior result is inconsistent or blocked")
            require(set(result.get("terminal_artifacts_sha256", {})) <= set(TERMINAL_FILES), "Unknown terminal artifact")
            for name, digest in result.get("terminal_artifacts_sha256", {}).items():
                require(sha256(segment / name) == digest, "Terminal evidence changed")
            if result["status"] == "completed":
                validate_completion(segment, reg, completed - start, result["supervisor"])
        segments.append({"segment": number, "start_windows": start, "end_windows": completed, "unsealed_tail": tail})
    return {"completed_windows": completed, "optimizer_steps": steps, "prefix": prefix, "last": previous, "segments": segments, "windows": windows}


def validate_completion(segment, reg, count, outcome):
    worker = read_json(segment / "worker_result.json")
    from .training_contract import PILOT
    require(outcome.get("status") == "completed" and outcome.get("returncode") == 0 and outcome.get("reason") == "worker_exited"
            and worker.get("schema") == "cnsbridge.mps_worker.v1" and worker.get("status") == "completed"
            and worker.get("training_run_id") == reg["training_run_id"] and worker.get("segment_id") == reg["segment_id"]
            and worker.get("completed_windows") == count == reg["windows"], "Not all registered windows completed")
    require(worker.get("raw_counts_sha256_before") == worker.get("raw_counts_sha256_after") == PILOT["raw_ordered_counts_sha256"], "Full raw graph evidence missing")
    require(worker.get("mps_memory", {}).get("error") is None and worker.get("mps_memory", {}).get("samples", 0) > 0, "MPS memory evidence failed")
