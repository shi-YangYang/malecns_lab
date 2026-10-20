"""Sequential persisted-window and segment evidence checking without Torch."""
from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from .training_archive import inspect_checkpoint, validate_state_digest
from .training_contract import (PARAMETERS, active_keys, bound, bytes_sha, cursor, expected_ids, next_selection,
                                prefix_line, read_arrays, read_json, require, sha256)
from .training_math import compare_r2, validate_window_numbers
from .resource_guard import GIB, resource_budget


def validate_segment_registration(registration, run, number, completed, steps, parent, prefix):
    bound(registration, "segment_id")
    require(set(registration) == {"schema", "training_run_id", "segment", "start_cursor", "parent_checkpoint_id",
                                  "prefix_sha256", "windows", "resources", "clock_monotonic", "segment_id"}, "Segment fields differ")
    require(registration["schema"] == "cnsbridge.training_segment.v1" and registration["training_run_id"] == run["training_run_id"]
            and registration["segment"] == number and registration["start_cursor"] == cursor(run["scenes"], completed, steps)
            and registration["parent_checkpoint_id"] == parent and registration["prefix_sha256"] == bytes_sha(prefix),
            "Segment start does not match validated prefix")
    count = registration["windows"]
    require(type(count) is int and 1 <= count <= 4 and completed + count <= sum(s["frame_count"] for s in run["scenes"]) // 2,
            "Segment window budget invalid")
    resources = registration["resources"]
    expected = resource_budget(resources["available_bytes_before_launch"], reserve_bytes=GIB, rss_bytes=6 * GIB)
    require(resources == {**expected, "wall_limit_seconds": 600, "sampling_seconds": .1}, "Resource budget changed")
    require(type(registration["clock_monotonic"]) is float and np.isfinite(registration["clock_monotonic"]), "Invalid resource clock")


def sampled_boundary(segment, registration, sealed_monotonic):
    with (segment / "resources.csv").open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["wall_seconds", "tree_rss_bytes", "system_available_bytes"], "Resource CSV fields differ")
        rows = list(reader)
    require(rows, "No resource samples")
    elapsed = sealed_monotonic - registration["clock_monotonic"]
    require(np.isfinite(elapsed) and elapsed >= 0, "Window precedes segment resource clock")
    last, covered, count = -1., False, 0
    for row in rows:
        wall, rss, available = float(row["wall_seconds"]), int(row["tree_rss_bytes"]), int(row["system_available_bytes"])
        require(np.isfinite(wall) and wall >= last and rss >= 0 and available >= 0, "Invalid resource sample")
        last = wall
        # Conservative: require the first sample at/after the seal also to be within budget.
        require(wall <= 600 and rss <= registration["resources"]["effective_rss_limit_bytes"] and available >= GIB,
                "Window lacks an in-budget sampled boundary")
        count += 1
        if wall >= elapsed:
            covered = True
            break
    require(covered, "Sealed window lacks a conservative post-seal resource sample; registration/supervisor clock offset unresolved")
    return count


def validate_resume(segment, registration, previous):
    report = read_json(segment / "resume.json")
    values = read_arrays(segment / "resume.npz")
    require(report["schema"] == "cnsbridge.training_restored_state.v1"
            and report["segment_id"] == registration["segment_id"]
            and report["checkpoint_id"] == previous["checkpoint_id"]
            and report["context"] == previous["context"] and report["state"] == previous["state"]
            and report["arrays_sha256"] == sha256(segment / "resume.npz"), "Restored state/context differs before forward")
    expected = {**{f"parameter.{k}": v for k, v in previous["parameters"].items()},
                **{f"adam.{k}": v for k, v in previous["moments"].items()}}
    require(set(values) == set(expected) and all(np.array_equal(values[k], v) and values[k].dtype == v.dtype
                                               for k, v in expected.items()), "Actual restored parameters/Adam differ")


def validate_window(path, run, data, registration, completed, steps, previous, prefix, *, production=True):
    """Validate one window, returning only its small continuation summary."""
    path = Path(path)
    number = completed + 1
    seal, event = read_json(path / "window.json"), read_json(path / "event.json")
    bound(seal, "window_id")
    require(set(seal) == {"schema", "training_run_id", "segment_id", "window", "files_sha256", "checkpoint_id",
                          "parent_checkpoint_id", "cursor", "sealed_monotonic", "window_id"}, "Window manifest fields differ")
    files = ("evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")
    require(set(seal["files_sha256"]) == set(files), "Window file inventory differs")
    for name in files:
        require(sha256(path / name) == seal["files_sha256"][name], "Window file SHA differs")
    parent = previous["checkpoint_id"] if previous else None
    require(seal["schema"] == "cnsbridge.training_window.v1" and seal["training_run_id"] == run["training_run_id"]
            and seal["segment_id"] == registration["segment_id"] and seal["window"] == number
            and seal["parent_checkpoint_id"] == parent, "Window run/parent/order mismatch")
    values = read_arrays(path / "evidence.npz")
    scene_index, selected = next_selection(data, completed)
    for key, value in selected.items():
        require(values[key].dtype == value.dtype and np.array_equal(values[key], value), "Window inputs differ from real dataset")
    numeric = validate_window_numbers(values, run["policy"], run["mode"], steps)
    after_cursor = cursor(run["scenes"], number, steps + numeric["updated"])
    require(seal["cursor"] == after_cursor, "Window cursor differs")
    expected_event_keys = {"schema", "training_run_id", "segment_id", "window", "scene_index", "scene_id", "before_cursor", "after_cursor",
                           "parent_checkpoint_id", "evidence_sha256", "start_state", "end_state", "numeric", "training_losses", "durations", "r2_comparison"}
    require(set(event) == expected_event_keys and event["schema"] == "cnsbridge.training_event.v1", "Event schema/fields differ")
    require(event["training_run_id"] == run["training_run_id"] and event["segment_id"] == registration["segment_id"]
            and event["window"] == number and event["scene_index"] == scene_index
            and event["scene_id"] == data.sequences[scene_index].scene["scene_id"]
            and event["before_cursor"] == cursor(run["scenes"], completed, steps) and event["after_cursor"] == after_cursor
            and event["parent_checkpoint_id"] == parent and event["evidence_sha256"] == sha256(path / "evidence.npz")
            and event["numeric"] == numeric, "Event numerical/cursor binding mismatch")
    for value in event["durations"].values():
        require(type(value) is float and np.isfinite(value) and value >= 0, "Invalid phase duration")
    require(set(event["training_losses"]) == {"action", "regularization", "total"}, "Training losses incomplete")
    for key, value in event["training_losses"].items():
        if numeric["updated"]:
            require(type(value) is float and np.isclose(value, numeric["metrics"][key], rtol=1e-6, atol=1e-7), "Tensor loss mismatch")
        else:
            require(value is None, "Skipped window reports a loss")
    validate_state_digest(event["start_state"], run["policy"], int(selected["indices"][0]))
    validate_state_digest(event["end_state"], run["policy"], after_cursor["next_frame"])
    if previous:
        require(all(np.array_equal(values[f"before_{k}"], v) for k, v in previous["parameters"].items()), "Parameters did not continue")
        require(all(np.array_equal(values[f"adam_before.{k}"], v) for k, v in previous["moments"].items()), "Old Adam moments did not continue")
        if selected["indices"][0] != 0:
            require(event["start_state"] == previous["state"], "Within-scene recurrent state reset/changed")
    elif production:
        initial = read_arrays(Path(run["prior_run"]["path"]) / "initial_parameters.npz")
        require(all(np.array_equal(values[f"before_{k}"], initial[k]) for k in (*PARAMETERS, "gains", "rho")), "Initial parameters differ from r2")
        initial_state = read_json(Path(run["prior_run"]["path"]) / "baseline.json")["initial_state"]
        require(event["start_state"] == initial_state, "Initial precondition differs from r2")
    expected_r2 = compare_r2(values, run["prior_run"]["path"]) if production and number == 1 and run["mode"] == "joint" else None
    require(event["r2_comparison"] == expected_r2 and (expected_r2 is None or expected_r2["passed"]), "r2 first-window gate failed")
    expected_prefix = prefix + prefix_line(registration["segment"], number, sha256(path / "event.json"))
    require((path / "prefix.jsonl").read_bytes() == expected_prefix, "Actual event prefix has wrong path/order/hash or duplicate entries")
    context = {**expected_ids(run), "scenes": run["scenes"], **after_cursor,
               "parent_checkpoint_id": parent, "log_prefix_sha256": bytes_sha(expected_prefix)}
    saved = inspect_checkpoint(path / "checkpoint", run, context)
    require(seal["checkpoint_id"] == saved["checkpoint_id"] and saved["state"] == event["end_state"], "Checkpoint state/ID mismatch")
    require(all(np.array_equal(saved["parameters"][k], values[f"after_{k}"]) for k in PARAMETERS), "Checkpoint updated parameters differ")
    require(all(np.array_equal(v, values[f"adam_after.{k}"]) for k, v in saved["moments"].items()), "Checkpoint actual Adam differs")
    require(type(seal["sealed_monotonic"]) is float and np.isfinite(seal["sealed_monotonic"]), "Invalid seal timestamp")
    saved.update(prefix=expected_prefix, path=str(path / "checkpoint"), window_id=seal["window_id"], numeric=numeric)
    return saved


def scan_run(directory, run, data, *, production=True, current_segment=None):
    """Follow the unique sealed chain; never roll back a corrupt sealed window."""
    directory = Path(directory)
    completed, steps, prefix, previous = 0, 0, b"", None
    segments, window_reports = [], []
    paths = sorted((directory / "segments").iterdir()) if (directory / "segments").exists() else []
    require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in paths), "Unexpected segment path")
    require([int(p.name) for p in paths] == list(range(1, len(paths) + 1)), "Segment sequence has gaps")
    for segment in paths:
        number = int(segment.name)
        registration = read_json(segment / "registration.json")
        parent = previous["checkpoint_id"] if previous else None
        if number != current_segment and (segment / "worker_result.json").exists():
            prior_worker = read_json(segment / "worker_result.json")
            require(prior_worker.get("failure_kind") != "numerical_or_protocol", "Prior numerical/protocol failure requires a new reviewed protocol; no automatic resume")
        validate_segment_registration(registration, run, number, completed, steps, parent, prefix)
        start_completed = completed
        windows = sorted((segment / "windows").iterdir()) if (segment / "windows").exists() else []
        require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in windows), "Unexpected window path")
        require([int(p.name) for p in windows] == list(range(completed + 1, completed + 1 + len(windows))), "Duplicate/gapped window numbers")
        require(len(windows) <= registration["windows"], "Segment exceeded requested window count")
        if windows and (windows[0] / "window.json").exists() and previous:
            validate_resume(segment, registration, previous)
        tail = False
        for path in windows:
            if not (path / "window.json").exists():
                tail = True
                continue
            require(not tail, "Sealed window follows an incomplete tail")
            saved = validate_window(path, run, data, registration, completed, steps, previous, prefix, production=production)
            sampled_boundary(segment, registration, read_json(path / "window.json")["sealed_monotonic"])
            completed += 1
            steps = saved["context"]["optimizer_steps"]
            prefix = saved["prefix"]
            previous = saved  # Drops the prior checkpoint arrays; full states already reduced to hashes.
            window_reports.append({"window": completed, "segment": number, "checkpoint_id": saved["checkpoint_id"],
                                   "cursor": saved["context"], "numeric": saved["numeric"]})
        result_path = segment / "result.json"
        if result_path.exists() and number != current_segment:
            result = read_json(result_path)
            if result.get("status") == "completed":
                validate_segment_completion(segment, registration, completed - start_completed)
            require(result.get("verified_end_windows") == completed, "Segment result progress differs from actual prefix")
            for name, digest in result.get("terminal_artifacts_sha256", {}).items():
                require(name in {"registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json", "supervisor_result.json",
                                 "resume.json", "resume.npz"}, "Unexpected terminal artifact path")
                require(sha256(segment / name) == digest, "Terminal artifact changed")
        segments.append({"segment": number, "start_windows": start_completed, "end_windows": completed,
                         "unsealed_tail": tail,
                         "supervisor": read_json(segment / "supervisor_result.json") if (segment / "supervisor_result.json").exists() else None})
    return {"completed_windows": completed, "optimizer_steps": steps, "prefix": prefix, "last": previous,
            "segments": segments, "windows": window_reports}


def validate_segment_completion(segment, registration, count, outcome=None):
    raw = outcome if outcome is not None else read_json(segment / "supervisor_result.json")
    require(raw.get("status") == "completed" and raw.get("returncode") == 0 and raw.get("reason") == "worker_exited",
            "Segment process was not successfully supervised")
    worker = read_json(segment / "worker_result.json")
    require(worker.get("schema") == "cnsbridge.training_worker.v1" and worker.get("status") == "completed"
            and worker.get("segment_id") == registration["segment_id"] and worker.get("training_run_id") == registration["training_run_id"]
            and worker.get("completed_windows") == count == registration["windows"], "Worker completion lacks all registered windows")
    from .training_contract import PILOT
    require(worker.get("raw_counts_sha256_before") == worker.get("raw_counts_sha256_after") == PILOT["raw_ordered_counts_sha256"],
            "Worker full raw graph verification missing")
    expected = {"phase.json"}
    if registration["parent_checkpoint_id"] is not None:
        expected |= {"resume.json", "resume.npz"}
    require(set(worker.get("artifacts_sha256", {})) == expected, "Worker terminal inventory missing")
    for name, digest in worker["artifacts_sha256"].items():
        require(sha256(segment / name) == digest, "Worker terminal artifact mismatch")
