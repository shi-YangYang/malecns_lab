"""Torch-free all-GPU genesis, correction-gate and bounded training-chain checks."""
from pathlib import Path
import re
import numpy as np
from ..brain.metadata import canonical_hash
from .resource_guard import GIB, resource_budget
from .training_contract import (PARAMETERS, bound, bytes_sha, cursor, expected_ids, next_selection, prefix_line,
                               read_arrays, read_json, require, sha256)
from .mps_evidence import memory_boundary
from .gpu_archive import inspect_checkpoint, validate_state, validate_state_digest, verify_resume
from .gpu_contract import initial_prefix
from .gpu_numbers import validate_numbers, validate_correction

TERMINAL_FILES = ("registration.json", "worker.claim", "worker_result.json", "resources.csv", "worker.log", "phase.json",
                  "supervisor_result.json", "resume.json", "resume.npz", "mps_memory.csv", "mps_budget.json", "timing.json", "seal.json", "checkpoint/checkpoint.json")


def validate_registration(reg, run, kind, previous, *, segment=0, completed=0, prefix=b""):
    bound(reg, "job_id")
    require(set(reg) == {"schema", "kind", "training_run_id", "segment", "start_cursor", "parent_checkpoint_id", "prefix_sha256",
                         "windows", "resources", "clock_monotonic", "parent_pid", "job_id"}, "Job fields differ")
    require(reg["schema"] == "cnsbridge.gpu_recovery_job.v1" and reg["kind"] == kind
            and reg["training_run_id"] == run["training_run_id"] and reg["segment"] == segment
            and reg["start_cursor"] == cursor(run["scenes"], completed, completed)
            and reg["parent_checkpoint_id"] == previous["checkpoint_id"] and reg["prefix_sha256"] == bytes_sha(prefix), "Job lineage differs")
    windows = reg["windows"]
    require(type(windows) is int and ((kind in ("initialize", "gate") and windows == 0)
            or (kind == "train" and 1 <= windows <= 4 and completed + windows <= 160)), "Job window budget differs")
    if kind == "train" and completed < 2:
        require(windows == 1, "First two windows require separate one-window workers")
    resources = reg["resources"]
    budget = resource_budget(resources["available_bytes_before_launch"], reserve_bytes=GIB, rss_bytes=6 * GIB)
    require(resources == {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}
            and type(reg["parent_pid"]) is int and reg["parent_pid"] > 0
            and type(reg["clock_monotonic"]) is float and np.isfinite(reg["clock_monotonic"]), "Invalid resources or parent")


def validate_completion(job, reg, count, outcome, run):
    worker = read_json(job / "worker_result.json")
    require(outcome.get("status") == "completed" and outcome.get("returncode") == 0 and outcome.get("reason") == "worker_exited"
            and worker.get("schema") == "cnsbridge.gpu_recovery_worker.v1" and worker.get("status") == "completed"
            and worker.get("training_run_id") == run["training_run_id"] and worker.get("job_id") == reg["job_id"]
            and worker.get("kind") == reg["kind"] and worker.get("completed_windows") == count == reg["windows"], "Job did not complete")
    expected = run["policy"]["model"]["graph"]["raw_ordered_counts_sha256"]
    require(worker.get("raw_counts_sha256_before") == worker.get("raw_counts_sha256_after") == expected
            and worker.get("actual_policy") == run["policy"] and worker.get("actual_optimizer") == run["optimizer"], "Actual graph/device identities differ")
    require(worker.get("mps_memory", {}).get("error") is None and worker.get("mps_memory", {}).get("samples", 0) > 0,
            "Missing device memory evidence")
    memory_boundary(job, reg, worker["sealed_monotonic"])


def terminal(job, run):
    result = read_json(job / "result.json")
    require(result.get("status") == "completed" and result.get("training_run_id") == run["training_run_id"], "Earlier job failed or incomplete")
    recorded = result.get("terminal_artifacts_sha256", {})
    expected = {name for name in TERMINAL_FILES if (job / name).exists()}
    require(set(recorded) == expected, "Terminal inventory differs")
    for name, digest in recorded.items():
        require(sha256(job / name) == digest, "Terminal evidence changed")
    reg = read_json(job / "registration.json")
    require(result.get("job_id") == reg["job_id"] and result.get("verified_end_windows") == reg["start_cursor"]["completed_windows"] + reg["windows"], "Terminal cursor differs")
    validate_completion(job, reg, reg["windows"], result["supervisor"], run)


def read_genesis(directory, run, parent, *, current=False):
    job = directory / "initialization"
    reg = read_json(job / "registration.json")
    validate_registration(reg, run, "initialize", parent)
    saved = inspect_checkpoint(job / "checkpoint", run)
    context = {**expected_ids(run), "scenes": run["scenes"], **cursor(run["scenes"], 0, 0),
               "parent_checkpoint_id": parent["checkpoint_id"], "log_prefix_sha256": bytes_sha(b"")}
    require(saved["context"] == context, "Genesis cursor differs")
    require(all(np.array_equal(v, saved["parameters"][k]) for k, v in parent["parameters"].items()), "Genesis changed parent parameters")
    require(all(np.count_nonzero(v) == 0 for v in saved["moments"].values()), "Genesis retained old optimizer state")
    config = run["policy"]["drive"]["config"]
    initial = np.random.Generator(np.random.PCG64(config["phase_seed"])).random(len(saved["arrays"]["drive.phase"]))
    initial = np.minimum(initial.astype(np.float32), np.nextafter(np.float32(1), np.float32(0)))
    require(np.array_equal(saved["arrays"]["drive.phase"], initial) and not np.any(saved["arrays"]["drive.event_counts"]), "Blank genesis drive differs")
    if not current:
        terminal(job, run)
    return saved


def validate_residency(record):
    expected = {"parameters", "raw_gradients", "clipped_gradients", "adam", "phase", "counts", "logits", "loss"}
    require(set(record) == expected, "Residency inventory differs")
    for key in ("parameters", "raw_gradients", "clipped_gradients"):
        require(set(record[key]) == set(PARAMETERS.values()) and set(record[key].values()) == {"mps:0"}, "Compute tensors left MPS")
    require(set(record["adam"]) == {f"{name}.{f}" for name in PARAMETERS.values() for f in ("step", "exp_avg", "exp_avg_sq")}
            and set(record["adam"].values()) == {"mps:0"}, "Adam state left MPS")
    require(all(record[k] == "mps:0" for k in ("phase", "counts", "logits", "loss")), "Policy compute left MPS")


def validate_record(record, values, run, previous, frame_start, frame_end):
    numeric = validate_numbers(values, run["policy"], previous["context"]["optimizer_steps"], run["protocol"]["theta_norm_limit"])
    require(record["numeric"] == numeric, "Saved numeric report differs")
    require(set(record["training_losses"]) == set(numeric["losses"]) and all(np.isclose(v, numeric["losses"][k], rtol=2e-5, atol=1e-6)
            for k, v in record["training_losses"].items()), "Tensor losses differ")
    require(all(type(v) is float and np.isfinite(v) and v >= 0 for v in record["durations"].values()), "Invalid timings")
    validate_residency(record["residency"])
    for key, value in previous["parameters"].items():
        require(np.array_equal(value, values[f"before_{key}"]), "Parameter continuation changed")
    for key, value in previous["moments"].items():
        require(np.array_equal(value, values[f"adam_before.{key}"]), "Optimizer continuation changed")
    validate_state_digest(record["start_state"], run["policy"], frame_start)
    validate_state_digest(record["end_state"], run["policy"], frame_end)
    if frame_start != 0 or previous["context"]["completed_windows"] == 0:
        require(record["start_state"] == previous["state"], "Recurrent state changed before window")
    return numeric


def read_gate(directory, run, data, genesis, *, current=False):
    job = directory / "gate"
    reg = read_json(job / "registration.json")
    validate_registration(reg, run, "gate", genesis, prefix=initial_prefix(run, genesis))
    verify_resume(job, reg, genesis, run)
    values, record = read_arrays(job / "evidence.npz"), read_json(job / "gate.json")
    require(set(record) == {"job_id", "evidence_sha256", "end_arrays_sha256", "end_metadata", "start_state", "end_state", "numeric",
                            "training_losses", "residency", "durations", "correction"}
            and record["job_id"] == reg["job_id"] and record["evidence_sha256"] == sha256(job / "evidence.npz")
            and record["end_arrays_sha256"] == sha256(job / "end_state.npz"), "Gate binding differs")
    _, selection = next_selection(data, 159)
    require(all(values[k].dtype == v.dtype and np.array_equal(values[k], v) for k, v in selection.items()), "Gate selected input differs")
    numeric = validate_record(record, values, run, genesis, 0, 2)
    require(validate_state(record["end_metadata"], read_arrays(job / "end_state.npz"), run["policy"], 2) == record["end_state"], "Gate final state differs")
    require(record["correction"] == validate_correction(values, numeric), "Gate correction report differs")
    # Bind nonterminal gate artifacts into the independently checked terminal.
    manifest = read_json(job / "seal.json")
    bound(manifest, "gate_id")
    require(set(manifest) == {"job_id", "files_sha256", "gate_id"} and manifest["job_id"] == reg["job_id"]
            and manifest["files_sha256"] == {name: sha256(job / name) for name in ("gate.json", "evidence.npz", "end_state.npz")}, "Gate seal differs")
    if not current:
        terminal(job, run)
    return record


def validate_window(path, run, data, reg, previous, completed, prefix):
    number = completed + 1
    seal, event = read_json(path / "window.json"), read_json(path / "event.json")
    bound(seal, "window_id")
    require(set(seal) == {"job_id", "window", "parent_checkpoint_id", "checkpoint_id", "files_sha256", "sealed_monotonic", "window_id"}
            and seal["job_id"] == reg["job_id"] and seal["window"] == number and seal["parent_checkpoint_id"] == previous["checkpoint_id"], "Window seal differs")
    require(seal["files_sha256"] == {name: sha256(path / name) for name in ("evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")}, "Window sealed files changed")
    require({p.name for p in path.iterdir()} == {"window.json", "event.json", "evidence.npz", "prefix.jsonl", "checkpoint"}, "Sealed window inventory differs")
    values = read_arrays(path / "evidence.npz")
    scene, selection = next_selection(data, completed)
    require(all(values[k].dtype == v.dtype and np.array_equal(values[k], v) for k, v in selection.items()), "Training input selection differs")
    expected = {"training_run_id": run["training_run_id"], "job_id": reg["job_id"], "window": number,
                "scene_index": scene, "scene_id": data.sequences[scene].scene["scene_id"],
                "parent_checkpoint_id": previous["checkpoint_id"], "evidence_sha256": sha256(path / "evidence.npz")}
    require(set(event) == {*expected, "start_state", "end_state", "numeric", "training_losses", "residency", "durations"}
            and all(event[k] == v for k, v in expected.items()), "Window event binding differs")
    validate_record(event, values, run, previous, int(selection["indices"][0]), int(selection["indices"][-1]) + 1)
    prefix += prefix_line(reg["segment"], number, sha256(path / "event.json"))
    require((path / "prefix.jsonl").read_bytes() == prefix, "Window prefix changed")
    saved = inspect_checkpoint(path / "checkpoint", run)
    context = {**expected_ids(run), "scenes": run["scenes"], **cursor(run["scenes"], number, number),
               "parent_checkpoint_id": previous["checkpoint_id"], "log_prefix_sha256": bytes_sha(prefix)}
    require(saved["context"] == context and saved["checkpoint_id"] == seal["checkpoint_id"] and saved["state"] == event["end_state"], "Saved window state differs")
    require(all(np.array_equal(v, values[f"after_{k}"]) for k, v in saved["parameters"].items())
            and all(np.array_equal(v, values[f"adam_after.{k}"]) for k, v in saved["moments"].items()), "Saved parameter or Adam update differs")
    memory_boundary(path.parents[1], reg, seal["sealed_monotonic"])
    return saved, prefix, event


def scan_run(directory, run, data, genesis, *, current_segment=None):
    completed, previous, prefix = 0, genesis, initial_prefix(run, genesis)
    segments, windows = [], []
    paths = sorted((directory / "segments").iterdir()) if (directory / "segments").exists() else []
    require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in paths)
            and [int(p.name) for p in paths] == list(range(1, len(paths) + 1)), "Segment sequence differs")
    for segment in paths:
        number, start = int(segment.name), completed
        reg = read_json(segment / "registration.json")
        validate_registration(reg, run, "train", previous, segment=number, completed=completed, prefix=prefix)
        children = sorted((segment / "windows").iterdir()) if (segment / "windows").exists() else []
        require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in children)
                and [int(p.name) for p in children] == list(range(completed + 1, completed + len(children) + 1))
                and len(children) <= reg["windows"], "Window sequence differs")
        if children:
            verify_resume(segment, reg, previous, run)
        tail = False
        for path in children:
            if not (path / "window.json").exists():
                tail = True
                continue
            require(not tail, "Window seal follows incomplete tail")
            previous, prefix, event = validate_window(path, run, data, reg, previous, completed, prefix)
            completed += 1
            windows.append({"window": completed, "checkpoint_id": previous["checkpoint_id"], "numeric": event["numeric"], "durations": event["durations"]})
        if number != current_segment:
            require(not tail and completed - start == reg["windows"], "Prior segment incomplete; preserve failure and start new protocol")
            terminal(segment, run)
        segments.append({"segment": number, "start_windows": start, "end_windows": completed, "unsealed_tail": tail})
    return {"completed_windows": completed, "optimizer_steps": completed, "prefix": prefix, "last": previous, "segments": segments, "windows": windows}
