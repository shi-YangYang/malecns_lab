"""Run one supervised 1–4 window segment; explicit invocation is required to resume."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

from ..brain.metadata import canonical_hash
from .resource_guard import GIB, resource_budget, supervise, write_json
from .training_contract import (SCHEMA, bytes_sha, cursor, read_json, register_run, require, sha256, validate_run)
from .training_evidence import scan_run, validate_segment_completion


@contextmanager
def workspace_lock():
    path = Path(__file__).resolve().parents[3] / "data" / ".continuous-training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another training parent owns the workspace lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def register_segment(directory, run, chain, windows):
    require(type(windows) is int and 1 <= windows <= 4, "--windows must be between 1 and 4")
    require(chain["completed_windows"] + windows <= run["protocol"]["windows"], "Requested windows exceed the epoch end")
    number = len(chain["segments"]) + 1
    segment = directory / "segments" / f"{number:06d}"
    segment.mkdir(parents=True, exist_ok=False)
    budget = resource_budget(psutil.virtual_memory().available, reserve_bytes=GIB, rss_bytes=6 * GIB)
    registration = {"schema": "cnsbridge.training_segment.v1", "training_run_id": run["training_run_id"],
        "segment": number, "start_cursor": cursor(run["scenes"], chain["completed_windows"], chain["optimizer_steps"]),
        "parent_checkpoint_id": chain["last"]["checkpoint_id"] if chain["last"] else None,
        "prefix_sha256": bytes_sha(chain["prefix"]), "windows": windows,
        "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}, "clock_monotonic": time.monotonic()}
    registration["segment_id"] = canonical_hash(registration)
    write_json(segment / "registration.json", registration)
    return segment, registration


def summarize_progress(chain, run):
    records = chain["windows"]
    return {"schema": "cnsbridge.training_progress.v1", "training_run_id": run["training_run_id"], "mode": run["mode"],
            "scope": "offline_teacher_trajectory_imitation_not_neural_closed_loop", "completed_windows": chain["completed_windows"],
            "optimizer_steps": chain["optimizer_steps"], "processed_frames": chain["completed_windows"] * 2,
            "valid_frames": sum(r["numeric"]["valid_frames"] for r in records),
            "epoch_complete": chain["completed_windows"] == run["protocol"]["windows"],
            "final_checkpoint": chain["last"]["path"] if chain["last"] else None,
            "label_axis_range": ([np.min([r["numeric"]["label_axis_range"][0] for r in records], axis=0).tolist(),
                                  np.max([r["numeric"]["label_axis_range"][1] for r in records], axis=0).tolist()] if records else None),
            "action_axis_range": ([np.min([r["numeric"]["action_axis_range"][0] for r in records], axis=0).tolist(),
                                   np.max([r["numeric"]["action_axis_range"][1] for r in records], axis=0).tolist()] if records else None),
            "windows": records, "segments": chain["segments"]}


def finalize_segment(directory, segment, run, data, registration, outcome, *, production=True):
    write_json(segment / "supervisor_result.json", outcome)
    result = {"schema": "cnsbridge.training_segment_result.v1", "status": "failed", "reason": outcome.get("reason"),
              "training_run_id": run["training_run_id"], "segment_id": registration["segment_id"],
              "verified_end_windows": None, "recoverable_prefix": None, "supervisor": outcome,
              "scope": "full_reference_training" if production else "numerical_test_only"}
    try:
        if production:
            data = validate_run(run)  # Recheck source/graph/data bytes after the child exits.
        chain = scan_run(directory, run, data, production=production, current_segment=registration["segment"])
        count = chain["completed_windows"] - registration["start_cursor"]["completed_windows"]
        result["verified_end_windows"] = chain["completed_windows"]
        result["recoverable_prefix"] = ({"checkpoint_id": chain["last"]["checkpoint_id"], "path": chain["last"]["path"],
                                          "cursor": chain["last"]["context"]} if chain["last"] else None)
        worker = read_json(segment / "worker_result.json") if (segment / "worker_result.json").exists() else {}
        if worker.get("failure_kind") == "numerical_or_protocol":
            result.update(recoverable_prefix=None, continuation_blocked_reason="numerical_or_protocol", detail=worker.get("reason"))
        if outcome.get("status") == "completed":
            validate_segment_completion(segment, registration, count, outcome)
            result.update(status="completed", reason="registered_windows_verified")
        write_json(directory / "progress.json", summarize_progress(chain, run))
    except Exception as error:
        result.update(reason="invalid_segment_evidence", detail=f"{type(error).__name__}:{error}")
    names = ("registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json", "supervisor_result.json",
             "resume.json", "resume.npz")
    result["terminal_artifacts_sha256"] = {name: sha256(segment / name) for name in names if (segment / name).exists()}
    write_json(segment / "result.json", result)
    return result


def run_segment(*, directory, windows, graph=None, dataset=None, mode=None, resume=False):
    directory = Path(directory).resolve()
    require(type(windows) is int and 1 <= windows <= 4, "--windows must be between 1 and 4")
    with workspace_lock():
        if resume:
            require(all(value is None for value in (graph, dataset, mode)), "Resume cannot change registered sources/mode")
            run = read_json(directory / "run.json")
        else:
            require(graph is not None and dataset is not None and mode is not None, "New run requires graph/dataset/mode")
            require(not directory.exists(), "New run directory already exists")
            directory.mkdir(parents=True, exist_ok=False)
            try:
                run = register_run(graph, dataset, mode)
                write_json(directory / "run.json", run)
            except Exception as error:
                write_json(directory / "registration_failure.json", {"status": "failed", "reason": f"{type(error).__name__}:{error}"})
                raise
        data = validate_run(run)
        chain = scan_run(directory, run, data)
        segment, registration = register_segment(directory, run, chain, windows)
        budget = registration["resources"]
        if not budget["can_launch"]:
            outcome = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                       "returncode": None, "wall_seconds": 0., "peak_observed_rss_bytes": 0}
        else:
            outcome = supervise([sys.executable, "-m", "cnsbridge.learning.train_snn", "--worker", str(segment)], segment,
                                rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                wall_seconds=600, poll_seconds=.1)
        return finalize_segment(directory, segment, run, data, registration, outcome)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "dataset", "out", "run"):
        value.add_argument("--" + name, type=Path)
    value.add_argument("--mode", choices=("joint", "frozen_brain"))
    value.add_argument("--windows", type=int)
    value.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return value


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    new = (args.graph, args.dataset, args.out, args.mode)
    if args.worker:
        if args.run is not None or args.windows is not None or any(v is not None for v in new):
            cli.error("--worker cannot be combined with public options")
        from .training_worker import run_worker
        result = run_worker(args.worker)
    else:
        if args.windows is None or not 1 <= args.windows <= 4:
            cli.error("--windows must be between 1 and 4")
        if args.run:
            if any(v is not None for v in new):
                cli.error("--run cannot be combined with graph/dataset/out/mode")
        elif any(v is None for v in new):
            cli.error("A new run requires --graph --dataset --out --mode")
        try:
            result = run_segment(directory=args.run or args.out, windows=args.windows, graph=args.graph, dataset=args.dataset,
                                 mode=args.mode, resume=args.run is not None)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "detail", "verified_end_windows", "recoverable_prefix")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
