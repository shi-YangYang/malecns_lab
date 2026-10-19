"""Resume one to four complete windows on MPS with original CPU master Adam."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import psutil
from ..brain.metadata import canonical_hash
from .resource_guard import GIB, resource_budget, supervise, write_json
from .training_contract import bytes_sha, cursor, read_json, require, sha256
from .train_snn import workspace_lock
from .mps_contract import register_run, validate_run
from .mps_evidence import TERMINAL_FILES, scan_run, validate_completion


def run_segment(*, directory, windows, cpu_run=None):
    require(type(windows) is int and 1 <= windows <= 4, "--windows must be between 1 and 4")
    directory = Path(directory).resolve()
    with workspace_lock():
        if cpu_run is not None:
            require(not directory.exists(), "New MPS run directory already exists")
            directory.mkdir(parents=True, exist_ok=False)
            try:
                run = register_run(cpu_run)
                write_json(directory / "run.json", run)
            except Exception as error:
                write_json(directory / "registration_failure.json", {"reason": f"{type(error).__name__}:{error}"})
                raise
        else:
            run = read_json(directory / "run.json")
        data, initial = validate_run(run)
        chain = scan_run(directory, run, data, initial)
        require(chain["completed_windows"] + windows <= 160, "Requested windows exceed inherited epoch budget")
        number = len(chain["segments"]) + 1
        segment = directory / "segments" / f"{number:06d}"
        segment.mkdir(parents=True, exist_ok=False)
        budget = resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB)
        reg = {"schema": "cnsbridge.mps_segment.v1", "training_run_id": run["training_run_id"], "segment": number,
               "start_cursor": cursor(run["scenes"], chain["completed_windows"], chain["optimizer_steps"]),
               "parent_checkpoint_id": chain["last"]["checkpoint_id"], "prefix_sha256": bytes_sha(chain["prefix"]),
               "windows": windows, "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1},
               "clock_monotonic": time.monotonic(), "parent_pid": os.getpid()}
        reg["segment_id"] = canonical_hash(reg)
        write_json(segment / "registration.json", reg)
        if budget["can_launch"]:
            outcome = supervise([sys.executable, "-m", "cnsbridge.learning.train_mps", "--worker", str(segment)], segment,
                                rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB, wall_seconds=600, poll_seconds=.1)
        else:
            outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None, "wall_seconds": 0.}
        write_json(segment / "supervisor_result.json", outcome)
        result = {"schema": "cnsbridge.mps_segment_result.v1", "status": "failed", "reason": outcome["reason"],
                  "training_run_id": run["training_run_id"], "segment_id": reg["segment_id"], "supervisor": outcome,
                  "verified_end_windows": None, "recoverable_prefix": None}
        try:
            data, initial = validate_run(run)
            chain = scan_run(directory, run, data, initial, current_segment=number)
            result.update(verified_end_windows=chain["completed_windows"],
                          recoverable_prefix={"checkpoint_id": chain["last"]["checkpoint_id"], "path": chain["last"]["path"]})
            worker = read_json(segment / "worker_result.json") if (segment / "worker_result.json").exists() else {}
            if worker.get("failure_kind") == "numerical_or_protocol":
                result.update(continuation_blocked_reason="numerical_or_protocol", recoverable_prefix=None, detail=worker.get("reason"))
            if outcome["status"] == "completed":
                validate_completion(segment, reg, chain["completed_windows"] - reg["start_cursor"]["completed_windows"], outcome)
                result.update(status="completed", reason="registered_mps_windows_verified")
            write_json(directory / "progress.json", {"training_run_id": run["training_run_id"], "backend": run["runtime"],
                "cpu_inherited_windows": run["cpu_parent"]["context"]["completed_windows"],
                "completed_windows": chain["completed_windows"], "optimizer_steps": chain["optimizer_steps"],
                "gpu_completed_windows": len(chain["windows"]), "epoch_complete": chain["completed_windows"] == 160,
                "final_checkpoint": chain["last"]["path"], "windows": chain["windows"], "segments": chain["segments"]})
        except Exception as error:
            result.update(reason="invalid_segment_evidence", detail=f"{type(error).__name__}:{error}", continuation_blocked_reason="invalid_segment_evidence")
        result["terminal_artifacts_sha256"] = {name: sha256(segment / name) for name in TERMINAL_FILES if (segment / name).exists()}
        write_json(segment / "result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu-run", "out", "run", "worker"):
        parser.add_argument("--" + name, type=Path, help=argparse.SUPPRESS if name == "worker" else None)
    parser.add_argument("--windows", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if any(v is not None for v in (args.cpu_run, args.out, args.run, args.windows)):
            parser.error("--worker cannot be combined with public arguments")
        from .mps_worker import run_worker
        result = run_worker(args.worker)
    else:
        if args.windows is None or not 1 <= args.windows <= 4:
            parser.error("--windows must be between 1 and 4")
        if not ((args.run and args.cpu_run is None and args.out is None) or (args.run is None and args.cpu_run and args.out)):
            parser.error("Choose --cpu-run CPU --out NEW or --run EXISTING")
        try:
            result = run_segment(directory=args.run or args.out, cpu_run=args.cpu_run, windows=args.windows)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "detail", "verified_end_windows", "recoverable_prefix")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
