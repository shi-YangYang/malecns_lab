"""Initialize, verify a disposable correction, then train the all-GPU branch."""
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
from .gpu_contract import register_run, validate_run, initial_prefix
from .gpu_evidence import (TERMINAL_FILES, read_genesis, read_gate, scan_run, validate_registration, validate_completion)


def run_job(*, directory, kind, windows=0, parent_run=None, credit_horizon=None, theta_norm_limit=None):
    require(kind in ("initialize", "gate", "train"), "Unknown job kind")
    require(type(windows) is int and (1 <= windows <= 4 if kind == "train" else windows == 0), "Invalid window request")
    require((parent_run is not None) == (kind == "initialize"), "Only initialization accepts a parent run")
    directory = Path(directory).resolve()
    with workspace_lock():
        if kind == "initialize":
            require(credit_horizon == 100 and theta_norm_limit == 1000, "Registered recovery requires horizon100 and theta norm ceiling1000")
            require(not directory.exists(), "New GPU recovery directory already exists")
            directory.mkdir(parents=True, exist_ok=False)
            try:
                run = register_run(parent_run, credit_horizon, theta_norm_limit)
                write_json(directory / "run.json", run)
            except Exception as error:
                write_json(directory / "registration_failure.json", {"reason": f"{type(error).__name__}:{error}"})
                raise
        else:
            require(credit_horizon is None and theta_norm_limit is None, "Resume cannot change frozen mathematics")
            run = read_json(directory / "run.json")
        data, parent, _ = validate_run(run)
        number, completed, prefix, previous = 0, 0, b"", parent
        if kind == "initialize":
            job = directory / "initialization"
        else:
            genesis = read_genesis(directory, run, parent)
            previous, prefix = genesis, initial_prefix(run, genesis)
            if kind == "gate":
                require(not (directory / "segments").exists(), "Gate must precede training")
                job = directory / "gate"
            else:
                read_gate(directory, run, data, genesis)
                chain = scan_run(directory, run, data, genesis)
                completed, previous, prefix = chain["completed_windows"], chain["last"], chain["prefix"]
                number = len(chain["segments"]) + 1
                job = directory / "segments" / f"{number:06d}"
        budget = resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB)
        reg = {"schema": "cnsbridge.gpu_recovery_job.v1", "kind": kind, "training_run_id": run["training_run_id"], "segment": number,
               "start_cursor": cursor(run["scenes"], completed, completed), "parent_checkpoint_id": previous["checkpoint_id"],
               "prefix_sha256": bytes_sha(prefix), "windows": windows,
               "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1},
               "clock_monotonic": time.monotonic(), "parent_pid": os.getpid()}
        reg["job_id"] = canonical_hash(reg)
        validate_registration(reg, run, kind, previous, segment=number, completed=completed, prefix=prefix)
        job.mkdir(parents=True, exist_ok=False)
        write_json(job / "registration.json", reg)
        if budget["can_launch"]:
            outcome = supervise([sys.executable, "-m", "malecns_lab.learning.train_gpu", "--worker", str(job)], job,
                                rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB, wall_seconds=600, poll_seconds=.1)
        else:
            outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None, "wall_seconds": 0.}
        write_json(job / "supervisor_result.json", outcome)
        result = {"schema": "cnsbridge.gpu_recovery_result.v1", "status": "failed", "reason": outcome["reason"],
                  "training_run_id": run["training_run_id"], "job_id": reg["job_id"], "kind": kind, "supervisor": outcome,
                  "verified_end_windows": None}
        try:
            # Revalidate files/source identities after the supervised child exits.
            data, parent, _ = validate_run(run)
            if kind == "initialize":
                latest = read_genesis(directory, run, parent, current=True)
                verified = 0
            elif kind == "gate":
                genesis = read_genesis(directory, run, parent)
                read_gate(directory, run, data, genesis, current=True)
                latest, verified = genesis, 0
            else:
                genesis = read_genesis(directory, run, parent)
                chain = scan_run(directory, run, data, genesis, current_segment=number)
                latest, verified = chain["last"], chain["completed_windows"]
            result.update(verified_end_windows=verified, final_checkpoint=latest["path"])
            validate_completion(job, reg, verified - completed, outcome, run)
            result.update(status="completed", reason="registered_all_gpu_evidence_verified")
            if kind == "train":
                write_json(directory / "progress.json", {"training_run_id": run["training_run_id"], "completed_windows": verified,
                    "optimizer_steps": verified, "epoch_complete": verified == 160, "final_checkpoint": latest["path"],
                    "windows": chain["windows"], "segments": chain["segments"], "parent_windows": 160,
                    "interpretation": "New objective/optimizer epoch; not old same-protocol window161"})
        except Exception as error:
            result.update(reason="invalid_or_failed_gpu_job", detail=f"{type(error).__name__}:{error}", continuation_blocked=True)
        result["terminal_artifacts_sha256"] = {name: sha256(job / name) for name in TERMINAL_FILES if (job / name).exists()}
        write_json(job / "result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--credit-horizon", type=int)
    parser.add_argument("--theta-norm-limit", type=float)
    parser.add_argument("--windows", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if any(v for k, v in vars(args).items() if k != "worker"):
            parser.error("Worker cannot combine public arguments")
        from .gpu_worker import run_worker
        result = run_worker(args.worker)
    else:
        if args.initialize:
            if args.gate or args.run or args.windows is not None or not args.parent_run or not args.out:
                parser.error("Initialize requires --parent-run/--out/--credit-horizon/--theta-norm-limit only")
            kind = "initialize"
        else:
            if not args.run or args.parent_run or args.out or args.credit_horizon is not None or args.theta_norm_limit is not None:
                parser.error("Gate/train requires only --run plus --gate or --windows")
            if args.gate == (args.windows is not None):
                parser.error("Choose exactly one of --gate or --windows")
            kind = "gate" if args.gate else "train"
        try:
            result = run_job(directory=args.out or args.run, kind=kind, windows=args.windows or 0,
                             parent_run=args.parent_run, credit_horizon=args.credit_horizon, theta_norm_limit=args.theta_norm_limit)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "detail", "verified_end_windows", "final_checkpoint")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
