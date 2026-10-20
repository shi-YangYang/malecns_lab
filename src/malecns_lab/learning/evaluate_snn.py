"""Register final-epoch policies, explicitly run a bounded scene segment, or summarize."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import psutil

from ..brain.metadata import canonical_hash
from .evaluation_contract import (CONDITIONS, VAL, TEST, episode_payload, load_condition, path_inside,
                                  register_suite, validate_suite)
from .evaluation_evidence import (TERMINAL_FILES, check_order, episode_report, scan_episode, scan_suite,
                                  seal_validation, validate_completion)
from .resource_guard import GIB, resource_budget, supervise, write_json
from .train_snn import workspace_lock
from .training_contract import bytes_sha, read_json, require, sha256


def register_segment(directory, run, episode, chain, frames):
    require(type(frames) is int and 1 <= frames <= 40, "--frames must be in 1..40")
    require(not chain["terminal"] and chain["next_frame"] + frames <= episode["planned_frames"], "Requested frames exceed scene end")
    number = len(chain["segments"]) + 1
    segment = Path(directory) / "segments" / f"{number:06d}"
    segment.mkdir(parents=True, exist_ok=False)
    budget = resource_budget(psutil.virtual_memory().available, reserve_bytes=GIB, rss_bytes=6 * GIB)
    reg = {"schema": "cnsbridge.evaluation_segment.v1", "evaluation_run_id": run["evaluation_run_id"],
           "episode_id": episode["episode_id"], "condition_id": episode["condition_id"], "parameter_set_id": episode["parameter_set_id"],
           "segment": number, "start_frame": chain["next_frame"], "parent_checkpoint_id": chain["last"]["checkpoint_id"] if chain["last"] else None,
           "prefix_sha256": bytes_sha(chain["prefix"]), "frames": frames,
           "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}, "clock_monotonic": time.monotonic()}
    reg["segment_id"] = canonical_hash(reg)
    write_json(segment / "registration.json", reg)
    return segment, reg


def finalize_segment(suite, directory, segment, run, episode, parameters, reg, outcome, *, production=True):
    write_json(segment / "supervisor_result.json", outcome)
    result = {"schema": "cnsbridge.evaluation_segment_result.v1", "status": "failed", "reason": outcome.get("reason"),
              "evaluation_run_id": run["evaluation_run_id"], "segment_id": reg["segment_id"], "verified_end_frame": None,
              "recoverable_prefix": None, "supervisor": outcome,
              "scope": "full_reference_closed_loop" if production else "numerical_test_only"}
    try:
        if production:
            require(validate_suite(suite) == run, "Suite changed during worker execution")
        chain = scan_episode(directory, run, episode, parameters, production=production, current_segment=reg["segment"])
        result["verified_end_frame"] = chain["next_frame"]
        if chain["last"]:
            result["recoverable_prefix"] = {"checkpoint_id": chain["last"]["checkpoint_id"], "path": chain["last"]["path"],
                                             "context": chain["last"]["context"]}
        worker = read_json(segment / "worker_result.json") if (segment / "worker_result.json").exists() else {}
        if worker.get("failure_kind") == "numerical_or_protocol":
            result.update(recoverable_prefix=None, continuation_blocked_reason="numerical_or_protocol", detail=worker.get("reason"))
        if outcome.get("status") == "completed":
            validate_completion(segment, reg, chain["next_frame"] - reg["start_frame"], chain["terminal"], production=production)
            result.update(status="completed", reason="registered_frames_verified")
        # Only this parent's validated outcome can complete the formal scene.
        # A sealed terminal prefix from a failed worker remains diagnostic.
        chain["segments"][-1].update(status=result["status"], reason=result["reason"])
        chain["evidence_complete"] = chain["terminal"] and result["status"] == "completed"
        result["episode"] = episode_report(chain, run, episode)
    except Exception as error:
        result.update(reason="invalid_segment_evidence", recoverable_prefix=None, detail=f"{type(error).__name__}:{error}")
    result["terminal_artifacts_sha256"] = {name: sha256(segment / name) for name in TERMINAL_FILES if (segment / name).exists()}
    write_json(segment / "result.json", result)
    return result


def run_segment(directory, name, scene, frames):
    directory = Path(directory).resolve()
    require(type(frames) is int and 1 <= frames <= 40, "--frames must be in 1..40")
    with workspace_lock():
        run = validate_suite(directory)
        summary = scan_suite(directory, run)
        check_order(directory, summary, name, scene)
        condition, parameters = load_condition(directory, run, name)
        episode = episode_payload(run, condition, scene)
        episode_dir = path_inside(directory, f"episodes/{name}/{scene}")
        if not episode_dir.exists():
            episode_dir.mkdir(parents=True, exist_ok=False)
            write_json(episode_dir / "episode.json", episode)
        chain = scan_episode(episode_dir, run, episode, parameters)
        if name == "joint" and scene == VAL[0] and chain["next_frame"] < 2:
            require(frames == 1, "The first actual gate requires two separate one-frame workers")
        segment, reg = register_segment(episode_dir, run, episode, chain, frames)
        if not reg["resources"]["can_launch"]:
            outcome = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                       "returncode": None, "wall_seconds": 0., "peak_observed_rss_bytes": 0}
        else:
            outcome = supervise([sys.executable, "-m", "malecns_lab.learning.evaluate_snn", "--worker", str(segment)], segment,
                                rss_limit_bytes=reg["resources"]["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                wall_seconds=600, poll_seconds=.1)
        return finalize_segment(directory, episode_dir, segment, run, episode, parameters, reg, outcome)


def summarize(directory):
    with workspace_lock():
        run = validate_suite(directory)
        result = scan_suite(directory, run)
        if result["val_complete"]:
            seal_validation(directory, result)
        write_json(Path(directory) / "summary.json", result)
        return {"status": "completed" if result["evidence_valid"] else "failed",
                "reason": "verified_evidence_summary" if result["evidence_valid"] else "summary_contains_failed_evidence", **result}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    sub = cli.add_subparsers(dest="command")
    register = sub.add_parser("register")
    register.add_argument("--joint-run", type=Path, required=True)
    register.add_argument("--frozen-run", type=Path, required=True)
    register.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--condition", choices=CONDITIONS, required=True)
    run.add_argument("--scene", choices=VAL + TEST, required=True)
    run.add_argument("--frames", type=int, required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("--suite", type=Path, required=True)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.worker and args.command:
        cli.error("Worker and public commands are mutually exclusive")
    if not args.worker and not args.command:
        cli.error("register, run or summary is required")
    try:
        if args.worker:
            from .evaluation_worker import run_worker
            result = run_worker(args.worker)
        elif args.command == "register":
            with workspace_lock():
                run = register_suite(args.joint_run, args.frozen_run, args.out)
            result = {"status": "completed", "reason": "final_epoch_parameters_registered", "evaluation_run_id": run["evaluation_run_id"]}
        elif args.command == "run":
            result = run_segment(args.suite, args.condition, args.scene, args.frames)
        else:
            result = summarize(args.suite)
    except Exception as error:
        result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
