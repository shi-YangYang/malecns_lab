"""Explicit supervised segments of the single registered complete MaleCNS R graph."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys

import psutil

from .resource_guard import GIB, resource_budget, supervise, write_json
from .rewiring_contract import canonical, read_json, register_run, require, validate_run
from .rewiring_evidence import evidence_hashes, scan_run, validate_completion, validate_registration


@contextmanager
def workspace_lock():
    # Same OS lock as training, without importing its numeric/model dependency tree.
    path = Path(__file__).resolve().parents[3] / "data" / ".continuous-training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another training/generation parent owns the workspace lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def register_segment(directory, run, chain, proposals):
    require(not chain["terminal"], "Final candidate already accepted; no further segments")
    require(type(proposals) is int and 1 <= proposals <= run["protocol"]["max_segment_proposals"], "Invalid proposal quota")
    number = len(chain["segments"]) + 1
    budget = resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB)
    value = {"schema": "cnsbridge.rewiring_segment.v1", "run_id": run["run_id"], "segment": number,
             "parent_checkpoint_id": chain["checkpoint_id"], "prefix_before": chain["prefix"], "start_state": chain["state"],
             "proposals": proposals, "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}}
    value["segment_id"] = canonical(value)
    validate_registration(value, run, chain)
    segment = directory / "segments" / f"{number:06d}"
    segment.mkdir(parents=True, exist_ok=False)
    write_json(segment / "registration.json", value)
    return segment, value


def finalize_segment(segment, run, registration, outcome):
    # Preserve supervisor result verbatim. Neither a zero exit nor a candidate NPZ is approval.
    write_json(segment / "supervisor_result.json", outcome)
    accepted = None
    try:
        validate_run(run)
        accepted = validate_completion(segment, run, registration)
        status, reason = "completed", "registered_proposals_and_terminal_evidence_verified"
    except Exception as error:
        status, reason = "failed", f"{type(error).__name__}:{error}"
    assessment = {"run_id": run["run_id"], "segment_id": registration["segment_id"], "status": status,
                  "reason": reason, "accepted": accepted, "files_sha256": evidence_hashes(segment)}
    assessment["assessment_id"] = canonical(assessment)
    write_json(segment / "assessment.json", assessment)
    return assessment


def run_segment(directory, proposals, *, graph=None, resume=False):
    directory = Path(directory).resolve()
    with workspace_lock():
        if not resume:
            require(graph is not None, "A new generation requires --graph")
            run = register_run(graph)
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "run.json", run)
        else:
            run = read_json(directory / "run.json")
        validate_run(run)
        chain = scan_run(directory, run)
        segment, registration = register_segment(directory, run, chain, proposals)
        budget = registration["resources"]
        if budget["can_launch"]:
            outcome = supervise([sys.executable, "-m", "cnsbridge.learning.generate_rewiring", "--worker", str(segment)],
                                segment, rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                wall_seconds=600, poll_seconds=.1)
        else:
            outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None}
            write_json(segment / "result.json", outcome)
        return finalize_segment(segment, run, registration, outcome)


def accepted_candidate(directory):
    """Return the unique final asset only after the complete parent-approved chain.

    Later neural research loaders must use this gate, not a worker's NPZ path.
    This function checks identities and files, without loading a graph in its caller.
    """
    directory = Path(directory).resolve()
    run = read_json(directory / "run.json")
    validate_run(run)
    chain = scan_run(directory, run)
    require(chain["terminal"], "No parent-approved terminal research candidate")
    segment = Path(chain["checkpoint_path"]).parent
    return {"path": str(segment / "candidate.npz"), "run_id": run["run_id"],
            "checkpoint_id": chain["checkpoint_id"], "prefix": chain["prefix"],
            "graph_sha256": chain["state"]["graph_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "out", "run", "worker"):
        parser.add_argument("--" + name, type=Path, help=argparse.SUPPRESS if name == "worker" else None)
    parser.add_argument("--proposals", type=int, help="Explicit new proposal quota, 1..1000000; never an automatic loop")
    args = parser.parse_args(argv)
    if args.worker:
        if any(value is not None for value in (args.graph, args.out, args.run, args.proposals)):
            parser.error("--worker is exclusive")
        from .rewiring_worker import run_worker
        result = run_worker(args.worker)
    else:
        if args.proposals is None:
            parser.error("--proposals is required for every segment")
        if args.run:
            if args.graph or args.out:
                parser.error("Resume --run cannot include --graph or --out")
        elif not args.graph or not args.out:
            parser.error("New generation requires --graph and --out")
        try:
            result = run_segment(args.run or args.out, args.proposals, graph=args.graph, resume=args.run is not None)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
