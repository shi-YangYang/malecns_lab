"""Supervised fixed window159->160 MPS gradient diagnosis (no training)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import psutil

from ..brain.metadata import canonical_hash
from .resource_guard import GIB, resource_budget, supervise, write_json
from .train_snn import workspace_lock
from .training_contract import read_json, require, sha256
from .gradient_probe_evidence import (PROTOCOL, SOURCE_RUN, load_anchor, source_hashes, validate_completed, validate_registration)


def run_probe(directory):
    directory = Path(directory).resolve()
    require(not directory.exists(), "Diagnostic output must be a new directory")
    require(directory != SOURCE_RUN and SOURCE_RUN not in directory.parents, "Diagnostic cannot write inside its source run")
    with workspace_lock():
        directory.mkdir(parents=True, exist_ok=False)
        try:
            _, _, _, _, _, _, anchor = load_anchor()
            resources = resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB)
            reg = {"protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "anchor": anchor,
                   "registered_utc": datetime.now(timezone.utc).isoformat(), "implementation_sha256": source_hashes(),
                   "resources": {**resources, "wall_limit_seconds": 600, "sampling_seconds": .1},
                   "parent_pid": os.getpid(), "clock_monotonic": time.monotonic()}
            reg["registration_id"] = canonical_hash(reg)
            write_json(directory / "registration.json", reg)
            if resources["can_launch"]:
                outcome = supervise([sys.executable, "-m", "malecns_lab.learning.gradient_probe", "--worker", str(directory)],
                                    directory, rss_limit_bytes=resources["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                    wall_seconds=600, poll_seconds=.1)
            else:
                outcome = {"status": "failed", "reason": "resource_preflight_failed", "returncode": None}
            write_json(directory / "supervisor_result.json", outcome)
            result = {"status": "failed", "reason": outcome["reason"], "registration_id": reg["registration_id"],
                      "supervisor": outcome, "role": "diagnostic_not_training", "optimizer_updates": 0}
            try:
                _, previous, final, selected, old, event, anchor = load_anchor()
                validate_registration(reg, anchor)
                require(outcome["status"] == "completed", "Diagnostic worker did not complete")
                worker = validate_completed(directory, reg, final, old, event)
                from .training_contract import read_arrays
                import numpy as np
                resume = read_arrays(directory / "resume.npz")
                restored = read_json(directory / "resume.json")
                require(restored["parent_checkpoint_id"] == previous["checkpoint_id"]
                        and restored["exact"] is True and set(resume) == set(previous["arrays"])
                        and all(resume[k].dtype == v.dtype and np.array_equal(resume[k], v)
                                for k, v in previous["arrays"].items()), "Complete restore evidence differs")
                actual_selection = read_arrays(directory / "selection.npz")
                require(set(actual_selection) == set(selected) and all(np.array_equal(actual_selection[k], v)
                        and actual_selection[k].dtype == v.dtype for k, v in selected.items()), "Fixed inputs changed")
                result.update(status="completed", reason="repeatability_and_fixed_horizon_diagnostic_verified",
                              completed_branches=worker["completed_branches"])
            except Exception as error:
                result.update(reason="diagnostic_evidence_failed", detail=f"{type(error).__name__}:{error}")
            result["terminal_artifacts_sha256"] = {name: sha256(directory / name) for name in
                ("registration.json", "worker.claim", "worker_result.json", "worker.log", "resources.csv", "mps_memory.csv",
                 "mps_budget.json", "phase.json", "supervisor_result.json") if (directory / name).exists()}
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
        write_json(directory / "result.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        from .gradient_probe_worker import run_worker
        result = run_worker(args.worker)
    else:
        try:
            result = run_probe(args.out)
        except Exception as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
