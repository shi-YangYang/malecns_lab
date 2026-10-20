"""One registered full-MaleCNS action-loss update; all neural work is supervised."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import psutil

from ..brain.metadata import canonical_hash
from .joint_probe_contract import (FINAL_SCHEMA, PROTOCOL, SCHEMA, graph_registration, require, select_data,
                                   sha256, sources)
from .joint_probe_history import history_binding
from .joint_probe_evidence import validate_joint_evidence
from .resource_guard import GIB, resource_budget, supervise, write_json


def register(graph, dataset, directory):
    """New directory, full byte/source bindings and exact inputs before worker launch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    select_data(dataset)  # Sealed dataset validation is light and never constructs a brain.
    graph_info = graph_registration(graph)
    implementation = sources()
    prior_run = history_binding(Path(dataset).resolve().parent / "joint_update_20261019_r1", implementation, graph_info)
    budget = resource_budget(psutil.virtual_memory().available, reserve_bytes=GIB, rss_bytes=6 * GIB)
    registration = {
        "schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL),
        "graph": graph_info, "dataset_path": str(Path(dataset).resolve()),
        "dataset_id": PROTOCOL["dataset_id"], "dataset_manifest_sha256": sha256(Path(dataset) / "manifest.json"),
        "implementation_sha256": implementation, "prior_run": prior_run,
        "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": 0.1},
        "environment": {"python": sys.version, "platform": platform.platform(), "psutil": psutil.__version__},
    }
    registration["registration_id"] = canonical_hash(registration)
    write_json(directory / "registration.json", registration)
    return registration


def finalize_run(directory, outcome):
    """Keep raw supervision separate; classify negative scientific outcomes honestly."""
    directory = Path(directory)
    write_json(directory / "supervisor_result.json", outcome)
    result = {"schema": FINAL_SCHEMA, "status": "failed", "protocol_complete": False,
              "joint_numeric_update": None, "internal_action_effect": None, "reason": outcome.get("reason"),
              "supervisor": outcome}
    if outcome.get("status") == "completed":
        try:
            evidence = validate_joint_evidence(directory, outcome)
            result.update(evidence, status="completed", reason="protocol_complete")
        except Exception as error:
            result.update(reason="invalid_joint_evidence", detail=f"{type(error).__name__}:{error}")
    result["terminal_artifacts_sha256"] = {name: sha256(directory / name) for name in
        ("registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json", "supervisor_result.json")
        if (directory / name).is_file()}
    write_json(directory / "result.json", result)
    return result


def run_registered(graph, dataset, directory):
    directory = Path(directory)
    # Never overwrite any old directory, even after a failed attempt.
    require(not directory.exists(), "Run output must be a new directory")
    try:
        registration = register(graph, dataset, directory)
    except Exception as error:
        directory.mkdir(parents=True, exist_ok=True)
        result = {"schema": FINAL_SCHEMA, "status": "failed", "protocol_complete": False,
                  "joint_numeric_update": None, "internal_action_effect": None, "reason": "registration_failed",
                  "detail": f"{type(error).__name__}:{error}"}
        write_json(directory / "result.json", result)
        return result
    budget = registration["resources"]
    if not budget["can_launch"]:
        outcome = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                   "peak_observed_rss_bytes": 0, "wall_seconds": 0.0, "returncode": None,
                   "worker_status": "not_launched", "last_phase": None}
    else:
        outcome = supervise([sys.executable, "-m", "malecns_lab.learning.joint_probe", "--worker", str(directory.resolve())],
                            directory, rss_limit_bytes=budget["effective_rss_limit_bytes"], wall_seconds=600,
                            reserve_bytes=GIB, poll_seconds=0.1)
    return finalize_run(directory, outcome)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if any(value is not None for value in (args.graph, args.dataset, args.out)):
            parser.error("--worker cannot be combined with a new registration")
        from .joint_probe_worker import run_worker  # All Torch imports are inside the supervised child.
        result = run_worker(args.worker)
    else:
        if args.graph is None or args.dataset is None or args.out is None:
            parser.error("--graph, --dataset and --out are required")
        result = run_registered(args.graph, args.dataset, args.out)
    print(json.dumps({k: result.get(k) for k in ("status", "reason", "protocol_complete", "joint_numeric_update",
                                                "internal_action_effect", "detail")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
