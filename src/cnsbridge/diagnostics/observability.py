"""Register and supervise a complete-graph paired observation; no device commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import psutil

from .observability_protocol import (REGISTRATION_SCHEMA, canonical_hash, implementation_hashes, ledger,
                                     read_json, template, validate_protocol)
from .observability_evidence import validate_evidence, validate_run
from ..learning.resource_guard import resource_budget, supervise, write_json


def register(graph: Path, protocol_path: Path, directory: Path) -> dict:
    protocol = validate_protocol(read_json(protocol_path))
    resources = protocol["resources"]
    budget = resource_budget(psutil.virtual_memory().available, rss_bytes=resources["rss_bytes"],
                             reserve_bytes=resources["reserve_bytes"])
    registration = {"schema": REGISTRATION_SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(),
                    "graph_path": str(graph.resolve()), "protocol": protocol, "protocol_id": canonical_hash(protocol),
                    "ledger": ledger(protocol), "budget": budget,
                    "environment": {"python": sys.version, "platform": platform.platform(), "psutil": psutil.__version__},
                    "implementation_sha256": implementation_hashes()}
    registration["registration_id"] = canonical_hash(registration)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "registration.json", registration)
    return registration


def finalize_run(directory: Path, outcome: dict) -> dict:
    """Keep the original supervisor evidence; a clean exit alone never completes."""
    write_json(directory / "supervisor.json", outcome)
    if outcome.get("status") != "completed":
        result = {"schema": "cnsbridge.observability_failure.v1", "status": "incomplete",
                  "reason": outcome.get("reason", "supervision_failed")}
    else:
        try:
            result = validate_evidence(directory)
        except Exception as error:
            result = {"schema": "cnsbridge.observability_failure.v1", "status": "incomplete",
                      "reason": "invalid_observation_evidence", "detail": f"{type(error).__name__}:{error}"}
    write_json(directory / "result.json", result)
    return result


def run_registered(graph: Path, protocol: Path, directory: Path) -> dict:
    registration = register(graph, protocol, directory)
    budget, resources = registration["budget"], registration["protocol"]["resources"]
    if not budget["can_launch"]:
        outcome = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                   "peak_observed_rss_bytes": 0, "wall_seconds": 0.0, "returncode": None,
                   "worker_status": "not_launched", "last_phase": None}
    else:
        outcome = supervise([sys.executable, "-m", "cnsbridge.diagnostics.observability", "--worker", str(directory.resolve())],
                            directory, rss_limit_bytes=budget["effective_rss_limit_bytes"],
                            wall_seconds=resources["wall_seconds"], reserve_bytes=resources["reserve_bytes"],
                            poll_seconds=resources["poll_seconds"])
    return finalize_run(directory, outcome)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="Verified complete full_reference NPZ")
    parser.add_argument("--protocol", type=Path, help="Explicit protocol file, frozen before this run")
    parser.add_argument("--out", type=Path, help="New output directory; never overwritten")
    parser.add_argument("--write-template", type=Path, help="Write candidate protocol; does not execute it")
    parser.add_argument("--validate-run", type=Path, help="Read-only evidence validation; no Torch or graph load")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    modes = sum(bool(value) for value in (args.write_template, args.validate_run, args.worker))
    if modes > 1 or (modes and any((args.graph, args.protocol, args.out))):
        parser.error("Use exactly one operation")
    try:
        if args.write_template:
            with args.write_template.open("x") as stream:
                json.dump(template(), stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            print(f"Candidate protocol written: {args.write_template}; not executed.")
            return 0
        if args.validate_run:
            result = validate_run(args.validate_run)
        elif args.worker:
            from .observability_collect import run_worker
            result = run_worker(args.worker)
            print(json.dumps({"status": result["status"], "reason": result.get("reason")}, indent=2))
            return 0 if result["status"] == "collected" else 1
        else:
            if not all((args.graph, args.protocol, args.out)):
                parser.error("New runs require --graph, --protocol and --out")
            result = run_registered(args.graph, args.protocol, args.out)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "protocol_complete" else 1
    except Exception as error:
        print(f"OBSERVABILITY INCOMPLETE: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
