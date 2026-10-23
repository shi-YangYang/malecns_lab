"""Opt-in supervised full-MaleCNS worker for the X2 virtual-marker loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="Record bounded actual-spike telemetry")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        if args.execute or args.output or args.sim_only or args.telemetry:
            parser.error("Internal worker cannot take public execution flags")
        from malecns_lab.learning.x2_session import run_worker
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode": "preview", "brain_loaded": False, "ros_connected": False,
                          "sensor": "live odometry virtual marker", "rounds": 2, "telemetry": args.telemetry,
                          "forward_threshold": .03, "pulse_seconds": 1., "wall_seconds": 240,
                          "brain_timeout_seconds": 30., "observation_timeout_seconds": 45., "pose_timeout_seconds": 40.,
                          "initial_requirement": "GPU VM and CPU simulator stopped; coordinator starts VM after ready"}))
        return 0
    if not args.sim_only or args.output is None:
        parser.error("--execute requires --sim-only and a new --output beneath project data/")
    try:
        from malecns_lab.learning.x2_session import execute
        result = execute(args.output, telemetry=args.telemetry)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
