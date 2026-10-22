"""Explicit, supervised full-reference inference; preview/help are stdlib-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run one bounded frozen full-brain baseline")
    parser.add_argument("--output", type=Path, help="A new directory beneath this project's data/ or logs/")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        if args.execute or args.output is not None:
            parser.error("Worker invocation cannot include public execution options")
        from malecns_lab.learning.x2_inference import run_worker
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode": "preview", "execute": False, "brain_loaded": False,
                          "inputs_m": [2.0, 1.5, 1.0], "new_scene_blank_steps": 600,
                          "steps_per_frame": 500, "wall_limit_seconds": 240,
                          "requested_rss_gib": 6, "system_reserve_gib": 1,
                          "requires": "GPU VM and CPU simulator stopped; shared complete-brain lock",
                          "scope": "frozen offline inference only; no ROS, training or robot motion"}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires --output pointing to a new directory")
    try:
        from malecns_lab.learning.x2_inference import execute
        result = execute(args.output)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "reason", "output")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
