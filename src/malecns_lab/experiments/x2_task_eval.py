"""Explicit, supervised five-scene frozen-brain task evaluation; preview is inert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the fixed offline 5-scene / 15-frame task screen")
    parser.add_argument("--output", type=Path, help="New directory beneath this project's data/ or logs/")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        if args.execute or args.output is not None:
            parser.error("Worker invocation cannot include public execution options")
        from malecns_lab.learning.x2_task_eval import run_worker
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        from malecns_lab.learning.x2_task_eval import protocol
        print(json.dumps({"mode": "preview", "execute": False, "brain_loaded": False, "protocol": protocol(),
                          "wall_limit_seconds": 240, "requested_rss_gib": 6, "system_reserve_gib": 1,
                          "requires": "GPU VM and CPU simulator stopped; shared complete-brain lock"}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires --output pointing to a new directory")
    try:
        from malecns_lab.learning.x2_task_eval import execute
        result = execute(args.output)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("status", "reason", "output", "behavior_pass")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
