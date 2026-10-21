"""Independent complete MaleCNS yaw-learning experiment commands."""
import argparse
import json
from pathlib import Path
from .protocol import register
from .training import run_gate, train_segment, evaluate, run_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="Register an immutable new joint/frozen run; does not execute a brain")
    reg.add_argument("--out", type=Path, required=True)
    reg.add_argument("--graph", type=Path, required=True)
    reg.add_argument("--mode", choices=("joint", "frozen"), required=True)
    gate = sub.add_parser("gate", help="Disposable full-graph GPU update; discarded before formal training")
    gate.add_argument("--run", type=Path, required=True)
    train = sub.add_parser("train", help="One supervised 1–4 window segment")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--windows", type=int, default=4)
    evaluation = sub.add_parser("evaluate", help="Fixed final models on validation or validation-gated locked test")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--frozen-run", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), required=True)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            run = register(args.out, args.graph, args.mode)
            result = {"status": "registered", "run_id": run["run_id"], "family_id": run["family_id"]}
        elif args.command == "gate":
            result = run_gate(args.run)
        elif args.command == "train":
            result = train_segment(args.run, args.windows)
        elif args.command == "evaluate":
            result = evaluate(args.run, args.frozen_run, args.split)
        else:
            result = run_worker(args.job)
    except Exception as error:
        result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    print(json.dumps({key: result[key] for key in ("status", "reason", "detail", "run_id", "family_id", "completed_windows", "split", "control_gate") if key in result}, indent=2))
    return 0 if result["status"] in ("completed", "registered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
