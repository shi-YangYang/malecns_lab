"""Preregistered full-brain X2 training/evaluation; default preview is inert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("probe", "probe-stabilized", "readout", "joint", "evaluate"), default="probe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Selected new X2 checkpoint; never the old three-axis head")
    parser.add_argument("--feature-cache", type=Path, help="Explicit fixed readout cache; no neural recomputation fallback")
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--initialization", choices=("migrated", "neutral"), default="migrated")
    parser.add_argument("--spike-backward-gain", type=float, choices=(1.0, 0.1),
                        help="Explicit estimator; checkpoint continuation inherits its registered gain")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from malecns_lab.learning.x2_behavior_training import contract, execute, run_worker, inspect_checkpoint
    if args._worker is not None:
        if args.execute or args.output is not None or args.checkpoint is not None or args.feature_cache is not None:
            parser.error("Worker cannot include public execution options")
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    try:
        gain = args.spike_backward_gain
        if args.checkpoint is not None and gain is None and args.mode in ("readout", "joint", "evaluate"):
            gain = inspect_checkpoint(args.checkpoint)["policy"].get("backward", {}).get("spike_backward_gain", 1.0)
        cfg = contract(args.mode, seed=args.seed, max_updates=args.max_updates,
            train_episodes=args.train_episodes, validation_episodes=args.validation_episodes,
            wall_seconds=args.wall_seconds, initialization=args.initialization, spike_backward_gain=gain)
        if args.feature_cache is not None:
            from malecns_lab.learning.x2_behavior_cache import validate_request
            from malecns_lab.learning.x2_inference import ROOT
            validate_request(cfg, args.checkpoint, args.feature_cache, ROOT)
        if not args.execute:
            print(json.dumps({"mode": "preview", "brain_loaded": False, "execute": False,
                              "configuration": cfg, "feature_cache": str(args.feature_cache) if args.feature_cache else None, "checkpoint": str(args.checkpoint) if args.checkpoint else None}, indent=2))
            return 0
        if args.output is None:
            parser.error("--execute requires a new --output under project data/")
        result = execute(args.output, configuration=cfg, checkpoint_path=args.checkpoint, feature_cache=args.feature_cache)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({name: result[name] for name in ("status", "reason", "output", "mode", "behavior_pass")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
