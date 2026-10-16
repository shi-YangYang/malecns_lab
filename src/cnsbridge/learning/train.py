"""Train and evaluate on separate scenes; save an offline candidate only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cnsbridge.learning.dataset import MotionDataset, concatenate_datasets, evaluate_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--test-scenes", nargs="+", required=True, help="Entire scene IDs held out from fitting")
    parser.add_argument("--ridge", type=float, default=1.0, help="Choose before examining held-out results")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for candidate and report")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error("--out-dir already exists; use a new experiment directory")
    try:
        datasets = [MotionDataset.load(path) for path in args.datasets]
        policies = {data.metadata.get("policy_id") for data in datasets}
        if len(policies) != 1 or None in policies:
            raise ValueError("Datasets must declare the same desired-action policy_id")
        data = concatenate_datasets(datasets)
        model, report = evaluate_split(data, tuple(args.test_scenes), args.ridge)
        report["dataset_files"] = [str(path.resolve()) for path in args.datasets]
        report["policy_id"] = next(iter(policies))
        report["status"] = "offline_candidate_not_validated_for_actuation"
        report["limitations"] = [
            "Grounded imitation error does not measure collision avoidance or closed-loop stability.",
            "The visual baseline uses the full Retina grids but may have a different parameter count.",
            "No random-connectome control or physical device-transfer experiment has been performed.",
            "Do not tune on the held-out scene; collect new evaluation scenes after model selection.",
        ]
        args.out_dir.mkdir(parents=True, exist_ok=False)
        model.save(args.out_dir / "candidate.json")
        with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        print(f"OFFLINE CANDIDATE: {(args.out_dir / 'candidate.json').resolve()}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"TRAINING FAILED: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
