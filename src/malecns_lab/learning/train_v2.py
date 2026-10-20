"""Prepare or fit V2 offline readouts; never connects to a device or simulates a brain."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from .contracts_v2 import RIDGE_CANDIDATES, digest, file_digest, write_json
from .dataset_v2 import DatasetV2, load_dataset
from .decoder_v2 import MaskedRidgeV2
from .evaluation_v2 import paired_scene_comparison, score_predictions


def for_input_seed(dataset: DatasetV2, seed: int) -> DatasetV2:
    use = np.array([row["run_seed"] == seed for row in dataset.rows], dtype=bool)
    segments = [segment for segment in dataset.segments if segment["run_seed"] == seed]
    if not segments:
        raise ValueError("Input seed missing from registered segments")
    return DatasetV2(dataset.protocol, dataset.schemas, {kind: x[use] for kind, x in dataset.features.items()},
                     dataset.targets[use], dataset.mask[use], dataset.eligible[use],
                     [row for row, keep in zip(dataset.rows, use) if keep], segments, dataset.provenance)


def fit_readouts(dataset: DatasetV2, *, deadline: float | None = None) -> dict:
    """One graph AND input seed per fit. No statistics or selection use test rows."""
    seeds = {segment["run_seed"] for segment in dataset.segments}
    if len(seeds) != 1:
        raise ValueError("Fit each graph/input seed independently using for_input_seed")
    train = dataset.eligible & dataset.split_rows("train")
    validation = dataset.eligible & dataset.split_rows("val")
    if not train.any() or not validation.any():
        raise ValueError("Need eligible train and validation observations")
    if any(not np.any(dataset.mask[train, axis]) for axis in (0, 5)):
        raise ValueError("Main vx/wz comparison requires independent training labels for both axes")
    models, selection = {}, {}
    for representation in ("neural", "visual"):
        x, schema = dataset.features[representation], dataset.schemas[representation]
        candidates, best, best_score = [], None, float("inf")
        for ridge in RIDGE_CANDIDATES:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Registered learning wall budget exhausted; no candidates silently skipped")
            model = MaskedRidgeV2.fit(schema, x[train], dataset.targets[train], dataset.mask[train], ridge=ridge)
            prediction = model.predict_batch(x, schema=schema, prediction_valid=validation)
            metrics = score_predictions(dataset, prediction, model.supported, split="val")
            if metrics["coverage_status"] != "all_scenes_axes_seeds_scored":
                raise ValueError("Validation lacks scored axes/scenes; retain coverage and collect data, do not tune on a subset")
            candidates.append({"ridge": ridge, "validation": metrics})
            if metrics["normalized_mae"] < best_score:
                best, best_score = model, metrics["normalized_mae"]
        models[representation] = best
        selection[representation] = {"candidates": candidates, "selected_ridge": best.ridge,
                                     "criterion": "scene-equal normalized MAE on val only; first minimum wins",
                                     "candidate_count": len(RIDGE_CANDIDATES)}
    mean = np.zeros(6)
    supported = np.zeros(6, dtype=bool)
    for axis in (0, 5):
        use = train & dataset.mask[:, axis]
        mean[axis], supported[axis] = dataset.targets[use, axis].mean(), True
    active_splits = ("train", "val", "test") if dataset.protocol["phase"] == "confirmatory" else ("train", "val")
    valid = dataset.eligible & np.array([row["split"] in active_splits for row in dataset.rows])
    predictions = {name: model.predict_batch(dataset.features[name], schema=dataset.schemas[name], prediction_valid=valid)
                   for name, model in models.items()}
    for name, value in (("mean", mean), ("zero", np.zeros(6))):
        predictions[name] = np.full(dataset.targets.shape, np.nan)
        predictions[name][valid] = value
    metrics = {split: {name: score_predictions(dataset, prediction, supported, split=split)
                       for name, prediction in predictions.items()} for split in active_splits}
    comparisons = {}
    for split in active_splits:
        comparisons[split] = {name: paired_scene_comparison(metrics[split]["neural"], metrics[split][name],
                              practical_delta=dataset.protocol["practical_delta"],
                              seed=dataset.protocol["bootstrap"]["seed"], replicates=dataset.protocol["bootstrap"]["replicates"])
                              for name in ("visual", "mean", "zero")}
    # The training identity deliberately excludes test paths, labels and features.
    development = [segment for segment in dataset.segments if segment["split"] in ("train", "val")]
    training_identity = {"schema": dataset.schemas["neural"].as_dict(), "task": dataset.protocol["task"],
                         "ridge_candidates": list(RIDGE_CANDIDATES), "development_segments": development,
                         "input_seed": next(iter(seeds)), "stage": "A_frozen_brain_readout_only"}
    return {"models": models, "selection": selection, "mean_target": mean.tolist(), "supported_axes": ["vx", "wz"],
            "predictions": predictions, "metrics": metrics, "comparisons": comparisons,
            "training_identity": training_identity, "training_id": digest(training_identity),
            "active_splits": list(active_splits)}


def save_fit(out: Path, dataset: DatasetV2, result: dict):
    out.mkdir(parents=True, exist_ok=False)
    for name, model in result["models"].items():
        model.save(out / f"model_{name}.json")
    selected = np.array([row["split"] in result["active_splits"] for row in dataset.rows], dtype=bool)
    arrays = {"targets": dataset.targets[selected], "supervised_mask": dataset.mask[selected],
              "prediction_valid": dataset.eligible[selected],
              "row_key": np.asarray([row["row_key"] for row in dataset.rows])[selected],
              **{f"prediction_{name}": prediction[selected] for name, prediction in result["predictions"].items()}}
    with (out / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(out / "rows.json", [row for row, use in zip(dataset.rows, selected) if use])
    write_json(out / "fit_report.json", {key: value for key, value in result.items() if key not in ("models", "predictions")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Predeclared cnsbridge.learning_protocol.v2 JSON")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrite")
    parser.add_argument("--prepare-only", action="store_true", help="Validate provenance/splits/labels and save coverage without fitting")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        # Snapshot registration before touching any held-out artifacts or computing metrics.
        registered = json.loads(args.protocol.read_text(encoding="utf-8"))
        write_json(args.out / "registration.json", {"protocol": registered, "sha256": file_digest(args.protocol),
                                                    "claim": "user predeclared protocol; not proof test data were previously unseen"})
        dataset = load_dataset(args.protocol)
        write_json(args.out / "dataset_manifest.json", dataset.provenance)
        write_json(args.out / "coverage.json", [{"segment": item["segment"], "source": item["input_source"],
                                                "split": item["split"], "run_seed": item["run_seed"], "coverage": item["coverage"]}
                                               for item in dataset.segments])
        seeds = sorted({segment["run_seed"] for segment in dataset.segments})
        if not args.prepare_only:
            for seed in seeds:
                current = for_input_seed(dataset, seed)
                result = fit_readouts(current, deadline=started + registered["budget_seconds"])
                if time.perf_counter() >= started + registered["budget_seconds"]:
                    raise TimeoutError("Registered wall budget exhausted before completed fit persistence")
                save_fit(args.out / f"input_seed_{seed}", current, result)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        source_files = ("contracts_v2.py", "dataset_v2.py", "decoder_v2.py", "evaluation_v2.py", "train_v2.py")
        write_json(args.out / "run_summary.json", {"format": "cnsbridge.learning_run.v2", "status": "prepared" if args.prepare_only else "complete",
                    "phase": registered["phase"], "input_seeds": seeds, "wall_seconds": time.perf_counter() - started,
                    "peak_process_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
                    "code_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in source_files},
                    "test_evaluated": not args.prepare_only and registered["phase"] == "confirmatory",
                    "scope": "offline stage A tools/results only; no proof of biological cognition, flight, transfer or connectome advantage"})
        print(f"Saved {args.out}; {'prepared without fitting' if args.prepare_only else 'separate fits for each input seed'}")
    except BaseException as error:
        write_json(args.out / "run_failed.json", {"error": repr(error), "wall_seconds": time.perf_counter() - started,
                                                 "status": "failed; partial artifacts are not completed results"})
        raise


if __name__ == "__main__":
    main()
