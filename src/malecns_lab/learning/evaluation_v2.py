"""Scene-equal offline metrics and paired, family-cluster uncertainty."""

from __future__ import annotations

import numpy as np

from ..control.body import BODY_AXES
from .contracts_v2 import digest
from .dataset_v2 import DatasetV2


def score_predictions(dataset: DatasetV2, prediction, supported, *, split: str) -> dict:
    prediction, supported = np.asarray(prediction, float), np.asarray(supported)
    if prediction.shape != dataset.targets.shape or supported.shape != (6,) or supported.dtype != np.bool_:
        raise ValueError("Aligned N x 6 predictions and six-axis supported mask required")
    mask = dataset.scoring_mask(split)
    if np.any(mask & ~supported[None, :]):
        raise ValueError("A scored axis has no trained support")
    if not np.isfinite(prediction[mask]).all():
        raise ValueError("Missing predictions on the shared scoring manifest")
    task = dataset.protocol["task"]
    scenes = sorted({s["input_source"]["scene_id"] for s in dataset.segments if s["split"] == split})
    result = []
    for scene in scenes:
        scene_rows = np.array([row["scene_id"] == scene for row in dataset.rows], dtype=bool)
        segments = [s for s in dataset.segments if s["input_source"]["scene_id"] == scene and s["split"] == split]
        coverage = {field: sum(s["coverage"][field] for s in segments) for field in
                    ("planned_frames", "attempted_frames", "valid_input_frames", "failed_frames", "unprocessed_tail_frames",
                     "warmup_frames", "prediction_valid_frames")}
        coverage["eligible_unlabelled_frames"] = int(np.sum(scene_rows & dataset.split_rows(split) & dataset.eligible & ~dataset.mask.any(axis=1)))
        coverage["scored_frames"] = int(np.sum(scene_rows & mask.any(axis=1)))
        coverage["time_coverage"] = coverage["scored_frames"] / coverage["planned_frames"]
        failures = {}
        for segment in segments:
            for reason, count in segment["coverage"].get("failure_reasons", {}).items():
                failures[reason] = failures.get(reason, 0) + count
        coverage["failure_reasons"] = failures
        axis_metrics = {}
        for axis in (0, 5):
            name, values = BODY_AXES[axis], []
            per_seed = []
            for seed in sorted({s["run_seed"] for s in segments}):
                use = scene_rows & mask[:, axis] & np.array([row["run_seed"] == seed for row in dataset.rows])
                if not use.any():
                    per_seed.append({"run_seed": seed, "count": 0, "mae": None})
                    continue
                truth, pred = dataset.targets[use, axis], prediction[use, axis]
                error = np.abs(pred - truth)
                values.append(float(error.mean()))
                tol = task["stationary_tolerances"][name]
                moving, stopped = np.abs(truth) > tol, np.abs(truth) <= tol
                per_seed.append({"run_seed": seed, "count": int(use.sum()), "mae": float(error.mean()),
                                 "direction_error_fraction": float(np.mean(np.sign(pred[moving]) != np.sign(truth[moving]))) if moving.any() else None,
                                 "stationary_false_action_fraction": float(np.mean(np.abs(pred[stopped]) > tol)) if stopped.any() else None})
            axis_metrics[name] = {"mae": float(np.mean(values)) if values else None, "unit": "m/s" if axis == 0 else "rad/s",
                                  "per_input_seed": per_seed, "missing_seed_coverage": any(item["count"] == 0 for item in per_seed)}
        all_axes = all(axis_metrics[name]["mae"] is not None for name in ("vx", "wz"))
        primary = float(np.mean([axis_metrics[name]["mae"] / task["tolerances"][name] for name in ("vx", "wz")])) if all_axes else None
        result.append({"scene_id": scene, "cluster_id": segments[0]["cluster_id"], "axes": axis_metrics,
                       "normalized_mae": primary, "coverage": coverage})
    scores = [scene["normalized_mae"] for scene in result if scene["normalized_mae"] is not None]
    complete = bool(result) and len(scores) == len(result) and all(
        not scene["axes"][axis]["missing_seed_coverage"] for scene in result for axis in ("vx", "wz"))
    scoring_identity = [{"row_key": row["row_key"], "mask": mask[index].tolist(),
                         "target": np.where(mask[index], dataset.targets[index], 0).tolist()}
                        for index, row in enumerate(dataset.rows) if row["split"] == split]
    return {"split": split, "scenes": result, "scored_scene_count": len(scores), "registered_scene_count": len(result),
            "scoring_manifest_sha256": digest(scoring_identity),
            "scoreable_scene_fraction": len(scores) / len(result) if result else 0.0,
            "normalized_mae": float(np.mean(scores)) if scores else None,
            "coverage_status": "all_scenes_axes_seeds_scored" if complete else "insufficient_evidence",
            "aggregation": "within scene/axis: average each input-seed MAE equally; then axes/tolerances; then scenes equally",
            "scope": "offline labelled observations only; no flight, collision, trajectory or transfer evidence"}


def paired_scene_comparison(real: dict, control: dict, *, practical_delta: float, seed: int, replicates: int) -> dict:
    """CI conditions on these graph/input seeds; no frame-level pseudoreplication."""
    if practical_delta < 0 or not np.isfinite(practical_delta) or replicates < 1:
        raise ValueError("Registered delta and positive bootstrap repetitions required")
    left, right = {s["scene_id"]: s for s in real["scenes"]}, {s["scene_id"]: s for s in control["scenes"]}
    if real["split"] != control["split"] or set(left) != set(right):
        raise ValueError("Paired reports need identical split and registered scene sets")
    if not real.get("scoring_manifest_sha256") or real["scoring_manifest_sha256"] != control.get("scoring_manifest_sha256"):
        raise ValueError("Paired reports require identical row identities, supervision masks and targets")
    differences = []
    for scene in sorted(left):
        a, b = left[scene], right[scene]
        if a["cluster_id"] != b["cluster_id"] or a["coverage"] != b["coverage"]:
            raise ValueError("Paired conditions differ in scene cluster or scoring coverage")
        delta = None if a["normalized_mae"] is None or b["normalized_mae"] is None else b["normalized_mae"] - a["normalized_mae"]
        differences.append({"scene_id": scene, "cluster_id": a["cluster_id"], "control_minus_real": delta})
    valid = [row for row in differences if row["control_minus_real"] is not None]
    clusters = sorted({row["cluster_id"] for row in valid})
    result = {"differences": differences, "practical_delta": practical_delta, "independent_clusters": len(clusters),
              "mean_difference": float(np.mean([row["control_minus_real"] for row in valid])) if valid else None,
              "ci95": None, "interpretation": "insufficient_evidence", "bootstrap_seed": seed, "bootstrap_replicates": replicates,
              "uncertainty_scope": "conditional on registered graph/input seeds; graph variability requires separate reporting; not a p-value"}
    if len(clusters) < 2 or len(valid) != len(differences) or any(r["coverage_status"] != "all_scenes_axes_seeds_scored" for r in (real, control)):
        return result
    rng, samples = np.random.default_rng(seed), []
    grouped = {cluster: [row["control_minus_real"] for row in valid if row["cluster_id"] == cluster] for cluster in clusters}
    for _ in range(replicates):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        samples.append(np.mean([value for cluster in draw for value in grouped[cluster]]))
    ci = np.percentile(samples, [2.5, 97.5]).tolist()
    if ci[0] > practical_delta:
        interpretation = "practical_advantage_on_scored_data"
    elif ci[1] < practical_delta:
        interpretation = "no_confirmed_practical_advantage"
    else:
        interpretation = "insufficient_evidence"
    result.update(ci95=ci, interpretation=interpretation)
    return result
