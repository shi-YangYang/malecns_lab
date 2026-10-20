"""Versioned, fail-closed diagnostic records and scene coverage; no action labels."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ARRAY_FIELDS = {"neural_counts": (33, np.int32, -1), "neural_rates_hz": (33, np.float64, np.nan),
                "neural_history_hz": (132, np.float64, np.nan), "visual_features": (14, np.float64, np.nan),
                "visual_history": (98, np.float64, np.nan), "input_rates_hz": (14, np.float64, np.nan),
                "input_event_counts": (14, np.int64, -1)}
EYE_ARRAYS = ("brightness_grid", "centers_px", "flow_px_s", "texture", "min_eigenvalue", "condition_number", "flow_valid")
EYE_SCALARS = ("expansion_x_s_inv", "expansion_y_s_inv", "expansion_s_inv", "expansion_ema_s_inv", "loom", "loom_speed_s_inv")


def write_json(path: Path, data: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def pack_rows(rows: list[dict], group_count: int) -> dict:
    arrays = {}
    for name, (size, dtype, fill) in ARRAY_FIELDS.items():
        arrays[name] = np.array([row.get(name, np.full(size, fill)) for row in rows], dtype=dtype).reshape(len(rows), size)
    arrays["group_rates_hz"] = np.array([row.get("group_rates_hz", np.full(group_count, np.nan)) for row in rows]).reshape(len(rows), group_count)
    for name in ("frame_index", "valid", "prediction_valid", "target_available", "input_cutoff_s", "target_time_s",
                 "simulation_start_ms", "simulation_end_ms", "processing_ms", "host_dequeue_time_s"):
        dtype = bool if name in ("valid", "prediction_valid", "target_available") else (np.int64 if name == "frame_index" else np.float64)
        arrays[name] = np.asarray([row[name] for row in rows], dtype=dtype)
    arrays["initialization"] = np.array([bool(row.get("vision") and row["vision"].initialization) for row in rows])
    arrays["invalid_reason"] = np.asarray([row["vision"].invalid_reason or "" if row.get("vision") else row.get("error", "") for row in rows], dtype=str)
    arrays["invalid_eye"] = np.asarray([row["vision"].invalid_eye or "" if row.get("vision") else "" for row in rows], dtype=str)
    for side in "LR":
        for name in EYE_ARRAYS:
            shape = (6, 8, 2) if name in ("centers_px", "flow_px_s") else (6, 8)
            dtype, fill = (bool, False) if name == "flow_valid" else (np.float64, np.nan)
            values = []
            for row in rows:
                eye = row["vision"].eyes.get(side) if row.get("vision") else None
                values.append(getattr(eye, name) if eye else np.full(shape, fill))
            arrays[f"{side}_{name}"] = np.asarray(values, dtype=dtype).reshape((len(rows), *shape))
        for name in EYE_SCALARS:
            values = []
            for row in rows:
                eye = row["vision"].eyes.get(side) if row.get("vision") else None
                value = getattr(eye, name) if eye else None
                values.append(np.nan if value is None else value)
            arrays[f"{side}_{name}"] = np.asarray(values, dtype=np.float64)
    for name in ("spikes", "active_neurons", "silent_fraction", "high_rate_fraction", "rate_p50_hz", "rate_p99_hz", "max_rate_hz"):
        arrays[f"network_{name}"] = np.array([row.get("network", {}).get(name, np.nan) for row in rows], dtype=np.float64)
    return arrays


def source_coverage(summaries: list[dict]) -> list[dict]:
    """Account for the whole source, including intentionally unregistered gaps."""
    grouped = {}
    for summary in summaries:
        segment, source = summary["segment"], summary["input_source"]
        key = (segment.get("source_index", 0), segment.get("repetition", 0))
        row = grouped.setdefault(key, {
            "source_index": key[0], "repetition": key[1], "scene_id": source["scene_id"],
            "session_id": source["session_id"], "family_id": source["family_id"],
            "source_frames": source["total_frames"], "predeclared_frames": 0,
            "valid_input_frames": 0, "failed_frames": 0, "unprocessed_tail_frames": 0, "prediction_valid_frames": 0,
        })
        coverage = summary["coverage"]
        row["predeclared_frames"] += coverage["planned_frames"]
        for name in ("valid_input_frames", "failed_frames", "unprocessed_tail_frames", "prediction_valid_frames"):
            row[name] += coverage[name]
    for row in grouped.values():
        row["outside_predeclared_segments"] = row["source_frames"] - row["predeclared_frames"]
        row["valid_fraction_of_whole_source"] = row["valid_input_frames"] / row["source_frames"]
        row["prediction_fraction_of_whole_source"] = row["prediction_valid_frames"] / row["source_frames"]
    return list(grouped.values())


def run_segment(engine, source, segment: dict, out_dir: Path, *, deadline: float | None = None) -> dict:
    """Preserve valid prefix, failure row and unprocessed tail; never auto-restart."""
    out_dir.mkdir(parents=True, exist_ok=False)
    engine.reset()
    rows, status, exception = [], "complete", None
    start, end = segment["start_frame"], segment["end_frame"]
    began = time.perf_counter()
    for index, image, host_time in source.frames(start, end):
        if deadline is not None and time.perf_counter() >= deadline:
            status = "budget_exhausted"
            break
        frame_started = time.perf_counter()
        sim_start = engine.net.t_ms
        try:
            row = engine.step(image, index / 20, target_available=index + 1 < end)
        except (ValueError, FloatingPointError, RuntimeError) as error:
            exception = repr(error)
            row = {"valid": False, "prediction_valid": False, "target_available": index + 1 < end,
                   "input_cutoff_s": index / 20, "target_time_s": (index + 1) / 20,
                   "simulation_start_ms": sim_start, "simulation_end_ms": engine.net.t_ms,
                   "processing_ms": (time.perf_counter() - frame_started) * 1000, "error": exception}
        row.update(frame_index=index, host_dequeue_time_s=host_time)
        rows.append(row)
        if not row["valid"]:
            status = "segment_stopped"
            break
    arrays = pack_rows(rows, len(engine.group_names))
    planned = end - start
    valid = int(arrays["valid"].sum())
    failures = {str(reason): int(count) for reason, count in zip(*np.unique(arrays["invalid_reason"][~arrays["valid"].astype(bool)], return_counts=True))}
    processing = arrays["processing_ms"]
    summary = {
        "format": "cnsbridge.neural_activity.v2", "status": status, "source_id": engine.source_id,
        "input_source": source.metadata, "segment": segment, "run_seed": engine.run_seed,
        "source_code": "model and preprocessing code hashes in parent run_manifest.json",
        "prediction_contract": "features only; no learned action decoder or action labels are present",
        "groups": list(engine.group_names), "input_groups": list(engine.encoder.groups),
        "input_neuron_counts": engine.encoder.neuron_counts.tolist(),
        "coverage": {"planned_frames": planned, "attempted_frames": len(rows), "valid_input_frames": valid,
                     "failed_frames": len(rows) - valid, "failure_reasons": failures,
                     "unprocessed_tail_frames": planned - len(rows), "warmup_frames": int(np.sum(arrays["valid"] & (arrays["simulation_end_ms"] <= 2000))),
                     "prediction_valid_frames": int(arrays["prediction_valid"].sum()),
                     "valid_input_fraction": valid / planned, "prediction_valid_fraction": float(arrays["prediction_valid"].sum() / planned),
                     "unprocessed_tail_is_not_measured_failure": True},
        "computation": {"wall_seconds": time.perf_counter() - began,
                        "processing_p50_ms": float(np.percentile(processing, 50)) if len(rows) else None,
                        "processing_p95_ms": float(np.percentile(processing, 95)) if len(rows) else None,
                        "simulation_seconds": engine.net.t_ms / 1000},
        "neural_summary": engine.segment_summary(), "exception": exception,
        "array_missing_values": "NaN for uncomputed floating values; -1 for uncomputed counts; use valid mask",
        "flow_quality": "per-eye flow_valid masks; initialization is not measured zero motion",
    }
    with (out_dir / "rows.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (out_dir / "neuron_totals.npz").open("xb") as stream:
        np.savez_compressed(stream, body_ids=engine.graph.body_ids, counts=engine.neuron_total_counts)
    write_json(out_dir / "metadata.json", summary)
    return summary
