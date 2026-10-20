"""Read immutable V2 neural replay segments with causal labels and split guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts_v2 import FeatureSchemaV2, digest, file_digest, load_labels, nonempty, target_at, validate_protocol


@dataclass
class DatasetV2:
    protocol: dict
    schemas: dict[str, FeatureSchemaV2]
    features: dict[str, np.ndarray]
    targets: np.ndarray
    mask: np.ndarray
    eligible: np.ndarray
    rows: list[dict]
    segments: list[dict]
    provenance: dict

    def split_rows(self, split: str) -> np.ndarray:
        return np.array([row["split"] == split for row in self.rows], dtype=bool)

    def scoring_mask(self, split: str) -> np.ndarray:
        return self.mask & (self.eligible & self.split_rows(split))[:, None]

    def row_keys(self, split: str) -> list[str]:
        return [row["row_key"] for row, keep in zip(self.rows, self.split_rows(split)) if keep]


def _split_guard(segments: list[dict]):
    assignments = {}
    for segment in segments:
        source, split = segment["input_source"], segment["split"]
        for key in ("scene_id", "session_id", "family_id", "source_id", "video_sha256"):
            value = nonempty(source.get(key), f"input_source.{key}")
            token = (key, value)
            if token in assignments and assignments[token] != split:
                raise ValueError(f"Split leakage across {key}={value}")
            assignments[token] = split


def scene_clusters(segments: list[dict]) -> dict[str, str]:
    """Connected scene components sharing a session, family or source content."""
    parents = {segment["input_source"]["scene_id"]: segment["input_source"]["scene_id"] for segment in segments}

    def root(item):
        while parents[item] != item:
            item = parents[item]
        return item

    seen = {}
    for segment in segments:
        source, scene = segment["input_source"], segment["input_source"]["scene_id"]
        for field in ("session_id", "family_id", "source_id", "video_sha256"):
            token = (field, source[field])
            if token in seen:
                parents[root(scene)] = root(seen[token])
            seen[token] = scene
    return {scene: root(scene) for scene in parents}


def load_dataset(protocol_path: str | Path) -> DatasetV2:
    protocol_path = Path(protocol_path).resolve()
    protocol = validate_protocol(json.loads(protocol_path.read_text(encoding="utf-8")))
    base = protocol_path.parent
    rows, segments, neural, visual, targets, masks, eligible = [], [], [], [], [], [], []
    schemas, seen_paths, seen_keys, labels_per_source = {}, set(), set(), {}
    for entry in protocol["segments"]:
        path = (base / entry["path"]).resolve()
        if path in seen_paths:
            raise ValueError("Duplicate segment path")
        seen_paths.add(path)
        metadata_path, run_path, arrays_path = path / "metadata.json", path.parent / "run_manifest.json", path / "rows.npz"
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if meta.get("format") != "cnsbridge.neural_activity.v2" or run.get("format") != "cnsbridge.neural_replay_run.v2":
            raise ValueError("Only V2 replay artifacts are accepted")
        if meta["source_id"] != run["model_source_id"] or digest(run["model_provenance"]) != run["model_source_id"]:
            raise ValueError("Neural model provenance/source hash mismatch")
        if meta["segment"] not in run["planned_segments"] or meta["input_source"] not in run["input_sources"]:
            raise ValueError("Segment/source missing from predeclared replay run")
        source = meta["input_source"]
        if source.get("kind") != "recorded_video":
            raise ValueError("Generated diagnostic stimuli are not action-training recordings")
        if digest({k: v for k, v in source.items() if k != "source_id"}) != source["source_id"]:
            raise ValueError("Input source metadata digest mismatch")
        if meta["run_seed"] != run["run_seed"]:
            raise ValueError("Run seed differs between segment and run")
        provenance = run["model_provenance"]
        if provenance["time"]["simulation_window_ms"] != 50 or provenance["time"]["target_horizon_s"] != 0.05:
            raise ValueError("Replay time contract changed")
        for kind in ("neural", "visual"):
            candidate = FeatureSchemaV2(run["model_source_id"], tuple(provenance["features"][f"{kind}_names"]), kind)
            if kind in schemas and schemas[kind] != candidate:
                raise ValueError("Mixed graph/feature identities; fit each graph condition separately")
            schemas[kind] = candidate
        label_path = (base / entry["labels"]).resolve()
        labels, label_meta = load_labels(label_path, source_id=source["source_id"], task=protocol["task"],
                                         legacy=entry.get("label_format") == "legacy_v1_intervals")
        source_key = source["source_id"]
        if source_key in labels_per_source and labels_per_source[source_key] != label_meta["file_sha256"]:
            raise ValueError("Same source has inconsistent labels across replays/seeds")
        labels_per_source[source_key] = label_meta["file_sha256"]
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in ("frame_index", "input_cutoff_s", "target_time_s", "valid",
                      "prediction_valid", "target_available", "neural_history_hz", "visual_history", "simulation_end_ms")}
        count = len(arrays["frame_index"])
        segment = meta["segment"]
        start, end = segment["start_frame"], segment["end_frame"]
        if not np.array_equal(arrays["frame_index"], np.arange(start, start + count)) or count > end - start:
            raise ValueError("Replay rows must be a contiguous prefix of the predeclared segment")
        for field in ("valid", "prediction_valid", "target_available"):
            if arrays[field].shape != (count,) or arrays[field].dtype != np.bool_:
                raise ValueError(f"Invalid replay {field} mask")
        for field in ("input_cutoff_s", "target_time_s", "simulation_end_ms"):
            if arrays[field].shape != (count,) or not np.isfinite(arrays[field]).all():
                raise ValueError(f"Invalid replay {field}")
        invalid_indices = np.flatnonzero(~arrays["valid"])
        if len(invalid_indices) and (len(invalid_indices) != 1 or invalid_indices[0] != count - 1):
            raise ValueError("Failed replay must stop at its first invalid row")
        if not np.allclose(arrays["simulation_end_ms"][arrays["valid"]],
                           (np.arange(count)[arrays["valid"]] + 1) * 50, rtol=0, atol=1e-8):
            raise ValueError("Valid replay windows must begin at reset and advance by 50 ms")
        if not np.allclose(arrays["input_cutoff_s"], arrays["frame_index"] / 20, rtol=0, atol=1e-10):
            raise ValueError("Input cutoff differs from frame playback time")
        if not np.allclose(arrays["target_time_s"], (arrays["frame_index"] + 1) / 20, rtol=0, atol=1e-10):
            raise ValueError("Target horizon mismatch; do not relabel at current frame time")
        if not np.array_equal(arrays["target_available"], arrays["frame_index"] + 1 < end):
            raise ValueError("Target availability differs from segment boundary")
        expected = arrays["valid"] & (arrays["simulation_end_ms"] > 2000) & arrays["target_available"]
        if not np.array_equal(arrays["prediction_valid"], expected):
            raise ValueError("Prediction validity inconsistent with warmup/input/target boundary")
        for field, size in (("neural_history_hz", 132), ("visual_history", 98)):
            if arrays[field].shape != (count, size) or not np.isfinite(arrays[field][arrays["valid"]]).all():
                raise ValueError(f"Invalid replay {field} shape or valid-row values")
        if np.any(arrays["neural_history_hz"][arrays["valid"]] < 0):
            raise ValueError("Neural histories must be nonnegative Hz")
        if np.any((arrays["visual_history"][arrays["valid"]] < 0) | (arrays["visual_history"][arrays["valid"]] > 1)):
            raise ValueError("Visual histories must lie in [0,1]")
        coverage = meta["coverage"]
        if (coverage["planned_frames"] != end - start or coverage["attempted_frames"] != count
                or coverage["unprocessed_tail_frames"] != end - start - count
                or coverage["prediction_valid_frames"] != int(expected.sum())
                or coverage["valid_input_frames"] != int(arrays["valid"].sum())
                or coverage["failed_frames"] != int((~arrays["valid"]).sum())
                or coverage["warmup_frames"] != int(np.sum(arrays["valid"] & (arrays["simulation_end_ms"] <= 2000)))):
            raise ValueError("Replay coverage does not match rows")
        segment_record = {"path": str(path), "split": entry["split"], "input_source": source, "run_seed": meta["run_seed"],
                          "segment": segment, "coverage": coverage, "labels": label_meta,
                          "hashes": {"metadata": file_digest(metadata_path), "run_manifest": file_digest(run_path),
                                     "rows": file_digest(arrays_path)}, "model_source_id": meta["source_id"]}
        segments.append(segment_record)
        for i, frame in enumerate(arrays["frame_index"]):
            y, mask, reason = target_at(labels, float(arrays["target_time_s"][i]))
            key_fields = {**{field: source[field] for field in ("source_id", "scene_id", "session_id", "family_id", "video_sha256")},
                          "segment_start": start, "segment_end": end, "frame_index": int(frame),
                          "repetition": segment["repetition"], "run_seed": meta["run_seed"]}
            key = digest(key_fields)
            if key in seen_keys:
                raise ValueError("Duplicate material/segment/frame/seed row")
            seen_keys.add(key)
            rows.append({**key_fields, "row_key": key, "split": entry["split"],
                         "input_cutoff_s": float(arrays["input_cutoff_s"][i]),
                         "target_time_s": float(arrays["target_time_s"][i]), "label_reason": reason,
                         "prediction_valid": bool(expected[i])})
            targets.append(y)
            masks.append(mask)
        neural.extend(arrays["neural_history_hz"])
        visual.extend(arrays["visual_history"])
        eligible.extend(expected)
    _split_guard(segments)
    clusters = scene_clusters(segments)
    for row in rows:
        row["cluster_id"] = clusters[row["scene_id"]]
    for segment in segments:
        segment["cluster_id"] = clusters[segment["input_source"]["scene_id"]]
    return DatasetV2(protocol, schemas, {"neural": np.asarray(neural).reshape(-1, 132), "visual": np.asarray(visual).reshape(-1, 98)},
                     np.asarray(targets).reshape(-1, 6), np.asarray(masks, dtype=bool).reshape(-1, 6), np.asarray(eligible, dtype=bool),
                     rows, segments, {"protocol_sha256": file_digest(protocol_path), "protocol": protocol, "segments": segments})


def assert_paired_rows(left: DatasetV2, right: DatasetV2, split: str):
    """Fail rather than silently intersect different graph-condition coverage."""
    li, ri = left.split_rows(split), right.split_rows(split)
    if left.row_keys(split) != right.row_keys(split):
        raise ValueError("Paired conditions require identical ordered row identities")
    if not np.array_equal(left.eligible[li], right.eligible[ri]) or not np.array_equal(left.mask[li], right.mask[ri]):
        raise ValueError("Paired conditions differ in validity or supervision coverage")
    if not np.array_equal(left.targets[li], right.targets[ri]):
        raise ValueError("Paired conditions have different independent target labels")
