"""Export existing V2 diagnostics and soma coordinates without running a brain.

The reference archive is opened lazily: only ``body_ids`` and ``meta`` are read.
Its recorded topology identity is compared, not recomputed from edge arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash
from ..diagnostics.sources_v2 import GeneratedSourceV2
from ..paths import project_asset_path

SCHEMA = "malecns_lab.atlas.v1"
DEFAULT_GRAPH = "data/malecns_v2/full_reference.npz"
DEFAULT_ANNOTATIONS = "data/malecns_v1/body-annotations-male-cns-v1.0-minconf-0.5.feather"
DEFAULT_RECORDINGS = "data/diagnostics_v2/standard_stimuli_20261016"
DEFAULT_OUT = "apps/atlas/public/data"
LABELS = {"static": "静止纹理", "left": "向左平移", "right": "向右平移", "up": "向上平移", "down": "向下平移", "expansion": "膨胀刺激"}


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _ids(values, label: str) -> np.ndarray:
    values = np.asarray(values)
    _require(values.ndim == 1 and values.dtype.kind in "iu", f"{label}: expected integer body IDs")
    _require(np.all(values > 0) and len(np.unique(values)) == len(values), f"{label}: duplicate or invalid body IDs")
    return values


def _reference(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        body_ids = _ids(archive["body_ids"], "reference")
        meta = json.loads(str(archive["meta"].item()))
    _require(len(body_ids) > 0 and np.all(body_ids[1:] > body_ids[:-1]), "Reference IDs must be nonempty and ascending")
    _require(meta.get("asset_type") == "full_reference" and meta.get("format_version") == 2, "A V2 full reference is required")
    _require(canonical_hash(meta["provenance"]) == meta["source_id"], "Reference provenance source_id mismatch")
    _require(meta["graph_sha256"] == meta["provenance"]["graph_sha256"], "Reference graph identity mismatch")
    _require(meta["statistics"]["neurons"] == len(body_ids), "Reference node count mismatch")
    _require(type(meta["statistics"]["connections"]) is int and meta["statistics"]["connections"] >= 0, "Invalid reference edge count")
    return body_ids, meta


def _manifest(path: Path, ids: np.ndarray, meta: dict) -> tuple[dict, dict]:
    manifest = _json(path)
    _require(manifest.get("format") == "cnsbridge.neural_replay_run.v2", "Unsupported replay manifest")
    model = manifest["model_provenance"]
    _require(model.get("protocol") == "cnsbridge.neural_replay.v2", "Unsupported recorded model")
    _require(canonical_hash(model) == manifest["model_source_id"], "Recorded model source_id mismatch")
    for key, expected in (("graph_source_id", meta["source_id"]), ("graph_sha256", meta["graph_sha256"]),
                          ("node_count", len(ids)), ("edge_count", meta["statistics"]["connections"])):
        _require(model.get(key) == expected, f"Recorded model {key} differs from reference")
    mapping = model["group_mapping"]
    _require(mapping == meta["provenance"]["group_mapping"], "Recorded group mapping differs from reference")
    _require(set(mapping["body_ids"]) == set(mapping["rules"]), "Group names and mapping rules differ")
    known = set(ids.tolist())
    for name, members in mapping["body_ids"].items():
        _require(isinstance(name, str) and bool(name), "Invalid group name")
        group_ids = _ids(members, f"group {name}") if members else np.empty(0, dtype=np.int64)
        _require(set(group_ids.tolist()) <= known, f"Group {name} contains unknown reference IDs")
    _require(isinstance(manifest["planned_segments"], list) and bool(manifest["planned_segments"]), "No declared recording segments")
    return manifest, mapping


def _atlas(path: Path, ids: np.ndarray, meta: dict, mapping: dict) -> tuple[dict, dict]:
    import pyarrow.feather as feather

    sources = meta["provenance"]["sources"]
    expected = sources.get(path.name)
    _require(isinstance(expected, dict) and bool(expected.get("sha256")), "Annotation file is not identified in reference provenance")
    digest = _sha256(path)
    _require(digest == expected["sha256"], "Annotation SHA256 differs from reference provenance")
    # Projection avoids loading unrelated annotation columns or any synapse table.
    table = feather.read_table(path, columns=["bodyId", "somaLocation"])
    annotation_ids = table["bodyId"].to_pylist()
    _require(all(type(i) is int and i > 0 for i in annotation_ids), "Invalid annotation body IDs")
    _require(len(set(annotation_ids)) == len(annotation_ids), "Duplicate annotation body IDs")
    positions_by_id = dict(zip(annotation_ids, table["somaLocation"].to_pylist()))
    _require(all(int(i) in positions_by_id for i in ids), "Reference IDs missing from annotation table")
    positioned_ids, coordinates = [], []
    for body_id in ids:
        coordinate = positions_by_id[int(body_id)]
        if coordinate is None:
            continue
        _require(isinstance(coordinate, list) and len(coordinate) == 3, f"Malformed somaLocation for {body_id}")
        if any(value is None for value in coordinate):
            continue
        coordinate = np.asarray(coordinate, dtype=np.float64)
        if not np.isfinite(coordinate).all():
            continue
        positioned_ids.append(str(body_id))
        coordinates.append(coordinate)
    xyz = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    minimum = xyz.min(axis=0) if len(xyz) else np.zeros(3)
    maximum = xyz.max(axis=0) if len(xyz) else np.zeros(3)
    center = (minimum + maximum) / 2
    span = float(np.max(maximum - minimum))
    scale = 2 / span if span else 1.0
    xyz = (xyz - center) * scale
    lookup = {int(body_id): index for index, body_id in enumerate(positioned_ids)}
    groups = []
    for name in sorted(mapping["body_ids"]):
        members = mapping["body_ids"][name]
        points = [lookup[i] for i in members if i in lookup]
        groups.append({"name": name, "point_indices": points, "total_neurons": len(members),
                       "positioned_neurons": len(points), "centroid": xyz[points].mean(axis=0).tolist() if points else None})
    atlas = {"schema": SCHEMA, "node_count": len(ids), "positioned_count": len(xyz), "missing_count": len(ids) - len(xyz),
             "positions": xyz.ravel().tolist(), "body_ids": positioned_ids, "groups": groups,
             "coordinate_transform": {"source_field": "somaLocation", "source_axes": ["x", "y", "z"],
                                      "source_units": "official annotation coordinate units; no physical-unit conversion",
                                      "center": center.tolist(), "scale": scale, "axis_order": [0, 1, 2],
                                      "axis_sign": [1, 1, 1], "formula": "display_xyz = (source_xyz - center) * scale",
                                      "source_bounds": {"min": minimum.tolist(), "max": maximum.tolist()},
                                      "description": "胞体位置统一居中和等比缩放；不是完整神经形态或突触连接。"}}
    return atlas, {"path": str(path), "sha256": digest, "official_url": expected.get("url"), "columns": ["bodyId", "somaLocation"]}


def _numeric(values: np.ndarray, shape: tuple, name: str, valid: np.ndarray | None = None) -> np.ndarray:
    _require(values.shape == shape and values.dtype.kind in "iuf", f"{name}: invalid array shape or type")
    _require(not np.isinf(values).any(), f"{name}: infinite values")
    _require(not np.any(values[np.isfinite(values)] < 0), f"{name}: negative values")
    if valid is not None:
        _require(np.isfinite(values[valid]).all(), f"{name}: missing values on valid frames")
    return values


def _nullable(values: np.ndarray) -> list:
    array = values.astype(object)
    array[~np.isfinite(values)] = None
    return array.tolist()


def _png(frames: list[np.ndarray]) -> bytes:
    """Encode one vertical RGB sprite strip using only the standard library."""
    height, width, channels = frames[0].shape
    _require(channels == 3 and all(f.shape == frames[0].shape and f.dtype == np.uint8 for f in frames), "Invalid preview frames")
    raw = b"".join(b"\x00" + row.tobytes() for frame in frames for row in frame)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height * len(frames), 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _preview(source: dict, indices: np.ndarray, run_id: str) -> tuple[dict | None, bytes | None, dict]:
    if source.get("kind") != "generated_engineering_stimulus":
        return None, None, {"available": False, "reason": "No supported recorded generator parameters; no substitute video created"}
    generator = GeneratedSourceV2(source["stimulus_kind"], observation_seconds=source["observation_s"], texture_seed=source["texture_seed"])
    for key in ("total_frames", "nominal_fps", "warmup_s", "translation_px_s", "expansion_s_inv"):
        _require(source.get(key) == generator.metadata[key], f"Stimulus reconstruction parameter mismatch: {key}")
    _require(isinstance(source.get("generator_sha256"), str) and len(source["generator_sha256"]) == 64, "Missing original generator SHA256")
    provenance = {"kind": "reconstructed", "original_generator_sha256": source["generator_sha256"],
                  "current_generator_sha256": generator.metadata["generator_sha256"],
                  "generator_hash_matches": source["generator_sha256"] == generator.metadata["generator_sha256"],
                  "original_source_id": source["source_id"], "reconstructed_source_id": generator.metadata["source_id"],
                  "parameters": source, "description": "按原 texture_seed、刺激参数和输入帧序号使用当前生成器重建；不是原录像。"}
    if not len(indices):
        return None, None, provenance
    frames = [generator.frame(int(index)) for index in indices]
    preview = {"url": f"../stimuli/{run_id}.png", "width": frames[0].shape[1], "height": frames[0].shape[0],
               "frames": len(frames), "layout": "vertical", "kind": "reconstructed"}
    return preview, _png(frames), provenance


def _run(directory: Path, segment: dict, manifest: dict, ids: np.ndarray, mapping: dict) -> tuple[dict, dict, bytes | None]:
    meta_path = directory / "metadata.json"
    meta = _json(meta_path)
    _require(meta.get("format") == "cnsbridge.neural_activity.v2", "Unsupported activity record")
    _require(meta["segment"] == segment, "Segment metadata differs from run manifest")
    _require(meta["source_id"] == manifest["model_source_id"], "Segment model source_id mismatch")
    _require(meta["run_seed"] == manifest["run_seed"], "Segment run_seed mismatch")
    source = meta["input_source"]
    _require(source == manifest["input_sources"][segment["source_index"]], "Segment input source differs from manifest")
    _require(canonical_hash({k: v for k, v in source.items() if k != "source_id"}) == source["source_id"], "Input source_id mismatch")
    frame_hz = float(source["nominal_fps"])
    _require(frame_hz == 20.0, "V2 recording must have a 20 Hz input frame timebase")
    groups = meta["groups"]
    _require(groups == sorted(mapping["body_ids"]), "Recorded group columns differ from model mapping order")
    start, end = segment["start_frame"], segment["end_frame"]
    _require(type(start) is int and type(end) is int and 0 <= start < end <= source["total_frames"], "Invalid declared segment range")
    rows_path = directory / "rows.npz"
    with np.load(rows_path, allow_pickle=False) as rows:
        indices = rows["frame_index"]
        _require(indices.ndim == 1 and indices.dtype.kind in "iu", "Invalid frame_index array")
        n = len(indices)
        _require(n <= end - start and np.array_equal(indices, np.arange(start, start + n)), "Recorded frame order is not the declared contiguous prefix")
        flags = {}
        for key in ("valid", "prediction_valid", "initialization"):
            flags[key] = rows[key]
            _require(flags[key].shape == (n,) and flags[key].dtype.kind == "b", f"Invalid {key} mask")
        valid = flags["valid"]
        _require(not np.any(flags["prediction_valid"] & ~valid), "Prediction validity includes invalid input frames")
        cutoff = _numeric(rows["input_cutoff_s"], (n,), "input_cutoff_s", valid)
        time_s = indices / frame_hz
        _require(np.allclose(cutoff[np.isfinite(cutoff)], time_s[np.isfinite(cutoff)], rtol=0, atol=1e-9), "Input frame timestamps disagree with frame indices")
        rates = _numeric(rows["group_rates_hz"], (n, len(groups)), "group_rates_hz")
        nonempty = [i for i, name in enumerate(groups) if mapping["body_ids"][name]]
        _require(np.isfinite(rates[valid][:, nonempty]).all(), "Group rates missing on valid frames")
        spikes = _numeric(rows["network_spikes"], (n,), "network_spikes", valid)
        active = _numeric(rows["network_active_neurons"], (n,), "network_active_neurons", valid)
        for name, values in (("network_spikes", spikes), ("network_active_neurons", active)):
            finite = values[np.isfinite(values)]
            _require(np.all(finite == np.floor(finite)), f"{name}: counts must be integers")
        _require(np.all(active[np.isfinite(active)] <= len(ids)), "Active neuron count exceeds reference size")
        simulation_start = _numeric(rows["simulation_start_ms"], (n,), "simulation_start_ms", np.ones(n, bool))
        simulation_end = _numeric(rows["simulation_end_ms"], (n,), "simulation_end_ms", np.ones(n, bool))
        _require(np.all(simulation_end >= simulation_start), "Simulation window ends before it starts")
        _require(not n or simulation_start[0] == 0, "Segment simulation does not start at zero")
        _require(np.allclose(simulation_start[1:], simulation_end[:-1], rtol=0, atol=1e-8), "Simulation windows are discontinuous")
        _require(np.allclose((simulation_end - simulation_start)[valid], 1000 / frame_hz, rtol=0, atol=1e-8), "Valid simulation windows differ from recorded frame period")
        invalid_reason = rows["invalid_reason"]
        _require(invalid_reason.shape == (n,) and invalid_reason.dtype.kind in "US", "Invalid invalid_reason array")
    coverage = meta["coverage"]
    for key, expected in (("planned_frames", end - start), ("attempted_frames", n), ("valid_input_frames", int(valid.sum())),
                          ("failed_frames", int((~valid).sum())), ("unprocessed_tail_frames", end - start - n),
                          ("prediction_valid_frames", int(flags["prediction_valid"].sum())),
                          ("warmup_frames", int(np.sum(valid & (simulation_end <= 2000))))):
        _require(coverage[key] == expected, f"Coverage {key} disagrees with recorded arrays")
    _require(meta["status"] in ("complete", "segment_stopped", "budget_exhausted"), "Unknown recording status")
    _require(meta["status"] != "complete" or (n == end - start and valid.all()), "Complete recording has missing or invalid frames")
    totals_path = directory / "neuron_totals.npz"
    with np.load(totals_path, allow_pickle=False) as totals:
        _require(np.array_equal(_ids(totals["body_ids"], "neuron totals"), ids), "Neuron totals body ID order differs from reference")
        counts = totals["counts"]
        _require(counts.shape == ids.shape and counts.dtype.kind in "iu" and np.all(counts >= 0), "Invalid cumulative neuron counts")
    _require(meta["neural_summary"]["network_nodes"] == len(ids), "Neural summary node count mismatch")
    _require(int(counts.sum()) == meta["neural_summary"]["spikes"], "Cumulative count and neural summary mismatch")
    if valid.all():
        _require(np.isclose(counts.sum(), spikes.sum(), rtol=0, atol=1e-6), "Cumulative and per-frame spike counts disagree")
        for column in nonempty:
            members = np.searchsorted(ids, mapping["body_ids"][groups[column]])
            _require(np.isclose(counts[members].mean() * frame_hz, rates[:, column].sum(), rtol=1e-10, atol=1e-7),
                     f"Group rate column does not match cumulative membership: {groups[column]}")
    run_id = segment["segment_id"]
    preview, png, reconstruction = _preview(source, indices, run_id)
    warmup = float(source.get("warmup_s", 2.0))
    _require(np.isfinite(warmup) and warmup >= 0, "Invalid warmup duration")
    stimulus = source.get("stimulus_kind", "unknown")
    label = LABELS.get(stimulus, stimulus)
    notes = ["固定 LIF 诊断模型的仿真记录；不是最新训练策略，也不是活体测量。",
             "逐帧颜色为 50 ms 窗口的群组平均放电率；同组节点共享组均值，未记录节点保持静态背景。",
             "神经元累计计数仅用于校验，不作为逐时刻活动。", "仿真动作与实体反馈均未接入。",
             "time_s 是原输入帧序号 / 20；simulation_start_s 和 simulation_end_s 是相应仿真窗口。"]
    if reconstruction.get("generator_hash_matches") is False:
        notes.append("当前生成器源码摘要与原记录不同；预览为按原参数使用当前生成器重建，不宣称原源码逐字节复现。")
    run = {"schema": SCHEMA, "id": run_id, "stimulus": stimulus, "label": label, "groups": groups,
           "time_s": time_s.tolist(), "frame_index": indices.tolist(), "valid": valid.tolist(),
           "group_rates_hz": _nullable(rates), "network_spikes": _nullable(spikes), "network_active_neurons": _nullable(active),
           "warmup_s": warmup, "frame_hz": frame_hz, "source_id": source["source_id"], "model_source_id": meta["source_id"],
           "simulation_start_s": (simulation_start / 1000).tolist(), "simulation_end_s": (simulation_end / 1000).tolist(),
           "prediction_valid": flags["prediction_valid"].tolist(), "initialization": flags["initialization"].tolist(),
           "invalid_reason": invalid_reason.tolist(), "status": meta["status"], "coverage": coverage,
           "stimulus_preview": preview, "notes": notes,
           "provenance": {"recording_directory": str(directory), "input_source": source, "segment": segment,
                          "run_seed": meta["run_seed"], "stimulus_reconstruction": reconstruction,
                          "source_files_sha256": {p.name: _sha256(p) for p in (meta_path, rows_path, totals_path)},
                          "timebase": "frame_index / nominal_fps; no display interpolation", "cumulative_counts_role": "validation only"}}
    catalog_run = {"id": run_id, "stimulus": stimulus, "label": label, "url": f"runs/{run_id}.json", "frame_count": n,
                   "duration_s": n / frame_hz, "warmup_s": warmup}
    return run, catalog_run, png


def _write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.parent.is_symlink(), f"Refusing symlink output directory: {path.parent}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".atlas-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_atlas(*, graph: Path, annotations: Path, recordings: Path, out: Path) -> dict:
    """Validate all inputs, then write only new browser display artifacts."""
    graph, annotations, recordings, out = (Path(p).resolve() for p in (graph, annotations, recordings, out))
    for protected in (graph.parent, annotations.parent, recordings):
        _require(not out.is_relative_to(protected) and not protected.is_relative_to(out), "Output must be separate from source asset directories")
    for folder in (out / "runs", out / "stimuli"):
        _require(not folder.is_symlink(), f"Refusing symlink output directory: {folder}")
    ids, meta = _reference(graph)
    manifest_path = recordings / "run_manifest.json"
    manifest, mapping = _manifest(manifest_path, ids, meta)
    atlas, annotation_provenance = _atlas(annotations, ids, meta, mapping)
    artifacts, catalog_runs, seen = {}, [], set()
    for segment in manifest["planned_segments"]:
        name = segment["segment_id"]
        _require(isinstance(name, str) and name not in (".", "..") and Path(name).name == name and name not in seen, "Unsafe or duplicate segment ID")
        _require(type(segment["source_index"]) is int and 0 <= segment["source_index"] < len(manifest["input_sources"]), "Invalid source index")
        seen.add(name)
        run, catalog_run, png = _run(recordings / name, segment, manifest, ids, mapping)
        artifacts[catalog_run["url"]] = run
        catalog_runs.append(catalog_run)
        if png is not None:
            artifacts[f"stimuli/{name}.png"] = png
    catalog = {"schema": SCHEMA, "graph": {"node_count": len(ids), "edge_count": meta["statistics"]["connections"],
                                         "source_id": meta["source_id"], "graph_sha256": meta["graph_sha256"]},
               "atlas_url": "atlas.json", "model": {"name": "固定 LIF 完整参考诊断模型", "source_id": manifest["model_source_id"],
                                                     "trained": False, "description": "已有 V2 诊断仿真回放；不是最新训练策略或活体神经测量。"},
               "recording": {"name": recordings.name, "source_path": str(recordings), "frame_hz": 20.0}, "runs": catalog_runs,
               "provenance": {"reference_path": str(graph), "annotation": annotation_provenance, "license": meta.get("license"),
                              "manifest_sha256": _sha256(manifest_path), "exporter_sha256": _sha256(Path(__file__)),
                              "recorded_model_provenance": manifest["model_provenance"], "recorded_runner_sha256": manifest.get("runner_sha256"),
                              "identity_validation": "Reference metadata provenance, recorded graph identity, annotation SHA256, ordered neuron totals and group membership checked",
                              "topology_hash_validation": "Recorded graph digest compared across metadata; edge arrays not loaded and topology hash not recomputed",
                              "graph_arrays_loaded": ["body_ids", "meta"], "simulation_executed": False},
               "notes": ["全部参考节点计入总数；仅展示有官方 somaLocation 的胞体点，不虚构缺失位置、完整形态或突触连线。",
                         "逐帧活动为记录的群组均值；未记录节点为静态背景。", "刺激预览为按原参数重建；仿真动作和实体反馈均未接入。"]}
    artifacts["atlas.json"] = atlas
    artifacts["catalog.json"] = catalog  # Publish catalog last, after validation and its dependencies.
    for relative, value in artifacts.items():
        content = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        _write(out / relative, content)
    return {"output": str(out), "node_count": len(ids), "positioned_count": atlas["positioned_count"], "missing_count": atlas["missing_count"],
            "groups": len(atlas["groups"]), "runs": len(catalog_runs), "frames": sum(r["frame_count"] for r in catalog_runs), "files": len(artifacts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only export of recorded V2 activity and official soma positions; no simulation or training.")
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        result = export_atlas(**{key: project_asset_path(value) for key, value in vars(args).items()})
    except (OSError, ValueError, KeyError, IndexError, TypeError, ImportError) as error:
        parser.exit(2, f"Atlas export failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
