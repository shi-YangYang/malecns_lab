"""Contract and failure-boundary checks for the read-only browser export."""

import hashlib
import json
import struct
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from malecns_lab.brain.metadata import canonical_hash
from malecns_lab.diagnostics.sources_v2 import GeneratedSourceV2
from malecns_lab.visualization.export import SCHEMA, export_atlas


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_assets(tmp_path, *, frames=50, empty_groups=False, no_positions=False):
    assets = tmp_path / "sources"
    assets.mkdir()
    graph = assets / "full_reference.npz"
    annotations = assets / "annotations.feather"
    recordings = assets / "recordings"
    run_dir = recordings / "source_00_repeat_00_segment_000"
    run_dir.mkdir(parents=True)
    ids = np.array([10001, 10002, 9007199254740993], dtype=np.int64)
    coordinates = [None] * 3 if no_positions else [[10, 20, 30], None, [30, 40, 50]]
    feather.write_feather(pa.table({"bodyId": pa.array(ids), "somaLocation": pa.array(coordinates, type=pa.list_(pa.int64()))}), annotations)
    mapping = {"rules": {} if empty_groups else {"A": {}, "B": {}, "Empty": {}},
               "body_ids": {} if empty_groups else {"A": [10001, 9007199254740993], "B": [10002], "Empty": []}}
    provenance = {"graph_sha256": "b" * 64, "sources": {annotations.name: {"sha256": hashlib.sha256(annotations.read_bytes()).hexdigest()}},
                  "group_mapping": mapping}
    graph_meta = {"asset_type": "full_reference", "format_version": 2, "source_id": canonical_hash(provenance),
                  "graph_sha256": "b" * 64, "statistics": {"neurons": 3, "connections": 2}, "provenance": provenance}
    # An object edge payload cannot be read with allow_pickle=False. Export needs no edges.
    np.savez(graph, body_ids=ids, meta=json.dumps(graph_meta), data=np.array([object()], dtype=object))
    model = {"protocol": "cnsbridge.neural_replay.v2", "graph_source_id": graph_meta["source_id"], "graph_sha256": "b" * 64,
             "node_count": 3, "edge_count": 2, "group_mapping": mapping}
    source = GeneratedSourceV2("left").metadata
    source["generator_sha256"] = "a" * 64  # Historical source intentionally differs from current source.
    source["source_id"] = canonical_hash({k: v for k, v in source.items() if k != "source_id"})
    segment = {"start_frame": 0, "end_frame": 50, "segment_id": run_dir.name, "source_index": 0, "repetition": 0}
    manifest = {"format": "cnsbridge.neural_replay_run.v2", "model_provenance": model, "model_source_id": canonical_hash(model),
                "run_seed": 0, "input_sources": [source], "planned_segments": [segment]}
    write_json(recordings / "run_manifest.json", manifest)
    meta = {"format": "cnsbridge.neural_activity.v2", "source_id": manifest["model_source_id"], "run_seed": 0,
            "segment": segment, "input_source": source, "groups": sorted(mapping["body_ids"]),
            "status": "complete" if frames == 50 else "budget_exhausted",
            "coverage": {"planned_frames": 50, "attempted_frames": frames, "valid_input_frames": frames,
                         "failed_frames": 0, "unprocessed_tail_frames": 50 - frames, "prediction_valid_frames": max(0, frames - 41),
                         "warmup_frames": min(frames, 40)}, "neural_summary": {"network_nodes": 3, "spikes": frames}}
    write_json(run_dir / "metadata.json", meta)
    rates = np.empty((frames, 0)) if empty_groups else np.tile([10.0, 0.0, np.nan], (frames, 1))
    index = np.arange(frames)
    rows = {"frame_index": index, "valid": np.ones(frames, dtype=bool), "prediction_valid": (index >= 40) & (index < 49),
            "initialization": index == 0, "input_cutoff_s": index / 20, "group_rates_hz": rates,
            "network_spikes": np.ones(frames), "network_active_neurons": np.ones(frames),
            "simulation_start_ms": index * 50.0, "simulation_end_ms": (index + 1) * 50.0,
            "invalid_reason": np.full(frames, "", dtype=str)}
    np.savez(run_dir / "rows.npz", **rows)
    np.savez(run_dir / "neuron_totals.npz", body_ids=ids, counts=np.array([frames, 0, 0]))
    return {"graph": graph, "annotations": annotations, "recordings": recordings, "out": tmp_path / "browser"}, run_dir


def test_real_contract_precise_ids_timestamps_positions_and_reconstruction(tmp_path):
    paths, run_dir = fixture_assets(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths["graph"].parent.rglob("*") if p.is_file()}
    result = export_atlas(**paths)
    assert result == {"output": str(paths["out"]), "node_count": 3, "positioned_count": 2, "missing_count": 1,
                      "groups": 3, "runs": 1, "frames": 50, "files": 4}
    catalog = json.loads((paths["out"] / "catalog.json").read_text())
    atlas = json.loads((paths["out"] / catalog["atlas_url"]).read_text())
    run_path = paths["out"] / catalog["runs"][0]["url"]
    run = json.loads(run_path.read_text())
    assert catalog["schema"] == atlas["schema"] == run["schema"] == SCHEMA
    assert catalog["model"]["trained"] is False
    assert atlas["body_ids"] == ["10001", "9007199254740993"]
    assert atlas["positions"] == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    assert atlas["groups"][0]["point_indices"] == [0, 1]
    assert atlas["groups"][1]["total_neurons"] == 1 and atlas["groups"][1]["centroid"] is None
    assert atlas["groups"][2]["total_neurons"] == 0 and atlas["groups"][2]["centroid"] is None
    assert run["time_s"][-1] == 2.45 and run["simulation_end_s"][-1] == 2.5
    assert run["group_rates_hz"][0] == [10.0, 0.0, None]
    assert run["frame_index"] == list(range(50))
    assert catalog["runs"][0]["duration_s"] == 2.5
    reconstruction = run["provenance"]["stimulus_reconstruction"]
    assert reconstruction["original_generator_sha256"] == "a" * 64
    assert reconstruction["generator_hash_matches"] is False
    assert len(reconstruction["current_generator_sha256"]) == 64
    preview = run["stimulus_preview"]
    png = (run_path.parent / preview["url"]).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (96, 3600)
    # Decode the exported strip independently and compare its actual pixels.
    offset, compressed = 8, b""
    while offset < len(png):
        size = struct.unpack(">I", png[offset:offset + 4])[0]
        kind, payload = png[offset + 4:offset + 8], png[offset + 8:offset + 8 + size]
        if kind == b"IDAT":
            compressed += payload
        offset += size + 12
    raw = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(3600, 289)
    assert np.all(raw[:, 0] == 0)
    pixels = raw[:, 1:].reshape(50, 72, 96, 3)
    generator = GeneratedSourceV2("left")
    assert np.array_equal(pixels[0], generator.frame(0))
    assert np.array_equal(pixels[-1], generator.frame(49))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


@pytest.mark.parametrize("problem,expected", [
    ("identity", "source_id"), ("node_order", "body ID order"), ("group_order", "group columns"),
    ("frame_order", "frame order"), ("nonfinite", "infinite"), ("valid_nan", "missing on valid"),
    ("coverage", "attempted_frames"), ("annotation", "Annotation SHA256"), ("spike_totals", "per-frame spike"),
])
def test_rejects_mismatched_or_corrupt_evidence_before_writing(tmp_path, problem, expected):
    paths, run_dir = fixture_assets(tmp_path)
    if problem in ("identity", "group_order", "coverage"):
        path = run_dir / "metadata.json"
        meta = json.loads(path.read_text())
        if problem == "identity":
            meta["source_id"] = "wrong"
        elif problem == "group_order":
            meta["groups"] = list(reversed(meta["groups"]))
        else:
            meta["coverage"]["attempted_frames"] = 49
        write_json(path, meta)
    elif problem == "annotation":
        with paths["annotations"].open("ab") as stream:
            stream.write(b"changed")
    elif problem == "node_order":
        path = run_dir / "neuron_totals.npz"
        with np.load(path) as archive:
            data = dict(archive)
        data["body_ids"] = data["body_ids"][::-1]
        np.savez(path, **data)
    else:
        path = run_dir / "rows.npz"
        with np.load(path) as archive:
            data = dict(archive)
        if problem == "frame_order":
            data["frame_index"][1] = 0
        elif problem == "spike_totals":
            data["network_spikes"][0] = 2
        else:
            data["group_rates_hz"][0, 0] = np.inf if problem == "nonfinite" else np.nan
        np.savez(path, **data)
    with pytest.raises(ValueError, match=expected):
        export_atlas(**paths)
    assert not paths["out"].exists()


def test_empty_recording_groups_and_coordinates_are_explicit(tmp_path):
    paths, _ = fixture_assets(tmp_path, frames=0, empty_groups=True, no_positions=True)
    result = export_atlas(**paths)
    assert result["frames"] == result["groups"] == result["positioned_count"] == 0
    assert result["missing_count"] == 3
    catalog = json.loads((paths["out"] / "catalog.json").read_text())
    atlas = json.loads((paths["out"] / "atlas.json").read_text())
    run = json.loads((paths["out"] / catalog["runs"][0]["url"]).read_text())
    assert atlas["positions"] == atlas["body_ids"] == atlas["groups"] == []
    assert run["time_s"] == run["group_rates_hz"] == run["valid"] == []
    assert run["stimulus_preview"] is None and run["status"] == "budget_exhausted"


def test_refuses_output_inside_source_assets(tmp_path):
    paths, _ = fixture_assets(tmp_path)
    paths["out"] = paths["recordings"] / "display"
    with pytest.raises(ValueError, match="separate from source"):
        export_atlas(**paths)
