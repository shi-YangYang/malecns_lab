"""One fresh, complete-reference experiment; invoked only by the local manager."""

from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from pathlib import Path

import numpy as np

from ..brain.metadata import file_hashes
from ..diagnostics.replay_storage_v2 import pack_rows
from ..diagnostics.runtime_v2 import NeuralReplayV2
from .export import LABELS, SCHEMA, _atlas, _nullable, _png, _write
from .stimulus import AtlasStimulus
from .storage import DISPLAY_ASSETS, EVIDENCE_FILES, read_json, write_json


def run_experiment(directory: Path, graph: Path, annotations: Path) -> dict:
    """The real worker entrypoint used by CLI and integration verification."""
    request = read_json(directory / "request.json")
    from .parameters import validate_parameters

    parameters = validate_parameters(request["parameters"])
    source = AtlasStimulus(parameters)
    began = time.monotonic()
    progress = {"status": "loading", "frames_completed": 0, "total_frames": source.total_frames, "simulation_s": 0.0}
    write_json(directory / "progress.json", progress)
    # load_reference inside NeuralReplayV2 validates full integer graph contents,
    # annotations and membership. The parent monitors memory before this starts.
    engine = NeuralReplayV2(graph, run_seed=0)
    meta = engine.graph.meta
    if (meta.get("asset_type") != "full_reference" or engine.graph.n != meta["statistics"]["neurons"]
            or engine.graph.n_connections != meta["statistics"]["connections"]):
        raise ValueError("完整连接组校验失败。")
    manifest = {"format": "malecns_lab.atlas_experiment.v1", "job_id": request["id"], "parameters": request["parameters"],
                "model_source_id": engine.source_id, "model_provenance": engine.provenance,
                "graph": {"node_count": engine.graph.n, "edge_count": engine.graph.n_connections,
                          "source_id": meta["source_id"], "graph_sha256": meta["graph_sha256"]},
                "input_source": source.metadata, "run_seed": 0, "texture_seed": 7,
                "time": {"frame_hz": 20, "warmup_s": 2, "snn_dt_ms": engine.params.dt, "steps_per_frame": 100},
                "source_sha256": {name: file_hashes(Path(__file__).with_name(name))["sha256"]
                                  for name in ("worker.py", "stimulus.py", "storage.py", "export.py")},
                "groups": list(engine.group_names), "graph_path": str(graph), "annotation_path": str(annotations),
                "claim_scope": "fixed LIF simulation; no training, action decoder or device control"}
    write_json(directory / "manifest.json", manifest)
    rows, frames = [], []
    try:
        for index in range(source.total_frames):
            frame = source.frame(index)
            frames.append(frame)
            row = engine.step(frame, index / 20, target_available=index + 1 < source.total_frames)
            row.update(frame_index=index, host_dequeue_time_s=float("nan"))
            rows.append(row)
            if not row["valid"]:
                raise ValueError(f"第 {index + 1} 帧视觉输入无效：{row['vision'].invalid_reason}。")
            progress.update(status="running", frames_completed=index + 1, simulation_s=engine.net.t_ms / 1000)
            write_json(directory / "progress.json", progress)
    finally:
        # A numerical or vision failure retains the attempted prefix, not success.
        arrays = pack_rows(rows, len(engine.group_names))
        np.savez_compressed(directory / "records.npz", **arrays)
        np.savez_compressed(directory / "inputs.npz", frames=np.asarray(frames, dtype=np.uint8),
                            frame_index=np.arange(len(frames)), time_s=np.arange(len(frames)) / 20)
        np.savez_compressed(directory / "neuron_totals.npz", body_ids=engine.graph.body_ids, counts=engine.neuron_total_counts)
    summary = engine.segment_summary()
    if summary["steps"] != source.total_frames * 100 or not np.isclose(summary["simulated_ms"], source.total_frames * 50):
        raise ValueError("仿真时间步未完整执行。")
    model_source_id = engine.source_id
    body_ids = engine.graph.body_ids.copy()
    del engine
    gc.collect()
    progress["status"] = "exporting"
    write_json(directory / "progress.json", progress)
    atlas, annotation_identity = _atlas(annotations, body_ids, meta, meta["provenance"]["group_mapping"])
    write_json(directory / "atlas.json", atlas)
    _write(directory / "stimulus.png", _png(frames))
    count = source.total_frames
    coverage = {"planned_frames": count, "attempted_frames": count, "valid_input_frames": int(arrays["valid"].sum()),
                "failed_frames": 0, "unprocessed_tail_frames": 0, "warmup_frames": 40,
                "prediction_valid_frames": int(arrays["prediction_valid"].sum())}
    run = {"schema": SCHEMA, "id": request["id"], "stimulus": source.stimulus, "label": LABELS[source.stimulus],
           "groups": manifest["groups"], "frame_index": arrays["frame_index"].tolist(), "time_s": (arrays["frame_index"] / 20).tolist(),
           "valid": arrays["valid"].tolist(), "group_rates_hz": _nullable(arrays["group_rates_hz"]),
           "network_spikes": arrays["network_spikes"].tolist(), "network_active_neurons": arrays["network_active_neurons"].tolist(),
           "warmup_s": 2.0, "frame_hz": 20.0, "source_id": source.metadata["source_id"], "model_source_id": model_source_id,
           "simulation_start_s": (arrays["simulation_start_ms"] / 1000).tolist(),
           "simulation_end_s": (arrays["simulation_end_ms"] / 1000).tolist(),
           "prediction_valid": arrays["prediction_valid"].tolist(), "initialization": arrays["initialization"].tolist(),
           "invalid_reason": arrays["invalid_reason"].tolist(), "status": "complete", "coverage": coverage,
           "stimulus_preview": {"url": "stimulus.png", "width": 96, "height": 72, "frames": count, "layout": "vertical", "kind": "generated"},
           "provenance": {"input_source": source.metadata, "run_seed": 0, "parameters": request["parameters"],
                          "timebase": "frame_index / 20; exact generated input frames; no interpolation",
                          "source_files_sha256": {name: file_hashes(directory / name)["sha256"] for name in EVIDENCE_FILES}},
           "notes": ["固定 LIF 模型的完整连接组仿真；颜色为群组平均放电率。", "无训练、动作输出或设备反馈。"]}
    write_json(directory / "run.json", run)
    catalog = {"schema": SCHEMA, "graph": manifest["graph"], "atlas_url": "atlas.json",
               "model": {"name": "固定 LIF 完整参考诊断模型", "source_id": model_source_id, "trained": False,
                         "description": "本次独立运行产生的固定 LIF 仿真响应。"},
               "recording": {"name": request["id"], "source_path": str(directory), "frame_hz": 20.0},
               "runs": [{"id": request["id"], "stimulus": source.stimulus, "label": LABELS[source.stimulus], "url": "run.json",
                         "frame_count": count, "duration_s": count / 20, "warmup_s": 2.0}],
               "provenance": {"reference_path": str(graph), "annotation": annotation_identity, "license": meta.get("license"),
                              "recorded_model_provenance": manifest["model_provenance"], "simulation_executed": True,
                              "topology_hash_validation": "Full integer edge arrays rehashed by load_reference before simulation",
                              "manifest_sha256": file_hashes(directory / "manifest.json")["sha256"]},
               "notes": ["仅显示有官方胞体坐标的节点。", "群组平均放电率；未记录节点保持静态。"]}
    write_json(directory / "catalog.json", catalog)
    result = {"status": "completed", "frames_completed": count, "total_frames": count, "simulation_s": summary["simulated_ms"] / 1000,
              "neural_steps": summary["steps"], "neural_summary": summary, "graph": manifest["graph"],
              "model_source_id": model_source_id, "coverage": coverage, "worker_elapsed_s": time.monotonic() - began,
              "artifacts": {name: file_hashes(directory / name)["sha256"] for name in (*DISPLAY_ASSETS, *EVIDENCE_FILES)}}
    write_json(directory / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    # Also terminate during graph loading if the local server is killed abruptly.
    def watch_parent():
        while True:
            if os.getppid() != args.parent_pid:
                os._exit(3)
            time.sleep(0.2)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        run_experiment(args.directory, args.graph, args.annotations)
        return 0
    except Exception as error:
        phase = read_json(args.directory / "progress.json").get("status", "loading") if (args.directory / "progress.json").exists() else "loading"
        if isinstance(error, ValueError) and "视觉输入无效" in str(error):
            message = "视觉输入未通过检查，请提高对比度或降低速度后重试。"
        elif phase == "loading":
            message = "完整脑模型加载或校验失败，请检查本地数据和依赖。"
        elif phase == "exporting":
            message = "结果保存失败，请检查胞体坐标文件和可用磁盘空间。"
        else:
            message = "神经仿真未完成，已保留记录，请重试或查看本地错误记录。"
        write_json(args.directory / "failure.json", {"error": message, "detail": repr(error), "phase": phase})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
