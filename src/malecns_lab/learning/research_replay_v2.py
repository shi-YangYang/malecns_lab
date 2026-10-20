"""Explicit OFFLINE replay of a verified research graph; no device entry point."""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path

from ..brain.lif import LIFNetwork
from ..brain.metadata import canonical_hash, file_hashes
from ..diagnostics.replay_storage_v2 import run_segment, source_coverage, write_json
from ..diagnostics.runtime_v2 import NeuralReplayV2
from ..diagnostics.sources_v2 import STIMULUS_KINDS, GeneratedSourceV2, VideoSourceV2, segment_ranges
from .graph_controls import load_research_control


def computation_identity(provenance: dict) -> str:
    """Compare all non-graph computation, preserving groups, bias and code hashes."""
    return canonical_hash({name: value for name, value in provenance.items() if name not in
                           {"graph_source_id", "graph_sha256", "edge_count", "research_graph"}})


class ResearchNeuralReplayV2(NeuralReplayV2):
    """Same P1 computation with independently validated research wiring installed."""

    def __init__(self, parent_path: str | Path, control_path: str | Path, *, run_seed: int = 0):
        super().__init__(parent_path, run_seed=run_seed)
        common_identity = computation_identity(self.provenance)
        parent_source_id = self.graph.meta["source_id"]
        control = load_research_control(control_path, self.graph)
        # The dedicated loader has established exact node/metadata/group identity.
        # Replace the actual propagation matrix, buffers and counters, not just a label.
        del self.net
        self.graph = control
        self.net = LIFNetwork(control.weights, self.params, seed=run_seed)
        for name, bias in self.provenance["bias_mv"].items():
            self.net.set_bias(control.group(name), bias)
        self.groups = {name: control.group(name) for name in self.group_names}
        self.provenance = copy.deepcopy(self.provenance)
        self.provenance.update(graph_source_id=control.meta["source_id"], graph_sha256=control.meta["graph_sha256"],
                               edge_count=control.n_connections)
        self.provenance["research_graph"] = {
            "asset_type": "research_control", "parent_source_id": parent_source_id,
            "transform": control.meta["provenance"]["transform"],
            "graph_seed": control.meta["provenance"]["seed"],
            "parameters": control.meta["provenance"]["parameters"],
            "common_computation_id": common_identity,
            "adapter_source_sha256": {name: file_hashes(Path(__file__).with_name(name))["sha256"] for name in
                                       ("research_replay_v2.py", "graph_controls.py", "graph_diagnostics.py")},
            "scope": "explicit offline research only; production Brain remains closed to this asset",
        }
        if computation_identity(self.provenance) != common_identity:
            raise ValueError("Research replay changed non-graph computation")
        self.source_id = canonical_hash(self.provenance)
        self.reset()


def record_replay(engine, sources: list, planned: list[tuple], out: Path, *, budget_seconds: float,
                  started: float | None = None, registration_id: str | None = None) -> dict:
    """Reuse P1 storage verbatim for explicit G or research experiment jobs."""
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("Positive finite replay wall budget required")
    started = time.perf_counter() if started is None else started
    out.mkdir(parents=True, exist_ok=False)
    try:
        manifest = {"format": "cnsbridge.neural_replay_run.v2", "model_source_id": engine.source_id,
                    "model_provenance": engine.provenance, "run_seed": engine.run_seed,
                    "planned_segments": [segment for _, segment in planned],
                    "input_sources": [source.metadata for source in sources], "budget_seconds": budget_seconds,
                    "common_computation_id": computation_identity(engine.provenance),
                    "experiment_registration_id": registration_id,
                    "research_role": engine.provenance.get("research_graph", {"asset_type": "full_reference", "transform": "G"}),
                    "runner_sha256": {"research_replay_v2.py": file_hashes(Path(__file__))["sha256"]}}
        write_json(out / "run_manifest.json", manifest)
        summaries = [run_segment(engine, source, segment, out / segment["segment_id"], deadline=started + budget_seconds)
                     for source, segment in planned]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"status": "complete" if all(row["status"] == "complete" for row in summaries) else "incomplete",
                  "segments": summaries, "source_coverage": source_coverage(summaries),
                  "wall_seconds": time.perf_counter() - started,
                  "peak_process_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
                  "claim_scope": "offline replay only; no learned action, task performance, hardware or graph advantage evidence"}
        write_json(out / "run_summary.json", result)
        return result
    except BaseException as error:
        write_json(out / "run_failed.json", {"status": "failed", "error": repr(error),
                                             "wall_seconds": time.perf_counter() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stimulus", choices=STIMULUS_KINDS)
    source.add_argument("--video", type=Path)
    source.add_argument("--recording", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--session-id")
    parser.add_argument("--family-id")
    parser.add_argument("--stimulus-kind", choices=("object_motion", "camera_motion", "mixed", "unknown"), default="unknown")
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--texture-seed", type=int, default=7)
    parser.add_argument("--observation-seconds", type=float, default=0.1)
    parser.add_argument("--budget-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output already exists")
    if args.stimulus:
        if args.segments:
            parser.error("Generated sources have one complete independent segment")
        current = GeneratedSourceV2(args.stimulus, observation_seconds=args.observation_seconds, texture_seed=args.texture_seed)
    else:
        scene, session = args.scene_id, args.session_id
        if args.recording:
            manifest = json.loads((args.recording / "session.json").read_text())
            scene, session = scene or manifest.get("scene_id"), session or manifest.get("session_id")
        current = VideoSourceV2(args.video or args.recording / "video.avi", scene_id=scene, session_id=session,
                                family_id=args.family_id, stimulus_kind=args.stimulus_kind, recording_dir=args.recording)
    planned = [(current, {**segment, "source_index": 0, "repetition": 0}) for segment in segment_ranges(args.segments, current.total_frames)]
    started = time.perf_counter()
    engine = ResearchNeuralReplayV2(args.parent, args.control, run_seed=args.run_seed)
    result = record_replay(engine, [current], planned, args.out, budget_seconds=args.budget_seconds, started=started)
    print(f"Saved {args.out}; status={result['status']}; wall={result['wall_seconds']:.3f}s")


if __name__ == "__main__":
    main()
