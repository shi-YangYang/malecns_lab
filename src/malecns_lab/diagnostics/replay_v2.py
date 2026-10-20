"""Offline full-reference V2 visual/neural recording; never controls a device."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

from ..brain.metadata import file_hashes
from ..paths import source_project_root
from .replay_storage_v2 import run_segment, source_coverage, write_json
from .runtime_v2 import NeuralReplayV2
from .sources_v2 import STIMULUS_KINDS, GeneratedSourceV2, VideoSourceV2, segment_ranges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--recording", type=Path)
    source_group.add_argument("--video", type=Path)
    source_group.add_argument("--stimulus", choices=(*STIMULUS_KINDS, "all"))
    parser.add_argument("--brain", type=Path, default=source_project_root() / "data/malecns_v2/full_reference.npz")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrite")
    parser.add_argument("--scene-id")
    parser.add_argument("--session-id")
    parser.add_argument("--family-id", help="Required for recorded material; related clips share this ID")
    parser.add_argument("--stimulus-kind", choices=("object_motion", "camera_motion", "mixed", "unknown"), default="unknown")
    parser.add_argument("--segments", type=Path, help="Predeclared JSON frame ranges; each starts a fresh 2 s warmup")
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--texture-seed", type=int, default=7)
    parser.add_argument("--observation-seconds", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--budget-seconds", type=float, default=300.0, help="Wall budget, checked between 50 ms simulation windows")
    args = parser.parse_args(argv)
    if args.repeat < 1 or not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("repeat and budget must be positive")
    if args.stimulus and args.segments:
        parser.error("generated stimuli already define independent complete segments")
    if args.out.exists():
        parser.error("output already exists")
    if args.stimulus:
        sources = [GeneratedSourceV2(name, observation_seconds=args.observation_seconds, texture_seed=args.texture_seed)
                   for name in (STIMULUS_KINDS if args.stimulus == "all" else (args.stimulus,))]
    else:
        scene, session = args.scene_id, args.session_id
        if args.recording:
            manifest = json.loads((args.recording / "session.json").read_text())
            scene, session = scene or manifest.get("scene_id"), session or manifest.get("session_id")
        sources = [VideoSourceV2(args.video or args.recording / "video.avi", scene_id=scene, session_id=session,
                                family_id=args.family_id, stimulus_kind=args.stimulus_kind, recording_dir=args.recording)]
    planned = []
    for source_index, source in enumerate(sources):
        for repetition in range(args.repeat):
            for segment in segment_ranges(args.segments, source.total_frames):
                planned.append((source, {**segment, "source_index": source_index, "repetition": repetition,
                                         "segment_id": f"source_{source_index:02d}_repeat_{repetition:02d}_{segment['segment_id']}"}))
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        engine = NeuralReplayV2(args.brain, run_seed=args.run_seed)
        manifest = {"format": "cnsbridge.neural_replay_run.v2", "model_source_id": engine.source_id,
                    "model_provenance": engine.provenance, "run_seed": args.run_seed, "planned_segments": [item for _, item in planned],
                    "input_sources": [source.metadata for source in sources], "budget_seconds": args.budget_seconds,
                    "predeclared_segments_sha256": file_hashes(args.segments)["sha256"] if args.segments else None,
                    "runner_sha256": {name: file_hashes(Path(__file__).with_name(name))["sha256"]
                                      for name in ("replay_v2.py", "replay_storage_v2.py", "sources_v2.py")}}
        write_json(args.out / "run_manifest.json", manifest)
        summaries = []
        for source, segment in planned:
            summary = run_segment(engine, source, segment, args.out / segment["segment_id"], deadline=started + args.budget_seconds)
            summaries.append(summary)
            print(f"{segment['segment_id']}: {summary['status']} | {summary['coverage']} | {summary['computation']}", flush=True)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"status": "complete" if all(row["status"] == "complete" for row in summaries) else "incomplete",
                  "segments": summaries, "wall_seconds": time.perf_counter() - started,
                  "source_coverage": source_coverage(summaries),
                  "peak_process_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
                  "claim_scope": "offline signal diagnostic only; no action labels, learned decoder, behavior or device evidence"}
        write_json(args.out / "run_summary.json", result)
        print(f"Saved {args.out}; status={result['status']}; wall={result['wall_seconds']:.3f}s", flush=True)
    except BaseException as error:
        write_json(args.out / "run_failed.json", {"error": repr(error), "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
