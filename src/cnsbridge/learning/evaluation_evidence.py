"""Torch-free sealed-chain, resource, resume and closed-loop evidence checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import numpy as np

from ..brain.metadata import canonical_hash
from .evaluation_checkpoint import (context_for, inspect_checkpoint, validate_arrays)
from .evaluation_contract import (CONDITIONS, NEURAL, VAL, TEST, episode_payload, load_condition,
                                  parameter_identity, path_inside)
from .evaluation_rollout import environment_for, environment_summary, metrics, replay_chunk
from .resource_guard import GIB, resource_budget, write_json
from .training_archive import validate_state_digest
from .training_contract import PILOT, bound, bytes_sha, read_arrays, read_json, require, sha256
from .training_evidence import sampled_boundary

TERMINAL_FILES = ("registration.json", "phase.json", "resources.csv", "worker.log", "resume.json", "resume.npz",
                  "worker_result.json", "supervisor_result.json")


def jsonl(path):
    # Use the strict JSON reader for each line, without temporary files.
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "Duplicate trajectory field")
            out[key] = value
        return out
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    lines = Path(path).read_text().splitlines()
    result = [json.loads(line, object_pairs_hook=unique, parse_constant=invalid) for line in lines]
    require(all(isinstance(value, dict) for value in result), "Trajectory objects required")
    return result


def prefix_line(segment, chunk, digest):
    return (json.dumps({"chunk": chunk, "event": f"segments/{segment:06d}/chunks/{chunk:06d}/event.json", "sha256": digest},
                       sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_registration(reg, run, episode, number, frame, parent, prefix):
    bound(reg, "segment_id")
    keys = {"schema", "evaluation_run_id", "episode_id", "condition_id", "parameter_set_id", "segment", "start_frame",
            "parent_checkpoint_id", "prefix_sha256", "frames", "resources", "clock_monotonic", "segment_id"}
    require(set(reg) == keys and reg["schema"] == "cnsbridge.evaluation_segment.v1", "Segment fields differ")
    require(reg["evaluation_run_id"] == run["evaluation_run_id"] and reg["episode_id"] == episode["episode_id"]
            and reg["condition_id"] == episode["condition_id"] and reg["parameter_set_id"] == episode["parameter_set_id"]
            and type(reg["segment"]) is int and reg["segment"] == number
            and type(reg["start_frame"]) is int and reg["start_frame"] == frame
            and reg["parent_checkpoint_id"] == parent and reg["prefix_sha256"] == bytes_sha(prefix), "Segment continuation differs")
    require(type(reg["frames"]) is int and 1 <= reg["frames"] <= 40 and frame + reg["frames"] <= 80, "Frame budget invalid")
    resources = reg["resources"]
    require(type(resources["available_bytes_before_launch"]) is int, "Resource available bytes invalid")
    expected = resource_budget(resources["available_bytes_before_launch"], reserve_bytes=GIB, rss_bytes=6 * GIB)
    require(resources == {**expected, "wall_limit_seconds": 600, "sampling_seconds": .1}, "Resource budget changed")
    require(type(reg["clock_monotonic"]) is float and np.isfinite(reg["clock_monotonic"]), "Invalid segment clock")


def validate_resume(segment, reg, run, episode, previous, environment, parameters):
    report = read_json(path_inside(segment, "resume.json"))
    values = read_arrays(path_inside(segment, "resume.npz"))
    require(set(report) == {"schema", "segment_id", "checkpoint_id", "next_frame", "environment", "state", "state_digest",
                            "parameter_identity", "arrays_sha256"}, "Resume report inventory differs")
    require(report["schema"] == "cnsbridge.evaluation_restored_state.v1" and report["segment_id"] == reg["segment_id"]
            and report["checkpoint_id"] == (previous["checkpoint_id"] if previous else None)
            and type(report["next_frame"]) is int and report["next_frame"] == reg["start_frame"]
            and report["environment"] == environment_summary(environment, reg["start_frame"])
            and report["arrays_sha256"] == sha256(segment / "resume.npz"), "Restored environment/cursor differs before forward")
    if parameters is None:
        require(values == {} and report["state"] is None and report["state_digest"] is None
                and report["parameter_identity"] is None, "Baseline fabricated resume state")
    else:
        actual = {key.removeprefix("parameter."): value for key, value in values.items() if key.startswith("parameter.")}
        arrays = {key: value for key, value in values.items() if not key.startswith("parameter.")}
        require(parameter_identity(run["policy"], actual) == run["parameters"][episode["condition"]]
                == report["parameter_identity"], "Actual restored parameters differ")
        digest = validate_arrays(report["state"], arrays, run["policy"], reg["start_frame"])
        require(digest == report["state_digest"] and (previous is None or digest == previous["state"]),
                "Actual restored full state differs")
    return report["state_digest"]


def validate_chunk(path, run, episode, reg, number, previous, prefix, environment, records, parameters, start_state):
    seal = read_json(path_inside(path, "chunk.json"))
    bound(seal, "chunk_id")
    keys = {"schema", "evaluation_run_id", "episode_id", "segment_id", "chunk", "start_frame", "end_frame", "files_sha256",
            "checkpoint_id", "parent_checkpoint_id", "sealed_monotonic", "chunk_id"}
    require(set(seal) == keys and seal["schema"] == "cnsbridge.evaluation_chunk.v1", "Chunk fields differ")
    files = ("evidence.npz", "trajectory.jsonl", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")
    require(set(seal["files_sha256"]) == set(files), "Chunk file inventory differs")
    for name in files:
        require(sha256(path_inside(path, name)) == seal["files_sha256"][name], "Chunk file SHA differs")
    parent = previous["checkpoint_id"] if previous else None
    start = len(records)
    require(seal["evaluation_run_id"] == run["evaluation_run_id"] and seal["episode_id"] == episode["episode_id"]
            and seal["segment_id"] == reg["segment_id"] and type(seal["chunk"]) is int and seal["chunk"] == number
            and type(seal["start_frame"]) is int and seal["start_frame"] == start
            and seal["parent_checkpoint_id"] == parent, "Chunk source/order/parent differs")
    new_records, values = jsonl(path / "trajectory.jsonl"), read_arrays(path / "evidence.npz")
    end = start + len(new_records)
    require(type(seal["end_frame"]) is int and seal["end_frame"] == end
            and end <= reg["start_frame"] + reg["frames"], "Chunk exceeds frame budget")
    replay_chunk(environment, values, new_records, run, episode, start, parameters)
    require(len(new_records) == min(10, reg["start_frame"] + reg["frames"] - start) or environment.done,
            "Short chunk outside segment end or physical termination")
    event = read_json(path / "event.json")
    expected_event = {"schema", "evaluation_run_id", "episode_id", "segment_id", "chunk", "start_frame", "end_frame",
                      "parent_checkpoint_id", "parameter_identity", "start_state", "end_state", "environment",
                      "evidence_sha256", "trajectory_sha256", "durations"}
    require(set(event) == expected_event and event["schema"] == "cnsbridge.evaluation_event.v1", "Event fields differ")
    require(all(event[key] == seal[key] for key in ("evaluation_run_id", "episode_id", "segment_id", "chunk", "start_frame",
                                                   "end_frame", "parent_checkpoint_id")), "Event/chunk binding differs")
    require(event["evidence_sha256"] == sha256(path / "evidence.npz") and event["trajectory_sha256"] == sha256(path / "trajectory.jsonl")
            and event["environment"] == environment_summary(environment, end)
            and event["parameter_identity"] == (run["parameters"][episode["condition"]] if parameters is not None else None),
            "Event arrays/environment/parameters differ")
    require(event["start_state"] == (previous["state"] if previous else start_state), "Recurrent state changed/reset within scene")
    if parameters is not None:
        validate_state_digest(event["start_state"], run["policy"], start)
        validate_state_digest(event["end_state"], run["policy"], end)
    else:
        require(event["start_state"] is None and event["end_state"] is None, "Baseline fabricated neural evidence")
    require(set(event["durations"]) == {"forward_seconds", "environment_seconds"} and
            all(type(x) is float and np.isfinite(x) and x >= 0 for x in event["durations"].values()), "Invalid timing")
    expected_prefix = prefix + prefix_line(reg["segment"], number, sha256(path / "event.json"))
    require((path / "prefix.jsonl").read_bytes() == expected_prefix, "Prefix order/path/SHA differs")
    context = context_for(run, episode, end, event["environment"], parent, bytes_sha(expected_prefix))
    saved = inspect_checkpoint(path / "checkpoint", run, episode, context)
    if parameters is not None:
        path_inside(path, "checkpoint/state.npz")
    require(saved["checkpoint_id"] == seal["checkpoint_id"] and saved["state"] == event["end_state"], "Saved state differs from end state")
    require(type(seal["sealed_monotonic"]) is float and np.isfinite(seal["sealed_monotonic"]), "Invalid seal clock")
    records.extend(new_records)
    return {**saved, "prefix": expected_prefix, "chunk_id": seal["chunk_id"], "durations": event["durations"]}


def validate_completion(segment, reg, count, terminal, *, production=True):
    supervisor = read_json(segment / "supervisor_result.json")
    require(supervisor.get("status") == "completed" and supervisor.get("reason") == "worker_exited"
            and supervisor.get("returncode") == 0, "Worker was not successfully supervised")
    worker = read_json(segment / "worker_result.json")
    require(worker.get("schema") == "cnsbridge.evaluation_worker.v1" and worker.get("status") == "completed"
            and worker.get("segment_id") == reg["segment_id"] and worker.get("processed_frames") == count
            and worker.get("terminal") == terminal and (count == reg["frames"] or terminal), "Worker completion lacks evidence")
    require(worker.get("parameters_unchanged") is True, "Worker parameter preservation missing")
    if production and reg["parameter_set_id"] is not None:
        require(worker.get("raw_counts_sha256_before") == worker.get("raw_counts_sha256_after")
                == PILOT["raw_ordered_counts_sha256"],
                "Full graph preservation missing")


def scan_episode(directory, run, episode, parameters, *, production=True, current_segment=None):
    directory = Path(directory)
    require(read_json(directory / "episode.json") == episode, "Episode registration differs")
    environment, records, prefix, previous = environment_for(run, episode), [], b"", None
    reports, chunks, durations = [], 0, {"forward_seconds": 0., "environment_seconds": 0.}
    paths = sorted((directory / "segments").iterdir()) if (directory / "segments").exists() else []
    require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in paths), "Unexpected segment directory")
    require([int(p.name) for p in paths] == list(range(1, len(paths) + 1)), "Gapped segment sequence")
    for segment in paths:
        path_inside(directory, str(segment.relative_to(directory)))
        number, start = int(segment.name), len(records)
        require(not environment.done, "Segment follows terminal scene")
        reg = read_json(path_inside(segment, "registration.json"))
        validate_registration(reg, run, episode, number, start, previous["checkpoint_id"] if previous else None, prefix)
        worker = read_json(segment / "worker_result.json") if (segment / "worker_result.json").exists() else {}
        if number != current_segment:
            require(worker.get("failure_kind") != "numerical_or_protocol", "Prior numerical/protocol failure blocks continuation")
        children = sorted((segment / "chunks").iterdir()) if (segment / "chunks").exists() else []
        require(all(p.is_dir() and re.fullmatch(r"\d{6}", p.name) for p in children), "Unexpected chunk directory")
        require([int(p.name) for p in children] == list(range(chunks + 1, chunks + 1 + len(children))), "Duplicate/gapped chunk sequence")
        require(len(children) <= (reg["frames"] + 9) // 10, "Chunk count exceeds segment budget")
        start_state = None
        if children and (children[0] / "chunk.json").exists():
            start_state = validate_resume(segment, reg, run, episode, previous, environment, parameters)
        tail = False
        for path in children:
            path_inside(segment, str(path.relative_to(segment)))
            if not (path / "chunk.json").exists():
                tail = True
                continue
            require(not tail and not environment.done, "Sealed chunk follows incomplete tail or terminal state")
            previous = validate_chunk(path, run, episode, reg, chunks + 1, previous, prefix, environment, records, parameters, start_state)
            if production:
                sampled_boundary(segment, reg, read_json(path / "chunk.json")["sealed_monotonic"])
            chunks += 1
            prefix = previous["prefix"]
            for key in durations:
                durations[key] += previous["durations"][key]
        result = read_json(segment / "result.json") if (segment / "result.json").exists() else None
        if number != current_segment:
            require(result is not None, "Previous segment lacks finalized parent result")
            require(result.get("schema") == "cnsbridge.evaluation_segment_result.v1"
                    and result.get("evaluation_run_id") == run["evaluation_run_id"]
                    and result.get("segment_id") == reg["segment_id"], "Parent result identity differs")
            require(result.get("verified_end_frame") == len(records), "Parent result progress differs")
            require(result.get("reason") != "invalid_segment_evidence" and worker.get("failure_kind") != "numerical_or_protocol",
                    "Invalid evidence cannot resume")
            require(result.get("status") in ("completed", "failed"), "Unknown segment status")
            expected_files = {name for name in TERMINAL_FILES if (segment / name).exists()}
            require(set(result.get("terminal_artifacts_sha256", {})) == expected_files, "Terminal inventory differs")
            for name, digest in result["terminal_artifacts_sha256"].items():
                require(sha256(path_inside(segment, name)) == digest, "Terminal artifact changed")
            require(result.get("supervisor") == read_json(segment / "supervisor_result.json"), "Raw supervisor was rewritten")
            if result["status"] == "completed":
                require(result.get("reason") == "registered_frames_verified", "Completed parent reason differs")
                validate_completion(segment, reg, len(records) - start, environment.done, production=production)
        resource_summary = None
        if (segment / "resources.csv").exists():
            with (segment / "resources.csv").open() as stream:
                samples = list(csv.DictReader(stream))
            if samples:
                resource_summary = {"samples": len(samples),
                    "minimum_available_bytes": min(int(row["system_available_bytes"]) for row in samples),
                    "peak_sampled_rss_bytes": max(int(row["tree_rss_bytes"]) for row in samples)}
        reports.append({"segment": number, "resource_samples": resource_summary, "start_frame": start, "end_frame": len(records), "unsealed_tail": tail,
                        "status": result.get("status") if result else "in_progress", "reason": result.get("reason") if result else None,
                        "supervisor": read_json(segment / "supervisor_result.json") if (segment / "supervisor_result.json").exists() else None,
                        "worker_wall_seconds": worker.get("wall_seconds")})
    return {"next_frame": len(records), "chunks": chunks, "last": previous, "prefix": prefix, "records": records,
            "environment": environment, "terminal": environment.done, "segments": reports, "durations": durations,
            "evidence_complete": bool(environment.done and reports and reports[-1]["status"] == "completed"
                                      and reports[-1]["segment"] != current_segment),
            "metrics": metrics(records, episode["planned_frames"])}


def episode_report(chain, run, episode):
    status = "partial"
    if chain["segments"] and chain["segments"][-1]["status"] == "failed":
        reason = chain["segments"][-1]["reason"]
        status = "resource_failure" if reason in ("wall_limit", "rss_limit", "system_reserve", "resource_preflight_failed") else "io_or_worker_failure"
    elif chain["evidence_complete"]:
        status = "physical_failure" if chain["metrics"]["failure"] else "duration_complete"
    elif chain["terminal"]:
        status = "terminal_evidence_incomplete"
    scored = ({**chain["metrics"], "success": bool(chain["evidence_complete"] and chain["metrics"]["success"])}
              if chain["metrics"] is not None else None)
    return {"status": status,"condition": episode["condition"], "scene_id": episode["scene_id"], "episode_id": episode["episode_id"],
            "terminal": chain["terminal"], "evidence_complete": chain["evidence_complete"],
            "next_frame": chain["next_frame"], "metrics": scored,
            "diagnostic_metrics": chain["metrics"] if not chain["evidence_complete"] else None,
            "checkpoint_id": chain["last"]["checkpoint_id"] if chain["last"] else None,
            "prefix_sha256": bytes_sha(chain["prefix"]), "segments": chain["segments"], "durations": chain["durations"],
            "environment_seconds": chain["next_frame"] * run["protocol"]["dt_s"],
            "neural_simulated_ms": ((run["policy"]["precondition_steps"] + chain["next_frame"] * run["policy"]["frame_steps"])
                                    * run["policy"]["model"]["config"]["dt_ms"] if episode["condition"] in NEURAL and chain["last"] else None)}


def scan_suite(directory, run, *, production=True):
    directory = Path(directory)
    reports = []
    root = directory / "episodes"
    if root.exists():
        require(all(p.is_dir() and p.name in CONDITIONS for p in root.iterdir()), "Unexpected condition directory")
        for condition_dir in root.iterdir():
            require(all(p.is_dir() and p.name in VAL + TEST for p in condition_dir.iterdir()), "Unexpected episode directory")
    pending_seen = False
    for split in (VAL, TEST):
        for name in CONDITIONS:
            condition, parameters = load_condition(directory, run, name)
            for scene in split:
                target = path_inside(directory, f"episodes/{name}/{scene}")
                if target.exists():
                    require(not pending_seen, "Persisted episodes violate frozen condition/scene order")
                    if scene in TEST:
                        require((directory / "validation_summary.json").exists(), "Test lacks sealed validation summary")
                    episode = episode_payload(run, condition, scene)
                    try:
                        chain = scan_episode(target, run, episode, parameters, production=production)
                        reports.append(episode_report(chain, run, episode))
                        pending_seen = not chain["evidence_complete"]
                    except (ValueError, OSError, KeyError, TypeError) as error:
                        # Preserve diagnostic visibility without presenting an unchecked prefix as valid metrics.
                        status = "numerical_or_protocol_failure" if "numerical/protocol failure" in str(error) else "invalid_evidence"
                        reports.append({"condition": name, "scene_id": scene, "episode_id": episode["episode_id"],
                            "terminal": False, "evidence_complete": False, "status": status, "next_frame": None, "metrics": None,
                            "error": f"{type(error).__name__}:{error}"})
                        pending_seen = True
                else:
                    pending_seen = True
    present = {(r["condition"], r["scene_id"]): r for r in reports}
    val_complete = all(present.get((name, scene), {}).get("evidence_complete", False) for name in CONDITIONS for scene in VAL)
    test_complete = val_complete and all(present.get((name, scene), {}).get("evidence_complete", False) for name in CONDITIONS for scene in TEST)
    summary = {"schema": "cnsbridge.evaluation_summary.v1", "evaluation_run_id": run["evaluation_run_id"],
               "val_complete": val_complete, "test_complete": test_complete, "scope": run["protocol"]["scope"],
               "episodes": reports, "missing": [{"condition": name, "scene_id": scene} for split in (VAL, TEST)
                                                  for name in CONDITIONS for scene in split if (name, scene) not in present]}
    summary["evidence_valid"] = all(r["status"] not in ("invalid_evidence", "numerical_or_protocol_failure") for r in reports)
    summary["comparisons"] = comparisons(present)
    return summary


def comparisons(present):
    keys = ("horizontal_mae", "vertical_mae", "distance_mae_m", "qualified_fraction", "valid_observation_fraction", "success")
    result = {}
    for split_name, scenes in (("val", VAL), ("test", TEST)):
        means, paired = {}, {}
        for name in CONDITIONS:
            group = [present.get((name, scene)) for scene in scenes]
            complete = all(item and item["evidence_complete"] for item in group)
            means[name] = {"complete": complete, "scenes_expected": len(scenes),
                           "scenes_terminal": sum(bool(item and item["terminal"]) for item in group),
                           "scenes_evidence_complete": sum(bool(item and item["evidence_complete"]) for item in group), "means": None}
            if complete:
                means[name]["means"] = {key: float(np.mean([item["metrics"][key] for item in group]))
                                        if all(item["metrics"][key] is not None for item in group) else None for key in keys}
        for name in CONDITIONS[1:]:
            rows = []
            for scene in scenes:
                a, b = present.get(("joint", scene)), present.get((name, scene))
                rows.append({"scene_id": scene, "joint_minus_other": {key: float(a["metrics"][key]) - float(b["metrics"][key])
                    if a["metrics"][key] is not None and b["metrics"][key] is not None else None for key in keys}
                    if a and b and a["evidence_complete"] and b["evidence_complete"] else None})
            paired[name] = rows
        result[split_name] = {"equal_scene_weight": means, "paired": paired, "statistical_significance": "not_estimated"}
    return result


def seal_validation(directory, summary):
    require(summary["val_complete"], "All five conditions' validation episodes require completed evidence first")
    payload = {"schema": "cnsbridge.evaluation_validation_seal.v1", "evaluation_run_id": summary["evaluation_run_id"],
               "episodes": [item for item in summary["episodes"] if item["scene_id"] in VAL],
               "comparisons": summary["comparisons"]["val"]}
    payload["validation_id"] = canonical_hash(payload)
    path = Path(directory) / "validation_summary.json"
    if path.exists():
        require(read_json(path) == payload, "Sealed validation summary differs")
    else:
        write_json(path, payload)
    return payload


def check_order(directory, summary, name, scene):
    present = {(r["condition"], r["scene_id"]): r for r in summary["episodes"]}
    ordered = [(n, s) for split in (VAL, TEST) for n in CONDITIONS for s in split]
    requested = (name, scene)
    require(requested in ordered, "Unknown condition/scene")
    first = next((item for item in ordered if not present.get(item, {}).get("evidence_complete", False)), None)
    require(requested == first, "Condition/scene violates frozen evaluation order or is already terminal")
    require(not present.get(requested, {}).get("terminal", False), "Terminal episode lacks completed evidence; no forward or automatic recovery")
    if scene in TEST:
        require(summary["val_complete"] and (Path(directory) / "validation_summary.json").exists(), "Test requires a sealed validation summary")
        seal_validation(directory, summary)
