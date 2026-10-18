"""Lightweight parent chain and terminal evidence verification; no graph loading."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from .resource_guard import GIB, resource_budget
from .rewiring_contract import bind, boundary, canonical, prefix_hash, read_json, require, sha
from .rewiring_steps import counters, restore_rng, validate_counters


def initial_state(run):
    import numpy as np
    value = {"mother_source_id": run["mother"]["source_id"], "graph_sha256": run["mother"]["graph_sha256"],
             "node_metadata_sha256": run["mother"]["provenance"]["node_metadata_sha256"], "counts": counters(),
             "rng": np.random.PCG64(run["protocol"]["seed"]).state}
    return {**value, "state_id": canonical(value)}


def checkpoint_context(registration):
    return {key: registration[key] for key in ("segment_id", "segment", "parent_checkpoint_id", "prefix_before")}


def validate_registration(registration, run, chain):
    bind(registration, "segment_id")
    require(registration["schema"] == "cnsbridge.rewiring_segment.v1" and registration["run_id"] == run["run_id"],
            "Segment run differs")
    require(type(registration["segment"]) is int and registration["segment"] == len(chain["segments"]) + 1,
            "Segment sequence differs")
    quota = registration["proposals"]
    require(type(quota) is int and 1 <= quota <= run["protocol"]["max_segment_proposals"], "Invalid explicit segment quota")
    require(registration["parent_checkpoint_id"] == chain["checkpoint_id"]
            and registration["prefix_before"] == chain["prefix"] and registration["start_state"] == chain["state"],
            "Segment start/parent/prefix differs")
    require(not boundary(chain["state"]["counts"], run["protocol"]), "Generation already reached its terminal boundary")
    resources = registration["resources"]
    available = resources["available_bytes_before_launch"]
    require(type(available) is int and available >= 0, "Invalid available memory observation")
    require(resources == {**resource_budget(available, rss_bytes=6 * GIB, reserve_bytes=GIB),
                          "wall_limit_seconds": 600, "sampling_seconds": .1}, "Segment resource protocol differs")


def verify_resources(segment, registration, result):
    budget = registration["resources"]
    require(result["status"] == "completed" and result["reason"] == "worker_exited"
            and type(result["returncode"]) is int and result["returncode"] == 0
            and result["worker_status"] == "completed", "Supervisor/worker did not complete")
    require(math.isfinite(result["wall_seconds"]) and 0 < result["wall_seconds"] <= 600,
            "Wall budget not met")
    rows, last, peak = 0, -1., 0
    with (segment / "resources.csv").open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["wall_seconds", "tree_rss_bytes", "system_available_bytes"], "Resource columns differ")
        for row in reader:
            t, rss, available = float(row["wall_seconds"]), int(row["tree_rss_bytes"]), int(row["system_available_bytes"])
            require(math.isfinite(t) and t >= last and 0 <= t <= 600 and 0 <= rss <= budget["effective_rss_limit_bytes"]
                    and available >= GIB, "Resource observation breached budget")
            last, peak, rows = t, max(peak, rss), rows + 1
    require(rows > 0 and peak == result["peak_observed_rss_bytes"], "Resource peak/records differ")


def validate_completion(segment, run, registration):
    from .rewiring_archive import inspect_checkpoint
    segment = Path(segment)
    result = read_json(segment / "supervisor_result.json")
    verify_resources(segment, registration, result)
    worker = read_json(segment / "worker_result.json")
    require(worker["status"] == "completed" and worker["run_id"] == run["run_id"]
            and worker["segment_id"] == registration["segment_id"], "Worker identity/terminal differs")
    files = worker["files_sha256"]
    expected = {"resume.json", "checkpoint/checkpoint.json", "checkpoint/arrays.npz"}
    if worker["terminal"]:
        expected |= {"candidate.npz", "diagnostics.json", "mechanical_vectors.npz", "candidate_verification.json"}
    require(set(files) == expected, "Worker output inventory differs")
    for name, digest in files.items():
        require(sha(segment / name) == digest, "Worker output changed: " + name)
    resume = read_json(segment / "resume.json")
    require(resume == {"run_id": run["run_id"], "segment_id": registration["segment_id"],
                       "parent_checkpoint_id": registration["parent_checkpoint_id"],
                       "actual_state": registration["start_state"], "verified_before_proposals": True},
            "Actual pre-proposal restoration proof differs")
    checkpoint = inspect_checkpoint(segment / "checkpoint", run, checkpoint_context(registration))
    state, protocol = checkpoint["state"], run["protocol"]
    require(state["node_metadata_sha256"] == run["mother"]["provenance"]["node_metadata_sha256"], "Checkpoint metadata changed")
    validate_counters(state["counts"], protocol["max_attempts"], protocol["target_swaps"])
    restore_rng(state["rng"])
    start, end = registration["start_state"]["counts"], state["counts"]
    require(end["successful_swaps"] >= start["successful_swaps"]
            and all(end["rejections"][k] >= v for k, v in start["rejections"].items()), "Counters moved backwards")
    delta = end["attempts"] - start["attempts"]
    expected_delta = min(registration["proposals"], protocol["max_attempts"] - start["attempts"])
    require(0 <= delta <= expected_delta and (delta == expected_delta or end["successful_swaps"] == protocol["target_swaps"]),
            "Segment stopped before its quota/total boundary")
    terminal = boundary(end, protocol)
    require(type(worker["terminal"]) is bool and worker["terminal"] == terminal
            and worker["checkpoint_id"] == checkpoint["checkpoint_id"] and worker["state_id"] == state["state_id"],
            "Worker checkpoint/boundary differs")
    if terminal:
        proof = read_json(segment / "candidate_verification.json")
        require(end["successful_swaps"] > 0 and proof["status"] == "validated_research_control"
                and proof["graph_sha256"] == state["graph_sha256"] and proof["run_id"] == run["run_id"]
                and proof["checkpoint_id"] == checkpoint["checkpoint_id"] and proof["counts"] == end
                and proof["candidate_sha256"] == files["candidate.npz"]
                and proof["node_count"] == protocol["nodes"] and proof["edge_count"] == protocol["edges"]
                and proof["synapse_count"] == protocol["synapses"] and proof["self_edges"] == protocol["self_edges"]
                and proof["inputs"] == protocol["inputs"] and proof["outputs"] == protocol["outputs"]
                and proof["per_source_weight_multiset"] is True and proof["actual_reloaded"] is True,
                "Final candidate mechanical/provenance proof differs")
    return {"checkpoint_id": checkpoint["checkpoint_id"], "state": state, "terminal": terminal,
            "prefix": prefix_hash(registration["prefix_before"], checkpoint["checkpoint_id"], registration["segment_id"])}


def evidence_hashes(segment):
    """Retain raw terminal evidence, including failed or unsealed files."""
    return {str(path.relative_to(segment)): sha(path) for path in sorted(segment.rglob("*"))
            if path.is_file() and path.name not in {"assessment.json", "assessment.json.tmp"}}


def scan_run(directory, run, *, stop_before=None):
    directory = Path(directory)
    chain = {"segments": [], "checkpoint_id": None, "checkpoint_path": None, "prefix": canonical([]),
             "state": initial_state(run), "terminal": False}
    root = directory / "segments"
    for expected, segment in enumerate(sorted(root.iterdir()) if root.exists() else [], 1):
        require(segment.is_dir() and segment.name == f"{expected:06d}", "Unexpected generation segment path")
        if segment.name == stop_before:
            break
        require(not chain["terminal"], "Extra segment after terminal candidate")
        registration = read_json(segment / "registration.json")
        validate_registration(registration, run, chain)
        assessment = read_json(segment / "assessment.json")  # Missing means interrupted parent, never silently restart.
        bind(assessment, "assessment_id")
        require(assessment["run_id"] == run["run_id"] and assessment["segment_id"] == registration["segment_id"]
                and assessment["files_sha256"] == evidence_hashes(segment), "Parent terminal evidence changed")
        if assessment["status"] == "completed":
            actual = validate_completion(segment, run, registration)
            require(assessment["accepted"] == actual, "Parent accepted state differs")
            chain.update(actual, checkpoint_path=str(segment / "checkpoint"))
        else:
            require(assessment["status"] == "failed" and assessment["accepted"] is None, "Invalid failure terminal")
            # A failed segment never contributes its checkpoint or candidate, even when fully written.
        chain["segments"].append(str(segment))
    return chain
