"""Lightweight independent evidence checks; never load Torch or a graph asset."""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from .observability_protocol import (CONDITIONS, LAMINA, MODES, WORKER_SCHEMA, array_hash, canonical_hash,
                                     file_hash, identity, ledger, masks, read_json, require, validate_registration)
from .observability_summary import report_text, summaries_from_files

BINDINGS = {"registration_id", "protocol_id", "model_id", "role"}
SAMPLE_KEYS = {"time_ms", "phase_index", "dn_spikes", "dn_v_mv", "dn_g_mv", "lamina_spikes", "lamina_v_mv",
               "lamina_g_mv", "input_events", "input_spikes", "node_phase_spikes", "input_phase_events",
               "phase_start_steps", "phase_end_steps", "phase_wall_seconds", "input_phase_end_phase",
               "input_phase_end_counts", "drive_phase_start", "drive_counts_start", "drive_step_start",
               "drive_phase_end", "drive_counts_end", "drive_step_end", "start_state_sha256", "end_state_sha256"}
STATE_KEYS = {"v", "g", "refractory_until", "pending", "step", "model_id"}
MODE_FILES = {"base_state.npz", "precondition.npz", "parameters.npz", "artifacts.json", "identity.json",
              *(f"{c}.npz" for c in CONDITIONS)}
ROOT_FILES = {"source_identity.json", "selections.npz", "summary.json", "differences.npz", "report.md"}
CONFIG = {"dt_ms": 0.1, "v_rest_mv": -52.0, "v_reset_mv": -52.0, "v_threshold_mv": -45.0, "tau_m_ms": 20.0,
          "tau_syn_ms": 5.0, "refractory_ms": 2.2, "delay_ms": 1.8, "weight_mv": 0.275, "surrogate_scale_mv": 1.0}


def arrays(path: Path, keys: set[str]) -> dict:
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == keys, f"Wrong array keys: {path}")
        return {name: data[name].copy() for name in data.files}


def vector(value, shape, dtype, name, *, nonnegative=False):
    require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(), f"Invalid array: {name}")
    if nonnegative:
        require(np.all(value >= 0), f"Negative count/clock: {name}")


def scalar_text(value, expected, name):
    require(value.shape == () and value.dtype.kind == "U" and value.item() == expected, f"Text binding mismatch: {name}")


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def state_hash(values):
    return canonical_hash({key: array_hash(value) for key, value in values.items()})


def _manifest(directory: Path, manifest: dict, keys: set[str]):
    require(isinstance(manifest, dict) and set(manifest) == keys, f"Incomplete manifest: {directory}")
    for name in keys:
        require(file_hash(directory / name) == manifest[name], f"Artifact changed: {directory / name}")


def validate_selections(selection: dict, graph: dict, policy: dict, spatial: dict, *, role="full_reference"):
    """Identity/array contract shared by production and explicit numerical tests."""
    identity(graph, "training_graph_id")
    require(graph.get("role") == role, "Graph role mismatch: production requires full_reference")
    n = graph["nodes"]
    require(type(n) is int and n > 0, "Invalid node count")
    for count in ("edges", "synapse_count"):
        require(type(graph[count]) is int and graph[count] >= 0, f"Invalid {count}")
    require(digest_string(graph["raw_ordered_counts_sha256"]), "Invalid raw graph digest")
    if role == "full_reference":
        require(graph["edges"] > 0 and graph["synapse_count"] >= graph["edges"], "Invalid production graph counts")
        require(canonical_hash(graph["reference_provenance"]) == graph["reference_source_id"]
                and graph["reference_graph_sha256"] == graph["reference_provenance"]["graph_sha256"], "Graph source binding mismatch")
    vector(selection["body_ids"], (n,), "int64", "body IDs")
    require(np.all(np.diff(selection["body_ids"]) > 0), "Body IDs must be strictly ordered")
    vector(selection["group_index"], (n,), "int64", "groups", nonnegative=True)
    grouping = graph["grouping"]
    names = grouping["names"]
    require(names[0] is None and all(isinstance(name, str) and name.strip() for name in names[1:])
            and names[1:] == sorted(set(names[1:])) and grouping["missing_index"] == 0
            and grouping["schema"] == "cnsbridge.presynaptic_raw_type.v1", "Invalid type grouping")
    require(np.all(selection["group_index"] < len(names)) and hashlib.sha256(
        selection["group_index"].astype("<i8").tobytes()).hexdigest() == grouping["mapping_sha256"], "Type grouping hash mismatch")
    for name in ("input_indices", "dn_indices", *LAMINA):
        indices = selection[name]
        vector(indices, (len(indices),), "int64", name, nonnegative=True)
        require(len(indices) > 0 and np.all(np.diff(indices) > 0) and np.all(indices < n), f"Invalid selection: {name}")
    for name, raw in (("input_indices", "R1-R6"), *((name, name) for name in LAMINA)):
        require(raw in names and np.array_equal(selection[name], np.flatnonzero(selection["group_index"] == names.index(raw))),
                f"Incomplete raw type selection: {raw}")
    identity(policy, "policy_map_id")
    identity(spatial, "spatial_map_id")
    require(policy["schema"] == "cnsbridge.policy_neurons.v1" and policy["role"] == role and policy["nodes"] == n
            and policy["training_graph_id"] == graph["training_graph_id"]
            and policy["reference_source_id"] == graph["reference_source_id"]
            and policy.get("reference_graph_sha256") == graph.get("reference_graph_sha256"), "Policy source mismatch")
    require(policy["selection"] == {"input": "raw type == R1-R6", "output": "raw superclass == descending_neuron",
                                   "ordering": "unchanged ascending graph body ID", "side_filter": None}, "Policy selection changed")
    if role == "full_reference":
        require(policy["node_metadata_sha256"] == graph["reference_provenance"]["node_metadata_sha256"], "Annotation hash mismatch")
    for label, indices_key in (("inputs", "input_indices"), ("outputs", "dn_indices")):
        group = policy[label]
        indices = selection[indices_key]
        require(group["indices"] == indices.tolist() and group["body_ids"] == selection["body_ids"][indices].tolist(),
                f"Policy ID/order mismatch: {label}")
        for side in ("root_side", "soma_side"):
            require(len(group[side]) == len(indices) and all(isinstance(s, str) for s in group[side])
                    and len(group[side + "_missing"]) == len(indices)
                    and all(type(v) is bool for v in group[side + "_missing"]), "Invalid side annotations")
    require(spatial["schema"] == "cnsbridge.spatial_input_map.v1" and spatial["role"] == role and spatial["nodes"] == n
            and spatial["policy_map_id"] == policy["policy_map_id"] and spatial["training_graph_id"] == graph["training_graph_id"]
            and spatial["shape"] == [12, 16] and spatial["seed"] == 1701
            and spatial["numpy_version"] == np.__version__, "Spatial map binding mismatch")
    require(spatial["input_indices"] == policy["inputs"]["indices"]
            and spatial["input_body_ids"] == policy["inputs"]["body_ids"], "Spatial input selection mismatch")
    # Reconstruct the registered engineering mapping without importing production mapping code.
    inputs = policy["inputs"]
    side, missing = np.array(inputs["root_side"]), np.array(inputs["root_side_missing"], bool)
    ids = np.array(inputs["body_ids"], np.int64)
    cells = np.empty(len(ids), np.int64)
    rng = np.random.Generator(np.random.PCG64(1701))
    groups = ([None] if missing.any() else []) + sorted(set(side[~missing].tolist()))
    coverage = []
    for label in groups:
        members = np.flatnonzero(missing if label is None else ~missing & (side == label))
        members = members[np.argsort(ids[members], kind="stable")]
        cells[members[rng.permutation(len(members))]] = np.arange(len(members)) % 192
        counts = np.bincount(cells[members], minlength=192)
        coverage.append({"root_side": label, "missing": label is None, "neurons": len(members), "cell_counts": counts.tolist(),
                         "uncovered_cells": np.flatnonzero(counts == 0).tolist(), "uncovered_count": int(np.count_nonzero(counts == 0))})
    require(spatial["pixel_cells"] == cells.tolist() and spatial["coverage"] == coverage, "Spatial mapping/coverage mismatch")


def validate_mode_identity(record, mode, graph, policy, spatial, selection, sources):
    model, drive = record["model"], record["drive"]
    identity(model, "model_id")
    identity(drive, "drive_id")
    expected_schema = "cnsbridge.trainable_lif.v1" if mode == MODES[0] else "cnsbridge.lamina_tonic_lif.v1"
    require(model["schema"] == expected_schema and model["graph"] == graph and model["config"] == CONFIG
            and model["dtype"] == "torch.float32" and model["backend"] == "scipy_cpu_csr_forward_torch_csr_adjoint"
            and model["input_body_ids"] == policy["inputs"]["body_ids"], "Model identity/config mismatch")
    files = ("trainable.py", "trainable_graph.py") + (("tonic.py",) if mode == MODES[1] else ())
    require(model["implementation_sha256"] == {name: sources[f"brain/{name}"] for name in files}, "Model source mismatch")
    if mode == MODES[1]:
        tonic = model["tonic"]
        members = [{"type": name, "count": len(selection[name]), "body_ids_sha256": hashlib.sha256(
            selection["body_ids"][selection[name]].astype("<i8").tobytes()).hexdigest()} for name in LAMINA]
        mapping = np.zeros(graph["nodes"], np.int64)
        for number, name in enumerate(LAMINA, start=1):
            mapping[selection[name]] = number
        require(tonic["mode"] == "fixed" and tonic["types"] == list(LAMINA) and tonic["members"] == members
                and tonic["member_count"] == sum(len(selection[name]) for name in LAMINA)
                and tonic["mapping_sha256"] == hashlib.sha256(mapping.astype("<i8").tobytes()).hexdigest()
                and tonic["range_mv"] == {"lower": 0, "upper": 16, "inclusive": False}
                and tonic["initial_rho"] == 0 and tonic["initial_offset_mv"] == 8
                and model["base_dynamics_schema"] == "cnsbridge.trainable_lif.v1", "Tonic identity mismatch")
    require(drive["schema"] == "cnsbridge.spatial_drive.v1" and drive["spatial_map_id"] == spatial["spatial_map_id"]
            and drive["config"] == {"dt_ms": 0.1, "max_rate_hz": 150.0, "event_voltage_mv": 68.75, "phase_seed": 1702}
            and drive["phase_dtype"] == "float64" and drive["numpy_version"] == np.__version__, "Drive identity mismatch")


def validate_samples(values, selection, spatial, protocol, phases, binding, expected_start_phase, expected_start_counts,
                     expected_start_step, expected_state_hash):
    """Check one complete recording, including an independently integrated event clock."""
    require(set(values) == SAMPLE_KEYS | BINDINGS, "Wrong sample array keys")
    for key, expected in binding.items():
        scalar_text(values[key], expected, key)
    bins = sum(steps for _, steps in phases) // 10
    n, ni, nd, np_ = len(selection["body_ids"]), len(selection["input_indices"]), len(selection["dn_indices"]), len(phases)
    for name in ("dn_spikes", "dn_v_mv", "dn_g_mv"):
        vector(values[name], (bins, nd), "int64" if name.endswith("spikes") else "float64", name, nonnegative=name.endswith("spikes"))
    for name in ("lamina_spikes", "lamina_v_mv", "lamina_g_mv"):
        vector(values[name], (bins, 5), "int64" if name.endswith("spikes") else "float64", name, nonnegative=name.endswith("spikes"))
    for name in ("input_events", "input_spikes", "phase_index"):
        vector(values[name], (bins,), "int64", name, nonnegative=True)
    vector(values["time_ms"], (bins,), "float64", "time_ms")
    for name in ("phase_start_steps", "phase_end_steps"):
        vector(values[name], (np_,), "int64", name, nonnegative=True)
    vector(values["phase_wall_seconds"], (np_,), "float64", "phase times", nonnegative=True)
    vector(values["node_phase_spikes"], (np_, n), "int64", "node phase counts", nonnegative=True)
    for name in ("input_phase_events", "input_phase_end_counts"):
        vector(values[name], (np_, ni), "int64", name, nonnegative=True)
    vector(values["input_phase_end_phase"], (np_, ni), "float64", "phase states")
    for name in ("drive_phase_start", "drive_phase_end"):
        vector(values[name], (ni,), "float64", name)
        require(np.all((values[name] >= 0) & (values[name] < 1)), "Invalid phase range")
    for name in ("drive_counts_start", "drive_counts_end"):
        vector(values[name], (ni,), "int64", name, nonnegative=True)
    for name in ("drive_step_start", "drive_step_end"):
        vector(values[name], (), "int64", name, nonnegative=True)
    scalar_text(values["start_state_sha256"], expected_state_hash, "paired starting state")
    require(values["end_state_sha256"].shape == () and values["end_state_sha256"].dtype.kind == "U"
            and digest_string(values["end_state_sha256"].item()), "Invalid final state digest")
    require(np.array_equal(values["drive_phase_start"], expected_start_phase)
            and np.array_equal(values["drive_counts_start"], expected_start_counts)
            and values["drive_step_start"].item() == expected_start_step, "Unpaired initial input state")
    phase = expected_start_phase.copy()
    counts = expected_start_counts.copy()
    step, row = expected_start_step, 0
    cells = np.asarray(spatial["pixel_cells"], np.int64)
    for index, (mask, steps) in enumerate(phases):
        require(values["phase_start_steps"][index] == step, "Discontinuous phase start")
        before = counts.copy()
        for _ in range(steps // 10):
            total = 0
            for _ in range(10):
                # Preserve floating operation order of the declared recurrence.
                phase += mask.ravel()[cells] * 150.0 * (0.1 / 1000)
                emitted = phase >= 1
                phase -= emitted
                counts += emitted
                total += int(emitted.sum())
                step += 1
            require(values["time_ms"][row] == step / 10 and values["phase_index"][row] == index,
                    "Wrong absolute time or phase index")
            require(values["input_events"][row] == total, "Input bucket differs from registered event recurrence")
            row += 1
        rows = values["phase_index"] == index
        full_counts = values["node_phase_spikes"][index]
        require(np.all(full_counts <= steps) and np.all(values["dn_spikes"][rows] <= 10), "Impossible neural spike count")
        require(np.array_equal(values["dn_spikes"][rows].sum(axis=0), full_counts[selection["dn_indices"]]), "DN bucket/count mismatch")
        for column, name in enumerate(LAMINA):
            require(values["lamina_spikes"][rows, column].sum() == full_counts[selection[name]].sum()
                    and np.all(values["lamina_spikes"][rows, column] <= 10 * len(selection[name])), "Lamina bucket/count mismatch")
        require(values["input_spikes"][rows].sum() == full_counts[selection["input_indices"]].sum()
                and np.all(values["input_spikes"][rows] <= 10 * ni), "R1 bucket/count mismatch")
        require(np.array_equal(values["input_phase_events"][index], counts - before)
                and np.array_equal(values["input_phase_end_counts"][index], counts)
                and np.array_equal(values["input_phase_end_phase"][index], phase)
                and values["phase_end_steps"][index] == step, "Phase event ledger mismatch")
    require(np.array_equal(values["drive_phase_end"], phase) and np.array_equal(values["drive_counts_end"], counts)
            and values["drive_step_end"].item() == step, "Final input clock mismatch")


def validate_supervision(directory: Path, registration: dict, worker: dict):
    supervisor = read_json(directory / "supervisor.json")
    require(supervisor.get("schema") == "cnsbridge.resource_run.v1" and supervisor.get("status") == "completed"
            and supervisor.get("reason") == "worker_exited" and supervisor.get("returncode") == 0
            and supervisor.get("worker_status") == "collected", "Missing clean supervised completion")
    budget, resources = registration["budget"], registration["protocol"]["resources"]
    require(budget["can_launch"], "Resource preflight did not permit this run")
    wall = supervisor["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= resources["wall_seconds"]
            and 0 <= worker["wall_seconds"] <= wall, "Invalid supervised wall time")
    with (directory / "resources.csv").open() as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["wall_seconds", "tree_rss_bytes", "system_available_bytes"], "Wrong resource sample schema")
        rows = list(reader)
    require(bool(rows), "Missing resource samples")
    peak, previous = 0, -1.0
    for row in rows:
        elapsed, rss, available = float(row["wall_seconds"]), int(row["tree_rss_bytes"]), int(row["system_available_bytes"])
        require(math.isfinite(elapsed) and previous <= elapsed <= wall and 0 <= rss <= budget["effective_rss_limit_bytes"]
                and available >= resources["reserve_bytes"], "Resource limit breached or malformed sample")
        previous, peak = elapsed, max(peak, rss)
    require(supervisor["peak_observed_rss_bytes"] == peak, "Peak RSS disagrees with samples")
    final_phase = read_json(directory / "phase.json")
    require(supervisor["last_phase"] == final_phase and final_phase["phase"] == "finished"
            and final_phase["completed_steps"] == registration["ledger"]["total_steps"], "Incomplete final phase")
    return supervisor


def validate_mode_artifacts(folder: Path, record: dict, mode: str, selection: dict, graph: dict, policy: dict,
                            spatial: dict, protocol: dict, run_identity: dict, sources: dict,
                            collection_wall_seconds: float):
    """Numeric/persistence contract; caller separately enforces production role.

    Explicit numerical fixtures exercise the same arrays without certifying a
    production run or changing their numerical_test_only identity.
    """
    stimuli, steps = masks(), ledger(protocol)["phase_steps"]
    ni, n = len(selection["input_indices"]), graph["nodes"]
    validate_mode_identity(record, mode, graph, policy, spatial, selection, sources)
    _manifest(folder, record["artifacts_sha256"], MODE_FILES)
    require(read_json(folder / "artifacts.json") == {name: digest for name, digest in record["artifacts_sha256"].items()
                                                    if name != "artifacts.json"}, "Incremental artifact ledger mismatch")
    model_id = record["model"]["model_id"]
    binding = {"registration_id": run_identity["registration_id"], "protocol_id": run_identity["protocol_id"],
               "model_id": model_id, "role": graph["role"]}
    require(read_json(folder / "identity.json") == {"model": record["model"], "drive": record["drive"], "run": binding},
            "Durable model/drive identity mismatch")
    params = arrays(folder / "parameters.npz", {"theta_before", "theta_after", "rho_before", "rho_after"})
    for name, value in params.items():
        count = len(graph["grouping"]["names"]) if name.startswith("theta") else (5 if mode == MODES[1] else 0)
        vector(value, (count,), "float32", name)
        require(np.all(value == 0), "Model parameters changed")
    base = arrays(folder / "base_state.npz", STATE_KEYS)
    for name in ("v", "g"):
        vector(base[name], (n,), "float32", name)
    vector(base["pending"], (18, n), "float32", "pending queue")
    vector(base["refractory_until"], (n,), "int64", "refractory", nonnegative=True)
    vector(base["step"], (), "int64", "base clock", nonnegative=True)
    scalar_text(base["model_id"], model_id, "base model")
    require(base["step"].item() == steps["precondition"] and state_hash(base) == record["base_state_sha256"], "Base state mismatch")
    reset = {"v": np.full(n, -52, np.float32), "g": np.zeros(n, np.float32),
             "pending": np.zeros((18, n), np.float32), "refractory_until": np.zeros(n, np.int64),
             "step": np.array(0, np.int64), "model_id": np.array(model_id)}
    initial_phase = np.random.Generator(np.random.PCG64(1702)).random(ni)
    warmup = arrays(folder / "precondition.npz", SAMPLE_KEYS | BINDINGS)
    validate_samples(warmup, selection, spatial, protocol, [(stimuli["blank"], steps["precondition"])], binding,
                     initial_phase, np.zeros(ni, np.int64), 0, state_hash(reset))
    scalar_text(warmup["end_state_sha256"], state_hash(base), "Precondition/base state")
    phase_wall = warmup["phase_wall_seconds"].sum()
    for condition in CONDITIONS:
        values = arrays(folder / f"{condition}.npz", SAMPLE_KEYS | BINDINGS)
        validate_samples(values, selection, spatial, protocol,
                         [(stimuli[condition], steps["stimulus"]), (stimuli["blank"], steps["withdrawal"])], binding,
                         warmup["drive_phase_end"], warmup["drive_counts_end"], steps["precondition"], state_hash(base))
        phase_wall += values["phase_wall_seconds"].sum()
    require(phase_wall <= collection_wall_seconds + 1e-6, "Phase timings exceed collection duration")


def validate_summary_files(directory: Path, protocol: dict) -> dict:
    """Recompute all descriptive outputs; shared with numerical evidence fixtures."""
    expected_summary, expected_differences = summaries_from_files(directory, protocol)
    require(read_json(directory / "summary.json") == expected_summary, "Summary differs from raw observations")
    differences = arrays(directory / "differences.npz", set(expected_differences))
    for name, expected in expected_differences.items():
        require(differences[name].dtype == expected.dtype and np.array_equal(differences[name], expected), "Raw paired differences mismatch")
    require((directory / "report.md").read_text() == report_text(expected_summary, protocol), "Report differs from summary")
    return expected_summary


def validate_evidence(directory: Path) -> dict:
    """Production-only validation, independent of worker success flags and summaries."""
    registration = read_json(directory / "registration.json")
    protocol = validate_registration(registration)
    worker = read_json(directory / "worker_result.json")
    require(worker.get("schema") == WORKER_SCHEMA and worker.get("status") == "collected", "Worker did not collect protocol")
    for name in ("registration_id", "protocol_id", "implementation_sha256", "graph_path"):
        require(worker[name] == registration[name], f"Worker binding mismatch: {name}")
    require(set(worker["modes"]) == set(MODES) and worker["completed_steps"] == ledger(protocol)["total_steps"]
            and worker["ledger"] == ledger(protocol), "Incomplete mode/step ledger")
    environment = worker["environment"]
    require(environment["torch_threads"] == 4 and environment["device"] == "cpu" and environment["grad_enabled"] is False
            and environment["numpy"] == np.__version__ and all(isinstance(environment.get(k), str) and environment[k]
                                                               for k in ("torch", "scipy", "numpy")), "Worker environment mismatch")
    supervisor = validate_supervision(directory, registration, worker)
    _manifest(directory, worker["artifacts_sha256"], ROOT_FILES)
    selection = arrays(directory / "selections.npz", {"body_ids", "group_index", "input_indices", "dn_indices", *LAMINA})
    graph, policy, spatial = worker["graph"], worker["policy"], worker["spatial"]
    require(read_json(directory / "source_identity.json") == {
        "graph": graph, "policy": policy, "spatial": spatial, "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol_id"]}, "Durable graph/mapping identity mismatch")
    validate_selections(selection, graph, policy, spatial)
    require(worker["raw_counts_sha256_after"] == graph["raw_ordered_counts_sha256"], "Raw graph changed")
    stages = worker["stages_seconds"]
    require(set(stages) == {"imports", "load_graph", "verify_graph", "summarize", *(f"construct:{m}" for m in MODES),
                            *(f"collect:{m}" for m in MODES)}, "Incomplete worker stages")
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in stages.values())
            and sum(stages.values()) <= worker["wall_seconds"] + 1e-6, "Invalid stage timings")
    for mode in MODES:
        validate_mode_artifacts(directory / mode, worker["modes"][mode], mode, selection, graph, policy, spatial,
                                protocol, registration, registration["implementation_sha256"], stages[f"collect:{mode}"])
    expected_summary = validate_summary_files(directory, protocol)
    return {"schema": "cnsbridge.observability_validation.v1", "status": "protocol_complete",
            "registration_id": registration["registration_id"], "protocol_id": registration["protocol_id"],
            "completed_steps": worker["completed_steps"], "wall_seconds": supervisor["wall_seconds"],
            "peak_observed_rss_bytes": supervisor["peak_observed_rss_bytes"],
            "response_observation": {mode: expected_summary["modes"][mode]["response_observation"] for mode in MODES}}


def validate_run(directory: Path) -> dict:
    result = validate_evidence(directory)
    saved = read_json(directory / "result.json")
    require(saved == result, "Final domain result differs from independently validated evidence")
    return result
