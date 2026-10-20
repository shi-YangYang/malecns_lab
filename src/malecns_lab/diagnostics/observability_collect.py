"""Child-only full-state paired sampling. Explicit numerical fixtures stay test-only."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np

from .observability_protocol import (CONDITIONS, LAMINA, MODES, WORKER_SCHEMA, array_hash, canonical_hash,
                                     file_hash, ledger, masks, read_json, require, validate_registration)
from ..learning.resource_guard import write_json


def state_arrays(state) -> dict:
    return {"v": state.v.numpy().copy(), "g": state.g.numpy().copy(),
            "refractory_until": state.refractory_until.numpy().copy(),
            "pending": np.stack([item.numpy() for item in state.pending]),
            "step": np.array(state.step, dtype=np.int64), "model_id": np.array(state.model_id)}


def state_digest(arrays: dict) -> str:
    return canonical_hash({key: array_hash(value) for key, value in arrays.items()})


def raw_graph_digest(graph) -> str:
    digest = hashlib.sha256()
    for values in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
        for start in range(0, len(values), 1_000_000):
            digest.update(np.asarray(values[start:start + 1_000_000], dtype="<i8").tobytes())
    return digest.hexdigest()


def selections(graph, policy) -> dict:
    require(policy.identity["training_graph_id"] == graph.identity["training_graph_id"], "Policy/graph mismatch")
    result = {"body_ids": graph.body_ids.astype(np.int64), "group_index": graph.group_index,
              "input_indices": policy.inputs.indices, "dn_indices": policy.outputs.indices}
    for name in LAMINA:
        require(name in graph.group_names, f"Missing exact lamina type: {name}")
        result[name] = np.flatnonzero(graph.group_index == graph.group_names.index(name)).astype(np.int64)
        require(len(result[name]) > 0, f"Empty exact lamina type: {name}")
    return result


def _advance(model, state, drive, selection, phases, binding, progress):
    """Aggregate post-reset states using float64 sums, with no retained autograd graph."""
    import torch
    start_state_hash = state_digest(state_arrays(state))
    start_drive = drive.export_state()
    n_bins = sum(steps for _, _, steps in phases) // 10
    dn = selection["dn_indices"]
    inputs = selection["input_indices"]
    lamina = [selection[name] for name in LAMINA]
    result = {"time_ms": np.empty(n_bins, np.float64), "phase_index": np.empty(n_bins, np.int64),
              "dn_spikes": np.zeros((n_bins, len(dn)), np.int64),
              "dn_v_mv": np.zeros((n_bins, len(dn)), np.float64),
              "dn_g_mv": np.zeros((n_bins, len(dn)), np.float64),
              "lamina_spikes": np.zeros((n_bins, 5), np.int64),
              "lamina_v_mv": np.zeros((n_bins, 5), np.float64),
              "lamina_g_mv": np.zeros((n_bins, 5), np.float64),
              "input_events": np.zeros(n_bins, np.int64), "input_spikes": np.zeros(n_bins, np.int64),
              "node_phase_spikes": np.zeros((len(phases), model.graph.n), np.int64),
              "input_phase_events": np.zeros((len(phases), len(inputs)), np.int64),
              "phase_start_steps": np.empty(len(phases), np.int64),
              "phase_end_steps": np.empty(len(phases), np.int64),
              "phase_wall_seconds": np.empty(len(phases), np.float64),
              "input_phase_end_phase": np.empty((len(phases), len(inputs)), np.float64),
              "input_phase_end_counts": np.empty((len(phases), len(inputs)), np.int64),
              "drive_phase_start": start_drive.phase, "drive_counts_start": start_drive.event_counts,
              "drive_step_start": np.array(start_drive.step, np.int64),
              "start_state_sha256": np.array(start_state_hash), **{k: np.array(v) for k, v in binding.items()}}
    row = 0
    with torch.no_grad():
        for phase_index, (phase_name, mask, steps) in enumerate(phases):
            phase_started = time.monotonic()
            result["phase_start_steps"][phase_index] = state.step
            prior_counts = drive.export_state().event_counts
            for _ in range(steps // 10):
                for _ in range(10):
                    events = drive.step(mask)
                    spikes, state = model.step(events.to_torch(), state)
                    fired = spikes.numpy().astype(np.int64)
                    v, g = state.v.numpy(), state.g.numpy()
                    result["node_phase_spikes"][phase_index] += fired
                    result["dn_spikes"][row] += fired[dn]
                    result["dn_v_mv"][row] += v[dn].astype(np.float64)
                    result["dn_g_mv"][row] += g[dn].astype(np.float64)
                    for column, members in enumerate(lamina):
                        result["lamina_spikes"][row, column] += fired[members].sum(dtype=np.int64)
                        result["lamina_v_mv"][row, column] += v[members].sum(dtype=np.float64) / len(members)
                        result["lamina_g_mv"][row, column] += g[members].sum(dtype=np.float64) / len(members)
                    result["input_events"][row] += len(events.input_indices)
                    result["input_spikes"][row] += fired[inputs].sum(dtype=np.int64)
                result["time_ms"][row] = state.step / 10
                result["phase_index"][row] = phase_index
                progress(phase_name, state.step, 10)
                row += 1
            end_drive = drive.export_state()
            result["phase_end_steps"][phase_index] = state.step
            result["input_phase_events"][phase_index] = end_drive.event_counts - prior_counts
            result["input_phase_end_phase"][phase_index] = end_drive.phase
            result["input_phase_end_counts"][phase_index] = end_drive.event_counts
            result["phase_wall_seconds"][phase_index] = time.monotonic() - phase_started
    for field in ("dn_v_mv", "dn_g_mv", "lamina_v_mv", "lamina_g_mv"):
        result[field] /= 10
    end_drive = drive.export_state()
    result.update(drive_phase_end=end_drive.phase, drive_counts_end=end_drive.event_counts,
                  drive_step_end=np.array(end_drive.step, np.int64),
                  end_state_sha256=np.array(state_digest(state_arrays(state))))
    return result, state


def collect_mode(model, policy, spatial, protocol, directory: Path, binding: dict, progress) -> dict:
    """One mode at a time. Numerical callers must retain numerical_test_only role."""
    import torch
    from ..senses.spatial_drive import SpatialDrive
    role = model.graph.identity["role"]
    require(role in ("full_reference", "numerical_test_only") and binding["role"] == role
            and policy.identity["role"] == role and spatial.identity["role"] == role, "Collection role mismatch")
    require(binding["model_id"] == model.model_id and model.config.dt_ms == protocol["dt_ms"], "Model binding mismatch")
    require(not torch.count_nonzero(model.theta), "Observability requires theta=0")
    if hasattr(model, "rho"):
        require(model.tonic_mode == "fixed" and not torch.count_nonzero(model.rho), "Expected fixed initial tonic background")
    directory.mkdir(exist_ok=False)
    artifacts = {}

    def save(name, values):
        np.savez(directory / name, **values)
        artifacts[name] = file_hash(directory / name)
        # This ledger is durable after every completed branch, even if a later
        # branch is interrupted and no complete mode/worker result is written.
        write_json(directory / "artifacts.json", artifacts)

    drive = SpatialDrive(spatial)
    write_json(directory / "identity.json", {"model": model.identity, "drive": drive.identity, "run": binding})
    artifacts["identity.json"] = file_hash(directory / "identity.json")
    write_json(directory / "artifacts.json", artifacts)
    selection = selections(model.graph, policy)
    phase_steps = ledger(protocol)["phase_steps"]
    stimuli = masks()
    params = {"theta_before": model.theta.detach().numpy().copy(),
              "rho_before": model.rho.detach().numpy().copy() if hasattr(model, "rho") else np.empty(0, np.float32)}
    with torch.no_grad():
        warmup, base_state = _advance(model, model.reset(), drive, selection,
                                     [("precondition", stimuli["blank"], phase_steps["precondition"])], binding,
                                     lambda phase, step, count: progress("precondition", phase, step, count))
        save("precondition.npz", warmup)
        base_drive = drive.export_state()
        base = state_arrays(base_state)
        save("base_state.npz", base)
        for condition in CONDITIONS:
            # LIFState tensors are immutable by the step contract. Each branch
            # begins from the same stored object; no branch's returned state leaks.
            drive.restore_state(base_drive)
            branch, _ = _advance(model, base_state, drive, selection,
                                 [("stimulus", stimuli[condition], phase_steps["stimulus"]),
                                  ("withdrawal", stimuli["blank"], phase_steps["withdrawal"])], binding,
                                 lambda phase, step, count, c=condition: progress(c, phase, step, count))
            require(branch["start_state_sha256"].item() == state_digest(base), "Base state mutated across branches")
            save(f"{condition}.npz", branch)
    params.update(theta_after=model.theta.detach().numpy().copy(),
                  rho_after=model.rho.detach().numpy().copy() if hasattr(model, "rho") else np.empty(0, np.float32))
    save("parameters.npz", params)
    require(all(np.all(value == 0) for value in params.values()), "Parameters changed during observation")
    return {"model": model.identity, "drive": drive.identity, "base_state_sha256": state_digest(base),
            "artifacts_sha256": {**artifacts, "artifacts.json": file_hash(directory / "artifacts.json")}}


def run_worker(directory: Path) -> dict:
    """Only this child loads Torch and the verified complete production graph."""
    started = time.monotonic()
    result = {"schema": WORKER_SCHEMA, "status": "incomplete", "modes": {}}
    completed = 0
    stages = {}
    stage_started = started
    current = None

    def stage(name):
        nonlocal current, stage_started
        now = time.monotonic()
        if current:
            stages[current] = now - stage_started
        current, stage_started = name, now
        write_json(directory / "phase.json", {"phase": name, "completed_steps": completed,
                                              "wall_seconds": now - started})

    def progress(mode, condition, phase, neural_step, count):
        nonlocal completed
        completed += count
        write_json(directory / "phase.json", {"mode": mode, "condition": condition, "phase": phase,
                                              "neural_step": neural_step, "completed_steps": completed,
                                              "wall_seconds": time.monotonic() - started})
    try:
        stage("imports")
        import torch
        import scipy
        from ..brain.trainable import TrainableLIF
        from ..brain.tonic import LaminaTonicLIF
        from ..brain.trainable_graph import load_training_graph
        from ..brain.policy_groups import load_policy_neuron_map
        from ..senses.spatial_drive import build_spatial_input_map
        from .observability_summary import write_summaries

        registration = read_json(directory / "registration.json")
        protocol = validate_registration(registration)
        require(registration["budget"]["can_launch"], "Unlaunchable registration")
        torch.set_num_threads(protocol["torch_threads"])
        torch.set_num_interop_threads(1)
        result.update(registration_id=registration["registration_id"], protocol_id=registration["protocol_id"],
                      implementation_sha256=registration["implementation_sha256"], graph_path=registration["graph_path"],
                      environment={"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
                                   "torch_threads": torch.get_num_threads(), "device": "cpu", "grad_enabled": False})
        stage("load_graph")
        graph = load_training_graph(registration["graph_path"])
        policy = load_policy_neuron_map(graph, registration["graph_path"])
        spatial = build_spatial_input_map(policy)
        result.update(graph=graph.identity, policy=policy.identity, spatial=spatial.identity)
        write_json(directory / "source_identity.json", {
            "graph": graph.identity, "policy": policy.identity, "spatial": spatial.identity,
            "registration_id": registration["registration_id"], "protocol_id": registration["protocol_id"]})
        np.savez(directory / "selections.npz", **selections(graph, policy))
        for mode in MODES:
            stage(f"construct:{mode}")
            model = (TrainableLIF(graph, policy.inputs.body_ids) if mode == MODES[0]
                     else LaminaTonicLIF(graph, policy.inputs.body_ids, mode="fixed"))
            binding = {"registration_id": registration["registration_id"], "protocol_id": registration["protocol_id"],
                       "model_id": model.model_id, "role": graph.identity["role"]}
            stage(f"collect:{mode}")
            result["modes"][mode] = collect_mode(model, policy, spatial, protocol, directory / mode, binding,
                                                lambda c, p, s, n, m=mode: progress(m, c, p, s, n))
            del model
            gc.collect()
        stage("verify_graph")
        result["raw_counts_sha256_after"] = raw_graph_digest(graph)
        require(result["raw_counts_sha256_after"] == graph.identity["raw_ordered_counts_sha256"], "Raw graph changed")
        stage("summarize")
        write_summaries(directory, protocol)
        result["artifacts_sha256"] = {name: file_hash(directory / name) for name in
                                       ("source_identity.json", "selections.npz", "summary.json", "differences.npz", "report.md")}
        result.update(status="collected", completed_steps=completed, ledger=ledger(protocol))
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        stage("finished")
        result.update(wall_seconds=time.monotonic() - started, stages_seconds=stages, completed_steps=completed)
        write_json(directory / "worker_result.json", result)
    return result
