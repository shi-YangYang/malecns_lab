"""Registered, supervised full-MaleCNS one-update numerical probe. No devices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import psutil

from .resource_guard import GIB, resource_budget, supervise, write_json
from .probe_evidence import (ARTIFACTS, REGISTRATION_SCHEMA, RESULT_SCHEMA, canonical_hash, file_sha256,
                             implementation_hashes, validate_probe_evidence, validate_registration)

PROTOCOL = {
    "schema": "cnsbridge.fullgraph_internal_probe.v1", "role": "numerical_feasibility_only",
    "input_raw_type": "R1-R6", "observation_raw_type": "L1", "dt_ms": 0.1,
    "steps": 200, "simulated_ms": 20.0, "pulse_period_steps": 67, "pulse_first_step": 0,
    "pulse_increment_mv": 68.75, "target_final_voltage_mv": -50.0,
    "objective": "mean squared final L1 membrane voltage error",
    "optimizer": "Adam", "learning_rate": 0.01, "updates": 1,
    "theta_initial": 0.0, "trainable_parameters": "internal presynaptic type theta only",
    "dtype": "torch.float32", "device": "cpu", "torch_threads": 4,
    "restoration_atol_mv": 1e-5, "restoration_rtol": 1e-6,
}


def register(graph: Path, directory: Path) -> dict:
    """Write the fixed protocol before importing the model or launching a child."""
    directory.mkdir(parents=True, exist_ok=False)
    budget = resource_budget(psutil.virtual_memory().available)
    registration = {"schema": REGISTRATION_SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(),
                    "graph_path": str(graph.resolve()), "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL),
                    "resources": {**budget, "wall_limit_seconds": 600, "sampling_seconds": 0.1},
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "psutil": psutil.__version__},
                    "implementation_sha256": implementation_hashes()}
    registration["registration_id"] = canonical_hash(registration)
    write_json(directory / "registration.json", registration)
    return registration


def run_registered(graph: Path, directory: Path) -> dict:
    registration = register(graph, directory)
    budget = registration["resources"]
    if not budget["can_launch"]:
        result = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                  "detail": "Available memory does not leave the registered 4 GiB system reserve",
                  "peak_observed_rss_bytes": 0, "wall_seconds": 0.0, "returncode": None,
                  "worker_status": "not_launched", "last_phase": None}
        write_json(directory / "result.json", result)
        return result
    outcome = supervise([sys.executable, "-m", "cnsbridge.learning.probe", "--worker", str(directory.resolve())],
                        directory, rss_limit_bytes=budget["effective_rss_limit_bytes"],
                        wall_seconds=600, reserve_bytes=4 * GIB)
    return finalize_run(directory, outcome)


def finalize_run(directory: Path, outcome: dict) -> dict:
    """A normal process exit is promoted only after validating its P1 evidence."""
    result = dict(outcome)
    if result["status"] == "completed":
        result["supervisor_reason"] = result["reason"]
        try:
            if result.get("returncode") != 0 or result["reason"] != "worker_exited":
                raise ValueError("Process did not complete with a clean supervised exit")
            result["evidence_validation"] = validate_probe_evidence(directory, PROTOCOL)
            result.update(status="passed", reason="numeric_internal_update_verified")
        except Exception as error:
            result.update(status="failed", reason="invalid_probe_evidence", detail=f"{type(error).__name__}:{error}")
    write_json(directory / "result.json", result)
    return result


def run_worker(directory: Path) -> dict:
    """Child-only imports and numerical work are all inside the parent's budget."""
    started = time.monotonic()
    phases = {}
    current = None
    phase_start = started
    result = {"schema": RESULT_SCHEMA, "status": "failed", "phases_seconds": phases}

    def phase(name, **details):
        nonlocal current, phase_start
        now = time.monotonic()
        if current is not None:
            phases[current] = phases.get(current, 0.0) + now - phase_start
        current, phase_start = name, now
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": now - started, **details})

    try:
        phase("imports")
        import gc
        import numpy as np
        import scipy
        import torch
        from cnsbridge.brain.trainable import TrainableLIF, TrainableLIFConfig
        from cnsbridge.brain.trainable_graph import load_training_graph
        from .checkpoint import load_checkpoint, save_checkpoint

        registration = json.loads((directory / "registration.json").read_text())
        run_identity = validate_registration(registration, PROTOCOL)
        result.update(**run_identity, implementation_sha256=registration["implementation_sha256"],
                      graph_path=registration["graph_path"])
        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.set_num_interop_threads(1)
        result["environment"] = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                 "torch": torch.__version__, "psutil": psutil.__version__,
                                 "torch_threads": torch.get_num_threads(), "device": "cpu"}
        phase("load_graph")
        graph = load_training_graph(registration["graph_path"])
        result["graph"] = graph.identity

        def original_hash():
            digest = hashlib.sha256()
            for values in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
                # Bounded temporary chunks also support large int32 index arrays.
                for offset in range(0, len(values), 1_000_000):
                    digest.update(np.asarray(values[offset:offset + 1_000_000], dtype="<i8").tobytes())
            return digest.hexdigest()

        inputs = graph.body_ids[graph.group_index == graph.group_names.index(PROTOCOL["input_raw_type"])]
        observation_indices = np.flatnonzero(graph.group_index == graph.group_names.index(PROTOCOL["observation_raw_type"]))
        if not len(inputs) or not len(observation_indices) or np.intersect1d(inputs, graph.body_ids[observation_indices]).size:
            raise ValueError("Nonempty disjoint input and observation groups are required")
        result["input_neurons"], result["observation_neurons"] = len(inputs), len(observation_indices)
        phase("construct_model")
        model = TrainableLIF(graph, inputs, config=TrainableLIFConfig(dt_ms=PROTOCOL["dt_ms"]), dtype=torch.float32)
        result["model"] = model.identity
        write_json(directory / "model_identity.json", model.identity)
        indices = torch.tensor(observation_indices)
        zero = torch.zeros(graph.n, dtype=torch.float32)
        pulse = zero.clone()
        pulse[model.input_mask] = PROTOCOL["pulse_increment_mv"]

        def response(active_model, name):
            phase(name)
            state = active_model.reset()
            for step in range(PROTOCOL["steps"]):
                _, state = active_model.step(pulse if step % PROTOCOL["pulse_period_steps"] == 0 else zero, state)
                if (step + 1) % 25 == 0:
                    write_json(directory / "phase.json", {"phase": name, "completed_steps": step + 1,
                                                          "wall_seconds": time.monotonic() - started})
            return state.v[indices]

        phase("construct_optimizer")
        optimizer = torch.optim.Adam([model.theta], lr=PROTOCOL["learning_rate"])
        initial_theta = model.theta.detach().numpy().copy()
        baseline = response(model, "baseline_forward")
        initial = baseline.detach().numpy().copy()
        np.savez(directory / "baseline.npz", observation_body_ids=graph.body_ids[observation_indices], voltage_mv=initial,
                 **run_identity)
        loss = ((baseline - PROTOCOL["target_final_voltage_mv"]) ** 2).mean()
        result["loss_before"] = float(np.mean((initial.astype(np.float64) - PROTOCOL["target_final_voltage_mv"]) ** 2))
        phase("backward")
        loss.backward()
        gradient = model.theta.grad.detach().numpy().copy()
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite internal gradient")
        result["gradient_l2"] = float(np.linalg.norm(gradient.astype(np.float64)))
        result["gradient_nonzero_groups"] = int(np.count_nonzero(gradient))
        phase("optimizer_step")
        optimizer.step()
        updated_theta = model.theta.detach().numpy().copy()
        updated_gain = model.gains.detach().numpy().copy()
        if not np.isfinite(updated_theta).all():
            raise ValueError("Nonfinite updated theta")
        result.update(theta_nonzero_groups=int(np.count_nonzero(updated_theta)),
                      theta_max_abs_change=float(np.max(np.abs(updated_theta))),
                      gain_nonzero_groups=int(np.count_nonzero(updated_gain - 1)),
                      gain_max_abs_change=float(np.max(np.abs(updated_gain - 1))))
        np.savez(directory / "update.npz", theta_before=initial_theta, gradient=gradient, theta=updated_theta,
                 gain=updated_gain, **run_identity)
        del baseline, loss, optimizer
        gc.collect()
        with torch.no_grad():
            updated = response(model, "updated_forward").numpy().copy()
            result["loss_after"] = float(np.mean((updated.astype(np.float64) - PROTOCOL["target_final_voltage_mv"]) ** 2))
            result["loss_decreased"] = result["loss_after"] < result["loss_before"]
            phase("checkpoint_save")
            saved = save_checkpoint(model, directory / "checkpoint", run_identity=run_identity)
            result["checkpoint_id"] = saved["checkpoint_id"]
            model.theta.zero_()
            restored = response(model, "restored_zero_forward").numpy().copy()
            result["zero_restoration_matches"] = bool(np.allclose(initial, restored, atol=PROTOCOL["restoration_atol_mv"],
                                                                 rtol=PROTOCOL["restoration_rtol"]))
            del model
            gc.collect()
            phase("checkpoint_load")
            reloaded_model = load_checkpoint(directory / "checkpoint", reference=graph)
            reloaded = response(reloaded_model, "checkpoint_forward").numpy().copy()
        result["checkpoint_response_matches"] = bool(np.allclose(updated, reloaded, atol=PROTOCOL["restoration_atol_mv"],
                                                                 rtol=PROTOCOL["restoration_rtol"]))
        result["response_changed_neurons"] = int(np.count_nonzero(updated != initial))
        result["response_max_abs_change_mv"] = float(np.max(np.abs(updated - initial)))
        np.savez(directory / "responses.npz", observation_body_ids=graph.body_ids[observation_indices],
                 baseline_mv=initial, updated_mv=updated, restored_zero_mv=restored, checkpoint_mv=reloaded, **run_identity)
        phase("verify_raw_graph")
        result["raw_counts_sha256_after"] = original_hash()
        result["raw_graph_unchanged"] = result["raw_counts_sha256_after"] == graph.identity["raw_ordered_counts_sha256"]
        result["simulated_ms_per_forward"] = PROTOCOL["simulated_ms"]
        result["artifacts_sha256"] = {name: file_sha256(directory / name) for name in ARTIFACTS}
        criteria = [result["gradient_nonzero_groups"] > 0, result["theta_nonzero_groups"] > 0, result["gain_nonzero_groups"] > 0,
                    result["response_changed_neurons"] > 0, result["zero_restoration_matches"],
                    result["checkpoint_response_matches"], result["raw_graph_unchanged"]]
        result["status"] = "passed" if all(criteria) else "failed"
        result["reason"] = "numeric_internal_update_verified" if all(criteria) else "numeric_criteria_failed"
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        phase("finished")
        result["wall_seconds"] = time.monotonic() - started
        write_json(directory / "worker_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="Verified full_reference NPZ (required for a new run)")
    parser.add_argument("--out", type=Path, help="New artifact directory; existing directories are rejected")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = run_worker(args.worker)
    else:
        if args.graph is None or args.out is None:
            parser.error("--graph and --out are required")
        result = run_registered(args.graph, args.out)
    print(json.dumps(result if not args.worker else {"status": result["status"], "reason": result["reason"]}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
