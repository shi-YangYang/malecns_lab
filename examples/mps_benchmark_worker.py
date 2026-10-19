"""Supervised fullgraph MPS experiment; imported only by the claimed child process."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from fullgraph_mps_benchmark import (ROOT, PROTOCOL, environment, implementation_hashes,
                                    observed_source_files, validate_registration, file_sha256)
from cnsbridge.learning.resource_guard import write_json


def run_worker(directory):
    directory = directory.resolve()
    preliminary = json.loads((directory / "registration.json").read_text())
    if os.getppid() != preliminary.get("parent_pid"):
        raise ValueError("Worker must be the child of its registered supervising parent")
    with (ROOT / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Supervising parent does not hold the workspace lock")
    with (directory / "worker.claim").open("x") as claim:
        claim.write(datetime.now(timezone.utc).isoformat() + "\n")
    result = {"schema": "cnsbridge.mps_benchmark_worker.v1", "status": "failed", "reason": None, "mps_kernel_candidate": False}
    started, current, phase_started, phases, sampler = time.perf_counter(), None, time.perf_counter(), {}, None

    def phase(name, **details):
        nonlocal current, phase_started
        now = time.perf_counter()
        if current is not None:
            phases[current] = phases.get(current, 0.0) + now - phase_started
        current, phase_started = name, now
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": now - started, **details})
        if sampler is not None:
            sampler.sample(name)

    try:
        phase("validate_registration")
        registration = validate_registration(directory)
        identity = {key: registration[key] for key in ("registration_id", "protocol_id")}
        result.update(identity)
        phase("imports")
        import numpy as np
        import torch
        from cnsbridge.brain.trainable import TrainableLIF, TrainableLIFConfig
        from cnsbridge.brain.trainable_graph import load_training_graph
        from adjoint_benchmark_kernel import array_sha256, make_inputs
        from mps_benchmark_memory import MPSMemorySampler
        from mps_benchmark_kernel import build_cache, verify_cache, measure_cases, assess_cases, autograd_check

        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.set_num_interop_threads(PROTOCOL["torch_interop_threads"])
        result["environment"] = {**environment(), "mps_available": torch.backends.mps.is_available(),
                                 "mps_built": torch.backends.mps.is_built(), "torch_build": torch.__config__.show(),
                                 "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
        if not result["environment"]["mps_available"]:
            raise ValueError("MPS unavailable; CPU fallback forbidden")
        if result["environment"]["threads"] != 4 or result["environment"]["interop_threads"] != 1:
            raise ValueError("Actual thread setting differs")
        sampler = MPSMemorySampler(directory / "mps_memory.csv", registration["resources"]["effective_rss_limit_bytes"])
        result["mps_budget"] = sampler.budget
        write_json(directory / "mps_budget.json", sampler.budget)
        observed = observed_source_files()
        if not all(registration["implementation_sha256"].get(name) == digest for name, digest in observed.items()):
            raise ValueError("Actual local imports escape registered source closure")
        result["observed_imports_sha256"] = observed
        phase("load_graph")
        graph = load_training_graph(registration["graph_path"])
        result["graph"] = graph.identity
        if (graph.identity["role"] != "full_reference"
                or any(graph.identity[key] != count for key, count in PROTOCOL["graph_counts"].items())
                or graph.identity["reference_graph_sha256"] != PROTOCOL["graph_sha256"]
                or graph.identity["raw_ordered_counts_sha256"] != PROTOCOL["raw_ordered_counts_sha256"]):
            raise ValueError("Loaded graph differs from registered complete logical graph")

        def raw_counts_hash():
            digest = hashlib.sha256()
            for array in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
                for offset in range(0, len(array), 1_000_000):
                    digest.update(np.asarray(array[offset:offset + 1_000_000], dtype="<i8").tobytes())
            return digest.hexdigest()

        result["raw_counts_sha256_before"] = raw_counts_hash()
        if result["raw_counts_sha256_before"] != PROTOCOL["raw_ordered_counts_sha256"]:
            raise ValueError("Initial raw graph digest differs")
        phase("construct_model")
        input_ids = graph.body_ids[graph.group_index == graph.group_names.index(PROTOCOL["model_input_raw_type"])]
        model = TrainableLIF(graph, input_ids, config=TrainableLIFConfig(weight_mv=PROTOCOL["weight_mv"]), dtype=torch.float32)
        result["model"] = model.identity
        write_json(directory / "model_identity.json", model.identity)
        cpu = {"forward": model.base_weights, "adjoint": model.base_weights_adjoint}
        caches, cache_evidence = {}, {}
        for direction in PROTOCOL["directions"]:
            phase("construct_mps_cache", direction=direction)
            caches[direction], cache_evidence[direction] = build_cache(cpu[direction], PROTOCOL["verification_chunk_edges"])
            write_json(directory / "cache_before.json", cache_evidence)
        result["cache_before"] = cache_evidence
        phase("construct_inputs")
        inputs, selected = make_inputs(graph.n, PROTOCOL["seed"])
        result["input_sha256_before"] = {name: array_sha256(value) for name, value in inputs.items()}
        result["selected_sha256_before"] = array_sha256(selected)
        np.savez(directory / "inputs.npz", **inputs, sparse_selected_indices=selected, **identity)
        records, transfers = measure_cases(model._forward_csr, model.base_weights_adjoint, caches, inputs,
                                          directory, PROTOCOL, identity, phase)
        if len(records) != PROTOCOL["total_matvec_calls"]:
            raise ValueError("Incomplete fixed call schedule")
        result["matvec_calls"], result["input_transfer_seconds"] = len(records), transfers
        phase("compare_outputs")
        result.update(assess_cases(directory, records, graph.n, PROTOCOL))
        phase("explicit_autograd")
        result["autograd"] = autograd_check(model._forward_csr, model.base_weights_adjoint, caches, inputs["dense"],
                                            directory, PROTOCOL, identity)
        phase("verify_unchanged")
        after = {}
        for direction in PROTOCOL["directions"]:
            coo = cpu[direction].to_sparse_coo().coalesce()
            after[direction] = verify_cache(coo, caches[direction], PROTOCOL["verification_chunk_edges"])
            del coo
        result["cache_after"] = after
        result["raw_counts_sha256_after"] = raw_counts_hash()
        result["input_sha256_after"] = {name: array_sha256(value) for name, value in inputs.items()}
        result["selected_sha256_after"] = array_sha256(selected)
        observed = observed_source_files()
        result["observed_imports_sha256"] = observed
        result["integrity"] = {
            "raw_graph_unchanged": result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"],
            "cache_unchanged": all(all(after[d][key] == cache_evidence[d][key] for key in
                                       ("indices_sha256", "values_sha256", "nnz", "shape")) for d in PROTOCOL["directions"]),
            "inputs_unchanged": result["input_sha256_after"] == result["input_sha256_before"]
                                and result["selected_sha256_after"] == result["selected_sha256_before"],
            "source_unchanged": implementation_hashes() == registration["implementation_sha256"],
            "source_closure_complete": all(registration["implementation_sha256"].get(name) == digest for name, digest in observed.items()),
            "graph_file_unchanged": file_sha256(Path(registration["graph_path"])) == registration["graph_file_sha256"],
        }
        result["integrity_pass"] = all(result["integrity"].values())
        result["numeric_pass"] = result["numeric_comparison_pass"] and result["autograd"]["numeric_pass"]
        result["mps_kernel_candidate"] = result["integrity_pass"] and result["numeric_pass"] and result["faster_all_inputs"]
        result["status"] = "protocol_complete" if result["integrity_pass"] else "failed"
        result["reason"] = "kernel_evidence_recorded" if result["integrity_pass"] else "integrity_failure"
        result["interpretation"] = "Complete graph kernel evidence only. No SNN window, optimizer, migration or flight was run."
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        # Even a sampling error must leave worker evidence instead of masking the original failure.
        if sampler is not None:
            try:
                result["mps_memory"] = sampler.close()
                if result["mps_memory"]["error"] is not None:
                    raise RuntimeError(result["mps_memory"]["error"])
            except BaseException as error:
                result["status"], result["mps_kernel_candidate"] = "failed", False
                result["reason"] = "memory_sampling_failure"
                result["memory_sampling_error"] = f"{type(error).__name__}:{error}"
        now = time.perf_counter()
        if current is not None:
            phases[current] = phases.get(current, 0.0) + now - phase_started
        result["phases_seconds"], result["wall_seconds"] = phases, now - started
        write_json(directory / "phase.json", {"phase": "finished", "wall_seconds": result["wall_seconds"]})
        result["artifacts_sha256"] = {path.name: file_sha256(path) for path in sorted(directory.iterdir())
                                      if path.is_file() and path.suffix in (".npz", ".csv")}
        write_json(directory / "worker_result.json", result)
    return result
