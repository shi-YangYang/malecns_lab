"""Supervised full-graph worker for one registered R proposal segment."""
from __future__ import annotations

from pathlib import Path
import time
import traceback

from .resource_guard import write_json
from .rewiring_contract import boundary, read_json, require, sha, validate_run
from .rewiring_evidence import checkpoint_context, scan_run, validate_registration


def run_worker(segment):
    segment = Path(segment).resolve()
    started = time.monotonic()
    def phase(name):
        write_json(segment / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - started})
    try:
        import numpy as np
        from ..brain.reference import load_reference, graph_hash
        from .graph_controls import _clone, _stamp, load_research_control
        from .graph_diagnostics import check_matrix, validate_rw, graph_diagnostics, complete_policy_diagnostics
        from .rewiring_archive import restore_checkpoint, save_checkpoint, state_identity
        from .rewiring_steps import advance, counters, report
        phase("validate_registration")
        run = read_json(segment.parents[1] / "run.json")
        validate_run(run)
        chain = scan_run(segment.parents[1], run, stop_before=segment.name)
        registration = read_json(segment / "registration.json")
        validate_registration(registration, run, chain)
        protocol = run["protocol"]
        phase("load_complete_mother")
        parent = load_reference(run["mother"]["path"])
        check_matrix(parent)
        require(parent.n == protocol["nodes"] and parent.n_connections == protocol["edges"]
                and parent.n_synapses == protocol["synapses"]
                and int(np.count_nonzero(parent.weights.diagonal())) == protocol["self_edges"], "Complete mother dimensions differ")
        require(int((parent.annotations["type"] == "R1-R6").sum()) == protocol["inputs"]
                and int((parent.annotations["superclass"] == "descending_neuron").sum()) == protocol["outputs"],
                "Complete policy selections differ")
        phase("restore_actual_state")
        if chain["checkpoint_path"]:
            previous = read_json(Path(chain["checkpoint_path"]) / "checkpoint.json")
            graph, rng, counts, restored = restore_checkpoint(chain["checkpoint_path"], parent=parent, run=run,
                                                               context=previous["context"])
            require(restored["checkpoint_id"] == chain["checkpoint_id"], "Restored parent checkpoint differs")
        else:
            graph, rng, counts = _clone(parent), np.random.default_rng(protocol["seed"]), counters()
        actual = state_identity(graph, rng, counts, run["mother"])
        require(actual == registration["start_state"], "Actual pre-proposal restoration differs")
        write_json(segment / "resume.json", {"run_id": run["run_id"], "segment_id": registration["segment_id"],
            "parent_checkpoint_id": registration["parent_checkpoint_id"], "actual_state": actual, "verified_before_proposals": True})
        phase("proposals")
        advance(graph.weights, rng, counts, max_attempts=protocol["max_attempts"], target_swaps=protocol["target_swaps"],
                proposals=registration["proposals"])
        phase("validate_complete_mechanics")
        validate_rw(parent, graph, "R")
        phase("seal_state")
        checkpoint = save_checkpoint(segment / "checkpoint", graph=graph, rng=rng, counts=counts, run=run,
                                     context=checkpoint_context(registration))
        terminal = boundary(counts, protocol)
        paths = ["resume.json", "checkpoint/checkpoint.json", "checkpoint/arrays.npz"]
        if terminal:
            require(counts["successful_swaps"] > 0, "Zero successful swaps cannot form a topology counterfactual")
            phase("final_diagnostics")
            diagnostics = graph_diagnostics(parent, graph, paths=False)
            details, vectors = complete_policy_diagnostics(parent, graph)
            diagnostics["complete_policy"] = details
            diagnostics["transformation"] = report(graph.weights, counts, max_attempts=protocol["max_attempts"],
                                                    target_swaps=protocol["target_swaps"])
            np.savez(segment / "mechanical_vectors.npz", **vectors)
            diagnostics["complete_vectors_sha256"] = sha(segment / "mechanical_vectors.npz")
            write_json(segment / "diagnostics.json", diagnostics)
            _stamp(parent, graph, transform="R", seed=protocol["seed"], protocol=run["run_id"],
                   parameters={"max_attempts": protocol["max_attempts"], "target_swaps": protocol["target_swaps"],
                       "self_edges": "fixed", "generation_run_id": run["run_id"], "generation_checkpoint_id": checkpoint["checkpoint_id"],
                       "generation_terminal": True, "with_paths": "complete_policy_dual_multisource"}, diagnostics=diagnostics)
            phase("seal_candidate")
            graph.save(segment / "candidate.npz", overwrite=False)
            del graph, vectors
            phase("reload_candidate")
            loaded = load_research_control(segment / "candidate.npz", parent)
            require(graph_hash(loaded) == checkpoint["state"]["graph_sha256"], "Final candidate differs from final state")
            write_json(segment / "candidate_verification.json", {"status": "validated_research_control", "run_id": run["run_id"],
                "checkpoint_id": checkpoint["checkpoint_id"], "counts": counts, "source_id": loaded.meta["source_id"],
                "graph_sha256": graph_hash(loaded), "candidate_sha256": sha(segment / "candidate.npz"),
                "node_count": loaded.n, "edge_count": loaded.n_connections, "synapse_count": loaded.n_synapses,
                "self_edges": int(np.count_nonzero(loaded.weights.diagonal())), "inputs": protocol["inputs"],
                "outputs": protocol["outputs"], "per_source_weight_multiset": True, "actual_reloaded": True})
            paths += ["candidate.npz", "diagnostics.json", "mechanical_vectors.npz", "candidate_verification.json"]
        phase("completed")
        result = {"status": "completed", "run_id": run["run_id"], "segment_id": registration["segment_id"],
                  "checkpoint_id": checkpoint["checkpoint_id"], "state_id": checkpoint["state"]["state_id"],
                  "terminal": terminal, "files_sha256": {path: sha(segment / path) for path in paths},
                  "wall_seconds": time.monotonic() - started}
    except Exception as error:
        traceback.print_exc()
        result = {"status": "failed", "reason": f"{type(error).__name__}:{error}", "wall_seconds": time.monotonic() - started}
    write_json(segment / "worker_result.json", result)
    return result
