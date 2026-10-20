"""Low-cost algorithm/archive fixtures, never labelled as a biological reference."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from malecns_lab.brain.connectome import Connectome
from malecns_lab.brain.metadata import annotation_hash
from malecns_lab.brain.reference import graph_hash
from malecns_lab.learning.graph_controls import _clone, _rewire
from malecns_lab.learning.graph_diagnostics import validate_rw, policy_path_vectors, incoming_strength_vectors
from malecns_lab.learning.rewiring_steps import advance, counters, restore_rng, report
from malecns_lab.learning.rewiring_archive import save_checkpoint, restore_checkpoint, state_identity
from malecns_lab.learning.rewiring_contract import canonical, sha
from malecns_lab.learning.rewiring_evidence import initial_state, checkpoint_context, scan_run, evidence_hashes, validate_completion
from malecns_lab.learning.generate_rewiring import register_segment, finalize_segment
from malecns_lab.learning.resource_guard import write_json


def fixture_graph():
    n = 9
    sources = np.repeat(np.arange(8), 4)
    targets = np.asarray([(i + d) % 8 for i in range(8) for d in (0, 1, 3, 5)])
    signs = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1], np.int8)
    values = np.tile([1, 2, 3, 4], 8) * signs[sources]
    matrix = sparse.csc_matrix((values, (targets, sources)), shape=(n, n), dtype=np.int64)
    types = np.asarray(["R1-R6", "a", "b", "", "c", "d", "e", "f", "isolated"])
    superclass = np.array(["fixture"] * 7 + ["descending_neuron", "descending_neuron"])
    ann = {"type": types, "superclass": superclass, "nt_sign": signs}
    missing = {k: np.zeros(n, bool) for k in ann}
    missing["type"][3] = True
    return Connectome("algorithm_fixture", matrix, types, np.full(n, ""), superclass, np.arange(n) + 100,
                      annotations=ann, annotation_missing=missing, groups={"empty": np.array([], np.int64)},
                      meta={"roles": {"empty": "input"}})


def fixture_run(graph, *, max_attempts=100, target_swaps=100):
    return {"run_id": "numerical-fixture-only", "mother": {"source_id": "numeric-fixture",
            "graph_sha256": graph_hash(graph), "provenance": {"node_metadata_sha256": annotation_hash(graph)}},
            "protocol": {"seed": 1901, "max_attempts": max_attempts, "target_swaps": target_swaps,
                         "max_segment_proposals": 1000000}, "runtime": {"fixture": True}, "implementation_sha256": {}}


def old_reference(matrix, rng, state, attempts, successes):
    # Independent frozen pre-extraction algorithm, kept only as an equivalence oracle.
    def contains(source, target):
        a, b = matrix.indptr[source:source + 2]
        pos = a + np.searchsorted(matrix.indices[a:b], target)
        return pos < b and matrix.indices[pos] == target
    def move(source, pos, target):
        a, b = matrix.indptr[source:source + 2]
        ins = a + int(np.searchsorted(matrix.indices[a:b], target))
        weight = matrix.data[pos]
        if ins > pos:
            ins -= 1
            matrix.indices[pos:ins] = matrix.indices[pos + 1:ins + 1]
            matrix.data[pos:ins] = matrix.data[pos + 1:ins + 1]
        else:
            matrix.indices[ins + 1:pos + 1] = matrix.indices[ins:pos]
            matrix.data[ins + 1:pos + 1] = matrix.data[ins:pos]
        matrix.indices[ins], matrix.data[ins] = target, weight
    while matrix.nnz and state["attempts"] < attempts and state["successful_swaps"] < successes:
        state["attempts"] += 1
        a, b = rng.integers(matrix.nnz, size=2)
        u, w = np.searchsorted(matrix.indptr, [a, b], side="right") - 1
        v, x = matrix.indices[[a, b]]
        reason = ("same_source" if u == w else "self_edge" if u == v or w == x else
                  "different_source_sign" if np.sign(matrix.data[a]) != np.sign(matrix.data[b]) else
                  "same_target" if v == x else "new_self_edge" if u == x or w == v else
                  "duplicate" if contains(u, x) or contains(w, v) else None)
        if reason:
            state["rejections"][reason] += 1
        else:
            move(u, a, x)
            move(w, b, v)
            state["successful_swaps"] += 1


def test_original_kernel_continuous_and_true_os_resume_exact(tmp_path):
    parent, rng, count = fixture_graph(), np.random.default_rng(1901), counters()
    run = fixture_run(parent, max_attempts=501, target_swaps=120)
    original, oracle_rng, oracle_count = _clone(parent), np.random.default_rng(1901), counters()
    old_reference(original.weights, oracle_rng, oracle_count, 501, 120)
    whole, measurements = _rewire(parent, seed=1901, max_attempts=501, target_swaps=120)
    assert graph_hash(whole) == graph_hash(original) and measurements["rejections"] == oracle_count["rejections"]
    graph = _clone(parent)
    advance(graph.weights, rng, count, max_attempts=501, target_swaps=120, proposals=137)
    context = {"fixture": "first"}
    first = save_checkpoint(tmp_path / "first", graph=graph, rng=rng, counts=count, run=run, context=context)
    (tmp_path / "run.json").write_text(json.dumps(run))
    # A real new Python process reconstructs its own graph and consumes the persisted RNG/CSC.
    code = '''import json,sys,numpy as np
from pathlib import Path
from test_bounded_rewiring import fixture_graph
from malecns_lab.learning.rewiring_archive import restore_checkpoint,save_checkpoint
from malecns_lab.learning.rewiring_steps import advance
p=Path(sys.argv[1]);run=json.loads((p/'run.json').read_text())
g,r,c,m=restore_checkpoint(p/'first',parent=fixture_graph(),run=run,context={'fixture':'first'})
advance(g.weights,r,c,max_attempts=501,target_swaps=120,proposals=364)
save_checkpoint(p/'second',graph=g,rng=r,counts=c,run=run,context={'fixture':'second'})
'''
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve()) + os.pathsep + str(Path("tests").resolve())}
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    resumed, resumed_rng, resumed_count, _ = restore_checkpoint(tmp_path / "second", parent=parent, run=run,
                                                               context={"fixture": "second"})
    assert graph_hash(resumed) == graph_hash(original)
    assert resumed_rng.bit_generator.state == oracle_rng.bit_generator.state and resumed_count == oracle_count
    assert first["state"]["counts"]["attempts"] == 137
    validate_rw(parent, resumed, "R")


def test_boundaries_zero_and_every_rejection():
    parent = fixture_graph()
    for budget, goal in ((0, 50), (1000, 0), (1000, 3), (3, 1000)):
        graph, stats = _rewire(parent, seed=1901, max_attempts=budget, target_swaps=goal)
        assert stats["attempts"] <= budget and stats["successful_swaps"] <= goal
        assert stats["attempts"] == stats["successful_swaps"] + sum(stats["rejections"].values())
    _, stats = _rewire(parent, seed=1901, max_attempts=5000, target_swaps=5000)
    assert all(value > 0 for value in stats["rejections"].values())
    empty = _clone(parent)
    empty.weights = sparse.csc_matrix((parent.n, parent.n), dtype=np.int64)
    assert _rewire(empty, seed=1, max_attempts=10, target_swaps=1)[1]["stop_reason"] == "no_edges"


def test_per_source_multiset_rejects_global_and_strength_preserving_counterexample():
    parent = fixture_graph()
    # Same-sign sources exchange [2,4] with [3,3]: equal edge count, sums and global histogram.
    for source, values in ((0, [1, 2, 4, 7]), (1, [1, 3, 3, 7])):
        a, b = parent.weights.indptr[source:source + 2]
        parent.weights.data[a:b] = values
    changed = _clone(parent)
    a, b = parent.weights.indptr[0:2]
    c, d = parent.weights.indptr[1:3]
    # Keep actual self positions fixed while swapping two nonself weights.
    first = np.flatnonzero(parent.weights.indices[a:b] != 0)[:2] + a
    second = np.flatnonzero(parent.weights.indices[c:d] != 1)[:2] + c
    parent.weights.data[first] = [2, 4]
    parent.weights.data[second] = [3, 3]
    changed = _clone(parent)
    changed.weights.data[first], changed.weights.data[second] = [3, 3], [2, 4]
    assert np.array_equal(parent.weights.sum(axis=0), changed.weights.sum(axis=0))
    with pytest.raises(ValueError, match="per-source weight multiset"):
        validate_rw(parent, changed, "R")


def test_directed_policy_paths_unreachable_and_complete_strengths():
    graph = fixture_graph()
    # Only 0 -> 1 -> 7. Isolated DN8 is unreachable; reverse biological direction would fail.
    graph.weights = sparse.csc_matrix(([2, -3], ([1, 7], [0, 1])), shape=(9, 9), dtype=np.int64)
    paths = policy_path_vectors(graph)
    assert paths["any_input_to_dn_hops"].tolist() == [2, -1]
    assert paths["input_to_any_dn_hops"].tolist() == [2]
    assert paths["output_body_ids"].tolist() == [107, 108]
    other = _clone(graph)
    other.weights.data[:] = [4, -5]
    strength = incoming_strength_vectors(graph, other)
    assert strength["delta_positive"].tolist() == [0, 2, 0, 0, 0, 0, 0, 0, 0]
    assert strength["delta_negative"][7] == -2 and strength["delta_absolute"][7] == 2
    assert len(strength["body_ids"]) == 9


def checkpoint_fixture(tmp_path):
    parent = fixture_graph()
    run = fixture_run(parent)
    rng, count = np.random.default_rng(1901), counters()
    directory = tmp_path / "checkpoint"
    meta = save_checkpoint(directory, graph=parent, rng=rng, counts=count, run=run, context={"parent": None})
    return parent, run, directory, meta


@pytest.mark.parametrize("change", ["mother", "runtime", "source", "rng", "boolean", "conservation", "parent", "array"])
def test_corrupt_checkpoint_rejected(tmp_path, change):
    parent, run, directory, meta = checkpoint_fixture(tmp_path)
    if change == "mother":
        run["mother"]["source_id"] = "wrong"
    elif change == "runtime":
        run["runtime"] = {"changed": True}
    elif change == "source":
        run["implementation_sha256"] = {"wrong": "sha"}
    elif change == "rng":
        meta["state"]["rng"]["has_uint32"] = True
    elif change == "boolean":
        meta["state"]["counts"]["attempts"] = False
    elif change == "conservation":
        meta["state"]["counts"]["attempts"] = 3
    elif change == "parent":
        meta["context"] = {"parent": "wrong"}
    elif change == "array":
        with (directory / "arrays.npz").open("ab") as stream:
            stream.write(b"corruption")
    meta["state"]["state_id"] = canonical({k: v for k, v in meta["state"].items() if k != "state_id"})
    meta["checkpoint_id"] = canonical({k: v for k, v in meta.items() if k != "checkpoint_id"})
    write_json(directory / "checkpoint.json", meta)
    with pytest.raises(ValueError):
        restore_checkpoint(directory, parent=parent, run=run, context={"parent": None})


def test_manifest_last_and_intermediate_rejects_production(tmp_path, monkeypatch):
    from malecns_lab.brain.brain import load_connectome
    parent, run, directory, meta = checkpoint_fixture(tmp_path)
    with pytest.raises((ValueError, KeyError)):
        load_connectome(directory / "arrays.npz")
    def fail(*args, **kwargs):
        raise OSError("interrupted save")
    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path / "broken", graph=parent, rng=np.random.default_rng(1901), counts=counters(), run=run, context={})
    assert not (tmp_path / "broken" / "checkpoint.json").exists()
    assert (tmp_path / "broken" / "arrays.npz.tmp").exists()


def simulated_segment(tmp_path):
    parent, run = fixture_graph(), fixture_run(fixture_graph(), max_attempts=1000, target_swaps=1000)
    chain = scan_run(tmp_path, run)
    segment, registration = register_segment(tmp_path, run, chain, 20)
    rng, count, graph = np.random.default_rng(1901), counters(), _clone(parent)
    write_json(segment / "resume.json", {"run_id": run["run_id"], "segment_id": registration["segment_id"],
        "parent_checkpoint_id": None, "actual_state": initial_state(run), "verified_before_proposals": True})
    advance(graph.weights, rng, count, max_attempts=1000, target_swaps=1000, proposals=20)
    meta = save_checkpoint(segment / "checkpoint", graph=graph, rng=rng, counts=count, run=run,
                           context=checkpoint_context(registration))
    files = ["resume.json", "checkpoint/checkpoint.json", "checkpoint/arrays.npz"]
    write_json(segment / "worker_result.json", {"status": "completed", "run_id": run["run_id"],
        "segment_id": registration["segment_id"], "checkpoint_id": meta["checkpoint_id"], "state_id": meta["state"]["state_id"],
        "terminal": False, "files_sha256": {f: sha(segment / f) for f in files}})
    (segment / "resources.csv").write_text("wall_seconds,tree_rss_bytes,system_available_bytes\n0.1,1000,4294967296\n")
    outcome = {"status": "completed", "reason": "worker_exited", "returncode": 0, "worker_status": "completed",
               "wall_seconds": .2, "peak_observed_rss_bytes": 1000}
    write_json(segment / "result.json", outcome)
    return run, segment, registration, outcome


def test_parent_chain_corruption_and_terminal_reject(tmp_path, monkeypatch):
    monkeypatch.setattr("malecns_lab.learning.generate_rewiring.validate_run", lambda run: None)
    run, segment, registration, outcome = simulated_segment(tmp_path)
    result = finalize_segment(segment, run, registration, outcome)
    assert result["status"] == "completed"
    chain = scan_run(tmp_path, run)
    assert chain["state"]["counts"]["attempts"] == 20
    chain["terminal"] = True
    with pytest.raises(ValueError, match="Final candidate"):
        register_segment(tmp_path, run, chain, 20)
    registration["parent_checkpoint_id"] = "invented"
    registration["segment_id"] = canonical({k: v for k, v in registration.items() if k != "segment_id"})
    write_json(segment / "registration.json", registration)
    with pytest.raises(ValueError, match="start/parent/prefix"):
        scan_run(tmp_path, run)


@pytest.mark.parametrize("failure", ["rss_limit", "wall_limit", "missing_artifact", "bad_resume", "zero_exit_only"])
def test_failed_segment_never_adopts_checkpoint_or_final_npz(tmp_path, failure, monkeypatch):
    monkeypatch.setattr("malecns_lab.learning.generate_rewiring.validate_run", lambda run: None)
    run, segment, registration, outcome = simulated_segment(tmp_path)
    (segment / "candidate.npz").write_bytes(b"unapproved candidate")
    if failure in {"rss_limit", "wall_limit"}:
        outcome.update(status="failed", reason=failure)
    elif failure == "missing_artifact":
        (segment / "checkpoint" / "arrays.npz").unlink()
    elif failure == "bad_resume":
        data = json.loads((segment / "resume.json").read_text())
        data["verified_before_proposals"] = False
        write_json(segment / "resume.json", data)
    else:
        write_json(segment / "worker_result.json", {"status": "failed"})
    result = finalize_segment(segment, run, registration, outcome)
    assert result["status"] == "failed" and result["accepted"] is None
    chain = scan_run(tmp_path, run)
    assert chain["checkpoint_id"] is None and chain["state"]["counts"]["attempts"] == 0
    assert (segment / "candidate.npz").exists()
    following, reg = register_segment(tmp_path, run, chain, 20)
    assert reg["segment"] == 2 and reg["parent_checkpoint_id"] is None


@pytest.mark.parametrize("corruption", ["duplicate_target", "out_of_range", "zero_weight", "wrong_dtype", "pointer"])
def test_structural_csc_corruption_rejected_even_after_file_and_array_hashes_resealed(tmp_path, corruption):
    from malecns_lab.learning.rewiring_archive import array_description
    parent, run, directory, meta = checkpoint_fixture(tmp_path)
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if corruption == "duplicate_target":
        arrays["indices"][1] = arrays["indices"][0]
    elif corruption == "out_of_range":
        arrays["indices"][0] = parent.n
    elif corruption == "zero_weight":
        arrays["data"][0] = 0
    elif corruption == "wrong_dtype":
        arrays["data"] = arrays["data"].astype(np.float64)
    else:
        arrays["indptr"][0] = 1
    np.savez(directory / "arrays.npz", **arrays)
    meta["arrays"] = {k: array_description(v) for k, v in arrays.items()}
    meta["arrays_sha256"] = sha(directory / "arrays.npz")
    meta["checkpoint_id"] = canonical({k: v for k, v in meta.items() if k != "checkpoint_id"})
    write_json(directory / "checkpoint.json", meta)
    with pytest.raises(ValueError, match="CSC"):
        restore_checkpoint(directory, parent=parent, run=run, context={"parent": None})


def test_terminal_parent_approval_required_and_bound_to_every_file(tmp_path, monkeypatch):
    from malecns_lab.learning.generate_rewiring import accepted_candidate
    monkeypatch.setattr("malecns_lab.learning.generate_rewiring.validate_run", lambda run: None)
    run, segment, registration, outcome = simulated_segment(tmp_path)
    # Exercise terminal evidence admission only; these opaque files are NOT graph/model fixtures.
    run["protocol"].update(max_attempts=20, nodes=9, edges=32, synapses=80, self_edges=8, inputs=1, outputs=2)
    write_json(tmp_path / "run.json", run)
    meta = json.loads((segment / "checkpoint" / "checkpoint.json").read_text())
    assert meta["state"]["counts"]["successful_swaps"] > 0
    for filename in ("candidate.npz", "mechanical_vectors.npz"):
        (segment / filename).write_bytes(b"opaque identity admission fixture; not a graph")
    write_json(segment / "diagnostics.json", {"scope": "admission fixture only"})
    proof = {"status": "validated_research_control", "run_id": run["run_id"], "checkpoint_id": meta["checkpoint_id"],
             "counts": meta["state"]["counts"], "graph_sha256": meta["state"]["graph_sha256"],
             "candidate_sha256": sha(segment / "candidate.npz"), "node_count": 9, "edge_count": 32,
             "synapse_count": 80, "self_edges": 8, "inputs": 1, "outputs": 2,
             "per_source_weight_multiset": True, "actual_reloaded": True}
    write_json(segment / "candidate_verification.json", proof)
    worker = json.loads((segment / "worker_result.json").read_text())
    worker["terminal"] = True
    for name in ("candidate.npz", "mechanical_vectors.npz", "diagnostics.json", "candidate_verification.json"):
        worker["files_sha256"][name] = sha(segment / name)
    write_json(segment / "worker_result.json", worker)
    with pytest.raises(FileNotFoundError):
        accepted_candidate(tmp_path)  # Complete worker files do not substitute for parent's terminal seal.
    assessed = finalize_segment(segment, run, registration, outcome)
    assert assessed["status"] == "completed" and assessed["accepted"]["terminal"]
    final = accepted_candidate(tmp_path)
    assert final["path"] == str(segment / "candidate.npz")
    with pytest.raises(ValueError, match="Final candidate"):
        register_segment(tmp_path, run, scan_run(tmp_path, run), 1)
    (segment / "candidate.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        accepted_candidate(tmp_path)


def test_parent_rechecks_source_after_worker_exit(tmp_path, monkeypatch):
    run, segment, registration, outcome = simulated_segment(tmp_path)
    def changed(run):
        raise ValueError("Generation source/runtime changed")
    monkeypatch.setattr("malecns_lab.learning.generate_rewiring.validate_run", changed)
    result = finalize_segment(segment, run, registration, outcome)
    assert result["status"] == "failed" and "source/runtime changed" in result["reason"]
    assert scan_run(tmp_path, run)["checkpoint_id"] is None
