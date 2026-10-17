"""Algorithm fixtures only; these small graphs are never brain/runtime substitutes."""

import copy

import numpy as np
import pytest
from scipy import sparse

from cnsbridge.brain.brain import load_connectome
from cnsbridge.brain.connectome import Connectome
from cnsbridge.brain.metadata import annotation_hash, canonical_hash, resolve_reference_groups
from cnsbridge.brain.reference import graph_hash, load_reference
from cnsbridge.learning.graph_controls import (
    _rewire,
    _shuffle,
    create_intervention_pair,
    create_research_control,
    load_research_control,
)
from cnsbridge.learning.graph_diagnostics import edge_overlap, input_output_paths, validate_rw


def _fixture_graph(offset=0):
    types = np.asarray(["DNg02"] * 29 + ["DNp03"] * 2 + ["DNp01"] * 2 + ["R1-R6"] * 2 + ["T4a"] * 6)
    sides = np.asarray(["L"] * 15 + ["R"] * 14 + ["L", "R", "L", "R"] + ["L", "R"] + ["L", "R"] * 3)
    n = len(types)
    sources = np.repeat(np.arange(n), 5)
    targets = np.asarray([(source + d) % n for source in range(n) for d in (0, 1, 7, 19, 27)])
    signs = np.where(np.arange(n) % 2, -1, 1).astype(np.int8)
    values = np.tile([1, 2, 3, 1, 2], n) * signs[sources]
    values[0] += offset
    matrix = sparse.csc_matrix((values, (targets, sources)), shape=(n, n), dtype=np.int64)
    annotations = {"type": types.copy(), "somaSide": sides.copy(), "rootSide": sides.copy(),
                   "subclass": np.full(n, ""), "superclass": np.full(n, "fixture"), "nt_sign": signs}
    graph = Connectome("algorithm-fixture-not-biological-evidence", matrix, types, sides,
                        annotations["superclass"], np.arange(n) + 1000,
                        annotations=annotations, annotation_missing={key: np.zeros(n, bool) for key in annotations})
    mapping = resolve_reference_groups(graph)
    provenance = {"graph_sha256": graph_hash(graph), "node_metadata_sha256": annotation_hash(graph),
                  "group_mapping": mapping, "fixture_only": True}
    graph.meta = {"format_version": 2, "asset_type": "full_reference", "source_id": canonical_hash(provenance),
                  "provenance": provenance, "roles": {name: rule["role"] for name, rule in mapping["rules"].items()}}
    return graph


@pytest.fixture
def reference(tmp_path):
    graph = _fixture_graph()
    path = graph.save(tmp_path / "parent.npz", overwrite=False)
    return graph, path


def test_rewire_preserves_exact_invariants_and_parent(reference):
    parent, _ = reference
    before = graph_hash(parent)
    graph, measurements = _rewire(parent, seed=7, max_attempts=1000, target_swaps=100)
    repeated, repeated_measurements = _rewire(parent, seed=7, max_attempts=1000, target_swaps=100)
    validate_rw(parent, graph, "R")
    assert graph_hash(graph) == graph_hash(repeated)
    assert measurements == repeated_measurements
    assert measurements["successful_swaps"] == 100
    assert measurements["attempts"] == 100 + sum(measurements["rejections"].values())
    assert edge_overlap(parent, graph)["retained_edge_fraction"] < 1
    assert graph_hash(parent) == before
    assert np.array_equal(parent.weights.diagonal(), graph.weights.diagonal())
    for source in range(parent.n):
        start, stop = parent.weights.indptr[source:source + 2]
        assert np.array_equal(np.sort(parent.weights.data[start:stop]), np.sort(graph.weights.data[start:stop]))


def test_rewire_zero_success_and_noops_are_explicit(reference):
    parent, _ = reference
    graph, report = _rewire(parent, seed=0, max_attempts=0, target_swaps=20)
    assert graph_hash(parent) == graph_hash(graph)
    assert report["attempts"] == report["successful_swaps"] == 0
    assert report["target_reached"] is False
    self_only = copy.deepcopy(parent)
    self_only.weights = sparse.diags(parent.annotations["nt_sign"], format="csc", dtype=np.int64)
    graph, report = _rewire(self_only, seed=0, max_attempts=20, target_swaps=1)
    assert report["successful_swaps"] == 0
    assert graph_hash(graph) == graph_hash(self_only)
    empty = copy.deepcopy(parent)
    empty.weights = sparse.csc_matrix(parent.weights.shape, dtype=np.int64)
    _, report = _rewire(empty, seed=0, max_attempts=20, target_swaps=1)
    assert report["stop_reason"] == "no_edges"
    assert report["effective_edge_moves_per_nonself_edge"] is None


def test_shuffle_keeps_topology_and_self_weights(reference):
    parent, _ = reference
    graph, report = _shuffle(parent, seed=22)
    validate_rw(parent, graph, "W")
    assert report["changed_weight_edges"] > 0
    assert edge_overlap(parent, graph)["retained_edge_fraction"] == 1
    assert np.array_equal(parent.weights.diagonal(), graph.weights.diagonal())
    altered = copy.deepcopy(graph)
    altered.weights.data[0] += 1
    with pytest.raises(ValueError):
        validate_rw(parent, altered, "W")


def test_path_diagnostic_direction_and_output_ids(reference):
    parent, _ = reference
    paths = input_output_paths(parent)
    assert paths["output_body_ids"] == list(range(1000, 1033))
    assert paths["groups"]["R16_L"]["reachable_output_neurons"] == 33
    assert paths["groups"]["HAL_L"]["reachable_output_neurons"] == 0
    assert paths["groups"]["R16_L"]["output_distances"][0] > 0


@pytest.mark.parametrize("transform", ["R", "W"])
def test_builder_offline_loader_identity_overwrite_and_production_rejection(reference, tmp_path, transform):
    parent, path = reference
    output = tmp_path / "control.npz"
    args = {"max_attempts": 100, "target_swaps": 20} if transform == "R" else {}
    report = create_research_control(path, output, transform=transform, seed=101,
                                     protocol="fixture-mechanical-only", **args)
    loaded = load_research_control(output, path)
    assert loaded.meta["asset_type"] == "research_control"
    assert report["diagnostics"]["paths"]["status"] == "not_executed"
    assert report["diagnostics"]["mixing"]["status"] == "not_established"
    assert loaded.meta["provenance"]["parent_source_id"] == parent.meta["source_id"]
    assert graph_hash(load_reference(path)) == graph_hash(parent)
    with pytest.raises(FileExistsError):
        create_research_control(path, output, transform=transform, seed=101, protocol="fixture", **args)
    with pytest.raises(ValueError, match="research_control|Research"):
        load_connectome(output)
    wrong_parent = _fixture_graph(offset=1)
    with pytest.raises(ValueError, match="different full reference"):
        load_research_control(output, wrong_parent)
    loaded.weights.data[0] += 1
    corrupt = loaded.save(tmp_path / "corrupt.npz", overwrite=False)
    with pytest.raises(ValueError, match="content identity"):
        load_research_control(corrupt, path)


def test_no_implicit_swap_budget_or_research_parent(reference, tmp_path):
    _, path = reference
    with pytest.raises(ValueError, match="max_attempts"):
        create_research_control(path, tmp_path / "missing.npz", transform="R", seed=0, protocol="fixture")
    output = tmp_path / "control.npz"
    create_research_control(path, output, transform="W", seed=0, protocol="fixture")
    with pytest.raises(ValueError, match="full_reference"):
        create_research_control(output, tmp_path / "chained.npz", transform="W", seed=0, protocol="fixture")


@pytest.mark.parametrize("include_self", [True, False])
def test_intervention_pair_exact_weight_matching_and_protocol(reference, tmp_path, include_self):
    parent, path = reference
    directory = tmp_path / "pair"
    result = create_intervention_pair(path, directory, source_body_ids=[1000], include_self=include_self,
                                      seed=3, protocol="source-1000-before-scores", intervention_mode="acute")
    target = load_research_control(result["assets"]["P_target"]["path"], path)
    matched = load_research_control(result["assets"]["P_matched"]["path"], path)
    summary = result["assets"]["P_target"]["diagnostics"]["removal"]
    match_summary = result["assets"]["P_matched"]["diagnostics"]["removal"]
    assert summary["signed_weight_histogram"] == match_summary["signed_weight_histogram"]
    assert summary["removed_edges"] == (5 if include_self else 4)
    assert np.array_equal(matched.weights[:, 0].toarray(), parent.weights[:, 0].toarray())
    assert target.weights[0, 0] == (0 if include_self else parent.weights[0, 0])
    for graph in (target, matched):
        params = graph.meta["provenance"]["parameters"]
        assert params["pair_id"] == result["pair_id"]
        assert params["fit_status"] == "not_executed"
        assert params["intervention_mode"] == "acute"
        assert graph.n == parent.n
        assert graph.meta["asset_type"] == "research_control"
    assert graph_hash(load_reference(path)) == graph_hash(parent)
    with pytest.raises(FileExistsError):
        create_intervention_pair(path, directory, source_body_ids=[1000], include_self=include_self,
                                  seed=3, protocol="fixture", intervention_mode="acute")


def test_intervention_insufficient_exact_match_fails_without_relaxing(reference, tmp_path):
    parent, path = reference
    positive_sources = parent.body_ids[parent.annotations["nt_sign"] > 0].tolist()
    with pytest.raises(ValueError, match="no relaxation"):
        create_intervention_pair(path, tmp_path / "unmatchable", source_body_ids=positive_sources,
                                  include_self=True, seed=3, protocol="fixture", intervention_mode="retrained")
    assert not (tmp_path / "unmatchable").exists()


def test_research_metadata_tampering_rejected(reference, tmp_path):
    _, path = reference
    output = tmp_path / "control.npz"
    create_research_control(path, output, transform="W", seed=0, protocol="fixture")
    graph = Connectome.load(output)
    graph.groups["DNg02_L"] = graph.groups["DNg02_L"][1:]
    corrupt = graph.save(tmp_path / "changed-probes.npz", overwrite=False)
    with pytest.raises(ValueError, match="groups"):
        load_research_control(corrupt, path)
