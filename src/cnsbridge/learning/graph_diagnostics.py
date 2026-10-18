"""Mechanical graph checks; no neural simulation, fitting or hardware access."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from ..brain.connectome import Connectome


def require_same_nodes(parent: Connectome, graph: Connectome) -> None:
    """Require the entire ordered neuron universe, nullable metadata and probes."""
    for name in ("body_ids", "types", "sides", "superclass"):
        if not np.array_equal(getattr(parent, name), getattr(graph, name)):
            raise ValueError(f"Research control changed node {name}")
    for name in ("annotations", "annotation_missing", "groups"):
        left, right = getattr(parent, name), getattr(graph, name)
        if left.keys() != right.keys() or any(not np.array_equal(left[key], right[key]) for key in left):
            raise ValueError(f"Research control changed {name}")
    if parent.weights.shape != graph.weights.shape:
        raise ValueError("Research controls must retain every neuron")


def check_matrix(graph: Connectome) -> None:
    matrix = graph.weights
    if not sparse.isspmatrix_csc(matrix) or not matrix.has_canonical_format:
        raise ValueError("Canonical CSC without duplicate edges is required")
    if not np.issubdtype(matrix.data.dtype, np.integer) or np.any(matrix.data == 0):
        raise ValueError("Nonzero exact integer synapse weights are required")
    signs = graph.annotations["nt_sign"]
    for start in range(0, matrix.nnz, 1_000_000):
        end = min(start + 1_000_000, matrix.nnz)
        sources = np.searchsorted(matrix.indptr, np.arange(start, end), side="right") - 1
        if np.any(np.sign(matrix.data[start:end]) != signs[sources]):
            raise ValueError("Edges must retain their source neuron sign")


def _degrees(matrix: sparse.csc_matrix) -> tuple:
    return (np.diff(matrix.indptr), np.bincount(matrix.indices, minlength=matrix.shape[0]),
            np.bincount(matrix.indices[matrix.data > 0], minlength=matrix.shape[0]),
            np.bincount(matrix.indices[matrix.data < 0], minlength=matrix.shape[0]))


def validate_rw(parent: Connectome, graph: Connectome, transform: str) -> None:
    """Exact discrete invariants, including signed incoming degree and autapses."""
    require_same_nodes(parent, graph)
    check_matrix(graph)
    old, new = parent.weights, graph.weights
    if transform not in {"R", "W"}:
        raise ValueError("Expected R or W")
    if any(not np.array_equal(a, b) for a, b in zip(_degrees(old), _degrees(new), strict=True)):
        raise ValueError("Research control changed per-node or signed incoming degrees")
    if not np.array_equal(old.diagonal(), new.diagonal()):
        raise ValueError("R/W must preserve original self edges and their weights")
    for absolute in (False, True):
        a, b = (abs(old), abs(new)) if absolute else (old, new)
        if not np.array_equal(np.asarray(a.sum(axis=0)), np.asarray(b.sum(axis=0))):
            raise ValueError("Research control changed per-source outgoing strength")
    for source in range(parent.n):
        a, b = old.indptr[source:source + 2], new.indptr[source:source + 2]
        if not np.array_equal(np.sort(old.data[a[0]:a[1]]), np.sort(new.data[b[0]:b[1]])):
            raise ValueError(f"Research control changed per-source weight multiset: {source}")
    old_values, old_counts = np.unique(old.data, return_counts=True)
    new_values, new_counts = np.unique(new.data, return_counts=True)
    if not np.array_equal(old_values, new_values) or not np.array_equal(old_counts, new_counts):
        raise ValueError("Research control changed the global signed weight multiset")
    if transform == "W" and (not np.array_equal(old.indices, new.indices)
                             or not np.array_equal(old.indptr, new.indptr)):
        raise ValueError("W must preserve topology")


def edge_overlap(parent: Connectome, graph: Connectome) -> dict:
    """Count common body pairs and changed weights in bounded column chunks."""
    old, new, n = parent.weights, graph.weights, parent.n
    common = same_weight = 0
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        a, b = slice(old.indptr[start], old.indptr[stop]), slice(new.indptr[start], new.indptr[stop])
        old_keys = np.repeat(np.arange(start, stop, dtype=np.int64), np.diff(old.indptr[start:stop + 1])) * n + old.indices[a]
        new_keys = np.repeat(np.arange(start, stop, dtype=np.int64), np.diff(new.indptr[start:stop + 1])) * n + new.indices[b]
        positions = np.searchsorted(old_keys, new_keys)
        valid = positions < old_keys.size
        matched = np.zeros(len(new_keys), bool)
        matched[valid] = old_keys[positions[valid]] == new_keys[valid]
        common += int(matched.sum())
        same_weight += int((old.data[a][positions[matched]] == new.data[b][matched]).sum())
    return {"parent_edges": old.nnz, "control_edges": new.nnz, "retained_edges": common,
            "added_edges": new.nnz - common, "removed_edges": old.nnz - common,
            "retained_edge_fraction": common / old.nnz if old.nnz else None,
            "changed_weight_on_common_edges": common - same_weight,
            "changed_weight_fraction_of_common_edges": (common - same_weight) / common if common else None}


def type_pair_changes(parent: Connectome, graph: Connectome, *, field: str = "type") -> dict:
    """Compare full type-pair edge counts without per-edge Python objects."""
    labels, codes = np.unique(parent.annotations[field], return_inverse=True)
    size = len(labels)
    histograms = []
    for matrix in (parent.weights, graph.weights):
        pairs = np.repeat(codes.astype(np.int64), np.diff(matrix.indptr)) * size + codes[matrix.indices]
        histograms.append(np.unique(pairs, return_counts=True))
    (keys_a, counts_a), (keys_b, counts_b) = histograms
    union = np.union1d(keys_a, keys_b)
    delta = np.zeros(len(union), dtype=np.int64)
    delta[np.searchsorted(union, keys_a)] -= counts_a
    delta[np.searchsorted(union, keys_b)] += counts_b
    changed = np.flatnonzero(delta)
    top = changed[np.argsort(-np.abs(delta[changed]), kind="stable")[:20]]
    return {"status": "measured", "annotation_field": field, "missing_labels": "empty string retained as category",
            "changed_type_pairs": int(changed.size), "edge_count_l1_change": int(np.abs(delta).sum()),
            "largest_changes": [{"source_type": str(labels[union[i] // size]),
                                 "target_type": str(labels[union[i] % size]), "edge_delta": int(delta[i])} for i in top]}


def input_output_paths(graph: Connectome) -> dict:
    """Exact directed unweighted multi-source distance per registered input group."""
    rules = graph.meta["roles"]
    outputs = np.unique(np.concatenate([graph.groups[name] for name, role in rules.items() if role == "output"]))
    matrix = graph.weights
    # W[post, pre] CSC has the same layout as A[pre, post] CSR.
    adjacency = sparse.csr_matrix((np.ones(matrix.nnz, np.uint8), matrix.indices, matrix.indptr), shape=matrix.shape)
    result = {}
    for name, role in rules.items():
        if role != "input":
            continue
        seeds = graph.groups[name]
        distances = (dijkstra(adjacency, directed=True, indices=seeds, unweighted=True, min_only=True)[outputs]
                     if len(seeds) else np.full(len(outputs), np.inf))
        result[name] = {"input_neurons": len(seeds), "reachable_output_neurons": int(np.isfinite(distances).sum()),
                        "output_distances": [int(value) if np.isfinite(value) else None for value in distances]}
    return {"status": "measured", "meaning": "shortest directed edge count from any group member; not functional propagation",
            "output_body_ids": graph.body_ids[outputs].tolist(), "groups": result,
            "haltere": "structural paths only; main experiment does not drive HAL"}


def graph_diagnostics(parent: Connectome, graph: Connectome, *, paths: bool) -> dict:
    diagonal = graph.weights.diagonal()
    return {"edge_overlap": edge_overlap(parent, graph), "type_pairs": type_pair_changes(parent, graph),
            "self_edges": int(np.count_nonzero(diagonal)),
            "self_edge_fraction": float(np.count_nonzero(diagonal) / graph.weights.nnz) if graph.weights.nnz else None,
            "paths": {"parent": input_output_paths(parent), "control": input_output_paths(graph)} if paths else {
                "status": "not_executed", "reason": "with_paths=False explicitly selected; no reachability claim",
                "method": "input_output_paths: exact directed unweighted multi-source shortest paths for each input group"},
            "neural_activity": {"status": "not_executed", "reason": "requires separate offline neural replay"},
            "mixing": {"status": "not_established", "reason": "mechanical validity and swap counts do not establish mixing"}}


def policy_path_vectors(graph: Connectome) -> dict:
    """Complete policy selections; two multi-source queries, never all pairs."""
    inputs = np.flatnonzero(graph.annotations["type"] == "R1-R6")
    outputs = np.flatnonzero(graph.annotations["superclass"] == "descending_neuron")
    matrix = graph.weights
    adjacency = sparse.csr_matrix((np.ones(matrix.nnz, np.uint8), matrix.indices, matrix.indptr), shape=matrix.shape)
    forward = (dijkstra(adjacency, directed=True, indices=inputs, unweighted=True, min_only=True)[outputs]
               if inputs.size else np.full(outputs.size, np.inf))
    reverse = (dijkstra(adjacency.T, directed=True, indices=outputs, unweighted=True, min_only=True)[inputs]
               if outputs.size else np.full(inputs.size, np.inf))
    def integral(values):
        result = np.full(values.shape, -1, np.int64)
        reachable = np.isfinite(values)
        result[reachable] = values[reachable].astype(np.int64)
        return result
    return {"input_body_ids": graph.body_ids[inputs], "output_body_ids": graph.body_ids[outputs],
            "input_to_any_dn_hops": integral(reverse), "any_input_to_dn_hops": integral(forward)}


def incoming_strength_vectors(parent: Connectome, graph: Connectome) -> dict:
    """Signed negative sums, positive sums and absolute sums for every target."""
    result = {"body_ids": parent.body_ids.copy()}
    for label, connectome in (("parent", parent), ("control", graph)):
        matrix = connectome.weights
        positive, negative = np.zeros(parent.n, np.int64), np.zeros(parent.n, np.int64)
        for start in range(0, matrix.nnz, 1_000_000):
            rows, weights = matrix.indices[start:start + 1_000_000], matrix.data[start:start + 1_000_000]
            np.add.at(positive, rows[weights > 0], weights[weights > 0])
            np.add.at(negative, rows[weights < 0], weights[weights < 0])
        result[label + "_positive"] = positive
        result[label + "_negative"] = negative
        result[label + "_absolute"] = positive - negative
    for field in ("positive", "negative", "absolute"):
        result["delta_" + field] = result["control_" + field] - result["parent_" + field]
    return result


def complete_policy_diagnostics(parent: Connectome, graph: Connectome) -> tuple[dict, dict]:
    """Return small explanation and complete NPZ-ready mechanical evidence."""
    vectors = incoming_strength_vectors(parent, graph)
    summary = {"incoming_strength": {}, "paths": {},
               "paths_meaning": "any input to each DN; each input to any DN using reverse adjacency; -1 unreachable; not all pairs",
               "selection": "type == R1-R6; superclass == descending_neuron; original body ID order"}
    for field in ("positive", "negative", "absolute"):
        delta = vectors["delta_" + field]
        summary["incoming_strength"][field] = {"changed_targets": int(np.count_nonzero(delta)),
            "delta_min": int(delta.min(initial=0)), "delta_max": int(delta.max(initial=0)),
            "delta_l1": int(np.abs(delta).sum(dtype=np.int64))}
    for label, connectome in (("parent", parent), ("control", graph)):
        paths = policy_path_vectors(connectome)
        vectors.update({label + "_" + key: value for key, value in paths.items()})
        summary["paths"][label] = {"inputs": len(paths["input_body_ids"]), "outputs": len(paths["output_body_ids"]),
            "unreachable_inputs": int((paths["input_to_any_dn_hops"] < 0).sum()),
            "unreachable_outputs": int((paths["any_input_to_dn_hops"] < 0).sum())}
    return summary, vectors
