"""Offline R/W and exactly weight-matched edge intervention controls.

Every public builder starts with ``load_reference``. Derived files are marked
``research_control`` and are deliberately rejected by production Brain loaders.
No default swap budget, neural run, fitted decoder or mixing claim is supplied.

P removes outgoing edges of predeclared source body IDs, retaining all nodes.
Its paired control samples edges of other sources with exactly the same signed
weight histogram. This is an *edge* comparison: affected neuron counts, anatomy
and paths are not matched. Acute/retrained are protocol intentions, not fits.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from ..brain.connectome import Connectome
from ..brain.metadata import annotation_hash, canonical_hash, file_hashes
from ..brain.reference import graph_hash, load_reference, validate_reference
from .graph_diagnostics import (
    check_matrix,
    edge_overlap,
    graph_diagnostics,
    require_same_nodes,
    validate_rw,
)


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < int(positive):
        raise ValueError(f"{name} must be an {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _clone(parent: Connectome) -> Connectome:
    return Connectome(
        f"{parent.name}-offline-control", parent.weights.copy(), parent.types.copy(), parent.sides.copy(),
        None if parent.superclass is None else parent.superclass.copy(), parent.body_ids.copy(),
        groups={name: indices.copy() for name, indices in parent.groups.items()},
        annotations={name: values.copy() for name, values in parent.annotations.items()},
        annotation_missing={name: values.copy() for name, values in parent.annotation_missing.items()},
        meta={"roles": copy.deepcopy(parent.meta["roles"])},
    )


def _self_positions(matrix) -> np.ndarray:
    sources = np.flatnonzero(matrix.diagonal())
    return np.asarray([matrix.indptr[source] + np.searchsorted(
        matrix.indices[matrix.indptr[source]:matrix.indptr[source + 1]], source) for source in sources], dtype=np.int64)


def _replace_target(matrix, source: int, position: int, target: int) -> None:
    """Move one weighted edge within its CSC column, keeping targets sorted."""
    start, end = matrix.indptr[source:source + 2]
    insertion = start + int(np.searchsorted(matrix.indices[start:end], target))
    weight = matrix.data[position]
    if insertion > position:
        insertion -= 1
        matrix.indices[position:insertion] = matrix.indices[position + 1:insertion + 1]
        matrix.data[position:insertion] = matrix.data[position + 1:insertion + 1]
    else:
        matrix.indices[insertion + 1:position + 1] = matrix.indices[insertion:position]
        matrix.data[insertion + 1:position + 1] = matrix.data[insertion:position]
    matrix.indices[insertion], matrix.data[insertion] = target, weight


def _contains(matrix, source: int, target: int) -> bool:
    start, end = matrix.indptr[source:source + 2]
    position = start + np.searchsorted(matrix.indices[start:end], target)
    return bool(position < end and matrix.indices[position] == target)


def _rewire(parent: Connectome, *, seed: int, max_attempts: int, target_swaps: int) -> tuple[Connectome, dict]:
    """Uniform edge-pair proposals with sign-preserving double-edge swaps."""
    rng = np.random.default_rng(seed)
    graph = _clone(parent)
    matrix = graph.weights
    rejections = dict.fromkeys(("same_source", "self_edge", "different_source_sign", "same_target", "new_self_edge", "duplicate"), 0)
    attempts = successful = 0
    while matrix.nnz and attempts < max_attempts and successful < target_swaps:
        attempts += 1
        a, b = rng.integers(matrix.nnz, size=2)
        u, w = np.searchsorted(matrix.indptr, [a, b], side="right") - 1
        v, x = matrix.indices[[a, b]]
        reason = None
        if u == w:
            reason = "same_source"
        elif u == v or w == x:
            reason = "self_edge"
        elif np.sign(matrix.data[a]) != np.sign(matrix.data[b]):
            reason = "different_source_sign"
        elif v == x:
            reason = "same_target"
        elif u == x or w == v:
            reason = "new_self_edge"
        elif _contains(matrix, u, x) or _contains(matrix, w, v):
            reason = "duplicate"
        if reason:
            rejections[reason] += 1
            continue
        _replace_target(matrix, u, a, x)
        _replace_target(matrix, w, b, v)
        successful += 1
    nonself = matrix.nnz - len(_self_positions(matrix))
    return graph, {"attempts": attempts, "successful_swaps": successful, "rejections": rejections,
                   "target_reached": successful == target_swaps,
                   "stop_reason": "target_reached" if successful == target_swaps else "proposal_budget_exhausted" if matrix.nnz else "no_edges",
                   "effective_edge_moves_per_nonself_edge": 2 * successful / nonself if nonself else None,
                   "proposal": "independent uniform edge positions, with replacement; all listed invalid proposals rejected",
                   "budget_interpretation": "engineering budget, not experimentally validated mixing"}


def _shuffle(parent: Connectome, *, seed: int) -> tuple[Connectome, dict]:
    rng = np.random.default_rng(seed)
    graph = _clone(parent)
    matrix = graph.weights
    eligible_sources = eligible_edges = changed = 0
    for source in range(parent.n):
        start, stop = matrix.indptr[source:source + 2]
        positions = np.arange(start, stop)[matrix.indices[start:stop] != source]
        if positions.size < 2:
            continue
        eligible_sources += 1
        eligible_edges += len(positions)
        old = matrix.data[positions].copy()
        matrix.data[positions] = rng.permutation(old)
        changed += int((matrix.data[positions] != old).sum())
    return graph, {"eligible_sources": eligible_sources, "eligible_nonself_edges": eligible_edges,
                   "ineligible_sources": parent.n - eligible_sources, "changed_weight_edges": changed,
                   "changed_weight_fraction_of_eligible_edges": changed / eligible_edges if eligible_edges else None,
                   "permutations": "one seeded permutation per eligible source; identities and equal weights are allowed"}


def _stamp(parent: Connectome, graph: Connectome, *, transform: str, seed: int,
           protocol: str, parameters: dict, diagnostics: dict) -> None:
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("A nonempty predeclared protocol identifier is required")
    graph_identity = graph_hash(graph)
    provenance = {
        "format": "cnsbridge.research_graph.v1", "parent_source_id": parent.meta["source_id"],
        "parent_graph_sha256": graph_hash(parent), "node_metadata_sha256": annotation_hash(graph),
        "graph_sha256": graph_identity, "transform": transform, "seed": seed,
        "protocol": protocol, "parameters": parameters, "diagnostics": diagnostics,
        "implementation_sha256": {path.name: file_hashes(path)["sha256"] for path in
                                   (Path(__file__), Path(__file__).with_name("graph_diagnostics.py"))},
        "scope": "offline research only; no hardware, fit, task accuracy or biological advantage demonstrated",
    }
    graph.meta = {"format_version": 2, "asset_type": "research_control", "source_id": canonical_hash(provenance),
                  "graph_sha256": graph_identity, "source": parent.meta.get("source"),
                  "license": parent.meta.get("license"), "roles": copy.deepcopy(parent.meta["roles"]),
                  "provenance": provenance}


def _resource(started: float) -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "method": "perf_counter and process-lifetime getrusage peak; includes parent loading and diagnostics"}


def create_research_control(parent_path: str | Path, out_path: str | Path, *, transform: str,
                            seed: int, protocol: str, max_attempts: int | None = None,
                            target_swaps: int | None = None, with_paths: bool = False) -> dict:
    """Derive a single R/W asset with explicit budgets; never overwrite assets."""
    started, out_path = time.perf_counter(), Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    seed = _integer(seed, "seed")
    if transform == "R":
        max_attempts = _integer(max_attempts, "max_attempts")
        target_swaps = _integer(target_swaps, "target_swaps")
    elif transform != "W" or max_attempts is not None or target_swaps is not None:
        raise ValueError("Select R with explicit budgets or W without swap budgets")
    parent = load_reference(parent_path)
    check_matrix(parent)
    if transform == "R":
        graph, changes = _rewire(parent, seed=seed, max_attempts=max_attempts, target_swaps=target_swaps)
        parameters = {"max_attempts": max_attempts, "target_swaps": target_swaps}
    else:
        graph, changes = _shuffle(parent, seed=seed)
        parameters = {"permutations_per_source": 1}
    validate_rw(parent, graph, transform)
    diagnostics = graph_diagnostics(parent, graph, paths=with_paths)
    diagnostics["transformation"] = changes
    _stamp(parent, graph, transform=transform, seed=seed, protocol=protocol,
           parameters={**parameters, "with_paths": with_paths, "self_edges": "fixed"}, diagnostics=diagnostics)
    graph.save(out_path, overwrite=False)
    return {"status": "created", "asset": str(out_path.resolve()), "source_id": graph.meta["source_id"],
            "graph_sha256": graph.meta["graph_sha256"], "parent_source_id": parent.meta["source_id"],
            "transform": transform, "diagnostics": diagnostics, "resource_measurement": _resource(started)}


def _sources(parent: Connectome, body_ids: list[int]) -> np.ndarray:
    ids = np.asarray(body_ids)
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Explicit nonempty integer source_body_ids required")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate source body IDs")
    positions = np.searchsorted(parent.body_ids, ids)
    if np.any(positions >= parent.n) or not np.array_equal(parent.body_ids[positions], ids):
        raise ValueError("Intervention source body IDs are absent from the parent")
    return np.sort(positions)


def _outgoing_positions(parent: Connectome, sources: np.ndarray, include_self: bool) -> np.ndarray:
    matrix = parent.weights
    positions = np.concatenate([np.arange(matrix.indptr[source], matrix.indptr[source + 1]) for source in sources])
    if not include_self and len(positions):
        positions = positions[matrix.indices[positions] != np.searchsorted(matrix.indptr, positions, side="right") - 1]
    return positions


def _matching_positions(parent: Connectome, sources: np.ndarray, target: np.ndarray,
                        *, seed: int, include_self: bool) -> np.ndarray:
    matrix, rng = parent.weights, np.random.default_rng(seed)
    eligible_sources = np.ones(parent.n, bool)
    eligible_sources[sources] = False
    eligible = np.repeat(eligible_sources, np.diff(matrix.indptr))
    if not include_self:
        eligible[_self_positions(matrix)] = False
    pool = np.flatnonzero(eligible)
    del eligible
    pool = pool[np.argsort(matrix.data[pool], kind="stable")]
    sorted_weights = matrix.data[pool]
    values, counts = np.unique(matrix.data[target], return_counts=True)
    selected = []
    for value, count in zip(values, counts, strict=True):
        start, stop = np.searchsorted(sorted_weights, value, side="left"), np.searchsorted(sorted_weights, value, side="right")
        if stop - start < count:
            raise ValueError(f"Exact matching unavailable for signed weight {value}: need {count}, available {stop - start}; no relaxation")
        selected.append(pool[start + rng.choice(stop - start, size=count, replace=False)])
    return np.sort(np.concatenate(selected)) if selected else np.zeros(0, dtype=np.int64)


def _remove(parent: Connectome, positions: np.ndarray) -> Connectome:
    graph = _clone(parent)
    graph.weights.data[positions] = 0
    graph.weights.eliminate_zeros()
    return graph


def _removal_summary(parent: Connectome, positions: np.ndarray) -> dict:
    values, counts = np.unique(parent.weights.data[positions], return_counts=True)
    sources = np.searchsorted(parent.weights.indptr, positions, side="right") - 1
    targets = parent.weights.indices[positions]
    return {"removed_edges": len(positions), "removed_synapses": int(np.abs(parent.weights.data[positions]).sum(dtype=np.int64)),
            "signed_weight_histogram": {str(value): int(count) for value, count in zip(values, counts, strict=True)},
            "affected_sources": int(np.unique(sources).size), "affected_targets": int(np.unique(targets).size),
            "removed_self_edges": int((sources == targets).sum())}


def _validate_p(parent: Connectome, graph: Connectome, transform: str, parameters: dict) -> None:
    require_same_nodes(parent, graph)
    check_matrix(graph)
    source_indices = _sources(parent, parameters["source_body_ids"])
    target_positions = _outgoing_positions(parent, source_indices, parameters["include_self"])
    expected = _removal_summary(parent, target_positions)
    overlap = edge_overlap(parent, graph)
    if overlap["added_edges"] or overlap["changed_weight_on_common_edges"]:
        raise ValueError("P may only delete parent edges")
    removed = parent.weights - graph.weights
    weights, counts = np.unique(removed.data, return_counts=True)
    histogram = {str(value): int(count) for value, count in zip(weights, counts, strict=True)}
    if histogram != expected["signed_weight_histogram"]:
        raise ValueError("P signed-weight matching failed")
    if transform == "P_target":
        expected_graph = _remove(parent, target_positions)
        if (expected_graph.weights != graph.weights).nnz:
            raise ValueError("P target differs from predeclared outgoing edge intervention")
    elif transform == "P_matched":
        if np.any(np.diff(removed.indptr)[source_indices]):
            raise ValueError("Matched P control removed edges from excluded target sources")
        if not parameters["include_self"] and np.any(removed.diagonal()):
            raise ValueError("P self edge exclusion violated")
    else:
        raise ValueError("Unknown P transform")


def create_intervention_pair(parent_path: str | Path, out_dir: str | Path, *, source_body_ids: list[int],
                             include_self: bool, seed: int, protocol: str,
                             intervention_mode: str, with_paths: bool = False) -> dict:
    """Delete nominated outgoing edges and an exactly matched non-target set."""
    started, out_dir = time.perf_counter(), Path(out_dir)
    seed = _integer(seed, "seed")
    if type(include_self) is not bool or intervention_mode not in {"acute", "retrained"}:
        raise ValueError("Specify include_self bool and intervention_mode acute or retrained")
    if out_dir.exists():
        raise FileExistsError(out_dir)
    parent = load_reference(parent_path)
    check_matrix(parent)
    sources = _sources(parent, source_body_ids)
    target = _outgoing_positions(parent, sources, include_self)
    if not target.size:
        raise ValueError("Predeclared intervention removes no edges")
    matched = _matching_positions(parent, sources, target, seed=seed, include_self=include_self)
    parameters = {"source_body_ids": parent.body_ids[sources].tolist(), "include_self": include_self,
                  "intervention_mode": intervention_mode, "fit_status": "not_executed",
                  "candidate_rule": "edges from all non-target sources; exact signed integer weight strata; no replacement",
                  "matching": "edge count and signed-weight histogram exactly; node counts, anatomy and paths unmatched",
                  "with_paths": with_paths}
    pair_id = canonical_hash({"parent_source_id": parent.meta["source_id"], "seed": seed, "protocol": protocol,
                              "parameters": parameters})
    parameters["pair_id"] = pair_id
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "created", "pair_id": pair_id, "parent_source_id": parent.meta["source_id"], "assets": {}}
    try:
        for transform, positions in (("P_target", target), ("P_matched", matched)):
            graph = _remove(parent, positions)
            _validate_p(parent, graph, transform, parameters)
            diagnostics = graph_diagnostics(parent, graph, paths=with_paths)
            diagnostics["removal"] = _removal_summary(parent, positions)
            _stamp(parent, graph, transform=transform, seed=seed, protocol=protocol,
                   parameters=parameters, diagnostics=diagnostics)
            path = graph.save(out_dir / f"{transform}.npz", overwrite=False)
            result["assets"][transform] = {"path": str(path.resolve()), "source_id": graph.meta["source_id"],
                                          "graph_sha256": graph.meta["graph_sha256"], "diagnostics": diagnostics}
        result["resource_measurement"] = _resource(started)
        (out_dir / "pair.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException as error:
        (out_dir / "failed.json").write_text(json.dumps({"status": "failed", "error": repr(error), "partial": result}, indent=2))
        raise
    return result


def load_research_control(path: str | Path, parent_reference: str | Path | Connectome) -> Connectome:
    """Explicit offline-only loader, requiring its separately validated mother graph."""
    parent = parent_reference if isinstance(parent_reference, Connectome) else load_reference(parent_reference)
    if isinstance(parent_reference, Connectome):
        validate_reference(parent)
    graph = Connectome.load(path)
    if graph.meta.get("asset_type") != "research_control" or graph.meta.get("format_version") != 2:
        raise ValueError("Expected an explicitly marked research_control asset")
    provenance = graph.meta["provenance"]
    if provenance.get("format") != "cnsbridge.research_graph.v1" or canonical_hash(provenance) != graph.meta["source_id"]:
        raise ValueError("Research control provenance identity mismatch")
    if (provenance["parent_source_id"] != parent.meta["source_id"]
            or provenance["parent_graph_sha256"] != graph_hash(parent)):
        raise ValueError("Research control belongs to a different full reference")
    if graph_hash(graph) != provenance["graph_sha256"] or graph.meta["graph_sha256"] != provenance["graph_sha256"]:
        raise ValueError("Research graph content identity mismatch")
    if annotation_hash(graph) != provenance["node_metadata_sha256"] or graph.meta["roles"] != parent.meta["roles"]:
        raise ValueError("Research control metadata or roles changed")
    transform = provenance["transform"]
    if transform in {"R", "W"}:
        validate_rw(parent, graph, transform)
    elif transform in {"P_target", "P_matched"}:
        parameters = provenance["parameters"]
        expected_pair = canonical_hash({"parent_source_id": parent.meta["source_id"], "seed": provenance["seed"],
                                       "protocol": provenance["protocol"],
                                       "parameters": {key: value for key, value in parameters.items() if key != "pair_id"}})
        if expected_pair != parameters["pair_id"] or parameters["intervention_mode"] not in {"acute", "retrained"}:
            raise ValueError("P protocol pair identity mismatch")
        _validate_p(parent, graph, transform, parameters)
    else:
        raise ValueError("Unrecognized research transform")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new NPZ for R/W; new directory for P")
    parser.add_argument("--transform", choices=("R", "W", "P"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", required=True, help="predeclared protocol identifier, not a score-selected label")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--target-swaps", type=int)
    parser.add_argument("--with-paths", action="store_true", help="also compute exact group-to-output shortest paths")
    parser.add_argument("--source-body-ids", nargs="+", type=int)
    parser.add_argument("--self-edges", choices=("include", "exclude"))
    parser.add_argument("--intervention-mode", choices=("acute", "retrained"))
    args = parser.parse_args()
    if args.transform == "P":
        if args.source_body_ids is None or args.self_edges is None or args.intervention_mode is None:
            parser.error("P requires source-body-ids, self-edges and intervention-mode")
        if args.max_attempts is not None or args.target_swaps is not None:
            parser.error("P does not use swap budgets")
        result = create_intervention_pair(args.parent, args.out, source_body_ids=args.source_body_ids,
                                         include_self=args.self_edges == "include", seed=args.seed, protocol=args.protocol,
                                         intervention_mode=args.intervention_mode, with_paths=args.with_paths)
    else:
        if args.source_body_ids is not None or args.self_edges is not None or args.intervention_mode is not None:
            parser.error("P-specific arguments cannot be used with R/W")
        result = create_research_control(args.parent, args.out, transform=args.transform, seed=args.seed,
                                         protocol=args.protocol, max_attempts=args.max_attempts,
                                         target_swaps=args.target_swaps, with_paths=args.with_paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
