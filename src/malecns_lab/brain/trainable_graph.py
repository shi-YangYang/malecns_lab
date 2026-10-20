"""Unpruned reference loading and reproducible presynaptic type grouping.

Only ``load_training_graph`` is a production graph entry point. Tiny matrices
can be constructed through ``numerical_test_graph`` solely for numerical tests.
Neither path modifies the original integer synapse counts.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .metadata import canonical_hash
from .reference import load_reference


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def group_by_type(types: np.ndarray, missing: np.ndarray) -> tuple[tuple[str | None, ...], np.ndarray]:
    """Missing/blank types are group 0; other raw strings sort by Unicode order."""
    types, missing = np.asarray(types), np.asarray(missing)
    if types.ndim != 1 or types.dtype.kind not in "US" or missing.shape != types.shape or missing.dtype != bool:
        raise ValueError("Types must be a string vector with a matching boolean missing mask")
    absent = missing | (np.char.strip(types.astype(str)) == "")
    names = sorted(set(types[~absent].tolist()))
    lookup = {name: index + 1 for index, name in enumerate(names)}
    groups = np.array([0 if absent[index] else lookup[name] for index, name in enumerate(types)], dtype=np.int64)
    return (None, *names), _readonly(groups)


@dataclass(frozen=True)
class TrainingGraph:
    """Read-only raw counts plus identities; floating computation is separate."""

    counts: sparse.csc_matrix  # exact signed int64 (post, pre), never trainable
    body_ids: np.ndarray
    group_names: tuple[str | None, ...]
    group_index: np.ndarray
    _identity: dict

    @property
    def n(self) -> int:
        return self.counts.shape[0]

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)


def _graph(counts, body_ids, types, missing, identity) -> TrainingGraph:
    if not sparse.issparse(counts) or len(counts.shape) != 2 or counts.shape[0] != counts.shape[1] or counts.shape[0] == 0:
        raise ValueError("A nonempty square sparse count matrix is required")
    if not np.issubdtype(counts.dtype, np.signedinteger):
        raise ValueError("Raw signed synapse counts must have integer dtype")
    counts = sparse.csc_matrix(counts, dtype=np.int64, copy=True)
    if not counts.has_canonical_format or np.any(counts.data == 0):
        raise ValueError("Counts must have canonical unique nonzero edges; no aggregation or pruning is performed")
    body_ids = np.asarray(body_ids)
    if body_ids.shape != (counts.shape[0],) or not np.issubdtype(body_ids.dtype, np.integer):
        raise ValueError("Each neuron requires an integer body ID")
    if np.any(body_ids[1:] <= body_ids[:-1]):
        raise ValueError("Body IDs must be unique and ascending")
    names, group_index = group_by_type(types, missing)
    if group_index.shape != body_ids.shape:
        raise ValueError("Type annotations must cover every neuron")
    digest = hashlib.sha256()
    for array in (body_ids, counts.indptr, counts.indices, counts.data):
        digest.update(np.asarray(array, dtype="<i8").tobytes())
    group_hash = hashlib.sha256(np.asarray(group_index, dtype="<i8").tobytes()).hexdigest()
    identity = {
        **identity, "nodes": len(body_ids), "edges": counts.nnz,
        "synapse_count": int(np.abs(counts.data).sum(dtype=np.int64)),
        "raw_ordered_counts_sha256": digest.hexdigest(),
        "grouping": {"schema": "cnsbridge.presynaptic_raw_type.v1", "names": list(names),
                     "mapping_sha256": group_hash, "missing_index": 0,
                     "ordering": "missing first, then exact nonblank raw strings in Unicode order",
                     "missing_rule": "annotation missing mask OR empty/whitespace-only string"},
    }
    identity["training_graph_id"] = canonical_hash(identity)
    for array in (counts.data, counts.indices, counts.indptr):
        array.setflags(write=False)
    return TrainingGraph(counts, _readonly(body_ids.astype(np.int64)), names, group_index, identity)


def load_training_graph(path: str | Path) -> TrainingGraph:
    """Validate through the full-reference loader; no subgraph/fallback option."""
    reference = load_reference(path)
    # The loader verifies provenance, exact graph hash, metadata and reference
    # groups. Preserve that evidence instead of silently accepting a research graph.
    return _graph(reference.weights, reference.body_ids, reference.annotations["type"],
                  reference.annotation_missing["type"], {
                      "role": "full_reference", "reference_source_id": reference.meta["source_id"],
                      "reference_graph_sha256": reference.meta["provenance"]["graph_sha256"],
                      "reference_provenance": deepcopy(reference.meta["provenance"]),
                  })


def numerical_test_graph(counts, types=None, missing=None) -> TrainingGraph:
    """Explicit non-biological fixture. Never an alternative production model."""
    n = counts.shape[0]
    types = np.array(["test"] * n) if types is None else np.asarray(types)
    missing = np.zeros(n, dtype=bool) if missing is None else np.asarray(missing)
    return _graph(counts, np.arange(n, dtype=np.int64), types, missing,
                  {"role": "numerical_test_only", "reference_source_id": None})
