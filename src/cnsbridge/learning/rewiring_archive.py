"""Manifest-last intermediate CSC/PCG64 archives; these are not brain assets."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
from scipy import sparse

from ..brain.metadata import annotation_hash
from ..brain.reference import graph_hash
from .graph_controls import _clone
from .graph_diagnostics import validate_rw
from .resource_guard import write_json
from .rewiring_contract import bind, canonical, read_json, require, sha
from .rewiring_steps import restore_rng, validate_counters

SCHEMA = "cnsbridge.rewiring_generation_state.v1"
PURPOSE = "intermediate_CSC_PCG64_only_not_a_brain_asset"


def array_description(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": __import__("hashlib").sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()}


def state_identity(graph, rng, counts, mother):
    value = {"mother_source_id": mother["source_id"], "graph_sha256": graph_hash(graph),
             "node_metadata_sha256": annotation_hash(graph), "counts": copy.deepcopy(counts),
             "rng": copy.deepcopy(rng.bit_generator.state)}
    value["state_id"] = canonical(value)
    return value


def save_checkpoint(directory, *, graph, rng, counts, run, context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    matrix = graph.weights
    arrays = {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr}
    # Context and state are explicit; sparse arrays cannot masquerade as a Connectome NPZ.
    manifest = {"schema": SCHEMA, "purpose": PURPOSE, "run_id": run["run_id"], "context": context,
                "state": state_identity(graph, rng, counts, run["mother"]),
                "shape": list(matrix.shape), "arrays": {key: array_description(value) for key, value in arrays.items()},
                "implementation_sha256": run["implementation_sha256"], "runtime": run["runtime"]}
    temporary = directory / "arrays.npz.tmp"
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(directory / "arrays.npz")
    manifest["arrays_sha256"] = sha(directory / "arrays.npz")
    manifest["checkpoint_id"] = canonical(manifest)
    write_json(directory / "checkpoint.json", manifest)
    return manifest


def inspect_checkpoint(directory, run, context):
    """Lightweight parent checks; actual array validation is worker-only."""
    directory = Path(directory)
    meta = read_json(directory / "checkpoint.json")
    bind(meta, "checkpoint_id")
    require(set(meta) == {"schema", "purpose", "run_id", "context", "state", "shape", "arrays", "implementation_sha256",
                          "runtime", "arrays_sha256", "checkpoint_id"}, "Checkpoint fields differ")
    require(meta["schema"] == SCHEMA and meta["purpose"] == PURPOSE and meta["run_id"] == run["run_id"]
            and meta["context"] == context, "Checkpoint purpose/run/parent context differs")
    require(meta["runtime"] == run["runtime"] and meta["implementation_sha256"] == run["implementation_sha256"],
            "Checkpoint sources/runtime differ")
    require(meta["arrays_sha256"] == sha(directory / "arrays.npz"), "Checkpoint arrays file changed")
    bind(meta["state"], "state_id")
    require(meta["state"]["mother_source_id"] == run["mother"]["source_id"], "Checkpoint mother differs")
    return meta


def restore_checkpoint(directory, *, parent, run, context):
    meta = inspect_checkpoint(directory, run, context)
    protocol = run["protocol"]
    state = meta["state"]
    validate_counters(state["counts"], protocol["max_attempts"], protocol["target_swaps"])
    rng = restore_rng(state["rng"])
    with np.load(Path(directory) / "arrays.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"data", "indices", "indptr"}, "CSC archive fields differ")
        arrays = {key: archive[key] for key in archive.files}
    require(meta["shape"] == list(parent.weights.shape), "CSC shape differs from complete mother")
    require(meta["arrays"] == {key: array_description(value) for key, value in arrays.items()}, "Actual CSC bytes/dtype/shape differ")
    data, indices, indptr = arrays["data"], arrays["indices"], arrays["indptr"]
    for name, value in arrays.items():
        original = getattr(parent.weights, name)
        require(value.dtype == original.dtype and value.ndim == 1 and value.shape == original.shape,
                "CSC dtype/dimension/edge count changed")
        require(np.issubdtype(value.dtype, np.signedinteger), "Exact signed integer CSC required")
    require(indptr[0] == 0 and indptr[-1] == len(data) and np.all(indptr[1:] >= indptr[:-1])
            and np.array_equal(indptr, parent.weights.indptr), "CSC pointer/out-degree corruption")
    require(np.all((indices >= 0) & (indices < parent.n)) and np.all(data != 0), "CSC target/weight range invalid")
    for source in range(parent.n):
        column = indices[indptr[source]:indptr[source + 1]]
        require(np.all(column[1:] > column[:-1]), "Unsorted or duplicate CSC targets")
    graph = _clone(parent)
    graph.weights = sparse.csc_matrix((data, indices, indptr), shape=parent.weights.shape, copy=False)
    validate_rw(parent, graph, "R")
    actual = state_identity(graph, rng, state["counts"], run["mother"])
    require(actual == state, "Actual restored CSC/RNG/counts/metadata differ")
    return graph, rng, copy.deepcopy(state["counts"]), meta
