"""Private first-construction cache installation and complete sparse correspondence."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.sparse import csr_array
import torch

from ..brain.metadata import canonical_hash
from ..brain.tonic import LAMINA_TYPES, LaminaTonicLIF
from ..brain.trainable_graph import group_by_type
from .rewiring_contract import require

CHUNK = 1_000_000


def _same(left, right, message):
    require(left.shape == right.shape, message)
    for start in range(0, left.size, CHUNK):
        require(np.array_equal(left[start:start + CHUNK], right[start:start + CHUNK]), message)


def _ordered_digest(graph):
    digest = hashlib.sha256()
    for value in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
        for start in range(0, len(value), CHUNK):
            digest.update(np.asarray(value[start:start + CHUNK], dtype="<i8").tobytes())
    return digest.hexdigest()


def _release_caches(brain):
    # Release SciPy first: its NumPy views retain all three forward Torch storages.
    del brain._forward_csr
    del brain.base_weights
    del brain.base_weights_adjoint


def _install_caches(brain):
    """Only the private fresh-brain constructor calls this; no state migration API."""
    require(not any(hasattr(brain, name) for name in ("_forward_csr", "base_weights", "base_weights_adjoint")),
            "Old graph caches must be released before installation")
    graph, dtype = brain.graph, brain._dtype
    csr = graph.counts.tocsr()
    forward = torch.sparse_csr_tensor(torch.tensor(csr.indptr, dtype=torch.int64),
        torch.tensor(csr.indices, dtype=torch.int64), torch.tensor(csr.data, dtype=dtype) * brain.config.weight_mv,
        size=csr.shape, check_invariants=True)
    brain.register_buffer("base_weights", forward, persistent=False)
    del csr
    adjoint = torch.sparse_csr_tensor(torch.tensor(graph.counts.indptr, dtype=torch.int64),
        torch.tensor(graph.counts.indices, dtype=torch.int64),
        torch.tensor(graph.counts.data, dtype=dtype) * brain.config.weight_mv,
        size=graph.counts.shape[::-1], check_invariants=True)
    brain.register_buffer("base_weights_adjoint", adjoint, persistent=False)
    views = (forward.values().numpy(), forward.col_indices().numpy(), forward.crow_indices().numpy())
    for view in views:
        view.flags.writeable = False
    brain._forward_csr = csr_array(views, shape=tuple(forward.shape), copy=False)
    for view in (brain._forward_csr.data, brain._forward_csr.indices, brain._forward_csr.indptr):
        view.flags.writeable = False


def validate_cache_correspondence(brain):
    """Check every edge, pointer, value, dtype and shared view; never densify."""
    require(type(brain) is LaminaTonicLIF and brain._dtype == torch.float32, "Exact CPU float32 LaminaTonicLIF required")
    graph, raw = brain.graph, brain.graph.counts
    identity = graph.identity
    require(identity["training_graph_id"] == canonical_hash({k: v for k, v in identity.items() if k != "training_graph_id"}),
            "Graph identity is unbound")
    require(raw.dtype == np.int64 and raw.has_canonical_format and not np.any(raw.data == 0), "Canonical integer counts required")
    require(all(not a.flags.writeable for a in (raw.data, raw.indices, raw.indptr, graph.body_ids, graph.group_index)),
            "Original graph buffers must remain read-only")
    require(_ordered_digest(graph) == identity["raw_ordered_counts_sha256"], "Actual raw counts differ from graph identity")
    require(identity["nodes"] == graph.n == len(graph.body_ids) and identity["edges"] == raw.nnz
            and identity["synapse_count"] == int(np.abs(raw.data).sum(dtype=np.int64)), "Graph dimensions/counts differ")
    for matrix in (brain.base_weights, brain.base_weights_adjoint):
        require(matrix.layout == torch.sparse_csr and matrix.device.type == "cpu" and matrix.dtype == torch.float32
                and tuple(matrix.shape) == raw.shape and matrix.values().numel() == raw.nnz,
                "Sparse cache shape/dtype/backend differs")
        require(matrix.crow_indices().dtype == matrix.col_indices().dtype == torch.int64
                and not matrix.values().requires_grad, "Sparse cache index dtype/trainability differs")
    # A bounded conversion supplies independent full CSR ordering; it is released
    # before validating the transpose already encoded by the original CSC.
    expected = raw.tocsr()
    for actual, integer in ((brain.base_weights, expected), (brain.base_weights_adjoint, raw)):
        _same(actual.crow_indices().numpy(), integer.indptr, "CSR pointer correspondence differs")
        _same(actual.col_indices().numpy(), integer.indices, "CSR index correspondence differs")
        for start in range(0, raw.nnz, CHUNK):
            stop = start + CHUNK
            values = torch.tensor(integer.data[start:stop], dtype=torch.float32) * brain.config.weight_mv
            require(torch.equal(actual.values()[start:stop], values), "CSR signed weight/order correspondence differs")
        if actual is brain.base_weights:
            del expected
    scipy = brain._forward_csr
    require(type(scipy) is csr_array and scipy.shape == raw.shape, "SciPy cache shape/type differs")
    for view, tensor in zip((scipy.data, scipy.indices, scipy.indptr),
                           (brain.base_weights.values(), brain.base_weights.col_indices(), brain.base_weights.crow_indices()), strict=True):
        source = tensor.numpy()
        require(view.shape == source.shape and view.dtype == source.dtype and view.ctypes.data == source.ctypes.data
                and not view.flags.writeable, "SciPy cache must share read-only forward storage")
    return {"schema": "cnsbridge.research_cache_correspondence.v1", "training_graph_id": identity["training_graph_id"],
            "raw_ordered_counts_sha256": identity["raw_ordered_counts_sha256"], "nodes": graph.n, "edges_checked": raw.nnz,
            "checks": "all integer CSC, forward CSR, transposed CSR, values and shared read-only SciPy buffers",
            "weight_order": "int64 -> torch.float32 -> multiply weight_mv", "chunk_edges": CHUNK}


def validate_membership(brain, mapping, annotations, missing):
    graph = brain.graph
    names, groups = group_by_type(annotations["type"], missing["type"])
    require(names == graph.group_names and np.array_equal(groups, graph.group_index), "Graph type grouping differs")
    require(brain.group_index.dtype == torch.int64 and brain.group_index.device.type == "cpu"
            and np.array_equal(brain.group_index.numpy(), groups), "Brain type grouping differs")
    expected_inputs = np.flatnonzero(annotations["type"] == "R1-R6")
    expected_outputs = np.flatnonzero(annotations["superclass"] == "descending_neuron")
    for selection, indices in ((mapping.inputs, expected_inputs), (mapping.outputs, expected_outputs)):
        require(np.array_equal(selection.indices, indices) and np.array_equal(selection.body_ids, graph.body_ids[indices]),
                "Complete input/output body order or indices differ")
    require(brain.input_body_ids == tuple(graph.body_ids[expected_inputs]) and brain.input_mask.dtype == torch.bool
            and brain.input_mask.device.type == "cpu"
            and np.array_equal(brain.input_mask.numpy(), np.isin(np.arange(graph.n), expected_inputs)), "Input mask differs")
    tonic = np.zeros(graph.n, np.int64)
    for index, name in enumerate(LAMINA_TYPES, 1):
        members = (annotations["type"] == name) & ~missing["type"]
        require(np.any(members), "Missing required complete lamina type")
        tonic[members] = index
    require(brain.tonic_types == LAMINA_TYPES and brain.tonic_mode == "fixed"
            and brain.tonic_group_index.dtype == torch.int64 and brain.tonic_group_index.device.type == "cpu"
            and np.array_equal(brain.tonic_group_index.numpy(), tonic), "Fixed complete lamina membership differs")
    require(brain.rho.dtype == torch.float32 and brain.rho.device.type == "cpu" and tuple(brain.rho.shape) == (5,)
            and not brain.rho.requires_grad and not torch.count_nonzero(brain.rho)
            and torch.equal(brain.tonic_offsets_mv, torch.full((5,), 8., dtype=torch.float32)), "Fixed 8 mV background differs")
