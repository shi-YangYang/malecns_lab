"""Complete, separately identified policy input/output selections.

Production selection reuses an already validated TrainingGraph. Reading the
reference annotations here is not a second validation of its edge contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .metadata import annotation_hash, canonical_hash
from .trainable_graph import TrainingGraph, group_by_type

POLICY_MAP_SCHEMA = "cnsbridge.policy_neurons.v1"
_REQUIRED_FIELDS = ("type", "superclass", "rootSide", "somaSide")


def _immutable(values: np.ndarray) -> np.ndarray:
    """A bytes-backed array cannot be made writable by the caller."""
    value = np.ascontiguousarray(values)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class NeuronSelection:
    body_ids: np.ndarray
    indices: np.ndarray
    root_side: np.ndarray
    root_side_missing: np.ndarray
    soma_side: np.ndarray
    soma_side_missing: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _immutable(getattr(self, name)))

    def to_dict(self) -> dict:
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PolicyNeuronMap:
    n_nodes: int
    inputs: NeuronSelection
    outputs: NeuronSelection
    _identity_json: str
    _map_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_map_id", json.loads(self._identity_json)["policy_map_id"])

    @property
    def identity(self) -> dict:
        return json.loads(self._identity_json)

    @property
    def map_id(self) -> str:
        return self._map_id


def _validate_annotations(annotations: dict, missing: dict, n: int) -> None:
    if not set(_REQUIRED_FIELDS).issubset(annotations) or set(annotations) != set(missing):
        raise ValueError("Policy annotations require type, superclass, rootSide, somaSide and all missing masks")
    for field, values in annotations.items():
        mask = missing[field]
        if values.shape != (n,) or mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"Invalid policy annotation shape or missing mask: {field}")
        if field in _REQUIRED_FIELDS and values.dtype.kind not in "US":
            raise ValueError(f"Policy selection annotations must contain strings: {field}")
        if values.dtype.hasobject:
            raise ValueError(f"Object annotations are unsupported: {field}")


def _read_bound_metadata(path: str | Path, body_ids: np.ndarray, provenance: dict,
                         source_id: str, *, asset_type: str) -> tuple[dict, dict]:
    """Lazy NPZ metadata read; never request sparse data/indices/indptr.

    The same binding checks are exercised with explicitly labelled numerical
    metadata fixtures. Only the production entry point accepts full_reference.
    """
    with np.load(path, allow_pickle=False) as asset:
        meta = json.loads(str(asset["meta"]))
        if meta.get("format_version") != 2 or meta.get("asset_type") != asset_type:
            raise ValueError("Policy metadata asset role/version mismatch")
        if (meta.get("source_id") != source_id or meta.get("provenance") != provenance
                or canonical_hash(provenance) != source_id):
            raise ValueError("Policy metadata source identity mismatch")
        ids = asset["body_ids"]
        if ids.dtype.kind not in "iu" or ids.shape != body_ids.shape or not np.array_equal(ids, body_ids):
            raise ValueError("Policy metadata body ID order does not match the validated graph")
        fields = asset["annotation_fields"]
        if (fields.ndim != 1 or fields.dtype.kind not in "US" or len(set(fields.tolist())) != len(fields)
                or any(not name for name in fields.tolist())):
            raise ValueError("Invalid policy annotation field list")
        annotations = {str(name): asset[f"annotation_{name}"] for name in fields}
        missing = {str(name): asset[f"missing_{name}"] for name in fields}
    _validate_annotations(annotations, missing, len(body_ids))
    observed_hash = annotation_hash(SimpleNamespace(annotations=annotations, annotation_missing=missing))
    if observed_hash != provenance.get("node_metadata_sha256"):
        raise ValueError("Policy node annotation hash differs from the validated graph")
    return annotations, missing


def _build_policy_map(graph: TrainingGraph, annotations: dict, missing: dict, *, annotation_digest: str) -> PolicyNeuronMap:
    _validate_annotations(annotations, missing, graph.n)
    names, groups = group_by_type(annotations["type"], missing["type"])
    if names != graph.group_names or not np.array_equal(groups, graph.group_index):
        raise ValueError("Policy type metadata differs from the training graph grouping")
    input_indices = np.flatnonzero(annotations["type"] == "R1-R6")
    output_indices = np.flatnonzero(annotations["superclass"] == "descending_neuron")
    if not len(input_indices) or not len(output_indices):
        raise ValueError("Policy requires nonempty complete R1-R6 and descending_neuron selections")

    def selection(indices: np.ndarray) -> NeuronSelection:
        return NeuronSelection(graph.body_ids[indices], indices.astype(np.int64),
                               annotations["rootSide"][indices], missing["rootSide"][indices],
                               annotations["somaSide"][indices], missing["somaSide"][indices])

    inputs, outputs = selection(input_indices), selection(output_indices)
    identity = {
        "schema": POLICY_MAP_SCHEMA, "role": graph.identity["role"], "nodes": graph.n,
        "training_graph_id": graph.identity["training_graph_id"],
        "reference_source_id": graph.identity["reference_source_id"],
        "reference_graph_sha256": graph.identity.get("reference_graph_sha256"),
        "node_metadata_sha256": annotation_digest,
        "selection": {"input": "raw type == R1-R6", "output": "raw superclass == descending_neuron",
                      "ordering": "unchanged ascending graph body ID", "side_filter": None},
        "inputs": inputs.to_dict(), "outputs": outputs.to_dict(),
        "boundary": "independent policy mapping; no change to reference groups or connectivity",
    }
    identity["policy_map_id"] = canonical_hash(identity)
    return PolicyNeuronMap(graph.n, inputs, outputs, json.dumps(identity, sort_keys=True))


def load_policy_neuron_map(graph: TrainingGraph, reference_path: str | Path) -> PolicyNeuronMap:
    """Bind annotations to a full graph already verified by load_training_graph.

    Do not construct TrainingGraph manually to bypass its production loader.
    This function verifies metadata and ID binding, not the NPZ edge arrays.
    """
    if not isinstance(graph, TrainingGraph) or graph.identity.get("role") != "full_reference":
        raise ValueError("Policy production mapping requires an already validated full_reference TrainingGraph")
    identity = graph.identity
    claimed_id = identity.pop("training_graph_id", None)
    if claimed_id != canonical_hash(identity):
        raise ValueError("Training graph identity payload mismatch")
    provenance = identity.get("reference_provenance", {})
    if identity.get("reference_graph_sha256") != provenance.get("graph_sha256"):
        raise ValueError("Training graph provenance binding mismatch")
    annotations, missing = _read_bound_metadata(reference_path, graph.body_ids, provenance,
                                                identity["reference_source_id"], asset_type="full_reference")
    return _build_policy_map(graph, annotations, missing, annotation_digest=provenance["node_metadata_sha256"])


def numerical_test_policy_map(graph: TrainingGraph, annotations: dict, missing: dict) -> PolicyNeuronMap:
    """Interface fixture only; never a substitute for a production full brain."""
    if not isinstance(graph, TrainingGraph) or graph.identity.get("role") != "numerical_test_only":
        raise ValueError("Numerical policy fixtures require an explicitly numerical_test_only graph")
    _validate_annotations(annotations, missing, graph.n)
    digest = annotation_hash(SimpleNamespace(annotations=annotations, annotation_missing=missing))
    return _build_policy_map(graph, annotations, missing, annotation_digest=digest)
