"""Offline terminal-R policy factory; call only inside a registered resource worker.

No research training, archive, device or numerical-fixture public entry point.
The exact existing dynamics and policy methods are used without substitution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from ..brain.metadata import canonical_hash
from ..brain.policy_groups import _build_policy_map, load_policy_neuron_map
from ..brain.tonic import LaminaTonicLIF
from ..brain.trainable import TrainableLIFConfig
from ..brain.trainable_graph import _graph, load_training_graph
from ..senses.spatial_drive import SpatialDriveConfig, build_spatial_input_map
from ..senses.target import TargetCamera
from .graph_controls import load_research_control
from .neural_policy import NeuralPolicy, NeuralPolicyConfig
from .research_admission import admit_research_graph, revalidate_admission
from .research_cache import _install_caches, _release_caches, validate_cache_correspondence, validate_membership
from .research_contract import common_numeric_contract
from .rewiring_contract import PROTOCOL, require


def _validate_configs(config, camera, drive_config, lif_config):
    for value, cls in ((config, NeuralPolicyConfig), (camera, TargetCamera),
                       (drive_config, SpatialDriveConfig), (lif_config, TrainableLIFConfig)):
        require(type(value) is cls, "Explicit exact existing configuration types required")
        cls(**asdict(value))
    require(all(not isinstance(v, bool) for v in asdict(lif_config).values()), "Boolean LIF numbers are not allowed")
    require(drive_config.dt_ms == lif_config.dt_ms, "Sensory and neural dt differ")
    config.steps(lif_config.dt_ms)


def _new_brain(reference_loader, replacement_loader, *, lif_config):
    """Private constructor, never accepts a brain, policy, state or delayed queue.

    Tests use explicitly numerical loaders; the sole public caller supplies the
    strict reference and revalidated terminal-R loaders below.
    """
    original, old_map = reference_loader()
    brain = LaminaTonicLIF(original, old_map.inputs.body_ids, mode="fixed", config=lif_config, dtype=torch.float32)
    body_ids, groups = original.body_ids.copy(), original.group_index.copy()
    names = original.group_names
    _release_caches(brain)
    del brain.graph
    del original  # No original integer matrix retained while loading/validating R.
    graph, mapping, annotations, missing = replacement_loader()
    require(np.array_equal(graph.body_ids, body_ids) and graph.group_names == names
            and np.array_equal(graph.group_index, groups), "Research body order/type groups changed")
    for name in ("inputs", "outputs"):
        require(getattr(mapping, name).to_dict() == getattr(old_map, name).to_dict(), "Complete policy selections changed")
    old_spatial, new_spatial = build_spatial_input_map(old_map), build_spatial_input_map(mapping)
    require(np.array_equal(old_spatial.pixel_cells, new_spatial.pixel_cells), "Spatial encoding values changed")
    brain.graph = graph
    validate_membership(brain, mapping, annotations, missing)
    _install_caches(brain)
    evidence = validate_cache_correspondence(brain)
    return brain, mapping, evidence


def _reference_loader(path):
    graph = load_training_graph(path)
    return graph, load_policy_neuron_map(graph, path)


def _research_loader(admission):
    revalidate_admission(admission)
    candidate, mother = admission["candidate"], admission["mother"]
    actual = load_research_control(candidate["path"], mother["path"])
    require(actual.meta["source_id"] == candidate["source_id"]
            and actual.meta["graph_sha256"] == candidate["graph_sha256"], "Actual loaded research source changed")
    require(actual.n == PROTOCOL["nodes"] and actual.n_connections == PROTOCOL["edges"]
            and actual.n_synapses == PROTOCOL["synapses"] and np.count_nonzero(actual.weights.diagonal()) == PROTOCOL["self_edges"],
            "Actual complete graph dimensions differ")
    graph = _graph(actual.weights, actual.body_ids, actual.annotations["type"], actual.annotation_missing["type"], {
        "role": "research_control", "reference_source_id": mother["source_id"],
        "reference_graph_sha256": mother["graph_sha256"], "reference_provenance": deepcopy(mother["provenance"]),
        "research": {"source_id": candidate["source_id"], "graph_sha256": candidate["graph_sha256"], "transform": "R",
                     "provenance": deepcopy(actual.meta["provenance"]), "admission": deepcopy(admission)},
    })
    annotations, missing = actual.annotations, actual.annotation_missing
    del actual
    mapping = _build_policy_map(graph, annotations, missing, annotation_digest=mother["provenance"]["node_metadata_sha256"])
    require(len(mapping.inputs.indices) == PROTOCOL["inputs"] and len(mapping.outputs.indices) == PROTOCOL["outputs"],
            "Actual complete R1-R6/DN selection differs")
    return graph, mapping, annotations, missing


def _finish_policy(brain, mapping, cache_evidence, admission, *, config, camera, drive_config):
    brain._identity["graph"] = brain.graph.identity
    factory = {"schema": "cnsbridge.offline_research_factory.v1", "admission_id": admission["admission_id"],
               "implementation_sha256": admission["factory_sources"], "cache_correspondence": cache_evidence}
    brain._identity["research_factory"] = deepcopy(factory)
    brain.model_id = canonical_hash(brain._identity)
    policy = NeuralPolicy.__new__(NeuralPolicy)
    policy._initialize(brain, mapping, background="fixed_lamina_8mv", config=config, camera=camera,
                       drive_config=drive_config, role=brain.graph.identity["role"])
    policy._identity["research_factory"] = factory
    policy._identity["common_numeric_contract"] = common_numeric_contract(policy)
    policy.policy_id = canonical_hash(policy._identity)
    return policy


def from_terminal_research_run(generation_directory, *, config, camera, drive_config, lif_config):
    """Construct a fresh genuine research_control policy from an approved final run.

    This may load complete matrices and must be resource-supervised. It performs
    no reset, preconditioning, forward, optimizer step or hardware operation.
    """
    _validate_configs(config, camera, drive_config, lif_config)
    admission = admit_research_graph(generation_directory)
    brain, mapping, caches = _new_brain(lambda: _reference_loader(admission["mother"]["path"]),
                                       lambda: _research_loader(admission), lif_config=lif_config)
    require(brain.graph.identity["role"] == mapping.identity["role"] == "research_control", "Research role required")
    policy = _finish_policy(brain, mapping, caches, admission, config=config, camera=camera, drive_config=drive_config)
    require(policy.brain.identity == policy.identity["model"] and policy.mapping.identity == policy.identity["neurons"],
            "Returned policy nested identities differ")
    require(validate_cache_correspondence(brain) == caches, "Cache installation changed before return")
    revalidate_admission(admission)
    return policy
