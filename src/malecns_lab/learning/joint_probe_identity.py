"""P3b policy identity checks without importing Torch or allocating a graph."""
from __future__ import annotations

import json
import re

import numpy as np

from ..brain.metadata import canonical_hash
from ..brain.policy_groups import NeuronSelection, PolicyNeuronMap, POLICY_MAP_SCHEMA
from ..senses.spatial_drive import SpatialDrive, SpatialDriveConfig, build_spatial_input_map
from .joint_probe_contract import PARAMETERS, PROTOCOL, bound, require


def _selection(payload, count, nodes):
    keys = {"body_ids", "indices", "root_side", "root_side_missing", "soma_side", "soma_side_missing"}
    require(set(payload) == keys and all(isinstance(v, list) and len(v) == count for v in payload.values()),
            "Neuron selection shape mismatch")
    for key in ("body_ids", "indices"):
        require(all(type(value) is int for value in payload[key]) and payload[key] == sorted(set(payload[key])),
                "Neuron selection must be ascending integers")
    require(all(0 <= i < nodes for i in payload["indices"]), "Neuron indices out of bounds")
    for key in ("root_side", "soma_side"):
        require(all(type(v) is str for v in payload[key]) and all(type(v) is bool for v in payload[key + "_missing"]),
                "Neuron side/missing metadata malformed")
    return NeuronSelection(*(np.array(payload[k], dtype=np.int64 if k in ("body_ids", "indices") else
                                      bool if k.endswith("missing") else str)
                             for k in ("body_ids", "indices", "root_side", "root_side_missing", "soma_side", "soma_side_missing")))


def validate_policy_identity(policy, registration):
    bound(policy, "policy_id")
    require(policy["schema"] == "cnsbridge.stateful_neural_policy.v1" and policy["role"] == "full_reference",
            "Production joint trial requires full_reference policy")
    model, neurons = policy["model"], policy["neurons"]
    graph = model["graph"]
    for obj, key in ((model, "model_id"), (graph, "training_graph_id"), (neurons, "policy_map_id"),
                     (policy["spatial"], "spatial_map_id"), (policy["drive"], "drive_id")):
        bound(obj, key)
    require(graph["role"] == "full_reference" and all(graph[key] == PROTOCOL[key] for key in
            ("nodes", "edges", "synapse_count", "raw_ordered_counts_sha256")), "Full graph count/digest mismatch")
    require(graph["reference_source_id"] == registration["graph"]["source_id"]
            and graph["reference_provenance"] == registration["graph"]["provenance"]
            and graph["reference_graph_sha256"] == registration["graph"]["provenance"]["graph_sha256"],
            "Graph differs from preregistered provenance")
    grouping = graph["grouping"]
    names = grouping["names"]
    require(grouping["schema"] == "cnsbridge.presynaptic_raw_type.v1" and grouping["missing_index"] == 0
            and names[0] is None and all(type(n) is str and n.strip() for n in names[1:])
            and names[1:] == sorted(set(names[1:])) and "R1-R6" in names, "Invalid gain grouping")
    require(neurons["schema"] == POLICY_MAP_SCHEMA and neurons["role"] == "full_reference"
            and neurons["nodes"] == graph["nodes"] and neurons["training_graph_id"] == graph["training_graph_id"]
            and neurons["reference_source_id"] == graph["reference_source_id"]
            and neurons["reference_graph_sha256"] == graph["reference_graph_sha256"], "Policy neuron-map binding differs")
    inputs = _selection(neurons["inputs"], PROTOCOL["inputs"], graph["nodes"])
    outputs = _selection(neurons["outputs"], PROTOCOL["outputs"], graph["nodes"])
    require(model["input_body_ids"] == inputs.body_ids.tolist() and not np.intersect1d(inputs.body_ids, outputs.body_ids).size,
            "Model inputs differ from the registered neuron selection")
    mapping = PolicyNeuronMap(graph["nodes"], inputs, outputs, json.dumps(neurons, sort_keys=True))
    spatial = build_spatial_input_map(mapping)
    require(policy["spatial"] == spatial.identity
            and policy["drive"] == SpatialDrive(spatial, SpatialDriveConfig(**PROTOCOL["drive_config"])).identity,
            "Spatial/drive mapping differs")
    require(policy["background"] == PROTOCOL["background"] and policy["config"] == PROTOCOL["config"]
            and policy["camera"] == PROTOCOL["camera"] and policy["dtype"] == model["dtype"] == PROTOCOL["dtype"]
            and policy["frame_steps"] == 500 and policy["precondition_steps"] == 600
            and model["config"] == PROTOCOL["lif_config"], "Policy background/dtype/camera/clocks differ")
    fixed = {"schema": "cnsbridge.lamina_tonic_lif.v1", "base_dynamics_schema": "cnsbridge.trainable_lif.v1",
             "backend": "scipy_cpu_csr_forward_torch_csr_adjoint",
             "gain": "1 + 0.5*tanh(theta[type(pre)]); initial theta=0; nextafter guards strict floating bounds",
             "gradient": "fast_sigmoid; reset/refractory gates detached; all-cell sparse adjoint",
             "event_order": "arrivals; exact active integration; explicit voltage input; strict threshold; reset; delayed enqueue"}
    require(all(model[key] == value for key, value in fixed.items()), "Model dynamics semantic mismatch")
    tonic = model["tonic"]
    fixed_tonic = {"types": ["L1", "L2", "L3", "L4", "L5"], "mode": "fixed", "initial_rho": 0, "initial_offset_mv": 8,
             "formula": "16*sigmoid(rho[type]); nextafter guards strict floating bounds; nonmembers=0",
             "range_mv": {"lower": 0, "upper": 16, "inclusive": False},
             "integration": "active only: integrated_v += (1-exp(-dt/tau_m))*offset; g unchanged",
             "hypothesis": "engineering equilibrium offset, not a measured physiological constant"}
    require(all(tonic[key] == value for key, value in fixed_tonic.items()) and len(tonic["members"]) == 5,
            "Fixed background semantic mismatch")
    for name, entry in zip(fixed_tonic["types"], tonic["members"]):
        require(entry["type"] == name and name in names and type(entry["count"]) is int and entry["count"] > 0
                and re.fullmatch("[0-9a-f]{64}", entry["body_ids_sha256"]) is not None, "Invalid tonic members")
    require(tonic["member_count"] == sum(item["count"] for item in tonic["members"])
            and tonic["member_count"] <= graph["nodes"] - PROTOCOL["inputs"], "Invalid tonic member count")
    implementation = registration["implementation_sha256"]
    require(model["implementation_sha256"] == {name: implementation["brain/" + name] for name in
                                                ("trainable.py", "trainable_graph.py", "tonic.py")}, "Model sources differ")
    policy_sources = ("learning/neural_policy.py", "learning/follow_task.py", "senses/target.py", "senses/spatial_drive.py",
                      "brain/policy_groups.py", "brain/metadata.py", "control/body.py")
    require(policy["implementation_sha256"] == {key: implementation[key] for key in policy_sources}, "Policy sources differ")
    width = 2 * PROTOCOL["outputs"]
    expected_features = {"ordering": "first_half_DN_then_second_half_DN; ascending body ID in each half",
                         "body_ids": outputs.body_ids.tolist(), "width": width, "sample": "post_reset_v_each_neural_step",
                         "reduction": "arithmetic_mean_per_half", "raw_units": "mV",
                         "normalization": "(mean_v-v_rest)/(v_threshold-v_rest)", "offset_mv": -52.0, "scale_mv": 7.0}
    expected_readout = {"architecture": "linear_bias_then_tanh_then_axis_scales", "axes": ["vx", "vz", "wz"],
                       "scales": PROTOCOL["axis_scales"], "units": ["m/s", "m/s", "rad/s"], "frame": "body_flu",
                       "weight_initialization": "local_CPU_Generator.normal(std=0.01/sqrt(feature_width)); bias=0"}
    require(policy["features"] == expected_features and policy["readout"] == expected_readout
            and policy["training_parameters"] == list(PARAMETERS.values()), "Features/readout declaration differs")
    return {"theta": (len(names),), "readout_weight": (3, width), "readout_bias": (3,),
            "gains": (len(names),), "rho": (5,)}
