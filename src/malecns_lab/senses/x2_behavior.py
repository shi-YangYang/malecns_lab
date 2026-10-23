"""Fixed engineering codebook for task/feedback/vision into every real R1–R6.

No action teacher or neural readout lives here. Invalid data has explicit masks;
IDs, clocks, target labels and future frames are never stimulation features.
"""
from __future__ import annotations

import hashlib
import json
import math

from ..control.x2_behavior import Behavior, ExecutionStatus, Observation, Task, digest, require

SCHEMA = "malecns_lab.x2_behavior_input.v1"
MAX_RATE_HZ = 150.
EVENT_VOLTAGE_MV = 68.75
DT_MS = .1
CODE_SEED = "x2-behavior-r1-codebook-2401"


def _clip(value):
    return min(1., max(0., float(value)))


def channel_values(observation):
    """Named [0,1] channels, in frozen order, from currently observed inputs."""
    require(isinstance(observation, Observation), "Observation required")
    o, f, p = observation, observation.feedback, observation.perception
    channels = {f"task.{task.value}": float(o.task == task) for task in Task}
    channels["request.changed"] = float(o.request_changed)
    channels.update({f"capability.{b.value}": float(b in o.capabilities) for b in Behavior})
    channels["feedback.valid"] = float(f.valid)
    channels["feedback.age"] = _clip(f.age_s / 30.)
    channels["feedback.same_request"] = float(f.valid and f.request_id == o.request_id)
    channels.update({f"feedback.behavior.{b.value}": float(f.valid and f.behavior == b) for b in Behavior})
    channels.update({f"feedback.status.{s.value}": float(f.valid and f.status == s) for s in ExecutionStatus})
    channels["body.standing"] = float(f.valid and f.standing)
    channels["body.mc_stand"] = float(f.valid and f.mc_mode == "STAND_DEFAULT")
    channels["body.mc_walk"] = float(f.valid and f.mc_mode == "WALK")
    for name, value, scale in (("yaw", f.yaw_rad, math.pi), ("vx", f.vx_m_s, .5),
                               ("wz", f.wz_rad_s, 1.), ("heading_error", f.heading_error_rad, math.pi)):
        channels[f"body.{name}.positive"] = _clip(value / scale) if f.valid else 0.
        channels[f"body.{name}.negative"] = _clip(-value / scale) if f.valid else 0.
    channels["vision.valid"] = float(p.valid)
    channels["vision.age"] = _clip(p.age_s / 30.)
    channels.update({f"vision.cell.{i:02d}": x if p.valid else 0. for i, x in enumerate(p.visual_cells)})
    return channels


# Obtain channel names from an explicitly missing-data protocol fixture, not a
# synthetic graph. This creates neither neural state nor a production mapping.
CHANNELS = tuple(channel_values(Observation("schema", "schema", 0, "schema", Task.GREET, 0.)))


def input_contract():
    return {"schema": SCHEMA, "channels": list(CHANNELS), "code_seed": CODE_SEED,
            "assignment": "sort input positions by SHA256(seed:body_id), tie by body_id; channel=rank%channel_count",
            "aggregation": "one channel per R1-R6; all input neurons assigned exactly once; no DN stimulation",
            "saturation": "all channels clipped to [0,1]; invalid measured value channels exactly zero",
            "scales": {"yaw_rad": math.pi, "heading_error_rad": math.pi, "vx_m_s": .5,
                       "wz_rad_s": 1., "age_s": 30.},
            "visual_cells": {"rows": 4, "columns": 4, "order": "row-major", "source": "measured occupancy; missing masked"},
            "dt_ms": DT_MS, "max_rate_hz": MAX_RATE_HZ, "event_voltage_mv": EVENT_VOLTAGE_MV,
            "units": "encoded occupancy; device event rate=occupancy*max_rate_hz",
            "excluded": ["teacher", "future", "timestamps", "sequence_numbers", "run_id", "request_id", "execution_id"],
            "boundary": "engineering code; no anatomical language/proprioception claim"}


class BehaviorEncoder:
    """Production mapping is accepted only from the existing validated map API.

    Import is light inside this module. Binding imports metadata machinery;
    encoding imports NumPy, but neither operation loads the graph or Torch.
    """
    @classmethod
    def from_policy_map(cls, mapping):
        from ..brain.policy_groups import PolicyNeuronMap
        require(isinstance(mapping, PolicyNeuronMap), "Validated production PolicyNeuronMap required")
        identity = mapping.identity
        claimed = identity.pop("policy_map_id", None)
        require(claimed == digest(identity) and claimed == mapping.map_id, "Policy map identity mismatch")
        require(identity["role"] == "full_reference" and mapping.n_nodes == 166700
                and len(mapping.inputs.body_ids) == 3377 and len(mapping.outputs.body_ids) == 1314,
                "Complete real MaleCNS 166700/3377/1314 mapping required")
        inputs = tuple(int(x) for x in mapping.inputs.body_ids)
        indices = tuple(int(x) for x in mapping.inputs.indices)
        outputs = tuple(int(x) for x in mapping.outputs.body_ids)
        require(len(set(inputs)) == len(inputs) and len(set(indices)) == len(indices)
                and not set(inputs).intersection(outputs), "Input IDs must be unique and disjoint from DN outputs")
        require(tuple(sorted(inputs)) == inputs and all(0 <= x < mapping.n_nodes for x in indices), "Input order/index mismatch")
        require(identity["inputs"] == mapping.inputs.to_dict() and identity["outputs"] == mapping.outputs.to_dict(), "Map arrays differ from bound identity")
        result = cls.__new__(cls)
        result.input_body_ids, result.input_indices = inputs, indices
        order = sorted(range(len(inputs)), key=lambda i: (hashlib.sha256(f"{CODE_SEED}:{inputs[i]}".encode()).digest(), inputs[i]))
        assignments = [0] * len(inputs)
        for rank, index in enumerate(order):
            assignments[index] = rank % len(CHANNELS)
        result.channel_indices = tuple(assignments)
        payload = {**input_contract(), "role": "full_reference", "nodes": mapping.n_nodes,
                   "policy_map_id": mapping.map_id, "training_graph_id": identity["training_graph_id"],
                   "reference_graph_sha256": identity["reference_graph_sha256"],
                   "input_body_ids": list(inputs), "input_indices": list(indices),
                   "output_body_ids_sha256": digest(list(outputs)), "outputs": len(outputs),
                   "channel_indices": assignments}
        result.input_map_id = digest(payload)
        result._identity_json = json.dumps({**payload, "input_map_id": result.input_map_id}, sort_keys=True)
        return result

    @property
    def identity(self):
        return json.loads(self._identity_json)

    def encode(self, observation):
        import numpy as np
        values = channel_values(observation)
        require(tuple(values) == CHANNELS, "Input channel ordering changed")
        table = tuple(values.values())
        encoded = np.asarray([table[i] for i in self.channel_indices], dtype=np.float32)
        require(encoded.shape == (3377,) and np.isfinite(encoded).all()
                and (encoded >= 0).all() and (encoded <= 1).all(), "Invalid neural drive")
        return encoded
