"""Stable task-polarity identity; no dynamics, graph, readout or runner changes."""
from copy import deepcopy
from pathlib import Path

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_task_polarity_policy.v1"
INPUT_SCHEMA = "malecns_lab.x2_behavior_input.task_polarity.v1"
TASK_CHANNELS = ("task.GREET", "task.FACE_LEFT", "task.FACE_RIGHT",
                 "task.SHOW_FORWARD", "task.END_INTERACTION")


class TaskPolarityEncoder:
    """(1-one_hot)/4 on the five original task groups; every other port unchanged."""
    def __init__(self, original):
        self.original = original
        self.input_body_ids, self.input_indices = original.input_body_ids, original.input_indices
        self.channel_indices = original.channel_indices
        previous = original.identity
        frozen.require(previous["schema"] == "malecns_lab.x2_behavior_input.no_absolute_yaw.v1"
            and previous["max_rate_hz"] == 150., "Expected original relative-heading 150 Hz encoding")
        channels = previous["channels"]
        frozen.require({name for name in channels if name.startswith("task.")} == set(TASK_CHANNELS),
                       "Expected exactly five original task channels")
        indices = {channels.index(name) for name in TASK_CHANNELS}
        self.task_ports = tuple(i for i,c in enumerate(self.channel_indices) if c in indices)
        frozen.require(len(self.channel_indices) == len(self.input_indices) == 3377
            and all(self.channel_indices.count(channels.index(name)) == 60 for name in TASK_CHANNELS),
            "Expected unchanged five 60-port task assignments")
        payload = deepcopy(previous)
        payload.pop("input_map_id")
        payload.update(schema=INPUT_SCHEMA, parent_input_map_id=previous["input_map_id"],
            task_polarity={"channels":list(TASK_CHANNELS),"ports":list(self.task_ports),
                "encoding":"(1-one_hot)/4", "selected_occupancy":0., "other_task_occupancy":.25,
                "nominal_task_occupancy_sum":60.,
                "finite_window_events":"measured; equal nominal sum does not imply equal actual event counts",
                "mapping":"original R1-R6 only; non-task ports and scheduler unchanged; no DN stimulation"},
            task_polarity_implementation_sha256=frozen.sha256(Path(__file__)))
        self.input_map_id = frozen.canonical_hash(payload)
        self._identity = {**payload,"input_map_id":self.input_map_id}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def encode(self, observation):
        import numpy as np
        values = self.original.encode(observation).copy()
        tasks = values[list(self.task_ports)]
        frozen.require(np.all((tasks == 0.) | (tasks == 1.)) and np.count_nonzero(tasks) == 60,
                       "Original task input must be one-hot over 60 ports")
        values[list(self.task_ports)] = (1.-tasks)/4.
        return values


def bind_task_polarity_policy(policy):
    """Rebind one strictly restored policy, preserving all parameters and methods."""
    from .x2_interaction_policy import SCHEMA as INTERACTION_SCHEMA
    previous = policy.identity
    frozen.require(previous["schema"] == INTERACTION_SCHEMA, "Expected original interaction policy")
    encoder = TaskPolarityEncoder(policy.encoder)
    payload = deepcopy(previous)
    payload.pop("policy_id")
    payload.update(schema=SCHEMA,parent_interaction_policy_id=previous["policy_id"],input_map=encoder.identity,
        task_polarity_implementation_sha256=frozen.sha256(Path(__file__)),
        task_polarity_intervention={"encoding":"(1-one_hot)/4", "graph_theta_heads":"unchanged",
            "state":"same archived arrays rebound to new identity for diagnosis only", "diagnostic_only":True})
    policy.encoder = encoder
    policy.policy_id = frozen.canonical_hash(payload)
    policy._identity = {**payload,"policy_id":policy.policy_id}
    return policy
