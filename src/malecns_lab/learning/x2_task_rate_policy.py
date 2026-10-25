"""Stable, explicit task-input-rate intervention; graph and heads stay unchanged."""
from copy import deepcopy
from pathlib import Path
from types import MethodType

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_task_rate_policy.v1"
INPUT_SCHEMA = "malecns_lab.x2_behavior_input.task_rate_x2.v1"
TASK_RATE_HZ = 300.
OTHER_RATE_HZ = 150.
TASK_CHANNELS = ("task.GREET", "task.FACE_LEFT", "task.FACE_RIGHT",
                 "task.SHOW_FORWARD", "task.END_INTERACTION")


class TaskRateEncoder:
    """The original occupancy and R1–R6 positions, with a new event-rate identity."""
    def __init__(self, original):
        self.original = original
        self.input_body_ids, self.input_indices = original.input_body_ids, original.input_indices
        self.channel_indices = original.channel_indices
        previous = original.identity
        frozen.require(previous["schema"] == "malecns_lab.x2_behavior_input.no_absolute_yaw.v1"
            and previous["max_rate_hz"] == OTHER_RATE_HZ, "Expected original relative-heading 150 Hz encoding")
        channels = previous["channels"]
        frozen.require({name for name in channels if name.startswith("task.")} == set(TASK_CHANNELS),
                       "Expected exactly the five existing task channels")
        indices = {channels.index(name) for name in TASK_CHANNELS}
        self.task_ports = tuple(i for i, channel in enumerate(self.channel_indices) if channel in indices)
        frozen.require(len(self.channel_indices) == len(self.input_indices) == 3377
            and all(self.channel_indices.count(channels.index(name)) == 60 for name in TASK_CHANNELS),
            "Expected unchanged five 60-port task assignments")
        self.rates_hz = tuple(TASK_RATE_HZ if channel in indices else OTHER_RATE_HZ
                              for channel in self.channel_indices)
        payload = deepcopy(previous)
        payload.pop("input_map_id")
        payload.update(schema=INPUT_SCHEMA, parent_input_map_id=previous["input_map_id"],
            max_rate_hz=TASK_RATE_HZ, units="encoded occupancy; event rate=occupancy*per_port_rate_hz",
            per_port_rate_hz=list(self.rates_hz),
            task_rate_intervention={"channels": list(TASK_CHANNELS), "task_ports": list(self.task_ports),
                "task_rate_hz": TASK_RATE_HZ, "other_rate_hz": OTHER_RATE_HZ,
                "occupancy": "unchanged [0,1]", "event_voltage_mv": previous["event_voltage_mv"],
                "mapping": "unchanged existing R1-R6 input positions; no DN stimulation"},
            task_rate_implementation_sha256=frozen.sha256(Path(__file__)))
        self.input_map_id = frozen.canonical_hash(payload)
        self._identity = {**payload, "input_map_id": self.input_map_id}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def encode(self, observation):
        # The original encoder still validates occupancy in [0,1]. Gain is in
        # the event scheduler, never encoded as an out-of-range occupancy.
        return self.original.encode(observation)


def _task_rate_step(self, occupancy, brain, phase, event_counts):
    import torch
    from ..senses.x2_behavior import DT_MS, EVENT_VOLTAGE_MV
    with torch.no_grad():
        advanced = phase + (occupancy * self.task_rates_hz) * (DT_MS / 1000)
        fired = advanced >= 1
        phase = (advanced - fired.to(torch.float32)).clamp_max(.9999999403953552)
        event_counts = event_counts + fired.to(torch.int64)
        increment = torch.zeros(self.brain.graph.n, dtype=torch.float32, device=occupancy.device)
        increment.index_copy_(0, self.input_indices, fired.to(torch.float32) * EVENT_VOLTAGE_MV)
    spikes, brain = self.brain.step(increment, brain, spike_backward_gain=self.spike_backward_gain)
    return brain, phase, event_counts, spikes


def bind_task_rate_policy(policy):
    """Rebind one already restored policy; allocate no second graph or parameters."""
    import torch
    from .x2_interaction_policy import SCHEMA as INTERACTION_SCHEMA
    previous = policy.identity
    frozen.require(previous["schema"] == INTERACTION_SCHEMA and not hasattr(policy, "task_rates_hz"),
                   "Expected unmodified strictly restored interaction policy")
    encoder = TaskRateEncoder(policy.encoder)
    policy.register_buffer("task_rates_hz", torch.tensor(encoder.rates_hz, dtype=torch.float32,
                           device=policy.input_indices.device), persistent=False)
    policy.encoder = encoder
    policy._step = MethodType(_task_rate_step, policy)
    payload = deepcopy(previous)
    payload.pop("policy_id")
    payload.update(schema=SCHEMA, parent_interaction_policy_id=previous["policy_id"],
        input_map=encoder.identity, task_rate_implementation_sha256=frozen.sha256(Path(__file__)),
        task_rate_intervention={"task_rate_hz": TASK_RATE_HZ, "other_rate_hz": OTHER_RATE_HZ,
            "graph_theta_heads": "unchanged; no optimization", "diagnostic_only": True})
    policy.policy_id = frozen.canonical_hash(payload)
    policy._identity = {**payload, "policy_id": policy.policy_id}
    return policy
