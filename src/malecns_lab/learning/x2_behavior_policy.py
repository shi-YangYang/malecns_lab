"""New full-connectome X2 policy; engineered sensory input, neural-only heads.

This is not an identity-preserving optimization of the historical RGB policy.
The original MPS LIF equations are reused; the sensory contract, three heads,
checkpoint and persistent session state have separate versioned identities.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..brain.mps_trainable import MPSLIF
from ..brain.policy_groups import load_policy_neuron_map
from ..brain.trainable import LIFState, TrainableLIFConfig
from ..brain.trainable_graph import load_training_graph
from ..control.x2_behavior import BEHAVIORS, PHASES, Decision, Observation, digest, require
from ..senses.x2_behavior import BehaviorEncoder, DT_MS, EVENT_VOLTAGE_MV, MAX_RATE_HZ

SCHEMA = "malecns_lab.x2_behavior_policy.v1"
PARAMETERS = ("brain.theta", "behavior_head.weight", "behavior_head.bias", "phase_head.weight",
              "phase_head.bias", "parameter_head.weight", "parameter_head.bias")
BLOCK_STEPS = 50


def backward_identity(gain):
    require(type(gain) in (int, float) and gain in (1.0, 0.1), "Unregistered spike backward gain")
    payload = {"schema": "malecns_lab.x2_spike_backward.v1", "spike_backward_gain": float(gain),
               "algorithm": "spikes.detach()+gain*(spikes-spikes.detach()); gain=1 returns original tensor",
               "meaning": "multiply each original surrogate derivative; unchanged binary forward and curve width",
               "credit": "no intermediate state detach inside two 500-step decisions",
               "loss_backward_scale": 1.0}
    return {**payload, "backward_id": digest(payload)}


@dataclass(frozen=True)
class BehaviorState:
    brain: LIFState
    phase: torch.Tensor
    event_counts: torch.Tensor
    frame_index: int
    observation_seq: int
    session_id: str
    policy_id: str

    def detach(self):
        return BehaviorState(self.brain.detach(), self.phase.detach(), self.event_counts.detach(),
                             self.frame_index, self.observation_seq, self.session_id, self.policy_id)


@dataclass(frozen=True)
class BehaviorOutput:
    behavior_logits: torch.Tensor
    phase_logits: torch.Tensor
    parameter_raw: torch.Tensor
    intensity: torch.Tensor
    features_mv: torch.Tensor
    state: BehaviorState


def array_identity(array):
    array = np.asarray(array)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


class X2BehaviorPolicy(nn.Module):
    frame_steps = 500
    precondition_steps = 600

    @classmethod
    def from_reference(cls, path, source_identity, *, theta, initialization="migrated", seed=2401,
                       spike_backward_gain=1.0):
        require(initialization in ("migrated", "neutral"), "Unknown theta initialization")
        graph = load_training_graph(path)
        require(graph.identity == source_identity["model"]["graph"]
                and graph.n == 166700 and graph.counts.nnz == 25582938
                and graph.identity["synapse_count"] == 124177617, "Full graph/type order differs from migration source")
        mapping = load_policy_neuron_map(graph, path)
        require(mapping.identity == source_identity["neurons"], "Complete input/DN mapping differs from migration source")
        brain = MPSLIF(graph, mapping.inputs.body_ids,
                       config=TrainableLIFConfig(**source_identity["model"]["config"]))
        return cls(brain, mapping, theta=theta, initialization=initialization, seed=seed,
                   migration_source=source_identity["policy_id"], spike_backward_gain=spike_backward_gain)

    def __init__(self, brain, mapping, *, theta, initialization, seed, migration_source, spike_backward_gain=1.0):
        super().__init__()
        require(type(brain) is MPSLIF and brain.graph.identity["role"] == "full_reference",
                "Production requires the actual complete MPS brain")
        require(brain.graph.n == 166700 and brain.graph.counts.nnz == 25582938
                and len(mapping.outputs.body_ids) == 1314 and brain.config.dt_ms == DT_MS,
                "Complete graph or neural clock differs")
        require(type(seed) is int and 0 <= seed < 2**32, "Invalid initialization seed")
        self.brain, self.mapping = brain, mapping
        self.spike_backward_gain = backward_identity(spike_backward_gain)["spike_backward_gain"]
        self.encoder = BehaviorEncoder.from_policy_map(mapping)
        require(tuple(brain.input_body_ids) == self.encoder.input_body_ids, "Sensory ordering differs")
        require(isinstance(theta, np.ndarray) and theta.dtype == np.float32
                and theta.shape == tuple(brain.theta.shape) and np.isfinite(theta).all(), "Invalid migrated theta")
        self.register_buffer("input_indices", torch.tensor(self.encoder.input_indices, dtype=torch.int64, device="mps"), persistent=False)
        self.register_buffer("dn_indices", torch.tensor(mapping.outputs.indices.copy(), dtype=torch.int64, device="mps"), persistent=False)
        with torch.no_grad():
            values = theta.copy() if initialization == "migrated" else np.zeros_like(theta)
            brain.theta.copy_(torch.from_numpy(values).to("mps"))
            require(np.array_equal(brain.theta.cpu().numpy(), values), "Theta device transfer changed values")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # CPU initialization is reproducible and does not inherit old heads.
            self.behavior_head = nn.Linear(2628, 5).to("mps")
            self.phase_head = nn.Linear(2628, 3).to("mps")
            self.parameter_head = nn.Linear(2628, 1).to("mps")
        payload = {"schema": SCHEMA, "model": brain.identity, "neurons": mapping.identity,
                   "input_map": self.encoder.identity, "features": {"width": 2628,
                       "body_ids": mapping.outputs.body_ids.tolist(), "aggregation": "DN mean voltage in consecutive 250-step halves",
                       "normalization": "(voltage-v_rest)/(v_threshold-v_rest)"},
                   "heads": {"behavior": [str(x) for x in BEHAVIORS], "phase": [str(x) for x in PHASES],
                             "parameter": "sigmoid of one linear output; all three heads use only neural features"},
                   "parameters": list(PARAMETERS), "frame_steps": 500, "precondition_steps": 600,
                   "state": "persistent within session; reset only explicit new session; credit cut is training-only",
                   "initialization": {"theta": initialization, "source_policy_id": migration_source,
                       "theta_sha256": array_identity(values)["sha256"], "head_seed": seed,
                       "heads": "torch.nn.Linear default Kaiming uniform weight and uniform bias on CPU; 5 then 3 then 1",
                       "optimizer": "new; no inherited Adam state", "phase": "PCG64(2401), float32, clamped nextbelow(1)"},
                   "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "torch_version": torch.__version__, "numpy_version": np.__version__}
        # Forward LIF semantics stay historical, but its source hash and this
        # estimator are new identities, never an in-place checkpoint rewrite.
        payload["backward"] = backward_identity(self.spike_backward_gain)
        payload["mps_step_sha256"] = hashlib.sha256(Path(__file__).parents[1].joinpath("brain/mps_trainable.py").read_bytes()).hexdigest()
        self.policy_id = digest(payload)
        self._identity = {**payload, "policy_id": self.policy_id}
        self.validate_parameters()

    @property
    def identity(self):
        return deepcopy(self._identity)

    def validate_parameters(self):
        require(tuple(dict(self.named_parameters())) == PARAMETERS, "Unexpected trainable parameter inventory")
        require(all(p.device.type == "mps" and p.dtype == torch.float32 and torch.isfinite(p).all()
                    for p in self.parameters()), "Nonfinite or foreign parameters")
        require(not self.brain.rho.requires_grad and not torch.count_nonzero(self.brain.rho), "Fixed lamina background changed")

    def validate_state(self, state):
        require(isinstance(state, BehaviorState) and state.policy_id == self.policy_id
                and type(state.frame_index) is int and state.frame_index >= 0
                and type(state.observation_seq) is int and state.observation_seq >= -1
                and isinstance(state.session_id, str) and state.session_id, "Invalid behavior state identity")
        require(state.brain.step == self.precondition_steps + state.frame_index * self.frame_steps, "Behavior neural clock differs")
        self.brain._validate_state(state.brain)
        require(state.phase.shape == state.event_counts.shape == (3377,)
                and state.phase.dtype == torch.float32 and state.event_counts.dtype == torch.int64
                and state.phase.device.type == state.event_counts.device.type == "mps"
                and not state.phase.requires_grad and not state.event_counts.requires_grad
                and torch.isfinite(state.phase).all() and not (state.phase < 0).any()
                and not (state.phase >= 1).any() and not (state.event_counts < 0).any()
                and not (state.event_counts > state.brain.step).any(), "Invalid sensory state")

    def _step(self, occupancy, brain, phase, event_counts):
        # Fixed encoding, actual MPSLIF equations; no teacher/current task in heads.
        with torch.no_grad():
            advanced = phase + (occupancy * MAX_RATE_HZ) * (DT_MS / 1000)
            fired = advanced >= 1
            phase = (advanced - fired.to(torch.float32)).clamp_max(.9999999403953552)
            event_counts = event_counts + fired.to(torch.int64)
            increment = torch.zeros(self.brain.graph.n, dtype=torch.float32, device="mps")
            increment.index_copy_(0, self.input_indices, fired.to(torch.float32) * EVENT_VOLTAGE_MV)
        spikes, brain = self.brain.step(increment, brain, spike_backward_gain=self.spike_backward_gain)
        return brain, phase, event_counts, spikes

    @torch.no_grad()
    def initial_state(self, session_id, *, telemetry=None):
        self.validate_parameters()
        require(isinstance(session_id, str) and session_id, "Explicit session ID required")
        initial = np.random.Generator(np.random.PCG64(2401)).random(3377).astype(np.float32)
        phase = torch.from_numpy(initial).to("mps").clamp_max(.9999999403953552)
        counts = torch.zeros(3377, dtype=torch.int64, device="mps")
        brain, blank = self.brain.reset(), torch.zeros(3377, dtype=torch.float32, device="mps")
        for step in range(self.precondition_steps):
            brain, phase, counts, spikes = self._step(blank, brain, phase, counts)
            if telemetry is not None:
                telemetry.capture(spikes, step)
                if (step + 1) % BLOCK_STEPS == 0:
                    telemetry.flush(step + 1)
        state = BehaviorState(brain, phase, counts, 0, -1, session_id, self.policy_id)
        self.validate_state(state)
        return state

    def read_features(self, features_mv):
        require(tuple(features_mv.shape) == (2628,), "All DN feature halves are required")
        cfg = self.brain.config
        normalized = (features_mv - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
        behavior = self.behavior_head(normalized)
        phase = self.phase_head(normalized)
        raw = self.parameter_head(normalized)
        return behavior, phase, raw, torch.sigmoid(raw)

    def forward(self, observation, state, *, checkpoint_blocks=True, telemetry=None, boundary_observer=None):
        require(isinstance(observation, Observation) and observation.session_id == state.session_id
                and observation.observation_seq > state.observation_seq, "Observation is foreign, duplicate or reordered")
        require(telemetry is None or not torch.is_grad_enabled(), "Telemetry cannot capture autograd recomputation")
        self.validate_parameters()
        self.validate_state(state)
        occupancy = torch.from_numpy(self.encoder.encode(observation)).to("mps")
        brain, phase, counts = state.brain, state.phase, state.event_counts
        sums = [torch.zeros(1314, device="mps") for _ in range(2)]
        for start in range(0, self.frame_steps, BLOCK_STEPS):
            offset, brain_step = start, brain.step

            def block(v, g, refractory, sensory, events, *pending_and_sums,
                      offset=offset, brain_step=brain_step):
                current = LIFState(v, g, refractory, tuple(pending_and_sums[:self.brain.delay_steps]),
                                   brain_step, self.brain.model_id)
                totals = list(pending_and_sums[self.brain.delay_steps:])
                for index in range(BLOCK_STEPS):
                    current, sensory, events, spikes = self._step(occupancy, current, sensory, events)
                    half = (offset + index) // 250
                    totals[half] = totals[half] + current.v[self.dn_indices]
                    if telemetry is not None:
                        telemetry.capture(spikes, current.step - 1)
                return current.v, current.g, current.refractory_until, sensory, events, *current.pending, *totals

            args = (brain.v, brain.g, brain.refractory_until, phase, counts, *brain.pending, *sums)
            values = (checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)
                      if checkpoint_blocks and torch.is_grad_enabled() else block(*args))
            brain = LIFState(values[0], values[1], values[2], tuple(values[5:5+self.brain.delay_steps]),
                             brain.step + BLOCK_STEPS, self.brain.model_id)
            phase, counts, sums = values[3], values[4], list(values[-2:])
            if boundary_observer is not None:
                boundary_observer(brain)
            if telemetry is not None:
                telemetry.flush(brain.step)
        features = torch.cat(sums) / 250
        next_state = BehaviorState(brain, phase, counts, state.frame_index + 1,
                                   observation.observation_seq, state.session_id, self.policy_id)
        self.validate_state(next_state)
        behavior, phase_logits, raw, intensity = self.read_features(features)
        require(all(torch.isfinite(x).all() for x in (features, behavior, phase_logits, raw)), "Nonfinite neural readout")
        return BehaviorOutput(behavior, phase_logits, raw, intensity, features, next_state)

    def state_arrays(self, state):
        self.validate_state(state)
        values = {"v": state.brain.v, "g": state.brain.g, "refractory_until": state.brain.refractory_until,
                  "phase": state.phase, "event_counts": state.event_counts,
                  **{f"pending.{i:02d}": x for i, x in enumerate(state.brain.pending)}}
        return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}

    def state_identity(self, state, arrays=None):
        arrays = self.state_arrays(state) if arrays is None else arrays
        payload = {"schema": "malecns_lab.x2_behavior_state.v1", "policy_id": self.policy_id,
                   "session_id": state.session_id, "frame_index": state.frame_index,
                   "observation_seq": state.observation_seq, "neural_step": state.brain.step,
                   "arrays": {name: array_identity(value) for name, value in arrays.items()}}
        return {**payload, "state_id": digest(payload)}

    def decision(self, observation, output, *, checkpoint_id, state_before_id, decision_seq):
        behavior = tuple(float(x) for x in output.behavior_logits.detach().cpu().tolist())
        phase = tuple(float(x) for x in output.phase_logits.detach().cpu().tolist())
        raw = tuple(float(x) for x in output.parameter_raw.detach().cpu().tolist())
        return Decision(observation.run_id, observation.session_id, decision_seq, observation.request_id,
            observation.observation_seq, state_before_id, self.state_identity(output.state)["state_id"],
            behavior, phase, raw, BEHAVIORS[max(range(5), key=behavior.__getitem__)],
            PHASES[max(range(3), key=phase.__getitem__)], float(output.intensity.detach().cpu()[0]),
            self.policy_id, checkpoint_id, self.encoder.input_map_id,
            output.state.brain.step - self.frame_steps, output.state.brain.step)
