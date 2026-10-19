"""Complete-connectome policy with GPU sensory state, readout and credit cuts."""
from dataclasses import dataclass
import numpy as np
import torch
from ..senses.gpu_drive import GPUDrive, GPUDriveState
from ..brain.mps_trainable import MPSLIF
from ..brain.trainable import TrainableLIFConfig
from ..brain.trainable_graph import load_training_graph
from ..brain.policy_groups import load_policy_neuron_map
from ..senses.spatial_drive import MASK_SHAPE
from .mps_policy import MPSPolicy
from .neural_policy import PolicyState
from .gpu_recompute import checkpoint_frame
from .gpu_contract import policy_identity
from .training_contract import PARAMETERS, require


@dataclass(frozen=True)
class GPUOutput:
    logits: torch.Tensor
    action: torch.Tensor
    features_mv: torch.Tensor
    state: PolicyState


class GPUPolicy(MPSPolicy):
    @classmethod
    def from_reference(cls, path, cpu_identity, *, credit_horizon):
        require(cpu_identity["role"] == "full_reference", "Production requires the complete reference")
        graph = load_training_graph(path)
        mapping = load_policy_neuron_map(graph, path)
        brain = MPSLIF(graph, mapping.inputs.body_ids, config=TrainableLIFConfig(**cpu_identity["model"]["config"]))
        return cls.from_brain(brain, mapping, cpu_identity, credit_horizon=credit_horizon)

    @classmethod
    def from_brain(cls, brain, mapping, cpu_identity, *, credit_horizon):
        instance = super().from_brain(brain, mapping, cpu_identity)
        return instance._configure(cpu_identity, credit_horizon)

    def _configure(self, cpu_identity, horizon):
        require(type(horizon) is int and horizon in (100, 250, 500, 1000), "Explicit supported credit horizon required")
        self.credit_horizon = horizon
        self.drive = GPUDrive(self.spatial_map, self.drive_config, self.camera)
        identity = policy_identity(cpu_identity, horizon)
        require(identity["drive"] == self.drive.identity and identity["model"] == self.brain.identity,
                "Actual GPU model or drive identity differs")
        self.policy_id = identity.pop("policy_id")
        self._identity = identity
        return self

    @torch.no_grad()
    def restore_parameters(self, values):
        require(set(values) == set(PARAMETERS), "Parameter inventory differs")
        named = dict(self.named_parameters())
        for key, name in PARAMETERS.items():
            value, parameter = values[key], named[name]
            require(value.dtype == np.float32 and value.shape == tuple(parameter.shape)
                    and np.isfinite(value).all(), "Invalid inherited parameter")
            parameter.copy_(torch.from_numpy(value.copy()).to("mps"))
            require(np.array_equal(parameter.cpu().numpy(), value), "Parameter device transfer changed values")
        self._restored = True
        self._validate_parameters()

    def validate_state(self, state):
        require(isinstance(state, PolicyState) and isinstance(state.drive, GPUDriveState)
                and state.policy_id == self.policy_id and type(state.frame_index) is int and state.frame_index >= 0,
                "GPU policy state identity differs")
        step = self.precondition_steps + state.frame_index * self.frame_steps
        require(state.brain.step == state.drive.step == step, "GPU clocks differ")
        self.brain._validate_state(state.brain)
        self.drive.validate(state.drive)

    @torch.no_grad()
    def initial_state(self):
        self._validate_parameters()
        brain, sensory = self.brain.reset(), self.drive.initial_state()
        blank = torch.zeros(MASK_SHAPE, dtype=torch.float32, device="mps")
        for _ in range(self.precondition_steps):
            increment, sensory = self.drive.step(blank, sensory)
            _, brain = self.brain.step(increment, brain)
        return PolicyState(brain, sensory, 0, self.policy_id)

    def forward_tensor(self, frame, state, *, frame_offset=0, direct_numerical_fixture=False):
        require(type(frame_offset) is int and frame_offset in (0, self.frame_steps), "Invalid two-frame credit offset")
        require(not direct_numerical_fixture or self.brain.graph.identity["role"] == "numerical_test_only",
                "Direct execution is fixture-only")
        self._validate_parameters()
        self.validate_state(state)
        mask = self.drive.mask(frame)
        sums = [state.brain.v.new_zeros(len(self.mapping.outputs.indices)) for _ in range(2)]
        brain, sensory, sums = checkpoint_frame(self, mask, state.brain, state.drive, sums,
                                                 frame_offset=frame_offset, direct=direct_numerical_fixture)
        features = torch.cat(sums) / (self.frame_steps // 2)
        cfg = self.brain.config
        normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
        logits = self.readout(normalized)
        return GPUOutput(logits, torch.tanh(logits) * self.action_scales, features,
                         PolicyState(brain, sensory, state.frame_index + 1, self.policy_id))
