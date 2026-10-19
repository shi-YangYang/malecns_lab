"""Distinct MPS policy preserving the registered CPU sensory and readout rules."""
import numpy as np
import torch
from torch import nn

from ..brain.mps_identity import policy_identity
from ..brain.mps_trainable import MPSLIF
from ..brain.policy_groups import load_policy_neuron_map
from ..brain.trainable import TrainableLIFConfig
from ..brain.trainable_graph import load_training_graph
from ..senses.spatial_drive import MASK_SHAPE, SpatialDrive, SpatialDriveConfig, build_spatial_input_map
from ..senses.target import TargetCamera, target_mask
from .neural_policy import NeuralPolicy, NeuralPolicyConfig, PolicyState, NeuralPolicyOutput, ACTION_SCALES, _bound_identity


class MPSPolicy(NeuralPolicy):
    """Reuse explicit state/clock validation; never initialize or relabel a CPU brain."""
    @classmethod
    def from_reference(cls, path, cpu_identity):
        if cpu_identity["role"] != "full_reference":
            raise ValueError("Production MPS policy requires full_reference")
        graph = load_training_graph(path)
        mapping = load_policy_neuron_map(graph, path)
        brain = MPSLIF(graph, mapping.inputs.body_ids, config=TrainableLIFConfig(**cpu_identity["model"]["config"]))
        return cls.from_brain(brain, mapping, cpu_identity)

    @classmethod
    def from_brain(cls, brain, mapping, cpu_identity):
        """Construct against an identity already validated by the CPU archive reader."""
        _bound_identity(cpu_identity, "policy_id")
        if (type(brain) is not MPSLIF or cpu_identity["background"] != "fixed_lamina_8mv"
                or cpu_identity["dtype"] != "torch.float32" or cpu_identity["model"]["graph"] != brain.graph.identity
                or cpu_identity["neurons"] != mapping.identity
                or tuple(mapping.inputs.body_ids) != brain.input_body_ids
                or cpu_identity["role"] != brain.graph.identity["role"]
                or cpu_identity["model"]["config"] != brain.identity["config"]):
            raise ValueError("Source identity, full graph, neuron map or dynamics differs")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.brain, instance.mapping = brain, mapping
        instance.config = NeuralPolicyConfig(**cpu_identity["config"])
        instance.camera = TargetCamera(**cpu_identity["camera"])
        instance.drive_config = SpatialDriveConfig(**cpu_identity["drive"]["config"])
        instance.background = "fixed_lamina_8mv"
        instance.frame_steps, instance.precondition_steps = instance.config.steps(brain.config.dt_ms)
        instance.spatial_map = build_spatial_input_map(mapping)
        drive = SpatialDrive(instance.spatial_map, instance.drive_config)
        if (drive.identity != cpu_identity["drive"] or instance.spatial_map.identity != cpu_identity["spatial"]
                or instance.frame_steps != cpu_identity["frame_steps"] or instance.precondition_steps != cpu_identity["precondition_steps"]
                or cpu_identity["features"]["body_ids"] != mapping.outputs.body_ids.tolist()
                or cpu_identity["features"]["width"] != 2 * len(mapping.outputs.indices)):
            raise ValueError("Sensory map, output selection or clocks differ")
        instance.register_buffer("dn_indices", torch.tensor(mapping.outputs.indices.copy(), device="mps"), persistent=False)
        instance.register_buffer("action_scales", torch.tensor(ACTION_SCALES, device="mps"), persistent=False)
        # Only a shape allocation; all parameters must be restored before any forward.
        with torch.random.fork_rng(devices=[]):
            instance.readout = nn.Linear(cpu_identity["features"]["width"], 3, device="cpu").to("mps")
        instance._restored = False
        identity = policy_identity(cpu_identity)
        if identity["model"] != brain.identity:
            raise ValueError("Actual MPS model differs from declared source migration")
        instance.policy_id = identity.pop("policy_id")
        instance._identity = identity
        return instance

    def _validate_parameters(self):
        if not self._restored:
            raise ValueError("MPS continuation requires explicit parameter restoration")
        if set(dict(self.named_parameters())) != {"brain.theta", "readout.weight", "readout.bias"}:
            raise ValueError("Unexpected trainable parameter")
        if any(p.device.type != "mps" or p.dtype != torch.float32 or not p.requires_grad or not torch.isfinite(p).all()
               for p in self.parameters()):
            raise ValueError("Parameters must remain finite MPS float32")
        if self.brain.rho.requires_grad or torch.count_nonzero(self.brain.rho):
            raise ValueError("Fixed background changed")
        if (not np.array_equal(self.dn_indices.cpu().numpy(), self.mapping.outputs.indices)
                or self.dn_indices.device.type != "mps" or self.dn_indices.dtype != torch.int64
                or self.action_scales.device.type != "mps" or self.action_scales.dtype != torch.float32
                or not torch.equal(self.action_scales.cpu(), torch.tensor(ACTION_SCALES))):
            raise ValueError("Readout selection/scales changed")

    @torch.no_grad()
    def initial_state(self):
        self._validate_parameters()
        brain, drive = self.brain.reset(), SpatialDrive(self.spatial_map, self.drive_config)
        blank = np.zeros(MASK_SHAPE, np.float32)
        for _ in range(self.precondition_steps):
            _, brain = self.brain.step(drive.step(blank).to_torch(dtype=torch.float32).to("mps"), brain)
        return PolicyState(brain, drive.export_state(), 0, self.policy_id)

    def forward_tensor(self, frame, state):
        mask = target_mask(frame, self.camera, MASK_SHAPE)
        self._validate_parameters()
        drive = self._drive_for(state)
        brain, half_steps = state.brain, self.frame_steps // 2
        sums = [brain.v.new_zeros(len(self.mapping.outputs.indices)) for _ in range(2)]
        for step in range(self.frame_steps):
            increment = drive.step(mask).to_torch(dtype=torch.float32).to("mps")
            _, brain = self.brain.step(increment, brain)
            half = step // half_steps
            sums[half] = sums[half] + brain.v[self.dn_indices]
        features = torch.cat(sums) / half_steps
        cfg = self.brain.config
        normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
        action = torch.tanh(self.readout(normalized)) * self.action_scales
        return NeuralPolicyOutput(action, features, PolicyState(brain, drive.export_state(), state.frame_index + 1, self.policy_id))
