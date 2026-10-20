"""Stateful, differentiable visual-to-neural control interface.

This module defines a policy, not a trained controller. Production construction
always verifies the complete reference. Small circuits have a separate, visibly
numerical-only entry point. No environment truth or geometric errors feed the
readout; the sole sensory path is pixels -> registered input events -> brain.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ..brain.metadata import canonical_hash, file_hashes
from ..brain.policy_groups import PolicyNeuronMap, load_policy_neuron_map
from ..brain.tonic import LaminaTonicLIF
from ..brain.trainable import LIFState, TrainableLIF, TrainableLIFConfig
from ..brain.trainable_graph import load_training_graph
from ..control.body import BodyVelocity
from ..senses.spatial_drive import MASK_SHAPE, SpatialDrive, SpatialDriveConfig, SpatialDriveState, build_spatial_input_map
from ..senses.target import TargetCamera, target_mask
from .follow_task import FollowPolicyInput

POLICY_SCHEMA = "cnsbridge.stateful_neural_policy.v1"
ACTION_AXES = ("vx", "vz", "wz")
ACTION_SCALES = (0.3, 0.2, 0.5)


@dataclass(frozen=True)
class NeuralPolicyConfig:
    """All clocks and initialization are explicit; no presumed training window."""

    frame_ms: float
    precondition_ms: float
    control_dt_s: float
    readout_seed: int

    def __post_init__(self) -> None:
        for name in ("frame_ms", "precondition_ms", "control_dt_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (name != "precondition_ms" and value == 0)):
                raise ValueError("Policy times must be finite, with positive frame/control durations and nonnegative precondition")
        if type(self.readout_seed) is not int or not 0 <= self.readout_seed < 2 ** 63:
            raise ValueError("readout_seed must be an integer in [0, 2**63)")

    def steps(self, dt_ms: float) -> tuple[int, int]:
        counts = []
        for duration in (self.frame_ms, self.precondition_ms):
            ratio = duration / dt_ms
            if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
                raise ValueError("Policy durations must be exactly divisible by neural dt")
            counts.append(round(ratio))
        if counts[0] < 2 or counts[0] % 2:
            raise ValueError("A frame must contain an even positive number of neural steps")
        return counts[0], counts[1]


@dataclass(frozen=True)
class PolicyState:
    brain: LIFState
    drive: SpatialDriveState
    frame_index: int
    policy_id: str

    def detach(self) -> PolicyState:
        """TBPTT boundary: retain every value, pending arrival, phase and clock."""
        return PolicyState(self.brain.detach(), self.drive, self.frame_index, self.policy_id)


@dataclass(frozen=True)
class NeuralPolicyOutput:
    action: Tensor  # vx, vz, wz; m/s, m/s, rad/s, FLU
    features_mv: Tensor  # first-half DN means, then second-half DN means
    state: PolicyState


def _bound_identity(identity: dict, field: str) -> None:
    payload = deepcopy(identity)
    claimed = payload.pop(field, None)
    if claimed != canonical_hash(payload):
        raise ValueError(f"Invalid {field} identity payload")


class NeuralPolicy(nn.Module):
    """Explicit-state CPU policy; construct through a role-specific factory."""

    def __init__(self):
        raise TypeError("Use NeuralPolicy.from_reference or NeuralPolicy.for_numerical_test")

    @classmethod
    def from_reference(cls, reference_path: str | Path, *, background: str,
                       config: NeuralPolicyConfig, camera: TargetCamera,
                       drive_config: SpatialDriveConfig, lif_config: TrainableLIFConfig | None = None,
                       dtype: torch.dtype = torch.float32) -> NeuralPolicy:
        """Verify all reference nodes/edges and metadata; no reduced/fallback path."""
        if background not in ("zero_background", "fixed_lamina_8mv"):
            raise ValueError("Choose zero_background or fixed_lamina_8mv explicitly")
        graph = load_training_graph(reference_path)
        mapping = load_policy_neuron_map(graph, reference_path)
        kwargs = {"config": lif_config, "dtype": dtype}
        brain = (TrainableLIF(graph, mapping.inputs.body_ids, **kwargs) if background == "zero_background" else
                 LaminaTonicLIF(graph, mapping.inputs.body_ids, mode="fixed", **kwargs))
        instance = cls.__new__(cls)
        instance._initialize(brain, mapping, background=background, config=config, camera=camera,
                             drive_config=drive_config, role="full_reference")
        return instance

    @classmethod
    def for_numerical_test(cls, brain: TrainableLIF, mapping: PolicyNeuronMap, *, background: str,
                           config: NeuralPolicyConfig, camera: TargetCamera,
                           drive_config: SpatialDriveConfig) -> NeuralPolicy:
        """Use an existing explicitly numerical circuit for interface checks only."""
        instance = cls.__new__(cls)
        instance._initialize(brain, mapping, background=background, config=config, camera=camera,
                             drive_config=drive_config, role="numerical_test_only")
        return instance

    def _initialize(self, brain, mapping, *, background, config, camera, drive_config, role):
        nn.Module.__init__(self)
        if type(brain) not in (TrainableLIF, LaminaTonicLIF) or not isinstance(mapping, PolicyNeuronMap):
            raise ValueError("Use an existing supported brain and validated policy neuron map")
        if not isinstance(config, NeuralPolicyConfig) or not isinstance(camera, TargetCamera):
            raise ValueError("Explicit NeuralPolicyConfig and TargetCamera are required")
        if not isinstance(drive_config, SpatialDriveConfig) or drive_config.dt_ms != brain.config.dt_ms:
            raise ValueError("Spatial drive dt must match the neural model dt")
        self.frame_steps, self.precondition_steps = config.steps(brain.config.dt_ms)
        graph_identity, map_identity = brain.graph.identity, mapping.identity
        _bound_identity(brain.identity, "model_id")
        _bound_identity(graph_identity, "training_graph_id")
        _bound_identity(map_identity, "policy_map_id")
        if (graph_identity["role"] != role or map_identity["role"] != role
                or map_identity["training_graph_id"] != graph_identity["training_graph_id"]
                or mapping.n_nodes != brain.graph.n or map_identity["nodes"] != brain.graph.n):
            raise ValueError("Policy graph and neuron-map role/identity must agree")
        for name, selected in (("inputs", mapping.inputs), ("outputs", mapping.outputs)):
            ids, indices = selected.body_ids, selected.indices
            if (selected.to_dict() != map_identity[name] or ids.ndim != 1 or not len(ids)
                    or indices.shape != ids.shape or ids.dtype != np.int64 or indices.dtype != np.int64
                    or np.any(np.diff(ids) <= 0) or np.any(indices < 0) or np.any(indices >= brain.graph.n)
                    or not np.array_equal(brain.graph.body_ids[indices], ids)):
                raise ValueError("Neuron selections must retain bound ascending body IDs and graph indices")
        if (tuple(mapping.inputs.body_ids) != brain.input_body_ids or "R1-R6" not in brain.graph.group_names
                or not np.array_equal(mapping.inputs.indices,
                                      np.flatnonzero(brain.graph.group_index == brain.graph.group_names.index("R1-R6")))):
            raise ValueError("Policy input must be the complete registered R1-R6 selection")
        if background == "zero_background":
            if type(brain) is not TrainableLIF:
                raise ValueError("zero_background requires the original TrainableLIF")
        elif background == "fixed_lamina_8mv":
            if type(brain) is not LaminaTonicLIF or brain.tonic_mode != "fixed":
                raise ValueError("fixed_lamina_8mv requires fixed tonic dynamics")
        else:
            raise ValueError("Choose zero_background or fixed_lamina_8mv explicitly")
        self.brain, self.mapping = brain, mapping
        self.config, self.camera, self.drive_config, self.background = config, camera, drive_config, background
        self.spatial_map = build_spatial_input_map(mapping)
        drive = SpatialDrive(self.spatial_map, drive_config)
        self.register_buffer("dn_indices", torch.tensor(mapping.outputs.indices.copy(), dtype=torch.int64), persistent=False)
        self.register_buffer("action_scales", torch.tensor(ACTION_SCALES, dtype=brain._dtype), persistent=False)
        # Linear.reset_parameters consumes the global generator. Restore that
        # state, then initialize entirely through an independent CPU generator.
        with torch.random.fork_rng(devices=[]):
            self.readout = nn.Linear(2 * len(mapping.outputs.indices), 3, dtype=brain._dtype, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(config.readout_seed)
        with torch.no_grad():
            self.readout.weight.normal_(std=0.01 / math.sqrt(self.readout.in_features), generator=generator)
            self.readout.bias.zero_()
        self._validate_parameters()
        package = Path(__file__).resolve().parents[1]
        sources = ("learning/neural_policy.py", "learning/follow_task.py", "senses/target.py", "senses/spatial_drive.py",
                   "brain/policy_groups.py", "brain/metadata.py", "control/body.py")
        self._identity = {
            "schema": POLICY_SCHEMA, "role": role, "model": brain.identity, "neurons": mapping.identity,
            "spatial": self.spatial_map.identity, "drive": drive.identity, "camera": asdict(camera),
            "config": asdict(config), "background": background, "dtype": str(brain._dtype),
            "frame_steps": self.frame_steps, "precondition_steps": self.precondition_steps,
            "features": {"ordering": "first_half_DN_then_second_half_DN; ascending body ID in each half",
                         "body_ids": mapping.outputs.body_ids.tolist(), "width": self.readout.in_features,
                         "sample": "post_reset_v_each_neural_step", "reduction": "arithmetic_mean_per_half",
                         "raw_units": "mV", "normalization": "(mean_v-v_rest)/(v_threshold-v_rest)",
                         "offset_mv": brain.config.v_rest_mv,
                         "scale_mv": brain.config.v_threshold_mv - brain.config.v_rest_mv},
            "readout": {"architecture": "linear_bias_then_tanh_then_axis_scales", "axes": list(ACTION_AXES),
                        "scales": list(ACTION_SCALES), "units": ["m/s", "m/s", "rad/s"], "frame": "body_flu",
                        "weight_initialization": "local_CPU_Generator.normal(std=0.01/sqrt(feature_width)); bias=0"},
            "training_parameters": ["brain.theta", "readout.weight", "readout.bias"],
            "boundary": "engineered input and readout; explicit neural/control clocks; no trained behavior asserted",
            "implementation_sha256": {name: file_hashes(package / name)["sha256"] for name in sources},
        }
        self.policy_id = canonical_hash(self._identity)

    @property
    def identity(self) -> dict:
        return {**deepcopy(self._identity), "policy_id": self.policy_id}

    def _validate_parameters(self) -> None:
        if set(dict(self.named_parameters())) != {"brain.theta", "readout.weight", "readout.bias"}:
            raise ValueError("Only brain.theta and the linear readout may be policy parameters")
        for value in self.parameters():
            if value.device.type != "cpu" or value.dtype != self.brain._dtype or not torch.isfinite(value).all():
                raise ValueError("Policy parameters must retain their finite CPU dtype")
        if self.background == "fixed_lamina_8mv" and (self.brain.rho.requires_grad or torch.count_nonzero(self.brain.rho)):
            raise ValueError("The fixed 8 mV policy requires unchanged, nontrainable rho=0")
        if (self.dn_indices.device.type != "cpu" or self.dn_indices.dtype != torch.int64
                or not np.array_equal(self.dn_indices.numpy(), self.mapping.outputs.indices)
                or self.action_scales.device.type != "cpu" or self.action_scales.dtype != self.brain._dtype
                or not torch.equal(self.action_scales, self.action_scales.new_tensor(ACTION_SCALES))):
            raise ValueError("Policy output selection, axes and dtype must retain their registered values")

    def _drive_for(self, state: PolicyState) -> SpatialDrive:
        if (not isinstance(state, PolicyState) or state.policy_id != self.policy_id
                or not isinstance(state.brain, LIFState) or not isinstance(state.drive, SpatialDriveState)
                or type(state.frame_index) is not int or state.frame_index < 0):
            raise ValueError("Policy state identity/frame clock mismatch")
        expected_step = self.precondition_steps + state.frame_index * self.frame_steps
        if state.brain.step != expected_step or state.drive.step != expected_step:
            raise ValueError("Policy neural and drive clocks must agree with the frame index")
        self.brain._validate_state(state.brain)
        drive = SpatialDrive(self.spatial_map, self.drive_config)
        drive.restore_state(state.drive)
        return drive

    @torch.no_grad()
    def initial_state(self) -> PolicyState:
        """Explicit scene reset and blank preconditioning; not a steady-state claim."""
        self._validate_parameters()
        brain, drive = self.brain.reset(), SpatialDrive(self.spatial_map, self.drive_config)
        blank = np.zeros(MASK_SHAPE, dtype=np.float32)
        for _ in range(self.precondition_steps):
            _, brain = self.brain.step(drive.step(blank).to_torch(dtype=self.brain._dtype), brain)
        return PolicyState(brain, drive.export_state(), 0, self.policy_id)

    def forward_tensor(self, frame: np.ndarray, state: PolicyState) -> NeuralPolicyOutput:
        """Advance one explicit frame; all neural features and actions retain gradients."""
        mask = target_mask(frame, self.camera, MASK_SHAPE)  # Validate before advancing either clock.
        self._validate_parameters()
        drive = self._drive_for(state)
        brain = state.brain
        half_steps = self.frame_steps // 2
        sums = [brain.v.new_zeros(len(self.mapping.outputs.indices)) for _ in range(2)]
        for step in range(self.frame_steps):
            _, brain = self.brain.step(drive.step(mask).to_torch(dtype=self.brain._dtype), brain)
            half = step // half_steps
            sums[half] = sums[half] + brain.v[self.dn_indices]
        features = torch.cat(sums) / half_steps
        cfg = self.brain.config
        normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
        action = torch.tanh(self.readout(normalized)) * self.action_scales
        return NeuralPolicyOutput(action, features,
                                  PolicyState(brain, drive.export_state(), state.frame_index + 1, self.policy_id))


class NeuralController:
    """Deployment/rollout boundary; reset explicitly before each scene.

    Observation errors and confidence are deliberately unused. The existing
    rollout owns target-loss intervention and logs the original neural action.
    """

    def __init__(self, policy: NeuralPolicy):
        if not isinstance(policy, NeuralPolicy):
            raise ValueError("A NeuralPolicy is required")
        self.policy, self.state = policy, None

    def reset(self) -> PolicyState:
        self.state = self.policy.initial_state()
        return self.state

    @property
    def neural_time_ms(self) -> float | None:
        return None if self.state is None else self.state.brain.step * self.policy.brain.config.dt_ms

    @property
    def control_time_s(self) -> float | None:
        return None if self.state is None else self.state.frame_index * self.policy.config.control_dt_s

    @torch.no_grad()
    def __call__(self, input_: FollowPolicyInput) -> BodyVelocity:
        if self.state is None:
            raise ValueError("Reset the neural controller explicitly at the scene boundary")
        if not isinstance(input_, FollowPolicyInput) or input_.camera != self.policy.camera:
            raise ValueError("Controller input must use its registered camera")
        if (isinstance(input_.time_s, bool) or not isinstance(input_.time_s, (int, float))
                or not math.isfinite(input_.time_s)
                or not math.isclose(input_.time_s, self.control_time_s, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Controller input time must match its registered control clock; reset for a new scene")
        output = self.policy.forward_tensor(input_.frame, self.state)
        vx, vz, wz = output.action.tolist()
        command = BodyVelocity(vx=vx, vz=vz, wz=wz)
        self.state = output.state
        return command
