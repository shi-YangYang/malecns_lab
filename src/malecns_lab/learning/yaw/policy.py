"""Independent camera-only MPS policy over every node/edge of the reference."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...brain.metadata import canonical_hash, file_hashes
from ...brain.mps_trainable import MPSLIF
from ...brain.policy_groups import load_policy_neuron_map
from ...brain.trainable import LIFState, TrainableLIFConfig
from ...brain.trainable_graph import load_training_graph
from ...senses.encoding_v2 import INPUT_GROUPS
from ...senses.yaw_drive import YawDrive, YawDriveState
from ...senses.yaw_vision import YawVision, YawVisionState
from .recompute import advance_frame

PARAMETER_NAMES = ("brain.theta", "readout.weight", "readout.bias")
SOURCE_FILES = ("learning/yaw/policy.py", "learning/yaw/recompute.py", "senses/yaw_vision.py", "senses/yaw_drive.py",
                "senses/vision_v2.py", "senses/encoding_v2.py", "brain/policy_groups.py", "brain/metadata.py")


@dataclass(frozen=True)
class YawPolicyState:
    brain: LIFState
    drive: YawDriveState
    vision: YawVisionState
    frame_index: int
    baseline_sum: torch.Tensor
    baseline: torch.Tensor
    baseline_count: int
    policy_id: str

    def detach(self):
        return YawPolicyState(self.brain.detach(), self.drive, self.vision, self.frame_index,
                              self.baseline_sum.detach(), self.baseline.detach(), self.baseline_count, self.policy_id)


@dataclass(frozen=True)
class YawPolicyOutput:
    logit: torch.Tensor
    action: torch.Tensor
    features_mv: torch.Tensor
    group_rates_hz: torch.Tensor
    network_spikes: torch.Tensor
    state: YawPolicyState
    visual_features: torch.Tensor


class YawVisualPolicy(nn.Module):
    frame_steps = 500
    warmup_frames = 40

    @classmethod
    def from_reference(cls, path):
        graph = load_training_graph(path)
        if graph.identity["role"] != "full_reference" or graph.n != 166700 or graph.counts.nnz != 25582938:
            raise ValueError("Yaw production policy requires the registered complete MaleCNS reference")
        mapping = load_policy_neuron_map(graph, path)
        recorded = graph.identity["reference_provenance"]["group_mapping"]["body_ids"]
        groups = {}
        for name, ids in recorded.items():
            body_ids = np.asarray(ids, dtype=np.int64)
            if not len(body_ids) or not np.isin(body_ids, graph.body_ids).all():
                raise ValueError("Empty or foreign reference display/input group")
            groups[name] = np.searchsorted(graph.body_ids, body_ids)
        input_indices = np.sort(np.concatenate([groups[name] for name in INPUT_GROUPS]))
        if len(np.unique(input_indices)) != len(input_indices):
            raise ValueError("Visual input groups overlap")
        brain = MPSLIF(graph, graph.body_ids[input_indices], config=TrainableLIFConfig())
        return cls._build(brain, groups, mapping.outputs.indices)

    @classmethod
    def for_numerical_test(cls, brain, groups, output_indices):
        if brain.graph.identity["role"] != "numerical_test_only":
            raise ValueError("Explicit numerical-test-only brain required")
        return cls._build(brain, groups, output_indices)

    @classmethod
    def _build(cls, brain, groups, output_indices):
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        if type(brain) is not MPSLIF or brain.config != TrainableLIFConfig():
            raise ValueError("Unmodified MPS LIF and registered 0.1ms dynamics required")
        instance.brain = brain
        inputs = np.sort(np.concatenate([np.asarray(groups[name], dtype=np.int64) for name in INPUT_GROUPS]))
        channels = np.empty(len(inputs), dtype=np.int64)
        for index, name in enumerate(INPUT_GROUPS):
            channels[np.searchsorted(inputs, groups[name])] = index
        if tuple(int(v) for v in brain.graph.body_ids[inputs]) != brain.input_body_ids:
            raise ValueError("Brain input IDs differ from group drive")
        outputs = np.asarray(output_indices, dtype=np.int64)
        if outputs.ndim != 1 or not len(outputs) or np.any(outputs[1:] <= outputs[:-1]) or np.any(outputs < 0) or np.any(outputs >= brain.graph.n):
            raise ValueError("Invalid complete descending-neuron selection")
        instance.drive, instance.vision = YawDrive(brain.graph.n, inputs, channels), YawVision()
        instance.register_buffer("dn_indices", torch.tensor(outputs.copy(), device="mps"), persistent=False)
        instance.group_names = tuple(sorted(groups))
        instance.groups = {name: torch.tensor(np.asarray(groups[name], dtype=np.int64), device="mps") for name in instance.group_names}
        width = 2 * len(outputs)
        # Small nonzero deterministic weights retain an internal gradient on update 1.
        rng = np.random.default_rng(1703)
        weights = rng.normal(0., .01 / np.sqrt(width), (1, width)).astype(np.float32)
        with torch.random.fork_rng(devices=[]):
            instance.readout = nn.Linear(width, 1, device="cpu").to("mps")
        with torch.no_grad():
            instance.readout.weight.copy_(torch.tensor(weights, device="mps"))
            instance.readout.bias.zero_()
        package = Path(__file__).parents[2]
        identity = {"schema": "malecns_lab.yaw_visual_policy.v1", "role": brain.graph.identity["role"],
                    "graph": brain.graph.identity, "model": brain.identity, "input_mapping": instance.drive.identity,
                    "vision": instance.vision.identity, "output_body_ids": brain.graph.body_ids[outputs].tolist(),
                    "group_names": list(instance.group_names),
                    "group_body_ids": {name: brain.graph.body_ids[np.asarray(groups[name], dtype=np.int64)].tolist()
                                       for name in instance.group_names},
                    "config": {"frame_ms": 50, "dt_ms": .1, "frame_steps": 500, "block_steps": 50,
                               "credit_steps": 100, "warmup_frames": 40, "baseline_last_frames": 20,
                               "readout_seed": 1703, "readout_initialization": "PCG64 normal(0,.01/sqrt(2*DN)); bias0",
                               "features": "DN first/second half mean membrane minus frozen last20warmup mean; /7mV",
                               "action": ".35*tanh(logit); only FLU wz rad/s"},
                    "source_sha256": {name: file_hashes(package / name)["sha256"] for name in SOURCE_FILES}}
        instance.policy_id = canonical_hash(identity)
        instance._identity = {**identity, "policy_id": instance.policy_id}
        return instance

    @property
    def identity(self):
        return deepcopy(self._identity)

    def validate_parameters(self):
        named = dict(self.named_parameters())
        if set(named) != set(PARAMETER_NAMES):
            raise ValueError("Unexpected yaw policy parameter inventory")
        if any(p.device.type != "mps" or p.dtype != torch.float32 or not torch.isfinite(p).all() for p in named.values()):
            raise ValueError("Yaw parameters must remain finite MPS float32")
        if not self.readout.weight.requires_grad or not self.readout.bias.requires_grad:
            raise ValueError("Readout must remain trainable")
        if not self.brain.theta.requires_grad and torch.count_nonzero(self.brain.theta):
            raise ValueError("Frozen brain must retain initial theta")
        if self.brain.rho.requires_grad or torch.count_nonzero(self.brain.rho):
            raise ValueError("Fixed background changed")

    def validate_state(self, state):
        if (not isinstance(state, YawPolicyState) or state.policy_id != self.policy_id
                or type(state.frame_index) is not int or state.frame_index < 0):
            raise ValueError("Foreign yaw policy state or invalid frame clock")
        if state.brain.step != state.frame_index * 500 or state.drive.step != state.brain.step or state.vision.frame != state.frame_index:
            raise ValueError("Yaw neural/sensory clocks disagree")
        self.brain._validate_state(state.brain)
        self.drive.validate(state.drive)
        self.vision.validate(state.vision)
        expected_count = min(20, max(0, state.frame_index - 20))
        if type(state.baseline_count) is not int or state.baseline_count != expected_count:
            raise ValueError("Invalid baseline accumulation count")
        for value in (state.baseline, state.baseline_sum):
            if (value.shape != (2 * len(self.dn_indices),) or value.dtype != torch.float32 or value.device.type != "mps"
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError("Invalid frozen neural baseline")
        if state.frame_index < 40 and torch.count_nonzero(state.baseline):
            raise ValueError("Baseline may only freeze after 40 warmup frames")
        if state.frame_index <= 20 and torch.count_nonzero(state.baseline_sum):
            raise ValueError("Baseline accumulation may only start on warmup frame 20")
        if state.frame_index >= 40 and not torch.equal(state.baseline, state.baseline_sum / 20):
            raise ValueError("Frozen baseline differs from warmup accumulation")

    @torch.no_grad()
    def initial_state(self):
        self.validate_parameters()
        zero = torch.zeros(2 * len(self.dn_indices), device="mps")
        return YawPolicyState(self.brain.reset(), self.drive.initial_state(), self.vision.initial_state(),
                              0, zero, zero.clone(), 0, self.policy_id)

    def forward_tensor(self, image_bgr, state, *, frame_offset=0, direct_numerical_fixture=False):
        if type(frame_offset) is not int or frame_offset not in (0, 500):
            raise ValueError("Expected first/second frame credit offset")
        self.validate_parameters()
        self.validate_state(state)
        visual, vision = self.vision.encode(image_bgr, state.vision)
        rates = self.drive.rates(visual)
        brain, drive, features, counts = advance_frame(self, rates, state.brain, state.drive,
                                                       frame_offset=frame_offset, direct=direct_numerical_fixture)
        total, baseline, count = state.baseline_sum, state.baseline, state.baseline_count
        if 20 <= state.frame_index < 40:
            total = total + features.detach()
            count += 1
            if count == 20:
                baseline = total / 20
        logit = self.readout((features - baseline) / 7.).squeeze()
        action = .35 * torch.tanh(logit) if state.frame_index >= 40 else logit * 0.
        group_rates = torch.stack([counts[idx].to(torch.float32).mean() * 20 for idx in self.groups.values()])
        next_state = YawPolicyState(brain, drive, vision, state.frame_index + 1, total, baseline, count, self.policy_id)
        return YawPolicyOutput(logit, action, features, group_rates, counts.sum(), next_state, visual)

    def parameters_numpy(self):
        self.validate_parameters()
        return {name: value.detach().cpu().numpy().copy() for name, value in self.named_parameters()}

    @torch.no_grad()
    def load_parameters(self, values):
        named = dict(self.named_parameters())
        if not isinstance(values, dict) or set(values) != set(PARAMETER_NAMES):
            raise ValueError("Yaw parameter names differ; old three-axis checkpoints are unsupported")
        for name, parameter in named.items():
            value = values[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != tuple(parameter.shape) or not np.isfinite(value).all():
                raise ValueError("Yaw parameter shape/dtype/value differs")
        if not self.brain.theta.requires_grad and np.count_nonzero(values["brain.theta"]):
            raise ValueError("Cannot restore trained theta into frozen control")
        for name, parameter in named.items():
            parameter.copy_(torch.from_numpy(values[name].copy()).to("mps"))
        self.validate_parameters()

    def state_arrays(self, state):
        self.validate_state(state)
        tensors = {"v": state.brain.v, "g": state.brain.g, "refractory": state.brain.refractory_until,
                   "phase": state.drive.phase, "event_counts": state.drive.event_counts,
                   "vision_gray": state.vision.gray, "vision_ema": state.vision.expansion_ema,
                   "vision_loom": state.vision.previous_loom, "baseline_sum": state.baseline_sum, "baseline": state.baseline,
                   **{f"pending_{i:02}": v for i, v in enumerate(state.brain.pending)}}
        return {**{name: value.detach().cpu().numpy().copy() for name, value in tensors.items()},
                **{name: np.array(value, dtype=np.int64) for name, value in
                   {"brain_step": state.brain.step, "drive_step": state.drive.step, "vision_frame": state.vision.frame,
                    "frame_index": state.frame_index, "baseline_count": state.baseline_count}.items()},
                "policy_id": np.array(self.policy_id)}

    def restore_state(self, arrays):
        n, width, inputs = self.brain.graph.n, 2 * len(self.dn_indices), len(self.drive.input_indices)
        shapes = {"v": (n,), "g": (n,), "refractory": (n,), "phase": (inputs,), "event_counts": (inputs,),
                  "vision_gray": (72, 96), "vision_ema": (2,), "vision_loom": (2,),
                  "baseline_sum": (width,), "baseline": (width,),
                  **{f"pending_{i:02}": (n,) for i in range(self.brain.delay_steps)}}
        integers = {"brain_step", "drive_step", "vision_frame", "frame_index", "baseline_count"}
        if not isinstance(arrays, dict) or set(arrays) != {*shapes, *integers, "policy_id"}:
            raise ValueError("Unexpected yaw state archive fields")
        for name in integers:
            if not isinstance(arrays[name], np.ndarray) or arrays[name].dtype != np.int64 or arrays[name].shape != ():
                raise ValueError("Invalid integer yaw archive field")
        identifier = arrays["policy_id"]
        if not isinstance(identifier, np.ndarray) or identifier.shape != () or identifier.dtype.kind != "U" or str(identifier) != self.policy_id:
            raise ValueError("Yaw state belongs to a different policy")
        tensors = {}
        for name, shape in shapes.items():
            value = arrays[name]
            dtype = np.int64 if name in ("refractory", "event_counts") else np.float32
            if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid yaw archive tensor")
            tensors[name] = torch.from_numpy(value.copy()).to("mps")
        brain = LIFState(tensors["v"], tensors["g"], tensors["refractory"],
                         tuple(tensors[f"pending_{i:02}"] for i in range(self.brain.delay_steps)),
                         int(arrays["brain_step"]), self.brain.model_id)
        state = YawPolicyState(brain, YawDriveState(tensors["phase"], tensors["event_counts"], int(arrays["drive_step"])),
                               YawVisionState(tensors["vision_gray"], tensors["vision_ema"], tensors["vision_loom"], int(arrays["vision_frame"])),
                               int(arrays["frame_index"]), tensors["baseline_sum"], tensors["baseline"],
                               int(arrays["baseline_count"]), self.policy_id)
        self.validate_state(state)
        return state
