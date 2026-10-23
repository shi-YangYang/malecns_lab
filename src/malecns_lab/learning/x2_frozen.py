"""Separately identified, no-grad execution of a verified historical GPUPolicy.

This wrapper neither loads a graph nor supplies a numerical-fixture fallback.
The caller authenticates the historical source/checkpoint and owns resource
supervision. State/output classes and sparse/spike operators come from that
policy's loaded source. Full-graph equivalence is an external acceptance gate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path

import torch

BLOCK_STEPS = 50


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _StickyError:
    """No host scalar extraction until check(); errors survive later recovery."""

    def __init__(self, device):
        self.flag = torch.zeros((), dtype=torch.int32, device=device)

    def add(self, condition, bit):
        self.flag = self.flag | (condition.any().to(torch.int32) * bit)

    def finite(self, *values):
        for value in values:
            self.add(~torch.isfinite(value), 1)

    def check(self, step):
        bits = int(self.flag.item())
        if bits:
            raise ValueError(f"Frozen inference numerical fault at/before step {step}: sticky_bits={bits}")


class FrozenInference:
    """Immutable-parameter wrapper; requires the already restored full policy.

    ``execution_identity`` identifies this execution separately from the old
    policy_id retained in compatible recurrent states. checkpoint_id is an
    authenticated caller-supplied reference, not a claim of checkpoint loading.
    """

    @torch.no_grad()
    def __init__(self, policy, *, checkpoint_id=None, telemetry=None):
        _require(type(policy).__module__ == "cnsbridge.learning.gpu_policy",
                 "A historical cnsbridge GPUPolicy is required")
        module = importlib.import_module(type(policy).__module__)
        _require(type(policy) is module.GPUPolicy and type(policy.brain) is module.MPSLIF
                 and type(policy.drive) is module.GPUDrive, "Historical policy/model/drive types differ")
        _require(policy.brain.graph.identity.get("role") == "full_reference",
                 "Frozen inference requires the complete reference; no fixture fallback")
        _require(checkpoint_id is None or (isinstance(checkpoint_id, str) and len(checkpoint_id) == 64
                 and all(c in "0123456789abcdef" for c in checkpoint_id)), "Invalid checkpoint_id reference")
        self.policy = policy
        self.telemetry = telemetry
        self._policy_state, self._output = module.PolicyState, module.GPUOutput
        self._drive_state = module.GPUDriveState
        brain_module = importlib.import_module(type(policy.brain).__module__)
        self._brain_state = brain_module.LIFState
        self._matvec, self._spike = brain_module.fixed_mps_matvec, brain_module.surrogate_spike
        self._mask_shape = module.MASK_SHAPE
        self._configs = self._configuration()
        self._validate_configuration()
        policy._validate_parameters()
        self._parameter_hash = self._parameters_digest()
        self._tensor_bindings = self._bindings()
        self._objects = (policy.brain, policy.drive, policy.mapping, policy.spatial_map,
                         policy.brain.graph, policy.readout)
        self._ids = (policy.policy_id, policy.brain.model_id, policy.drive.drive_id,
                     policy.drive.mapping.map_id)
        source_modules = (module, brain_module, importlib.import_module(type(policy.drive).__module__),
                          importlib.import_module("cnsbridge.learning.gpu_recompute"))
        identity = {
            "schema": "malecns_lab.x2_frozen_execution.v1",
            "backend": "historical_mps_equations_frozen_sticky50",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_policy_id": policy.policy_id,
            "source_policy_identity_sha256": _digest(policy.identity),
            "source_model_id": policy.brain.model_id, "source_drive_id": policy.drive.drive_id,
            "source_files_sha256": {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                    for m in source_modules},
            "checkpoint_id": checkpoint_id, "parameters_sha256": self._parameter_hash,
            "graph": deepcopy(policy.brain.graph.identity),
            "configuration": deepcopy(self._configs), "torch_version": torch.__version__,
            "checks": "GPU sticky at every step; host every 50 steps and final remainder",
            "scope": "no_grad; unchanged requires_grad; no optimizer; same sparse/spike operators",
            "state_contract": "source policy_id retained for state compatibility; execution_id is separate",
        }
        self._identity = {**identity, "execution_id": _digest(identity)}

    @property
    def execution_identity(self):
        return deepcopy(self._identity)

    def _configuration(self):
        p = self.policy
        return {"brain": asdict(p.brain.config), "drive": asdict(p.drive.config),
                "policy": asdict(p.config), "camera": asdict(p.drive.camera),
                "frame_steps": p.frame_steps, "precondition_steps": p.precondition_steps,
                "credit_horizon": p.credit_horizon, "delay_steps": p.brain.delay_steps,
                "refractory_steps": p.brain.refractory_steps}

    def _validate_configuration(self):
        p, b, d = self.policy, self.policy.brain, self.policy.drive
        for config in (b.config, d.config, p.config, d.camera):
            type(config)(**asdict(config))
        _require(p.config.steps(b.config.dt_ms) == (p.frame_steps, p.precondition_steps)
                 and d.config.dt_ms == b.config.dt_ms and d.config == p.drive_config
                 and d.camera == p.camera and p.credit_horizon in (100, 250, 500, 1000)
                 and b.delay_steps == b.config.integer_steps(b.config.delay_ms)
                 and b.refractory_steps == b.config.integer_steps(b.config.refractory_ms),
                 "Frozen configuration/clocks differ")
        graph = b.graph.identity
        _require(graph["role"] == "full_reference" and graph["nodes"] == b.graph.n == d.mapping.n_nodes
                 and graph["edges"] == b.graph.counts.nnz == b.base_weights._nnz() == b.base_weights_adjoint._nnz()
                 and tuple(b.base_weights.shape) == tuple(b.base_weights_adjoint.shape) == (b.graph.n, b.graph.n)
                 and d.mapping.map_id == p.spatial_map.map_id, "Full graph or sensory map differs")
        _require(b.rho.dtype == torch.float32 and b.rho.device.type == "mps"
                 and b.rho.shape == (len(b.tonic_types),) and not b.rho.requires_grad
                 and not torch.count_nonzero(b.rho), "Fixed background changed")

    def _parameters_digest(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.policy.named_parameters()):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def _bindings(self):
        p = self.policy
        tensors = dict(p.named_buffers()) | dict(p.named_parameters())
        # GPUDrive is not an nn.Module; include its immutable device maps too.
        tensors.update({f"drive.{name}": getattr(p.drive, name) for name in
                        ("input_indices", "pixel_cells", "image_cells", "cell_sizes")})
        return {name: (id(t), t._version, tuple(t.shape), t.dtype, t.device, t.requires_grad)
                for name, t in tensors.items()}

    def _boundary(self):
        p = self.policy
        _require(all(a is b for a, b in zip(self._objects,
                 (p.brain, p.drive, p.mapping, p.spatial_map, p.brain.graph, p.readout))),
                 "Frozen policy objects changed")
        _require(self._ids == (p.policy_id, p.brain.model_id, p.drive.drive_id, p.drive.mapping.map_id)
                 and self._configs == self._configuration(), "Frozen identity/configuration changed")
        _require(self._tensor_bindings == self._bindings(), "Frozen tensors changed")
        p._validate_parameters()
        _require(self._parameter_hash == self._parameters_digest(), "Frozen parameters changed")

    def _constants(self, mask):
        b, d = self.policy.brain, self.policy.drive
        _require(isinstance(mask, torch.Tensor) and mask.device.type == "mps" and mask.dtype == torch.float32
                 and tuple(mask.shape) == self._mask_shape and torch.isfinite(mask).all()
                 and not (mask < 0).any() and not (mask > 1).any(), "Invalid registered occupancy mask")
        cfg = b.config
        a, decay = math.exp(-cfg.dt_ms / cfg.tau_m_ms), math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
        coupling = (cfg.dt_ms / cfg.tau_m_ms * a if cfg.tau_m_ms == cfg.tau_syn_ms else
                    cfg.tau_syn_ms / (cfg.tau_m_ms - cfg.tau_syn_ms) * (a - decay))
        offsets = 16 * torch.sigmoid(b.rho)
        equilibrium = torch.cat((offsets.new_zeros(1), offsets))[b.tonic_group_index]
        # Keep the original parentheses and device operations for all constants.
        phase_increment = (mask.flatten()[d.pixel_cells] * d.config.max_rate_hz) * (d.config.dt_ms / 1000)
        return a, decay, coupling, (1 - a) * equilibrium, b.gains[b.group_index], phase_increment

    def _step(self, state, sensory, constants, errors):
        b, d, cfg = self.policy.brain, self.policy.drive, self.policy.brain.config
        a, decay, coupling, background, gains, phase_increment = constants
        phase = sensory.phase + phase_increment
        fired_input = phase >= 1
        residual = (phase - fired_input.to(torch.float32)).clamp_max(0.9999999403953552)
        counts = sensory.event_counts + fired_input.to(torch.int64)
        increment = torch.zeros(d.mapping.n_nodes, dtype=torch.float32, device=sensory.phase.device)
        increment.index_copy_(0, d.input_indices, fired_input.to(torch.float32) * d.config.event_voltage_mv)
        errors.finite(phase, residual, increment)
        errors.add((phase < 0) | (phase > 2) | (residual < 0) | (residual >= 1), 2)
        errors.add((counts < 0) | (counts > sensory.step + 1), 4)
        next_sensory = self._drive_state(residual, counts, sensory.step + 1, d.drive_id, d.mapping.map_id)

        active = (state.step >= state.refractory_until) | b.input_mask
        arrival_g = state.g + state.pending[0]
        integrated_v = cfg.v_rest_mv + (state.v - cfg.v_rest_mv) * a + coupling * arrival_g
        errors.finite(arrival_g, integrated_v)
        integrated_v = integrated_v + background
        v = torch.where(active, integrated_v, state.v) + increment * b.input_mask
        g = torch.where(active, arrival_g * decay, arrival_g)
        distance = v - cfg.v_threshold_mv
        spikes = self._spike(distance, cfg.surrogate_scale_mv) * active
        fired = spikes.detach().bool()
        next_v, next_g = torch.where(fired, cfg.v_reset_mv, v), torch.where(fired, 0., g)
        refractory = torch.where(fired & ~b.input_mask, state.step + b.refractory_steps, state.refractory_until)
        transmitted = self._matvec(b.base_weights, b.base_weights_adjoint, gains * spikes)
        # Inspect pre-reset values too: a later finite reset cannot erase a fault.
        errors.finite(integrated_v, v, g, distance, spikes, transmitted)
        errors.add(refractory < 0, 8)
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.capture(spikes, state.step)
        return (self._brain_state(next_v, next_g, refractory, state.pending[1:] + (transmitted,),
                                  state.step + 1, b.model_id), next_sensory)

    def _run(self, brain, sensory, mask, count, *, sums=None, frame_offset=0):
        _require(type(count) is int and count >= 0
                 and max(brain.step, sensory.step) + count + self.policy.brain.refractory_steps < 2**63,
                 "Frozen clock would overflow")
        constants = self._constants(mask)
        errors = _StickyError(brain.v.device)
        errors.finite(*constants[3:])
        for offset in range(count):
            brain, sensory = self._step(brain, sensory, constants, errors)
            if sums is not None:
                index = offset // (self.policy.frame_steps // 2)
                sums[index] = sums[index] + brain.v[self.policy.dn_indices]
                errors.finite(sums[index])
            completed = offset + 1
            if completed % BLOCK_STEPS == 0 or completed == count:
                errors.check(brain.step)
                telemetry = getattr(self, "telemetry", None)
                if telemetry is not None:
                    telemetry.flush(brain.step)
                credit = frame_offset + completed
                if sums is not None and credit % self.policy.credit_horizon == 0 and credit < 2 * self.policy.frame_steps:
                    brain = brain.detach()
        if count == 0:
            errors.check(brain.step)
        return brain, sensory, sums

    @torch.no_grad()
    def initial_state(self):
        self._boundary()
        p = self.policy
        brain, sensory = p.brain.reset(), p.drive.initial_state()
        p.brain._validate_state(brain)
        p.drive.validate(sensory)
        blank = torch.zeros(self._mask_shape, dtype=torch.float32, device="mps")
        brain, sensory, _ = self._run(brain, sensory, blank, p.precondition_steps)
        state = self._policy_state(brain, sensory, 0, p.policy_id)
        p.validate_state(state)
        self._boundary()
        return state

    @torch.no_grad()
    def forward_tensor(self, frame, state, *, frame_offset=0):
        p = self.policy
        _require(type(frame_offset) is int and frame_offset in (0, p.frame_steps), "Invalid two-frame credit offset")
        self._boundary()
        p.validate_state(state)
        mask = p.drive.mask(frame)
        sums = [state.brain.v.new_zeros(len(p.mapping.outputs.indices)) for _ in range(2)]
        brain, sensory, sums = self._run(state.brain, state.drive, mask, p.frame_steps,
                                         sums=sums, frame_offset=frame_offset)
        features = torch.cat(sums) / (p.frame_steps // 2)
        cfg = p.brain.config
        normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
        logits = p.readout(normalized)
        action = torch.tanh(logits) * p.action_scales
        errors = _StickyError(brain.v.device)
        errors.finite(features, normalized, logits, action)
        errors.check(brain.step)
        next_state = self._policy_state(brain, sensory, state.frame_index + 1, p.policy_id)
        p.validate_state(next_state)
        self._boundary()
        return self._output(logits, action, features, next_state)
