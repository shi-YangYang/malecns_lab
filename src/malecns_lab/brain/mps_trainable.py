"""MPS float32 LIF; complete fixed COO forward/adjoint and all-cell gradients.

The event equations are those of trainable.py. This is a distinct numerical
backend, not a claim of bitwise CPU equivalence or measured fly physiology.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import os

import numpy as np
import torch
from torch import nn

from .mps_identity import model_identity
from .tonic import LAMINA_TYPES
from .trainable import LIFState, TrainableLIFConfig, SHIU_REFERENCE, surrogate_spike
from .trainable_graph import numerical_test_graph


def require_mps():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise ValueError("An available MPS device and PYTORCH_ENABLE_MPS_FALLBACK=0 are required")


class _Matvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, adjoint, vector):
        ctx.save_for_backward(adjoint)
        return torch.sparse.mm(matrix, vector[:, None])[:, 0]

    @staticmethod
    def backward(ctx, gradient):
        (adjoint,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(adjoint, gradient[:, None])[:, 0]


def fixed_mps_matvec(matrix, adjoint, vector):
    return _Matvec.apply(matrix, adjoint, vector)


def scale_spike_backward(spikes, gain):
    """Keep the binary forward exactly; explicitly rescale its surrogate only."""
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("Spike backward gain must be finite and in (0, 1]")
    if gain == 1.0:
        return spikes
    detached = spikes.detach()
    return detached + gain * (spikes - detached)


def checked_coo(matrix, weight):
    """Verify every ordered coordinate/value on the actual device, in chunks."""
    if not matrix.has_canonical_format:
        raise ValueError("COO cache requires canonical unique edges")
    csr = matrix.tocsr()
    compressed = torch.sparse_csr_tensor(torch.tensor(csr.indptr, dtype=torch.int64),
                                         torch.tensor(csr.indices, dtype=torch.int64),
                                         torch.tensor(csr.data, dtype=torch.float32) * weight,
                                         size=csr.shape, check_invariants=True)
    del csr
    # Canonical CSR already orders unique edges: do not allocate another full
    # coordinate sort/aggregation workspace by constructing uncoalesced COO.
    cpu = compressed.to_sparse_coo()
    del compressed
    if not cpu.is_coalesced() or cpu._nnz() != matrix.nnz:
        raise ValueError("CSR to COO changed canonical edge count/order")
    device = cpu.to("mps")
    if not device.is_coalesced() or device.shape != cpu.shape or device._nnz() != cpu._nnz():
        raise ValueError("COO cache changed shape/order")
    for start in range(0, cpu._nnz(), 262144):
        sl = slice(start, start + 262144)
        if (not torch.equal(device.indices()[:, sl].cpu(), cpu.indices()[:, sl])
                or not torch.equal(device.values()[sl].cpu(), cpu.values()[sl])):
            raise ValueError("COO cache lost coordinates/values")
    return device


class MPSLIF(nn.Module):
    def __init__(self, graph, input_body_ids, *, config=None, tonic_types=LAMINA_TYPES):
        super().__init__()
        require_mps()
        role = graph.identity.get("role")
        if role not in ("full_reference", "numerical_test_only"):
            raise ValueError("Verified full reference or explicit numerical fixture required")
        if (not tonic_types or tuple(tonic_types) != tuple(n for n in LAMINA_TYPES if n in tonic_types)
                or role == "full_reference" and tuple(tonic_types) != LAMINA_TYPES):
            raise ValueError("All exact lamina groups are required in production")
        self.graph, self.config, self._dtype = graph, config or TrainableLIFConfig(), torch.float32
        ids = np.asarray(tuple(input_body_ids))
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Input IDs must be integers")
        ids = np.sort(ids.astype(np.int64))
        if len(np.unique(ids)) != len(ids) or not np.isin(ids, graph.body_ids).all():
            raise ValueError("Input IDs must be unique graph members")
        self.input_body_ids = tuple(int(v) for v in ids)
        self.delay_steps = self.config.integer_steps(self.config.delay_ms)
        self.refractory_steps = self.config.integer_steps(self.config.refractory_ms)
        mapping = np.zeros(graph.n, np.int64)
        members = []
        for index, name in enumerate(tonic_types, 1):
            if name not in graph.group_names:
                raise ValueError(f"Missing exact tonic group {name}")
            selected = graph.group_index == graph.group_names.index(name)
            members.append({"type": name, "count": int(selected.sum()),
                            "body_ids_sha256": hashlib.sha256(np.asarray(graph.body_ids[selected], dtype="<i8").tobytes()).hexdigest()})
            if not selected.any() or np.isin(graph.body_ids[selected], ids).any():
                raise ValueError("Tonic group is empty or overlaps input")
            mapping[selected] = index
        self.theta = nn.Parameter(torch.zeros(len(graph.group_names), device="mps"))
        self.tonic_mode, self.tonic_types = "fixed", tuple(tonic_types)
        self.register_buffer("rho", torch.zeros(len(tonic_types), device="mps"))
        self.register_buffer("group_index", torch.tensor(graph.group_index.copy(), device="mps"), persistent=False)
        self.register_buffer("input_mask", torch.tensor(np.isin(graph.body_ids, ids), device="mps"), persistent=False)
        self.register_buffer("tonic_group_index", torch.tensor(mapping, device="mps"), persistent=False)
        self.register_buffer("base_weights", checked_coo(graph.counts, self.config.weight_mv), persistent=False)
        self.register_buffer("base_weights_adjoint", checked_coo(graph.counts.T, self.config.weight_mv), persistent=False)
        identity = model_identity(graph.identity, asdict(self.config), self.input_body_ids,
            {"mode": "fixed", "types": list(tonic_types), "members": members,
             "member_count": int(np.count_nonzero(mapping)),
             "mapping_sha256": hashlib.sha256(np.asarray(mapping, dtype="<i8").tobytes()).hexdigest(),
             "initial_offset_mv": 8}, SHIU_REFERENCE)
        self.model_id = identity.pop("model_id")
        self._identity = identity

    @classmethod
    def for_numerical_test(cls, counts, input_indices, *, types, tonic_types, config=None):
        return cls(numerical_test_graph(counts, types), input_indices, config=config, tonic_types=tonic_types)

    @property
    def identity(self):
        return {**deepcopy(self._identity), "model_id": self.model_id}

    @property
    def gains(self):
        # Constants match float32 nextafter bounds without invoking a device-dependent nextafter.
        return (1 + .5 * torch.tanh(self.theta)).clamp(float(np.nextafter(np.float32(.5), np.float32(1))),
                                                     float(np.nextafter(np.float32(1.5), np.float32(1))))

    def reset(self):
        zero = torch.zeros(self.graph.n, device="mps")
        return LIFState(torch.full_like(zero, self.config.v_rest_mv), zero,
                        torch.zeros(self.graph.n, dtype=torch.int64, device="mps"),
                        tuple(torch.zeros_like(zero) for _ in range(self.delay_steps)), 0, self.model_id)

    def _validate_state(self, state):
        if (not isinstance(state, LIFState) or state.model_id != self.model_id
                or type(state.step) is not int or state.step < 0 or len(state.pending) != self.delay_steps):
            raise ValueError("State identity/clock/delay differs")
        for value in (state.v, state.g, *state.pending):
            if (value.shape != (self.graph.n,) or value.dtype != torch.float32 or value.device.type != "mps"
                    or not torch.isfinite(value).all()):
                raise ValueError("State must be finite MPS float32 with full node shape")
        value = state.refractory_until
        if (value.shape != (self.graph.n,) or value.dtype != torch.int64 or value.device.type != "mps" or (value < 0).any()):
            raise ValueError("Invalid refractory state")

    def step(self, voltage_increment, state, *, spike_backward_gain=1.0):
        self._validate_state(state)
        if (self.theta.dtype != torch.float32 or self.theta.device.type != "mps" or not torch.isfinite(self.theta).all()
                or self.rho.requires_grad or self.rho.device.type != "mps" or self.rho.dtype != torch.float32
                or self.rho.shape != (len(self.tonic_types),) or torch.count_nonzero(self.rho)):
            raise ValueError("Model parameters/background changed")
        if (voltage_increment.shape != (self.graph.n,) or voltage_increment.dtype != torch.float32
                or voltage_increment.device.type != "mps" or not torch.isfinite(voltage_increment).all()
                or torch.count_nonzero(voltage_increment[~self.input_mask])):
            raise ValueError("Invalid registered voltage input")
        cfg = self.config
        active = (state.step >= state.refractory_until) | self.input_mask
        arrival_g = state.g + state.pending[0]
        a, b = math.exp(-cfg.dt_ms / cfg.tau_m_ms), math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
        coupling = cfg.dt_ms / cfg.tau_m_ms * a if cfg.tau_m_ms == cfg.tau_syn_ms else cfg.tau_syn_ms / (cfg.tau_m_ms - cfg.tau_syn_ms) * (a - b)
        integrated_v = cfg.v_rest_mv + (state.v - cfg.v_rest_mv) * a + coupling * arrival_g
        offsets = 16 * torch.sigmoid(self.rho)
        equilibrium = torch.cat((offsets.new_zeros(1), offsets))[self.tonic_group_index]
        integrated_v = integrated_v + (1 - a) * equilibrium
        v = torch.where(active, integrated_v, state.v) + voltage_increment * self.input_mask
        g = torch.where(active, arrival_g * b, arrival_g)
        spikes = surrogate_spike(v - cfg.v_threshold_mv, cfg.surrogate_scale_mv) * active
        # The historical/default branch has no added tensor operation. Callers
        # selecting a new estimator must bind this gain in their policy identity.
        if spike_backward_gain != 1.0:
            spikes = scale_spike_backward(spikes, spike_backward_gain)
        fired = spikes.detach().bool()
        next_v, next_g = torch.where(fired, cfg.v_reset_mv, v), torch.where(fired, 0., g)
        refractory = torch.where(fired & ~self.input_mask, state.step + self.refractory_steps, state.refractory_until)
        transmitted = fixed_mps_matvec(self.base_weights, self.base_weights_adjoint, self.gains[self.group_index] * spikes)
        return spikes, LIFState(next_v, next_g, refractory, state.pending[1:] + (transmitted,), state.step + 1, self.model_id)
