"""CPU full-graph LIF with bounded internal gains and surrogate gradients.

This is an explicitly ordered Shiu-equation-inspired model, not a bitwise
replication of Brian2. See docs/TRAINABLE_SNN.md for the numerical contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .metadata import canonical_hash, file_hashes
from .trainable_graph import TrainingGraph, load_training_graph, numerical_test_graph

MODEL_SCHEMA = "cnsbridge.trainable_lif.v1"
SHIU_REFERENCE = {
    "url": "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/model.py",
    "retrieved": "2026-10-17",
    "sha256": "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e",
    "relationship": "equations and parameter source; explicit CNSBridge event order, not Brian2 replication",
}


@dataclass(frozen=True)
class TrainableLIFConfig:
    dt_ms: float = 0.1
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    weight_mv: float = 0.275
    surrogate_scale_mv: float = 1.0

    def __post_init__(self):
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("All LIF parameters must be finite numbers")
        for name in ("dt_ms", "tau_m_ms", "tau_syn_ms", "refractory_ms", "delay_ms", "weight_mv", "surrogate_scale_mv"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.v_reset_mv >= self.v_threshold_mv or self.v_rest_mv >= self.v_threshold_mv:
            raise ValueError("Reset and resting voltage must be below threshold")
        self.integer_steps(self.delay_ms)
        self.integer_steps(self.refractory_ms)

    def integer_steps(self, duration_ms: float) -> int:
        ratio = duration_ms / self.dt_ms
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9) or round(ratio) < 1:
            raise ValueError("dt_ms must exactly divide positive delay and refractory durations")
        return round(ratio)


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(distance)
        ctx.scale = scale
        return (distance > 0).to(distance.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (distance,) = ctx.saved_tensors
        derivative = 1.0 / (ctx.scale * (1.0 + (distance / ctx.scale).abs()) ** 2)
        return gradient * derivative, None


def surrogate_spike(distance_mv: Tensor, scale_mv: float = 1.0) -> Tensor:
    """Strict binary forward; declared fast-sigmoid backward, including silence."""
    if not math.isfinite(scale_mv) or scale_mv <= 0:
        raise ValueError("Surrogate scale must be finite and positive")
    return _Spike.apply(distance_mv, scale_mv)


class _FixedSparseMatvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix: Tensor, vector: Tensor) -> Tensor:
        ctx.save_for_backward(matrix)
        return torch.sparse.mm(matrix, vector[:, None])[:, 0]

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (matrix,) = ctx.saved_tensors
        # Every presynaptic component contributes to the adjoint. Restricting
        # this product to actual spike indices would destroy silent-cell gradients.
        return None, torch.sparse.mm(matrix.transpose(0, 1), gradient[:, None])[:, 0]


def fixed_sparse_matvec(matrix: Tensor, vector: Tensor) -> Tensor:
    """Fixed sparse (post, pre) matrix; gradients only for its dense input."""
    return _FixedSparseMatvec.apply(matrix, vector)


@dataclass(frozen=True)
class LIFState:
    v: Tensor
    g: Tensor
    refractory_until: Tensor
    pending: tuple[Tensor, ...]
    step: int
    model_id: str

    def detach(self) -> LIFState:
        """Truncated BPTT boundary; values, delay queue and clock are preserved."""
        return LIFState(self.v.detach(), self.g.detach(), self.refractory_until.detach(),
                        tuple(value.detach() for value in self.pending), self.step, self.model_id)


class TrainableLIF(nn.Module):
    """Single-sequence model. Explicit state avoids hidden carry-over between runs."""

    def __init__(self, graph: TrainingGraph, input_body_ids: Iterable[int], *,
                 config: TrainableLIFConfig | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("The CPU training core supports only float32 or float64")
        if graph.identity.get("role") not in ("full_reference", "numerical_test_only"):
            raise ValueError("Only verified full_reference or explicit numerical test graphs are accepted")
        self.graph, self.config = graph, config or TrainableLIFConfig()
        ids = np.asarray(list(input_body_ids))
        if ids.size and (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("External input body IDs must be integers")
        ids = np.sort(ids.astype(np.int64))
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or not np.isin(ids, graph.body_ids).all():
            raise ValueError("External input body IDs must be unique members of the graph")
        self.input_body_ids = tuple(int(value) for value in ids)
        self.delay_steps = self.config.integer_steps(self.config.delay_ms)
        self.refractory_steps = self.config.integer_steps(self.config.refractory_ms)
        self.theta = nn.Parameter(torch.zeros(len(graph.group_names), dtype=dtype))
        self.register_buffer("group_index", torch.tensor(graph.group_index.copy(), dtype=torch.int64), persistent=False)
        self.register_buffer("input_mask", torch.tensor(np.isin(graph.body_ids, ids)), persistent=False)
        csr = graph.counts.tocsr()
        matrix = torch.sparse_csr_tensor(torch.tensor(csr.indptr, dtype=torch.int64),
                                         torch.tensor(csr.indices, dtype=torch.int64),
                                         torch.tensor(csr.data, dtype=dtype) * self.config.weight_mv,
                                         size=csr.shape, check_invariants=True)
        self.register_buffer("base_weights", matrix, persistent=False)
        self._dtype = dtype
        self._identity = {
            "schema": MODEL_SCHEMA, "graph": graph.identity, "config": asdict(self.config),
            "input_body_ids": list(self.input_body_ids), "dtype": str(dtype), "backend": "torch_cpu_csr",
            "gain": "1 + 0.5*tanh(theta[type(pre)]); initial theta=0; nextafter guards strict floating bounds",
            "gradient": "fast_sigmoid; reset/refractory gates detached; all-cell sparse adjoint",
            "event_order": "arrivals; exact active integration; explicit voltage input; strict threshold; reset; delayed enqueue",
            "reference": SHIU_REFERENCE,
            "implementation_sha256": {name: file_hashes(Path(__file__).with_name(name))["sha256"]
                                      for name in ("trainable.py", "trainable_graph.py")},
        }
        self.model_id = canonical_hash(self._identity)

    @classmethod
    def from_reference(cls, path: str | Path, input_body_ids: Iterable[int], **kwargs) -> TrainableLIF:
        """Production constructor: verify the complete reference, with no fallback."""
        return cls(load_training_graph(path), input_body_ids, **kwargs)

    @classmethod
    def for_numerical_test(cls, counts, input_indices=(), *, types=None, missing=None, **kwargs) -> TrainableLIF:
        """Small numerical circuit only; identity explicitly excludes production use."""
        return cls(numerical_test_graph(counts, types, missing), input_indices, **kwargs)

    @property
    def identity(self) -> dict:
        from copy import deepcopy
        return {**deepcopy(self._identity), "model_id": self.model_id}

    @property
    def gains(self) -> Tensor:
        low, high = self.theta.new_tensor(0.5), self.theta.new_tensor(1.5)
        one = self.theta.new_tensor(1.0)
        return (1.0 + 0.5 * torch.tanh(self.theta)).clamp(torch.nextafter(low, one), torch.nextafter(high, one))

    def reset(self) -> LIFState:
        """Fresh rest state; does not reset learned parameters."""
        zeros = torch.zeros(self.graph.n, dtype=self._dtype)
        return LIFState(torch.full_like(zeros, self.config.v_rest_mv), zeros,
                        torch.zeros(self.graph.n, dtype=torch.int64),
                        tuple(torch.zeros_like(zeros) for _ in range(self.delay_steps)), 0, self.model_id)

    def _validate_state(self, state: LIFState) -> None:
        if not isinstance(state, LIFState) or state.model_id != self.model_id or type(state.step) is not int or state.step < 0:
            raise ValueError("State identity/clock does not match this model")
        if len(state.pending) != self.delay_steps:
            raise ValueError("State delay queue has the wrong length")
        for value in (state.v, state.g, *state.pending):
            if not isinstance(value, Tensor) or value.shape != (self.graph.n,) or value.dtype != self._dtype or value.device.type != "cpu":
                raise ValueError("State vectors must match the CPU model shape and dtype")
            if not torch.isfinite(value).all():
                raise ValueError("State contains nonfinite values")
        if (not isinstance(state.refractory_until, Tensor) or state.refractory_until.shape != (self.graph.n,)
                or state.refractory_until.dtype != torch.int64
                or state.refractory_until.device.type != "cpu" or (state.refractory_until < 0).any()):
            raise ValueError("Invalid refractory clock")

    def _equilibrium_offset_mv(self) -> Tensor | None:
        """Optional intrinsic equilibrium shift; the original model has none."""
        return None

    def step(self, voltage_increment: Tensor, state: LIFState) -> tuple[Tensor, LIFState]:
        """Advance dt; input is one explicit voltage increment per neuron in mV."""
        self._validate_state(state)
        if self.theta.device.type != "cpu" or self.theta.dtype != self._dtype or not torch.isfinite(self.theta).all():
            raise ValueError("Model parameters must retain their finite CPU configuration")
        if (not isinstance(voltage_increment, Tensor) or voltage_increment.shape != (self.graph.n,) or voltage_increment.dtype != self._dtype
                or voltage_increment.device.type != "cpu" or not torch.isfinite(voltage_increment).all()):
            raise ValueError("Voltage input must be a finite CPU vector matching model shape and dtype")
        if torch.count_nonzero(voltage_increment[~self.input_mask]):
            raise ValueError("Nonzero voltage input outside the registered external input cells")
        equilibrium_offset = self._equilibrium_offset_mv()
        cfg = self.config
        active = (state.step >= state.refractory_until) | self.input_mask
        arrival_g = state.g + state.pending[0]
        a, b = math.exp(-cfg.dt_ms / cfg.tau_m_ms), math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
        if cfg.tau_m_ms == cfg.tau_syn_ms:
            coupling = cfg.dt_ms / cfg.tau_m_ms * a
        else:
            coupling = cfg.tau_syn_ms / (cfg.tau_m_ms - cfg.tau_syn_ms) * (a - b)
        integrated_v = cfg.v_rest_mv + (state.v - cfg.v_rest_mv) * a + coupling * arrival_g
        if equilibrium_offset is not None:
            integrated_v = integrated_v + (1 - a) * equilibrium_offset
        v = torch.where(active, integrated_v, state.v) + voltage_increment * self.input_mask
        g = torch.where(active, arrival_g * b, arrival_g)
        spikes = surrogate_spike(v - cfg.v_threshold_mv, cfg.surrogate_scale_mv) * active
        fired = spikes.detach().bool()
        next_v = torch.where(fired, cfg.v_reset_mv, v)
        next_g = torch.where(fired, 0.0, g)
        refractory = torch.where(fired & ~self.input_mask, state.step + self.refractory_steps, state.refractory_until)
        transmitted = fixed_sparse_matvec(self.base_weights, self.gains[self.group_index] * spikes)
        next_state = LIFState(next_v, next_g, refractory, state.pending[1:] + (transmitted,), state.step + 1, self.model_id)
        return spikes, next_state

    def run(self, voltage_increments: Iterable[Tensor], state: LIFState | None = None) -> tuple[Tensor, LIFState]:
        """Run contiguous steps; pass the returned state to continue another window."""
        state = self.reset() if state is None else state
        values = []
        for increment in voltage_increments:
            spikes, state = self.step(increment, state)
            values.append(spikes)
        return (torch.stack(values) if values else torch.empty((0, self.graph.n), dtype=self._dtype)), state
