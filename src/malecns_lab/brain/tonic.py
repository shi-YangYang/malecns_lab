"""Explicit lamina background hypothesis, not measured fly physiology.

The production variant retains the complete graph and selects every exact
L1--L5 type. Small circuits are available only through the numerical factory.
The gain-only checkpoint API deliberately does not support this dynamics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .metadata import canonical_hash, file_hashes
from .trainable import MODEL_SCHEMA, TrainableLIF, TrainableLIFConfig
from .trainable_graph import TrainingGraph, numerical_test_graph

TONIC_SCHEMA = "cnsbridge.lamina_tonic_lif.v1"
LAMINA_TYPES = ("L1", "L2", "L3", "L4", "L5")


class LaminaTonicLIF(TrainableLIF):
    """Optional fixed/learnable equilibrium offsets; original LIF is zero control."""

    def __init__(self, graph: TrainingGraph, input_body_ids: Iterable[int], *,
                 mode: str = "learnable", config: TrainableLIFConfig | None = None,
                 dtype: torch.dtype = torch.float32):
        self._initialize(graph, input_body_ids, LAMINA_TYPES, mode=mode, config=config, dtype=dtype)

    def _initialize(self, graph, input_body_ids, tonic_types, *, mode, config, dtype):
        if mode not in ("learnable", "fixed"):
            raise ValueError("Tonic mode must be explicitly learnable or fixed")
        if (not tonic_types or any(name not in LAMINA_TYPES for name in tonic_types)
                or tonic_types != tuple(name for name in LAMINA_TYPES if name in tonic_types)):
            raise ValueError("Tonic types must be a nonempty ordered subset of exact L1--L5 types")
        if tonic_types != LAMINA_TYPES and graph.identity.get("role") != "numerical_test_only":
            raise ValueError("Production tonic dynamics require all exact L1--L5 types")
        # Group 0 explicitly contains missing annotations; never infer its type.
        mapping = np.zeros(graph.n, dtype=np.int64)
        members = []
        for index, name in enumerate(tonic_types, start=1):
            if name not in graph.group_names:
                raise ValueError(f"Required tonic type {name} has no annotated members")
            selected = graph.group_index == graph.group_names.index(name)
            body_ids = graph.body_ids[selected]
            if not len(body_ids):
                raise ValueError(f"Required tonic type {name} has no annotated members")
            mapping[selected] = index
            members.append({"type": name, "count": len(body_ids),
                            "body_ids_sha256": hashlib.sha256(np.asarray(body_ids, dtype="<i8").tobytes()).hexdigest()})
        inputs = tuple(input_body_ids)
        if np.isin(graph.body_ids[mapping > 0], inputs).any():
            raise ValueError("Tonic cells must not overlap registered external input cells")
        super().__init__(graph, inputs, config=config, dtype=dtype)
        self.tonic_types, self.tonic_mode = tonic_types, mode
        rho = torch.zeros(len(tonic_types), dtype=dtype)
        if mode == "learnable":
            self.rho = nn.Parameter(rho)
        else:
            self.register_buffer("rho", rho)
        self.register_buffer("tonic_group_index", torch.tensor(mapping), persistent=False)
        self._identity.update({
            "schema": TONIC_SCHEMA,
            "base_dynamics_schema": MODEL_SCHEMA,
            "tonic": {
                "mode": mode, "types": list(tonic_types), "members": members,
                "member_count": int(np.count_nonzero(mapping)),
                "mapping_sha256": hashlib.sha256(np.asarray(mapping, dtype="<i8").tobytes()).hexdigest(),
                "formula": "16*sigmoid(rho[type]); nextafter guards strict floating bounds; nonmembers=0",
                "range_mv": {"lower": 0, "upper": 16, "inclusive": False},
                "initial_rho": 0, "initial_offset_mv": 8,
                "integration": "active only: integrated_v += (1-exp(-dt/tau_m))*offset; g unchanged",
                "hypothesis": "engineering equilibrium offset, not a measured physiological constant",
            },
        })
        self._identity["implementation_sha256"]["tonic.py"] = file_hashes(Path(__file__))["sha256"]
        self.model_id = canonical_hash(self._identity)

    @classmethod
    def for_numerical_test(cls, counts, input_indices=(), *, types=None, missing=None,
                           tonic_types=LAMINA_TYPES, mode="learnable", config=None,
                           dtype=torch.float32) -> LaminaTonicLIF:
        """Test-only circuit with explicit lamina subset; never a production brain."""
        graph = numerical_test_graph(counts, types, missing)
        instance = cls.__new__(cls)
        instance._initialize(graph, input_indices, tuple(tonic_types), mode=mode, config=config, dtype=dtype)
        return instance

    @property
    def tonic_offsets_mv(self) -> Tensor:
        """One bounded offset per declared type, with gradients in learnable mode."""
        if (not isinstance(self.rho, Tensor) or self.rho.shape != (len(self.tonic_types),)
                or self.rho.dtype != self._dtype or self.rho.device.type != "cpu"
                or not torch.isfinite(self.rho).all()):
            raise ValueError("Tonic rho must retain its finite CPU shape and dtype")
        low, high = self.rho.new_tensor(0), self.rho.new_tensor(16)
        return (16 * torch.sigmoid(self.rho)).clamp(torch.nextafter(low, high), torch.nextafter(high, low))

    def _equilibrium_offset_mv(self) -> Tensor:
        # Padding preserves exact zero for nonmembers without broadcasting rho.
        offsets = self.tonic_offsets_mv
        return torch.cat((offsets.new_zeros(1), offsets))[self.tonic_group_index]
