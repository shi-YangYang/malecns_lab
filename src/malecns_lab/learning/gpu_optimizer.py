"""Device-resident named fused Adam and overflow-safe MPS gradient clipping."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from ..brain.metadata import canonical_hash, file_hashes
from ..brain.mps_trainable import require_mps

NAMES = ("brain.theta", "readout.weight", "readout.bias")
FIELDS = ("step", "exp_avg", "exp_avg_sq")
OPTIONS = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
           "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
           "fused": True, "decoupled_weight_decay": False}


class GPUAdam:
    """The policy parameters are the sole masters; no CPU optimizer bridge."""

    def __init__(self, named_parameters, *, moments=None, steps=0):
        require_mps()
        self.parameters = dict(named_parameters)
        if set(self.parameters) != set(NAMES) or len({id(p) for p in self.parameters.values()}) != len(NAMES):
            raise ValueError("GPU Adam requires unique exact named policy parameters")
        if (type(steps) is not int or steps < 0 or steps >= 2 ** 24
                or (moments is None) != (steps == 0)):
            raise ValueError("Supply either zero state or complete named moments with a positive exact step")
        self.steps = steps
        self._validate_parameters()
        if (self.parameters["brain.theta"].ndim != 1 or self.parameters["readout.weight"].ndim != 2
                or self.parameters["readout.weight"].shape[0] != 3 or self.parameters["readout.bias"].shape != (3,)):
            raise ValueError("Unexpected neural/readout parameter shapes")
        self.groups = (("brain.theta",), ("readout.weight", "readout.bias"))
        self.optimizer = torch.optim.Adam([
            {"params": [self.parameters[n] for n in self.groups[0]], "lr": .001},
            {"params": [self.parameters[n] for n in self.groups[1]], "lr": .0001},
        ], **OPTIONS)
        expected = {f"{name}.{field}" for name in NAMES for field in FIELDS}
        if moments is not None and (not isinstance(moments, dict) or set(moments) != expected):
            raise ValueError("Missing/extra GPU Adam moments")
        for name, parameter in self.parameters.items():
            state = {}
            for field in FIELDS:
                shape = () if field == "step" else tuple(parameter.shape)
                if moments is None:
                    value = torch.zeros(shape, dtype=torch.float32, device="mps")
                else:
                    saved = moments[f"{name}.{field}"]
                    if (not isinstance(saved, np.ndarray) or saved.shape != shape or saved.dtype != np.float32
                            or not np.isfinite(saved).all() or field == "step" and float(saved) != steps
                            or field == "exp_avg_sq" and np.any(saved < 0)):
                        raise ValueError("Invalid GPU Adam archive array")
                    value = torch.from_numpy(saved.copy()).to("mps")
                state[field] = value
            self.optimizer.state[parameter] = state
        identity = {"schema": "cnsbridge.gpu_adam.v1", "backend": "torch_mps_fused_adam",
                    "groups": [{"parameters": list(names), "lr": lr} for names, lr in zip(self.groups, (.001, .0001))],
                    "options": {**OPTIONS, "betas": list(OPTIONS["betas"])}, "max_group_norm": 1., "clip_eps": 1e-6,
                    "clipping": "scale=max(maxabs,1); u=g/scale; q=sqrt(sum(u^2)); post=u*min(scale,1/(q+eps/scale))",
                    "device": "mps", "dtype": "float32", "torch_version": torch.__version__,
                    "implementation_sha256": file_hashes(Path(__file__))["sha256"]}
        self.identity = {**identity, "optimizer_id": canonical_hash(identity)}
        self.validate()

    def _validate_parameters(self):
        if any(not isinstance(p, torch.nn.Parameter) or not p.is_leaf or not p.requires_grad
               or p.device.type != "mps" or p.dtype != torch.float32 or p.numel() == 0
               or not torch.isfinite(p).all() for p in self.parameters.values()):
            raise ValueError("GPU master parameters must be finite nonempty trainable leaf MPS float32")

    def validate(self):
        self._validate_parameters()
        if type(self.optimizer) is not torch.optim.Adam or len(self.optimizer.param_groups) != 2:
            raise ValueError("GPU optimizer type/groups changed")
        if set(self.optimizer.state) != set(self.parameters.values()):
            raise ValueError("GPU optimizer state bindings changed")
        for names, lr, group in zip(self.groups, (.001, .0001), self.optimizer.param_groups):
            if (set(group) != {"params", "lr", *OPTIONS} or group["lr"] != lr
                    or any(group[k] != v for k, v in OPTIONS.items())
                    or len(group["params"]) != len(names)
                    or any(actual is not self.parameters[n] for actual, n in zip(group["params"], names))):
                raise ValueError("GPU optimizer options/bindings changed")
        for parameter in self.parameters.values():
            state = self.optimizer.state[parameter]
            if set(state) != set(FIELDS):
                raise ValueError("GPU optimizer state fields changed")
            for field, value in state.items():
                if (not isinstance(value, torch.Tensor) or value.device.type != "mps" or value.dtype != torch.float32
                        or value.requires_grad or value.grad_fn is not None
                        or value.shape != (() if field == "step" else parameter.shape) or not torch.isfinite(value).all()):
                    raise ValueError("Adam parameter/step/moments must remain on MPS")
            if (state["exp_avg_sq"] < 0).any() or state["step"] != self.steps:
                raise ValueError("GPU Adam moment or step differs")

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        self.validate()
        if self.steps >= 2 ** 24 - 1:
            raise ValueError("Float32 step would leave exact integer range")
        if any(p.grad is None or p.grad.device.type != "mps" or p.grad.dtype != torch.float32
               or p.grad.shape != p.shape or not torch.isfinite(p.grad).all() for p in self.parameters.values()):
            raise ValueError("All raw gradients must be finite MPS float32")
        raw = {n: p.grad.detach().clone() for n, p in self.parameters.items()}
        clipped, reports = {}, {}
        for label, names in zip(("theta", "readout"), self.groups):
            maximum = torch.stack([raw[n].abs().max() for n in names]).max()
            scale = maximum.clamp_min(1.)
            normalized = {n: raw[n] / scale for n in names}
            q = torch.stack([v.square().sum() for v in normalized.values()]).sum().sqrt()
            multiplier = torch.minimum(scale, 1. / (q + 1e-6 / scale))
            for name in names:
                clipped[name] = normalized[name] * multiplier
            reports[label] = {"scale": scale, "normalized_norm": q, "normalized_multiplier": multiplier,
                              "effective_coefficient": multiplier / scale,
                              "norm_after": torch.stack([clipped[n].square().sum() for n in names]).sum().sqrt(),
                              "nonzero_before": torch.stack([torch.count_nonzero(raw[n]) for n in names]).sum(),
                              "nonzero_after": torch.stack([torch.count_nonzero(clipped[n]) for n in names]).sum()}
            if not torch.isfinite(multiplier) or reports[label]["norm_after"] > 1 + 1e-6:
                raise ValueError("Invalid GPU clipping result")
        for name, parameter in self.parameters.items():
            parameter.grad.copy_(clipped[name])
        # Unsupported fused MPS operations raise; CPU fallback is disabled.
        self.optimizer.step()
        self.steps += 1
        self.validate()
        return {"raw": raw, "clipped": clipped, "groups": reports, "steps": self.steps}

    def snapshot(self):
        self.validate()
        return {"steps": self.steps, "optimizer_id": self.identity["optimizer_id"],
                "parameters": {n: p.detach().cpu().numpy().copy() for n, p in self.parameters.items()},
                "moments": {f"{n}.{field}": value.detach().cpu().numpy().copy()
                            for n, p in self.parameters.items() for field, value in self.optimizer.state[p].items()}}
