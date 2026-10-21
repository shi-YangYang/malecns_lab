"""Single-axis MPS fused Adam; frozen control never registers internal theta."""
from pathlib import Path
from copy import deepcopy
import math

import numpy as np
import torch

from ...brain.metadata import canonical_hash, file_hashes
from ...brain.mps_trainable import require_mps
from .policy import PARAMETER_NAMES

OPTIONS = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
           "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
           "fused": True, "decoupled_weight_decay": False}
FIELDS = ("step", "exp_avg", "exp_avg_sq")


def stable_norm(values):
    maximum = torch.stack([value.detach().abs().max() for value in values]).max()
    scale = maximum.clamp_min(1.)
    return scale * torch.stack([(value.detach() / scale).square().sum() for value in values]).sum().sqrt()


class YawAdam:
    def __init__(self, policy, mode="joint", *, snapshot=None):
        require_mps()
        if mode not in ("joint", "frozen"):
            raise ValueError("Choose joint or frozen yaw training")
        self.policy, self.mode = policy, mode
        named = dict(policy.named_parameters())
        if set(named) != set(PARAMETER_NAMES) or named["readout.weight"].ndim != 2 or named["readout.weight"].shape[0] != 1 or named["readout.bias"].shape != (1,):
            raise ValueError("Yaw optimizer requires the exact one-axis policy parameters")
        if mode == "frozen" and torch.count_nonzero(named["brain.theta"]):
            raise ValueError("Frozen control must begin with unmodified internal gains")
        named["brain.theta"].requires_grad_(mode == "joint")
        self.parameters = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
        self.groups = ({"theta": ("brain.theta",)} if mode == "joint" else {}) | {"readout": ("readout.weight", "readout.bias")}
        self.optimizer = torch.optim.Adam([{ "params": [self.parameters[name] for name in names], "lr": .001}
                                           for names in self.groups.values()], **OPTIONS)
        identity = {"schema": "malecns_lab.yaw_adam.v1", "mode": mode, "policy_id": policy.policy_id,
                    "groups": {key: list(names) for key, names in self.groups.items()}, "lr": .001,
                    "options": {**OPTIONS, "betas": list(OPTIONS["betas"])}, "max_group_norm": 1.,
                    "clip_eps": 1e-6, "dtype": "torch.float32", "device": "mps", "torch_version": torch.__version__,
                    "source_sha256": file_hashes(Path(__file__))["sha256"]}
        self.identity = {**identity, "optimizer_id": canonical_hash(identity)}
        if snapshot is not None and not isinstance(snapshot, dict):
            raise ValueError("Yaw optimizer snapshot must be a dictionary")
        self.steps = 0 if snapshot is None else snapshot.get("steps")
        if type(self.steps) is not int or not 0 <= self.steps < 2**24:
            raise ValueError("Invalid exact float32 optimizer step")
        expected = {f"{name}.{field}" for name in self.parameters for field in FIELDS}
        if snapshot is not None and (set(snapshot) != {"schema", "mode", "steps", "identity", "moments"}
                or snapshot["schema"] != "malecns_lab.yaw_adam_snapshot.v1" or snapshot["mode"] != mode
                or snapshot["identity"] != self.identity or set(snapshot["moments"]) != expected):
            raise ValueError("Yaw optimizer restore identity/inventory differs")
        for name, parameter in self.parameters.items():
            moments = {}
            for field in FIELDS:
                shape = () if field == "step" else tuple(parameter.shape)
                if snapshot is None:
                    value = torch.zeros(shape, dtype=torch.float32, device="mps")
                else:
                    saved = snapshot["moments"][f"{name}.{field}"]
                    if (not isinstance(saved, np.ndarray) or saved.dtype != np.float32 or saved.shape != shape
                            or not np.isfinite(saved).all() or field == "step" and float(saved) != self.steps
                            or field == "exp_avg_sq" and np.any(saved < 0)):
                        raise ValueError("Invalid yaw Adam state tensor")
                    value = torch.from_numpy(saved.copy()).to("mps")
                moments[field] = value
            self.optimizer.state[parameter] = moments
        self.validate()

    def validate(self):
        self.policy.validate_parameters()
        named = dict(self.policy.named_parameters())
        expected = set(PARAMETER_NAMES) if self.mode == "joint" else set(PARAMETER_NAMES[1:])
        if set(self.parameters) != expected or named["brain.theta"].requires_grad != (self.mode == "joint"):
            raise ValueError("Yaw optimizer mode/parameter binding changed")
        if type(self.optimizer) is not torch.optim.Adam or len(self.optimizer.param_groups) != len(self.groups):
            raise ValueError("Yaw Adam group/type changed")
        for names, group in zip(self.groups.values(), self.optimizer.param_groups):
            if (set(group) != {"params", "lr", *OPTIONS} or group["lr"] != .001
                    or any(group[key] != value for key, value in OPTIONS.items())
                    or len(group["params"]) != len(names)
                    or any(parameter is not self.parameters[name] for parameter, name in zip(group["params"], names))):
                raise ValueError("Yaw Adam options/parameter binding changed")
        if set(self.optimizer.state) != set(self.parameters.values()):
            raise ValueError("Yaw Adam state parameter binding changed")
        for name, parameter in self.parameters.items():
            if named[name] is not parameter or not parameter.is_leaf or not parameter.requires_grad:
                raise ValueError("Yaw parameter binding changed")
            state = self.optimizer.state[parameter]
            if set(state) != set(FIELDS):
                raise ValueError("Yaw Adam fields changed")
            for field, value in state.items():
                if (value.device.type != "mps" or value.dtype != torch.float32 or value.requires_grad
                        or value.shape != (() if field == "step" else parameter.shape) or not torch.isfinite(value).all()):
                    raise ValueError("Yaw Adam state must remain finite and on MPS")
            if state["step"] != self.steps or (state["exp_avg_sq"] < 0).any():
                raise ValueError("Yaw Adam step/second moment differs")
            if self.steps == 0 and (torch.count_nonzero(state["exp_avg"]) or torch.count_nonzero(state["exp_avg_sq"])):
                raise ValueError("Initial Adam moments must be zero")

    def zero_grad(self):
        self.policy.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        self.validate()
        if self.steps >= 2**24 - 1:
            raise ValueError("Adam step outside exact float32 range")
        before = {name: parameter.detach().clone() for name, parameter in self.policy.named_parameters()}
        raw = {}
        for name, parameter in self.parameters.items():
            if (parameter.grad is None or parameter.grad.device.type != "mps" or parameter.grad.dtype != torch.float32
                    or parameter.grad.shape != parameter.shape or not torch.isfinite(parameter.grad).all()):
                raise ValueError("Nonfinite, missing or foreign yaw gradient")
            raw[name] = parameter.grad.detach().clone()
        raw_norms, clipped_norms = {}, {}
        for label, names in self.groups.items():
            raw_norms[label] = float(stable_norm([raw[name] for name in names]).item())
            if not math.isfinite(raw_norms[label]):
                raise ValueError("Nonfinite yaw gradient norm")
            maximum = torch.stack([raw[name].abs().max() for name in names]).max()
            scale = maximum.clamp_min(1.)
            normalized = {name: raw[name] / scale for name in names}
            q = torch.stack([value.square().sum() for value in normalized.values()]).sum().sqrt()
            multiplier = torch.minimum(scale, 1. / (q + 1e-6 / scale))
            if not torch.isfinite(multiplier):
                raise ValueError("Nonfinite yaw gradient clip")
            for name in names:
                self.parameters[name].grad.copy_(normalized[name] * multiplier)
            norm = stable_norm([self.parameters[name].grad for name in names])
            if not torch.isfinite(norm) or norm > 1 + 1e-6:
                raise ValueError("Invalid clipped yaw gradient norm")
            clipped_norms[label] = float(norm.item())
        self.optimizer.step()
        self.steps += 1
        self.validate()
        changed = {name: int(torch.count_nonzero(parameter.detach() != before[name]).item()) for name, parameter in self.policy.named_parameters()}
        return {"steps": self.steps, "mode": self.mode, "raw_norms": raw_norms,
                "raw_nonzero": {name: int(torch.count_nonzero(value).item()) for name, value in raw.items()},
                "changed": changed, "post_clip_norms": clipped_norms,
                "residency": {"parameters": {name: str(value.device) for name, value in self.policy.named_parameters()},
                              "gradients": {name: str(value.device) for name, value in raw.items()},
                              "adam": {f"{name}.{field}": str(value.device) for name, parameter in self.parameters.items()
                                       for field, value in self.optimizer.state[parameter].items()}}}

    def snapshot(self):
        self.validate()
        return {"schema": "malecns_lab.yaw_adam_snapshot.v1", "mode": self.mode, "steps": self.steps,
                "identity": deepcopy(self.identity),
                "moments": {f"{name}.{field}": value.detach().cpu().numpy().copy() for name, parameter in self.parameters.items()
                            for field, value in self.optimizer.state[parameter].items()}}
