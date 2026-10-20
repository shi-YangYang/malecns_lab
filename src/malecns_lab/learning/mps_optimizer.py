"""CPU master Adam: preserve original clipping and named optimizer arithmetic."""
from collections import defaultdict
import time
import numpy as np
import torch
from .training_contract import PARAMETERS, require
from .training_optimizer import ADAM_OPTIONS, MOMENTS, optimizer_contract, validate_moments
from .training_math import clipping


def snapshot(policy):
    values = {key: dict(policy.named_parameters())[name].detach().cpu().numpy().copy() for key, name in PARAMETERS.items()}
    return {**values, "gains": policy.brain.gains.detach().cpu().numpy().copy(), "rho": policy.brain.rho.detach().cpu().numpy().copy()}


class CPUAdamBridge:
    def __init__(self, policy, parameters, moments, steps):
        require(set(parameters) == set(PARAMETERS), "Unexpected master parameter fields")
        self.policy, self.contract = policy, optimizer_contract("joint")
        named = dict(policy.named_parameters())
        require(set(named) == set(PARAMETERS.values()), "Unexpected MPS parameter fields")
        for key, value in parameters.items():
            require(value.dtype == np.float32 and value.shape == tuple(named[PARAMETERS[key]].shape)
                    and np.isfinite(value).all(), "Invalid master parameter shape/dtype/value")
        validate_moments(moments, self.contract, parameters, steps)
        self.masters = {name: torch.nn.Parameter(torch.from_numpy(parameters[key].copy())) for key, name in PARAMETERS.items()}
        groups = [{"params": [self.masters[n] for n in g["parameters"]], "lr": g["lr"]} for g in self.contract["groups"]]
        self.optimizer = torch.optim.Adam(groups, **{**ADAM_OPTIONS, "betas": tuple(ADAM_OPTIONS["betas"])})
        self.optimizer.state = defaultdict(dict, {p: {field: torch.from_numpy(moments[f"{name}.{field}"].copy()) for field in MOMENTS}
                                                   for name, p in self.masters.items()})
        self.steps = steps
        self.copy_to_device()
        policy._restored = True
        policy._validate_parameters()

    def validate(self):
        named = dict(self.policy.named_parameters())
        require(type(self.optimizer) is torch.optim.Adam and len(self.optimizer.param_groups) == len(self.contract["groups"]), "Adam type/groups changed")
        for group, expected in zip(self.optimizer.param_groups, self.contract["groups"]):
            require(set(group) == (set(expected) - {"parameters"}) | {"params"}, "Adam fields changed")
            actual = {k: list(v) if isinstance(v, tuple) else v for k, v in group.items() if k != "params"}
            require(actual == {k: v for k, v in expected.items() if k != "parameters"}, "Adam options changed")
            require(len(group["params"]) == len(expected["parameters"]) and all(p is self.masters[n] for p, n in zip(group["params"], expected["parameters"])), "Adam binding changed")
        require(set(self.optimizer.state) == set(self.masters.values()), "Unexpected Adam states")
        for name, parameter in self.masters.items():
            require(parameter.device.type == "cpu" and parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
                    and torch.equal(parameter.detach(), named[name].detach().cpu()), "GPU/CPU master parameters differ")

    @torch.no_grad()
    def copy_to_device(self):
        named = dict(self.policy.named_parameters())
        for name, parameter in self.masters.items():
            named[name].copy_(parameter)
            require(torch.equal(named[name].cpu(), parameter), "Parameter transfer was not exact")

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.policy.zero_grad(set_to_none=True)

    def moments(self):
        self.validate()
        result = {f"{name}.{field}": self.optimizer.state[parameter][field].detach().numpy().copy()
                  for name, parameter in self.masters.items() for field in MOMENTS}
        validate_moments(result, self.contract, {k: self.masters[n].detach().numpy() for k, n in PARAMETERS.items()}, self.steps)
        return result

    def step(self, raw):
        self.validate()
        torch.mps.synchronize()
        started = time.monotonic()
        scaled, report = clipping(raw, "joint")
        for key, name in PARAMETERS.items():
            self.masters[name].grad = torch.from_numpy(scaled[key].copy())
        self.optimizer.step()
        self.steps += 1
        cpu_finished = time.monotonic()
        self.copy_to_device()
        torch.mps.synchronize()
        return scaled, report, {"cpu_clip_adam_seconds": cpu_finished - started,
                                "parameter_transfer_seconds": time.monotonic() - cpu_finished}
