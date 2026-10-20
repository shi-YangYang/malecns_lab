"""Named, fixed CPU Adam contract for joint and frozen-brain continuations."""
from __future__ import annotations

import numpy as np
import torch

from . import policy_checkpoint as parameters

MODES = ("joint", "frozen_brain")
ADAM_OPTIONS = {
    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "amsgrad": False,
    "maximize": False, "foreach": False, "capturable": False, "differentiable": False,
    "fused": False, "decoupled_weight_decay": False,
}
MOMENTS = ("step", "exp_avg", "exp_avg_sq")


def optimizer_contract(mode):
    if mode not in MODES:
        raise ValueError("Unknown training mode")
    groups = []
    if mode == "joint":
        groups.append({"parameters": ["brain.theta"], "lr": 0.001, **ADAM_OPTIONS})
    groups.append({"parameters": ["readout.weight", "readout.bias"], "lr": 0.0001, **ADAM_OPTIONS})
    return parameters._json({"type": "torch.optim.Adam", "mode": mode, "groups": groups})


def _policy_parameters(policy):
    identity, _ = parameters._arrays(policy)
    if identity["dtype"] != "torch.float32":
        raise ValueError("Training continuation requires CPU float32")
    return dict(policy.named_parameters(remove_duplicate=False))


def make_training_optimizer(policy, mode):
    """Explicitly select the mode; changing a mode creates a new optimizer."""
    contract = optimizer_contract(mode)
    named = _policy_parameters(policy)
    if not all(named[name].requires_grad for name in ("readout.weight", "readout.bias")):
        raise ValueError("Both readout parameters must require gradients")
    named["brain.theta"].requires_grad_(mode == "joint")
    groups = [{"params": [named[name] for name in group["parameters"]], "lr": group["lr"]}
              for group in contract["groups"]]
    options = {**ADAM_OPTIONS, "betas": tuple(ADAM_OPTIONS["betas"])}
    return torch.optim.Adam(groups, **options)


def validate_optimizer(policy, optimizer, mode):
    """Reject positional/duplicate/extra parameter bindings and changed options."""
    named = _policy_parameters(policy)
    contract = optimizer_contract(mode)
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != len(contract["groups"]):
        raise ValueError("Optimizer must use the declared Adam groups")
    for name, value in named.items():
        if value.requires_grad != (mode == "joint" or name != "brain.theta"):
            raise ValueError("Parameter requires_grad differs from the selected mode")
    active = {}
    for group, expected in zip(optimizer.param_groups, contract["groups"]):
        if set(group) != (set(expected) - {"parameters"}) | {"params"}:
            raise ValueError("Unknown or missing optimizer group fields")
        options = parameters._json({k: v for k, v in group.items() if k != "params"})
        if options != {k: v for k, v in expected.items() if k != "parameters"}:
            raise ValueError("Optimizer configuration differs from the fixed contract")
        if (len(group["params"]) != len(expected["parameters"])
                or any(actual is not named[name] for actual, name in zip(group["params"], expected["parameters"]))):
            raise ValueError("Optimizer parameters must retain exact named bindings/order")
        active.update((name, named[name]) for name in expected["parameters"])
    if any(not any(key is value for value in active.values()) for key in optimizer.state):
        raise ValueError("Unexpected optimizer state parameter")
    return contract, active


def validate_moments(arrays, contract, parameter_arrays, steps):
    """Validate serialized actual float32 moments; zero second moments are valid."""
    names = [name for group in contract["groups"] for name in group["parameters"]]
    expected = {f"{name}.{key}" for name in names for key in MOMENTS}
    if set(arrays) != expected:
        raise ValueError("Optimizer archive has missing/extra named states")
    shapes = {parameters.PARAMETERS[key]: value.shape for key, value in parameter_arrays.items()}
    if type(steps) is not int or steps <= 0:
        raise ValueError("optimizer_steps must be a positive integer")
    for name in names:
        for key in MOMENTS:
            value = arrays[f"{name}.{key}"]
            shape = () if key == "step" else shapes[name]
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Adam states must retain finite float32 scalar/parameter shapes")
            if key == "step" and (float(value) != steps or not float(value).is_integer()):
                raise ValueError("Adam step must equal the positive optimizer_steps cursor")
            if key == "exp_avg_sq" and np.any(value < 0):
                raise ValueError("Adam second moment must be nonnegative")


def snapshot_optimizer(policy, optimizer, mode, parameter_arrays, steps):
    contract, active = validate_optimizer(policy, optimizer, mode)
    arrays = {}
    for name, parameter in active.items():
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or set(state) != set(MOMENTS):
            raise ValueError("A complete actual Adam update is required before sealing")
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.requires_grad or value.grad_fn is not None):
                raise ValueError("Adam states must be detached CPU tensors")
            arrays[f"{name}.{key}"] = value.numpy().copy()
    validate_moments(arrays, contract, parameter_arrays, steps)
    return contract, arrays
