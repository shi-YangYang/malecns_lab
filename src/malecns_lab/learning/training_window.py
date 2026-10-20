"""Torch-only two-frame update; state is retained and detached, never reset here."""
from __future__ import annotations

import time
import numpy as np
import torch

from .joint_probe_worker import snapshot, state_digest
from .training_contract import PARAMETERS, active_keys, require
from .training_math import clipping, validate_window_numbers
from .training_optimizer import snapshot_optimizer, validate_optimizer


def optimizer_snapshot(policy, optimizer, mode, steps):
    if not steps:
        validate_optimizer(policy, optimizer, mode)
        require(not optimizer.state, "Initial optimizer unexpectedly has moments")
        return {}
    _, result = snapshot_optimizer(policy, optimizer, mode,
                                    {key: value for key, value in snapshot(policy).items() if key in PARAMETERS}, steps)
    return result


def compute_window(policy, optimizer, state, selection, *, mode, before_steps, phase=lambda *a, **k: None,
                   failure_directory=None):
    """Shared numeric operation; production construction exists only in the worker.

    Explicit numerical policies may exercise this function. It does not register
    a production run, load a graph, or grant a protocol-complete result.
    """
    start = time.monotonic()
    before = snapshot(policy)
    start_state = state_digest(state)
    before_moments = optimizer_snapshot(policy, optimizer, mode, before_steps)
    values = {**selection, **{f"before_{key}": value for key, value in before.items()},
              **{f"adam_before.{key}": value for key, value in before_moments.items()},
              "adam_before_present": np.array(bool(before_steps), dtype=np.bool_)}
    outputs = []
    durations = {}
    optimizer.zero_grad(set_to_none=True)
    try:
        updated = bool(selection["valid"].any())
        # Build the same regularizer before forward as the accepted first-window pilot.
        reg = 1e-4 * (policy.brain.gains - 1).square().mean() if updated else None
        for index, frame in enumerate(selection["frames"]):
            phase("forward", window_frame=index)
            output = policy.forward_tensor(frame, state)
            state = output.state
            state_digest(state)  # Stop immediately for a nonfinite complete recurrent state.
            outputs.append(output)
        action = torch.stack([output.action for output in outputs])
        values.update(actions=action.detach().numpy().copy(),
                      features_mv=torch.stack([output.features_mv.detach() for output in outputs]).numpy().copy())
        require(all(np.isfinite(values[key]).all() for key in ("actions", "features_mv")), "Nonfinite neural output")
        durations["forward_seconds"] = time.monotonic() - start
        training_losses = {"action": None, "regularization": None, "total": None}
        if updated:
            phase("backward")
            then = time.monotonic()
            target = torch.tensor(selection["teacher_target"], dtype=torch.float32)
            valid = torch.from_numpy(selection["valid"])
            task = ((action[valid] - target[valid]) / policy.action_scales).square().mean()
            loss = task + reg
            require(bool(torch.isfinite(loss)), "Nonfinite window loss")
            training_losses = {"action": float(task.detach()), "regularization": float(reg.detach()), "total": float(loss.detach())}
            loss.backward()
            named = dict(policy.named_parameters())
            raw = {key: named[PARAMETERS[key]].grad.detach().numpy().copy() for key in active_keys(mode)}
            values.update({f"pre_clip_{key}": value for key, value in raw.items()})
            _, clip_record = clipping(raw, mode)
            for key in active_keys(mode):
                group = "theta" if key == "theta" else "readout"
                named[PARAMETERS[key]].grad.mul_(clip_record[group]["coefficient_float32"])
                values[f"post_clip_{key}"] = named[PARAMETERS[key]].grad.detach().numpy().copy()
            durations["backward_clip_seconds"] = time.monotonic() - then
            phase("optimizer_step")
            optimizer.step()
        else:
            require(before_steps > 0, "Cannot seal an all-invalid first window with a zero-step optimizer")
        state = state.detach()
        optimizer.zero_grad(set_to_none=True)
        values.update({f"after_{key}": value for key, value in snapshot(policy).items()})
        values.update({f"adam_after.{key}": value for key, value in
                       optimizer_snapshot(policy, optimizer, mode, before_steps + updated).items()})
        # Independent arithmetic checks happen before any completed-window seal.
        numeric = validate_window_numbers(values, policy.identity, mode, before_steps)
        for key in training_losses:
            if updated:
                require(np.isclose(training_losses[key], numeric["metrics"][key], rtol=1e-6, atol=1e-7), "Tensor/report loss mismatch")
        durations["compute_seconds"] = time.monotonic() - start
        record = {"start_state": start_state, "end_state": state_digest(state), "numeric": numeric,
                  "training_losses": training_losses, "durations": durations}
        return state, values, record
    except BaseException:
        if failure_directory is not None:
            # Keep reached raw values, including NaN/Inf, as diagnostic data only.
            np.savez(failure_directory / "failed_window.npz", **values)
        raise
