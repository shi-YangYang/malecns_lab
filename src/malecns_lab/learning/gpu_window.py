"""One joint all-GPU update; host copies exist only as boundary evidence."""
import time
import numpy as np
import torch
from .training_contract import PARAMETERS, require
from .gpu_state import state_digest
from .gpu_numbers import validate_numbers


def clock():
    torch.mps.synchronize()
    return time.monotonic()


def logit_loss(logits, target, scales):
    y = target / scales
    require(torch.isfinite(y).all() and torch.all(y.abs() <= 1), "Normalized teacher outside action contract")
    return (torch.nn.functional.softplus(2 * logits) - (1 + y) * logits).mean()


def policy_snapshot(policy):
    return {**{k: v.detach().cpu().numpy().copy() for k, name in PARAMETERS.items()
               for v in (dict(policy.named_parameters())[name],)},
            "gains": policy.brain.gains.detach().cpu().numpy().copy(), "rho": policy.brain.rho.detach().cpu().numpy().copy()}


def stable_norm(value):
    maximum = value.detach().abs().max()
    scale = maximum.clamp_min(1.)
    return scale * (value.detach() / scale).square().sum().sqrt()


def compute_window(policy, optimizer, state, selection, *, theta_norm_limit, phase=lambda *a, **k: None, failure_directory=None):
    started, before_steps = clock(), optimizer.steps
    initial = state_digest(policy, state)
    before = policy_snapshot(policy)
    old = optimizer.snapshot()
    values = {**selection, **{f"before_{k}": v for k, v in before.items()},
              **{f"adam_before.{k}": v for k, v in old["moments"].items()}}
    optimizer.zero_grad()
    durations = {}
    try:
        require(selection["valid"].all() and len(selection["frames"]) == 2, "Two valid frames required")
        reg = 1e-4 * (policy.brain.gains - 1).square().mean()
        outputs = []
        began = clock()
        for index, frame in enumerate(selection["frames"]):
            phase("forward", window_frame=index)
            output = policy.forward_tensor(frame, state, frame_offset=index * policy.frame_steps)
            state = output.state
            outputs.append(output)
        logits = torch.stack([o.logits for o in outputs])
        actions = torch.stack([o.action for o in outputs])
        features = torch.stack([o.features_mv for o in outputs])
        target = torch.tensor(selection["teacher_target"], dtype=torch.float32, device="mps")
        task = logit_loss(logits, target, policy.action_scales)
        mse = ((actions - target) / policy.action_scales).square().mean()
        loss = task + reg
        require(torch.isfinite(loss), "Nonfinite GPU loss")
        durations["forward_seconds"] = clock() - began
        losses = {"logit": float(task.detach().item()), "action_mse": float(mse.detach().item()),
                  "regularization": float(reg.detach().item()), "total": float(loss.detach().item())}
        phase("backward")
        began = clock()
        loss.backward()
        durations["backward_seconds"] = clock() - began
        named = dict(policy.named_parameters())
        require(all(named[n].grad is not None and named[n].grad.device.type == "mps"
                    and torch.isfinite(named[n].grad).all() for n in PARAMETERS.values()), "Invalid device raw gradients")
        values.update({f"pre_clip_{k}": named[n].grad.detach().cpu().numpy().copy() for k, n in PARAMETERS.items()})
        theta_norm = stable_norm(policy.brain.theta.grad)
        require(torch.isfinite(theta_norm) and theta_norm <= theta_norm_limit, "Raw theta gradient exceeded registered gate")
        require(torch.count_nonzero(policy.brain.theta.grad) > 0, "Missing internal neural gradient")
        # Save raw evidence before clipping; no CPU copy is ever used for an update.
        values.update(logits=logits.detach().cpu().numpy().copy(), actions=actions.detach().cpu().numpy().copy(),
                      features_mv=features.detach().cpu().numpy().copy())
        phase("gpu_clip_adam")
        began = clock()
        update = optimizer.step()
        durations["optimizer_seconds"] = clock() - began
        for label, collection in (("pre_clip", update["raw"]), ("post_clip", update["clipped"])):
            values.update({f"{label}_{k}": collection[n].cpu().numpy().copy() for k, n in PARAMETERS.items()})
        values.update({f"after_{k}": v for k, v in policy_snapshot(policy).items()})
        values.update({f"adam_after.{k}": v for k, v in optimizer.snapshot()["moments"].items()})
        with torch.no_grad():
            normalized = (features.detach() - policy.brain.config.v_rest_mv) / (
                policy.brain.config.v_threshold_mv - policy.brain.config.v_rest_mv)
            post = torch.stack([policy.readout(row) for row in normalized])
            values["post_fixed_logits"] = post.cpu().numpy().copy()
        state = state.detach()
        optimizer.zero_grad()
        numeric = validate_numbers(values, policy.identity, before_steps, theta_norm_limit)
        require(numeric["parameter_updates"]["theta"]["changed"] > 0, "No internal parameter update")
        for key, value in losses.items():
            require(np.isclose(value, numeric["losses"][key], rtol=2e-5, atol=1e-6), "Reported loss differs from saved values")
        durations["compute_seconds"] = clock() - started
        residency = {"parameters": {n: str(v.device) for n, v in named.items()},
                     "raw_gradients": {n: str(v.device) for n, v in update["raw"].items()},
                     "clipped_gradients": {n: str(v.device) for n, v in update["clipped"].items()},
                     "adam": {f"{n}.{k}": str(v.device) for n, p in named.items() for k, v in optimizer.optimizer.state[p].items()},
                     "phase": str(state.drive.phase.device), "counts": str(state.drive.event_counts.device),
                     "logits": str(logits.device), "loss": str(loss.device)}
        return state, values, {"start_state": initial, "end_state": state_digest(policy, state), "numeric": numeric,
                               "training_losses": losses, "residency": residency, "durations": durations}
    except BaseException:
        if failure_directory is not None:
            np.savez(failure_directory / "failed_window.npz", **values)
        raise
