"""One unchanged two-frame TBPTT update on MPS, with CPU master Adam."""
import time
import numpy as np
import torch
from .training_contract import PARAMETERS, require
from .training_math import validate_window_numbers
from .mps_optimizer import snapshot
from .mps_state import state_digest


def clock():
    torch.mps.synchronize()
    return time.monotonic()


def compute_window(policy, bridge, state, selection, *, phase=lambda *a, **k: None, failure_directory=None):
    start = clock()
    before_steps = bridge.steps
    before, before_moments = snapshot(policy), bridge.moments()
    start_state = state_digest(state)
    values = {**selection, **{f"before_{k}": v for k, v in before.items()},
              **{f"adam_before.{k}": v for k, v in before_moments.items()}, "adam_before_present": np.array(True)}
    durations = {}
    bridge.zero_grad()
    try:
        require(selection["valid"].all(), "Registered continuation dataset requires both frames valid")
        reg = 1e-4 * (policy.brain.gains - 1).square().mean()
        outputs = []
        then = clock()
        for index, frame in enumerate(selection["frames"]):
            phase("forward", window_frame=index)
            output = policy.forward_tensor(frame, state)
            state = output.state
            state_digest(state)
            outputs.append(output)
        actions = torch.stack([o.action for o in outputs])
        durations["forward_seconds"] = clock() - then
        then = clock()
        values.update(actions=actions.detach().cpu().numpy().copy(),
                      features_mv=torch.stack([o.features_mv.detach() for o in outputs]).cpu().numpy().copy())
        durations["output_transfer_seconds"] = clock() - then
        phase("backward")
        then = clock()
        target = torch.tensor(selection["teacher_target"], dtype=torch.float32, device="mps")
        valid = torch.from_numpy(selection["valid"]).to("mps")
        task = ((actions[valid] - target[valid]) / policy.action_scales).square().mean()
        loss = task + reg
        require(torch.isfinite(loss), "Nonfinite training loss")
        losses = {"action": float(task.detach().cpu()), "regularization": float(reg.detach().cpu()), "total": float(loss.detach().cpu())}
        loss.backward()
        durations["backward_seconds"] = clock() - then
        then = clock()
        named = dict(policy.named_parameters())
        require(all(named[n].grad is not None for n in PARAMETERS.values()), "Missing raw gradient")
        raw = {k: named[n].grad.detach().cpu().numpy().copy() for k, n in PARAMETERS.items()}
        values.update({f"pre_clip_{k}": v for k, v in raw.items()})
        durations["gradient_transfer_seconds"] = clock() - then
        phase("cpu_clip_adam")
        then = clock()
        scaled, _, bridge_durations = bridge.step(raw)
        values.update({f"post_clip_{k}": v for k, v in scaled.items()})
        durations.update(bridge_durations)
        durations["optimizer_bridge_seconds"] = clock() - then
        state = state.detach()
        bridge.zero_grad()
        values.update({f"after_{k}": v for k, v in snapshot(policy).items()})
        values.update({f"adam_after.{k}": v for k, v in bridge.moments().items()})
        numeric = validate_window_numbers(values, policy.identity, "joint", before_steps)
        for key, value in losses.items():
            require(np.isclose(value, numeric["metrics"][key], rtol=1e-6, atol=1e-7), "Tensor/report loss mismatch")
        require(numeric["clipping"]["theta"]["nonzero_before"] > 0
                and numeric["parameter_updates"]["theta"]["changed"] > 0, "No internal neural update observed")
        durations["compute_seconds"] = clock() - start
        return state, values, {"start_state": start_state, "end_state": state_digest(state), "numeric": numeric,
                               "training_losses": losses, "durations": durations}
    except BaseException:
        if failure_directory is not None:
            np.savez(failure_directory / "failed_window.npz", **values)
        raise
