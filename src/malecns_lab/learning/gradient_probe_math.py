"""Read-only MPS credit-horizon experiment; never updates a parameter.

The original checkpoint block and addition order are reused. A horizon bounds
the recurrent-state gradient path, not the DN-to-readout path: the running DN
sums remain differentiable across every cut and every forward step still runs.
"""
import math
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ..brain.mps_identity import ACTIVATION_CHECKPOINT
from ..brain.trainable import LIFState
from .mps_recompute import _block_function
from .mps_state import state_arrays, state_digest
from .neural_policy import NeuralPolicyOutput, PolicyState
from .training_contract import PARAMETERS, require
from ..senses.target import target_mask
from ..senses.spatial_drive import MASK_SHAPE


def stable_metrics(value):
    """Scale on MPS before squaring; multiply scalar norms in Python float64."""
    value = value.detach()
    finite = bool(torch.isfinite(value).all().item())
    if not finite:
        return {"finite": False, "l2": None, "max_abs": None, "nonzero": None}
    maximum = value.abs().max()
    peak = float(maximum.item())
    norm = peak * math.sqrt(float((value / maximum).square().sum().item())) if peak else 0.
    return {"finite": True, "l2": norm, "max_abs": peak,
            "nonzero": int(torch.count_nonzero(value).item())}


class BoundaryObserver:
    """Observe output-state adjoints without retaining large gradient tensors.

    At a cut these are the LEFT (pre-detach) state tensors. Their gradient can
    include DN losses before the cut; future recurrent credit is blocked. A
    missing hook call is recorded as absent, never reported as measured zero.
    Refractory state is integer/non-differentiable and is explicitly labelled.
    """
    def __init__(self):
        self.rows = []

    def observe(self, state, window_step, cut):
        row = {"window_step": window_step, "brain_step": state.step, "cut_after_boundary": cut,
               "side": "pre_detach_output", "refractory": "integer_nondifferentiable", "adjoints": {}}
        for name, tensor in (("v", state.v), ("g", state.g),
                             *((f"pending.{i}", v) for i, v in enumerate(state.pending))):
            item = {"requires_grad": tensor.requires_grad, "observed": False}
            row["adjoints"][name] = item
            if tensor.requires_grad:
                def hook(gradient, item=item):
                    item.update(observed=True, **stable_metrics(gradient))
                    # Returning None leaves the original gradient unchanged.
                tensor.register_hook(hook)
        self.rows.append(row)


def observed_frame(policy, frame, state, *, horizon, frame_offset, observer):
    """Same MPS frame math with explicit state-only cuts and boundary hooks."""
    mask = np.array(target_mask(frame, policy.camera, MASK_SHAPE), copy=True)
    mask.setflags(write=False)
    policy._validate_parameters()
    drive_state = policy._drive_for(state).export_state()
    brain = state.brain
    half_steps = policy.frame_steps // 2
    sums = [brain.v.new_zeros(len(policy.mapping.outputs.indices)) for _ in range(2)]
    block_steps = ACTIVATION_CHECKPOINT["neural_steps_per_block"]
    require(horizon > 0 and horizon % block_steps == 0, "Credit horizon must align to checkpoint blocks")
    for start in range(0, policy.frame_steps, block_steps):
        count = min(block_steps, policy.frame_steps - start)
        block = _block_function(policy, mask, drive_state, brain.step, start, count)
        values = checkpoint(block, brain.v, brain.g, brain.refractory_until, *brain.pending, *sums,
                            use_reentrant=False, preserve_rng_state=False, determinism_check="default", early_stop=True)
        brain = LIFState(values[0], values[1], values[2], tuple(values[3:3 + policy.brain.delay_steps]),
                         brain.step + count, brain.model_id)
        sums, drive_state = list(values[-3:-1]), values[-1]
        completed = frame_offset + start + count
        cut = completed % horizon == 0 and completed < 2 * policy.frame_steps
        observer.observe(brain, completed, cut)
        if cut:
            brain = brain.detach()
    features = torch.cat(sums) / half_steps
    cfg = policy.brain.config
    normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
    action = torch.tanh(policy.readout(normalized)) * policy.action_scales
    return NeuralPolicyOutput(action, features, PolicyState(brain, drive_state, state.frame_index + 1, policy.policy_id))


def clock():
    torch.mps.synchronize()
    return time.monotonic()


def compute_branch(policy, state, selection, horizon, *, observed=True, phase=lambda *a, **k: None):
    """Identical two-frame MSE + gain regularization for every branch."""
    require(selection["valid"].all() and len(selection["frames"]) == 2, "Two valid frames required")
    policy.zero_grad(set_to_none=True)
    start_state = state_digest(state)
    require(observed or horizon == 2 * policy.frame_steps, "Unobserved replay must use full credit")
    observer = BoundaryObserver() if observed else None
    began = clock()
    reg = 1e-4 * (policy.brain.gains - 1).square().mean()
    outputs = []
    for index, frame in enumerate(selection["frames"]):
        phase("forward", horizon=horizon, window_frame=index)
        if observed:
            output = observed_frame(policy, frame, state, horizon=horizon,
                                    frame_offset=index * policy.frame_steps, observer=observer)
        else:
            output = policy.forward_tensor(frame, state)
        state = output.state
        state_digest(state)
        outputs.append(output)
    actions = torch.stack([o.action for o in outputs])
    forward_end = clock()
    values = {"actions": actions.detach().cpu().numpy().copy(),
              "features_mv": torch.stack([o.features_mv.detach() for o in outputs]).cpu().numpy().copy()}
    end_state = state_digest(state)
    values.update({f"end.{key}": value for key, value in state_arrays(state).items()})
    phase("backward", horizon=horizon)
    began_backward = clock()
    target = torch.tensor(selection["teacher_target"], dtype=torch.float32, device="mps")
    valid = torch.from_numpy(selection["valid"]).to("mps")
    task = ((actions[valid] - target[valid]) / policy.action_scales).square().mean()
    loss = task + reg
    require(torch.isfinite(loss), "Nonfinite diagnostic loss")
    losses = {"action": float(task.detach().cpu()), "regularization": float(reg.detach().cpu()),
              "total": float(loss.detach().cpu())}
    loss.backward()
    backward_end = clock()
    named = dict(policy.named_parameters())
    require(all(named[n].grad is not None for n in PARAMETERS.values()), "Missing diagnostic gradient")
    gradient_metrics = {key: stable_metrics(named[name].grad) for key, name in PARAMETERS.items()}
    values.update({f"pre_clip_{key}": named[name].grad.detach().cpu().numpy().copy()
                   for key, name in PARAMETERS.items()})
    require(state_digest(state) == end_state, "Checkpoint recomputation mutated the final state")
    record = {"horizon_steps": horizon, "forward_steps": 2 * policy.frame_steps,
              "losses": losses, "start_state": start_state, "end_state": end_state,
              "gradients": gradient_metrics, "boundaries": observer.rows if observed else [],
              "observed": observed, "forward_path": "observed_frame" if observed else "policy.forward_tensor",
              "durations": {"forward_seconds": forward_end - began,
                            "backward_seconds": backward_end - began_backward,
                            "total_seconds": clock() - began},
              "interpretation": "State-only gradient cuts; all DN running sums retain gradient; no optimizer."}
    policy.zero_grad(set_to_none=True)
    return values, record


def logit_loss_check(policy, features_mv, selection):
    """Fixed saved features, no brain replay or parameter update, all math MPS."""
    cfg = policy.brain.config
    features = torch.tensor(features_mv, device="mps")
    normalized = (features - cfg.v_rest_mv) / (cfg.v_threshold_mv - cfg.v_rest_mv)
    # Per-frame linear calls preserve the original MPS inference operation shape.
    logits = torch.stack([policy.readout(row) for row in normalized])
    actions = torch.tanh(logits) * policy.action_scales
    targets = torch.tensor(selection["teacher_target"], device="mps", dtype=torch.float32) / policy.action_scales
    require(torch.all(targets.abs() <= 1), "Normalized teacher outside action contract")
    losses = torch.nn.functional.softplus(2 * logits) - (1 + targets) * logits
    loss = losses.mean()
    grad = torch.autograd.grad(loss, logits)[0]
    expected = (torch.tanh(logits) - targets) / logits.numel()
    require(torch.isfinite(loss) and torch.isfinite(grad).all()
            and torch.allclose(grad, expected, rtol=1e-5, atol=1e-7), "Incorrect logit loss gradient")
    # A local logit descent check is a diagnostic; it is NOT a model update.
    trial_logits = logits.detach() - .1 * grad
    trial_loss = (torch.nn.functional.softplus(2 * trial_logits) - (1 + targets) * trial_logits).mean()
    return {"logits": logits.detach().cpu().numpy().copy(), "actions": actions.detach().cpu().numpy().copy(),
            "normalized_targets": targets.detach().cpu().numpy().copy(), "gradient": grad.detach().cpu().numpy().copy(),
            "expected_gradient": expected.detach().cpu().numpy().copy()}, {
                "loss": float(loss.detach().cpu()), "trial_logit_loss": float(trial_loss.detach().cpu()),
                "trial_step": .1, "model_updated": False, "device": str(logits.device),
                "gradient": stable_metrics(grad), "finite": True}
