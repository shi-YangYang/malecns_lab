"""Causal camera-only rollouts and two-frame online imitation updates."""
from __future__ import annotations
import time
import numpy as np
from ...control.body import BodyVelocity
from ...drones.yaw_task import WARMUP_FRAMES, ACTION_LIMIT, YawTaskEnvironment, score_episode
from .protocol import PARAMETERS, require


def _host(value):
    return value.detach().cpu().numpy().copy()


def step_neural(policy, state, environment, *, frame_offset=0, force_zero=False):
    frame_index = environment.frame_index
    image = environment.frame()
    teacher = environment.teacher_action
    output = policy.forward_tensor(image, state, frame_offset=frame_offset)
    candidate = float(output.action.detach().item())
    command = 0. if force_zero or frame_index < WARMUP_FRAMES else candidate
    require(np.isfinite(candidate) and abs(candidate) <= ACTION_LIMIT + 1e-7, "Invalid neural action")
    motion = environment.advance(BodyVelocity(wz=command))
    row = {"frame_index": frame_index, "time_s": frame_index * .05, "frame_bgr": image,
           "teacher_wz_rad_s": teacher, "candidate_wz_rad_s": candidate, "command_wz_rad_s": command,
           "logit": float(output.logit.detach().item()), "features_mv": _host(output.features_mv),
           "visual_features": _host(output.visual_features),
           "group_rates_hz": _host(output.group_rates_hz), "network_spikes": int(output.network_spikes.detach().item()),
           "simulation_start_ms": frame_index * 50., "simulation_end_ms": (frame_index + 1) * 50., **motion}
    require(output.state.frame_index == environment.frame_index, "Policy and body clocks differ")
    return output, row


def warmup(policy, state, environment, *, progress=lambda **kwargs: None):
    import torch
    rows = []
    with torch.no_grad():
        while environment.frame_index < WARMUP_FRAMES:
            output, row = step_neural(policy, state, environment, force_zero=True)
            state = output.state
            rows.append(row)
            progress(phase="warmup", frame=environment.frame_index)
    return state.detach(), rows


def pack_rows(rows):
    require(bool(rows), "Cannot save empty rollout")
    keys = set(rows[0])
    require(all(set(row) == keys for row in rows), "Inconsistent frame fields")
    return {key: np.asarray([row[key] for row in rows]) for key in sorted(keys)}


def compute_window(policy, optimizer, state, environment, *, theta_raw_norm_limit=1000., progress=lambda **kwargs: None):
    import torch
    require(environment.frame_index >= WARMUP_FRAMES and environment.frame_index + 2 <= environment.scene.total_frames,
            "A training window requires two observation frames")
    before = policy.parameters_numpy()
    optimizer.zero_grad()
    outputs, rows = [], []
    started = time.monotonic()
    for index in range(2):
        progress(phase="forward", frame=environment.frame_index)
        output, row = step_neural(policy, state, environment, frame_offset=index * 500)
        state = output.state
        outputs.append(output)
        rows.append(row)
    logits = torch.stack([out.logit.reshape(()) for out in outputs])
    actions = torch.stack([out.action.reshape(()) for out in outputs])
    targets = torch.tensor([row["teacher_wz_rad_s"] for row in rows], dtype=torch.float32, device=logits.device)
    task = (torch.nn.functional.softplus(2 * logits) - (1 + targets / ACTION_LIMIT) * logits).mean()
    regularization = 1e-4 * (policy.brain.gains - 1).square().mean()
    loss = task + regularization
    require(bool(torch.isfinite(loss)), "Nonfinite training loss")
    progress(phase="backward", frame=environment.frame_index)
    loss.backward()
    gradients = {}
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing or nonfinite gradient")
            gradients[name] = _host(parameter.grad)
    if optimizer.mode == "joint":
        theta = dict(policy.named_parameters())["brain.theta"].grad
        maximum = theta.abs().max().clamp_min(1.)
        norm = maximum * (theta / maximum).square().sum().sqrt()
        require(bool(torch.isfinite(norm)) and float(norm) <= theta_raw_norm_limit, "Raw theta norm exceeds registered gate")
    report = optimizer.step()
    after = policy.parameters_numpy()
    require(set(before) == set(after) == set(PARAMETERS), "Parameter inventory differs")
    changed = {key: int(np.count_nonzero(before[key] != after[key])) for key in PARAMETERS}
    require(changed == report["changed"], "Optimizer changed-count report differs")
    if optimizer.mode == "frozen":
        require(changed["brain.theta"] == 0 and np.count_nonzero(after["brain.theta"]) == 0, "Frozen brain changed")
    values = {**pack_rows(rows), **{"before."+k: v for k,v in before.items()},
              **{"after."+k: v for k,v in after.items()}, **{"gradient."+k: v for k,v in gradients.items()}}
    loss_report = {"task": float(task.detach()), "regularization": float(regularization.detach()),
                   "total": float(loss.detach()), "action_mse": float((((actions-targets)/ACTION_LIMIT)**2).mean().detach()),
                   "direction_errors": int(((actions.detach()*targets < 0) & (targets.abs() > 1e-8)).sum()),
                   "nonzero_targets": int((targets.abs() > 1e-8).sum())}
    state = state.detach()
    optimizer.zero_grad()
    return state, values, {"losses": loss_report, "optimizer": report, "compute_seconds": time.monotonic() - started}


def evaluate_episode(scene, mode, policy=None, *, progress=lambda **kwargs: None):
    """Zero and visual-P intentionally have no fabricated neural observations."""
    from dataclasses import asdict
    environment = YawTaskEnvironment(scene)
    rows, state = [], None
    started = time.monotonic()
    require(mode in ("joint", "frozen", "untrained", "zero", "visual_p"), "Unknown controller")
    neural = mode in ("joint", "frozen", "untrained")
    if neural:
        import torch
        require(policy is not None, "Neural controller requires full policy")
        state = policy.initial_state()
        with torch.no_grad():
            for _ in range(scene.total_frames):
                output, row = step_neural(policy, state, environment)
                state = output.state
                rows.append(row)
                progress(phase="evaluate", frame=environment.frame_index)
    else:
        from ...senses.vision_v2 import VisionV2Encoder
        vision = VisionV2Encoder() if mode == "visual_p" else None
        for index in range(scene.total_frames):
            image, teacher = environment.frame(), environment.teacher_action
            candidate = 0.
            if vision is not None:
                observation = vision.encode(image, index * .05)
                require(observation.valid, "Visual-P input invalid")
                if not observation.initialization:
                    flow = np.concatenate([eye.flow_px_s[eye.flow_valid, 0] for eye in observation.eyes.values()])
                    candidate = float(np.clip(-2 * np.mean(flow) / 60., -ACTION_LIMIT, ACTION_LIMIT))
            command = 0. if index < WARMUP_FRAMES else candidate
            motion = environment.advance(BodyVelocity(wz=command))
            rows.append({"frame_index": index, "time_s": index*.05, "frame_bgr": image,
                         "teacher_wz_rad_s": teacher, "candidate_wz_rad_s": candidate, "command_wz_rad_s": command, **motion})
            progress(phase="evaluate", frame=environment.frame_index)
    arrays = pack_rows(rows)
    metrics = score_episode(scene, arrays["actual_wz_rad_s"], arrays["command_wz_rad_s"])
    summary = {"scene": asdict(scene), "mode": mode, "neural_executed": neural,
               "frames": scene.total_frames, "neural_steps": scene.total_frames*500 if neural else 0,
               "metrics": metrics, "wall_seconds": time.monotonic()-started,
               "group_names": list(policy.identity["group_names"]) if neural else [],
               "policy_identity": policy.identity if neural else None}
    return summary, arrays
