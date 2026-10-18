"""Child-only P3b computation. Never import this module from the supervisor."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import psutil
import scipy
import torch

from .joint_probe_contract import (BRANCHES, PARAMETERS, PROTOCOL, RESULT_SCHEMA, array_digest, bound,
                                   read_json, require, select_data, sha256, validate_registration)
from .neural_policy import NeuralPolicy, NeuralPolicyConfig
from .policy_checkpoint import load_policy_checkpoint, save_policy_checkpoint
from .resource_guard import write_json
from ..brain.trainable import TrainableLIFConfig
from ..senses.spatial_drive import SpatialDriveConfig
from ..senses.target import TargetCamera


def snapshot(policy):
    parameters = dict(policy.named_parameters())
    return {**{key: parameters[name].detach().numpy().copy() for key, name in PARAMETERS.items()},
            "gains": policy.brain.gains.detach().numpy().copy(), "rho": policy.brain.rho.detach().numpy().copy()}


def raw_counts_digest(graph):
    digest = hashlib.sha256()
    for values in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
        for offset in range(0, len(values), 1_000_000):
            digest.update(np.asarray(values[offset:offset + 1_000_000], dtype="<i8").tobytes())
    return digest.hexdigest()


def state_digest(state):
    """Digest every state value, including delayed arrivals; no per-step trajectory claim."""
    values = {"v": state.brain.v, "g": state.brain.g, "refractory_until": state.brain.refractory_until,
              **{f"pending_{i}": item for i, item in enumerate(state.brain.pending)}}
    arrays = {key: array_digest(value.detach().numpy()) for key, value in values.items()}
    arrays.update(drive_phase=array_digest(state.drive.phase), drive_event_counts=array_digest(state.drive.event_counts))
    record = {"policy_id": state.policy_id, "model_id": state.brain.model_id,
              "drive_id": state.drive.drive_id, "spatial_map_id": state.drive.spatial_map_id,
              "brain_step": state.brain.step, "drive_step": state.drive.step, "frame_index": state.frame_index,
              "drive_dt_ms": state.drive.dt_ms, "values": arrays}
    from ..brain.metadata import canonical_hash
    return {**record, "state_id": canonical_hash(record)}


def metrics(actions, targets, gains):
    # Float64 reporting is separately reproducible from saved float32 actions.
    per_axis = np.mean(((actions.astype(np.float64) - targets.astype(np.float32).astype(np.float64)) /
                        np.array(PROTOCOL["axis_scales"], dtype=np.float32).astype(np.float64)) ** 2, axis=0)
    regularization = PROTOCOL["regularization"] * np.mean((gains.astype(np.float64) - 1) ** 2)
    return {"action_mse_per_axis": per_axis.tolist(), "action_loss": float(per_axis.mean()),
            "regularization_loss": float(regularization), "total_loss": float(per_axis.mean() + regularization)}


def response(policy, frames, targets, name, directory, run_identity, phase):
    phase(name + "_precondition", completed_frames=0)
    state = policy.initial_state()
    start = state_digest(state)
    actions, features = [], []
    for index, frame in enumerate(frames):
        phase(name + "_forward", completed_frames=index)
        output = policy.forward_tensor(frame, state)
        state = output.state  # No detach/reset at the intermediate frame.
        actions.append(output.action)
        features.append(output.features_mv)
        phase(name + "_forward", completed_frames=index + 1)
    end = state_digest(state)
    action_tensor = torch.stack(actions)
    values = {"actions": action_tensor.detach().numpy().copy(),
              "features_mv": torch.stack(features).detach().numpy().copy(), **snapshot(policy)}
    require(all(np.isfinite(value).all() for value in values.values()), "Nonfinite branch output")
    np.savez(directory / f"{name}.npz", **values, **run_identity)
    record = {"initial_state": start, "end_state": end, "completed_frames": 2,
              "metrics": metrics(values["actions"], targets, values["gains"])}
    write_json(directory / f"{name}.json", record)
    # No state or feature graph escapes; actions alone retain the loss graph.
    return action_tensor, record


def optimizer_for(policy):
    config = PROTOCOL["optimizer"]
    return torch.optim.Adam([{"params": [policy.brain.theta], "lr": config["groups"][0]["lr"]},
                             {"params": [policy.readout.weight, policy.readout.bias], "lr": config["groups"][1]["lr"]}],
                            betas=tuple(config["betas"]), eps=config["eps"], weight_decay=0, amsgrad=False,
                            foreach=False, fused=False)


def clip_gradients(policy):
    parameters = dict(policy.named_parameters())
    ordered = [parameters[name] for name in PARAMETERS.values()]
    before = {key: parameters[name].grad.detach().numpy().copy() for key, name in PARAMETERS.items()}
    norm = torch.nn.utils.clip_grad_norm_(ordered, 1.0, error_if_nonfinite=True, foreach=False)
    coefficient = torch.clamp(norm.new_tensor(1.0) / (norm + 1e-6), max=1.0)
    after = {key: parameters[name].grad.detach().numpy().copy() for key, name in PARAMETERS.items()}
    return before, after, float(norm), float(coefficient)


def update_and_intervene(policy, frames, targets, directory, run_identity, phase):
    """Shared numerical experiment; production construction/reload stay in run_worker."""
    require(tuple(dict(policy.named_parameters())) == tuple(PARAMETERS.values()), "Wrong trainable parameter order")
    before = snapshot(policy)
    require(np.count_nonzero(before["theta"]) == 0 and np.all(before["gains"] == 1)
            and np.count_nonzero(before["rho"]) == 0, "Initial theta/gain/rho differs")
    np.savez(directory / "initial_parameters.npz", **before, **run_identity)
    optimizer = optimizer_for(policy)
    reg = PROTOCOL["regularization"] * (policy.brain.gains - 1).square().mean()
    reg_gradient = torch.autograd.grad(reg, policy.brain.theta, retain_graph=True)[0].detach().numpy().copy()
    require(float(reg.detach()) == 0 and not np.count_nonzero(reg_gradient), "Initial regularizer or gradient is nonzero")
    actions, baseline_record = response(policy, frames, targets, "baseline", directory, run_identity, phase)
    target_tensor = torch.tensor(targets, dtype=policy.brain._dtype)
    task = ((actions - target_tensor) / policy.action_scales).square().mean()
    loss = task + reg
    require(bool(torch.isfinite(loss)), "Nonfinite action loss")
    training_losses = {"action": float(task.detach()), "regularization": float(reg.detach()), "total": float(loss.detach())}
    phase("backward", completed_frames=2)
    loss.backward()
    parameters = dict(policy.named_parameters())
    # Preserve all reached gradients even if clipping rejects nonfinite values.
    np.savez(directory / "gradients_before_clip.npz",
             **{key: parameters[name].grad.detach().numpy().copy() for key, name in PARAMETERS.items()},
             reg_gradient=reg_gradient, **run_identity)
    pre, post, norm, coefficient = clip_gradients(policy)
    phase("optimizer_step")
    optimizer.step()  # The only optimizer step in this experiment.
    after = snapshot(policy)
    require(all(np.isfinite(value).all() for value in after.values()), "Nonfinite updated parameters")
    np.savez(directory / "update.npz", **{f"before_{key}": value for key, value in before.items()},
             **{f"after_{key}": value for key, value in after.items()},
             **{f"pre_clip_{key}": value for key, value in pre.items()},
             **{f"post_clip_{key}": value for key, value in post.items()}, reg_gradient=reg_gradient,
             **run_identity)
    optimizer_steps = {key: float(optimizer.state[parameters[name]]["step"]) for key, name in PARAMETERS.items()}
    details = {"optimizer": PROTOCOL["optimizer"], "optimizer_steps": optimizer_steps,
               "gradient_total_norm": norm, "clip_coefficient": coefficient, "training_losses": training_losses,
               "branches": {"baseline": baseline_record}}
    # Free the entire 100 ms autograd graph before further forward branches.
    del actions, target_tensor, task, loss, reg, optimizer, parameters
    gc.collect()
    with torch.no_grad():
        value, details["branches"]["updated"] = response(policy, frames, targets, "updated", directory, run_identity, phase)
        del value
        phase("checkpoint_save")
        checkpoint = save_policy_checkpoint(policy, directory / "checkpoint", run_identity=run_identity)
        details["checkpoint_id"] = checkpoint["checkpoint_id"]
        policy.brain.theta.zero_()
        value, details["branches"]["acute_restore"] = response(policy, frames, targets, "acute_restore", directory,
                                                               run_identity, phase)
        del value
        policy.readout.weight.copy_(torch.from_numpy(before["readout_weight"]))
        policy.readout.bias.copy_(torch.from_numpy(before["readout_bias"]))
        value, details["branches"]["full_restore"] = response(policy, frames, targets, "full_restore", directory,
                                                              run_identity, phase)
        del value
    return details  # JSON/NumPy-independent metadata only; never old model/tensor references.


def run_worker(directory):
    directory = Path(directory)
    started = time.monotonic()
    phase_start, current = started, None
    durations = {}
    result = {"schema": RESULT_SCHEMA, "status": "failed", "phases_seconds": durations}

    def phase(name, **details):
        nonlocal phase_start, current
        now = time.monotonic()
        if name != current:
            if current is not None:
                durations[current] = durations.get(current, 0) + now - phase_start
            current, phase_start = name, now
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": now - started, **details})

    try:
        phase("validate_registration")
        registration = read_json(directory / "registration.json")
        identity = validate_registration(registration)
        result.update(identity, implementation_sha256=registration["implementation_sha256"])
        require(sha256(registration["graph"]["path"]) == registration["graph"]["file_sha256"], "Graph file changed before run")
        selection, scene = select_data(registration["dataset_path"])
        np.savez(directory / "selection.npz", **selection, **identity)
        result["scene"] = scene
        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.set_num_interop_threads(PROTOCOL["torch_interop_threads"])
        result["environment"] = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                 "torch": torch.__version__, "psutil": psutil.__version__, "device": "cpu",
                                 "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}
        phase("load_policy")
        policy = NeuralPolicy.from_reference(registration["graph"]["path"], background=PROTOCOL["background"],
                    config=NeuralPolicyConfig(**PROTOCOL["config"]), camera=TargetCamera(**PROTOCOL["camera"]),
                    drive_config=SpatialDriveConfig(**PROTOCOL["drive_config"]),
                    lif_config=TrainableLIFConfig(**PROTOCOL["lif_config"]), dtype=torch.float32)
        # Apply the same strict lightweight production identity checks before computing.
        from .joint_probe_identity import validate_policy_identity
        validate_policy_identity(policy.identity, registration)
        write_json(directory / "policy_identity.json", policy.identity)
        np.savez(directory / "dn_ids.npz", body_ids=policy.mapping.outputs.body_ids.copy(), **identity)
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        result.update(update_and_intervene(policy, selection["frames"], selection["teacher_target"], directory, identity, phase))
        phase("verify_raw_graph")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        # Nothing containing a graph or a parameter survives this boundary.
        del policy
        gc.collect()
        phase("checkpoint_load")
        reloaded = load_policy_checkpoint(directory / "checkpoint", registration["graph"]["path"])
        require(reloaded.identity == read_json(directory / "policy_identity.json"), "Reload policy identity differs")
        with torch.no_grad():
            value, result["branches"]["checkpoint_reload"] = response(reloaded, selection["frames"], selection["teacher_target"],
                                                                      "checkpoint_reload", directory, identity, phase)
            del value
        result["raw_counts_sha256_reload"] = raw_counts_digest(reloaded.brain.graph)
        del reloaded
        gc.collect()
        phase("seal_artifacts")
        from .joint_probe_evidence import ARTIFACTS
        result["artifacts_sha256"] = {name: sha256(directory / name) for name in ARTIFACTS}
        result.update(status="completed", reason="five_branches_recorded")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        phase("finished")
        result["wall_seconds"] = time.monotonic() - started
        write_json(directory / "worker_result.json", result)
    return result
