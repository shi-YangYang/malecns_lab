"""Worker-only policy loading and bounded no-grad neural closed-loop steps."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ..brain.metadata import canonical_hash
from ..control.body import BodyVelocity
from .evaluation_checkpoint import (context_for, inspect_checkpoint, restore_state, save_checkpoint, state_payload, validate_arrays)
from .evaluation_contract import (NEURAL, episode_payload, load_condition, parameter_identity, validate_suite)
from .evaluation_evidence import prefix_line, scan_episode, validate_resume
from .evaluation_rollout import baseline_action, environment_summary, step_record, vector
from .joint_probe_worker import raw_counts_digest
from .policy_checkpoint import load_policy_checkpoint
from .resource_guard import write_json
from .training_contract import PILOT, bytes_sha, read_json, require, sha256


def parameters(policy):
    return {"theta": policy.brain.theta.detach().numpy().copy(), "readout_weight": policy.readout.weight.detach().numpy().copy(),
            "readout_bias": policy.readout.bias.detach().numpy().copy()}


def apply_effective_parameters(policy, condition, expected):
    """Acute intervention is applied before initialization, never after it."""
    source = parameters(policy)
    require(np.array_equal(source["readout_weight"], expected["readout_weight"])
            and np.array_equal(source["readout_bias"], expected["readout_bias"]), "Loaded readout differs from sealed source")
    if condition["name"] == "acute_restore":
        with torch.no_grad():
            policy.brain.theta.copy_(torch.from_numpy(expected["theta"]))
    require(parameter_identity(policy.identity, parameters(policy)) == condition["parameter_identity"],
            "Loaded effective parameters differ")
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def digest_state(state, identity):
    if state is None:
        return None
    info, arrays = state_payload(state)
    return validate_arrays(info, arrays, identity, state.frame_index)


def record_resume(segment, reg, run, episode, policy, state, chain):
    info, arrays = state_payload(state) if policy is not None else (None, {})
    actual = parameters(policy) if policy is not None else None
    digest = validate_arrays(info, arrays, run["policy"], chain["next_frame"]) if policy is not None else None
    np.savez(segment / "resume.npz", **arrays, **({f"parameter.{key}": value for key, value in actual.items()} if actual else {}))
    write_json(segment / "resume.json", {"schema": "cnsbridge.evaluation_restored_state.v1", "segment_id": reg["segment_id"],
               "checkpoint_id": chain["last"]["checkpoint_id"] if chain["last"] else None, "next_frame": chain["next_frame"],
               "environment": environment_summary(chain["environment"], chain["next_frame"]), "state": info, "state_digest": digest,
               "parameter_identity": parameter_identity(run["policy"], actual) if actual else None,
               "arrays_sha256": sha256(segment / "resume.npz")})
    validate_resume(segment, reg, run, episode, chain["last"], chain["environment"], actual)


def execute_segment(segment, run, episode, condition, reg, chain, policy=None, state=None, *, phase=lambda *a, **k: None):
    """Pure orchestration entry used by explicitly numerical tests as well.

    Production construction occurs only in run_worker and requires full_reference.
    Neural forward receives exactly (read-only BGR, PolicyState).
    """
    environment = chain["environment"]
    require(not environment.done and chain["next_frame"] == reg["start_frame"], "Cannot continue wrong/terminal cursor")
    actual = parameters(policy) if policy is not None else None
    if policy is not None:
        require(parameter_identity(run["policy"], actual) == condition["parameter_identity"], "Policy parameter set differs")
        if chain["last"] is None:
            phase("scene_precondition")
            state = policy.initial_state()
        require(state is not None and state.frame_index == chain["next_frame"], "Restored frame differs")
    else:
        require(state is None and condition["name"] not in NEURAL, "Neural condition requires a real policy")
    record_resume(segment, reg, run, episode, policy, state, chain)
    prefix, parent = chain["prefix"], chain["last"]["checkpoint_id"] if chain["last"] else None
    number, next_frame = chain["chunks"], chain["next_frame"]
    (segment / "chunks").mkdir(exist_ok=False)
    end = next_frame + reg["frames"]
    with torch.no_grad():
        while next_frame < end and not environment.done:
            number += 1
            chunk = segment / "chunks" / f"{number:06d}"
            chunk.mkdir(exist_ok=False)
            start_frame, start_state = next_frame, digest_state(state, run["policy"])
            rows, images, times, features, raw_actions, executed = [], [], [], [], [], []
            durations = {"forward_seconds": 0., "environment_seconds": 0.}
            for _ in range(min(10, end - next_frame)):
                if environment.done:
                    break
                image = environment.frame()
                image.setflags(write=False)
                images.append(image.copy())
                times.append(environment.time_s)
                phase("forward", frame=next_frame)
                started = time.monotonic()
                if policy is not None:
                    output = policy.forward_tensor(image, state)
                    require(output.action.shape == (3,) and output.action.dtype == torch.float32
                            and bool(torch.isfinite(output.action).all()) and bool(torch.isfinite(output.features_mv).all()),
                            "Nonfinite or malformed neural action/features")
                    state = output.state
                    policy._drive_for(state)  # Includes full-state finite/range checks before any action executes.
                    vx, vz, wz = output.action.tolist()
                    raw = BodyVelocity(vx=vx, vz=vz, wz=wz)
                    features.append(output.features_mv.numpy().copy())
                else:
                    raw = baseline_action(condition["name"], image, environment)
                durations["forward_seconds"] += time.monotonic() - started
                started = time.monotonic()
                row = step_record(environment, raw, next_frame)
                durations["environment_seconds"] += time.monotonic() - started
                rows.append(row)
                raw_actions.append(vector(raw))
                executed.append(vector(BodyVelocity(**row["executed_action"])))
                next_frame += 1
            arrays = {"frames": np.stack(images), "indices": np.arange(start_frame, next_frame, dtype=np.int64),
                      "time_s": np.asarray(times, dtype=np.float64), "raw_actions": np.stack(raw_actions),
                      "executed_actions": np.stack(executed)}
            if policy is not None:
                arrays["features_mv"] = np.stack(features)
                require(parameter_identity(run["policy"], parameters(policy)) == condition["parameter_identity"],
                        "Parameters changed during evaluation")
            np.savez(chunk / "evidence.npz", **arrays)
            (chunk / "trajectory.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
            summary = environment_summary(environment, next_frame)
            event = {"schema": "cnsbridge.evaluation_event.v1", "evaluation_run_id": run["evaluation_run_id"],
                     "episode_id": episode["episode_id"], "segment_id": reg["segment_id"], "chunk": number,
                     "start_frame": start_frame, "end_frame": next_frame, "parent_checkpoint_id": parent,
                     "parameter_identity": condition["parameter_identity"], "start_state": start_state,
                     "end_state": digest_state(state, run["policy"]), "environment": summary,
                     "evidence_sha256": sha256(chunk / "evidence.npz"), "trajectory_sha256": sha256(chunk / "trajectory.jsonl"),
                     "durations": durations}
            write_json(chunk / "event.json", event)
            prefix += prefix_line(reg["segment"], number, sha256(chunk / "event.json"))
            (chunk / "prefix.jsonl").write_bytes(prefix)
            context = context_for(run, episode, next_frame, summary, parent, bytes_sha(prefix))
            phase("checkpoint_save", frame=next_frame)
            metadata = save_checkpoint(chunk / "checkpoint", run, episode, context, state=state, actual_parameters=actual)
            seal = {"schema": "cnsbridge.evaluation_chunk.v1", "evaluation_run_id": run["evaluation_run_id"],
                    "episode_id": episode["episode_id"], "segment_id": reg["segment_id"], "chunk": number,
                    "start_frame": start_frame, "end_frame": next_frame, "parent_checkpoint_id": parent,
                    "checkpoint_id": metadata["checkpoint_id"], "files_sha256": {name: sha256(chunk / name) for name in
                        ("evidence.npz", "trajectory.jsonl", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")},
                    "sealed_monotonic": time.monotonic()}
            seal["chunk_id"] = canonical_hash(seal)
            write_json(chunk / "chunk.json", seal)
            parent = metadata["checkpoint_id"]
            phase("chunk_sealed", frame=next_frame)
    if policy is not None:
        require(parameter_identity(run["policy"], parameters(policy)) == condition["parameter_identity"], "Final parameters changed")
    return state, next_frame, environment.done


def run_worker(segment):
    segment = Path(segment).resolve()
    episode_dir, suite_dir = segment.parents[1], segment.parents[4]
    started = time.monotonic()
    result = {"schema": "cnsbridge.evaluation_worker.v1", "status": "failed", "processed_frames": 0}
    def phase(name, **details):
        write_json(segment / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - started, **details})
    try:
        phase("validate_registration")
        run = validate_suite(suite_dir)
        episode = read_json(episode_dir / "episode.json")
        condition, expected = load_condition(suite_dir, run, episode["condition"])
        require(episode == episode_payload(run, condition, episode["scene_id"]), "Worker episode differs")
        reg = read_json(segment / "registration.json")
        result.update(segment_id=reg["segment_id"])
        chain = scan_episode(episode_dir, run, episode, expected, current_segment=reg["segment"])
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        policy, state = None, None
        if condition["name"] in NEURAL:
            phase("load_policy")
            policy = load_policy_checkpoint(Path(condition["source"]["checkpoint_path"]) / "policy", run["graph"]["path"])
            require(policy.identity == run["policy"] and policy.identity["role"] == "full_reference", "Actual full policy differs")
            apply_effective_parameters(policy, condition, expected)
            result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
            require(result["raw_counts_sha256_before"] == PILOT["raw_ordered_counts_sha256"], "Full graph changed")
            if chain["last"]:
                archive = inspect_checkpoint(chain["last"]["path"], run, episode, chain["last"]["context"], keep_arrays=True)
                state = restore_state(archive, policy)
                del archive
        state, end, terminal = execute_segment(segment, run, episode, condition, reg, chain, policy, state, phase=phase)
        if policy is not None:
            result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
            require(result["raw_counts_sha256_before"] == result["raw_counts_sha256_after"], "Evaluation changed graph")
        result.update(status="completed", reason="registered_frames_sealed", processed_frames=end - reg["start_frame"],
                      terminal=terminal, parameters_unchanged=True)
    except BaseException as error:
        result.update(reason=f"{type(error).__name__}:{error}",
                      failure_kind="io_or_interrupt" if isinstance(error, (OSError, KeyboardInterrupt)) else "numerical_or_protocol")
        traceback.print_exc()
    finally:
        phase("finished")
        result["wall_seconds"] = time.monotonic() - started
        write_json(segment / "worker_result.json", result)
    return result
