"""Eight-frame task-versus-neural-history intervention; no fitting or deployment.

Two archived observations build a real second pre-state. At each of two
pre-states, GREET / SHOW_FORWARD / GREET-repeat use one otherwise identical
observation. All responses are recomputed by the unchanged complete brain.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import x2_inference as frozen
from .x2_behavior_trial import MINIMUM_BUDGET, output_directory as safe_output_directory
from .x2_interaction_trial import OWN_SOURCES as INTERACTION_SOURCES

SCHEMA = "malecns_lab.x2_interaction_task_causal.v1"
PREFIX = "x2_behavior_short_trial_interaction_causal_"
TASKS = ("GREET", "SHOW_FORWARD", "GREET")
CONTEXTS = ("initial", "after_two_frames")
SOURCE_FILES = (*INTERACTION_SOURCES, "learning/x2_interaction_causal.py")
INPUT_FILES = ("registration.json", "summary.json", "dataset.json", "initial-state.json",
               "validation-features.npz", "readout.npz", "checkpoint/checkpoint.json", "checkpoint/arrays.npz")


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 180, "context_replay_frames": 2,
        "branch_frames": 6, "total_frames": 8, "contexts": list(CONTEXTS), "branches": list(TASKS),
        "decision_steps": 500, "precondition_steps": 600, "graph_nodes": 166700, "graph_edges": 25582938,
        "intervention": "task only; all other observation fields and each branch pre-state held fixed",
        "repeat_control": "third branch exactly repeats GREET from the identical pre-state",
        "parameters": "all unchanged; no fitting, optimization or hyperparameter selection",
        "normalization": "archived train-only statistics, used for measurement only",
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "locked_test_opened": False, "automatic_deployment": False,
        "boundary": "known failure context diagnostic; no independent validation or physical behavior claim"}


def output_directory(output, root, **kwargs):
    frozen.require(Path(output).name.startswith(PREFIX) and len(Path(output).name) > len(PREFIX),
                   "Causal diagnostic requires its dedicated output prefix")
    return safe_output_directory(output, root, **kwargs)


def branch_observations(observation):
    from ..control.x2_behavior import Task
    frozen.require(observation.task == Task.GREET and observation.request_changed
        and observation.feedback.valid and observation.feedback.heading_error_rad == 0.,
        "Expected the recorded non-heading GREET request")
    values = [replace(observation, task=Task(task)) for task in TASKS]
    original = observation.to_dict()
    frozen.require(all({**value.to_dict(), "task": original["task"]} == original for value in values),
                   "Task intervention changed another observation field")
    return values


def input_audit(encoder, observations):
    import numpy as np
    encoded = np.asarray([encoder.encode(value) for value in observations])
    identity = encoder.identity
    allowed_channels = {identity["channels"].index("task.GREET"), identity["channels"].index("task.SHOW_FORWARD")}
    allowed = [i for i,c in enumerate(identity["channel_indices"]) if c in allowed_channels]
    changed = np.flatnonzero(encoded[1] != encoded[0]).tolist()
    frozen.require(encoded.shape == (3, 3377) and len(allowed) == 120 and changed == allowed
        and np.array_equal(encoded[0], encoded[2])
        and np.all(np.abs(encoded[1, allowed]-encoded[0, allowed]) == 1.),
        "Encoded intervention must change exactly the two 60-port task channels")
    return {"changed_indices": changed, "changed_ports": len(changed),
            "only_task_channels_changed": True, "repeat_input_exact": True}, encoded


def delta_measurements(features, center, scale):
    """Descriptive effect sizes; no learned transform or selection uses responses."""
    import numpy as np
    values = np.asarray(features, np.float64)
    frozen.require(values.shape == (2, 3, 2628) and np.isfinite(values).all()
        and np.shape(center) == np.shape(scale) == (2628,) and np.isfinite(center).all()
        and np.isfinite(scale).all() and (scale >= .1).all(), "Invalid actual neural responses or fixed statistics")
    def norm(value):
        return {"rms": float(np.sqrt(np.mean(value*value))), "l2": float(np.linalg.norm(value)),
                "max_abs": float(np.max(np.abs(value))), "nonzero": int(np.count_nonzero(value))}
    def cosine(a,b):
        denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
        return float(np.dot(a,b)/denominator) if denominator else None
    result = {}
    arrays = {}
    for name, matrix in (("raw_mv", values), ("train_standardized", (values-center)/scale)):
        task = matrix[:, 1]-matrix[:, 0]
        state = matrix[1, :2]-matrix[0, :2]
        task_rms = float(np.sqrt(np.mean(task*task)))
        state_rms = float(np.sqrt(np.mean(state*state)))
        result[name] = {"task_delta": {context:norm(task[i]) for i,context in enumerate(CONTEXTS)},
            "state_delta": {task_name:norm(state[i]) for i,task_name in enumerate(TASKS[:2])},
            "task_delta_cosine_between_contexts": cosine(task[0],task[1]),
            "state_delta_cosine_between_tasks": cosine(state[0],state[1]),
            "pooled_task_to_state_rms_ratio": task_rms/state_rms if state_rms else None}
        arrays[name+".task_delta"] = task
        arrays[name+".state_delta"] = state
    return result, arrays


def source_inputs(source_run):
    """Authenticate the saved candidate and fixed known failure, without a brain."""
    from .x2_behavior_training import inspect_checkpoint
    from .x2_behavior_session import checkpoint_registration_metadata
    from .x2_interaction_policy import SCHEMA as POLICY_SCHEMA
    from .x2_interaction_trial import check_dataset
    original = frozen.read_json(source_run / "registration.json")
    frozen.bound(original, "registration_id")
    summary = frozen.read_json(source_run / "summary.json")
    dataset = check_dataset(frozen.read_json(source_run / "dataset.json"))
    checkpoint = inspect_checkpoint(source_run / "checkpoint")
    frozen.require(summary["status"] == "completed" and summary["registration_id"] == original["registration_id"]
        and summary["worker"]["completed_frames"] == 34
        and checkpoint["policy"]["schema"] == POLICY_SCHEMA
        and summary["worker"]["checkpoint_id"] == checkpoint["checkpoint_id"]
        and checkpoint["context"]["registration_id"] == original["registration_id"]
        and checkpoint["context"]["dataset_id"] == dataset["identity"]["dataset_id"] == original["dataset_id"]
        and frozen.sha256(source_run / "dataset.json") == original["dataset_sha256"]
        and frozen.sha256(source_run / "readout.npz") == checkpoint["context"]["readout_arrays_sha256"],
        "Saved candidate, dataset or train normalization differs")
    episode = dataset["splits"]["validation"][0]
    frozen.require([f["observation"]["task"] for f in episode["frames"][:3]] == ["FACE_RIGHT", "FACE_RIGHT", "GREET"],
                   "Registered failure prefix changed")
    return episode, checkpoint_registration_metadata(checkpoint)


def execute(output, source_run, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import source_inventory
    root, source_run = Path(root).resolve(), Path(source_run).absolute()
    frozen.require(source_run.resolve() == source_run and source_run.is_relative_to(root / "data")
        and source_run.is_dir(), "Original local source run required; no symbolic links")
    output = output_directory(output, root, inputs=(source_run,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe complete-brain budget")
        _, metadata = source_inputs(source_run)
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        sources.update({name:frozen.sha256(root / "src/malecns_lab" / name) for name in SOURCE_FILES})
        for name in sources:
            target = output / "sources/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "src/malecns_lab" / name, target)
            target.chmod(0o444)
        registration = {"configuration": configuration(), "root": str(root), "output": str(output),
            "source_run": str(source_run), "parent_pid": os.getpid(), "deadline_monotonic": started+180,
            "checkpoint_metadata": metadata, "resources": resources, "source_sha256": sources,
            "input_sha256": {name:frozen.sha256(source_run / name) for name in INPUT_FILES}}
        registration["registration_id"] = frozen.canonical_hash(registration)
        write_json(output / "registration.json", registration)
        write_json(output / "simulator_preflight.json", preflight)
        remaining = registration["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining > 0, "Preparation exhausted budget")
        supervisor = supervise([sys.executable, "-m", "malecns_lab.learning.x2_interaction_causal", "--_worker", str(output)],
            output, rss_limit_bytes=resources["effective_rss_limit_bytes"], wall_seconds=remaining, reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
        result = {"status": "completed" if supervisor["status"] == worker.get("status") == "completed" else "failed",
            "registration_id": registration["registration_id"], "output": str(output), "supervisor": supervisor,
            "worker": worker, "wall_seconds": time.monotonic()-started}
        write_json(output / "summary.json", result)
        return result


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).absolute()
    registration = frozen.read_json(directory / "registration.json")
    frozen.bound(registration, "registration_id")
    root, source_run = Path(registration["root"]), Path(registration["source_run"])
    output_directory(directory, root, created=True, inputs=(source_run,))
    frozen.require(registration["parent_pid"] == os.getppid() and registration["output"] == str(directory)
        and registration["configuration"] == configuration(), "Registered supervising parent/configuration required")
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must own the shared complete-brain lock")
    with (directory / "worker.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    started, sampler = time.monotonic(), None
    result = {"status": "failed", "reason": None, "registration_id": registration["registration_id"],
              "completed_frames": 0, "required_frames": 8, "locked_test_opened": False, "automatic_deployment": False}
    def phase(name, **fields):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started,
            "completed_frames": result["completed_frames"], "required_frames": 8, **fields})
    def audit_sources():
        for name, expected in registration["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected
                and frozen.sha256(directory / "sources/malecns_lab" / name) == expected, "Registered source changed")
        for name, expected in registration["input_sha256"].items():
            frozen.require(frozen.sha256(source_run / name) == expected, "Archived input changed")
    try:
        audit_sources()
        episode, metadata = source_inputs(source_run)
        frozen.require(metadata == registration["checkpoint_metadata"], "Registered checkpoint changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from ..control.x2_behavior import Observation
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_policy import array_identity
        from .x2_behavior_training import hashlib_counts, _prediction
        from .x2_behavior_session import load_candidate_policy
        sampler = MPSMemorySampler(directory / "mps_memory.csv", registration["resources"]["effective_rss_limit_bytes"])
        phase("strict_candidate_restore")
        policy = load_candidate_policy(root, metadata, source_run / "checkpoint")
        policy.eval().requires_grad_(False)
        parameters = {name:array_identity(value.cpu().numpy()) for name,value in policy.named_parameters()}
        graph = hashlib_counts(policy)
        with np.load(source_run / "validation-features.npz", allow_pickle=False) as archive:
            prior = archive["features_mv"].copy()
        with np.load(source_run / "readout.npz", allow_pickle=False) as archive:
            center, scale = archive["center"].copy(), archive["scale"].copy()
        frozen.require(prior.shape == (10,2628) and prior.dtype == np.float32 and np.isfinite(prior).all(), "Invalid archived DN")
        phase("fixed_blank_precondition")
        with torch.no_grad():
            initial = policy.initial_state(episode["episode_id"])
        initial_identity = policy.state_identity(initial)
        frozen.require(initial_identity["arrays"] == frozen.read_json(source_run / "initial-state.json")["arrays"],
                       "Candidate initial neural state differs from the source run")
        def compute(observation, state):
            frozen.require(time.monotonic() < registration["deadline_monotonic"], "Wall budget exhausted")
            with torch.no_grad():
                output = policy(observation, state, checkpoint_blocks=False)
            torch.mps.synchronize()
            result["completed_frames"] += 1
            frozen.require(output.state.brain.step == state.brain.step+500, "Decision neural clock changed")
            return output
        history, replay = initial, []
        for i, frame in enumerate(episode["frames"][:2]):
            phase("replay_context", frame_index=i)
            output = compute(Observation.from_dict(frame["observation"]), history)
            actual = output.features_mv.cpu().numpy().copy()
            replay.append(actual)
            frozen.require(np.array_equal(actual, prior[i]), "Real prefix replay differs from archived neural response")
            history = output.state
        frozen.require(policy.state_identity(initial) == initial_identity, "Context replay mutated the initial pre-state")
        observations = branch_observations(Observation.from_dict(episode["frames"][2]["observation"]))
        input_check, inputs = input_audit(policy.encoder, observations)
        write_json(directory / "observations.json", {"observations": [o.to_dict() for o in observations], "audit": input_check})
        features, evidence = [], []
        for context, state in zip(CONTEXTS, (initial, history)):
            prestate = policy.state_identity(state)
            with (directory / f"{context}-state.npz").open("xb") as stream:
                np.savez_compressed(stream, **policy.state_arrays(state))
            context_features, branches = [], []
            for index, observation in enumerate(observations):
                phase("controlled_task_branch", context=context, branch=index, task=str(observation.task))
                before = policy.state_identity(state)
                frozen.require(before == prestate, "Previous branch polluted the fixed pre-state")
                output = compute(observation, state)
                context_features.append(output.features_mv.cpu().numpy().copy())
                raw = _prediction((output.behavior_logits, output.phase_logits, output.parameter_raw, output.intensity))
                after = policy.state_identity(state)
                frozen.require(after == before, "Forward mutated the branch pre-state")
                branches.append({"task": str(observation.task), "before": before, "prestate_after": after,
                    "result_state": policy.state_identity(output.state), "raw": raw,
                    "advance_minus_hold": raw["behavior_logits"][1]-raw["behavior_logits"][0],
                    "start_minus_finish": raw["phase_logits"][0]-raw["phase_logits"][2]})
            frozen.require(np.array_equal(context_features[0],context_features[2])
                and branches[0]["result_state"] == branches[2]["result_state"]
                and branches[0]["raw"] == branches[2]["raw"], "Identical-input repeat control differs")
            features.append(context_features)
            evidence.append({"context": context, "prestate": prestate, "state_npz_sha256": frozen.sha256(directory / f"{context}-state.npz"),
                "repeat_dn_state_logits_exact": True, "branches": branches})
        features = np.asarray(features)
        frozen.require(np.array_equal(features[1,0],prior[2]), "Controlled recorded GREET differs from archived failure response")
        measurements, deltas = delta_measurements(features,center,scale)
        with (directory / "responses.npz").open("xb") as stream:
            np.savez_compressed(stream, features_mv=features, replay_features_mv=np.array(replay), encoded_inputs=inputs,
                fixed_train_center=center, fixed_train_scale=scale, **deltas)
        write_json(directory / "comparison.json", {"configuration": configuration(), "input_audit": input_check,
            "contexts": evidence, "measurements": measurements, "arrays_sha256": frozen.sha256(directory / "responses.npz"),
            "prefix_and_recorded_greet_dn_exact": True})
        frozen.require(result["completed_frames"] == 8 and graph == hashlib_counts(policy)
            and parameters == {name:array_identity(value.cpu().numpy()) for name,value in policy.named_parameters()},
            "Neural computation count, graph or parameters changed")
        audit_sources()
        result.update(status="completed", reason="eight_controlled_frames_measured", parameters_unchanged=True,
            graph_unchanged=True, repeat_controls_exact=True, prefix_and_recorded_greet_dn_exact=True,
            checkpoint_id=metadata["checkpoint_id"], measurements=measurements)
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if sampler is not None:
            result["mps_memory"] = sampler.close()
        result["wall_seconds"] = time.monotonic()-started
        write_json(directory / "worker_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-run", type=Path, default=frozen.ROOT / "data/x2_behavior_short_trial_interaction_20261023_r1")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode": "preview", "brain_loaded": False, "configuration": configuration()}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output")
    try:
        result = execute(args.output,args.source_run)
    except Exception as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"], "reason": result["worker"].get("reason")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
