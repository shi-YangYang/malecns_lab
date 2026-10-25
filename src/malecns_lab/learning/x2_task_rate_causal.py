"""Eight real responses: 150/300 Hz task input at two authenticated fixed states."""
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
from . import x2_interaction_causal as prior

SCHEMA = "malecns_lab.x2_task_rate_causal.v1"
PREFIX = "x2_behavior_short_trial_task_rate_"
CAUSAL_ID = "baf2add5e9fa9058a36b92b304da79328b6456f8529d1a16c3ae3e52c54edefc"
COMPARISON_SHA256 = "6e1be9d075abbf05540bb2367836a6b0a08c26521a3a40703bcb8684aa678a65"
CAUSAL_FILES = ("registration.json", "summary.json", "comparison.json", "observations.json", "responses.npz",
                "initial-state.npz", "after_two_frames-state.npz")
SOURCE_FILES = (*prior.SOURCE_FILES, "learning/x2_task_rate_policy.py", "learning/x2_task_rate_causal.py")
PATH_BASELINE_ID = "9e070c12ac40b75744effa015922580a8e6c6cf82328371ba61ad5119b32bf01"
PATH_BASELINE_COMPARISON = "443eff449e562ce3a1f296cf8a195c1c113d522791600f48425881315dad0944"


def configuration(mode="rate"):
    if mode in ("task-path","task-polarity"):
        from .x2_task_path_probe import configuration as path_configuration
        return path_configuration(mode)
    frozen.require(mode == "rate", "Unknown causal diagnostic mode")
    return {"schema": SCHEMA, "wall_seconds": 180, "total_frames": 8, "context_replay_frames": 0,
        "conditions_hz": [150, 300], "contexts": list(prior.CONTEXTS), "tasks": list(prior.TASKS[:2]),
        "decision_steps": 500, "graph_nodes": 166700, "graph_edges": 25582938,
        "intervention": "only five existing task-channel event rates; occupancy and all other inputs unchanged",
        "prestates": "complete authenticated causal-r1 arrays; no synthetic state or new preconditioning",
        "parameters": "all unchanged; no fitting or selection", "normalization": "archived train-only statistics",
        "exploratory": "DN spike counts in the same two 250-step halves; never replace voltage acceptance",
        "continuation_screen": {"both_contexts_raw_and_standardized_task_rms_strictly_increase": True,
            "standardized_task_cosine_min": .2, "meaning": "preregistered engineering screen, not biological standard",
            "pass": "supports considering a separate 54-frame learning trial only",
            "fail": "no follow-up training or gain sweep on this hypothesis"},
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "locked_test_opened": False, "automatic_deployment": False,
        "boundary": "known-context diagnostic only; sequence and actual-click acceptance still required"}


def output_directory(output, root, *, mode="rate", **kwargs):
    prefix = PREFIX
    if mode == "task-path":
        from .x2_task_path_probe import PREFIX as prefix
    elif mode == "task-polarity":
        from .x2_task_path_probe import POLARITY_PREFIX as prefix
    else:
        frozen.require(mode == "rate", "Unknown causal diagnostic mode")
    frozen.require(Path(output).name.startswith(prefix) and len(Path(output).name) > len(prefix),
                   "Task-rate diagnostic requires its dedicated output prefix")
    return safe_output_directory(output, root, **kwargs)


def path_baseline(root, causal_run, metadata, causal_comparison, observations, archived):
    """Authenticate the already completed 150 Hz baseline, without rerunning it."""
    import numpy as np
    directory = root/"data/x2_behavior_short_trial_task_path_20261024_r1"
    frozen.require(directory.resolve() == directory and directory.is_dir(), "Local sealed task-path baseline required")
    registration = frozen.read_json(directory/"registration.json")
    frozen.bound(registration,"registration_id")
    frozen.require(registration["registration_id"] == PATH_BASELINE_ID
        and registration["root"] == str(root) and registration["output"] == str(directory)
        and registration["causal_run"] == str(causal_run)
        and registration["checkpoint_metadata"] == metadata
        and registration["configuration"] == configuration("task-path")
        and frozen.sha256(directory/"comparison.json") == PATH_BASELINE_COMPARISON,
        "Sealed baseline identity/configuration differs")
    summary = frozen.read_json(directory/"summary.json")
    worker = summary["worker"]
    frozen.require(summary["status"] == worker["status"] == "completed"
        and summary["registration_id"] == worker["registration_id"] == PATH_BASELINE_ID
        and worker["completed_frames"] == 8 and worker["parameters_unchanged"] and worker["graph_unchanged"]
        and worker["baseline_four_frames_exact"], "Baseline exact replay did not complete")
    comparison = frozen.read_json(directory/"comparison.json")
    frozen.bound(comparison["policy_identity"],"policy_id")
    # Registration carries the transport-safe summary; its canonical policy ID
    # still binds the complete identity preserved in the recorded comparison.
    frozen.require(all(comparison["policy_identity"].get(k) == v for k,v in metadata["policy"].items())
        and comparison["configuration"] == configuration("task-path") and comparison["baseline_four_frames_exact"]
        and frozen.sha256(directory/"responses.npz") == comparison["arrays_sha256"]
        and frozen.sha256(directory/"selection.json") == comparison["selection_sha256"]
        and frozen.canonical_hash(frozen.read_json(directory/"observations.json")["observations"])
            == frozen.canonical_hash([o.to_dict() for o in observations[:2]]), "Baseline recorded inputs/arrays changed")
    for index, context in enumerate(comparison["conditions"]):
        source = causal_comparison["contexts"][index]
        frozen.require(context["context"] == source["context"] and len(context["branches"]) == 2,
                       "Baseline context changed")
        for task, branch in enumerate(context["branches"]):
            frozen.require(branch["before"] == source["prestate"] and branch["prestate_unchanged"]
                and branch["frames"][0]["result_state"] == source["branches"][task]["result_state"]
                and branch["frames"][0]["raw"] == source["branches"][task]["raw"], "Baseline replay binding differs")
    with np.load(directory/"responses.npz",allow_pickle=False) as arrays:
        frozen.require(arrays["features_mv"].shape == (2,2,2,2628)
            and np.array_equal(arrays["features_mv"][:,:,0],archived["features_mv"][:,:2])
            and np.array_equal(arrays["fixed_train_center"],archived["fixed_train_center"])
            and np.array_equal(arrays["fixed_train_scale"],archived["fixed_train_scale"]), "Baseline features/normalization differ")
    names = ("registration.json","summary.json","comparison.json","responses.npz","selection.json","observations.json")
    return {"path":str(directory),"registration_id":PATH_BASELINE_ID,"parent_policy_id":metadata["policy"]["policy_id"],
        "file_sha256":{name:frozen.sha256(directory/name) for name in names},
        "four_first_50ms_responses_bound_to_causal_r1":True,"neural_frames_replayed":0}


def polarity_input_audit(encoder, observations, original):
    import numpy as np
    encoded = np.asarray([encoder.encode(o) for o in observations])
    expected = original.copy()
    ports = list(encoder.task_ports)
    expected[:,ports] = (1.-expected[:,ports])/4.
    frozen.require(encoded.shape == (3,3377) and np.array_equal(encoded,expected)
        and np.array_equal(encoded[0],encoded[2]) and np.all(encoded[:,ports].sum(1) == 60.),
        "Polarity changed unrelated inputs or nominal task occupancy")
    changed = np.flatnonzero(encoded[1] != encoded[0])
    channels = encoder.identity["channels"]
    allowed = [i for i,c in enumerate(encoder.channel_indices)
        if c in (channels.index("task.GREET"),channels.index("task.SHOW_FORWARD"))]
    frozen.require(changed.tolist() == allowed and len(changed) == 120
        and np.all(np.abs(encoded[1,changed]-encoded[0,changed]) == .25), "Task-only polarity contrast differs")
    return {"only_task_ports_changed_from_parent":True,"between_task_changed_ports":120,
        "between_task_delta_magnitude":.25,"nominal_task_occupancy_sum":60.,
        "actual_event_equality_assumed":False,"repeat_input_exact":True}, encoded


def validate_state_arrays(arrays, identity):
    """Every saved state component is mandatory, including all delay slots."""
    import numpy as np
    from .x2_behavior_policy import array_identity
    frozen.bound(identity, "state_id")
    keys = {"v", "g", "refractory_until", "phase", "event_counts", *(f"pending.{i:02d}" for i in range(18))}
    frozen.require(set(arrays) == set(identity["arrays"]) == keys, "Missing or extra archived state arrays")
    for name, value in arrays.items():
        shape = (3377,) if name in ("phase", "event_counts") else (166700,)
        dtype = np.int64 if name in ("refractory_until", "event_counts") else np.float32
        frozen.require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
            and array_identity(value) == identity["arrays"][name], f"Archived state array differs: {name}")
    frozen.require(np.all((arrays["phase"] >= 0) & (arrays["phase"] < 1))
        and np.all((arrays["event_counts"] >= 0) & (arrays["event_counts"] <= identity["neural_step"])),
        "Invalid archived sensory state")
    return arrays


def source_inputs(causal_run, root):
    """Bind the sealed causal artifact to its original candidate, states and inputs."""
    import numpy as np
    from ..control.x2_behavior import Observation
    original = frozen.read_json(causal_run / "registration.json")
    frozen.bound(original, "registration_id")
    frozen.require(original["registration_id"] == CAUSAL_ID
        and frozen.sha256(causal_run / "comparison.json") == COMPARISON_SHA256, "Expected sealed causal-r1 artifact")
    summary = frozen.read_json(causal_run / "summary.json")
    frozen.require(summary["status"] == summary["worker"]["status"] == "completed"
        and summary["registration_id"] == summary["worker"]["registration_id"] == CAUSAL_ID
        and summary["worker"]["completed_frames"] == 8 and summary["worker"]["parameters_unchanged"]
        and summary["worker"]["graph_unchanged"] and summary["worker"]["repeat_controls_exact"],
        "Source causal run did not complete its exact controls")
    source_run = Path(original["source_run"])
    frozen.require(source_run.resolve() == source_run and source_run.is_relative_to(root / "data"),
                   "Original candidate must remain local and not symbolic")
    for name, expected in original["input_sha256"].items():
        frozen.require(frozen.sha256(source_run / name) == expected, "Source candidate asset changed")
    episode, metadata = prior.source_inputs(source_run)
    frozen.require(metadata == original["checkpoint_metadata"]
        and metadata["checkpoint_id"] == summary["worker"]["checkpoint_id"], "Source checkpoint changed")
    comparison = frozen.read_json(causal_run / "comparison.json")
    frozen.require(comparison["configuration"] == original["configuration"] == prior.configuration()
        and comparison["prefix_and_recorded_greet_dn_exact"]
        and frozen.sha256(causal_run / "responses.npz") == comparison["arrays_sha256"], "Causal responses changed")
    observations = prior.branch_observations(Observation.from_dict(episode["frames"][2]["observation"]))
    frozen.require(frozen.canonical_hash(frozen.read_json(causal_run / "observations.json")["observations"])
        == frozen.canonical_hash([o.to_dict() for o in observations]),
                   "Archived task-only observations differ from the original failure")
    states = []
    for i, context in enumerate(prior.CONTEXTS):
        record = comparison["contexts"][i]
        identity = record["prestate"]
        frozen.require(record["context"] == context and record["repeat_dn_state_logits_exact"]
            and identity["policy_id"] == metadata["policy"]["policy_id"]
            and identity["session_id"] == episode["episode_id"]
            and identity["frame_index"] == (0, 2)[i] and identity["neural_step"] == (600, 1600)[i]
            and identity["observation_seq"] == (-1, episode["frames"][1]["observation"]["observation_seq"])[i]
            and frozen.sha256(causal_run / f"{context}-state.npz") == record["state_npz_sha256"],
            "Archived context provenance or state file differs")
        with np.load(causal_run / f"{context}-state.npz", allow_pickle=False) as archive:
            arrays = validate_state_arrays({name: archive[name].copy() for name in archive.files}, identity)
        frozen.require(all(b["before"] == b["prestate_after"] == identity for b in record["branches"]),
                       "Source branches did not retain the same pre-state")
        states.append((identity, arrays))
    with np.load(causal_run / "responses.npz", allow_pickle=False) as archive:
        responses = {name: archive[name].copy() for name in archive.files}
    frozen.require(responses["features_mv"].shape == (2,3,2628)
        and responses["features_mv"].dtype == np.float32
        and np.isfinite(responses["features_mv"]).all()
        and np.array_equal(responses["features_mv"][:,0], responses["features_mv"][:,2]), "Invalid source response array")
    with np.load(source_run / "readout.npz", allow_pickle=False) as archive:
        frozen.require(np.array_equal(responses["fixed_train_center"], archive["center"])
            and np.array_equal(responses["fixed_train_scale"], archive["scale"]), "Fixed train statistics changed")
    return source_run, metadata, comparison, observations, states, responses


def restore_state(policy, identity, arrays):
    import torch
    from ..brain.trainable import LIFState
    from .x2_behavior_policy import BehaviorState
    validate_state_arrays(arrays, identity)
    frozen.require(policy.policy_id == identity["policy_id"] and policy.brain.delay_steps == 18,
                   "State must first restore under its original policy")
    values = {name: torch.from_numpy(value.copy()).to(policy.input_indices.device) for name,value in arrays.items()}
    brain = LIFState(values["v"], values["g"], values["refractory_until"],
        tuple(values[f"pending.{i:02d}"] for i in range(18)), identity["neural_step"], policy.brain.model_id)
    state = BehaviorState(brain, values["phase"], values["event_counts"], identity["frame_index"],
        identity["observation_seq"], identity["session_id"], identity["policy_id"])
    frozen.require(policy.state_identity(state) == identity, "Device restore changed archived state")
    return state


def compare_rates(features, center, scale):
    """Reuse fixed descriptive metrics; the repeated entry is not a new response."""
    import numpy as np
    frozen.require(np.shape(features) == (2,2,2,2628), "All eight real responses required")
    measurements, deltas = {}, {}
    for i, label in enumerate(("150_hz", "300_hz")):
        # prior metric's third entry is an unused repeat-control slot.
        values = np.concatenate((features[i], features[i,:,0:1]), axis=1)
        measurements[label], arrays = prior.delta_measurements(values, center, scale)
        deltas.update({label+"."+name: value for name,value in arrays.items()})
    improvements = {name: {context: measurements["300_hz"][name]["task_delta"][context]["rms"]
        > measurements["150_hz"][name]["task_delta"][context]["rms"] for context in prior.CONTEXTS}
        for name in ("raw_mv", "train_standardized")}
    cosine = measurements["300_hz"]["train_standardized"]["task_delta_cosine_between_contexts"]
    screen = {"task_rms_increased": improvements, "standardized_cosine": cosine,
        "supports_considering_54_frame_trial": all(all(values.values()) for values in improvements.values())
            and cosine is not None and cosine >= configuration()["continuation_screen"]["standardized_task_cosine_min"],
        "deployment_qualified": False}
    return measurements, deltas, screen


class DNSpikeCollector:
    """Accumulate the existing telemetry hook on-device; no per-step disk/CPU I/O."""
    def __init__(self, dn_indices, start_step):
        import torch
        self.indices, self.start_step = dn_indices, start_step
        self.counts = torch.zeros((2, len(dn_indices)), dtype=torch.int64, device=dn_indices.device)
        self.steps = 0

    def capture(self, spikes, step):
        offset = step-self.start_step
        frozen.require(offset == self.steps and 0 <= offset < 500, "Unexpected DN telemetry clock")
        self.counts[offset//250].add_(spikes[self.indices].to(self.counts.dtype))
        self.steps += 1

    def flush(self, step):
        pass


def execute(output, causal_run, *, root=frozen.ROOT, mode="rate"):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import source_inventory
    root, causal_run = Path(root).resolve(), Path(causal_run).absolute()
    frozen.require(causal_run.resolve() == causal_run and causal_run.is_relative_to(root / "data")
        and causal_run.is_dir(), "Archived local causal run required; no symbolic links")
    output = output_directory(output, root, mode=mode, inputs=(causal_run,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe complete-brain budget")
        source_run, metadata, comparison, observations, saved_states, archived = source_inputs(causal_run, root)
        baseline = (path_baseline(root,causal_run,metadata,comparison,observations,archived)
                    if mode == "task-polarity" else None)
        del comparison,observations,saved_states,archived
        output_directory(output, root, mode=mode, inputs=(causal_run, source_run))
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        sources.update({name:frozen.sha256(root / "src/malecns_lab" / name) for name in SOURCE_FILES})
        if mode in ("task-path","task-polarity"):
            name = "learning/x2_task_path_probe.py"
            sources[name] = frozen.sha256(root / "src/malecns_lab" / name)
        if mode == "task-polarity":
            name = "learning/x2_task_polarity_policy.py"
            sources[name] = frozen.sha256(root / "src/malecns_lab" / name)
        for name in sources:
            target = output / "sources/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "src/malecns_lab" / name, target)
            target.chmod(0o444)
        registration = {"configuration": configuration(mode), "root": str(root), "output": str(output),
            "causal_run": str(causal_run), "source_run": str(source_run), "parent_pid": os.getpid(),
            "deadline_monotonic": started+180, "checkpoint_metadata": metadata, "resources": resources,
            "source_sha256": sources, "causal_sha256": {name:frozen.sha256(causal_run / name) for name in CAUSAL_FILES}}
        if baseline is not None:
            registration["completed_path_baseline"] = baseline
        registration["registration_id"] = frozen.canonical_hash(registration)
        write_json(output / "registration.json", registration)
        write_json(output / "simulator_preflight.json", preflight)
        remaining = registration["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining > 0, "Preparation exhausted budget")
        supervisor = supervise([sys.executable, "-m", "malecns_lab.learning.x2_task_rate_causal", "--_worker", str(output)],
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
    root, causal_run, source_run = (Path(registration[key]) for key in ("root", "causal_run", "source_run"))
    mode = {"malecns_lab.x2_task_path_causal.v1":"task-path",
            "malecns_lab.x2_task_polarity_causal.v1":"task-polarity"}.get(registration["configuration"].get("schema"),"rate")
    output_directory(directory, root, mode=mode, created=True, inputs=(causal_run, source_run))
    frozen.require(registration["parent_pid"] == os.getppid() and registration["output"] == str(directory)
        and registration["configuration"] == configuration(mode), "Registered supervising parent/configuration required")
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
        for name, expected in registration["causal_sha256"].items():
            frozen.require(frozen.sha256(causal_run / name) == expected, "Archived causal input changed")
        for name, expected in frozen.read_json(causal_run / "registration.json")["input_sha256"].items():
            frozen.require(frozen.sha256(source_run / name) == expected, "Archived candidate input changed")
        if mode == "task-polarity":
            baseline = registration["completed_path_baseline"]
            for name, expected in baseline["file_sha256"].items():
                frozen.require(frozen.sha256(Path(baseline["path"])/name) == expected, "Completed baseline changed")
    try:
        audit_sources()
        actual_source, metadata, comparison, observations, saved_states, archived = source_inputs(causal_run, root)
        frozen.require(actual_source == source_run and metadata == registration["checkpoint_metadata"], "Registered source changed")
        baseline = (path_baseline(root,causal_run,metadata,comparison,observations,archived)
                    if mode == "task-polarity" else None)
        if baseline is not None:
            frozen.require(baseline == registration["completed_path_baseline"], "Registered baseline binding changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_policy import array_identity
        from .x2_behavior_training import hashlib_counts, _prediction
        from .x2_behavior_session import load_candidate_policy
        from .x2_task_rate_policy import bind_task_rate_policy
        sampler = MPSMemorySampler(directory / "mps_memory.csv", registration["resources"]["effective_rss_limit_bytes"])
        phase("strict_candidate_and_archived_state_restore")
        policy = load_candidate_policy(root, metadata, source_run / "checkpoint")
        policy.eval().requires_grad_(False)
        parameters = {name:array_identity(value.cpu().numpy()) for name,value in policy.named_parameters()}
        graph, original_identity = hashlib_counts(policy), policy.identity
        states = [restore_state(policy, identity, arrays) for identity, arrays in saved_states]
        del saved_states
        input_check, occupancy = prior.input_audit(policy.encoder, observations)
        frozen.require(np.array_equal(occupancy, archived["encoded_inputs"]), "Original occupancy differs from causal cache")
        observation_record = {"observations":[o.to_dict() for o in observations[:2]],"audit":input_check}
        expected_identity = original_identity
        if mode == "task-polarity":
            from .x2_task_polarity_policy import bind_task_polarity_policy
            bind_task_polarity_policy(policy)
            expected_identity = policy.identity
            states = [replace(state,policy_id=policy.policy_id) for state in states]
            observation_record["polarity_audit"], _ = polarity_input_audit(policy.encoder,observations,occupancy)
        write_json(directory / "observations.json", observation_record)
        if mode in ("task-path","task-polarity"):
            from .x2_task_path_probe import run_probe
            def completed():
                result["completed_frames"] += 1
            evidence = run_probe(policy,states,observations,archived,comparison,directory,
                                 registration["deadline_monotonic"],completed,phase,mode=mode,baseline_evidence=baseline)
            frozen.require(result["completed_frames"] == 8 and policy.identity == expected_identity
                and graph == hashlib_counts(policy)
                and parameters == {name:array_identity(value.cpu().numpy()) for name,value in policy.named_parameters()},
                "Task-path observation changed frame count, graph, policy or parameters")
            audit_sources()
            result.update(status="completed",parameters_unchanged=True,graph_unchanged=True,
                          checkpoint_id=metadata["checkpoint_id"],**evidence)
            return result
        features, spike_counts, evidence, identities = [], [], [], [original_identity]
        for condition in (150, 300):
            if condition == 300:
                bind_task_rate_policy(policy)
                identities.append(policy.identity)
                frozen.require(np.array_equal(occupancy, np.asarray([policy.encoder.encode(o) for o in observations])),
                               "Task-rate change altered occupancy")
            condition_features, condition_counts, contexts = [], [], []
            for index, (context, saved_state) in enumerate(zip(prior.CONTEXTS, states)):
                # This diagnostic grafts the very same numerical state into the
                # intervention identity; production session reset is untouched.
                state = saved_state if condition == 150 else replace(saved_state, policy_id=policy.policy_id)
                before = policy.state_identity(state)
                source_identity = comparison["contexts"][index]["prestate"]
                frozen.require(before["arrays"] == source_identity["arrays"], "Intervention changed fixed pre-state arrays")
                context_features, context_counts, branches = [], [], []
                for branch, observation in enumerate(observations[:2]):
                    phase("controlled_rate_task_branch", task_hz=condition, context=context, task=str(observation.task))
                    frozen.require(time.monotonic() < registration["deadline_monotonic"], "Wall budget exhausted")
                    frozen.require(policy.state_identity(state) == before, "Previous branch polluted fixed pre-state")
                    telemetry = DNSpikeCollector(policy.dn_indices, state.brain.step)
                    with torch.no_grad():
                        output = policy(observation, state, checkpoint_blocks=False, telemetry=telemetry)
                    torch.mps.synchronize()
                    frozen.require(telemetry.steps == 500, "Incomplete exploratory DN spike capture")
                    result["completed_frames"] += 1
                    actual = output.features_mv.cpu().numpy().copy()
                    raw = _prediction((output.behavior_logits, output.phase_logits, output.parameter_raw, output.intensity))
                    after = policy.state_identity(state)
                    result_state = policy.state_identity(output.state)
                    frozen.require(after == before and output.state.brain.step == state.brain.step+500,
                                   "Branch mutated its fixed pre-state or changed neural duration")
                    if condition == 150:
                        old = comparison["contexts"][index]["branches"][branch]
                        frozen.require(np.array_equal(actual, archived["features_mv"][index,branch])
                            and raw == old["raw"] and result_state == old["result_state"],
                            "Original 150 Hz DN, logits or final state failed exact causal replay")
                    context_features.append(actual)
                    context_counts.append(telemetry.counts.cpu().numpy().copy())
                    branches.append({"task": str(observation.task), "prestate_after": after,
                        "result_state": result_state, "raw": raw,
                        "advance_minus_hold": raw["behavior_logits"][1]-raw["behavior_logits"][0],
                        "start_minus_finish": raw["phase_logits"][0]-raw["phase_logits"][2]})
                    del output
                condition_features.append(context_features)
                condition_counts.append(context_counts)
                contexts.append({"context": context, "source_state_id": source_identity["state_id"],
                    "before": before, "source_arrays_exact": True, "branches": branches,
                    "forward_minus_greet_margin": {name:branches[1][name]-branches[0][name]
                        for name in ("advance_minus_hold", "start_minus_finish")}})
            features.append(condition_features)
            spike_counts.append(condition_counts)
            evidence.append({"task_rate_hz": condition, "policy_id": policy.policy_id, "contexts": contexts})
        features = np.asarray(features)
        spike_counts = np.asarray(spike_counts)
        measurements, deltas, screen = compare_rates(features, archived["fixed_train_center"], archived["fixed_train_scale"])
        spike_metrics = {}
        for i, label in enumerate(("150_hz", "300_hz")):
            values = spike_counts[i].reshape(2,2,2628)
            repeated = np.concatenate((values, values[:,0:1]), axis=1)
            spike_metrics[label] = prior.delta_measurements(repeated, np.zeros(2628), np.ones(2628))[0]["raw_mv"]
        with (directory / "responses.npz").open("xb") as stream:
            np.savez_compressed(stream, features_mv=features, encoded_occupancy=occupancy[:2],
                dn_spike_counts=spike_counts,
                intervention_rates_hz=np.asarray(policy.encoder.rates_hz, np.float32),
                fixed_train_center=archived["fixed_train_center"], fixed_train_scale=archived["fixed_train_scale"], **deltas)
        write_json(directory / "comparison.json", {"configuration": configuration(), "conditions": evidence,
            "policy_identities": identities, "measurements": measurements, "continuation_screen": screen,
            "exploratory_dn_spike_counts": {"units": "spikes per DN per 250-step half", "acceptance_use": False,
                "condition_axis_hz": [150,300], "axes": ["condition", "context", "task", "half", "DN"],
                "measurements": spike_metrics},
            "arrays_sha256": frozen.sha256(directory / "responses.npz"), "baseline_four_frames_exact": True})
        frozen.require(result["completed_frames"] == 8 and graph == hashlib_counts(policy)
            and parameters == {name:array_identity(value.cpu().numpy()) for name,value in policy.named_parameters()},
            "Frame count, graph or parameters changed")
        audit_sources()
        result.update(status="completed", reason="eight_fixed_state_rate_responses_measured", parameters_unchanged=True,
            graph_unchanged=True, baseline_four_frames_exact=True, checkpoint_id=metadata["checkpoint_id"],
            measurements=measurements, continuation_screen=screen)
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
    parser.add_argument("--mode", choices=("rate","task-path","task-polarity"), default="rate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--causal-run", type=Path,
        default=frozen.ROOT / "data/x2_behavior_short_trial_interaction_causal_20261024_r1")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode": "preview", "brain_loaded": False, "configuration": configuration(args.mode)}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output")
    try:
        result = execute(args.output, args.causal_run, mode=args.mode)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"], "reason": result["worker"].get("reason")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
