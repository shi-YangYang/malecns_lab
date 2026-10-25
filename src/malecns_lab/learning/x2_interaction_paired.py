"""One 54-frame paired-task readout fit with an unseen continuous session.

Preview is stdlib-only. Execution requires an explicit fresh output and runs a
single supervised complete brain. It never selects a desktop configuration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import x2_inference as frozen
from .x2_behavior_trial import MINIMUM_BUDGET, output_directory as trial_output_directory
from .x2_interaction_trial import score_requests
from .x2_interaction_paired_policy import SCHEMA as POLICY_SCHEMA, ALPHA, bind_paired_policy

SCHEMA = "malecns_lab.x2_interaction_paired_trial.v1"
PREFIX = "x2_behavior_short_trial_interaction_paired_"
EXPECTED_FRAMES = {"train": 24, "validation": 12}
PAIR_FRAMES = (0, 4, 8)
PAIR_TASKS = ("SHOW_FORWARD", "END_INTERACTION", "GREET")
VALIDATION_ORDER = (("SHOW_FORWARD", False), ("END_INTERACTION", False), ("GREET", False),
    ("FACE_LEFT", False), ("SHOW_FORWARD", False), ("FACE_RIGHT", False),
    ("SHOW_FORWARD", True), ("END_INTERACTION", False))
VALIDATION_FAMILY = {"yaw": .67, "source_age": .055, "turn": .07, "inference_s": 7.6}
OWN_SOURCES = ("learning/x2_interaction_paired.py", "learning/x2_interaction_paired_policy.py", "learning/x2_interaction_trial.py", "learning/x2_interaction_policy.py",
    "learning/x2_interaction_data.py", "control/x2_interaction.py", "control/x2_skill_profile.py",
    "control/x2_skills.py", "control/x2.py",
    "senses/x2_relative_heading.py", "learning/x2_behavior_session.py", "learning/x2_behavior_telemetry.py",
    "learning/x2_telemetry.py", "learning/x2_yaw_diagnostic.py",
    "learning/x2_behavior_trial.py", "learning/x2_behavior_trial_policy.py")


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 600, "frames": EXPECTED_FRAMES,
        "sessions": {"train": 2, "validation": 1}, "decision_steps": 500, "precondition_steps": 600,
        "graph_nodes": 166700, "graph_edges": 25582938, "theta": "frozen source checkpoint",
        "paired_frames": 18, "train_fit_frames": 42, "total_neural_frames": 54,
        "pair_context_frames_per_train_session": list(PAIR_FRAMES), "pair_tasks": list(PAIR_TASKS),
        "validation_order": [list(item) for item in VALIDATION_ORDER], "validation_family": VALIDATION_FAMILY,
        "validation_requests": {"total": 8, "forward": 3, "end": 2},
        "readout": {"method": "ridge", "alpha": ALPHA, "fits": 1, "normalization_split": "42 train frames only",
                    "std_floor_mv": .1, "hyperparameter_selection": False},
        "state": "reset only at session start; request changes retain state; gate polls never enter brain",
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "coverage_limits": ["simulated feedback only; fixed observations do not react to candidate mistakes", "no WAVE capability", "no ordinary CONTINUE samples",
                            "only forward-to-END interruption; arbitrary switching not qualified"],
        "locked_test_opened": False, "qualification_only": True, "automatic_deployment": False}


def output_directory(output, root, **kwargs):
    frozen.require(Path(output).name.startswith(PREFIX) and len(Path(output).name) > len(PREFIX),
                   "Interaction trial requires its dedicated output prefix")
    return trial_output_directory(output, root, **kwargs)


def check_dataset(dataset):
    from .x2_interaction_data import validate_dataset
    validate_dataset(dataset)
    for split, required in EXPECTED_FRAMES.items():
        episodes = dataset["splits"][split]
        frozen.require(len(episodes) == configuration()["sessions"][split]
            and sum(len(e["frames"]) for e in episodes) == required, "Registered complete session count changed")
    return dataset


def paired_dataset(original):
    from .x2_interaction_data import _session
    from ..control.x2_behavior import digest
    value = deepcopy(original)
    value["splits"]["validation"] = [_session("validation", "paired-validation", value["identity"]["profile"], 241024,
        order=VALIDATION_ORDER, family=VALIDATION_FAMILY)]
    identity = value["identity"]
    identity["source_dataset_id"] = identity.pop("dataset_id")
    identity["session_families"] = {"train": ["train-a", "train-b"], "validation": ["paired-validation"]}
    identity["generator_source_sha256"] = frozen.sha256(Path(__file__).with_name("x2_interaction_data.py"))
    identity["split_rule"] = "original complete train sessions plus paired branches; new fixed held-out order and feedback family"
    identity["paired_trial_sha256"] = frozen.sha256(Path(__file__))
    identity["split_sha256"] = {split:digest(rows) for split,rows in value["splits"].items()}
    identity["dataset_id"] = digest(identity)
    check_dataset(value)
    return value


def branch_observations(observation):
    from ..control.x2_behavior import Task, ExecutionStatus
    frozen.require(observation.request_changed and observation.feedback.valid and observation.feedback.standing
        and observation.feedback.status in (ExecutionStatus.IDLE, ExecutionStatus.COMPLETED), "Pair requires a standing new-request pre-state")
    base = replace(observation, request_id=observation.request_id+"-paired",
                   feedback=replace(observation.feedback, heading_error_rad=0.))
    frozen.require(base.request_id != base.feedback.request_id, "Pair must be a fresh request")
    return [replace(base, task=Task(task)) for task in PAIR_TASKS]


def pair_targets(observations):
    from .x2_behavior_data import teacher_step
    return [teacher_step(observation,{}) for observation in observations]


def fit_readout(train, targets):
    """Fixed alpha; callers provide exactly 42 train rows, never validation."""
    import numpy as np
    from .x2_yaw_diagnostic import BEHAVIORS, PHASES
    values = np.asarray(train,np.float64)
    frozen.require(values.shape == (42,2628) and len(targets) == 42 and np.isfinite(values).all(), "Exactly 42 real train rows required")
    center, scale = values.mean(0), np.maximum(values.std(0),.1)
    z = (values-center)/scale
    y = np.asarray([[float(t["behavior"] == b) for b in BEHAVIORS]+[float(t["phase"] == p) for p in PHASES] for t in targets])
    bias = y.mean(0)
    weight = z.T@np.linalg.solve(z@z.T+ALPHA*np.eye(len(z)),y-bias)
    active = np.asarray([t["behavior"] in ("ADVANCE","TURN_LEFT","TURN_RIGHT") and t["phase"] in ("START","CONTINUE") for t in targets])
    intensity = np.asarray([t["intensity"] for t in targets],float)[active]
    frozen.require(active.any() and ((intensity>0)&(intensity<1)).all(), "Finite active parameter targets required")
    yp = np.log(intensity/(1-intensity))
    zp, pcenter = z[active],z[active].mean(0)
    pbias = float(yp.mean())
    pweight = (zp-pcenter).T@np.linalg.solve((zp-pcenter)@(zp-pcenter).T+ALPHA*np.eye(len(zp)),yp-pbias)
    return {"center":center,"scale":scale,"weight":weight,"bias":bias,
            "parameter_weight":pweight,"parameter_bias":np.asarray(pbias-pcenter@pweight)}


def execute(output, source_run, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import inspect_checkpoint, source_inventory
    from .x2_behavior_session import checkpoint_registration_metadata
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    checkpoint = source_run / "checkpoint"
    frozen.require(source_run.is_relative_to(root / "data"), "Archived project run required")
    output = output_directory(output, root, inputs=(source_run,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe full-brain memory budget")
        from .x2_interaction_policy import SCHEMA as INTERACTION_SCHEMA
        metadata = inspect_checkpoint(checkpoint)
        source_reg = frozen.read_json(source_run / "registration.json")
        frozen.bound(source_reg,"registration_id")
        frozen.require(metadata["policy"]["schema"] == INTERACTION_SCHEMA
            and metadata["context"]["registration_id"] == source_reg["registration_id"]
            and metadata["context"]["dataset_id"] == source_reg["dataset_id"]
            and frozen.sha256(source_run / "dataset.json") == source_reg["dataset_sha256"], "Original interaction source differs")
        dataset = paired_dataset(frozen.read_json(source_run / "dataset.json"))
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        sources.update({name: frozen.sha256(root / "src/malecns_lab" / name) for name in OWN_SOURCES})
        for name in sources:
            target = output / "sources/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "src/malecns_lab" / name, target)
            target.chmod(0o444)
        write_json(output / "dataset.json", dataset)
        reg = {"configuration": configuration(), "root": str(root), "output": str(output), "parent_pid": os.getpid(),
            "deadline_monotonic": started+600, "source_run": str(source_run), "checkpoint": str(checkpoint), "checkpoint_id": metadata["checkpoint_id"],
            "source_input_sha256": {name:frozen.sha256(source_run/name) for name in ("registration.json","dataset.json","train-features.npz","initial-state.json")},
            "checkpoint_metadata": checkpoint_registration_metadata(metadata),
            "dataset_id": dataset["identity"]["dataset_id"], "dataset_sha256": frozen.sha256(output / "dataset.json"),
            "source_sha256": sources, "resources": resources}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output / "registration.json", reg)
        write_json(output / "simulator_preflight.json", preflight)
        remaining = reg["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining > 0, "Preparation exhausted budget")
        supervisor = supervise([sys.executable, "-m", "malecns_lab.learning.x2_interaction_paired", "--_worker", str(output)],
            output, rss_limit_bytes=resources["effective_rss_limit_bytes"], wall_seconds=remaining, reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
        result = {"status": "completed" if supervisor["status"] == worker.get("status") == "completed" else "failed",
            "output": str(output), "registration_id": reg["registration_id"], "supervisor": supervisor,
            "worker": worker, "wall_seconds": time.monotonic()-started}
        write_json(output / "summary.json", result)
        return result


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).absolute()
    frozen.require(not directory.is_symlink() and directory.resolve() == directory, "Worker output cannot use symbolic links")
    reg = frozen.read_json(directory / "registration.json")
    frozen.bound(reg, "registration_id")
    frozen.require(reg["parent_pid"] == os.getppid() and reg["output"] == str(directory), "Supervising parent required")
    root = Path(reg["root"])
    output_directory(directory, root, created=True, inputs=(reg["source_run"],))
    frozen.require(reg["configuration"] == configuration(), "Registered trial configuration changed")
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
    result = {"status": "failed", "reason": None, "registration_id": reg["registration_id"],
        "completed_frames": 0, "required_frames": 54,
        "qualification_only": True, "automatic_deployment": False, "locked_test_opened": False}
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started,
            "completed_frames": result["completed_frames"], "required_frames": result["required_frames"], **details})
    def audit_sources():
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected
                and frozen.sha256(directory / "sources/malecns_lab" / name) == expected, "Registered source changed")
        for name,expected in reg["source_input_sha256"].items():
            frozen.require(frozen.sha256(Path(reg["source_run"])/name) == expected,"Archived source input changed")
    try:
        audit_sources()
        frozen.require(frozen.sha256(directory / "dataset.json") == reg["dataset_sha256"], "Dataset bytes changed")
        dataset = check_dataset(frozen.read_json(directory / "dataset.json"))
        frozen.require(dataset["identity"]["dataset_id"] == reg["dataset_id"], "Dataset changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from ..control.x2_behavior import Observation
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_policy import array_identity
        from .x2_behavior_training import inspect_checkpoint, save_checkpoint, hashlib_counts, _prediction
        from .x2_behavior_session import checkpoint_registration_metadata, load_candidate_policy
        from .x2_interaction_policy import install_readout, verify_fold
        from .x2_yaw_diagnostic import budget_projection
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        phase("load_complete_graph")
        metadata = inspect_checkpoint(reg["checkpoint"])
        frozen.require(metadata["checkpoint_id"] == reg["checkpoint_id"], "Source checkpoint changed")
        frozen.require(checkpoint_registration_metadata(metadata) == reg["checkpoint_metadata"], "Registered candidate differs")
        policy = load_candidate_policy(root, reg["checkpoint_metadata"], reg["checkpoint"])
        policy.eval().requires_grad_(False)
        frozen_theta = array_identity(policy.brain.theta.cpu().numpy())
        counts = hashlib_counts(policy)
        write_json(directory / "policy.json", policy.identity)
        phase("fixed_blank_precondition")
        with torch.no_grad():
            template = policy.initial_state("interaction-template")
        initial = policy.state_identity(template)
        frozen.require(initial["arrays"] == frozen.read_json(Path(reg["source_run"])/"initial-state.json")["arrays"], "Source initial state differs")
        with (directory / "initial-state.npz").open("xb") as stream:
            np.savez_compressed(stream, **policy.state_arrays(template))
        write_json(directory / "initial-state.json", initial)
        features, before, state_evidence, timing = {}, {}, [], []
        paired_features, paired_inputs, paired_target, paired_before, pair_evidence = [], [], [], [], []
        compute_seconds = 0.
        for split, episodes in dataset["splits"].items():
            feature_rows, input_rows, baseline = [], [], []
            for episode in episodes:
                state = replace(template, session_id=episode["episode_id"])
                initial_session = policy.state_identity(state)
                frozen.require(initial_session["arrays"] == initial["arrays"], "Session reset state differs")
                boundaries = []
                for frame in episode["frames"]:
                    observation = Observation.from_dict(frame["observation"])
                    if split == "train" and frame["frame_index"] in PAIR_FRAMES:
                        shared_before = policy.state_identity(state)
                        observations = branch_observations(observation)
                        inputs = np.array([policy.encoder.encode(value) for value in observations])
                        channel_indices = np.asarray(policy.encoder.channel_indices)
                        channels = policy.encoder.identity["channels"]
                        for i in (1,2):
                            allowed = np.isin(channel_indices,[channels.index("task."+PAIR_TASKS[j]) for j in (0,i)])
                            frozen.require(np.array_equal(np.flatnonzero(inputs[0]!=inputs[i]),np.flatnonzero(allowed))
                                and np.count_nonzero(allowed)==120,"Pair changed non-task stimulation")
                        targets = pair_targets(observations)
                        pair_rows = []
                        for index,value in enumerate(observations):
                            phase("paired_task_response",family=episode["family_id"],frame_index=frame["frame_index"],task=str(value.task))
                            frozen.require(policy.state_identity(state)==shared_before,"Previous pair branch contaminated pre-state")
                            begin=time.monotonic()
                            with torch.no_grad():
                                branch=policy(value,state,checkpoint_blocks=False)
                            torch.mps.synchronize()
                            compute_seconds+=time.monotonic()-begin
                            frozen.require(policy.state_identity(state)==shared_before
                                and branch.state.brain.step==state.brain.step+500,"Pair mutated pre-state or neural clock")
                            paired_features.append(branch.features_mv.cpu().numpy().copy())
                            paired_inputs.append(inputs[index])
                            paired_target.append(targets[index])
                            raw=_prediction((branch.behavior_logits,branch.phase_logits,branch.parameter_raw,branch.intensity))
                            paired_before.append(raw)
                            pair_rows.append({"observation":value.to_dict(),"target":targets[index],"baseline":raw,
                                "prestate_before":shared_before,"prestate_after":policy.state_identity(state)})
                            result["completed_frames"]+=1
                        pair_evidence.append({"family":episode["family_id"],"frame_index":frame["frame_index"],"branches":pair_rows})
                    phase("complete_neural_response", split=split, family=episode["family_id"], frame_index=frame["frame_index"])
                    previous_step, previous_index = state.brain.step, state.frame_index
                    begin = time.monotonic()
                    with torch.no_grad():
                        output = policy(observation, state, checkpoint_blocks=False)
                    torch.mps.synchronize()
                    compute_seconds += time.monotonic()-begin
                    frozen.require(output.state.brain.step == previous_step+500
                        and output.state.frame_index == previous_index+1, "One consumed observation must advance exactly 500 steps")
                    feature_rows.append(output.features_mv.cpu().numpy().copy())
                    input_rows.append(policy.encoder.encode(observation))
                    baseline.append(_prediction((output.behavior_logits, output.phase_logits, output.parameter_raw, output.intensity)))
                    boundaries.append({"request_id": observation.request_id, "observation_seq": observation.observation_seq,
                        "request_changed": observation.request_changed, "frame_index": state.frame_index,
                        "neural_step_start": previous_step, "neural_step_end": output.state.brain.step,
                        "gate_only_polls_skipped": len(frame["gate_only_feedback"])})
                    state = output.state
                    result["completed_frames"] += 1
                    if result["completed_frames"] % 5 == 0:
                        projection = budget_projection(result["completed_frames"], compute_seconds,
                            max(0., reg["deadline_monotonic"]-time.monotonic()), total=result["required_frames"], reserve_seconds=35.)
                        timing.append(projection)
                        write_json(directory / "budget-projections.json", {"projections": timing})
                        if not projection["can_finish"]:
                            with (directory / f"partial-{split}-features.npz").open("xb") as stream:
                                np.savez_compressed(stream, features_mv=np.array(feature_rows))
                            raise RuntimeError("Projected complete sessions exceed budget; no shortened success")
                state_evidence.append({"session_id": episode["episode_id"], "initial_state": initial_session,
                    "final_state": policy.state_identity(state), "boundaries": boundaries})
            features[split], before[split] = np.array(feature_rows), baseline
            frozen.require(len(feature_rows) == EXPECTED_FRAMES[split], "Incomplete continuous session responses")
            with (directory / f"{split}-features.npz").open("xb") as stream:
                np.savez_compressed(stream, features_mv=features[split], encoded_inputs=np.array(input_rows))
        write_json(directory / "state-evidence.json", {"sessions": state_evidence})
        write_json(directory / "paired-observations.json", {"contexts":pair_evidence})
        frozen.require(len(paired_features)==18 and len(pair_evidence)==6,"Incomplete paired train contexts")
        with (directory / "paired-train-features.npz").open("xb") as stream:
            np.savez_compressed(stream,features_mv=np.array(paired_features),encoded_inputs=np.array(paired_inputs))
        with np.load(Path(reg["source_run"])/"train-features.npz",allow_pickle=False) as cached:
            frozen.require(np.array_equal(features["train"],cached["features_mv"]),"Original 24 train DN responses changed")
        frozen.require(result["completed_frames"] == result["required_frames"], "Incomplete full-brain collection")
        phase("fixed_readout_fit")
        targets = [f["target"] for e in dataset["splits"]["train"] for f in e["frames"]]+paired_target
        train_features=np.concatenate((features["train"],np.array(paired_features)))
        readout = fit_readout(train_features, targets)
        with (directory / "readout.npz").open("xb") as stream:
            np.savez_compressed(stream, **readout)
        install_readout(policy, readout)
        all_features = np.concatenate((train_features,features["validation"]))
        def read_rows(candidate):
            with torch.no_grad():
                return [_prediction(candidate.read_features(torch.from_numpy(row).to("mps"))) for row in all_features]
        def logits(rows):
            return np.asarray([r["behavior_logits"]+r["phase_logits"]+r["parameter_raw"] for r in rows])
        fitted = read_rows(policy)
        fold_audit, expected = verify_fold(readout, all_features, logits(fitted))
        write_json(directory / "fold-audit.json", fold_audit)
        with (directory / "logit-comparison.npz").open("xb") as stream:
            np.savez_compressed(stream, float64_ridge_logits=expected, folded_mps_logits=logits(fitted))
        frozen.require(fold_audit["passed"], "Folded float32 logits/argmax differ from fixed ridge; candidate rejected")
        frozen.require(frozen_theta == array_identity(policy.brain.theta.cpu().numpy())
            and hashlib_counts(policy) == counts and policy.state_identity(template) == initial, "Frozen brain or initial state changed")
        paired_correct=[row["behavior"]==target["behavior"] and row["phase"]==target["phase"]
            for row,target in zip(fitted[24:42],paired_target)]
        comparison={"baseline":{split:score_requests(dataset["splits"][split],before[split]) for split in EXPECTED_FRAMES},
            "candidate":{"train":score_requests(dataset["splits"]["train"],fitted[:24]),
                         "validation":score_requests(dataset["splits"]["validation"],fitted[42:])},
            "paired_train":{"matched_frames":sum(paired_correct),"frames":18,"all_correct":all(paired_correct),
                "rows":[{"target":target,"baseline":old,"candidate":row} for target,old,row in zip(paired_target,paired_before,fitted[24:42])]}}
        validation=comparison["candidate"]["validation"]
        greet=[r for r in validation["requests"] if r["task"]=="GREET"]
        validation["greet_false_advance_requests"]=sum(any(row["prediction"]["behavior"]=="ADVANCE" for row in r["raw"]) for r in greet)
        validation["greet_hold_finish_exact"]=bool(greet and all(r["raw_sequence_match"] for r in greet))
        validation["forward_end_and_greet_pass"]=bool(validation["forward_candidate_pass"] and validation["greet_hold_finish_exact"])
        write_json(directory / "comparison.json", comparison)
        result.update(theta_unchanged=True,graph_unchanged=True,paired_train_matched=sum(paired_correct),paired_train_frames=18)
        frozen.require(all(paired_correct),"Paired training failed to distinguish all 18 same-prestate tasks; no candidate saved")
        bind_paired_policy(policy)
        write_json(directory / "paired-policy.json",policy.identity)
        checkpoint = directory / "checkpoint"
        saved = save_checkpoint(checkpoint, policy, {"registration_id": reg["registration_id"],
            "dataset_id": reg["dataset_id"], "source_checkpoint_id": metadata["checkpoint_id"],
            "readout_arrays_sha256": frozen.sha256(directory / "readout.npz"),
            "fit_count": 1, "ridge_alpha": ALPHA, "train_frames":42,"paired_frames_exact":18,
            "optimizer_steps": 0, "theta_updates": 0, "qualification_only": True})
        # Release the only complete brain before rebuilding through the exact
        # desktop factory. No second neural rollout or parallel graph is needed.
        del branch, output, state, template, policy
        gc.collect()
        torch.mps.empty_cache()
        phase("strict_desktop_restore")
        restored = load_candidate_policy(root, checkpoint_registration_metadata(saved), checkpoint)
        restored.eval().requires_grad_(False)
        recovered = read_rows(restored)
        recovered_logits = logits(recovered)
        restore_audit, _ = verify_fold(readout, all_features, recovered_logits)
        restore_audit["saved_reload_logits_exact"] = bool(np.array_equal(recovered_logits, logits(fitted)))
        restore_audit["theta_exact"] = frozen_theta == array_identity(restored.brain.theta.cpu().numpy())
        write_json(directory / "restore-audit.json", restore_audit)
        frozen.require(restore_audit["passed"] and restore_audit["saved_reload_logits_exact"] and restore_audit["theta_exact"],
                       "Shared desktop restoration changed the fitted candidate")
        audit_sources()
        result.update(status="completed", reason="complete_sessions_and_strict_restoration_checked",
            neural_compute_seconds=compute_seconds, theta_unchanged=True, graph_unchanged=True,
            checkpoint_path=str(checkpoint), checkpoint_id=saved["checkpoint_id"], policy_id=saved["policy"]["policy_id"],
            validation={name: {k:v for k,v in value["validation"].items() if k not in ("episodes", "requests")}
                        for name, value in comparison.items() if name in ("baseline","candidate")},
            paired_train_exact=18, original_train_dn_exact=True, coverage_limits=configuration()["coverage_limits"])
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
        parser.error("--execute requires a new --output")
    try:
        result = execute(args.output, args.source_run)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"], "reason": result["worker"].get("reason")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
