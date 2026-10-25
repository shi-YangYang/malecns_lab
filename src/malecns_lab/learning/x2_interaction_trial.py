"""One bounded frozen-brain/ridge candidate on complete continuous sessions.

Preview is stdlib-only. Execution requires an explicit fresh output and runs a
single supervised complete brain. It never selects a desktop configuration.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "malecns_lab.x2_interaction_readout_trial.v1"
PREFIX = "x2_behavior_short_trial_interaction_"
EXPECTED_FRAMES = {"train": 24, "validation": 10}
OWN_SOURCES = ("learning/x2_interaction_trial.py", "learning/x2_interaction_policy.py",
    "learning/x2_interaction_data.py", "control/x2_interaction.py", "control/x2_skill_profile.py",
    "control/x2_skills.py", "control/x2.py",
    "senses/x2_relative_heading.py", "learning/x2_behavior_session.py", "learning/x2_behavior_telemetry.py",
    "learning/x2_telemetry.py", "learning/x2_yaw_diagnostic.py",
    "learning/x2_behavior_trial.py", "learning/x2_behavior_trial_policy.py")


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 600, "frames": EXPECTED_FRAMES,
        "sessions": {"train": 2, "validation": 1}, "decision_steps": 500, "precondition_steps": 600,
        "graph_nodes": 166700, "graph_edges": 25582938, "theta": "frozen source checkpoint",
        "readout": {"method": "ridge", "alpha": 1., "fits": 1, "normalization_split": "train",
                    "std_floor_mv": .1, "hyperparameter_selection": False},
        "state": "reset only at session start; request changes retain state; gate polls never enter brain",
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "coverage_limits": ["simulated feedback only", "no WAVE capability", "no ordinary CONTINUE samples",
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


def score_requests(episodes, rows):
    """Raw outputs are scored before any gate/protection; no motion claim."""
    from .x2_behavior_data import score_predictions
    frozen.require(sum(len(e["frames"]) for e in episodes) == len(rows), "Missing complete session predictions")
    predictions, requests, offset = {}, [], 0
    for episode in episodes:
        current = rows[offset:offset+len(episode["frames"])]
        predictions[episode["episode_id"]] = current
        offset += len(current)
        for request in episode["requests"]:
            start, end = request["first_frame"], request["end_frame"]
            frames, outputs = episode["frames"][start:end], current[start:end]
            matched = [f["target"]["behavior"] == p["behavior"] and f["target"]["phase"] == p["phase"]
                       for f, p in zip(frames, outputs)]
            requests.append({"session_id": episode["episode_id"], "request_id": request["request_id"],
                "task": request["task"], "frames": len(frames), "matched_frames": sum(matched),
                "raw_sequence_match": all(matched), "fixture_outcome": request["outcome"],
                "will_be_interrupted": request["will_be_interrupted"],
                "capability_unavailable": request["outcome"] == "capability_unavailable",
                "start_behavior": outputs[0]["behavior"], "start_phase": outputs[0]["phase"],
                "raw": [{"observation_seq": f["observation"]["observation_seq"], "target": f["target"],
                         "prediction": p} for f, p in zip(frames, outputs)]})
    forward = [r for r in requests if r["task"] == "SHOW_FORWARD"]
    ending = [r for r in requests if r["task"] == "END_INTERACTION"]
    scores = score_predictions(episodes, predictions)
    scores.update(requests=requests, request_sequence_matches=sum(r["raw_sequence_match"] for r in requests),
        request_total=len(requests), forward_requests=len(forward),
        forward_starts=sum(r["start_behavior"] == "ADVANCE" and r["start_phase"] == "START" for r in forward),
        forward_sequences=sum(r["raw_sequence_match"] for r in forward),
        end_requests=len(ending), end_sequences=sum(r["raw_sequence_match"] for r in ending),
        forward_candidate_pass=bool(forward and ending and all(r["raw_sequence_match"] for r in forward+ending)),
        greet_motion_qualified=False, continue_qualified=False,
        present_target_behaviors=sorted({f["target"]["behavior"] for e in episodes for f in e["frames"]}),
        present_target_phases=sorted({f["target"]["phase"] for e in episodes for f in e["frames"]}))
    return scores


def execute(output, checkpoint, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import inspect_checkpoint, source_inventory
    from .x2_behavior_session import checkpoint_registration_metadata
    from .x2_interaction_data import generate_dataset
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    frozen.require(checkpoint.is_relative_to(root / "data"), "Archived project checkpoint required")
    output = output_directory(output, root, inputs=(checkpoint,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe full-brain memory budget")
        metadata, dataset = inspect_checkpoint(checkpoint), check_dataset(generate_dataset())
        frozen.require(metadata["policy"]["schema"] == "malecns_lab.x2_behavior_policy.v1", "Original source checkpoint required")
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
            "deadline_monotonic": started+600, "checkpoint": str(checkpoint), "checkpoint_id": metadata["checkpoint_id"],
            "checkpoint_metadata": checkpoint_registration_metadata(metadata),
            "dataset_id": dataset["identity"]["dataset_id"], "dataset_sha256": frozen.sha256(output / "dataset.json"),
            "source_sha256": sources, "resources": resources}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output / "registration.json", reg)
        write_json(output / "simulator_preflight.json", preflight)
        remaining = reg["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining > 0, "Preparation exhausted budget")
        supervisor = supervise([sys.executable, "-m", "malecns_lab.learning.x2_interaction_trial", "--_worker", str(output)],
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
    output_directory(directory, root, created=True, inputs=(reg["checkpoint"],))
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
        "completed_frames": 0, "required_frames": sum(EXPECTED_FRAMES.values()),
        "qualification_only": True, "automatic_deployment": False, "locked_test_opened": False}
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started,
            "completed_frames": result["completed_frames"], "required_frames": result["required_frames"], **details})
    def audit_sources():
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected
                and frozen.sha256(directory / "sources/malecns_lab" / name) == expected, "Registered source changed")
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
        from .x2_interaction_policy import bind_interaction_policy, install_readout, verify_fold
        from .x2_yaw_diagnostic import fit_readout, budget_projection
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        phase("load_complete_graph")
        metadata = inspect_checkpoint(reg["checkpoint"])
        frozen.require(metadata["checkpoint_id"] == reg["checkpoint_id"], "Source checkpoint changed")
        frozen.require(checkpoint_registration_metadata(metadata) == reg["checkpoint_metadata"], "Registered candidate differs")
        policy = load_candidate_policy(root, reg["checkpoint_metadata"], reg["checkpoint"])
        bind_interaction_policy(policy, source_checkpoint_id=metadata["checkpoint_id"], profile=dataset["identity"]["profile"])
        policy.eval().requires_grad_(False)
        frozen_theta = array_identity(policy.brain.theta.cpu().numpy())
        counts = hashlib_counts(policy)
        write_json(directory / "policy.json", policy.identity)
        phase("fixed_blank_precondition")
        with torch.no_grad():
            template = policy.initial_state("interaction-template")
        initial = policy.state_identity(template)
        with (directory / "initial-state.npz").open("xb") as stream:
            np.savez_compressed(stream, **policy.state_arrays(template))
        write_json(directory / "initial-state.json", initial)
        features, before, state_evidence, timing = {}, {}, [], []
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
        frozen.require(result["completed_frames"] == result["required_frames"], "Incomplete full-brain collection")
        phase("fixed_readout_fit")
        targets = [f["target"] for e in dataset["splits"]["train"] for f in e["frames"]]
        readout = fit_readout(features["train"], targets)
        with (directory / "readout.npz").open("xb") as stream:
            np.savez_compressed(stream, **readout)
        install_readout(policy, readout)
        all_features = np.concatenate([features[split] for split in EXPECTED_FRAMES])
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
        checkpoint = directory / "checkpoint"
        saved = save_checkpoint(checkpoint, policy, {"registration_id": reg["registration_id"],
            "dataset_id": reg["dataset_id"], "source_checkpoint_id": metadata["checkpoint_id"],
            "readout_arrays_sha256": frozen.sha256(directory / "readout.npz"),
            "fit_count": 1, "optimizer_steps": 0, "theta_updates": 0, "qualification_only": True})
        # Release the only complete brain before rebuilding through the exact
        # desktop factory. No second neural rollout or parallel graph is needed.
        del output, state, template, policy
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
        comparison, offset = {"baseline": {}, "candidate": {}}, 0
        for split in EXPECTED_FRAMES:
            count = EXPECTED_FRAMES[split]
            comparison["baseline"][split] = score_requests(dataset["splits"][split], before[split])
            comparison["candidate"][split] = score_requests(dataset["splits"][split], recovered[offset:offset+count])
            offset += count
        write_json(directory / "comparison.json", comparison)
        audit_sources()
        result.update(status="completed", reason="complete_sessions_and_strict_restoration_checked",
            neural_compute_seconds=compute_seconds, theta_unchanged=True, graph_unchanged=True,
            checkpoint_path=str(checkpoint), checkpoint_id=saved["checkpoint_id"], policy_id=saved["policy"]["policy_id"],
            validation={name: {k:v for k,v in value["validation"].items() if k not in ("episodes", "requests")}
                        for name, value in comparison.items()}, coverage_limits=configuration()["coverage_limits"])
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
    parser.add_argument("--checkpoint", type=Path, default=frozen.ROOT / "data/x2_behavior_readout_20261023_r2/checkpoints/0149")
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
        result = execute(args.output, args.checkpoint)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"], "reason": result["worker"].get("reason")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
