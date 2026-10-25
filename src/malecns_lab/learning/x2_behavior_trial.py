"""One bounded, balanced full-brain update with an untouched held-out prefix.

This diagnoses five task starts, not complete interaction ability. Default CLI
is a preview. Existing checkpoints and archived datasets are read-only inputs.
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

SCHEMA = "malecns_lab.x2_behavior_short_trial.v1"
TASKS = ("GREET", "FACE_LEFT", "FACE_RIGHT", "SHOW_FORWARD", "END_INTERACTION")
MINIMUM_BUDGET = 2 * 1024**3  # Old 1000-step probe peaks: RSS 1.63 GB / MPS 1.60 GB.
OWN_SOURCES = ("learning/x2_behavior_trial.py", "learning/x2_behavior_trial_policy.py")
OUTPUT_PREFIX = "x2_behavior_short_trial_"


def output_directory(output, root, *, created=False, inputs=()):
    """Keep trial writes separate from every archived input tree."""
    root = Path(root).resolve()
    data, output = root / "data", Path(output).absolute()
    frozen.require(data.is_dir() and data.resolve() == data
        and output.parent == data and output.name.startswith(OUTPUT_PREFIX)
        and len(output.name) > len(OUTPUT_PREFIX),
        "Output must be a direct data/ child named x2_behavior_short_trial_<new-name>")
    frozen.require(not output.is_symlink() and output.resolve() == output, "Output must not use symbolic links")
    for source in inputs:
        source = Path(source).resolve()
        frozen.require(not output.is_relative_to(source) and not source.is_relative_to(output),
                       "Output must be separate from archived input assets")
    frozen.require(output.is_dir() if created else not output.exists(),
                   "Registered output directory missing" if created else "Output already exists")
    return output


def select_prefixes(episodes, family):
    selected = []
    for task in TASKS:
        rows = [e for e in episodes if e["task"] == task and e["family_id"] == family and e["context"] == "normal"]
        frozen.require(len(rows) == 1 and len(rows[0]["frames"]) >= 2, "Missing unique normal task prefix")
        selected.append({**rows[0], "frames": rows[0]["frames"][:2]})
    return selected


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 600, "max_updates": 1,
        "train_family": "family-007", "validation_family": "family-107", "tasks": list(TASKS),
        "prefix_frames": 2, "decision_steps": 500, "credit_steps": 1000,
        "spike_backward_gain": .01, "normalization": "train feature mean/std; 0.1 mV std floor",
        "optimizer": {"name": "Adam", "theta_lr": .0001, "head_lr": .001,
                      "gradient_group_l2_cap": 1., "fresh_state": True},
        "gradient_admission": "finite raw values, stable float64 norms; separate theta/head norm caps before Adam",
        "boundary_gradient_claim": "finite nonzero first-boundary gradient under the combined two-frame loss; final-frame contribution not isolated",
        "validation": "same ten frames and exact fixed precondition state before/after the single balanced update",
        "claim": "prefix diagnostic only, not complete task qualification or a deployable checkpoint",
        "locked_test_opened": False, "system_reserve_gib": 1}


def execute(output, checkpoint, cache, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import inspect_checkpoint, source_inventory
    root, checkpoint, cache = map(lambda p: Path(p).resolve(), (root, checkpoint, cache))
    output = output_directory(output, root, inputs=(checkpoint, cache))
    frozen.require(checkpoint.is_relative_to(root / "data") and cache.is_relative_to(root / "data"), "Archived local inputs required")
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET,
                       f"Insufficient safe full-brain budget: {resources['effective_rss_limit_bytes']} < {MINIMUM_BUDGET}")
        meta = inspect_checkpoint(checkpoint)
        output.mkdir(exist_ok=False)
        write_json(output / "simulator_preflight.json", preflight)
        sources = source_inventory(root)
        sources.update({name: frozen.sha256(root / "src/malecns_lab" / name) for name in OWN_SOURCES})
        for name in sources:
            target = output / "sources/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "src/malecns_lab" / name, target)
            target.chmod(0o444)
        reg = {"configuration": configuration(), "root": str(root), "output": str(output),
            "parent_pid": os.getpid(), "checkpoint": str(checkpoint), "checkpoint_id": meta["checkpoint_id"],
            "cache": str(cache), "source_sha256": sources, "resources": resources}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output / "registration.json", reg)
        remaining = 600-(time.monotonic()-started)
        frozen.require(remaining > 0, "Preparation exhausted wall budget")
        result = supervise([sys.executable, "-m", "malecns_lab.learning.x2_behavior_trial", "--_worker", str(output)],
            output, rss_limit_bytes=resources["effective_rss_limit_bytes"], wall_seconds=remaining,
            reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
        summary = {"status": "completed" if result["status"] == worker.get("status") == "completed" else "failed",
            "output": str(output), "registration_id": reg["registration_id"], "supervisor": result,
            "worker": worker, "wall_seconds": time.monotonic()-started}
        write_json(output / "summary.json", summary)
        return summary


def cache_diagnostic(cache, selected_checkpoint):
    """Read train/validation only; one fixed linear probe, no test-directed search."""
    import numpy as np
    from .x2_behavior_data import load_split
    from .x2_behavior_trial_policy import normalization
    from .x2_behavior_training import inspect_checkpoint
    cache = Path(cache)
    cache_registration = frozen.read_json(cache / "registration.json")
    frozen.bound(cache_registration, "registration_id")
    reference = cache_registration["checkpoint"]
    source_path = Path(reference["path"])
    source = inspect_checkpoint(source_path)
    frozen.require(source["checkpoint_id"] == reference["checkpoint_id"]
        and frozen.sha256(source_path / "checkpoint.json") == reference["metadata_sha256"], "Cache initialization reference differs")
    for key in ("parameter.brain.theta",):
        frozen.require(source["arrays"][key] == selected_checkpoint["arrays"][key], "Cache theta differs from selected checkpoint")
    frozen.require(source["policy"] == selected_checkpoint["policy"], "Cache policy differs")
    selected = frozen.read_json(cache / "selected_episodes.json")
    partitions, matrices, labels, first_masks = {}, {}, {}, {}
    evidence = {"cache": str(cache), "policy_id": source["policy"]["policy_id"], "files": {}, "locked_test_opened": False}
    for split in ("train", "validation"):
        episodes = load_split(cache / "dataset", split)
        partitions[split] = episodes
        by_id = {e["episode_id"]: e for e in episodes}
        rows, target_rows, first = [], [], []
        for eid in selected[split]:
            for index, frame in enumerate(by_id[eid]["frames"]):
                stem = cache / f"{split}-features" / eid / f"frame-{index:04d}"
                meta = frozen.read_json(stem.with_suffix(".json"))
                file_hash = frozen.sha256(stem.with_suffix(".npz"))
                frozen.require(meta["arrays_sha256"] == file_hash and meta["observation"] == frame["observation"]
                    and meta["decision"]["policy_id"] == source["policy"]["policy_id"], "Cached feature evidence differs")
                evidence["files"][str(stem.relative_to(cache))] = file_hash
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
                    row = arrays["features_mv"].copy()
                frozen.require(row.shape == (2628,) and row.dtype == np.float32 and np.isfinite(row).all(), "Invalid DN feature")
                rows.append(row)
                target_rows.append((frame["target"]["behavior"], frame["target"]["phase"], by_id[eid]["task"]))
                first.append(index == 0)
        matrices[split], labels[split], first_masks[split] = np.array(rows), np.array(target_rows), np.array(first)
    center, scale = normalization(matrices["train"])
    report = {"normalization_fitted_split": "train", "fixed_ridge_alpha": 1., "diagnostic_only": True,
              "frames": {k: len(v) for k, v in matrices.items()}, "linear_decodability": {}}
    for name, divisor in (("raw", np.ones(2628)), ("normalized", scale.astype(np.float64))):
        train = (matrices["train"].astype(np.float64)-center)/divisor
        valid = (matrices["validation"].astype(np.float64)-center)/divisor
        gram = train @ train.T
        for column, target in enumerate(("behavior", "phase", "task")):
            classes = np.unique(labels["train"][:, column])
            truth = (labels["train"][:, column, None] == classes).astype(float)
            mean = truth.mean(0)
            coefficients = np.linalg.solve(gram+np.eye(len(train)), truth-mean)
            fitted = classes[(gram@coefficients+mean).argmax(1)]
            predicted = classes[((valid@train.T)@coefficients+mean).argmax(1)]
            report["linear_decodability"][f"{name}.{target}"] = {
                "train_accuracy": float((fitted == labels["train"][:, column]).mean()),
                "validation_accuracy": float((predicted == labels["validation"][:, column]).mean()),
                "validation_first_frame_accuracy": float((predicted[first_masks["validation"]] == labels["validation"][first_masks["validation"]][:, column]).mean())}
    return partitions, matrices, center, scale, evidence, report


def run_worker(directory):
    from .resource_guard import write_json
    from .x2_behavior_training import inspect_checkpoint, restore_checkpoint, save_checkpoint, _gradient, loss_for, _prediction, hashlib_counts
    directory = Path(directory).absolute()
    frozen.require(not directory.is_symlink() and directory.resolve() == directory,
                   "Worker output must not use symbolic links")
    reg = frozen.read_json(directory / "registration.json")
    frozen.bound(reg, "registration_id")
    frozen.require(os.getppid() == reg["parent_pid"] and reg["output"] == str(directory), "Supervising parent required")
    root = Path(reg["root"])
    output_directory(directory, root, created=True, inputs=(reg["checkpoint"], reg["cache"]))
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must own full-brain lock")
    (directory / "worker.claim").write_text(str(os.getpid()))
    started, sampler = time.monotonic(), None
    result = {"status": "failed", "reason": None, "registration_id": reg["registration_id"],
        "optimizer_steps": 0, "locked_test_opened": False, "deployable": False}
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started, **details})
    try:
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected, "Registered source changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_trial_policy import X2BehaviorTrialPolicy
        from .x2_behavior_policy import array_identity
        from .x2_behavior_data import score_predictions
        from ..control.x2_behavior import Observation
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        phase("cached_linear_diagnostic")
        metadata = inspect_checkpoint(reg["checkpoint"])
        frozen.require(metadata["checkpoint_id"] == reg["checkpoint_id"], "Checkpoint changed")
        partitions, matrices, center, scale, evidence, diagnostic = cache_diagnostic(reg["cache"], metadata)
        write_json(directory / "cache-evidence.json", evidence)
        write_json(directory / "linear-diagnostic.json", diagnostic)
        train, validation = select_prefixes(partitions["train"], "family-007"), select_prefixes(partitions["validation"], "family-107")
        write_json(directory / "selected-prefixes.json", {"train": train, "validation": validation})
        phase("load_complete_graph")
        anchors = frozen.read_anchors(root)
        values = frozen.load_parameters(root / frozen.CHECKPOINT / "arrays.npz", anchors["checkpoint"], anchors["run"]["policy"])
        policy = X2BehaviorTrialPolicy.from_reference(root / frozen.REFERENCE, anchors["run"]["policy"],
            theta=values["theta"], spike_backward_gain=.1)
        del values
        restore_checkpoint(reg["checkpoint"], policy)
        original_heads = {name: p.detach().cpu().numpy().copy() for name, p in policy.named_parameters() if name != "brain.theta"}
        policy.migrate_trial(center, scale, source_checkpoint_id=metadata["checkpoint_id"],
            statistics_evidence={"cache_policy_id": evidence["policy_id"], "split": "train", "frames": len(matrices["train"]),
                "evidence_sha256": frozen.sha256(directory / "cache-evidence.json")})
        # The change in readout coordinates must not manufacture better baseline answers.
        affine_errors = {}
        for name in ("behavior_head", "phase_head", "parameter_head"):
            head = getattr(policy, name)
            before = ((matrices["train"].astype(np.float64)+52)/7)@original_heads[name+".weight"].T + original_heads[name+".bias"]
            after = ((matrices["train"].astype(np.float64)-center)/scale)@head.weight.detach().cpu().numpy().T + head.bias.detach().cpu().numpy()
            affine_errors[name] = float(np.abs(after-before).max())
            frozen.require(np.allclose(before, after, rtol=1e-5, atol=1e-4), "Affine migration changed initial readout")
        write_json(directory / "affine-migration.json", {"max_abs_errors": affine_errors, "tolerance": {"rtol": 1e-5, "atol": 1e-4}})
        write_json(directory / "policy.json", policy.identity)
        initial = save_checkpoint(directory / "initial-checkpoint", policy, {"role": "diagnostic_initial", "optimizer_steps": 0})
        before_parameters = {n: p.detach().cpu().numpy().copy() for n, p in policy.named_parameters()}
        raw_counts = hashlib_counts(policy)
        phase("precondition")
        template = policy.initial_state("fixed-trial-precondition")
        fixed_state = policy.state_identity(template)
        with (directory / "fixed-initial-state.npz").open("xb") as stream:
            np.savez_compressed(stream, **policy.state_arrays(template))
        write_json(directory / "fixed-initial-state.json", fixed_state)

        def evaluate(label):
            phase(label)
            predictions, losses, features, inputs, paired_forward = {}, [], [], [], []
            with torch.no_grad():
                for episode in validation:
                    state = replace(template, session_id=episode["episode_id"])
                    predictions[episode["episode_id"]] = []
                    for frame in episode["frames"]:
                        observation = Observation.from_dict(frame["observation"])
                        output = policy(observation, state, checkpoint_blocks=False)
                        predictions[episode["episode_id"]].append(_prediction((output.behavior_logits, output.phase_logits, output.parameter_raw, output.intensity)))
                        losses.append(float(loss_for(output, frame["target"]).cpu()))
                        features.append(output.features_mv.cpu().numpy().copy())
                        inputs.append(policy.encoder.encode(observation))
                        if label == "baseline":
                            path = Path(reg["cache"]) / "validation-features" / episode["episode_id"] / f"frame-{frame['frame_index']:04d}.npz"
                            with np.load(path, allow_pickle=False) as archived:
                                prior = archived["features_mv"]
                            delta = float(np.abs(prior.astype(np.float64)-features[-1]).max())
                            paired_forward.append({"path": str(path), "exact": bool(np.array_equal(prior, features[-1])), "max_abs_mv": delta})
                            frozen.require(np.array_equal(prior, features[-1]), "New estimator changed baseline neural forward")
                        state = output.state
            features, inputs = np.array(features), np.array(inputs)
            first = features[::2].astype(np.float64)
            pairs = {f"{TASKS[i]}/{TASKS[j]}": {"input_changed_count": int(np.count_nonzero(inputs[2*i] != inputs[2*j])),
                "dn_l2_mv": float(np.linalg.norm(first[i]-first[j]))} for i in range(5) for j in range(i+1, 5)}
            report = {"loss": float(np.mean(losses)), "scores": score_predictions(validation, predictions),
                "predictions": predictions, "pairwise_first_frames": pairs, "fixed_initial_state_id": fixed_state["state_id"],
                "features_identity": array_identity(features), "prefix_only": True, "paired_forward": paired_forward}
            with (directory / f"{label}-features.npz").open("xb") as stream:
                np.savez_compressed(stream, features_mv=features, inputs=inputs)
            write_json(directory / f"{label}.json", report)
            return report

        baseline = evaluate("baseline")
        optimizer = torch.optim.Adam([{"params": [policy.brain.theta], "lr": .0001},
            {"params": [p for n, p in policy.named_parameters() if n != "brain.theta"], "lr": .001}])
        optimizer.zero_grad(set_to_none=True)
        audits = []
        for index, episode in enumerate(train):
            phase("balanced_training_backward", task=episode["task"], window=index+1, total=5)
            state, outputs = replace(template, session_id=episode["episode_id"]), []
            # Save differences of accumulated gradients to audit each task's contribution.
            earlier = {n: p.grad.detach().clone() if p.grad is not None else None for n, p in policy.named_parameters()}
            for frame_index, frame in enumerate(episode["frames"]):
                output = policy(Observation.from_dict(frame["observation"]), state)
                if frame_index == 0:
                    boundary = output.state.brain.v
                    boundary.retain_grad()
                outputs.append(output)
                state = output.state
            loss = torch.stack([loss_for(o, f["target"]) for o, f in zip(outputs, episode["frames"])]).mean()/5
            loss.backward()
            gradients = {n: _gradient(p.grad if earlier[n] is None else p.grad-earlier[n]) for n, p in policy.named_parameters()}
            audit = {"task": episode["task"], "loss": float(loss.detach().cpu())*5,
                "parameter_gradients": gradients, "first_boundary_v_gradient": _gradient(boundary.grad),
                "boundary_gradient_loss_scope": "combined_two_frame_loss",
                "final_frame_contribution_isolated": False}
            audits.append(audit)
            write_json(directory / "gradient-audits.json", {"windows": audits})
            frozen.require(all(g["finite"] for g in gradients.values()), "Nonfinite task gradient")
            frozen.require(audit["first_boundary_v_gradient"]["finite"] and audit["first_boundary_v_gradient"]["nonzero"] > 0,
                           "Missing finite nonzero boundary gradient under combined two-frame loss")
            del loss, outputs, output, state, boundary, earlier
        parameters = dict(policy.named_parameters())
        raw_gradients = {name: _gradient(p.grad) for name, p in parameters.items()}
        clips = {}
        for group_name, names in (("theta", ["brain.theta"]), ("heads", [n for n in parameters if n != "brain.theta"])):
            norm = float(np.linalg.norm([raw_gradients[n]["l2"] for n in names]))
            factor = min(1., 1./max(norm, 1e-30))
            clips[group_name] = {"raw_l2": norm, "applied_scale": factor, "cap": 1.}
            for name in names:
                parameters[name].grad.mul_(factor)
        phase("single_adam_update")
        optimizer.step()
        result["optimizer_steps"] = 1
        policy.validate_parameters()
        adam = {f"{name}.{key}": _gradient(value) for name, p in parameters.items()
                for key, value in optimizer.state[p].items() if torch.is_tensor(value)}
        frozen.require(all(value["finite"] for value in adam.values()), "Nonfinite Adam state")
        write_json(directory / "gradient-audits.json", {"windows": audits, "accumulated_raw": raw_gradients,
            "group_clipping": clips, "adam_state": adam, "scale_is_not_proof_of_learned_ability": True})
        final = save_checkpoint(directory / "final-checkpoint", policy,
            {"role": "diagnostic_only_not_selected_for_deployment", "optimizer_steps": 1}, optimizer=optimizer)
        after = evaluate("after")
        frozen.require(policy.state_identity(template) == fixed_state, "Validation initial state changed")
        frozen.require(hashlib_counts(policy) == raw_counts, "Raw graph changed")
        changes = {n: {"changed_count": int(np.count_nonzero(before_parameters[n] != p.detach().cpu().numpy())),
                      "delta_l2": float(np.linalg.norm(p.detach().cpu().numpy().astype(np.float64)-before_parameters[n]))}
                   for n, p in parameters.items()}
        frozen.require(changes["brain.theta"]["changed_count"] > 0, "No internal parameter update")
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected, "Source changed during trial")
        result.update(status="completed", reason="one_balanced_update_measured", changes=changes,
            baseline_loss=baseline["loss"], after_loss=after["loss"], validation_loss_improved=after["loss"] < baseline["loss"],
            baseline_scores={k:v for k,v in baseline["scores"].items() if k != "episodes"},
            after_scores={k:v for k,v in after["scores"].items() if k != "episodes"},
            initial_checkpoint_id=initial["checkpoint_id"], final_checkpoint_id=final["checkpoint_id"],
            raw_counts_unchanged=True, graph_nodes=166700, graph_edges=25582938)
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
    parser.add_argument("--cache", type=Path, default=frozen.ROOT / "data/x2_behavior_readout_20261023_r1")
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
        result = execute(args.output, args.checkpoint, args.cache)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": result["output"], "reason": result["worker"].get("reason")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
