"""Bounded whole-sequence test of irrelevant absolute heading stimulation.

Old encoding uses authenticated saved responses; the new encoding must run the
same complete brain on all 99 frames. Both fixed ridge readouts fit train only.
No controller, production checkpoint, old dataset, or locked test is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from .x2_behavior_trial import MINIMUM_BUDGET, TASKS, output_directory as trial_output_directory

SCHEMA = "malecns_lab.x2_absolute_yaw_diagnostic.v1"
OUTPUT_PREFIX = "x2_behavior_short_trial_yaw_"
EXPECTED_FRAMES = {"train": 42, "validation": 57}
FAMILIES = {"train": "family-007", "validation": "family-107"}
BEHAVIORS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")
PHASES = ("START", "CONTINUE", "FINISH")
OWN_SOURCES = ("learning/x2_yaw_diagnostic.py", "learning/x2_behavior_trial.py", "learning/x2_behavior_trial_policy.py")


class RelativeHeadingEncoder:
    """Preserve the input ports, observation, and relative heading channels."""
    def __init__(self, original):
        self.original = original
        identity = original.identity
        self.input_body_ids, self.input_indices = original.input_body_ids, original.input_indices
        self.channel_indices = original.channel_indices
        excluded = {identity["channels"].index(name) for name in ("body.yaw.positive", "body.yaw.negative")}
        self.suppressed_indices = tuple(i for i, channel in enumerate(self.channel_indices) if channel in excluded)
        frozen.require(len(self.suppressed_indices) == 118, "Expected both 59-neuron absolute-yaw ports")
        self.parent_input_map_id = identity.pop("input_map_id")
        identity.update(schema="malecns_lab.x2_behavior_input.no_absolute_yaw.v1",
            parent_input_map_id=self.parent_input_map_id,
            intervention={"channels": ["body.yaw.positive", "body.yaw.negative"], "encoded_value": 0.,
                "raw_observation_changed": False, "relative_heading_error_retained": True,
                "reason": "relative tasks must not depend on world-frame absolute yaw"})
        self.input_map_id = frozen.canonical_hash(identity)
        self._identity = {**identity, "input_map_id": self.input_map_id}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def encode(self, observation):
        values = self.original.encode(observation)
        values[list(self.suppressed_indices)] = 0.
        return values


def select_complete(episodes, split):
    frozen.require(split in FAMILIES, "Only train and validation may be selected")
    result = []
    for task in TASKS:
        matched = [e for e in episodes if e["task"] == task and e["family_id"] == FAMILIES[split] and e["context"] == "normal"]
        frozen.require(len(matched) == 1 and matched[0]["split"] == split, "Unique complete normal episode required")
        result.append(matched[0])
    frozen.require(sum(len(e["frames"]) for e in result) == EXPECTED_FRAMES[split], "Registered whole-sequence frame count changed")
    frozen.require(all(e["frames"][-1]["target"]["behavior"] == "HOLD"
                       and e["frames"][-1]["target"]["phase"] == "FINISH" for e in result), "Incomplete task sequence")
    return result


def budget_projection(completed, compute_seconds, remaining_seconds, *, total=99, reserve_seconds=15.):
    frozen.require(0 < completed <= total and compute_seconds >= 0 and remaining_seconds >= 0, "Invalid timing evidence")
    needed = compute_seconds/completed*(total-completed)+reserve_seconds
    return {"completed_frames": completed, "required_frames": total,
            "mean_compute_seconds_per_frame": compute_seconds/completed,
            "estimated_remaining_seconds": needed, "remaining_budget_seconds": remaining_seconds,
            "can_finish": needed <= remaining_seconds}


def fit_readout(train, targets):
    """One fixed ridge alpha, fitted once on train; no validation arguments."""
    import numpy as np
    values = np.asarray(train, dtype=np.float64)
    frozen.require(values.ndim == 2 and values.shape[1] == 2628 and len(values) == len(targets)
                   and np.isfinite(values).all(), "Invalid train features")
    center, scale = values.mean(0), np.maximum(values.std(0), .1)
    z = (values-center)/scale
    y = np.asarray([[float(t["behavior"] == b) for b in BEHAVIORS]
                    + [float(t["phase"] == p) for p in PHASES] for t in targets])
    bias = y.mean(0)
    weight = z.T @ np.linalg.solve(z@z.T+np.eye(len(z)), y-bias)
    # Parameter supervision keeps the original active-action mask. The known
    # 0.5 examples are fitted as logit targets, not hardcoded at inference.
    active = np.asarray([t["behavior"] in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT")
                         and t["phase"] in ("START", "CONTINUE") for t in targets])
    frozen.require(active.any(), "No active parameter examples")
    intensity = np.asarray([t["intensity"] for t in targets], dtype=float)[active]
    frozen.require(((intensity > 0) & (intensity < 1)).all(), "Finite logit targets required")
    yp = np.log(intensity/(1-intensity))
    zp, pcenter = z[active], z[active].mean(0)
    pbias = float(yp.mean())
    pweight = (zp-pcenter).T @ np.linalg.solve((zp-pcenter)@(zp-pcenter).T+np.eye(len(zp)), yp-pbias)
    return {"center": center, "scale": scale, "weight": weight, "bias": bias,
            "parameter_weight": pweight, "parameter_bias": np.asarray(pbias-pcenter@pweight)}


def predict(readout, features):
    import numpy as np
    z = (np.asarray(features, dtype=np.float64)-readout["center"])/readout["scale"]
    scores = z@readout["weight"]+readout["bias"]
    raw = z@readout["parameter_weight"]+readout["parameter_bias"]
    intensity = 1/(1+np.exp(-np.clip(raw, -700, 700)))
    return [{"behavior": BEHAVIORS[int(row[:5].argmax())], "phase": PHASES[int(row[5:].argmax())],
             "intensity": float(value), "behavior_scores": row[:5].tolist(), "phase_scores": row[5:].tolist()}
            for row, value in zip(scores, intensity)]


def score_complete(episodes, rows):
    from .x2_behavior_data import score_predictions
    predictions, position = {}, 0
    for episode in episodes:
        count = len(episode["frames"])
        predictions[episode["episode_id"]] = rows[position:position+count]
        position += count
    frozen.require(position == len(rows), "Predictions must cover every complete frame")
    scores = score_predictions(episodes, predictions)
    scores["phase_errors"] = {phase: sum(frame["target"]["phase"] == phase and row["phase"] != phase
        for e in episodes for frame, row in zip(e["frames"], predictions[e["episode_id"]])) for phase in PHASES}
    return {"scores": scores, "predictions": predictions}


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 600, "families": FAMILIES, "frames": EXPECTED_FRAMES,
        "task_order": TASKS, "decision_steps": 500, "precondition_steps": 600,
        "graph_nodes": 166700, "graph_edges": 25582938, "theta": "frozen selected checkpoint",
        "old_condition": "authenticated original encoding complete neural cache",
        "new_condition": "real complete-brain recomputation, only absolute yaw ports suppressed",
        "readout": {"method": "ridge one-hot behavior/phase, masked intensity logit", "alpha": 1.,
                    "normalization": "each condition's train mean/std only; 0.1 mV floor",
                    "fit_count_per_condition": 1, "hyperparameter_selection": False},
        "resource": {"wall_seconds": 600, "system_reserve_gib": 1, "minimum_safe_budget_gib": 2},
        "initial_state": "same frozen-parameter input-blank precondition shared across episode resets",
        "scope": "five complete offline normal sequences; continuous requests and unavailable WAVE remain unqualified",
        "locked_test_opened": False, "deployable": False}


def output_directory(output, root, **kwargs):
    frozen.require(Path(output).name.startswith(OUTPUT_PREFIX) and len(Path(output).name) > len(OUTPUT_PREFIX),
                   "Yaw diagnostic requires its dedicated output prefix")
    return trial_output_directory(output, root, **kwargs)


def load_old_condition(cache, metadata):
    import numpy as np
    from .x2_behavior_data import load_split
    from .x2_behavior_training import inspect_checkpoint
    cache = Path(cache)
    reg = frozen.read_json(cache / "registration.json")
    frozen.bound(reg, "registration_id")
    ref = reg["checkpoint"]
    source = inspect_checkpoint(ref["path"])
    frozen.require(source["checkpoint_id"] == ref["checkpoint_id"]
        and frozen.sha256(Path(ref["path"]) / "checkpoint.json") == ref["metadata_sha256"]
        and source["policy"] == metadata["policy"]
        and source["arrays"]["parameter.brain.theta"] == metadata["arrays"]["parameter.brain.theta"],
        "Old condition policy/theta or initialization changed")
    episodes, matrices = {}, {}
    evidence = {"cache": str(cache), "cache_registration_id": reg["registration_id"], "files": {},
                "checkpoint_id": source["checkpoint_id"], "locked_test_opened": False}
    for split in FAMILIES:
        episodes[split] = select_complete(load_split(cache / "dataset", split), split)
        rows = []
        for e in episodes[split]:
            for f in e["frames"]:
                stem = cache / f"{split}-features" / e["episode_id"] / f"frame-{f['frame_index']:04d}"
                record = frozen.read_json(stem.with_suffix(".json"))
                actual_hash = frozen.sha256(stem.with_suffix(".npz"))
                frozen.require(record["arrays_sha256"] == actual_hash and record["observation"] == f["observation"]
                    and record["decision"]["policy_id"] == source["policy"]["policy_id"], "Old neural frame identity differs")
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
                    row = arrays["features_mv"].copy()
                frozen.require(row.shape == (2628,) and row.dtype == np.float32 and np.isfinite(row).all(), "Invalid cached feature")
                rows.append(row)
                evidence["files"][str(stem.relative_to(cache))] = actual_hash
        matrices[split] = np.array(rows)
    return episodes, matrices, evidence


def execute(output, checkpoint, cache, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import inspect_checkpoint, source_inventory
    root, checkpoint, cache = (Path(p).resolve() for p in (root, checkpoint, cache))
    frozen.require(all(p.is_relative_to(root / "data") for p in (checkpoint, cache)), "Archived local inputs required")
    output = output_directory(output, root, inputs=(checkpoint, cache))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe full-brain memory budget")
        metadata = inspect_checkpoint(checkpoint)
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        sources.update({name: frozen.sha256(root / "src/malecns_lab" / name) for name in OWN_SOURCES})
        for name in sources:
            destination = output / "sources/malecns_lab" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / "src/malecns_lab" / name, destination)
            destination.chmod(0o444)
        reg = {"configuration": configuration(), "root": str(root), "output": str(output), "parent_pid": os.getpid(),
            "deadline_monotonic": started+600, "checkpoint": str(checkpoint), "checkpoint_id": metadata["checkpoint_id"],
            "cache": str(cache), "source_sha256": sources, "resources": resources}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output / "registration.json", reg)
        write_json(output / "simulator_preflight.json", preflight)
        remaining = reg["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining > 0, "Preparation exhausted budget")
        supervisor = supervise([sys.executable, "-m", "malecns_lab.learning.x2_yaw_diagnostic", "--_worker", str(output)],
            output, rss_limit_bytes=resources["effective_rss_limit_bytes"], wall_seconds=remaining, reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
        summary = {"status": "completed" if supervisor["status"] == worker.get("status") == "completed" else "failed",
            "output": str(output), "registration_id": reg["registration_id"], "supervisor": supervisor, "worker": worker,
            "wall_seconds": time.monotonic()-started}
        write_json(output / "summary.json", summary)
        return summary


def run_worker(directory):
    from .resource_guard import write_json
    from .x2_behavior_training import inspect_checkpoint, restore_checkpoint, hashlib_counts
    directory = Path(directory).absolute()
    frozen.require(not directory.is_symlink() and directory.resolve() == directory, "Worker output cannot use symbolic links")
    reg = frozen.read_json(directory / "registration.json")
    frozen.bound(reg, "registration_id")
    frozen.require(reg["parent_pid"] == os.getppid() and reg["output"] == str(directory), "Supervising parent required")
    root = Path(reg["root"])
    output_directory(directory, root, created=True, inputs=(reg["checkpoint"], reg["cache"]))
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
        "completed_frames": 0, "required_frames": 99, "locked_test_opened": False, "deployable": False}
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started,
            "completed_frames": result["completed_frames"], "required_frames": 99, **details})
    try:
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected
                and frozen.sha256(directory / "sources/malecns_lab" / name) == expected, "Registered source changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_policy import X2BehaviorPolicy, array_identity
        from ..control.x2_behavior import Observation
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        phase("authenticate_old_condition")
        metadata = inspect_checkpoint(reg["checkpoint"])
        frozen.require(metadata["checkpoint_id"] == reg["checkpoint_id"], "Checkpoint changed")
        episodes, old_features, evidence = load_old_condition(reg["cache"], metadata)
        write_json(directory / "old-condition-evidence.json", evidence)
        write_json(directory / "selected-episodes.json", episodes)
        phase("load_complete_graph")
        anchors = frozen.read_anchors(root)
        values = frozen.load_parameters(root / frozen.CHECKPOINT / "arrays.npz", anchors["checkpoint"], anchors["run"]["policy"])
        policy = X2BehaviorPolicy.from_reference(root / frozen.REFERENCE, anchors["run"]["policy"],
            theta=values["theta"], spike_backward_gain=metadata["policy"]["backward"]["spike_backward_gain"])
        del values
        restore_checkpoint(reg["checkpoint"], policy)
        old_policy = policy.identity
        raw_counts = hashlib_counts(policy)
        parameter_ids = {n: array_identity(p.detach().cpu().numpy()) for n, p in policy.named_parameters()}
        policy.encoder = RelativeHeadingEncoder(policy.encoder)
        identity = {**old_policy, "schema": SCHEMA, "parent_policy_id": old_policy["policy_id"],
            "input_map": policy.encoder.identity, "diagnostic_implementation_sha256": frozen.sha256(Path(__file__))}
        identity.pop("policy_id")
        policy.policy_id = frozen.canonical_hash(identity)
        policy._identity = {**identity, "policy_id": policy.policy_id}
        write_json(directory / "policy.json", policy.identity)
        phase("fixed_blank_precondition")
        template = policy.initial_state("yaw-diagnostic-initial")
        initial = policy.state_identity(template)
        first_episode = episodes["train"][0]
        old_state = Path(reg["cache"]) / "train-features" / first_episode["episode_id"] / "state-0000.json"
        state_evidence = frozen.read_json(old_state)
        frozen.require(frozen.sha256(old_state.with_suffix(".npz")) == state_evidence["file_sha256"]
            and initial["arrays"] == state_evidence["arrays"] and initial["neural_step"] == state_evidence["neural_step"],
            "Old and new conditions differ in initial neural state")
        with (directory / "fixed-initial-state.npz").open("xb") as stream:
            np.savez_compressed(stream, **policy.state_arrays(template))
        write_json(directory / "fixed-initial-state.json", {**initial, "old_state_evidence": str(old_state), "exact_array_identity": True})
        new_features, compute_seconds, timing = {}, 0., []
        input_audit = []
        for split in FAMILIES:
            feature_rows, input_rows = [], []
            for episode in episodes[split]:
                state = replace(template, session_id=episode["episode_id"])
                for frame in episode["frames"]:
                    phase("complete_neural_response", split=split, task=episode["task"], frame_index=frame["frame_index"])
                    observation = Observation.from_dict(frame["observation"])
                    old_input, new_input = policy.encoder.original.encode(observation), policy.encoder.encode(observation)
                    changed = np.flatnonzero(old_input != new_input)
                    frozen.require(set(changed).issubset(policy.encoder.suppressed_indices), "Intervention changed non-yaw input ports")
                    begin = time.monotonic()
                    with torch.no_grad():
                        output = policy(observation, state, checkpoint_blocks=False)
                    torch.mps.synchronize()
                    compute_seconds += time.monotonic()-begin
                    feature_rows.append(output.features_mv.cpu().numpy().copy())
                    input_rows.append(new_input)
                    input_audit.append({"split": split, "episode_id": episode["episode_id"], "frame_index": frame["frame_index"],
                        "old_input": array_identity(old_input), "new_input": array_identity(new_input), "changed_count": len(changed)})
                    state = output.state
                    result["completed_frames"] += 1
                    if result["completed_frames"] >= 10 and result["completed_frames"] % 10 == 0:
                        projection = budget_projection(result["completed_frames"], compute_seconds, max(0., reg["deadline_monotonic"]-time.monotonic()))
                        timing.append(projection)
                        write_json(directory / "budget-projections.json", {"projections": timing})
                        if not projection["can_finish"]:
                            with (directory / f"partial-{split}-features.npz").open("xb") as stream:
                                np.savez_compressed(stream, features_mv=np.array(feature_rows), encoded_inputs=np.array(input_rows))
                            raise RuntimeError("Projected complete-sequence cost exceeds remaining budget; no shortened success")
            new_features[split] = np.array(feature_rows)
            frozen.require(len(feature_rows) == EXPECTED_FRAMES[split], "Missing complete sequence frames")
            with (directory / f"new-{split}-features.npz").open("xb") as stream:
                np.savez_compressed(stream, features_mv=new_features[split], encoded_inputs=np.array(input_rows))
        write_json(directory / "input-audit.json", {"frames": input_audit, "only_absolute_yaw_changed": True})
        frozen.require(result["completed_frames"] == 99, "Incomplete neural computation")
        phase("fixed_readout_comparison")
        targets = [f["target"] for e in episodes["train"] for f in e["frames"]]
        comparison = {}
        for condition, features in (("old", old_features), ("no_absolute_yaw", new_features)):
            readout = fit_readout(features["train"], targets)
            with (directory / f"{condition}-readout.npz").open("xb") as stream:
                np.savez_compressed(stream, **readout)
            model = {"schema": "malecns_lab.x2_ridge_diagnostic_readout.v1", "condition": condition,
                "input_map_id": old_policy["input_map"]["input_map_id"] if condition == "old" else policy.encoder.input_map_id,
                "source_checkpoint_id": metadata["checkpoint_id"], "alpha": 1., "normalization_split": "train",
                "train_feature_identity": array_identity(features["train"]), "arrays_sha256": frozen.sha256(directory / f"{condition}-readout.npz"),
                "deployable": False}
            model["readout_id"] = frozen.canonical_hash(model)
            write_json(directory / f"{condition}-readout.json", model)
            comparison[condition] = {split: score_complete(episodes[split], predict(readout, features[split])) for split in FAMILIES}
        write_json(directory / "comparison.json", comparison)
        # Same task prefixes become identical after removing the sole original
        # input difference; longer held-out timings remain distinct sequences.
        prefixes, offset_train, offset_valid = [], 0, 0
        for train_episode, valid_episode in zip(episodes["train"], episodes["validation"]):
            a, b = new_features["train"][offset_train:offset_train+2], new_features["validation"][offset_valid:offset_valid+2]
            prefixes.append({"task": train_episode["task"], "first_two_dn_exact": bool(np.array_equal(a, b)),
                "max_abs_mv": float(np.abs(a.astype(np.float64)-b).max())})
            offset_train += len(train_episode["frames"])
            offset_valid += len(valid_episode["frames"])
        write_json(directory / "invariance.json", {"prefixes": prefixes,
            "boundary": "identical prefixes establish encoding invariance, not independent task generalization"})
        frozen.require(all(p["first_two_dn_exact"] for p in prefixes), "Expected yaw-invariant prefixes differ")
        frozen.require(policy.state_identity(template) == initial and hashlib_counts(policy) == raw_counts, "Initial state or graph changed")
        frozen.require(parameter_ids == {n: array_identity(p.detach().cpu().numpy()) for n, p in policy.named_parameters()}, "Frozen neural policy parameters changed")
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected, "Source changed during diagnostic")
        result.update(status="completed", reason="all_complete_sequences_compared", neural_compute_seconds=compute_seconds,
            theta_unchanged=True, raw_counts_unchanged=True, initial_state_unchanged=True,
            old_validation={k:v for k,v in comparison["old"]["validation"]["scores"].items() if k != "episodes"},
            new_validation={k:v for k,v in comparison["no_absolute_yaw"]["validation"]["scores"].items() if k != "episodes"},
            tasks={condition: comparison[condition]["validation"]["scores"]["episodes"] for condition in comparison})
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
