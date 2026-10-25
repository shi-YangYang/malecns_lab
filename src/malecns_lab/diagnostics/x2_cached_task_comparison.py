"""CPU-only, single-version data/cache gates followed by two fixed ridge fits.

Run this file directly. No production policy, brain, simulator or Torch imports.
The parent enforces the remaining wall budget; every failed gate stops the run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/x2_behavior_short_trial_interaction_paired_20261024_r1"
DATASET_ID = "3b9edf5ca1e4d7484d12bd21f7c1d1f669b4d4af3f401130c533bd9b19b2921d"
CHECKPOINT_ID = "e224de705e2406f8407e7831bba0693785a886fd073c20ac31d672c6b93f11e0"
BEHAVIORS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")
PHASES = ("START", "CONTINUE", "FINISH")
ALPHA = 1e-4
HISTORY = 12
# Reserve one second for the preserved initial import-guard failure (< 1 s).
WALL_LIMIT_SECONDS = 599


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(value, key):
    require(value[key] == digest({k: v for k, v in value.items() if k != key}), key + " mismatch")


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class NoNeuralRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "torch" or fullname.startswith("torch.")
                or fullname.startswith(("malecns_lab.brain", "malecns_lab.runtime", "malecns_lab.experiments"))):
            raise RuntimeError("Forbidden neural/device import: " + fullname)
        return None


def audit_data(out, np):
    reg, data = read(RUN / "registration.json"), read(RUN / "dataset.json")
    bound(reg, "registration_id")
    bound(data["identity"], "dataset_id")
    require(data["identity"]["dataset_id"] == reg["dataset_id"] == DATASET_ID, "Selected dataset changed")
    require(sha(RUN / "dataset.json") == reg["dataset_sha256"], "Dataset bytes changed")
    source_hashes = {}
    for name, expected in reg["source_sha256"].items():
        actual = sha(RUN / "sources/malecns_lab" / name)
        require(actual == expected, "Archived source changed: " + name)
        source_hashes[name] = actual
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RUN / "sources"))
    sys.meta_path.insert(0, NoNeuralRuntime())
    from malecns_lab.control.x2_behavior import Observation
    from malecns_lab.learning.x2_behavior_data import teacher_step
    # Load this verified leaf directly: senses/__init__.py imports the graph
    # encoder, although this leaf's channel_values has no neural dependency.
    spec = importlib.util.spec_from_file_location("malecns_lab.senses.x2_behavior",
        RUN / "sources/malecns_lab/senses/x2_behavior.py")
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    channel_values, CHANNELS = encoder.channel_values, encoder.CHANNELS

    policy = read(RUN / "policy.json")
    bound(policy, "policy_id")
    profile = data["identity"]["profile"]
    require(profile == policy["interaction_profile"], "Dataset/policy profile mismatch")
    require(profile["skill_profile_id"] == "malecns.x2.sim.advance1-turn2.v1"
            and not profile["wave_enabled"], "Wrong execution/capability version")
    require(data["identity"]["teacher_source_sha256"] == source_hashes["learning/x2_behavior_data.py"], "Wrong teacher")
    require(data["identity"]["generator_source_sha256"] == source_hashes["learning/x2_interaction_data.py"], "Wrong generator")
    require(data["identity"]["paired_trial_sha256"] == source_hashes["learning/x2_interaction_paired.py"], "Wrong paired protocol")
    require(list(CHANNELS) == policy["input_map"]["channels"], "Wrong channel order")

    def channels(observation):
        result = channel_values(Observation.from_dict(observation))
        # The selected archived encoder suppresses these exact two channels.
        result["body.yaw.positive"] = result["body.yaw.negative"] = 0.
        return np.asarray(list(result.values()), dtype=np.float32)

    rows = {"train": [], "validation": [], "paired": []}
    sessions, by_family, conflicts, seen = set(), {}, [], {}
    teacher_checked = 0

    def row(observation, target, history, **identity):
        nonlocal teacher_checked
        x = channels(observation)
        prefix = history + [x]
        require(len(prefix) <= HISTORY, "History cannot be truncated")
        # Hash the entire available encoded prefix, including its length.
        key = hashlib.sha256(np.stack(prefix).tobytes()).hexdigest()
        if key in seen and seen[key]["target"] != target:
            conflicts.append({"earlier": seen[key], "later": {**identity, "target": target}})
        seen[key] = {**identity, "target": target}
        padded = np.zeros((HISTORY, len(CHANNELS)), dtype=np.float64)
        mask = np.zeros(HISTORY, dtype=np.float64)
        padded[-len(prefix):] = prefix
        mask[-len(prefix):] = 1.
        return {**identity, "observation": observation, "target": target, "channel": x,
                "history": prefix, "raw_features": np.r_[padded.ravel(), mask]}

    for split, episodes in data["splits"].items():
        require(data["identity"]["split_sha256"][split] == digest(episodes), "Split hash mismatch")
        for episode in episodes:
            require(episode["episode_id"] not in sessions and episode["split"] == split, "Session leakage")
            sessions.add(episode["episode_id"])
            require(episode["sequence_sha256"] == digest(episode["frames"]), "Frame sequence mismatch")
            memory, history, previous_request, erows = {}, [], None, []
            for frame in episode["frames"]:
                o, target = frame["observation"], frame["target"]
                require(o["request_changed"] == (o["request_id"] != previous_request), "Request identity transition unavailable")
                require(o["capabilities"] == profile["capabilities"], "Capability mismatch")
                expected = teacher_step(Observation.from_dict(o), memory)
                require(expected == target, "Teacher needs information absent from consumed observation history")
                teacher_checked += 1
                r = row(o, target, history, split=split, session=episode["episode_id"],
                        family=episode["family_id"], frame_index=frame["frame_index"], request_index=frame["request_index"])
                rows[split].append(r)
                erows.append(r)
                history = r["history"]
                previous_request = o["request_id"]
            by_family[episode["family_id"]] = (episode, erows)

    pair_contexts = read(RUN / "paired-observations.json")["contexts"]
    for context in pair_contexts:
        episode, erows = by_family[context["family"]]
        require(episode["split"] == "train", "Validation branch entered fitting data")
        index = context["frame_index"]
        prefix = [r["channel"] for r in erows[:index]]
        for branch_index, branch in enumerate(context["branches"]):
            memory = {}
            for previous in episode["frames"][:index]:
                teacher_step(Observation.from_dict(previous["observation"]), memory)
            expected = teacher_step(Observation.from_dict(branch["observation"]), memory)
            require(expected == branch["target"], "Branch teacher uses hidden history")
            teacher_checked += 1
            rows["paired"].append(row(branch["observation"], branch["target"], prefix,
                split="paired_train", session=episode["episode_id"], family=episode["family_id"],
                frame_index=index, request_index=None, branch_index=branch_index))
    require([len(rows[k]) for k in ("train", "paired", "validation")] == [24, 18, 12], "Unexpected row inventory")
    require(not conflicts, "Conflicting targets for identical full encoded histories")

    def coverage(group):
        combos = Counter()
        for r in group:
            o, t = r["observation"], r["target"]
            f = o["feedback"]
            combos[(o["task"], t["behavior"], t["phase"], f["standing"],
                    f["status"], f["behavior"], o["request_changed"], tuple(o["capabilities"]))] += 1
        return {
            "frames": len(group), "task": dict(Counter(r["observation"]["task"] for r in group)),
            "behavior": dict(Counter(r["target"]["behavior"] for r in group)),
            "phase": dict(Counter(r["target"]["phase"] for r in group)),
            "intensities": sorted(set(r["target"]["intensity"] for r in group)),
            "standing_false": sum(not r["observation"]["feedback"]["standing"] for r in group),
            "request_changed": sum(r["observation"]["request_changed"] for r in group),
            "combinations": [{"task": k[0], "behavior": k[1], "phase": k[2], "standing": k[3],
                "feedback_status": k[4], "feedback_behavior": k[5], "request_changed": k[6],
                "capabilities": list(k[7]), "count": n} for k, n in combos.items()]}

    evidence = {"status": "passed_for_covered_subset_only", "dataset_id": DATASET_ID,
        "profile": profile, "teacher_recomputed_matches": teacher_checked,
        "identical_encoded_history_conflicts": conflicts,
        "information_boundary": "All consumed observations from session start; no labels, teacher memory, future or gate-only polls in model inputs",
        "history_representation": "12 slots, right-aligned complete 57-channel prefix plus presence mask; no truncation",
        "coverage": {k: coverage(v) for k, v in rows.items()},
        "cannot_test": ["WAVE/GREET motion", "ordinary CONTINUE", "non-END unsteady starts",
                        "arbitrary active task switching", "variable intensity", "real closed-loop recovery"],
        "coverage_scope": "Finite observed fixtures only; no assertion that all possible histories have consistent labels"}
    write(out / "data-audit.json", evidence)
    return reg, data, policy, rows, pair_contexts, evidence


def audit_cache(out, np, reg, data, policy, rows, contexts):
    meta, worker = read(RUN / "checkpoint/checkpoint.json"), read(RUN / "worker_result.json")
    bound(meta, "checkpoint_id")
    require(meta["checkpoint_id"] == CHECKPOINT_ID == worker["checkpoint_id"], "Wrong checkpoint")
    require(meta["context"]["registration_id"] == reg["registration_id"]
            and meta["context"]["dataset_id"] == DATASET_ID, "Checkpoint dataset mismatch")
    require(sha(RUN / "checkpoint/arrays.npz") == meta["arrays_sha256"], "Checkpoint array hash mismatch")
    require(sha(RUN / "readout.npz") == meta["context"]["readout_arrays_sha256"], "Readout hash mismatch")
    fitted_policy = meta["policy"]
    bound(fitted_policy, "policy_id")
    require(fitted_policy == read(RUN / "paired-policy.json"), "Policy copies differ")
    for name in ("model", "input_map", "features", "state", "frame_steps", "precondition_steps", "interaction_profile"):
        require(policy[name] == fitted_policy[name], "Neural/cache contract changed: " + name)
    require(worker["status"] == "completed" and worker["theta_unchanged"]
            and worker["graph_unchanged"] and worker["completed_frames"] == 54, "Incomplete cache producer")
    source = Path(reg["checkpoint"])
    require(source.is_relative_to(ROOT / "data"), "Foreign parent provenance")
    parent_meta = read(source / "checkpoint.json")
    bound(parent_meta, "checkpoint_id")
    require(parent_meta["checkpoint_id"] == reg["checkpoint_id"] == meta["context"]["source_checkpoint_id"], "Parent checkpoint mismatch")
    require(sha(source / "arrays.npz") == parent_meta["arrays_sha256"], "Parent array hash mismatch")
    theta_key = "parameter.brain.theta"
    require(meta["arrays"][theta_key] == parent_meta["arrays"][theta_key], "Cached brain theta differs")
    # Parent is used only for provenance, never as extra training/validation data.
    initial = read(RUN / "initial-state.json")
    bound(initial, "state_id")
    with np.load(RUN / "initial-state.npz", allow_pickle=False) as z:
        require(set(z.files) == set(initial["arrays"]), "Initial state inventory changed")
        for name in z.files:
            a = z[name]
            actual = {"shape": list(a.shape), "dtype": a.dtype.str,
                      "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
            require(actual == initial["arrays"][name], "Initial state changed: " + name)
    require(initial["neural_step"] == 600 and initial["frame_index"] == 0, "Wrong preconditioning")
    evidence_sessions = read(RUN / "state-evidence.json")["sessions"]
    episodes = [e for split in data["splits"].values() for e in split]
    require(len(evidence_sessions) == len(episodes) == 3, "Missing state history")
    for e, evidence in zip(episodes, evidence_sessions):
        require(evidence["session_id"] == e["episode_id"]
                and evidence["initial_state"]["arrays"] == initial["arrays"], "Session initialization mismatch")
        require(len(evidence["boundaries"]) == len(e["frames"]), "Missing neural boundaries")
        for i, (frame, boundary) in enumerate(zip(e["frames"], evidence["boundaries"])):
            o = frame["observation"]
            require(boundary["frame_index"] == i and boundary["neural_step_start"] == 600 + i*500
                    and boundary["neural_step_end"] == 1100 + i*500
                    and boundary["observation_seq"] == o["observation_seq"]
                    and boundary["request_id"] == o["request_id"]
                    and boundary["request_changed"] == o["request_changed"]
                    and boundary["gate_only_polls_skipped"] == len(frame["gate_only_feedback"]), "State continuation mismatch")
        require(evidence["final_state"]["neural_step"] == 600 + len(e["frames"])*500, "Final state clock mismatch")
    for context in contexts:
        for branch in context["branches"]:
            pre = branch["prestate_before"]
            bound(pre, "state_id")
            require(pre == branch["prestate_after"] and pre["neural_step"] == 600+500*context["frame_index"], "Branch mutated shared state")
    assignment = np.asarray(policy["input_map"]["channel_indices"])
    require(set(assignment.tolist()) == set(range(57)), "Some observation channels absent from brain")
    cache, load_times = {}, {}
    for split, filename in (("train", "train-features.npz"), ("paired", "paired-train-features.npz"), ("validation", "validation-features.npz")):
        start = time.perf_counter()
        with np.load(RUN / filename, allow_pickle=False) as z:
            features, encoded = z["features_mv"].copy(), z["encoded_inputs"].copy()
        load_times[split] = time.perf_counter()-start
        require(features.shape == (len(rows[split]), 2628) and np.isfinite(features).all(), "Incomplete/nonfinite DN cache")
        expected = np.stack([r["channel"][assignment] for r in rows[split]])
        require(np.array_equal(expected, encoded), "Cached encoded inputs do not match original observations")
        cache[split] = features
    with np.load(RUN / "readout.npz", allow_pickle=False) as z:
        readout = {k: z[k].copy() for k in z.files}
    all_dn = np.concatenate([cache[s] for s in ("train", "paired", "validation")]).astype(np.float64)
    z = (all_dn-readout["center"])/readout["scale"]
    logits = np.column_stack((z@readout["weight"]+readout["bias"],
                             z@readout["parameter_weight"]+readout["parameter_bias"]))
    with np.load(RUN / "logit-comparison.npz", allow_pickle=False) as z:
        archived, folded = z["float64_ridge_logits"], z["folded_mps_logits"]
    max_error = float(np.max(np.abs(logits-archived)))
    require(np.allclose(logits, archived, atol=1e-9, rtol=1e-9), "Cache/readout archived logits mismatch")
    require(np.array_equal(logits[:, :5].argmax(1), folded[:, :5].argmax(1))
            and np.array_equal(logits[:, 5:8].argmax(1), folded[:, 5:8].argmax(1)), "Archived float32 decisions differ")
    files = ["registration.json", "dataset.json", "paired-observations.json", "state-evidence.json",
             "initial-state.json", "initial-state.npz", "policy.json", "paired-policy.json",
             "train-features.npz", "paired-train-features.npz", "validation-features.npz",
             "readout.npz", "logit-comparison.npz", "checkpoint/checkpoint.json", "checkpoint/arrays.npz"]
    evidence = {"status": "passed_existing_archive_consistency", "source_run": str(RUN),
        "dataset_id": DATASET_ID, "checkpoint_id": CHECKPOINT_ID,
        "input_map_id": policy["input_map"]["input_map_id"], "theta": meta["arrays"][theta_key],
        "encoded_input_rows_exact": 54, "sessions": 3, "paired_shared_prestates": 6,
        "archived_logits_max_abs_error": max_error, "archived_float32_argmax_exact": True,
        "cache_load_seconds": load_times, "current_file_sha256": {n: sha(RUN/n) for n in files},
        "limit": "Archive metadata, input bytes, state evidence and saved logits agree; no independent neural replay performed. Current file hashes are not retroactive signatures."}
    write(out / "cache-audit.json", evidence)
    return cache, readout, evidence


def fit(np, x, targets, floor):
    center, scale = x.mean(0), np.maximum(x.std(0), floor)
    z = (x-center)/scale
    y = np.asarray([[float(t["behavior"] == b) for b in BEHAVIORS]
                    + [float(t["phase"] == p) for p in PHASES] for t in targets])
    bias = y.mean(0)
    weight = z.T @ np.linalg.solve(z@z.T + ALPHA*np.eye(len(z)), y-bias)
    require(set(t["intensity"] for t in targets) == {.5}, "Unexpected intensity task")
    return {"center": center, "scale": scale, "weight": weight, "bias": bias}


def predict(np, model, x):
    logits = ((x-model["center"])/model["scale"]) @ model["weight"] + model["bias"]
    require(np.isfinite(logits).all(), "Nonfinite classifier output")
    return [{"behavior": BEHAVIORS[int(r[:5].argmax())], "phase": PHASES[int(r[5:].argmax())],
             "intensity": .5, "behavior_logits": r[:5].tolist(), "phase_logits": r[5:].tolist()} for r in logits]


def score(rows, predictions, episodes=None):
    confusion = {b: {p: 0 for p in BEHAVIORS} for b in BEHAVIORS}
    phase_confusion = {b: {p: 0 for p in PHASES} for b in PHASES}
    for r, p in zip(rows, predictions):
        confusion[r["target"]["behavior"]][p["behavior"]] += 1
        phase_confusion[r["target"]["phase"]][p["phase"]] += 1
    correct = [r["target"]["behavior"] == p["behavior"] and r["target"]["phase"] == p["phase"] for r, p in zip(rows, predictions)]
    ends = [(r, p) for r, p in zip(rows, predictions) if r["target"]["phase"] == "FINISH"]
    explicit = [(r, p) for r, p in zip(rows, predictions) if r["observation"]["task"] == "END_INTERACTION"]
    result = {"frames": len(rows), "correct_behavior_and_phase": sum(correct),
        "true_counts": {b: sum(confusion[b].values()) for b in BEHAVIORS},
        "predicted_counts": {b: sum(confusion[t][b] for t in BEHAVIORS) for b in BEHAVIORS},
        "confusion": confusion, "phase_confusion": phase_confusion,
        "finish_target_frames": len(ends), "motion_when_should_finish": sum(p["behavior"] != "HOLD" for _, p in ends),
        "explicit_END_frames": len(explicit), "motion_on_explicit_END": sum(p["behavior"] != "HOLD" for _, p in explicit),
        "hold_fraction": sum(p["behavior"] == "HOLD" for p in predictions)/len(rows),
        "complete_requests": None, "requests": []}
    if episodes is not None:
        offset = 0
        for e in episodes:
            for request in e["requests"]:
                start, end = request["first_frame"], request["end_frame"]
                result["requests"].append({"session": e["episode_id"], "request_id": request["request_id"],
                    "task": request["task"], "frames": end-start,
                    "all_behavior_phase_correct": all(correct[offset+start:offset+end]),
                    "scope": "teacher sequence agreement on fixed offline observations, not physical completion"})
            offset += len(e["frames"])
        result["complete_requests"] = {"correct": sum(r["all_behavior_phase_correct"] for r in result["requests"]),
                                       "total": len(result["requests"])}
    return result


def compare(out, np, data, rows, cache, archived_readout):
    train_rows = rows["train"] + rows["paired"]
    targets = [r["target"] for r in train_rows]
    raw = {k: np.stack([r["raw_features"] for r in rows[k]]) for k in rows}
    models, metrics, predictions = {}, {}, {}
    for name, features, floor in (("A_raw_full_history", raw, 1e-6), ("B_cached_DN", cache, .1)):
        start = time.perf_counter()
        x = np.concatenate((features["train"], features["paired"])).astype(np.float64)
        model = fit(np, x, targets, floor)
        fit_seconds = time.perf_counter()-start
        if name == "B_cached_DN":
            require(all(np.allclose(model[k], archived_readout[k], atol=1e-10, rtol=1e-9)
                        for k in model), "Fixed B refit no longer reproduces archived estimator")
        models[name], predictions[name], timing = model, {}, {}
        # Exactly one pass per split: descriptive timings, no microbenchmark repetition.
        for split in ("train", "paired", "validation"):
            start = time.perf_counter()
            predictions[name][split] = predict(np, model, features[split].astype(np.float64))
            timing[split] = {"batch_seconds": time.perf_counter()-start, "rows": len(rows[split])}
        metrics[name] = {
            "train_all42": score(train_rows, predictions[name]["train"] + predictions[name]["paired"]),
            "train_continuous24": score(rows["train"], predictions[name]["train"], data["splits"]["train"]),
            "train_paired18": score(rows["paired"], predictions[name]["paired"]),
            "development12": score(rows["validation"], predictions[name]["validation"], data["splits"]["validation"]),
            "fit_seconds": fit_seconds, "inference_seconds": timing, "input_width": x.shape[1],
            "timing_scope": "CPU readout only: already materialized features, normalization, matmul, argmax and prediction serialization. Excludes history construction, cache reading and full MaleCNS dynamics."}
        with (out / (name + ".npz")).open("xb") as f:
            np.savez_compressed(f, **model)
        write(out / "metrics.json", metrics)
        write(out / "predictions.json", predictions)
    write(out / "row-index.json", {split: [{k: v for k, v in r.items() if k not in ("channel", "history", "raw_features")}
                                           for r in values] for split, values in rows.items()})
    return metrics


def worker(out):
    started = time.monotonic()
    result = {"status": "running", "stage": "data", "neural_calls": 0, "fits_completed": 0}
    def timeout(signum, frame):
        raise TimeoutError("Worker budget reached; no automatic retry")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT_SECONDS - 10)
    try:
        import numpy as np
        reg, data, policy, rows, contexts, data_audit = audit_data(out, np)
        result.update(stage="cache", data_gate=data_audit["status"])
        write(out / "result.json", result)
        cache, readout, cache_audit = audit_cache(out, np, reg, data, policy, rows, contexts)
        result.update(stage="comparison", cache_gate=cache_audit["status"])
        write(out / "result.json", result)
        metrics = compare(out, np, data, rows, cache, readout)
        require("torch" not in sys.modules, "Unexpected Torch import")
        result.update(status="completed", stage="done", fits_completed=len(metrics))
    except BaseException as error:
        result.update(status="stopped", reason=type(error).__name__ + ": " + str(error))
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["wall_seconds"] = time.monotonic()-started
        write(out / "result.json", result)
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.absolute()
    if args.worker:
        return worker(out)
    require(out.parent.resolve() == (ROOT / "data").resolve()
            and out.name.startswith("x2_cached_task_comparison_"), "Fresh project diagnostic output required")
    out.mkdir(exist_ok=False)
    config = {"schema": "malecns_lab.cached_task_comparison.v1", "source_run": str(RUN),
        "dataset_id": DATASET_ID, "checkpoint_id": CHECKPOINT_ID, "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "method": "two fixed one-hot ridge regressions; separate behavior/phase argmax; no gating or output correction",
        "alpha": ALPHA, "fits": 2, "parameter_search": False, "history_slots": HISTORY,
        "A": "57 current/raw observation channels with selected yaw suppression, complete consumed prefix plus padding mask",
        "B": "archived frozen DN 2628-dimensional features; no new brain forward",
        "standardization": {"fit_on": "42 train rows only", "A_std_floor": 1e-6, "B_std_floor_mv": .1},
        "intensity": "both constant .5, the only training target; not evidence of intensity learning",
        "evaluation": "same 24 continuous train +18 train branches and same12 development frames; full sessions not shuffled",
        "script_sha256": sha(__file__), "created_unix": time.time()}
    write(out / "configuration.json", config)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    with (out / "worker.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--output", str(out), "--worker"],
                                   stdout=log, stderr=log, env=env)
        try:
            returncode = process.wait(timeout=WALL_LIMIT_SECONDS)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()  # Only this explicitly budgeted diagnostic child.
            returncode = process.wait()
            timed_out = True
    write(out / "supervisor.json", {"returncode": returncode, "hard_timeout": timed_out,
        "wall_seconds": time.monotonic()-started, "existing_processes_touched": False})
    print(json.dumps({"output": str(out), "returncode": returncode, "hard_timeout": timed_out}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
