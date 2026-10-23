"""Registered whole-episode X2 demonstrations; teachers are offline-only.

Synthetic *feedback* fixtures are explicitly labeled; no synthetic neural graph
is created. Locked-test is a separately hashed file, never opened by load_split
for train or validation. No Torch/NumPy or device imports at module import.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "malecns_lab.x2_behavior_dataset.v1"
SPLIT_FILES = {"train": "train.json", "validation": "validation.json", "locked_test": "locked-test.json"}
FAMILIES = {"train": tuple(range(40)), "validation": tuple(range(100, 110)), "locked_test": tuple(range(200, 220))}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def teacher_step(observation, memory):
    """Explicit engineering demonstration rule; NEVER called by inference.

    Uses only the present observation and per-request earlier completion memory.
    Rejections stay failures, not invented completion. Returned targets are
    stored beside observations and are excluded by the sensory encoder API.
    """
    from ..control.x2_behavior import Behavior as B, ExecutionStatus as S, Phase as P, Task as T
    o, f = observation, observation.feedback
    if memory.get("request_id") != o.request_id:
        memory.clear()
        memory.update(request_id=o.request_id, completed=[], rejected=False)
    if f.valid and f.request_id == o.request_id:
        if f.status == S.COMPLETED and f.behavior.value not in memory["completed"]:
            memory["completed"].append(f.behavior.value)
        if f.status in (S.REJECTED, S.FAILED, S.UNKNOWN):
            memory["rejected"] = True
    behavior, phase, intensity = B.HOLD, P.FINISH, .5
    active = f.valid and f.status in (S.ACCEPTED, S.STARTED, S.RUNNING)
    if o.task == T.END_INTERACTION or not f.valid or memory["rejected"]:
        pass
    elif active:
        if f.request_id == o.request_id:
            behavior, phase = f.behavior, P.CONTINUE
        # A changed task asks to end the previous behavior first; the next
        # choice is still learned, not an executor task sequence.
    elif f.standing:
        done = memory["completed"]
        if B.WAVE_RIGHT.value not in done:
            primary = {T.FACE_LEFT: B.TURN_LEFT, T.FACE_RIGHT: B.TURN_RIGHT, T.SHOW_FORWARD: B.ADVANCE}.get(o.task)
            aligned = primary in (B.TURN_LEFT, B.TURN_RIGHT) and abs(f.heading_error_rad) <= .05
            behavior = primary if primary is not None and primary.value not in done and not aligned else B.WAVE_RIGHT
            if behavior in o.capabilities:
                phase = P.START
            else:
                behavior = B.HOLD
    return {"behavior": behavior.value, "phase": phase.value, "intensity": intensity}


def _context(index):
    # Every task has normal, stale/missing, rejected, repeat and switch contexts.
    return ("rejected", "missing_feedback", "waiting_stand", "already_aligned", "repeat_request",
            "switch_request", "slow_feedback", "normal", "normal", "normal")[index % 10]


def _episode(task, split, family, index, seed):
    from ..control.x2_behavior import Behavior as B, ExecutionStatus as S, Feedback, Observation, Perception, Task as T
    context = _context(index)
    family_parameters = {"initial_yaw_rad": .1 + family * .003, "feedback_interval_s": .11 + family * .0007,
                         "running_samples": 1 + family % 3, "settling_samples": 1 + (family // 3) % 2}
    identity = {"seed": seed, "split": split, "family": family, "task": task.value,
                "context": context, "parameters": family_parameters}
    episode_id = _digest(identity)
    request_id = episode_id + "-request"
    yaw = family_parameters["initial_yaw_rad"]
    error = 0. if context == "already_aligned" else (-.45 if task == T.FACE_RIGHT else .45)
    f = Feedback(valid=True, source="simulated_execution_feedback", standing=context != "waiting_stand",
                 yaw_rad=yaw, heading_error_rad=error, mc_mode="STAND_DEFAULT")
    if context == "missing_feedback":
        f = replace(f, valid=False, source="missing")
    if context == "switch_request" or task == T.END_INTERACTION:
        # END spans already idle, moving and waving, including an old request.
        if task != T.END_INTERACTION or index % 3:
            old_behavior = B.ADVANCE if index % 2 else B.WAVE_RIGHT
            f = replace(f, status=S.RUNNING, behavior=old_behavior, standing=False,
                        request_id="previous-request", execution_id=episode_id+"-previous", vx_m_s=.12 if old_behavior == B.ADVANCE else 0.)
    memory, frames = {}, []
    execution_index, pending, repeated, settling = 0, None, 0, 0
    for frame_index in range(24):
        elapsed = family_parameters["feedback_interval_s"] * frame_index
        o = Observation(episode_id, episode_id, frame_index, request_id, task, 10.+elapsed,
                        f, Perception(), tuple(B), frame_index == 0)
        target = teacher_step(o, memory)
        frames.append({"frame_index": frame_index, "observation": o.to_dict(), "target": target})
        selected = B(target["behavior"])
        if f.status in (S.REJECTED, S.FAILED, S.UNKNOWN) or not f.valid:
            if frame_index >= 2:
                break
            continue
        if selected == B.HOLD:
            if f.status in (S.ACCEPTED, S.STARTED, S.RUNNING):
                f = replace(f, status=S.CANCELLED, standing=False, vx_m_s=0., wz_rad_s=0.)
                settling = family_parameters["settling_samples"]
            elif not f.standing:
                settling -= 1
                if settling <= 0:
                    f = replace(f, standing=True, vx_m_s=0., wz_rad_s=0.)
            elif frame_index >= 2:
                break
            continue
        if target["phase"] == "START":
            execution_index += 1
            pending = selected
            if context == "rejected":
                f = replace(f, status=S.REJECTED, behavior=selected, request_id=request_id,
                            execution_id=episode_id+f"-execution-{execution_index}")
                continue
            repeated = family_parameters["running_samples"] + (1 if context in ("repeat_request", "slow_feedback") else 0)
            f = replace(f, status=S.ACCEPTED, behavior=selected, request_id=request_id,
                        execution_id=episode_id+f"-execution-{execution_index}", standing=False)
        else:
            repeated -= 1
            if repeated >= 0:
                f = replace(f, status=S.RUNNING, vx_m_s=.12 if pending == B.ADVANCE else 0.,
                            wz_rad_s=.1 if pending == B.TURN_LEFT else -.1 if pending == B.TURN_RIGHT else 0.)
            else:
                f = replace(f, status=S.COMPLETED, standing=False, vx_m_s=0., wz_rad_s=0.,
                            heading_error_rad=0. if pending in (B.TURN_LEFT, B.TURN_RIGHT) else f.heading_error_rad)
                settling = family_parameters["settling_samples"]
    _require(len(frames) >= 2 and frames[-1]["target"]["behavior"] == "HOLD", "Incomplete generated episode")
    outcome = "rejected" if memory.get("rejected") else "missing_feedback" if not f.valid else "normal_completion"
    return {"episode_id": episode_id, "family_id": f"family-{family:03d}", "split": split, "task": task.value,
            "context": context, "family_parameters": family_parameters, "feedback_source": "simulated_execution_feedback",
            "frames": frames, "outcome": outcome, "sequence_sha256": _digest(frames)}


def generate_dataset(seed=2401):
    from ..control.x2_behavior import Task
    from ..senses.x2_behavior import input_contract
    _require(type(seed) is int and seed >= 0, "Nonnegative integer seed required")
    splits = {name: [_episode(task, name, family, index, seed) for task in Task
                     for index, family in enumerate(families)] for name, families in FAMILIES.items()}
    identity = {"schema": SCHEMA, "seed": seed, "source": "offline engineering demonstration; simulated execution feedback",
                "task_order": [x.value for x in Task], "split_counts": {k: len(v) for k, v in splits.items()},
                "family_partition": {k: list(v) for k, v in FAMILIES.items()},
                "split_rule": "complete episodes; disjoint initial-yaw/timing parameter families; never adjacent-frame shuffling",
                "input_contract_sha256": _digest(input_contract()), "teacher": "teacher_step.v1",
                "teacher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reset_rule": "once at episode start; preserve brain state within episode",
                "split_sha256": {k: _digest(v) for k, v in splits.items()},
                "boundary": "offline labels are not actual robot motion or real simulation feedback; no Spec023 frames included"}
    identity["dataset_id"] = _digest(identity)
    result = {"schema": SCHEMA, "identity": identity, "splits": splits}
    validate_dataset(result)
    return result


def validate_dataset(dataset):
    identity, splits = dataset["identity"], dataset["splits"]
    _require(dataset["schema"] == SCHEMA and set(splits) == set(SPLIT_FILES), "Dataset schema/splits differ")
    _require(identity["dataset_id"] == _digest({k: v for k, v in identity.items() if k != "dataset_id"}), "Dataset identity differs")
    used_episodes, used_families = set(), set()
    for split, episodes in splits.items():
        _require(_digest(episodes) == identity["split_sha256"][split], "Split hash differs")
        families = {e["family_id"] for e in episodes}
        _require(not families.intersection(used_families), "Parameter-family leakage")
        used_families.update(families)
        for e in episodes:
            _require(e["episode_id"] not in used_episodes and e["split"] == split, "Episode leakage")
            used_episodes.add(e["episode_id"])
            _require(e["sequence_sha256"] == _digest(e["frames"]) and len(e["frames"]) >= 2, "Episode evidence differs")
            for i, row in enumerate(e["frames"]):
                o = row["observation"]
                _require(row["frame_index"] == i and o["observation_seq"] == i and o["session_id"] == e["episode_id"], "Episode order/session differs")
                _require("target" not in o and "teacher" not in o, "Teacher leaked into observation")
    return True


def write_dataset(output, *, seed=2401):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = generate_dataset(seed)
    manifest = {"schema": SCHEMA, "identity": dataset["identity"], "files": {}}
    for split, name in SPLIT_FILES.items():
        body = json.dumps(dataset["splits"][split], sort_keys=True, allow_nan=False) + "\n"
        (output / name).write_text(body)
        manifest["files"][split] = {"path": name, "sha256": hashlib.sha256(body.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def load_split(output, split):
    """Read exactly the named partition; training must never name locked_test."""
    _require(split in SPLIT_FILES, "Unknown split")
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest["identity"]
    _require(manifest["schema"] == SCHEMA and identity["dataset_id"] == _digest({k: v for k, v in identity.items() if k != "dataset_id"}), "Manifest identity differs")
    record = manifest["files"][split]
    _require(record["path"] == SPLIT_FILES[split], "Unexpected split path")
    body = (output / record["path"]).read_bytes()
    _require(hashlib.sha256(body).hexdigest() == record["sha256"], "Split file hash differs")
    episodes = json.loads(body)
    _require(_digest(episodes) == identity["split_sha256"][split], "Registered split content differs")
    return episodes


def score_predictions(episodes, predictions):
    """Score raw outputs in supplied episode order; no protection credit.

    This is offline teacher agreement, not physical task completion. Predictions
    are {episode_id: [{'behavior','phase',...}, ...]}; missing rows are an error.
    """
    _require(set(predictions) == {e["episode_id"] for e in episodes}, "Prediction episodes differ")
    behavior_pairs, phase_pairs, cases = [], [], []
    for episode in episodes:
        rows = predictions[episode["episode_id"]]
        _require(len(rows) == len(episode["frames"]), "Prediction frame count differs")
        correct, interventions = 0, 0
        for frame, predicted in zip(episode["frames"], rows):
            target = frame["target"]
            behavior_pairs.append((target["behavior"], predicted["behavior"]))
            phase_pairs.append((target["phase"], predicted["phase"]))
            protection = bool(predicted.get("protection_reason"))
            interventions += int(protection)
            correct += int(target["behavior"] == predicted["behavior"] and target["phase"] == predicted["phase"] and not protection)
        cases.append({"episode_id": episode["episode_id"], "task": episode["task"], "outcome": episode["outcome"],
                      "sequence_match": correct == len(rows), "matched_frames": correct, "frames": len(rows), "protection_interventions": interventions})
    def macro_f1(pairs, labels):
        scores = []
        for label in labels:
            tp = sum(a == label and b == label for a, b in pairs)
            fp = sum(a != label and b == label for a, b in pairs)
            fn = sum(a == label and b != label for a, b in pairs)
            scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        return sum(scores)/len(scores)
    return {"episodes": cases, "sequence_matches": sum(e["sequence_match"] for e in cases), "total": len(cases),
            "behavior_macro_f1": macro_f1(behavior_pairs, ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")),
            "phase_macro_f1": macro_f1(phase_pairs, ("START", "CONTINUE", "FINISH")),
            "boundary": "offline teacher sequence agreement; no robot task-completion claim"}
