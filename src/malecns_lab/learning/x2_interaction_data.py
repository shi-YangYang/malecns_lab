"""Continuous offline request sessions matching the desktop observation contract.

Feedback and execution timing are explicitly simulated fixtures, not robot
evidence. Teachers are used only here. No old dataset is read or modified.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

from ..control.x2_behavior import Behavior as B, ExecutionStatus as S, Feedback, Observation, Perception, Task, digest, require
from ..control.x2_interaction import (interaction_profile, validate_interaction_profile, profile_capabilities,
    relative_heading_goal, relative_heading_error, wrap_heading)
from .x2_behavior_data import teacher_step

SCHEMA = "malecns_lab.x2_continuous_interaction_dataset.v1"
ORDERS = {
    "train-a": (("SHOW_FORWARD", False), ("FACE_LEFT", False), ("SHOW_FORWARD", False),
              ("SHOW_FORWARD", True), ("END_INTERACTION", False), ("FACE_RIGHT", False),
              ("GREET", False), ("END_INTERACTION", False)),
    "train-b": (("FACE_RIGHT", False), ("SHOW_FORWARD", False), ("SHOW_FORWARD", False),
                ("FACE_LEFT", False), ("GREET", False), ("SHOW_FORWARD", True),
                ("END_INTERACTION", False), ("END_INTERACTION", False)),
    "validation": (("FACE_RIGHT", False), ("GREET", False), ("SHOW_FORWARD", False),
                   ("SHOW_FORWARD", True), ("END_INTERACTION", False), ("FACE_LEFT", False),
                   ("END_INTERACTION", False)),
}
FAMILIES = {
    "train-a": {"yaw": .13, "source_age": .04, "turn": .06, "inference_s": 7.2},
    "train-b": {"yaw": -1.12, "source_age": .07, "turn": .08, "inference_s": 6.8},
    "validation": {"yaw": -.31, "source_age": .09, "turn": .09, "inference_s": 8.1},
}
SPLITS = {"train": ("train-a", "train-b"), "validation": ("validation",)}


def _session(split, family_name, profile, seed, *, order=None, family=None):
    # Different orders and fixture families; no adjacent-frame split.
    family = dict(FAMILIES[family_name] if family is None else family)
    order = ORDERS[family_name] if order is None else tuple(order)
    session_id = digest({"schema": SCHEMA, "split": split, "family": family,
                         "seed": seed, "profile_id": profile["profile_id"]})
    caps = profile_capabilities(profile)
    feedback = Feedback(valid=True, age_s=family["source_age"], source="simulated_execution_feedback",
                        standing=True, yaw_rad=family["yaw"], mc_mode="STAND_DEFAULT")
    frames, requests, memory = [], [], {}
    now, observation_seq, execution_seq = 10., 0, 0
    pending_completion = None

    def complete(active):
        turn = family["turn"] * (1 if active.behavior == B.TURN_LEFT else -1 if active.behavior == B.TURN_RIGHT else 0)
        return replace(active, status=S.COMPLETED, standing=True, yaw_rad=wrap_heading(active.yaw_rad+turn),
                       vx_m_s=0., wz_rad_s=0., mc_mode="STAND_DEFAULT")

    def gate_poll(row, value, purpose, elapsed):
        nonlocal now, observation_seq
        now += elapsed
        observation_seq += 1
        value = replace(value, heading_error_rad=relative_heading_error(task, goal, value.yaw_rad))
        row["gate_only_feedback"].append({"observation_seq": observation_seq, "received_monotonic": now,
            "purpose": purpose, "feedback": value.__dict__, "neural_consumed": False})

    for index, (task, interrupt) in enumerate(order):
        request_id = f"{session_id}-request-{index}"
        goal = relative_heading_goal(task, feedback.yaw_rad)
        request = {"request_id": request_id, "task": task, "request_index": index,
                   "interrupts_active_request": feedback.status in (S.ACCEPTED, S.STARTED, S.RUNNING),
                   "will_be_interrupted": interrupt, "goal": goal, "first_frame": len(frames),
                   "capability_limitation": "wave_unavailable" if task == "GREET" else None}
        requests.append(request)
        for within_request in range(3):
            feedback = replace(feedback, heading_error_rad=relative_heading_error(task, goal, feedback.yaw_rad))
            observation_seq += 1
            observation = Observation(session_id, session_id, observation_seq, request_id, task, now,
                feedback, Perception(), caps, within_request == 0)
            target = teacher_step(observation, memory)
            row = {"frame_index": len(frames), "request_index": index, "observation": observation.to_dict(),
                   "target": target, "request_goal": goal, "gate_only_feedback": []}
            frames.append(row)
            # Actual worker refreshes execution events after inference without
            # replacing the observation that produced this neural decision.
            now += family["inference_s"]
            if pending_completion is not None:
                feedback = pending_completion
                pending_completion = None
            gate_poll(row, feedback, "post_inference_event_refresh", .02)
            execution_seq += 1
            chosen = B(target["behavior"])
            if chosen == B.HOLD:
                require(target["phase"] == "FINISH", "Offline HOLD must request finish")
                feedback = replace(feedback, status=S.COMPLETED, behavior=B.HOLD, standing=True,
                    request_id=request_id, execution_id=f"{session_id}-execution-{execution_seq}",
                    vx_m_s=0., wz_rad_s=0., mc_mode="STAND_DEFAULT")
                gate_poll(row, feedback, "execution_event_poll", .1)
                request["end_frame"] = len(frames)
                request["outcome"] = "capability_unavailable" if task == "GREET" else "fixture_hold_confirmed"
                break
            require(target["phase"] == "START" and chosen.value in caps, "Unexpected offline action")
            active = replace(feedback, status=S.ACCEPTED, behavior=chosen, standing=False,
                request_id=request_id, execution_id=f"{session_id}-execution-{execution_seq}", mc_mode="WALK")
            gate_poll(row, active, "execution_event_poll", .1)
            active = replace(active, status=S.RUNNING, vx_m_s=.12 if chosen == B.ADVANCE else 0.,
                             wz_rad_s=.1 if chosen == B.TURN_LEFT else -.1 if chosen == B.TURN_RIGHT else 0.)
            gate_poll(row, active, "execution_event_poll", .1)
            if interrupt:
                # New END arrives during active execution. The previous finite
                # pulse completes while the next full-brain decision computes.
                require(index+1 < len(order) and order[index+1][0] == "END_INTERACTION",
                        "Interrupted fixture must be followed by ordinary END")
                feedback, pending_completion = active, complete(active)
                request["end_frame"], request["outcome"] = len(frames), "new_request_during_execution"
                break
            feedback = complete(active)
            gate_poll(row, feedback, "execution_event_poll", 1. if chosen == B.ADVANCE else 2.)
        else:
            raise ValueError("Continuous fixture did not finish its request")
    return {"episode_id": session_id, "family_id": f"continuous-{family_name}", "split": split,
            "task": "CONTINUOUS_INTERACTION", "context": "multiple_requests_with_repeat_and_interrupt",
            "family_parameters": family, "requests": requests, "frames": frames,
            "feedback_source": "simulated_execution_feedback", "outcome": "offline_fixture_no_wave_qualification",
            "sequence_sha256": digest(frames)}


def generate_dataset(*, profile=None, seed=2401):
    profile = validate_interaction_profile(interaction_profile() if profile is None else profile)
    require(type(seed) is int and seed >= 0, "Nonnegative fixture seed required")
    splits = {split: [_session(split, family, profile, seed) for family in families] for split, families in SPLITS.items()}
    payload = {"schema": SCHEMA, "profile": profile, "seed": seed,
        "source": "offline engineering teacher and simulated feedback; no actual robot evidence",
        "reset_rule": profile["state_reset"], "sampling": profile["sampling"], "locked_test_opened": False,
        "session_families": {k:list(v) for k,v in SPLITS.items()},
        "split_rule": "complete continuous sessions; fixed held-out request order and feedback family",
        "teacher_source_sha256": hashlib.sha256(Path(__file__).with_name("x2_behavior_data.py").read_bytes()).hexdigest(),
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_sha256": {name: digest(rows) for name, rows in splits.items()}}
    result = {"schema": SCHEMA, "identity": {**payload, "dataset_id": digest(payload)}, "splits": splits}
    validate_dataset(result)
    return result


def validate_dataset(dataset):
    identity, splits = dataset["identity"], dataset["splits"]
    require(dataset["schema"] == identity["schema"] == SCHEMA and set(splits) == set(SPLITS), "Continuous dataset schema differs")
    require(identity["dataset_id"] == digest({k:v for k,v in identity.items() if k != "dataset_id"}), "Dataset identity differs")
    caps = profile_capabilities(identity["profile"])
    used = set()
    for split, episodes in splits.items():
        require(identity["split_sha256"][split] == digest(episodes), "Continuous split changed")
        for episode in episodes:
            require(episode["episode_id"] not in used and episode["split"] == split, "Session leakage")
            used.add(episode["episode_id"])
            require(episode["sequence_sha256"] == digest(episode["frames"]), "Continuous frames changed")
            previous, consumed, goals = -1, None, {}
            for i, frame in enumerate(episode["frames"]):
                observation = Observation.from_dict(frame["observation"])
                require(frame["frame_index"] == i and observation.session_id == episode["episode_id"]
                    and observation.observation_seq > previous and tuple(map(str, observation.capabilities)) == caps,
                    "Continuous observation identity/order/profile differs")
                require(observation.request_changed == (observation.request_id != consumed), "Request edge differs from neural consumption")
                require(observation.request_id not in goals or goals[observation.request_id] == frame["request_goal"], "Goal moved within request")
                goals[observation.request_id] = frame["request_goal"]
                require(abs(observation.feedback.heading_error_rad-relative_heading_error(
                    observation.task, frame["request_goal"], observation.feedback.yaw_rad)) <= 1e-8, "Relative input differs")
                previous = observation.observation_seq
                for poll in frame["gate_only_feedback"]:
                    require(poll["neural_consumed"] is False and poll["observation_seq"] > previous,
                            "Gate polling became neural input or reordered")
                    previous = poll["observation_seq"]
                consumed = observation.request_id
            require({r["task"] for r in episode["requests"]} == {t.value for t in Task}, "Missing requested task")
    return True
