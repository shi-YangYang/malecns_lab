"""Offline input/cadence contract checks; no graph, model, ROS or device IO."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from malecns_lab.control.x2_behavior import Feedback, Observation, Task
from malecns_lab.control.x2_interaction import (interaction_profile, profile_capabilities,
    relative_heading_goal, relative_heading_error)
from malecns_lab.control.x2_skill_profile import candidate_skill_profile
from malecns_lab.experiments.x2_behavior_bridge import RemoteExchange, registered_skill_profile
from malecns_lab.learning.x2_behavior_session import SCHEMA, CAPABILITIES
from malecns_lab.learning.x2_interaction_data import generate_dataset, validate_dataset


def test_shared_relative_goal_keeps_request_origin_and_zeros_other_tasks():
    goal = relative_heading_goal("FACE_LEFT", 3.)
    assert relative_heading_error("FACE_LEFT", goal, 3.) == pytest.approx(.45)
    assert relative_heading_error("FACE_LEFT", goal, -3.1) == pytest.approx(goal["target_yaw_rad"]+3.1)
    assert goal["origin_yaw_rad"] == 3.
    for task in ("GREET", "SHOW_FORWARD", "END_INTERACTION"):
        assert relative_heading_goal(task, .9) is None
        assert relative_heading_error(task, None, -.6) == 0.
    with pytest.raises(ValueError, match="changed"):
        relative_heading_error("FACE_RIGHT", goal, 3.)


def test_bridge_uses_the_same_goal_and_capabilities_as_offline_sessions():
    profile = interaction_profile()
    assert profile_capabilities(profile) == CAPABILITIES
    for index, task in enumerate(Task):
        remote = RemoteExchange("run", "session", lambda: 10.)
        query = dict(schema=SCHEMA, kind="observe", run_id="run", session_id="session", task=str(task),
            sequence=1, nonce=f"{index:032x}", request_id="request", last_consumed_request_id=None,
            host_sent_monotonic=100.)
        yaw = .27
        odom = {"sequence": 1, "remote_received_monotonic": 10.,
            "message": {"pose": {"pose": {"orientation": {"x": 0., "y": 0., "z": math.sin(yaw/2), "w": math.cos(yaw/2)}}}}}
        packet = remote.observe(query, Feedback(valid=True, age_s=0., source="fixture",
            standing=True, yaw_rad=yaw, heading_error_rad=.45), [], odom=odom)
        goal = relative_heading_goal(task, yaw)
        assert packet["feedback"]["heading_error_rad"] == pytest.approx(relative_heading_error(task, goal, yaw))
        assert tuple(packet["capabilities"]) == profile_capabilities(profile)


def test_host_and_bridge_require_same_explicit_interaction_profile(tmp_path):
    reg = dict(run_id="run", session_id="session", qualification_only=True, simulation_only=True,
               wave_enabled=False, skill_profile=candidate_skill_profile(), interaction_profile=interaction_profile())
    (tmp_path / "behavior_registration.json").write_text(json.dumps(reg))
    assert registered_skill_profile(tmp_path, reg) == candidate_skill_profile()
    changed = deepcopy(reg)
    changed["interaction_profile"]["capabilities"].append("WAVE_RIGHT")
    with pytest.raises(ValueError, match="Interaction profile"):
        registered_skill_profile(tmp_path, changed)


def test_continuous_dataset_preserves_requests_feedback_and_gate_only_waits():
    dataset = generate_dataset()
    assert validate_dataset(dataset)
    assert dataset["identity"]["locked_test_opened"] is False
    assert "never reset on request change" in dataset["identity"]["reset_rule"]
    session_ids = set()
    for split, episode in [(split, episode) for split, episodes in dataset["splits"].items() for episode in episodes]:
        assert episode["episode_id"] not in session_ids
        session_ids.add(episode["episode_id"])
        frames = episode["frames"]
        assert {row["observation"]["session_id"] for row in frames} == {episode["episode_id"]}
        assert {row["observation"]["task"] for row in frames} == {task.value for task in Task}
        for row in frames:
            o = Observation.from_dict(row["observation"])
            assert tuple(map(str, o.capabilities)) == CAPABILITIES
            assert o.feedback.source == "simulated_execution_feedback"
            assert row["target"]["behavior"] != "WAVE_RIGHT"
            if o.task not in (Task.FACE_LEFT, Task.FACE_RIGHT):
                assert o.feedback.heading_error_rad == 0.
            if o.feedback.status in ("ACCEPTED", "STARTED", "RUNNING"):
                assert o.task == "END_INTERACTION" and o.request_changed
                assert o.feedback.request_id != o.request_id
                assert row["target"]["behavior"] == "HOLD"
                assert row["gate_only_feedback"][0]["feedback"]["status"] == "COMPLETED"
            assert all(poll["neural_consumed"] is False for poll in row["gate_only_feedback"])
        requests = episode["requests"]
        repeated = [(a, b) for a,b in zip(requests, requests[1:]) if a["task"] == b["task"] == "SHOW_FORWARD"]
        assert repeated and repeated[0][0]["request_id"] != repeated[0][1]["request_id"]
        assert any(r["interrupts_active_request"] and r["task"] == "END_INTERACTION" for r in requests)
        assert next(r for r in requests if r["task"] == "GREET")["outcome"] == "capability_unavailable"
    assert [r["task"] for r in dataset["splits"]["train"][0]["requests"]] != [r["task"] for r in dataset["splits"]["validation"][0]["requests"]]


def test_two_train_sessions_have_different_order_and_feedback_not_just_ids():
    first, second = generate_dataset()["splits"]["train"]
    assert first["family_parameters"] != second["family_parameters"]
    assert [r["task"] for r in first["requests"]] != [r["task"] for r in second["requests"]]
    assert first["frames"][0]["observation"]["feedback"]["age_s"] != second["frames"][0]["observation"]["feedback"]["age_s"]
    validation, = generate_dataset()["splits"]["validation"]
    assert [r["task"] for r in validation["requests"]] == ["FACE_RIGHT", "GREET", "SHOW_FORWARD",
        "SHOW_FORWARD", "END_INTERACTION", "FACE_LEFT", "END_INTERACTION"]
    assert validation["family_parameters"] == {"yaw": -.31, "source_age": .09, "turn": .09, "inference_s": 8.1}


def test_shared_relative_encoder_matches_diagnostic_and_keeps_all_other_ports():
    import numpy as np
    from .test_x2_yaw_diagnostic import EncodingFixture
    from malecns_lab.learning.x2_yaw_diagnostic import RelativeHeadingEncoder as DiagnosticEncoder
    from malecns_lab.senses.x2_relative_heading import RelativeHeadingEncoder
    original = EncodingFixture()
    encoder, diagnostic = RelativeHeadingEncoder(original), DiagnosticEncoder(original)
    frame = generate_dataset()["splits"]["train"][0]["frames"][0]
    observation = Observation.from_dict(frame["observation"])
    assert encoder.identity == diagnostic.identity and encoder.input_map_id != encoder.parent_input_map_id
    np.testing.assert_array_equal(encoder.encode(observation), diagnostic.encode(observation))
    changed = replace(observation, feedback=replace(observation.feedback, yaw_rad=-1.7))
    np.testing.assert_array_equal(encoder.encode(observation), encoder.encode(changed))
    relative = replace(observation, feedback=replace(observation.feedback, heading_error_rad=-.45))
    assert np.any(encoder.encode(observation) != encoder.encode(relative))
    assert observation.to_dict() == frame["observation"]
    before, after = original.encode(observation), encoder.encode(observation)
    assert set(np.flatnonzero(before != after)).issubset(encoder.suppressed_indices)
    assert len(encoder.suppressed_indices) == 118
