"""Protocol-only fixtures: no device, Torch, production brain or simulator."""
from dataclasses import replace
import pytest

from malecns_lab.control.x2_behavior import (
    Behavior as B, Decision, DecisionGate, ExecutionEvent, ExecutionStatus as S,
    Feedback, Observation, Perception, Phase as P, Task as T,
)
from malecns_lab.senses.x2_behavior import BehaviorEncoder, CHANNELS, channel_values


def observation(**changes):
    return replace(Observation("run", "session", 0, "request", T.GREET, 10.,
                              Feedback(valid=True, source="sim_odom", standing=True),
                              capabilities=tuple(B)), **changes)


def decision(behavior=B.WAVE_RIGHT, phase=P.START, **changes):
    values = [0.] * 5
    values[tuple(B).index(behavior)] = 1.
    phases = [0.] * 3
    phases[tuple(P).index(phase)] = 1.
    return replace(Decision("run", "session", 0, "request", 0, "before", "after", tuple(values),
                            tuple(phases), (0.,), behavior, phase, .5, "policy", "checkpoint", "input", 600, 1100), **changes)


def gate():
    g = DecisionGate("run", "session", "policy", "checkpoint", "input", max_age_s=30.)
    g.observe(observation())
    return g


def event(status, seq=0, **changes):
    return replace(ExecutionEvent("run", "session", seq, 0, "exec", "request", "offline_fixture", "pack",
                                  status, B.WAVE_RIGHT, 12.+seq), **changes)


def test_roundtrip_and_raw_decoding_are_bound():
    for message in (observation(), decision(), event(S.ACCEPTED)):
        assert type(message).from_dict(message.to_dict()) == message
    with pytest.raises(ValueError, match="raw logits"):
        replace(decision(), behavior=B.ADVANCE)
    with pytest.raises(ValueError, match="raw parameter"):
        replace(decision(), intensity=.9)
    with pytest.raises(ValueError, match="raw heads"):
        replace(decision(), behavior_logits=(float("nan"), 0., 0., 0., 1.))


@pytest.mark.parametrize("change,now,reason", [({"run_id": "other"}, 11., "cross_run"),
    ({"checkpoint_id": "other"}, 11., "policy_identity"), ({}, 41., "expired_observation"),
    ({}, 9., "expired_observation"), ({"observation_seq": 1}, 11., "superseded_observation")])
def test_admission_rejects_without_rewriting_raw_output(change, now, reason):
    d = decision(**change)
    admission = gate().admit(d, now)
    assert not admission.accepted and admission.reason == reason and admission.decision is d


def test_new_request_duplicate_and_illegal_phase_do_not_emit_alternative_actions():
    g = gate()
    d = decision()
    assert g.admit(d, 11.).accepted
    assert g.admit(d, 11.).reason == "duplicate_or_reordered_decision"
    g.observe(observation(observation_seq=1, request_id="next", task=T.END_INTERACTION, received_monotonic=12.))
    assert g.admit(decision(decision_seq=1, observation_seq=1), 13.).reason == "superseded_request"
    assert g.admit(decision(B.HOLD, P.START, decision_seq=1, observation_seq=1, request_id="next"), 13.).reason == "invalid_hold_phase"
    # A valid neural HOLD asks for stopping but does not mark the active skill completed.
    assert g.admit(decision(B.HOLD, P.FINISH, decision_seq=1, observation_seq=1, request_id="next"), 13.).accepted
    assert g.active is not None


def test_execution_ack_cannot_be_completion_and_invalid_event_is_atomic():
    g = gate()
    assert g.admit(decision(), 11.).accepted
    g.execution_event(event(S.ACCEPTED))
    with pytest.raises(ValueError, match="ACK"):
        g.execution_event(event(S.COMPLETED, 1))
    assert g.last_event_seq == 0 and g.active["status"] == S.ACCEPTED
    g.execution_event(event(S.STARTED, 1))
    with pytest.raises(ValueError, match="Concurrent"):
        g.execution_event(event(S.COMPLETED, 2, execution_id="unrelated"))
    g.execution_event(event(S.COMPLETED, 2, stop_evidence={"joints_settled": True}))
    assert g.active is None
    with pytest.raises(ValueError, match="Duplicate"):
        g.execution_event(event(S.COMPLETED, 2))


def test_single_inflight_and_unavailable_skill():
    g = gate()
    g.observe(observation(observation_seq=1, capabilities=(B.HOLD,), received_monotonic=10.1))
    assert g.admit(decision(observation_seq=1), 11.).reason == "unavailable_skill"
    g = gate()
    assert g.admit(decision(), 11.).accepted
    assert g.admit(decision(B.TURN_LEFT, decision_seq=1), 12.).reason == "skill_already_active"
    assert g.admit(decision(B.WAVE_RIGHT, P.CONTINUE, decision_seq=1), 12.).reason == "awaiting_execution_acceptance"


def test_engineering_channels_use_masks_and_no_identity_or_teacher_inputs():
    o = observation()
    values = channel_values(o)
    assert tuple(values) == CHANNELS and all(0 <= x <= 1 for x in values.values())
    assert values == channel_values(replace(o, run_id="different", session_id="different", observation_seq=100, received_monotonic=999.))
    changed = channel_values(replace(o, task=T.FACE_LEFT))
    assert changed != values
    stale = replace(o, feedback=Feedback(valid=False, source="missing", yaw_rad=999., vx_m_s=-999., age_s=99.))
    invalid = channel_values(stale)
    assert invalid["feedback.valid"] == 0 and invalid["body.yaw.positive"] == 0 and invalid["feedback.age"] == 1
    with pytest.raises(ValueError, match="Missing perception"):
        Perception(visual_cells=(1.,) * 16)
    with pytest.raises(ValueError, match="Validated production"):
        BehaviorEncoder.from_policy_map(None)
