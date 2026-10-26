"""Finite stopped-angle correction with scripted sensors, no brain or ROS."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_skill_profile import (
    SIM_SETTLED_ACTION_PROFILE_ID, settled_action_skill_profile, measured_action_skill_profile,
    precision_action_skill_profile, validate_candidate_skill_profile,
)
from malecns_lab.control.x2_interaction import (
    settled_action_interaction_profile, measured_action_interaction_profile, validate_interaction_profile,
)
from malecns_lab.learning.x2_behavior_session import NeuralSession
from .test_x2_action_continuity import rig, step, submit


def settled_rig():
    return rig(settled_action_skill_profile, settled_action_interaction_profile)


def stopped(r, degrees, *, delay=0.):
    r.a = math.radians(degrees)
    end = r.t + delay
    while r.t < end:
        step(r, wz=.1)  # Keep real stop criteria unconfirmed during this scripted delay.
    for _ in range(80):
        step(r)
        if r.core.terminal or r.core.stop_at is None:
            return
    raise AssertionError('Scripted stop was not recognized')


def crossing(r, sign=1):
    step(r)
    commanded = r.core.resolved_velocity[2]
    r.a = sign * math.pi / 2 + math.copysign(.001, commanded)
    step(r, wz=commanded)
    assert r.core.stop_at is not None and r.commands[-1] == (0., 0., 0.)


def test_settled_profile_is_distinct_and_only_its_turn_poll_budget_grows():
    value = settled_action_skill_profile()
    assert value['profile_id'] == SIM_SETTLED_ACTION_PROFILE_ID
    assert validate_candidate_skill_profile(value) == value
    assert value['execution_poll_seconds_by_behavior'] == dict(TURN_LEFT=48., TURN_RIGHT=48.)
    assert 'settled_controller' not in measured_action_skill_profile()['turn_execution']
    assert 'settled_controller' not in precision_action_skill_profile()['turn_execution']
    interaction = settled_action_interaction_profile()
    assert validate_interaction_profile(interaction) == interaction
    baseline = measured_action_interaction_profile()
    assert {k: v for k, v in interaction.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k: v for k, v in baseline.items() if k not in ('profile_id', 'skill_profile_id')}
    for name, forged in [('nonzero_budget_seconds', 11.), ('max_corrections', 3), ('total_deadline_seconds', 47.)]:
        altered = deepcopy(value)
        altered['turn_execution']['settled_controller'][name] = forged
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(altered)
    policy = SimpleNamespace(policy_id='p', encoder=SimpleNamespace(input_map_id='i'),
        identity={'interaction_profile': interaction}, initial_state=lambda *a, **k: None,
        state_identity=lambda _: {'state_id': 's'}, set_runtime_context=lambda *a: None)
    session = NeuralSession(policy, 'c', 'r', 's', None, lambda *a, **k: None, continuous=True)
    assert session.execution_poll_seconds_by_behavior == dict(TURN_LEFT=48., TURN_RIGHT=48.)
    assert session.execution_poll_seconds == 32.


@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_settled_corrections_keep_brain_identity_origin_and_goal_across_long_stops(behavior, sign):
    r = settled_rig()
    assert not submit(r, behavior, phase='CONTINUE')
    assert submit(r, behavior)
    command, origin, identity = r.core.command, r.core.start_yaw, r.core.execution_id
    motion_end, hard_end = r.core.deadline, r.core.settled_hard_deadline
    crossing(r, sign)
    spent = r.core.settled_nonzero_seconds
    stopped(r, sign * 99.05, delay=4.1)
    assert r.core.settled_corrections == 1 and r.core.settled_nonzero_seconds == spent
    assert r.core.resolved_velocity == (0., 0., -sign * .8)
    assert submit(r, behavior, phase='CONTINUE', request=command.request_id)
    crossing(r, sign)
    stopped(r, sign * 96., delay=4.1)
    assert r.t - r.core.started_at > 10. and r.core.settled_corrections == 2
    assert not r.core.terminal
    crossing(r, sign)
    stopped(r, sign * 91.)
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    assert r.core.command == command and r.core.start_yaw == origin and r.core.execution_id == identity
    assert r.core.deadline == motion_end == r.core.started_at + 34.
    assert r.core.settled_hard_deadline == hard_end == r.core.started_at + 46.
    assert len(r.core.events) == 3  # No artificial events for the two corrections.
    trace = r.core.events[-1]['settled_turn']['trace']
    assert len([row for row in trace if row['reason'] == 'measured_stop_confirmed']) == 3
    active = [row for row in trace if row['reason'] == 'nonzero_published']
    zero = [row for row in trace if row['reason'] == 'zero_published']
    assert len(active) == len(zero) == 3
    assert r.core.settled_nonzero_seconds == pytest.approx(sum(b['monotonic']-a['monotonic'] for a, b in zip(active, zero)))
    assert r.core.settled_nonzero_seconds <= 10.


@pytest.mark.parametrize('final_degrees,reason', [(99.1, 'settled_turn_no_improvement'),
                                                (99.05, 'settled_turn_no_improvement')])
def test_non_improving_correction_fails_without_another_attempt(final_degrees, reason):
    r = settled_rig(); assert submit(r, 'TURN_LEFT')
    crossing(r); stopped(r, 99.05)
    crossing(r); stopped(r, final_degrees)
    assert r.core.status == 'FAILED' and r.core.reason == reason
    assert r.core.settled_corrections == 1 and r.commands[-1] == (0., 0., 0.)


def test_two_corrections_exhausted_even_when_stopped_error_keeps_improving():
    r = settled_rig(); assert submit(r, 'TURN_RIGHT')
    for degrees in (-100., -98., -96.):
        crossing(r, -1); stopped(r, degrees)
    assert r.core.status == 'FAILED' and r.core.reason == 'settled_turn_corrections_exhausted'
    assert r.core.settled_corrections == 2


@pytest.mark.parametrize('interruption', ['end', 'emergency', 'stale', 'tilt'])
def test_interrupted_or_unfresh_action_cannot_start_a_correction(interruption):
    r = settled_rig(); assert submit(r, 'TURN_RIGHT')
    step(r)
    r.a = -math.pi / 2
    r.feed()
    if interruption == 'end':
        assert submit(r, 'HOLD', phase='FINISH', request='end')
    elif interruption == 'emergency':
        r.core.cancel()
    elif interruption == 'stale':
        r.t += .3
    else:
        r.core.update_odom((0., 0., .5), (math.sin(.25), 0., 0., math.cos(.25)),
                           (0., 0., 0.), (0., 0., 0.), received=r.t)
    r.core.tick()
    assert r.commands[-1] == (0., 0., 0.)
    stopped(r, -99.05)
    assert r.core.terminal and r.core.status != 'COMPLETED' and r.core.settled_corrections == 0


def test_nonzero_budget_is_shared_by_segments_and_continue_does_not_renew_it():
    r = settled_rig(); assert submit(r, 'TURN_LEFT')
    step(r)
    until = r.t + 6.
    while r.t < until:
        r.a += .002; step(r, wz=.1)
    crossing(r); stopped(r, 99.)
    assert r.core.settled_corrections == 1
    remaining = r.core.settled_segment_deadline - r.t
    assert 0 < remaining < 4.
    before = r.core.settled_segment_deadline
    assert submit(r, 'TURN_LEFT', phase='CONTINUE', request=r.core.command.request_id)
    assert r.core.settled_segment_deadline == before
    while r.core.stop_at is None:
        step(r)
    assert r.core.settled_failure == 'settled_turn_motion_budget'
    assert r.core.settled_nonzero_seconds <= 10. + 1e-8
    stopped(r, 90.)  # Late target feedback cannot rescue budget failure.
    assert r.core.status == 'FAILED' and r.core.settled_corrections == 1


def test_last_twelve_seconds_are_stop_only_and_hard_deadline_cannot_be_renewed():
    r = settled_rig(); assert submit(r, 'TURN_LEFT')
    step(r)
    deadline = r.core.deadline
    r.t = deadline
    r.feed(); r.core.tick()
    assert r.commands[-1] == (0., 0., 0.) and r.core.stop_at is not None
    r.a = math.radians(99.)
    stopped(r, 99.)
    assert r.core.status == 'FAILED' and r.core.settled_corrections == 0
    assert r.core.deadline == deadline and r.core.settled_hard_deadline == deadline + 12.


def test_unconfirmed_stop_remains_unknown_at_original_twelve_second_watchdog():
    r = settled_rig(); assert submit(r, 'TURN_RIGHT')
    crossing(r, -1)
    start = r.core.stop_at
    while not r.core.terminal:
        step(r, wz=.1)
    assert r.core.status == 'UNKNOWN' and r.core.reason == 'settled_turn_stop_timeout'
    assert 12. <= r.t - start <= 12.03 and not r.core.events[-1]['stop_confirmed']
    assert r.core.settled_corrections == 0


@pytest.mark.parametrize('final_degrees', [90., 99.05])
def test_late_stopped_and_standing_cannot_complete_or_start_correction(final_degrees):
    r = settled_rig(); assert submit(r, 'TURN_LEFT')
    crossing(r)
    stop_at, hard_end = r.core.stop_at, r.core.settled_hard_deadline
    r.t = stop_at + 11.01
    r.a = math.radians(final_degrees)
    first_ready = None
    # Fresh sensor callbacks and the lower-level stop observer continue while
    # the skill completion poll is delayed. Do not stub STOPPED or standing.
    for _ in range(65):
        r.feed()
        standing = r.core.standing_ready()
        r.core.motion.tick()
        if first_ready is None and standing and r.core.motion.state == 'STOPPED':
            first_ready = r.t
    assert first_ready is not None and first_ready >= stop_at + 12.
    assert r.core.motion.state == 'STOPPED' and r.core.standing_ready()
    assert not r.core.motion.failures and not r.core.terminal
    r.core.tick()
    assert r.core.terminal and r.core.status == 'UNKNOWN'
    assert r.core.reason == 'settled_turn_stop_timeout'
    assert not r.core.events[-1]['stop_confirmed']
    assert r.core.settled_corrections == 0
    assert r.commands[-1] == r.core.resolved_velocity == r.core.motion.target == (0., 0., 0.)
    assert r.core.stop_at == stop_at and r.core.settled_hard_deadline == hard_end
