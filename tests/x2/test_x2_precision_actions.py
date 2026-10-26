"""Opt-in heading feedback contract; synthetic observations, no ROS or brain."""
from copy import deepcopy
import math

import pytest

from malecns_lab.control.x2_skill_profile import (
    SIM_PRECISION_ACTION_PROFILE_ID, precision_action_skill_profile,
    measured_action_skill_profile, tapered_action_skill_profile, validate_candidate_skill_profile,
)
from malecns_lab.control.x2_interaction import (
    precision_action_interaction_profile, measured_action_interaction_profile, validate_interaction_profile,
)
from .test_x2_action_continuity import rig, step, submit


def precision_rig():
    return rig(precision_action_skill_profile, precision_action_interaction_profile)


def settle(r):
    for _ in range(80):
        step(r)
        if r.core.terminal or r.core.stop_at is None:
            break


def test_precision_is_opt_in_and_preserves_previous_contracts_and_neural_inputs():
    value = precision_action_skill_profile()
    assert value['profile_id'] == SIM_PRECISION_ACTION_PROFILE_ID
    assert validate_candidate_skill_profile(value) == value
    old = measured_action_skill_profile()
    assert 'feedback_controller' not in old['turn_execution']
    assert 'feedback_controller' not in tapered_action_skill_profile()['turn_execution']
    for key in ('advance_execution', 'wave_contract', 'command_seconds_by_behavior'):
        assert value[key] == old[key]
    current_input = precision_action_interaction_profile()
    old_input = measured_action_interaction_profile()
    assert validate_interaction_profile(current_input) == current_input
    assert {k: v for k, v in current_input.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k: v for k, v in old_input.items() if k not in ('profile_id', 'skill_profile_id')}
    for key, forged_value in [('max_corrections', 3), ('final_tolerance_rad', .2), ('total_deadline_seconds', 23.)]:
        forged = deepcopy(value)
        forged['turn_execution']['feedback_controller'][key] = forged_value
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(forged)


@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_pd_uses_actual_error_and_rate_and_does_not_stop_at_fast_crossing(behavior, sign):
    r = precision_rig()
    assert not submit(r, behavior, phase='CONTINUE')
    assert all(command == (0., 0., 0.) for command in r.commands)
    assert submit(r, behavior)
    step(r)
    assert r.commands[-1] == (0., 0., sign * .8)
    r.a = sign * (math.pi / 2 - .2)
    step(r, wz=sign * .6)
    assert r.commands[-1] == pytest.approx((0., 0., sign * .25))
    r.a = sign * (math.pi / 2 + .01)
    step(r, wz=sign * .4)
    assert r.commands[-1] == pytest.approx((0., 0., -sign * .12))
    assert r.core.turn_target_reached and r.core.stop_at is None
    assert not r.core.terminal


@pytest.mark.parametrize('degrees', [88., 92.])
@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_symmetric_final_error_requires_real_stop_and_retains_measured_angle(degrees, behavior, sign):
    r = precision_rig(); assert submit(r, behavior)
    r.a = sign * math.radians(degrees); step(r)
    assert r.core.stop_at is not None and not r.core.terminal
    assert r.commands[-1] == (0., 0., 0.)
    settle(r)
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    assert r.core.events[-1]['angle_rad'] == pytest.approx(sign * math.radians(degrees))
    assert r.core.turn_target_reached == (degrees > 90.)
    assert r.core.precision_corrections == 0
    assert submit(r, 'HOLD', phase='FINISH')
    assert r.core.terminal and r.core.resolved_velocity == (0., 0., 0.)
    assert submit(r, behavior)  # Reuse only after the preceding measured stop.
    assert r.core.signed_angle == 0. and r.core.precision_corrections == 0


@pytest.mark.parametrize('final_degrees', [99.05, 84.46])
def test_measured_recoil_is_corrected_inside_same_skill_without_new_brain_events(final_degrees):
    r = precision_rig(); assert submit(r, 'TURN_RIGHT')
    origin, identity, deadline = r.core.start_yaw, r.core.execution_id, r.core.deadline
    request = r.core.command.request_id
    r.a = -math.radians(90.1); step(r, wz=-.1)
    r.a = -math.radians(final_degrees)
    settle(r)
    assert r.core.precision_corrections == 1 and not r.core.terminal
    assert r.core.stop_at is None
    assert r.core.start_yaw == origin and r.core.execution_id == identity and r.core.command.request_id == request
    assert r.core.deadline == deadline and r.core.motion.intent_deadline == deadline
    assert r.core.resolved_velocity[2] * (1 if final_degrees > 90 else -1) > 0
    assert len(r.core.events) == 2  # Original ACCEPTED and measured STARTED only.
    assert submit(r, 'TURN_RIGHT', phase='CONTINUE', request=request)
    assert r.core.deadline == deadline
    r.a = -math.pi / 2; step(r)
    settle(r)
    assert r.core.status == 'COMPLETED'
    trace = r.core.events[-1]['precision_turn']['trace']
    stops = [row for row in trace if row['phase'] == 'stopped']
    assert len(stops) == 2 and stops[0]['angle_rad'] == pytest.approx(-math.radians(final_degrees))
    assert len(r.core.events) == 3


def test_corrections_are_finite_and_overshoot_never_passes_as_goal_completion():
    r = precision_rig(); assert submit(r, 'TURN_LEFT')
    for attempt in range(3):
        r.a = math.pi / 2; step(r)
        r.a = math.radians(99.05)
        settle(r)
        if attempt < 2:
            assert not r.core.terminal and r.core.precision_corrections == attempt + 1
    assert r.core.status == 'FAILED' and r.core.reason == 'precision_turn_corrections_exhausted'
    assert r.core.precision_corrections == 2 and r.core.events[-1]['stop_confirmed']
    assert r.commands[-1] == (0., 0., 0.)


def test_stalled_control_stops_and_late_arrival_cannot_rescue_failure():
    r = precision_rig(); assert submit(r, 'TURN_LEFT')
    while r.core.stop_at is None:
        step(r)
    assert r.core.precision_failure == 'precision_turn_stalled'
    assert r.t - r.core.started_at < 2.05
    r.a = math.pi / 2
    settle(r)
    assert r.core.status == 'FAILED' and r.core.reason == 'precision_turn_stalled'
    assert r.core.precision_corrections == 0 and r.commands[-1] == (0., 0., 0.)


@pytest.mark.parametrize('interruption', ['end', 'emergency', 'stale'])
def test_interruption_never_restarts_internal_correction(interruption):
    r = precision_rig(); assert submit(r, 'TURN_RIGHT')
    r.a = -math.pi / 2; step(r)
    r.a = -math.radians(99.05); settle(r)
    assert r.core.precision_corrections == 1
    if interruption == 'end':
        assert submit(r, 'HOLD', phase='FINISH', request='END')
    elif interruption == 'emergency':
        r.core.cancel()
    else:
        r.t += .3
    r.core.tick()
    assert r.commands[-1] == (0., 0., 0.)
    settle(r)
    assert r.core.terminal and r.core.status == ('CANCELLED' if interruption == 'end' else 'UNKNOWN' if interruption == 'emergency' else 'FAILED')
    assert r.core.precision_corrections == 1 and r.commands[-1] == (0., 0., 0.)


def test_motion_deadline_is_absolute_and_terminal_late_feedback_cannot_pass():
    r = precision_rig(); assert submit(r, 'TURN_LEFT')
    deadline = r.core.deadline
    while r.t < deadline:
        r.a += .002
        step(r, wz=.2)
    assert r.core.precision_failure == 'precision_turn_motion_deadline'
    assert r.core.deadline == deadline and r.core.precision_hard_deadline == r.core.started_at + 22.
    assert r.commands[-1] == (0., 0., 0.)
    r.a = math.pi / 2; settle(r)
    assert r.core.status == 'FAILED' and r.core.reason == 'precision_turn_motion_deadline'


def test_stop_watchdog_preserves_unknown_when_rotation_does_not_stop():
    r = precision_rig(); assert submit(r, 'TURN_LEFT')
    r.a = math.pi / 2; step(r)
    while not r.core.terminal:
        step(r, wz=.1)
    assert r.core.status == 'UNKNOWN' and r.core.reason == 'precision_turn_stop_timeout'
    assert not r.core.events[-1]['stop_confirmed']
    assert r.core.precision_corrections == 0 and r.commands[-1] == (0., 0., 0.)


def test_stale_feedback_at_settle_threshold_cannot_bypass_active_guard():
    r = precision_rig(); assert submit(r, 'TURN_LEFT')
    r.a = math.pi / 2
    r.feed()  # Fresh measured target, but no executor tick yet.
    r.t += .3
    r.core.tick()
    assert 'odom_unavailable' in r.core.motion.failures
    assert r.commands[-1] == (0., 0., 0.)
    settle(r)
    assert r.core.status == 'FAILED'
    assert r.core.precision_corrections == 0
