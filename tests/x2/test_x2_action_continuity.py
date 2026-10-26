"""Executor continuity with fake measured feedback; no neural/physical claim."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import DecisionGate
from malecns_lab.control.x2_interaction import (
    feedback_full_interaction_profile, frozen_full_interaction_profile,
    validate_interaction_profile, high_turn_full_interaction_profile, measured_action_interaction_profile,
)
from malecns_lab.control.x2_skill_profile import (
    SIM_FEEDBACK_FULL_PROFILE_ID, feedback_full_skill_profile,
    frozen_full_skill_profile, validate_candidate_skill_profile,
    SIM_HIGH_TURN_FULL_PROFILE_ID, high_turn_full_skill_profile, measured_action_skill_profile,
)
from malecns_lab.control.x2_skills import RIGHT_ARM, SkillCommand, validate_command_durations
from malecns_lab.experiments.x2_behavior_bridge import candidate_executor, runtime_context_snapshot
from .test_x2_skills import Rig


ACTIONS = ('ADVANCE', 'TURN_LEFT', 'TURN_RIGHT', 'WAVE_RIGHT')


def rig(profile_factory=measured_action_skill_profile, interaction_factory=measured_action_interaction_profile):
    r = Rig(qualified=True)
    reg = dict(run_id='run', session_id='session', simulation_only=True,
        qualification_only=True, wave_enabled=True, skill_profile=profile_factory(),
        interaction_profile=interaction_factory())
    r.core = candidate_executor(reg, r.commands.append, preset=r.presets.append,
        clock=lambda: r.t, wave_poses=dict(first=dict.fromkeys(RIGHT_ARM, 0.),
                                         terminal=dict.fromkeys(RIGHT_ARM, 0.)))
    r.core.motion.arm(); r.core.motion.confirm_source_registration()
    r.seq = 0
    for _ in range(65):
        step(r)
    return r


def step(r, *, speed=0., wz=0.):
    r.feed(speed=speed, wz=wz)
    r.core.standing_ready()
    r.core.tick()


def submit(r, behavior, *, phase='START', request=None):
    r.seq += 1
    return r.core.submit(SkillCommand(r.seq, request or f'request-{r.seq}', behavior, phase, .5, r.t))


def complete(r, behavior):
    """Drive one admitted action from synthetic sensors, retaining same executor."""
    if behavior == 'WAVE_RIGHT':
        r.core.preset_query(0, 0, 1, received=r.t)
    assert submit(r, behavior)
    execution_id = r.core.execution_id
    assert r.core.signed_angle == r.core.forward_distance == 0.
    assert not r.core.moved and not r.core.turn_target_reached
    if behavior == 'WAVE_RIGHT':
        r.core.preset_dispatch()
        assert r.core.preset_sent(r.t)
        assert r.core.preset_ack(0, 0, 400)
        r.player, r.joint = 2, .3
        r.feed()
        r.core.preset_query(0, 0, 400, received=r.t)
        r.core.tick()
        assert r.core.status == 'STARTED'
        r.player = 0; r.joint = 0.
    for _ in range(400):
        speed = wz = 0.
        if behavior != 'WAVE_RIGHT' and r.core.stop_at is None:
            if behavior == 'ADVANCE':
                speed = .2
                r.x += .004 * math.cos(r.core.start_yaw)
                r.y += .004 * math.sin(r.core.start_yaw)
            else:
                wz = r.core.resolved_velocity[2]
                r.a += wz * .02
        r.feed(speed=speed, wz=wz)
        if behavior == 'WAVE_RIGHT':
            r.core.preset_query(0, 0, 1, received=r.t)
        r.core.standing_ready(); r.core.tick()
        if r.core.terminal:
            break
    assert r.core.terminal and r.core.status == 'COMPLETED'
    assert r.core.events[-1]['stop_confirmed']
    assert r.commands[-1] == (0., 0., 0.)
    assert not r.core.motion.failures
    if behavior.startswith('TURN_'):
        assert r.core.turn_target_reached
    assert runtime_context_snapshot(r.core, SimpleNamespace(valid=True), -1)['protection_state'] == 'READY'
    return execution_id


def test_feedback_profile_is_new_and_keeps_old_contract():
    old = frozen_full_skill_profile(); new = feedback_full_skill_profile(); high = high_turn_full_skill_profile()
    assert old['profile_id'] != new['profile_id'] == SIM_FEEDBACK_FULL_PROFILE_ID
    assert old['command_seconds_by_behavior'] == dict(ADVANCE=1., TURN_LEFT=2., TURN_RIGHT=2.)
    assert new['command_seconds_by_behavior'] == dict(ADVANCE=1., TURN_LEFT=5., TURN_RIGHT=5.)
    assert high['profile_id'] == SIM_HIGH_TURN_FULL_PROFILE_ID and high['profile_id'] != new['profile_id']
    assert (new['turn_execution']['left_rad_s'], new['turn_execution']['right_rad_s']) == (.4, -.4)
    assert (high['turn_execution']['left_rad_s'], high['turn_execution']['right_rad_s']) == (.8, -.8)
    comparable = deepcopy(high)
    comparable['profile_id'] = new['profile_id']
    for key in ('left_rad_s', 'right_rad_s', 'evidence_scope'):
        comparable['turn_execution'][key] = new['turn_execution'][key]
    assert comparable == new
    for profile in (old, new, high):
        assert validate_candidate_skill_profile(profile) == profile
    old_input = frozen_full_interaction_profile(); new_input = feedback_full_interaction_profile()
    high_input = high_turn_full_interaction_profile()
    for profile in (old_input, new_input, high_input):
        assert validate_interaction_profile(profile) == profile
    assert {k:v for k,v in old_input.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k:v for k,v in new_input.items() if k not in ('profile_id', 'skill_profile_id')}
    assert {k:v for k,v in high_input.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k:v for k,v in new_input.items() if k not in ('profile_id', 'skill_profile_id')}
    DecisionGate('run', 'session', 'policy', 'checkpoint', 'input', sim_wave_profile_id=new['profile_id'])
    DecisionGate('run', 'session', 'policy', 'checkpoint', 'input', sim_wave_profile_id=high['profile_id'])
    for change in ({'command_seconds': 6.}, {'left_rad_s': .5}, {'measured_stop_angle_rad': .02}):
        forged = deepcopy(new); forged['turn_execution'].update(change)
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(forged)
    with pytest.raises(ValueError):
        validate_command_durations(new['command_seconds_by_behavior'])


@pytest.mark.parametrize('profile,interaction,velocity,duration', [
    (frozen_full_skill_profile, frozen_full_interaction_profile, .4, 2.),
    (feedback_full_skill_profile, feedback_full_interaction_profile, .4, 5.),
    (high_turn_full_skill_profile, high_turn_full_interaction_profile, .8, 5.),
    (measured_action_skill_profile, measured_action_interaction_profile, .8, 10.)])
@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_profile_velocity_is_the_actual_published_pure_turn(profile, interaction, velocity, duration, behavior, sign):
    r = rig(profile, interaction)
    assert submit(r, behavior)
    r.core.tick()
    assert r.commands[-1] == r.core.resolved_velocity == (0., 0., velocity * sign)
    assert r.core.deadline - r.core.started_at == pytest.approx(duration)
    assert r.core.events[-1]['model_intensity'] == .5
    r.a += sign * (r.core.turn_target_rad + .01)
    step(r, wz=sign * velocity)
    assert r.commands[-1] == (0., 0., 0.) and r.core.turn_target_reached
    assert not r.core.terminal  # The unchanged measured stop window is still required.


@pytest.mark.parametrize('first', ACTIONS)
@pytest.mark.parametrize('second', ACTIONS)
def test_all_ordered_pairs_then_hold_finish_and_restart(first, second):
    r = rig()
    first_id = complete(r, first)
    assert submit(r, 'HOLD', phase='FINISH', request='finish-first')
    assert r.core.terminal and r.core.status == 'COMPLETED'
    second_id = complete(r, second)
    assert first_id != second_id
    assert submit(r, 'HOLD', phase='FINISH', request='END')
    assert r.core.terminal and r.core.status == 'COMPLETED'
    third_id = complete(r, first)
    assert third_id not in (first_id, second_id)
    admitted = [e['behavior'] for e in r.core.events if e['reason'] == 'skill_admitted']
    assert admitted == [first, 'HOLD', second, 'HOLD', first]


@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_turn_needs_target_and_late_motion_cannot_rescue_timeout(behavior, sign):
    r = rig(); assert submit(r, behavior)
    deadline = r.core.deadline
    assert submit(r, behavior, phase='CONTINUE', request=r.core.command.request_id)
    assert r.core.deadline == deadline
    # Some correctly directed motion is insufficient to meet this new contract.
    r.a += sign * .1
    while r.t < deadline:
        step(r)
    assert r.core.stop_at is not None and r.commands[-1] == (0., 0., 0.)
    r.a += sign * (r.core.turn_target_rad + .1)  # Late target cannot rescue timeout.
    for _ in range(70):
        step(r)
    assert r.core.status == 'FAILED' and r.core.reason == 'measured_action_target_not_reached'
    assert not r.core.turn_target_reached
    assert not submit(r, 'ADVANCE')


@pytest.mark.parametrize('behavior', ['ADVANCE', 'TURN_LEFT', 'TURN_RIGHT'])
def test_end_before_motion_is_cancellation_and_next_request_can_start(behavior):
    r = rig(); assert submit(r, behavior)
    old_request = r.core.command.request_id
    assert submit(r, 'HOLD', phase='FINISH', request='END')
    for _ in range(70):
        step(r)
    assert r.core.status == 'CANCELLED' and r.core.terminal
    assert not r.core.turn_target_reached and not r.core.moved
    assert not submit(r, behavior, phase='CONTINUE', request=old_request)
    complete(r, behavior)


def test_stale_feedback_stops_and_latches_without_retry():
    r = rig(); assert submit(r, 'TURN_LEFT')
    r.t += .3; r.core.tick()
    assert r.commands[-1] == (0., 0., 0.) and 'odom_unavailable' in r.core.motion.failures
    for _ in range(70):
        step(r)
    assert r.core.status == 'FAILED'
    assert not submit(r, 'TURN_LEFT')
    assert len([e for e in r.core.events if e['reason'] == 'skill_admitted']) == 1
