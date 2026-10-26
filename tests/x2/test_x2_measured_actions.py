"""Measured-action contract and desktop request ownership; fake feedback only."""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_interaction import (
    measured_action_interaction_profile, tapered_action_interaction_profile,
    relative_heading_goal, validate_interaction_profile,
)
from malecns_lab.control.x2_skill_profile import (
    SIM_MEASURED_ACTION_PROFILE_ID, measured_action_skill_profile,
    high_turn_full_skill_profile, validate_candidate_skill_profile,
    SIM_TAPERED_ACTION_PROFILE_ID, tapered_action_skill_profile,
)
from malecns_lab.experiments.x2_behavior_bridge import RemoteExchange
from malecns_lab.learning.x2_behavior_session import ClockExchange, validate_heading_target
from malecns_lab.runtime.x2.behavior import BehaviorRuntime, Commands
from .test_x2_action_continuity import rig, step, submit, complete
from .test_x2_behavior_desktop import Clock, heading_snapshot


def test_measured_contract_is_distinct_and_cannot_relax_guards():
    profile = measured_action_skill_profile()
    assert profile['profile_id'] == SIM_MEASURED_ACTION_PROFILE_ID
    assert validate_candidate_skill_profile(profile) == profile
    assert profile['command_seconds_by_behavior'] == dict(ADVANCE=10., TURN_LEFT=10., TURN_RIGHT=10.)
    assert profile['advance_execution']['measured_stop_distance_m'] == 1.
    assert profile['turn_execution']['measured_stop_angle_rad'] == math.pi / 2
    old = high_turn_full_skill_profile()
    assert old['command_seconds_by_behavior'] == dict(ADVANCE=1., TURN_LEFT=5., TURN_RIGHT=5.)
    assert old['turn_execution']['measured_stop_angle_rad'] == .35
    assert profile['wave_contract'] == old['wave_contract']
    for section, key, value in [('turn_execution', 'command_seconds', 11.),
            ('advance_execution', 'measured_stop_distance_m', .5)]:
        forged = deepcopy(profile); forged[section][key] = value
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(forged)


def test_tapered_profile_preserves_v4_and_changes_only_turn_speed_schedule():
    old, new = measured_action_skill_profile(), tapered_action_skill_profile()
    assert new['profile_id'] == SIM_TAPERED_ACTION_PROFILE_ID and new['profile_id'] != old['profile_id']
    assert validate_candidate_skill_profile(new) == new
    assert new['command_seconds_by_behavior'] == old['command_seconds_by_behavior']
    assert new['advance_execution'] == old['advance_execution'] and new['wave_contract'] == old['wave_contract']
    assert 'terminal_speed_rad_s' not in old['turn_execution']
    assert new['turn_execution']['measured_stop_angle_rad'] == old['turn_execution']['measured_stop_angle_rad']
    assert new['turn_execution']['terminal_speed_rad_s'] == .4
    assert new['turn_execution']['slowdown_remaining_angle_rad'] == .5
    interaction = tapered_action_interaction_profile()
    assert validate_interaction_profile(interaction) == interaction
    assert {k:v for k,v in interaction.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k:v for k,v in measured_action_interaction_profile().items() if k not in ('profile_id', 'skill_profile_id')}
    forged = deepcopy(new); forged['turn_execution']['terminal_speed_rad_s'] = .9
    with pytest.raises(ValueError):
        validate_candidate_skill_profile(forged)


@pytest.mark.parametrize('behavior,sign', [('TURN_LEFT', 1), ('TURN_RIGHT', -1)])
def test_terminal_turn_reduces_once_without_renewing_or_early_target(behavior, sign):
    r = rig(tapered_action_skill_profile, tapered_action_interaction_profile)
    assert submit(r, behavior)
    deadline, intent_deadline = r.core.deadline, r.core.motion.intent_deadline
    r.a = sign * (math.pi / 2 - .5001); step(r, wz=sign * .8)
    assert r.commands[-1] == (0., 0., sign * .8) and r.core.turn_slowdown_at is None
    r.a = sign * (math.pi / 2 - .4999); step(r, wz=sign * .8)
    assert r.commands[-1] == r.core.resolved_velocity == (0., 0., sign * .4)
    assert r.core.turn_slowdown_at == r.t and r.core.turn_slowdown_angle == pytest.approx(r.a)
    assert not r.core.turn_target_reached and r.core.stop_at is None
    assert submit(r, behavior, phase='CONTINUE', request=r.core.command.request_id)
    r.a = sign * (math.pi / 2 - .6); step(r)
    assert r.commands[-1] == (0., 0., sign * .4)  # No reacceleration or repeated ramp.
    assert r.core.deadline == deadline and r.core.motion.intent_deadline == intent_deadline
    r.a = sign * (math.pi / 2 + .002); step(r, wz=sign * .4)
    assert r.commands[-1] == (0., 0., 0.) and r.core.turn_target_reached
    assert not r.core.terminal
    for _ in range(70):
        step(r)
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    assert r.core.events[-1]['turn_slowdown_at'] is not None
    assert submit(r, behavior)
    assert r.core.turn_slowdown_at is None and r.core.resolved_velocity == (0., 0., sign * .8)


def test_terminal_turn_stall_and_stale_feedback_still_stop_with_original_guards():
    for stale in (False, True):
        r = rig(tapered_action_skill_profile, tapered_action_interaction_profile)
        assert submit(r, 'TURN_LEFT')
        r.a = math.pi / 2 - .4; step(r)
        assert r.commands[-1] == (0., 0., .4)
        if stale:
            r.t += .3; r.core.tick()
            assert r.commands[-1] == (0., 0., 0.) and 'odom_unavailable' in r.core.motion.failures
        else:
            while r.t < r.core.deadline:
                step(r)
            assert r.commands[-1] == (0., 0., 0.) and not r.core.turn_target_reached
        for _ in range(70):
            step(r)
        assert r.core.status == 'FAILED' and not submit(r, 'TURN_LEFT')


def test_advance_requires_one_measured_metre_then_stopped_feedback():
    r = rig(); assert submit(r, 'ADVANCE')
    r.x = .99; step(r, speed=.2)
    assert r.core.stop_at is None and r.commands[-1] == (.2, 0., 0.)
    r.x = 1.01; step(r, speed=.2)
    assert r.core.advance_target_reached and not r.core.terminal
    assert r.commands[-1] == (0., 0., 0.)
    for _ in range(70):
        step(r)
    assert r.core.status == 'COMPLETED' and r.core.forward_distance == pytest.approx(1.01)
    assert r.core.events[-1]['stop_confirmed']
    complete(r, 'TURN_RIGHT')


def test_advance_late_distance_cannot_rescue_timeout_or_extend_deadline():
    r = rig(); assert submit(r, 'ADVANCE')
    deadline = r.core.deadline
    assert submit(r, 'ADVANCE', phase='CONTINUE', request=r.core.command.request_id)
    assert r.core.deadline == deadline
    r.x = .8
    while r.t < deadline:
        step(r)
    assert r.core.stop_at is not None and r.commands[-1] == (0., 0., 0.)
    r.x = 1.2
    for _ in range(70):
        step(r)
    assert r.core.status == 'FAILED' and not r.core.advance_target_reached
    assert r.core.reason == 'measured_action_target_not_reached'


@pytest.mark.parametrize('task,sign', [('FACE_LEFT', 1), ('FACE_RIGHT', -1)])
def test_new_heading_goal_and_brain_observation_are_the_real_quarter_turn(task, sign):
    profile = measured_action_interaction_profile()
    host = ClockExchange('run', 'session', Clock(9000.), profile=profile)
    remote = RemoteExchange('run', 'session', Clock(12.), profile=profile)
    first, evidence, packet = heading_snapshot(host, remote, task, 'request', 3.)
    assert first.feedback.heading_error_rad == pytest.approx(sign * math.pi / 2)
    assert evidence['heading_target']['relative_offset_rad'] == sign * math.pi / 2
    assert relative_heading_goal(task, 3.)['relative_offset_rad'] == sign * .45
    remote.clock.value += .1
    measured_yaw = math.atan2(math.sin(3. + sign * math.pi / 2), math.cos(3. + sign * math.pi / 2))
    final, evidence, _ = heading_snapshot(host, remote, task, 'request', measured_yaw)
    assert final.feedback.heading_error_rad == pytest.approx(0., abs=1e-8)
    forged = deepcopy(packet['heading_target'])
    forged.update(relative_heading_goal(task, 3.))
    with pytest.raises(ValueError):
        validate_heading_target(forged, first.feedback,
            dict(run_id='run', session_id='session', task=task, request_id='request'),
            packet['remote_sequence'], packet['remote_issued_monotonic'], profile=profile)


def runtime(tmp_path):
    rows = []
    value = BehaviorRuntime(tmp_path, Path(__file__).resolve().parents[2], Commands(io.StringIO()), rows.append,
        tmp_path / 'unused-checkpoint', {'policy': {
            'interaction_profile': measured_action_interaction_profile(),
            'execution_profile': measured_action_skill_profile()}})
    value.tasks_open = True
    value.state = 'running'
    return value, rows


def test_busy_input_cannot_overwrite_and_stale_wait_cannot_unlock(tmp_path):
    value, rows = runtime(tmp_path)
    assert value.accept_task('SHOW_FORWARD', 1.)
    original = json.loads((tmp_path / 'task.json').read_text())
    identity = original['request_id']
    assert rows[-1]['phase'] == 'queued' and rows[-1]['task'] == 'SHOW_FORWARD'
    assert rows[-1]['active_request_id'] == identity and rows[-1]['request_pending'] and rows[-1]['action_busy']
    assert not value.accept_task('FACE_LEFT', 1.1)
    assert rows[-1]['phase'] == 'queued' and rows[-1]['notice'] == 'request_rejected'
    assert rows[-1]['task'] == 'SHOW_FORWARD' and rows[-1]['rejected_task'] == 'FACE_LEFT'
    assert rows[-1]['reason'] == 'action_busy' and not value.commands.stop.is_set()
    assert json.loads((tmp_path / 'task.json').read_text()) == original
    for stale_id in (None, 'old'):
        assert not value.phase_status(dict(phase='waiting_task', request_id=stale_id, stop_confirmed=True))
    assert value.request_pending and value.action_busy
    assert value.phase_status(dict(phase='computing', request_id=identity))
    assert not value.request_pending and value.action_busy
    assert value.phase_status(dict(phase='waiting_task', request_id=identity, stop_confirmed=False))
    assert value.action_busy
    assert value.phase_status(dict(phase='waiting_task', request_id=identity, stop_confirmed=True))
    assert not value.action_busy and value.active_request_id is None
    assert value.accept_task('FACE_LEFT', 2.)
    second = value.active_request_id
    assert not value.phase_status(dict(phase='waiting_task', request_id=identity, stop_confirmed=True))
    assert value.action_busy and value.active_request_id == second


def test_end_interrupt_is_once_and_old_finish_cannot_unlock_it(tmp_path):
    value, rows = runtime(tmp_path)
    assert value.accept_task('GREET', 1.)
    first = value.active_request_id
    assert value.accept_task('END_INTERACTION', 1.1)
    ending = value.active_request_id
    assert ending != first and value.active_task == 'END_INTERACTION'
    assert not value.accept_task('END_INTERACTION', 1.2)
    assert rows[-1]['reason'] == 'end_pending' and rows[-1]['rejected_task'] == 'END_INTERACTION'
    assert not value.phase_status(dict(phase='waiting_task', request_id=first, stop_confirmed=True))
    assert value.action_busy and value.request_pending
    assert value.phase_status(dict(phase='waiting_task', request_id=ending, stop_confirmed=True))
    assert value.accept_task('GREET', 2.)


def test_not_ready_and_task_budget_reject_without_session_failure(tmp_path):
    value, rows = runtime(tmp_path)
    value.tasks_open = False
    assert not value.accept_task('GREET', 1.)
    assert rows[-1]['reason'] == 'tasks_not_ready' and not (tmp_path / 'task.json').exists()
    value.tasks_open = True; value.task_sequence = 50
    value.phase_status(dict(phase='waiting_task'))
    assert not value.accept_task('GREET', 1.)
    assert rows[-1]['reason'] == 'task_budget' and rows[-1]['notice'] == 'request_rejected'
    assert rows[-1]['phase'] == 'waiting_task' and not rows[-1]['action_busy']
    assert value.accept_task('END_INTERACTION', 2.)
    assert not value.commands.stop.is_set()


@pytest.mark.parametrize('stop_claim', [False, True])
def test_request_failure_stops_session_while_worker_is_still_alive(tmp_path, stop_claim):
    value, rows = runtime(tmp_path)
    value.accept_task('SHOW_FORWARD', 1.)
    identity = value.active_request_id
    (tmp_path/'behavior_phase.json').write_text(json.dumps(dict(phase='request_failed',
        request_id=identity, reason='decision_budget_reached', task_success=False, stop_confirmed=stop_claim)))
    value.host = value.bridge = SimpleNamespace(poll=lambda: None)
    for name in ('preflight', 'prepare', 'start_simulator', 'start_behavior_bridge', 'guard'):
        setattr(value, name, lambda: None)
    value.command = lambda *args, **kwargs: SimpleNamespace(stdout='')
    stopped = []
    def shutdown(reason):
        stopped.append(reason)
        value.stop_confirmed = value.resources_stopped = True
    value.shutdown_behavior = shutdown
    outcome = value.execute_behavior()
    assert outcome['state'] == 'failed' and 'decision_budget_reached' in outcome['reason']
    assert len(stopped) == 1 and not value.tasks_open
    assert rows[-1]['state'] == 'failed' and rows[-1]['resources_stopped']
    assert not any(row.get('phase') == 'waiting_task' for row in rows)


def test_queue_overflow_keeps_end_and_emergency_available():
    tasks = [dict(command='task', task='GREET')] * 51 + [dict(command='task', task='END_INTERACTION'), dict(command='stop')]
    commands = Commands(io.StringIO(''.join(json.dumps(row) + '\n' for row in tasks)))
    commands.read()
    assert commands.tasks.qsize() == 50 and commands.task_overflow_count == 1
    assert commands.priority_end.get_nowait()[0] == 'END_INTERACTION'
    assert commands.reason == 'user_emergency_stop'
