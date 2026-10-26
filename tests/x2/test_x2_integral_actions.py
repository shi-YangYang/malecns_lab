"""Single opt-in PID candidate; fake sensors/clock only, no ROS or neural IO."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_interaction import (
    integral_action_interaction_profile, balance_sway_action_interaction_profile,
    validate_interaction_profile,
)
from malecns_lab.control.x2_skill_profile import (
    SIM_INTEGRAL_ACTION_PROFILE_ID, integral_action_skill_profile,
    measured_action_skill_profile, precision_action_skill_profile,
    balance_sway_action_skill_profile, quarter_turn_action_skill_profile,
    validate_candidate_skill_profile,
)
from malecns_lab.learning import x2_frozen_history_policy as frozen
from malecns_lab.learning.x2_behavior_session import NeuralSession
from malecns_lab.runtime.x2 import behavior as runtime
from malecns_lab.experiments import x2_continuity as driver
from .test_x2_action_continuity import rig, step, submit


ZERO = (0., 0., 0.)
ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = [('TURN_LEFT', 1), ('TURN_RIGHT', -1)]


def integral_rig(behavior='TURN_LEFT'):
    r = rig(integral_action_skill_profile, integral_action_interaction_profile)
    assert submit(r, behavior)
    return r


def settle(r):
    for _ in range(80):
        step(r)
        if r.core.terminal or r.core.stop_at is None:
            break


def park_then_recoil(r, final_degrees):
    sign = 1 if r.core.command.behavior == 'TURN_LEFT' else -1
    r.a = r.core.start_yaw + sign * math.pi / 2
    step(r)
    assert r.core.stop_at is not None and r.commands[-1] == ZERO
    r.a = r.core.start_yaw + sign * math.radians(final_degrees)
    settle(r)


def test_integral_contract_is_isolated_and_authenticates_exact_configuration():
    new, old = integral_action_skill_profile(), balance_sway_action_skill_profile()
    assert new['profile_id'] == SIM_INTEGRAL_ACTION_PROFILE_ID
    assert validate_candidate_skill_profile(new) == new
    assert new['execution_poll_seconds_by_behavior'] == dict(TURN_LEFT=36., TURN_RIGHT=36.)
    for key in ('advance_execution', 'wave_execution', 'wave_contract', 'command_seconds_by_behavior'):
        assert new[key] == old[key]
    p = new['turn_execution']['feedback_controller']
    assert (p['position_gain'], p['angular_rate_gain'], p['integral_gain']) == (2., .25, 1.)
    assert (p['nonzero_budget_seconds'], p['stop_timeout_seconds'], p['total_deadline_seconds']) == (10., 12., 34.)
    assert 'feedback_controller' not in measured_action_skill_profile()['turn_execution']
    assert 'feedback_controller' not in quarter_turn_action_skill_profile()['turn_execution']
    assert 'integral_gain' not in precision_action_skill_profile()['turn_execution']['feedback_controller']
    before, after = balance_sway_action_interaction_profile(), integral_action_interaction_profile()
    assert validate_interaction_profile(after) == after
    assert {k:v for k,v in before.items() if k not in ('profile_id', 'skill_profile_id')} == {
        k:v for k,v in after.items() if k not in ('profile_id', 'skill_profile_id')}
    policy = SimpleNamespace(policy_id='p', encoder=SimpleNamespace(input_map_id='i'),
        identity={'interaction_profile': after}, initial_state=lambda *a, **k:None,
        state_identity=lambda _: {'state_id':'s'}, set_runtime_context=lambda *a:None)
    session = NeuralSession(policy, 'c', 'r', 's', None, lambda *a, **k:None, continuous=True)
    assert session.execution_poll_seconds_by_behavior == dict(TURN_LEFT=36., TURN_RIGHT=36.)
    assert session.execution_poll_seconds == 32.
    for key, value in [('integral_gain', 2.), ('nonzero_budget_seconds', 11.), ('max_corrections', 2)]:
        bad = deepcopy(new); bad['turn_execution']['feedback_controller'][key] = value
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(bad)


def test_metadata_loader_and_cli_preserve_neural_identity_and_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(frozen, '_check', lambda *args:None)
    old = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, wave_profile='balance-sway')
    new = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE,
        turn_profile='integral', wave_profile='balance-sway')
    assert new['policy']['execution_profile'] == integral_action_skill_profile()
    assert new['checkpoint_id'] != old['checkpoint_id'] and not new['policy']['can_deploy']
    for key in ('numeric_engine_id', 'input_map_id', 'artifacts', 'readout', 'initialization', 'source_sha256'):
        assert new['policy'][key] == old['policy'][key]
    seen = []
    def inspect(root, path, *, turn_profile, wave_profile):
        seen.append((turn_profile, wave_profile)); return new
    monkeypatch.setattr(frozen, 'inspect_frozen_history_candidate', inspect)
    bad = deepcopy(new); bad['checkpoint_id'] = 'forged'
    with pytest.raises(ValueError, match='identity differs'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, bad)
    assert seen == [('integral', 'balance-sway')]
    bad['policy']['execution_profile']['turn_execution']['feedback_controller']['integral_gain'] = 2.
    with pytest.raises(ValueError, match='not an allowed contract'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, bad)
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *a, **k:pytest.fail('Started a process'))
    path = tmp_path/frozen.CANDIDATE_RELATIVE; path.mkdir(parents=True)
    assert runtime.candidate(tmp_path, path, turn_profile='integral', wave_profile='balance-sway')[1] == new
    legacy = tmp_path/'data/legacy'; legacy.mkdir()
    with pytest.raises(ValueError, match='requires the fixed B'):
        runtime.candidate(tmp_path, legacy, turn_profile='integral', wave_profile='balance-sway')
    for main in (runtime.main, driver.main):
        assert main([]) == 0
        default = json.loads(capsys.readouterr().out)
        assert (default['turn_profile'], default['wave_profile']) == ('measured', 'baseline')
        assert main(['--turn-profile', 'integral', '--wave-profile', 'balance-sway']) == 0
        selected = json.loads(capsys.readouterr().out)
        assert (selected['turn_profile'], selected['wave_profile']) == ('integral', 'balance-sway')
        with pytest.raises(ValueError, match='requires balance-sway'):
            main(['--turn-profile', 'integral'])
    base = driver.runtime_command('/python', tmp_path/'run', 'continuous')
    assert driver.runtime_command('/python', tmp_path/'run', 'continuous', 'integral', 'balance-sway') == (
        base + ['--turn-profile', 'integral', '--wave-profile', 'balance-sway'])


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
def test_unsaturated_stall_accumulates_i_past_old_two_second_deadline(behavior, sign):
    r = integral_rig(behavior)
    r.a = sign * (math.pi / 2 - .18)
    step(r)
    first = r.commands[-1][2]
    for _ in range(105):
        step(r)
    assert r.core.stop_at is None and not r.core.motion.failures
    assert abs(first) < abs(r.commands[-1][2]) < .8
    assert r.core.integral_value == pytest.approx(sign * .18 * .02 * 106)
    while r.core.stop_at is None and r.t < r.core.started_at + 8:
        step(r)
    assert r.core.precision_failure == 'integral_turn_saturated_stall'
    assert r.t - r.core.started_at > 4.
    assert r.core.integral_value == 0. and r.commands[-1] == ZERO
    r.a = sign * math.pi / 2; settle(r)
    assert r.core.status == 'FAILED'  # Late goal feedback never rescues a latched stall.


def test_only_fresh_new_samples_integrate_and_gap_and_sign_change_are_explicit():
    r = integral_rig(); r.a = math.pi / 2 - .18; step(r)
    initial = r.core.integral_value
    assert initial == pytest.approx(.18 * .02)
    count = len(r.core.integral_samples)
    r.core.tick(); r.core.tick()
    assert r.core.integral_value == initial and len(r.core.integral_samples) == count
    r.t += .04; step(r)  # Actual odom interval .06, within freshness but over integral dt cap.
    assert r.core.integral_value == initial
    assert r.core.integral_terms['integral_rule'] == 'sample_gap_skipped'
    step(r); assert r.core.integral_value == pytest.approx(initial * 2)
    r.a = math.pi / 2 + .18; step(r)
    assert r.core.integral_value == 0.
    assert r.core.integral_terms['integral_rule'] == 'error_sign_reset'


@pytest.mark.parametrize('sign', [1, -1])
def test_full_pid_saturation_holds_same_direction_and_unwinds_opposite(sign):
    r = integral_rig('TURN_LEFT' if sign > 0 else 'TURN_RIGHT')
    r.a = sign * (math.pi / 2 - .2)
    r.core.integral_value = sign * .5
    step(r)
    assert r.commands[-1][2] == sign * .8
    assert r.core.integral_value == sign * .5
    assert r.core.integral_terms['integral_rule'] == 'same_direction_saturation_hold'
    r.core.integral_value = -sign * .8
    step(r, wz=sign * 2.)  # D produces opposite saturation; I may unload toward the error.
    assert r.core.integral_value == pytest.approx(sign * (-.8 + .2 * .02))
    assert r.commands[-1][2] == -sign * .8
    r.core.integral_value = sign * .799
    step(r, wz=sign * 2.)  # Full PID is unsaturated despite high I; clamp I itself.
    assert r.core.integral_value == sign * .8
    assert abs(r.commands[-1][2]) <= .8


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('degrees', [88., 92.])
def test_final_tolerance_uses_real_stop_and_new_start_resets_i(behavior, sign, degrees):
    r = integral_rig(behavior); step(r)
    r.a = sign * (math.pi / 2 + .01); step(r, wz=sign * .4)
    assert r.core.turn_target_reached and r.core.stop_at is None and not r.core.terminal
    r.a = sign * math.radians(degrees); step(r)
    assert r.core.stop_at is not None and not r.core.terminal and r.core.integral_value == 0.
    settle(r)
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    assert r.core.events[-1]['angle_rad'] == pytest.approx(sign * math.radians(degrees))
    assert submit(r, behavior)
    assert r.core.integral_value == r.core.integral_nonzero_seconds == 0.
    assert not r.core.integral_samples and r.core.precision_corrections == 0
    assert r.core.signed_angle == 0.


@pytest.mark.parametrize('final_degrees,reason', [(91., None), (96., 'integral_turn_corrections_exhausted'),
    (99., 'integral_turn_no_improvement'), (100., 'integral_turn_no_improvement')])
def test_one_correction_preserves_original_request_goal_and_requires_improvement(final_degrees, reason):
    r = integral_rig('TURN_RIGHT'); step(r)
    origin, execution, request = r.core.start_yaw, r.core.execution_id, r.core.command.request_id
    deadlines = (r.core.deadline, r.core.precision_hard_deadline)
    park_then_recoil(r, 99.)
    assert r.core.precision_corrections == 1 and r.core.stop_at is None
    assert r.core.start_yaw == origin and r.core.execution_id == execution and r.core.command.request_id == request
    assert (r.core.deadline, r.core.precision_hard_deadline) == deadlines
    remaining = min(10. - r.core.integral_nonzero_seconds, r.core.deadline - r.t)
    assert r.core.motion.intent_deadline == pytest.approx(r.t + remaining)
    assert len(r.core.events) == 2  # ACCEPTED and actual STARTED, no manufactured brain/execution event.
    assert submit(r, 'TURN_RIGHT', phase='CONTINUE', request=request)
    assert (r.core.deadline, r.core.precision_hard_deadline) == deadlines
    step(r); assert r.commands[-1][2] > 0.  # Internal correction may oppose original brain action.
    park_then_recoil(r, final_degrees)
    assert r.core.terminal and r.core.precision_corrections == 1
    assert r.core.status == ('COMPLETED' if reason is None else 'FAILED')
    if reason:
        assert r.core.reason == reason
    assert r.core.events[-1]['stop_confirmed'] and r.commands[-1] == ZERO
    stops = [x for x in r.core.precision_trace if x['phase'] == 'stopped']
    assert stops[0]['angle_rad'] == pytest.approx(-math.radians(99.))
    assert len(r.core.events) == 3


@pytest.mark.parametrize('interrupt', ['end', 'emergency', 'stale_odom', 'stale_mc', 'ttl'])
def test_interruption_clears_i_and_never_enters_correction(interrupt):
    r = integral_rig(); r.a = math.pi / 2 - .18; step(r)
    assert r.core.integral_value > 0
    if interrupt == 'end':
        assert submit(r, 'HOLD', phase='FINISH', request='END')
    elif interrupt == 'emergency':
        r.core.cancel()
    elif interrupt == 'stale_odom':
        r.t += .3
    elif interrupt == 'stale_mc':
        for _ in range(26):
            r.t += .02
            r.core.update_odom((0.,0.,.65), (0.,0.,math.sin(r.a/2),math.cos(r.a/2)), ZERO, ZERO, received=r.t)
    else:
        r.core.motion._expires = r.t  # Model the adapter's independent intent deadline.
    r.core.tick()
    assert r.core.integral_value == 0. and r.commands[-1] == ZERO
    if interrupt == 'stale_mc':
        assert 'mc_unavailable' in r.core.motion.failures
    r.a = math.pi / 2; settle(r)
    assert r.core.status == ('CANCELLED' if interrupt == 'end' else 'UNKNOWN'
        if interrupt in ('emergency', 'stale_mc') else 'FAILED')
    assert r.core.precision_corrections == 0


def test_actual_nonzero_budget_excludes_parking_and_continue_does_not_renew():
    r = integral_rig(); step(r)
    for _ in range(20):
        r.a += .01; step(r, wz=.5)
    r.a = math.pi / 2; step(r)
    spent = r.core.integral_nonzero_seconds
    before = len(r.commands)
    for _ in range(60):
        step(r, wz=.1)  # Genuine angular speed prevents stopped confirmation.
    assert r.core.integral_nonzero_seconds == spent
    assert all(v == ZERO for v in r.commands[before:])
    r.a = math.radians(99.); settle(r)
    assert r.core.precision_corrections == 1
    deadlines = r.core.deadline, r.core.integral_segment_deadline, r.core.motion.intent_deadline
    assert submit(r, 'TURN_LEFT', phase='CONTINUE', request=r.core.command.request_id)
    assert (r.core.deadline, r.core.integral_segment_deadline, r.core.motion.intent_deadline) == deadlines
    # Exercise the cumulative published-command boundary independently of wall parking time.
    r.core.integral_nonzero_seconds = 10.
    r.core.tick()
    assert r.commands[-1] == ZERO and r.core.precision_failure == 'integral_turn_motion_budget'
    r.a = math.pi / 2; settle(r)
    assert r.core.status == 'FAILED'


@pytest.mark.parametrize('deadline_kind', ['motion', 'stop', 'hard'])
def test_absolute_deadlines_reject_late_feedback_and_never_restart(deadline_kind):
    r = integral_rig(); step(r)
    assert r.core.deadline == r.core.started_at + 22.
    assert r.core.precision_hard_deadline == r.core.started_at + 34.
    if deadline_kind == 'motion':
        park_then_recoil(r, 99.)
        assert r.core.precision_corrections == 1
        # Reach the fixed stop-only cutoff with fresh sensors, not a stale-input shortcut.
        r.t = r.core.deadline - .02
        r.feed(); r.core.tick()
        assert r.commands[-1] == ZERO and r.core.stop_at is not None
        r.a = math.pi / 2; settle(r)
        assert r.core.status == 'FAILED'
    else:
        r.a = math.pi / 2; step(r)
        r.a = math.radians(99.)
        cutoff = r.core.stop_at + 12. if deadline_kind == 'stop' else r.core.precision_hard_deadline
        r.t = cutoff - .02
        r.feed()
        # The first usable STOPPED+standing evidence arrives exactly at the deadline.
        r.core.motion.state = 'STOPPED'
        r.core.standing_ready = lambda:True
        r.core._integral_finish(r.t)
        assert r.core.status == 'UNKNOWN' and not r.core.events[-1]['stop_confirmed']
        assert r.core.precision_corrections == 0 and r.core.terminal


def test_integral_evidence_buffers_are_bounded_and_count_discarded_records():
    r = integral_rig(); r.a = math.pi / 2 - .18
    # Exercise only arithmetic/bookkeeping at 100Hz; no sink/model/ROS calls in this loop.
    for _ in range(520):
        r.t -= .01; r.feed()
        r.core._integral_velocity(r.t)
    assert len(r.core.integral_samples) == 512 and r.core.integral_sample_records_dropped == 8
    for _ in range(70):
        r.core._precision_stage('tracking', 'fixture')
    assert len(r.core.precision_trace) == 64 and r.core.integral_stage_records_dropped > 0
    details = r.core._execution_details()['precision_turn']['integral']
    assert details['sample_records_dropped'] == 8 and len(details['samples']) == 512
