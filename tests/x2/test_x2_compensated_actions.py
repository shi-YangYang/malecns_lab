"""v11 same-request residual compensation; synthetic sensors only, no ROS/brain."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_interaction import (
    integral_action_interaction_profile, compensated_action_interaction_profile, validate_interaction_profile,
)
from malecns_lab.control.x2_skill_profile import (
    SIM_COMPENSATED_ACTION_PROFILE_ID, integral_action_skill_profile,
    compensated_action_skill_profile, validate_candidate_skill_profile,
)
from malecns_lab.experiments import x2_continuity as driver
from malecns_lab.learning import x2_frozen_history_policy as frozen
from malecns_lab.learning.x2_behavior_session import NeuralSession
from malecns_lab.runtime.x2 import behavior as runtime
from .test_x2_action_continuity import rig, step, submit


ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = [('TURN_LEFT', 1), ('TURN_RIGHT', -1)]


def compensated_rig(behavior='TURN_LEFT'):
    r = rig(compensated_action_skill_profile, compensated_action_interaction_profile)
    assert submit(r, behavior)
    step(r)
    return r


def settle(r):
    for _ in range(80):
        step(r)
        if r.core.terminal or r.core.stop_at is None:
            break


def first_stop(r, degrees):
    sign = 1 if r.core.command.behavior == 'TURN_LEFT' else -1
    r.a = r.core.start_yaw + sign * math.pi/2
    step(r)
    assert r.core.stop_at is not None
    r.a = r.core.start_yaw + sign * math.radians(degrees)
    settle(r)


def test_explicit_v11_preserves_v10_contract_and_neural_inputs_and_poll_budget(tmp_path, monkeypatch, capsys):
    old, new = integral_action_skill_profile(), compensated_action_skill_profile()
    assert new['profile_id'] == SIM_COMPENSATED_ACTION_PROFILE_ID
    assert validate_candidate_skill_profile(new) == new
    assert 'reference_compensation' not in old['turn_execution']
    for key in ('feedback_controller','left_rad_s','right_rad_s','command_seconds','measured_stop_angle_rad','acceptance'):
        assert new['turn_execution'][key] == old['turn_execution'][key]
    for key in ('wave_execution','advance_execution','execution_poll_seconds_by_behavior'):
        assert new[key] == old[key]
    interaction = compensated_action_interaction_profile()
    assert validate_interaction_profile(interaction) == interaction
    assert {k:v for k,v in interaction.items() if k not in ('profile_id','skill_profile_id')} == {
        k:v for k,v in integral_action_interaction_profile().items() if k not in ('profile_id','skill_profile_id')}
    for key in ('max_first_stopped_error_rad','measured_angle_limit_rad'):
        bad = deepcopy(new); bad['turn_execution']['reference_compensation'][key] += .01
        with pytest.raises(ValueError):
            validate_candidate_skill_profile(bad)
    policy = SimpleNamespace(policy_id='p', encoder=SimpleNamespace(input_map_id='i'),
        identity={'interaction_profile':interaction}, initial_state=lambda *a, **k:None,
        state_identity=lambda _: {'state_id':'s'}, set_runtime_context=lambda *a:None)
    session = NeuralSession(policy,'c','r','s',None,lambda *a,**k:None,continuous=True)
    assert session.execution_poll_seconds_by_behavior == dict(TURN_LEFT=36.,TURN_RIGHT=36.)
    monkeypatch.setattr(frozen,'_check',lambda *args:None)
    before = frozen.inspect_frozen_history_candidate(ROOT,ROOT/frozen.CANDIDATE_RELATIVE,
        turn_profile='integral',wave_profile='balance-sway')
    after = frozen.inspect_frozen_history_candidate(ROOT,ROOT/frozen.CANDIDATE_RELATIVE,
        turn_profile='compensated',wave_profile='balance-sway')
    assert before['checkpoint_id'] != after['checkpoint_id'] and not after['policy']['can_deploy']
    for key in ('numeric_engine_id','input_map_id','artifacts','readout','initialization','source_sha256'):
        assert before['policy'][key] == after['policy'][key]
    seen = []
    def inspect(root,path,*,turn_profile,wave_profile):
        seen.append((turn_profile,wave_profile)); return after
    monkeypatch.setattr(frozen,'inspect_frozen_history_candidate',inspect)
    bad = deepcopy(after); bad['checkpoint_id'] = 'forged'
    with pytest.raises(ValueError,match='identity differs'):
        frozen.load_frozen_history_policy(ROOT,ROOT/frozen.CANDIDATE_RELATIVE,bad)
    assert seen == [('compensated','balance-sway')]
    path = tmp_path/frozen.CANDIDATE_RELATIVE; path.mkdir(parents=True)
    assert runtime.candidate(tmp_path,path,turn_profile='compensated',wave_profile='balance-sway')[1] == after
    monkeypatch.setattr(driver.subprocess,'Popen',lambda *a,**k:pytest.fail('Started a process'))
    for main in (runtime.main,driver.main):
        assert main([]) == 0
        assert json.loads(capsys.readouterr().out)['turn_profile'] == 'measured'
        assert main(['--turn-profile','compensated','--wave-profile','balance-sway']) == 0
        assert json.loads(capsys.readouterr().out)['turn_profile'] == 'compensated'
        with pytest.raises(ValueError,match='requires balance-sway'):
            main(['--turn-profile','compensated'])
    base = driver.runtime_command('/python',tmp_path/'run','continuous')
    assert driver.runtime_command('/python',tmp_path/'run','continuous','compensated','balance-sway') == (
        base + ['--turn-profile','compensated','--wave-profile','balance-sway'])


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('first_degrees', [80.,100.])
def test_reference_formula_only_changes_tracking_and_keeps_origin_identity_budgets(behavior,sign,first_degrees):
    r = compensated_rig(behavior)
    identity = (r.core.start_yaw,r.core.execution_id,r.core.command.request_id,r.core.started_at)
    deadlines = (r.core.deadline,r.core.precision_hard_deadline)
    first_stop(r,first_degrees)
    assert not r.core.terminal and r.core.precision_corrections == 1
    expected = sign*math.radians(180.-first_degrees)
    assert r.core.compensated_reference == pytest.approx(expected)
    assert r.core.compensated_residual == pytest.approx(sign*math.radians(90.-first_degrees))
    assert r.core.integral_value == 0. and r.core.stop_at is None
    assert (r.core.start_yaw,r.core.execution_id,r.core.command.request_id,r.core.started_at) == identity
    assert (r.core.deadline,r.core.precision_hard_deadline) == deadlines
    remaining = min(10.-r.core.integral_nonzero_seconds,r.core.deadline-r.t)
    assert r.core.motion.intent_deadline == pytest.approx(r.t+remaining)
    assert submit(r,behavior,phase='CONTINUE',request=r.core.command.request_id)
    assert (r.core.deadline,r.core.precision_hard_deadline) == deadlines
    evidence = r.core._execution_details()['precision_turn']['reference_compensation']
    assert evidence['original_signed_target_rad'] == sign*math.pi/2
    assert evidence['trigger']['request_id'] == identity[2] and evidence['trigger']['execution_id'] == identity[1]
    assert evidence['trigger']['odom_sequence'] == r.core.motion.odom.sequence
    assert evidence['first_stopped_angle_rad'] == pytest.approx(sign*math.radians(first_degrees))
    r.a = r.core.start_yaw+sign*math.pi/2; step(r)
    assert r.core._precision_error() == pytest.approx(0.)
    assert r.core._integral_tracking_error() == pytest.approx(sign*math.radians(90.-first_degrees))
    assert r.core.stop_at is None  # Original target crossing cannot terminate reference tracking.
    r.a = r.core.start_yaw+expected; step(r)
    assert r.core.stop_at is not None
    r.a = r.core.start_yaw+sign*math.pi/2; settle(r)
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    assert submit(r,behavior)
    assert r.core.compensated_reference is r.core.compensated_residual is r.core.compensated_trigger is None
    assert r.core.precision_corrections == 0 and r.core.integral_value == 0.


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
def test_compensated_reference_is_not_final_scoring_target(behavior,sign):
    r = compensated_rig(behavior); first_stop(r,80.)
    r.a = r.core.start_yaw+sign*math.radians(100.); step(r)
    assert abs(r.core._integral_tracking_error()) < 1e-12
    # A genuine small post-zero outward recoil makes "no improvement" strict;
    # do not rely on binary equality of the independently accumulated +/-10°.
    r.a = r.core.start_yaw+sign*math.radians(100.1)
    settle(r)
    assert r.core.status == 'FAILED' and r.core.reason == 'integral_turn_no_improvement'
    assert r.core.events[-1]['angle_rad'] == pytest.approx(sign*math.radians(100.1))
    assert abs(r.core._precision_error()) > math.radians(5.)


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('first_degrees,corrects', [(75.,True),(74.9,False),(85.1,False)])
def test_first_stopped_error_limit_is_bounded_without_clipping(behavior,sign,first_degrees,corrects):
    r = compensated_rig(behavior); first_stop(r,first_degrees)
    if corrects:
        assert r.core.compensated_reference == pytest.approx(sign*math.radians(105.))
        assert r.core.precision_corrections == 1 and not r.core.terminal
    elif first_degrees < 75:
        assert r.core.status == 'FAILED' and r.core.reason == 'compensated_turn_first_stopped_error_limit'
        assert r.core.precision_corrections == 0 and r.core.compensated_reference is None
    else:
        assert r.core.status == 'COMPLETED' and r.core.compensated_reference is None


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('degrees', [105.,105.01])
def test_strict_measured_angle_guard_latches_without_callback_publishing(behavior,sign,degrees):
    r = compensated_rig(behavior)
    r.a = sign*math.radians(degrees)
    before = len(r.commands)
    r.feed(wz=sign*.3)
    assert len(r.commands) == before
    if degrees > 105:
        assert r.core.precision_failure == 'compensated_turn_measured_angle_limit'
        assert r.core.stop_at is not None and r.core.motion.target == (0.,0.,0.)
        r.core.tick(); assert r.commands[-1] == (0.,0.,0.)
        r.a = sign*math.pi/2; settle(r)
        assert r.core.status == 'FAILED' and r.core.precision_corrections == 0
    else:
        assert r.core.precision_failure is None and r.core.stop_at is None


def test_angle_guard_also_latches_while_stopping_and_v10_does_not_enable_it():
    r = compensated_rig(); r.a = math.pi/2; step(r)
    r.a = math.radians(105.01); r.feed()
    assert r.core.precision_failure == 'compensated_turn_measured_angle_limit'
    r.a = math.pi/2; settle(r)
    assert r.core.status == 'FAILED' and r.core.compensated_reference is None
    old = rig(integral_action_skill_profile,integral_action_interaction_profile)
    assert submit(old,'TURN_LEFT')
    old.a = math.radians(106.); step(old,wz=.3)
    assert old.core.precision_failure is None and old.core.stop_at is None
    first_stop(old,80.)
    assert old.core.precision_corrections == 1 and old.core.compensated_reference is None
    assert old.core._integral_tracking_error() == old.core._precision_error()


@pytest.mark.parametrize('interrupt', ['end','stale','ttl'])
def test_cancel_and_original_guards_take_priority_over_compensation_and_105_limit(interrupt):
    r = compensated_rig()
    if interrupt == 'end':
        assert submit(r,'HOLD',phase='FINISH',request='END')
    elif interrupt == 'stale':
        r.t += .3
    else:
        r.core.motion._expires = r.t
    r.a = math.radians(106.); r.feed(); r.core.tick()
    assert r.core.precision_failure != 'compensated_turn_measured_angle_limit'
    assert r.commands[-1] == (0.,0.,0.) and r.core.compensated_reference is None
    r.a = math.pi/2; settle(r)
    assert r.core.status == ('CANCELLED' if interrupt == 'end' else 'FAILED')
    assert r.core.precision_corrections == 0


def test_first_stop_needs_standing_and_fresh_samples_before_reference_change():
    r = compensated_rig(); r.a = math.pi/2; step(r)
    r.a = math.radians(80.)
    ready = r.core.standing_ready
    r.core.standing_ready = lambda:False
    for _ in range(70):
        step(r)
    assert r.core.motion.state == 'STOPPED' and r.core.compensated_reference is None
    r.core.standing_ready = ready
    r.t += .3; r.core.tick()
    assert r.core.compensated_reference is None
    settle(r)
    assert r.core.compensated_reference == pytest.approx(math.radians(100.))


def test_compensation_never_renews_budget_or_rescues_expired_stop():
    r = compensated_rig(); r.a = math.pi/2; step(r)
    r.a = math.radians(80.)
    deadline = r.core.deadline
    r.core.integral_nonzero_seconds = 10.
    settle(r)
    assert r.core.status == 'FAILED' and r.core.reason == 'integral_turn_motion_budget'
    assert r.core.deadline == deadline and r.core.compensated_reference is None
    late = compensated_rig(); late.a = math.pi/2; step(late)
    late.a = math.radians(80.)
    stop = late.core.stop_at
    late.t = stop+12.
    late.core.tick()
    assert late.core.status == 'UNKNOWN' and late.core.compensated_reference is None
