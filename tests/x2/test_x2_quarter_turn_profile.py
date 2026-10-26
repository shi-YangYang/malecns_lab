"""Stopped quarter-turn contract fixtures; no brain, ROS or simulation runs."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_interaction import (
    quarter_turn_action_interaction_profile, balance_sway_action_interaction_profile,
    measured_action_interaction_profile, validate_interaction_profile)
from malecns_lab.control.x2_skill_profile import (
    SIM_QUARTER_TURN_ACTION_PROFILE_ID, FULL_ACTION_PROFILE_IDS, MEASURED_ACTION_PROFILE_IDS,
    quarter_turn_action_skill_profile, balance_sway_action_skill_profile,
    measured_action_skill_profile, validate_candidate_skill_profile)
from malecns_lab.control.x2_skills import RIGHT_ARM
from malecns_lab.experiments.x2_behavior_bridge import candidate_executor, _pump_callbacks
from malecns_lab.experiments import x2_continuity as driver
from malecns_lab.learning import x2_frozen_history_policy as frozen
from malecns_lab.runtime.x2 import behavior as runtime
from .test_x2_action_continuity import rig, step, submit

ROOT = Path(__file__).resolve().parents[2]
DIRECTIONS = [('TURN_LEFT', 1), ('TURN_RIGHT', -1)]


def quarter_rig():
    return rig(quarter_turn_action_skill_profile, quarter_turn_action_interaction_profile)


def cross_then_settle(r, behavior, sign, final_degrees):
    assert submit(r, behavior)
    deadline, request = r.core.deadline, r.core.command.request_id
    step(r)
    assert r.commands[-1] == (0., 0., sign * .8)
    assert submit(r, behavior, phase='CONTINUE', request=request)
    assert r.core.deadline == deadline == r.core.started_at + 10.
    r.a = sign * (math.pi / 2 + .001)
    step(r, wz=sign * .8)
    assert r.core.turn_target_reached and r.core.stop_at is not None
    assert not r.core.terminal and r.commands[-1] == (0., 0., 0.)
    r.a = sign * math.radians(final_degrees)
    for _ in range(80):
        step(r)
    assert r.core.motion.state == 'STOPPED' and r.core.terminal
    assert r.core.events[-1]['stop_confirmed'] and not r.core.motion.failures


def test_v9_changes_only_stopped_angle_contract_and_execution_identity():
    old, new = balance_sway_action_skill_profile(), quarter_turn_action_skill_profile()
    assert validate_candidate_skill_profile(new) == new
    assert new['profile_id'] == SIM_QUARTER_TURN_ACTION_PROFILE_ID
    assert new['profile_id'] in FULL_ACTION_PROFILE_IDS and new['profile_id'] in MEASURED_ACTION_PROFILE_IDS
    assert new['turn_execution'].pop('final_tolerance_rad') == math.radians(5.)
    for key in ('acceptance', 'evidence_scope'):
        new['turn_execution'][key] = old['turn_execution'][key]
    new['profile_id'] = old['profile_id']
    assert new == old  # Includes unchanged v8 wave .06/.03 and all motion/stop budgets.
    assert 'final_tolerance_rad' not in measured_action_skill_profile()['turn_execution']
    assert 'final_tolerance_rad' not in balance_sway_action_skill_profile()['turn_execution']
    new_interaction = quarter_turn_action_interaction_profile()
    assert validate_interaction_profile(new_interaction) == new_interaction
    old_interaction = balance_sway_action_interaction_profile()
    assert new_interaction['profile_id'] != old_interaction['profile_id']
    for key in ('profile_id', 'skill_profile_id'):
        new_interaction.pop(key); old_interaction.pop(key)
    assert new_interaction == old_interaction


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
def test_measured_crossing_then_88741_degree_stopped_return_completes(behavior, sign):
    r = quarter_rig()
    cross_then_settle(r, behavior, sign, math.degrees(1.548825043689792))
    assert r.core.status == 'COMPLETED' and r.core.moved
    assert r.core.signed_angle == pytest.approx(sign * 1.548825043689792)
    assert r.core.events[-1]['turn_final_tolerance_rad'] == math.radians(5.)
    assert [event['status'] for event in r.core.events] == ['ACCEPTED', 'STARTED', 'COMPLETED']


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('final_degrees', [84.9, 95.1])
def test_either_side_outside_five_degree_terminal_window_fails(behavior, sign, final_degrees):
    r = quarter_rig()
    cross_then_settle(r, behavior, sign, final_degrees)
    assert r.core.status == 'FAILED' and r.core.reason == 'measured_action_target_not_reached'


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
@pytest.mark.parametrize('late_degrees', [88.741, 90.1])
def test_no_target_crossing_before_deadline_cannot_be_rescued_by_late_feedback(behavior, sign, late_degrees):
    r = quarter_rig(); assert submit(r, behavior)
    r.a = sign * math.radians(88.741)
    while r.t < r.core.deadline:
        step(r)
    assert r.core.stop_at is not None and not r.core.turn_target_reached
    r.a = sign * math.radians(late_degrees)
    for _ in range(80):
        step(r)
    assert r.core.status == 'FAILED' and not r.core.turn_target_reached
    assert r.commands[-1] == (0., 0., 0.)


@pytest.mark.parametrize('profile,interaction', [
    (measured_action_skill_profile, measured_action_interaction_profile),
    (balance_sway_action_skill_profile, balance_sway_action_interaction_profile)])
def test_legacy_v4_v8_keep_original_one_sided_terminal_rule(profile, interaction):
    for final_degrees, expected in ((88.741, 'FAILED'), (96., 'COMPLETED')):
        r = rig(profile, interaction)
        cross_then_settle(r, 'TURN_LEFT', 1, final_degrees)
        assert r.core.status == expected and r.core.turn_final_tolerance_rad is None
        assert 'turn_final_tolerance_rad' not in r.core.events[-1]


@pytest.mark.parametrize('interrupt', ['end', 'stale'])
def test_cancellation_or_stale_feedback_cannot_be_reported_as_success(interrupt):
    r = quarter_rig(); assert submit(r, 'TURN_LEFT')
    step(r)
    if interrupt == 'end':
        r.a = math.pi / 2 + .001; step(r, wz=.8)
        assert submit(r, 'HOLD', phase='FINISH', request='end')
    else:
        # Expiry while motion is owned latches failure; STOPPING may instead
        # wait for new samples without weakening its measured stop window.
        r.t += .3; r.core.tick()
        assert 'odom_unavailable' in r.core.motion.failures
    r.a = math.radians(88.741)
    for _ in range(80):
        step(r)
    assert r.core.status == ('CANCELLED' if interrupt == 'end' else 'FAILED')


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
def test_cached_target_crossing_with_stale_feedback_cannot_complete(behavior, sign):
    r = quarter_rig(); assert submit(r, behavior)
    step(r)
    r.a = sign * (math.pi / 2 + .001)
    r.feed(wz=sign * .8)  # Receive the crossing, but delay the executor tick.
    assert r.core.motion.state == 'FORWARD' and not r.core.turn_target_reached
    r.t += .3
    assert r.t < r.core.deadline
    published_before_guard = len(r.commands)
    r.core.tick()
    assert 'odom_unavailable' in r.core.motion.failures
    assert not r.core.turn_target_reached and r.core.stop_at is not None
    r.a = sign * math.radians(88.741)
    for _ in range(80):
        step(r)
    assert r.core.terminal and r.core.status == 'FAILED'
    assert not any(event['status'] == 'COMPLETED' for event in r.core.events)
    assert r.commands[published_before_guard:]
    assert all(command == (0., 0., 0.) for command in r.commands[published_before_guard:])


@pytest.mark.parametrize('mismatch', ['tolerance', 'interaction', 'simulation'])
def test_quarter_turn_registration_requires_matching_fixed_contract(mismatch):
    reg = dict(run_id='run', session_id='session', simulation_only=True, qualification_only=True,
        wave_enabled=True, skill_profile=quarter_turn_action_skill_profile(),
        interaction_profile=quarter_turn_action_interaction_profile())
    if mismatch == 'tolerance':
        reg['skill_profile']['turn_execution']['final_tolerance_rad'] = math.radians(6.)
    elif mismatch == 'interaction':
        reg['interaction_profile'] = balance_sway_action_interaction_profile()
    else:
        reg['simulation_only'] = False
    with pytest.raises(ValueError):
        candidate_executor(reg, lambda _:pytest.fail('Published invalid contract'), preset=lambda _:None,
            wave_poses=dict(first=dict.fromkeys(RIGHT_ARM, 0.), terminal=dict.fromkeys(RIGHT_ARM, 0.)))


def test_metadata_and_loader_select_v9_without_changing_neural_inputs(monkeypatch):
    monkeypatch.setattr(frozen, '_check', lambda *args:None)
    old = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, wave_profile='balance-sway')
    new = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE,
        turn_profile='quarter-turn', wave_profile='balance-sway')
    assert new['policy']['execution_profile'] == quarter_turn_action_skill_profile()
    assert new['checkpoint_id'] != old['checkpoint_id'] and new['policy']['policy_id'] != old['policy']['policy_id']
    for key in ('numeric_engine_id', 'input_map_id', 'artifacts', 'readout', 'initialization', 'source_sha256'):
        assert new['policy'][key] == old['policy'][key]
    assert not new['policy']['can_deploy']
    seen = []
    def inspect(root, path, *, turn_profile, wave_profile):
        seen.append((turn_profile, wave_profile))
        return new
    monkeypatch.setattr(frozen, 'inspect_frozen_history_candidate', inspect)
    forged = deepcopy(new); forged['checkpoint_id'] = 'forged'
    with pytest.raises(ValueError, match='identity differs'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, forged)
    assert seen == [('quarter-turn', 'balance-sway')]
    forged['policy']['execution_profile']['turn_execution']['final_tolerance_rad'] = math.radians(6.)
    with pytest.raises(ValueError, match='not an allowed contract'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, forged)
    assert len(seen) == 1


def test_cli_and_candidate_forward_only_explicit_quarter_turn_balance_sway(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Preview started a process')
    monkeypatch.setattr(driver.subprocess, 'Popen', forbidden)
    def inspect(root, path, *, turn_profile, wave_profile):
        return dict(turn=turn_profile, wave=wave_profile)
    monkeypatch.setattr(frozen, 'inspect_frozen_history_candidate', inspect)
    path = tmp_path/frozen.CANDIDATE_RELATIVE; path.mkdir(parents=True)
    assert runtime.candidate(tmp_path, path)[1] == dict(turn='measured', wave='baseline')
    assert runtime.candidate(tmp_path, path, turn_profile='quarter-turn', wave_profile='balance-sway')[1] == dict(turn='quarter-turn', wave='balance-sway')
    legacy = tmp_path/'data/legacy'; legacy.mkdir()
    with pytest.raises(ValueError, match='requires the fixed B'):
        runtime.candidate(tmp_path, legacy, turn_profile='quarter-turn', wave_profile='balance-sway')
    for main in (runtime.main, driver.main):
        assert main([]) == 0
        default = json.loads(capsys.readouterr().out)
        assert (default['turn_profile'], default['wave_profile']) == ('measured', 'baseline')
        assert main(['--turn-profile', 'quarter-turn', '--wave-profile', 'balance-sway']) == 0
        selected = json.loads(capsys.readouterr().out)
        assert (selected['turn_profile'], selected['wave_profile']) == ('quarter-turn', 'balance-sway')
        with pytest.raises(ValueError, match='requires balance-sway'):
            main(['--turn-profile', 'quarter-turn'])
    old = driver.runtime_command('/python', tmp_path/'run', 'continuous')
    assert driver.runtime_command('/python', tmp_path/'run', 'continuous', 'quarter-turn', 'balance-sway') == old+['--turn-profile', 'quarter-turn', '--wave-profile', 'balance-sway']
    for turn, wave in [('quarter-turn', 'baseline'), ('precision', 'balance-sway'), ('settled', 'balance-sway')]:
        with pytest.raises(ValueError):
            driver.runtime_command('/python', tmp_path/'run', 'continuous', turn, wave)
        with pytest.raises(ValueError):
            runtime.candidate(tmp_path, path, turn_profile=turn, wave_profile=wave)


@pytest.mark.parametrize('behavior,sign', DIRECTIONS)
def test_callback_pump_returns_on_crossing_callback_then_original_tick_publishes_zero(behavior, sign):
    r = quarter_rig(); assert submit(r, behavior)
    step(r)
    assert r.commands[-1] == (0., 0., sign * .8)
    node = SimpleNamespace(calls=0)
    def spin_once(actual_node, *, timeout_sec):
        assert actual_node is node and timeout_sec == .0001
        node.calls += 1
        r.t += .001
        r.a = sign * (math.pi / 2 + .001) if node.calls >= 3 else sign * .3
        r.core.update_odom((r.x, r.y, .65), (0., 0., math.sin(r.a/2), math.cos(r.a/2)),
            (0., 0., 0.), (0., 0., sign * .8), received=r.t)
    before = len(r.commands)
    _pump_callbacks(spin_once, node, r.core, clock=lambda:r.t)
    assert node.calls == 3  # A ready fourth callback cannot delay the parking tick.
    assert len(r.commands) == before and r.core.stop_at is None and not r.core.turn_target_reached
    r.core.standing_ready(); r.core.tick()
    assert r.core.turn_target_reached and r.core.stop_at == r.t
    assert r.commands[before:] == [(0., 0., 0.)] and not r.core.motion.failures
    assert not r.core.terminal  # The original measured stop window is still required.


@pytest.mark.parametrize('case', ['v4', 'v8', 'no_command', 'hold', 'terminal', 'cancel', 'stopping', 'stopped', 'below_target'])
def test_callback_pump_nonapplicable_cases_keep_original_eight_ms_budget(case):
    core = SimpleNamespace(skill_profile_id=SIM_QUARTER_TURN_ACTION_PROFILE_ID,
        command=SimpleNamespace(behavior='TURN_LEFT'), terminal=False, cancel_requested=False,
        stop_at=None, motion=SimpleNamespace(state='FORWARD'), signed_angle=math.pi/2+.001,
        turn_target_rad=math.pi/2)
    if case in ('v4', 'v8'):
        core.skill_profile_id = (measured_action_skill_profile() if case == 'v4' else balance_sway_action_skill_profile())['profile_id']
    elif case == 'no_command': core.command = None
    elif case == 'hold': core.command.behavior = 'HOLD'
    elif case == 'terminal': core.terminal = True
    elif case == 'cancel': core.cancel_requested = True
    elif case == 'stopping': core.stop_at = 0.
    elif case == 'stopped': core.motion.state = 'STOPPED'
    else: core.signed_angle = math.pi/2-.001
    node = SimpleNamespace(calls=0)
    def spin_once(actual_node, *, timeout_sec):
        assert actual_node is node and timeout_sec == .0001
        node.calls += 1
    _pump_callbacks(spin_once, node, core, clock=lambda:node.calls*.001)
    assert node.calls == 8


def test_callback_pump_retains_sixty_four_callback_cap_even_if_clock_does_not_advance():
    node = SimpleNamespace(calls=0)
    core = SimpleNamespace(skill_profile_id=SIM_QUARTER_TURN_ACTION_PROFILE_ID, command=None)
    def spin_once(actual_node, *, timeout_sec):
        assert actual_node is node and timeout_sec == .0001
        node.calls += 1
    _pump_callbacks(spin_once, node, core, clock=lambda:0.)
    assert node.calls == 64


@pytest.mark.parametrize('expiry,reason', [('mc', 'mc_unavailable'), ('intent', 'intent_expired')])
def test_callback_pump_early_return_cannot_bypass_mc_or_intent_expiry(expiry, reason):
    r = quarter_rig(); assert submit(r, 'TURN_RIGHT')
    step(r)
    if expiry == 'mc':
        # Keep odometry continuous while the independent MC cache expires.
        for _ in range(5):
            r.t += .101
            r.core.update_odom((r.x, r.y, .65), (0., 0., 0., 1.),
                (0., 0., 0.), (0., 0., 0.), received=r.t)
    else:
        r.core.motion._expires = r.t - .001
    node = SimpleNamespace(calls=0)
    def spin_once(actual_node, *, timeout_sec):
        assert actual_node is node and timeout_sec == .0001
        node.calls += 1
        r.t += .001
        r.a = -math.pi/2-.001
        r.core.update_odom((r.x, r.y, .65), (0., 0., math.sin(r.a/2), math.cos(r.a/2)),
            (0., 0., 0.), (0., 0., -.8), received=r.t)
    _pump_callbacks(spin_once, node, r.core, clock=lambda:r.t)
    assert node.calls == 1 and not r.core.turn_target_reached
    r.core.standing_ready(); r.core.tick()
    assert reason in r.core.motion.failures and not r.core.turn_target_reached
    assert r.commands[-1] == (0., 0., 0.)
    r.a = -math.radians(88.741)
    for _ in range(80): step(r)
    assert r.core.status == ('UNKNOWN' if expiry == 'mc' else 'FAILED')
    assert not any(event['status'] == 'COMPLETED' for event in r.core.events)
