"""Opt-in wave envelope using fake sensors; no neural, ROS or VM execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from malecns_lab.control.x2_interaction import (
    measured_action_interaction_profile, balance_sway_action_interaction_profile)
from malecns_lab.control.x2_skill_profile import (
    measured_action_skill_profile, balance_sway_action_skill_profile, validate_candidate_skill_profile,
    SIM_BALANCE_SWAY_ACTION_PROFILE_ID, FULL_ACTION_PROFILE_IDS, MEASURED_ACTION_PROFILE_IDS)
from malecns_lab.control.x2_skills import RIGHT_ARM
from malecns_lab.experiments.x2_behavior_bridge import candidate_executor
from malecns_lab.experiments import x2_continuity as driver
from malecns_lab.learning import x2_frozen_history_policy as frozen
from malecns_lab.runtime.x2 import behavior as runtime
from .test_x2_skills import Rig

ROOT = Path(__file__).resolve().parents[2]


def registration(candidate):
    return dict(run_id='run', session_id='session', simulation_only=True, qualification_only=True,
        wave_enabled=True,
        skill_profile=balance_sway_action_skill_profile() if candidate else measured_action_skill_profile(),
        interaction_profile=balance_sway_action_interaction_profile() if candidate else measured_action_interaction_profile())


def wave_rig(candidate):
    r = Rig(qualified=True)
    r.core = candidate_executor(registration(candidate), r.commands.append, clock=lambda:r.t,
        preset=r.presets.append, wave_poses=dict(first=dict.fromkeys(RIGHT_ARM, 0.), terminal=dict.fromkeys(RIGHT_ARM, 0.)))
    r.core.motion.arm(); r.core.motion.confirm_source_registration()
    for _ in range(65):
        r.feed(); r.core.standing_ready(); r.core.tick()
    r.core.preset_query(0, 0, 1, received=r.t)
    assert r.submit('WAVE_RIGHT')
    r.core.preset_dispatch()
    assert r.core.preset_sent(r.t) and r.core.preset_ack(0, 0, 400)
    r.player, r.joint = 2, .3
    r.feed(); r.core.preset_query(0, 0, 400, received=r.t); r.core.tick()
    assert r.core.status == 'STARTED'
    return r


def test_wave_sway_profile_changes_only_explicit_wave_contract():
    old, new = measured_action_skill_profile(), balance_sway_action_skill_profile()
    assert validate_candidate_skill_profile(new) == new
    assert SIM_BALANCE_SWAY_ACTION_PROFILE_ID in FULL_ACTION_PROFILE_IDS
    assert SIM_BALANCE_SWAY_ACTION_PROFILE_ID in MEASURED_ACTION_PROFILE_IDS
    contract = new.pop('wave_execution')
    assert contract['active_yaw_limit_rad'] == .06 and contract['return_yaw_limit_rad'] == .03
    assert 'not an official safety limit' in contract['evidence_scope']
    new['profile_id'] = old['profile_id']
    assert new == old  # Includes unchanged helper/cancel facts, all deadlines and turn/advance contracts.
    old_interaction, new_interaction = measured_action_interaction_profile(), balance_sway_action_interaction_profile()
    for key in ('profile_id', 'skill_profile_id'):
        old_interaction.pop(key); new_interaction.pop(key)
    assert old_interaction == new_interaction


def test_wave_sway_active_tolerance_never_substitutes_for_final_stopped_return():
    baseline = wave_rig(False)
    baseline.a = .031
    baseline.feed(); baseline.core.preset_query(0, 0, 400, received=baseline.t); baseline.core.tick()
    assert baseline.core.status == 'UNKNOWN' and baseline.core.reason == 'wave_yaw_limit'
    r = wave_rig(True)
    r.a = .031
    r.feed(); r.core.preset_query(0, 0, 400, received=r.t); r.core.tick()
    assert r.core.status == 'STARTED' and not r.core.motion.failures
    r.player, r.joint = 0, 0.
    for _ in range(125):
        r.feed(); r.core.preset_query(0, 0, 1, received=r.t); r.core.tick()
    assert not r.core.terminal and not r.core.wave_return_confirmed
    assert not any(event.get('stop_confirmed') for event in r.core.events)
    r.a = .01
    for _ in range(125):
        r.feed(); r.core.preset_query(0, 0, 1, received=r.t); r.core.tick()
    assert r.core.status == 'COMPLETED' and r.core.terminal and r.core.wave_return_confirmed
    assert r.core.events[-1]['stop_confirmed'] is True and r.core.standing_ready()
    assert r.core.summary()['wave_active_yaw_limit_rad'] == .06
    assert r.core.summary()['wave_return_yaw_limit_rad'] == .03


@pytest.mark.parametrize('angle', [-.061, .061])
def test_wave_sway_still_latches_excess_active_yaw(angle):
    r = wave_rig(True)
    r.a = angle
    r.feed(); r.core.preset_query(0, 0, 400, received=r.t); r.core.tick()
    assert r.core.status == 'UNKNOWN' and r.core.reason == 'wave_yaw_limit'
    assert r.commands[-1] == (0., 0., 0.)


@pytest.mark.parametrize('mismatch', ['threshold', 'interaction', 'simulation'])
def test_wave_sway_registration_rejects_unmatched_or_modified_contract(mismatch):
    reg = registration(True)
    if mismatch == 'threshold':
        reg['skill_profile']['wave_execution']['active_yaw_limit_rad'] = .07
    elif mismatch == 'interaction':
        reg['interaction_profile'] = measured_action_interaction_profile()
    else:
        reg['simulation_only'] = False
    with pytest.raises(ValueError):
        candidate_executor(reg, lambda _:pytest.fail('Published during invalid registration'), preset=lambda _:None,
            wave_poses=dict(first=dict.fromkeys(RIGHT_ARM, 0.), terminal=dict.fromkeys(RIGHT_ARM, 0.)))


def test_wave_sway_metadata_changes_execution_identity_without_neural_changes(monkeypatch):
    monkeypatch.setattr(frozen, '_check', lambda *args:None)
    old = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE)
    new = frozen.inspect_frozen_history_candidate(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, wave_profile='balance-sway')
    assert new['policy']['execution_profile'] == balance_sway_action_skill_profile()
    assert new['checkpoint_id'] != old['checkpoint_id'] and new['policy']['policy_id'] != old['policy']['policy_id']
    for key in ('numeric_engine_id', 'input_map_id', 'artifacts', 'readout', 'initialization', 'source_sha256'):
        assert old['policy'][key] == new['policy'][key]
    assert not new['policy']['can_deploy']
    seen = []
    def inspect(root, path, *, turn_profile, wave_profile):
        seen.append((turn_profile, wave_profile))
        return new
    monkeypatch.setattr(frozen, 'inspect_frozen_history_candidate', inspect)
    forged = deepcopy(new); forged['checkpoint_id'] = 'forged'
    with pytest.raises(ValueError, match='identity differs'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, forged)
    assert seen == [('measured', 'balance-sway')]
    forged['policy']['execution_profile']['wave_execution']['return_yaw_limit_rad'] = .06
    with pytest.raises(ValueError, match='not an allowed contract'):
        frozen.load_frozen_history_policy(ROOT, ROOT/frozen.CANDIDATE_RELATIVE, forged)
    assert seen == [('measured', 'balance-sway')]  # No reconstruction/numerical loading for altered thresholds.


def test_wave_sway_cli_selection_and_forwarding_are_explicit(tmp_path, monkeypatch, capsys):
    def inspect(root, path, *, turn_profile, wave_profile):
        return dict(turn=turn_profile, wave=wave_profile)
    monkeypatch.setattr(frozen, 'inspect_frozen_history_candidate', inspect)
    path = tmp_path/frozen.CANDIDATE_RELATIVE; path.mkdir(parents=True)
    assert runtime.candidate(tmp_path, path)[1] == dict(turn='measured', wave='baseline')
    assert runtime.candidate(tmp_path, path, wave_profile='balance-sway')[1] == dict(turn='measured', wave='balance-sway')
    legacy = tmp_path/'data/legacy'; legacy.mkdir()
    with pytest.raises(ValueError, match='requires the fixed B'):
        runtime.candidate(tmp_path, legacy, wave_profile='balance-sway')
    for main in (runtime.main, driver.main):
        assert main([]) == 0
        assert json.loads(capsys.readouterr().out)['wave_profile'] == 'baseline'
        assert main(['--wave-profile', 'balance-sway']) == 0
        assert json.loads(capsys.readouterr().out)['wave_profile'] == 'balance-sway'
        with pytest.raises(ValueError, match='requires measured'):
            main(['--turn-profile', 'precision', '--wave-profile', 'balance-sway'])
    old = driver.runtime_command('/python', tmp_path/'run', 'continuous')
    assert '--wave-profile' not in old
    assert driver.runtime_command('/python', tmp_path/'run', 'continuous', wave_profile='balance-sway') == old+['--wave-profile', 'balance-sway']
    for turn in ('precision', 'settled'):
        with pytest.raises(ValueError, match='requires measured'):
            driver.runtime_command('/python', tmp_path/'run', 'continuous', turn, 'balance-sway')
        with pytest.raises(ValueError, match='requires measured'):
            runtime.candidate(tmp_path, path, turn_profile=turn, wave_profile='balance-sway')
