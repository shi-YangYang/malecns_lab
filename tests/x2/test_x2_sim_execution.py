"""Execution-binding checks with archived arrays and fake control feedback only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2 import SOURCE as INPUT_SOURCE
from malecns_lab.control.x2_behavior import Behavior, Phase, DecisionGate, digest
from malecns_lab.control.x2_interaction import interaction_profile, fixed_advance_interaction_profile
from malecns_lab.control.x2_skill_profile import (
    candidate_skill_profile, fixed_advance_skill_profile, validate_candidate_skill_profile,
)
from malecns_lab.experiments.x2_behavior_bridge import candidate_executor, registered_skill_profile, wrap_execution
from malecns_lab.learning import x2_sim_execution_policy as execution
from malecns_lab.learning.x2_behavior_session import checkpoint_registration_metadata, MAX_PACKET_BYTES
from .test_x2_behavior import observation, decision
from .test_x2_skills import Rig

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def archived_candidate(tmp_path):
    source = ROOT/execution.SOURCE['path']
    if not source.is_dir():
        pytest.skip('Pinned paired checkpoint is a local research asset')
    original = json.loads((source/'checkpoint.json').read_text())
    target_source = tmp_path/execution.SOURCE['path']
    target_source.mkdir(parents=True)
    candidate = tmp_path/'data/x2_behavior_sim_execution_test'
    candidate.mkdir()
    for name in ('checkpoint.json', 'arrays.npz'):
        shutil.copyfile(source/name, target_source/name)
    shutil.copyfile(source/'arrays.npz', candidate/'arrays.npz')
    metadata = dict(schema=original['schema'], policy=execution.execution_policy_identity(original['policy']),
        arrays=deepcopy(original['arrays']), arrays_sha256=original['arrays_sha256'], context=execution.derivation_context())
    metadata['checkpoint_id'] = digest(metadata)
    (candidate/'checkpoint.json').write_text(json.dumps(metadata))
    return tmp_path, candidate, original, metadata


def test_real_parameter_archive_and_identity_are_unchanged_and_tampering_rejected(archived_candidate):
    root, path, original, metadata = archived_candidate
    full, source, _ = execution.validate_execution_checkpoint(root, path, metadata)
    assert (path/'arrays.npz').read_bytes() == (source/'arrays.npz').read_bytes()
    assert len(full['arrays']) == 7 and full['arrays'] == original['arrays']
    assert full['checkpoint_id'] != original['checkpoint_id'] and full['policy']['policy_id'] != original['policy']['policy_id']
    for name in ('input_map', 'model', 'neurons', 'heads', 'source_checkpoint_id', 'readout_fit'):
        assert full['policy'][name] == original['policy'][name]
    old, new = deepcopy(interaction_profile()), deepcopy(fixed_advance_interaction_profile())
    for value in (old, new):
        value.pop('profile_id'); value.pop('skill_profile_id')
    assert old == new
    registered = checkpoint_registration_metadata(full)
    assert len(json.dumps(registered, separators=(',', ':')).encode()) < MAX_PACKET_BYTES
    assert registered['policy']['execution_source'] == execution.SOURCE
    forged = deepcopy(metadata)
    forged['policy']['execution_profile']['advance_execution']['velocity_m_s'][0] = .1
    forged['policy']['policy_id'] = digest({k:v for k,v in forged['policy'].items() if k != 'policy_id'})
    forged['checkpoint_id'] = digest({k:v for k,v in forged.items() if k != 'checkpoint_id'})
    (path/'checkpoint.json').write_text(json.dumps(forged))
    with pytest.raises(ValueError, match='Execution policy identity'):
        execution.validate_execution_checkpoint(root, path)
    (path/'checkpoint.json').write_text(json.dumps(metadata))
    (source/'checkpoint.json').write_text(json.dumps(original))  # Same values, different pinned bytes.
    with pytest.raises(ValueError, match='source assets changed'):
        execution.validate_execution_checkpoint(root, path)


def test_shared_loader_strictly_restores_parent_before_binding_new_execution(archived_candidate, monkeypatch):
    root, path, original, metadata = archived_candidate
    from malecns_lab.learning import x2_behavior_training as training, x2_inference as frozen
    from malecns_lab.learning import x2_interaction_policy, x2_interaction_paired_policy
    from malecns_lab.learning.x2_behavior_policy import X2BehaviorPolicy
    from malecns_lab.learning.x2_behavior_session import load_candidate_policy, NeuralSession
    class FixturePolicy:
        policy_id = original['policy']['policy_id']
        _identity = deepcopy(original['policy'])
        encoder = SimpleNamespace(input_map_id=original['policy']['input_map']['input_map_id'])
        @property
        def identity(self): return deepcopy(self._identity)
        def initial_state(self, *a, **kw): return None
        def state_identity(self, state): return {'state_id': 'fixture-state'}
    monkeypatch.setattr(frozen, 'read_anchors', lambda *_:{'run': {'policy': {}}, 'checkpoint': {}})
    monkeypatch.setattr(frozen, 'load_parameters', lambda *_:{'theta': None})
    monkeypatch.setattr(X2BehaviorPolicy, 'from_reference', lambda *a, **kw:FixturePolicy())
    monkeypatch.setattr(x2_interaction_policy, 'bind_interaction_policy', lambda *a, **kw:None)
    monkeypatch.setattr(x2_interaction_paired_policy, 'bind_paired_policy', lambda *a, **kw:None)
    restored = []
    def strict(directory, policy):
        full = training.inspect_checkpoint(directory)
        assert full['policy'] == policy.identity
        restored.append(full['checkpoint_id'])
        return full
    monkeypatch.setattr(training, 'restore_checkpoint', strict)
    policy = load_candidate_policy(root, checkpoint_registration_metadata(metadata), path)
    assert restored == [original['checkpoint_id'], metadata['checkpoint_id']]
    session = NeuralSession(policy, metadata['checkpoint_id'], 'run', 'session', None, lambda *a, **kw:None)
    assert session.profile == fixed_advance_interaction_profile() and policy.encoder.input_map_id == original['policy']['input_map']['input_map_id']


def registration():
    return dict(run_id='run', session_id='session', qualification_only=True, simulation_only=True, wave_enabled=False,
        skill_profile=fixed_advance_skill_profile(), interaction_profile=fixed_advance_interaction_profile())


def test_fixed_profile_registration_rejects_non_simulation_and_mismatches_before_publish(tmp_path):
    reg = registration()
    (tmp_path/'behavior_registration.json').write_text(json.dumps(reg))
    assert registered_skill_profile(tmp_path, reg) == fixed_advance_skill_profile()
    commands = []
    for change in ({'simulation_only': False}, {'qualification_only': False}, {'interaction_profile': interaction_profile()}):
        with pytest.raises(ValueError):
            candidate_executor({**reg, **change}, commands.append)
    forged = deepcopy(reg['skill_profile']); forged['command_seconds_by_behavior']['ADVANCE'] = 2.
    with pytest.raises(ValueError):
        validate_candidate_skill_profile(forged)
    with pytest.raises(ValueError):
        registered_skill_profile(tmp_path, {**reg, 'skill_profile': candidate_skill_profile()})
    assert commands == [] and candidate_skill_profile()['command_seconds_by_behavior']['ADVANCE'] == 1.


def fixed_rig():
    r = Rig.__new__(Rig)
    r.t, r.x, r.y, r.a, r.joint, r.player, r.source = 1., 0., 0., 0., 0., 0, INPUT_SOURCE
    r.commands, r.presets, r.timed = [], [], []
    def publish(command):
        r.commands.append(command); r.timed.append((r.t, command))
    r.core = candidate_executor(registration(), publish, clock=lambda:r.t)
    r.core.motion.arm(); r.core.motion.confirm_source_registration()
    for _ in range(65):
        r.feed(); r.core.standing_ready(); r.core.tick()
    return r


def start(r):
    raw = decision(Behavior.ADVANCE, Phase.START)
    gate = DecisionGate('run', 'session', 'policy', 'checkpoint', 'input')
    gate.observe(observation(received_monotonic=r.t, capabilities=tuple(fixed_advance_interaction_profile()['capabilities'])))
    assert gate.admit(raw, r.t).accepted
    assert r.core.submit_decision(raw, r.t)
    event = wrap_execution(registration(), r.core.events[0], 0, [], {})
    gate.execution_event(event)
    assert raw.intensity == r.core.command.intensity == event.mc_request['model_intensity'] == .5
    assert event.mc_request['resolved_velocity'] == (.2, 0., 0.)
    return gate


def test_fixed_advance_preserves_raw_decision_and_continue_cannot_extend_one_second():
    r = fixed_rig(); gate = start(r)
    deadline = r.core.deadline
    for i in range(65):
        r.x += .002 if r.t < deadline else 0.
        r.feed(speed=.1 if r.t < deadline else 0.)
        if i == 10:
            continuation = decision(Behavior.ADVANCE, Phase.CONTINUE, decision_seq=1)
            assert gate.admit(continuation, r.t).accepted and r.core.submit_decision(continuation, r.t)
            assert r.core.deadline == deadline
        r.core.tick()
    assert {command for _,command in r.timed if command != (0.,0.,0.)} == {(.2,0.,0.)}
    assert all(at < deadline for at,command in r.timed if command != (0.,0.,0.))
    assert r.commands[-1] == (0.,0.,0.)
    for _ in range(65): r.feed(); r.core.tick()
    assert r.core.status == 'COMPLETED' and r.core.events[-1]['stop_confirmed']
    old = Rig()
    assert old.core.submit_decision(decision(Behavior.ADVANCE, Phase.START), old.t)
    old.feed(); old.core.tick()
    assert old.commands[-1] == (.1,0.,0.)
    assert r.core.motion.resolve_velocity('TURN_LEFT', .5) == (0.,0.,.1)


@pytest.mark.parametrize('emergency', [False, True])
def test_fixed_advance_finish_and_emergency_still_zero_without_renewing(emergency):
    r = fixed_rig(); gate = start(r)
    r.feed(); r.core.tick()
    assert r.commands[-1] == (.2,0.,0.)
    if emergency:
        r.core.cancel('user_emergency_stop')
    else:
        finish = decision(Behavior.HOLD, Phase.FINISH, decision_seq=1)
        assert gate.admit(finish, r.t).accepted and r.core.submit_decision(finish, r.t)
    before = len(r.commands)
    # Existing adapter schedules zero for the next 50 Hz publication tick.
    for _ in range(10): r.feed(); r.core.tick()
    assert r.commands[-1] == (0.,0.,0.)
    assert all(command == (0.,0.,0.) for command in r.commands[before:])
