"""Continuous lifecycle/ACK fixtures only: no neural inference, ROS or VM."""
from copy import deepcopy
import io
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import DecisionGate, Feedback, ExecutionEvent, Observation
from malecns_lab.control.x2_interaction import measured_action_interaction_profile
from malecns_lab.control.x2_skill_profile import measured_action_skill_profile
from malecns_lab.control.x2_skills import SkillCommand
from malecns_lab.experiments.x2_behavior_bridge import RemoteExchange, SupervisorLease, WaveSession, wrap_execution
from malecns_lab.learning.x2_behavior_session import (
    ClockExchange, NeuralSession, SCHEMA, host_event, session_profile_limits, validate_session_profile,
)
from malecns_lab.runtime.x2.behavior import BehaviorRuntime, Commands, main
from .test_x2_action_continuity import rig
from .test_x2_behavior_desktop import Clock, decision, FixturePolicy


def runtime(tmp_path, profile='continuous'):
    return BehaviorRuntime(tmp_path, Path(__file__).resolve().parents[2], Commands(io.StringIO()), lambda row: None,
        tmp_path/'unused', {'policy': {'interaction_profile': measured_action_interaction_profile(),
                                    'execution_profile': measured_action_skill_profile()}}, session_profile=profile)


@pytest.mark.parametrize('outcome', ['ready', 'timeout', 'end', 'latched'])
def test_ack_rechecks_current_body_before_unlock_with_original_deadline(outcome):
    clock = Clock(10.)
    session = NeuralSession.__new__(NeuralSession)
    session.gate = SimpleNamespace(active=None, blocked_reason=None)
    session.runtime_contexts = {}
    records = []
    session.record = lambda kind, **fields: records.append(kind)
    count = 0
    def snapshot():
        nonlocal count
        count += 1
        session.runtime_contexts[count] = dict(protection_state='LATCHED' if outcome == 'latched' else 'READY')
        return SimpleNamespace(request_id='request', observation_seq=count,
            feedback=SimpleNamespace(valid=True, age_s=0., standing=count > 1 and outcome == 'ready'))
    def wait(seconds):
        clock.value += seconds
    call = lambda: session.confirm_closed_request_ready('request', snapshot,
        lambda: outcome == 'end' and count > 0, lambda: None, deadline=10.25, clock=clock, wait=wait)
    if outcome == 'ready':
        assert call() and count == 2
        assert records == ['request_ack_wait_stopped', 'request_ack_stopped_restored']
    elif outcome == 'end':
        assert not call()
    else:
        with pytest.raises(ValueError, match='timed out|protection changed'):
            call()


def test_registration_has_explicit_none_limits_and_qualification_is_unchanged(capsys):
    old, new = session_profile_limits(), session_profile_limits('continuous')
    assert old['session_contract_id'] == 'malecns.x2.session.qualification.v1'
    assert new['session_contract_id'] == 'malecns.x2.session.continuous.v2'
    assert new['request_scope'] == 'measured_single_brain_selected_action.v1'
    assert (old['vm_seconds'], old['max_decisions'], old['max_requests']) == (480., 160, 50)
    assert (old['wall_seconds'], old['worker_wall_seconds'], old['bridge_wall_seconds']) == (900., 865., 445.)
    for key in ('wall_seconds', 'vm_seconds', 'worker_wall_seconds', 'bridge_wall_seconds', 'max_decisions', 'max_requests'):
        assert new[key] is None
    assert validate_session_profile(new) == new
    forged = dict(old, vm_seconds=None)
    with pytest.raises(ValueError, match='limits'):
        validate_session_profile(forged)
    assert main(['--session-profile', 'continuous']) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview['session_contract_id'] == new['session_contract_id']
    assert not preview['brain_loaded'] and not preview['simulator_started']


def single_action_fixture(task='SHOW_FORWARD', behavior='ADVANCE', *, continuous=True):
    policy = FixturePolicy()
    policy.identity = {'interaction_profile': measured_action_interaction_profile()}
    policy.set_runtime_context = lambda *args: None
    policy.behavior, policy.phase = behavior, 'FINISH' if behavior == 'HOLD' else 'START'
    records = []
    session = NeuralSession(policy, 'checkpoint', 'run', 'session',
        SimpleNamespace(begin_decision=lambda *args: None), lambda kind, **fields: records.append((kind, fields)),
        continuous=continuous)
    count, current_status = 0, 'IDLE'
    def snapshot(*, request='request', current_task=task, standing=True):
        nonlocal count
        count += 1
        observation = Observation('run', 'session', count, request, current_task, time.monotonic(),
            capabilities=('HOLD', 'ADVANCE', 'TURN_LEFT', 'TURN_RIGHT', 'WAVE_RIGHT'),
            feedback=Feedback(valid=True, age_s=.03, source='official_fixture', standing=standing,
                status=current_status, behavior=behavior, request_id='request' if current_status != 'IDLE' else '',
                execution_id='skill-1' if current_status != 'IDLE' else ''))
        session.runtime_context(observation, dict(protection_state='READY', event_sequence=session.gate.last_event_seq,
            facts=dict(failures=[], core_status=current_status, source_registered=True, armed=True,
                       stable_preparation_completed=True, fresh_body=True, ownership_supported=True)))
        return observation
    start, admission = session.infer(snapshot())
    assert admission.accepted
    def event(status, *, stopped=True):
        nonlocal current_status
        current_status = status
        value = ExecutionEvent('run', 'session', session.gate.last_event_seq + 1, start.decision_seq,
            'skill-1', start.request_id, 'sim', 'skills', status, behavior, time.monotonic(),
            stop_evidence={'stop_confirmed': stopped and status in ('COMPLETED', 'CANCELLED')})
        session.execution_event(value)
    return session, policy, records, snapshot, event, start


@pytest.mark.parametrize('task,behavior', [('SHOW_FORWARD', 'ADVANCE'), ('FACE_LEFT', 'TURN_LEFT'),
    ('FACE_RIGHT', 'TURN_RIGHT'), ('GREET', 'WAVE_RIGHT')])
def test_single_action_completion_drains_raw_wave_without_second_skill_then_allows_new_request(task, behavior):
    session, policy, records, snapshot, event, start = single_action_fixture(task, behavior)
    for status in ('ACCEPTED', 'STARTED', 'COMPLETED'):
        event(status)
    assert session.gate.completed_action('request') == start and session.closure_permitted(start)
    policy.behavior, policy.phase = 'WAVE_RIGHT', 'START'
    assert session.drain_finished_request(start, snapshot, lambda: False, lambda: None, decision_limit=None)
    raw = [row for kind, row in records if kind == 'raw_decision'][1:]
    assert len(raw) == 3 and all(row['decision']['behavior'] == 'WAVE_RIGHT' for row in raw)
    assert all(row['closed_request_completion_decision_seq'] == start.decision_seq for row in raw)
    assert all(not row['accepted'] and row['reason'] == 'closed_request_event_drain'
               for kind, row in records if kind == 'admission' and row['decision_seq'] > start.decision_seq)
    assert session.gate.last_event_seq == 2 and list(session.gate._admitted) == [start.decision_seq]
    assert not session.pending_neural_events and session.consumed_event_sequence == 2
    assert next(row for kind, row in records if kind == 'request_closing')['completion_basis'] == 'measured_single_action'
    session.retire_closed_request(1)
    policy.behavior, policy.phase = behavior, 'START'
    _, admission = session.infer(snapshot(request='next'))
    assert admission.accepted and policy.inputs[-1].request_changed


def test_single_action_completion_during_refresh_blocks_raw_wave_and_late_remote_admission():
    session, policy, records, snapshot, event, start = single_action_fixture()
    event('ACCEPTED')
    policy.behavior, policy.phase = 'WAVE_RIGHT', 'START'
    def refresh():
        event('STARTED'); event('COMPLETED')
        snapshot()
    raw, admission = session.infer(snapshot(), refresh)
    assert raw.behavior == 'WAVE_RIGHT' and not admission.accepted
    assert admission.reason == 'completed_action_request' and session.gate.active is None
    assert list(session.gate._admitted) == [start.decision_seq]
    assert [e.event_seq for e in session.pending_neural_events] == [1, 2]
    assert session.drain_finished_request(start, snapshot, lambda: False, lambda: None, decision_limit=None)
    # Same shared gate on the bridge consumes a late decision but emits no event or skill.
    late_observation = snapshot()
    session.gate.observe(late_observation)
    late = decision(late_observation, 100, 'WAVE_RIGHT', 'START')
    admission = session.gate.admit(late, time.monotonic())
    assert admission.reason == 'completed_action_request' and session.gate.active is None
    assert session.gate.last_decision_seq == 100 and session.gate.last_event_seq == 2
    assert 100 not in session.gate._admitted


@pytest.mark.parametrize('case', ['early_hold', 'wrong_behavior', 'cancelled', 'failed', 'unconfirmed_stop'])
def test_single_action_never_qualifies_hold_wrong_behavior_or_unsuccessful_execution(case):
    behavior = 'HOLD' if case == 'early_hold' else 'TURN_LEFT' if case == 'wrong_behavior' else 'ADVANCE'
    session, _, _, _, event, start = single_action_fixture(behavior=behavior)
    event('ACCEPTED'); event('STARTED')
    event('CANCELLED' if case == 'cancelled' else 'FAILED' if case == 'failed' else 'COMPLETED',
          stopped=case != 'unconfirmed_stop')
    assert session.gate.completed_action('request') is None and not session.closure_permitted(start)


def test_single_action_closure_waits_for_fresh_standing_and_end_can_supersede_drain():
    session, policy, _, snapshot, event, start = single_action_fixture()
    for status in ('ACCEPTED', 'STARTED', 'COMPLETED'):
        event(status)
    calls = []
    def wait(seconds):
        assert len(policy.inputs) == 1 and len(session.pending_neural_events) == 3
        calls.append(seconds)
    assert session.drain_finished_request(start, lambda: snapshot(standing=bool(calls)),
        lambda: False, lambda: None, decision_limit=None, wait=wait)
    assert calls == [.1]
    session2, _, records, snapshot2, event2, start2 = single_action_fixture()
    for status in ('ACCEPTED', 'STARTED', 'COMPLETED'):
        event2(status)
    assert not session2.drain_finished_request(start2, snapshot2, lambda: True, lambda: None, decision_limit=None)
    assert len(session2.pending_neural_events) == 3 and session2.consumed_event_sequence == -1
    assert not any(kind == 'request_closing_complete' for kind, _ in records)


def test_single_action_end_still_requires_actual_brain_hold_and_qualification_keeps_old_scope():
    session, _, _, snapshot, event, finish = single_action_fixture('END_INTERACTION', 'HOLD')
    for status in ('ACCEPTED', 'STARTED', 'COMPLETED'):
        event(status)
    assert session.finish_ends_request(finish) and session.closure_permitted(finish)
    assert session.gate.completed_action('request') is None
    assert session.drain_finished_request(finish, snapshot, lambda: False, lambda: None, decision_limit=None)
    old, _, _, _, old_event, start = single_action_fixture(continuous=False)
    for status in ('ACCEPTED', 'STARTED', 'COMPLETED'):
        old_event(status)
    assert old.gate.completed_action('request') is None and not old.closure_permitted(start)


def test_continuous_idle_removes_only_total_deadline_and_request_count(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    assert math.isfinite(value.deadline())  # Startup remains bounded.
    value.session_running = True
    value.started -= 10000.; value.vm_started = value.started + 100.
    assert math.isinf(value.deadline())
    assert value.budget_record()['wall_deadline_monotonic'] is None
    value.tasks_open = True; value.task_sequence = 500
    assert value.accept_task('GREET', 1.)
    assert not value.accept_task('FACE_LEFT', 1.1)  # Busy still rejects.
    assert value.accept_task('END_INTERACTION', 1.2)
    assert not value.accept_task('END_INTERACTION', 1.3)
    monkeypatch.setattr('malecns_lab.runtime.x2.behavior.shutil.disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='disk reserve'):
        value.guard()
    value.cleanup = lambda: None
    value.shutdown_behavior('fixture')
    assert math.isfinite(value.deadline(True)) and value.cleanup_deadline is not None
    old = runtime(tmp_path, 'qualification')
    old.tasks_open = True; old.task_sequence = 50
    assert not old.accept_task('GREET', 1.)
    assert math.isfinite(old.deadline())


def test_supervisor_lease_uses_local_elapsed_time_and_never_renews_on_reread():
    clock = Clock(10.)
    lease = SupervisorLease('run', 'session', clock=clock)
    heartbeat = dict(run_id='run', session_id='session', sequence=1, host_monotonic=999999.)
    lease.check(heartbeat)
    clock.value = 14.9; lease.check(heartbeat)
    clock.value = 15.
    with pytest.raises(ValueError, match='expired'):
        lease.check(heartbeat)
    heartbeat['sequence'] = 2; lease.check(heartbeat)
    with pytest.raises(ValueError, match='Invalid'):
        lease.check(dict(heartbeat, sequence=1))


def test_host_request_watchdog_survives_a_stalled_worker_and_end_has_its_own_request(tmp_path):
    value = runtime(tmp_path)
    value.session_running = value.tasks_open = True
    assert value.accept_task('GREET', 100.)
    value.guard_request_deadline(219.9)
    with pytest.raises(ValueError, match='120 s'):
        value.guard_request_deadline(220.)
    assert value.accept_task('END_INTERACTION', 221.)
    value.guard_request_deadline(230.)
    assert value.active_request_deadline == 341.
    assert value.phase_status(dict(phase='waiting_task', request_id=value.active_request_id, stop_confirmed=True))
    value.guard_request_deadline(100000.)
    assert value.active_request_deadline is None


@pytest.mark.parametrize('profile,allowed', [('qualification', False), ('continuous', True)])
def test_wave_count_is_not_a_continuous_lifetime_limit(profile, allowed, tmp_path):
    from malecns_lab.experiments.x2_preset_compat import FIXED
    starts = []
    wave = WaveSession(tmp_path, {'session_profile': profile}, object(), {}, lambda *a, **k: None,
        None, None, None, boundary_factory=lambda *a, **k: None)
    wave.actions = 50
    wave.stand_preparation = SimpleNamespace(request=None, start=lambda core: starts.append(core))
    if allowed:
        wave.preset(FIXED)
        assert wave.actions == 51 and len(starts) == 1
    else:
        with pytest.raises(ValueError):
            wave.preset(FIXED)
        assert not starts


def test_two_hundred_closed_requests_retire_only_consumed_history_and_keep_replay_guards():
    profile = measured_action_interaction_profile()
    r = rig()
    hclock = Clock(1000.)
    host = ClockExchange('run', 'session', hclock, profile=profile)
    remote = RemoteExchange('run', 'session', lambda: r.t, profile=profile, continuous=True)
    gate = DecisionGate('run', 'session', 'policy', 'checkpoint', 'input', monotonic_execution_ids=True)
    policy = SimpleNamespace(policy_id='policy', encoder=SimpleNamespace(input_map_id='input'),
        identity={'interaction_profile': profile}, initial_state=lambda *a, **k: object(),
        state_identity=lambda state: {'state_id': 'fixture'}, set_runtime_context=lambda *a: None)
    session = NeuralSession(policy, 'checkpoint', 'run', 'session', None, lambda *a, **k: None, continuous=True)
    events, next_event = [], 0
    def execution(raw):
        nonlocal next_event
        event = wrap_execution(dict(run_id='run', session_id='session'), raw, next_event, [], {})
        next_event += 1
        gate.execution_event(event)
        events.append(event.to_dict())
        session.execution_event(host_event(event.to_dict(), 'run', 'session', hclock()))
    r.core.record = execution
    ack = None
    feedback = Feedback(valid=True, source='official_fixture', standing=True)
    first_message = None
    for number in range(1, 201):
        request = 'request-'+str(number)
        hclock.value += .01
        query = host.request('GREET', request, next_event-1, request_sequence=number, closed_request_ack=ack)
        packet = remote.observe(query, feedback, [])
        remote.acknowledge_closed_history(query, gate, r.core, events, next_event)
        obs, _ = host.accept(packet)
        if ack is not None:
            host.retire_requests(ack['request_sequence'], request)
        chosen = decision(obs, number)
        session.gate.observe(obs)
        assert session.gate.admit(chosen, hclock()).accepted
        message = dict(schema=SCHEMA, kind='decision', run_id='run', session_id='session',
                       nonce=query['nonce'], decision=chosen.to_dict())
        first_message = first_message or deepcopy(message)
        chosen, remote_obs, _ = remote.consume(message)
        gate.observe(remote_obs)
        assert gate.admit(chosen, r.t).accepted
        assert r.core.submit(SkillCommand(number, request, 'HOLD', 'FINISH', .5, r.t))
        assert len(events) == len(session.pending_neural_events) == len(r.core.events) == 3
        with pytest.raises(ValueError, match='unconsumed'):
            session.retire_closed_request(number)
        assert len(events) == len(session.pending_neural_events) == 3
        # Fixture consumer: actual model inference is intentionally not run here.
        while session.pending_neural_events:
            session.consumed_event_sequence = session.pending_neural_events.popleft().event_seq
        session.last_consumed_request_id = request
        ack = session.retire_closed_request(number, retire=False)
        query = host.request('GREET', request, next_event-1, request_sequence=number, closed_request_ack=ack)
        packet = remote.observe(query, feedback, [])
        remote.acknowledge_closed_history(query, gate, r.core, events, next_event)
        host.accept(packet)
        session.retire_closed_request(number)
        assert not events and not r.core.events and not session.execution_results
        assert not gate._admitted and not gate._used_executions
        assert len(remote.used) <= len(remote.issued) <= 32
        assert len(host.heading_targets) == len(remote.heading_targets) == 1
    assert next_event == r.core.event_sequence == 600
    assert gate.last_event_seq == session.gate.last_event_seq == 599
    assert gate.last_decision_seq == 200
    with pytest.raises(ValueError, match='Unknown/reused'):
        remote.consume(first_message)
    stale_query = dict(query, sequence=query['sequence']+1, nonce='f'*32, request_sequence=1)
    with pytest.raises(ValueError, match='Replayed'):
        remote.observe(stale_query, feedback, [])
    # A new decision may not reuse an old execution ID after the set is retired.
    hclock.value += .01
    query = host.request('GREET', 'next', 599, request_sequence=201, closed_request_ack=ack)
    packet = remote.observe(query, feedback, [])
    obs, _ = host.accept(packet)
    chosen = decision(obs, 201)
    gate.observe(obs); assert gate.admit(chosen, hclock()).accepted
    from malecns_lab.control.x2_behavior import ExecutionEvent
    with pytest.raises(ValueError, match='monotonic decision'):
        gate.execution_event(ExecutionEvent('run', 'session', 600, 201, 'skill-1', 'next',
            'sim', 'skills', 'ACCEPTED', 'HOLD', hclock()))
