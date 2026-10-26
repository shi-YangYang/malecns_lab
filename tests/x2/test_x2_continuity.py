"""Offline protocol/evidence checks; no neural runtime, SDK or VM is started."""
import json
import io
import math
from pathlib import Path

import pytest

from malecns_lab.experiments import x2_continuity as diagnostic


RUN = 'desktop-20261026-010000-12345678'
RUN_PATH = Path('/fixture') / RUN


def status(**updates):
    return dict(schema='malecns_lab.desktop_event.v1', type='status', output=str(RUN_PATH),
        run_id=RUN, state='running', phase='waiting_task', brain_controlled=True,
        action_busy=False, request_pending=False, active_request_id=None, **updates)


def decision(request='request-1', behavior='ADVANCE', sequence=1):
    return dict(kind='raw_decision', decision=dict(run_id=RUN, session_id='session', request_id=request,
        decision_seq=sequence, behavior=behavior, phase='FINISH' if behavior == 'HOLD' else 'START',
        intensity=.5, checkpoint_id='fixed-checkpoint', policy_id='fixed-policy', input_map_id='v4-input'))


def event(request='request-1', behavior='ADVANCE', sequence=1, event_sequence=0,
          state='COMPLETED', angle=math.pi / 2, stop=True, arm_position=None):
    raw = dict(forward_distance_m=1.21, lateral_distance_m=.01, angle_rad=angle,
        advance_direction_ok=True, joint_return_rad=.04, return_reference='official_animation_terminal_frame')
    if arm_position is not None:
        raw['sensors'] = dict(arm=dict(message=dict(joints=[dict(name=name, position=arm_position,
            velocity=.01, error_code=0) for name in diagnostic.ARM])))
    return dict(kind='execution', event=dict(run_id=RUN, session_id='session', event_seq=event_sequence,
        decision_seq=sequence, execution_id='skill-' + str(sequence), request_id=request,
        behavior=behavior, status=state, fact='status', protection_reason=None,
        raw_feedback=raw, stop_evidence=dict(stop_confirmed=stop)))


def driver(tasks=('SHOW_FORWARD', 'FACE_LEFT')):
    result = diagnostic.SequenceDriver(tasks, RUN_PATH, 120.)
    result.status_event(status())
    assert result.advance(10., caught_up=True) == dict(command='task', task=tasks[0])
    queued = status(task=tasks[0], request_id='request-1')
    queued.update(phase='queued', action_busy=True, request_pending=True, active_request_id='request-1')
    result.status_event(queued)
    return result


@pytest.mark.parametrize('field', ['notice', 'phase'])
def test_matched_rejection_fails_immediately_for_both_protocol_versions(field):
    current = driver()
    value = status(rejected_task='GREET', reason='action_busy')
    value[field] = 'request_rejected'
    current.status_event(value)  # An unrelated rejection cannot fail this request.
    value['rejected_task'] = 'SHOW_FORWARD'
    with pytest.raises(ValueError, match='Runtime rejected request'):
        current.status_event(value)


def test_default_covers_all_16_directed_pairs_and_end_restart():
    base = diagnostic.default_sequence()
    assert len(base) == 17
    assert len(set(zip(base, base[1:]))) == 16
    assert set(zip(base, base[1:])) == {(a, b) for a in diagnostic.TASKS for b in diagnostic.TASKS}
    planned = diagnostic.planned_tasks(cycles=2)
    assert planned == base * 2 + ['END_INTERACTION', 'SHOW_FORWARD']
    assert diagnostic.planned_tasks('GREET,SHOW_FORWARD', 3) == ['GREET', 'SHOW_FORWARD'] * 3
    with pytest.raises(ValueError):
        diagnostic.planned_tasks('ADVANCE')  # Execution/SDK commands cannot masquerade as task enums.


def test_preview_never_starts_a_subprocess_or_creates_output(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Preview attempted to launch a process')
    monkeypatch.setattr(diagnostic.subprocess, 'Popen', forbidden)
    output = tmp_path / 'absent'
    assert diagnostic.main(['--output', str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'preview' and not result['brain_loaded'] and not result['simulator_started']
    assert result['max_wall_seconds'] == 480 and result['runtime_session_profile'] == 'qualification'
    assert not output.exists()
    old = diagnostic.runtime_command('/python', RUN_PATH, 'qualification')
    assert '--session-profile' not in old
    assert diagnostic.runtime_command('/python', RUN_PATH, 'continuous') == old + ['--session-profile', 'continuous']


@pytest.mark.parametrize('selection', ['precision', 'settled'])
def test_turn_profile_preview_and_runtime_forwarding_are_explicit(monkeypatch, capsys, selection):
    monkeypatch.setattr(diagnostic, 'execute', lambda *args: pytest.fail('Preview attempted execution'))
    assert diagnostic.main([]) == 0
    assert json.loads(capsys.readouterr().out)['turn_profile'] == 'measured'
    assert diagnostic.main(['--turn-profile', selection]) == 0
    assert json.loads(capsys.readouterr().out)['turn_profile'] == selection
    old = diagnostic.runtime_command('/python', RUN_PATH, 'continuous')
    assert '--turn-profile' not in old
    assert diagnostic.runtime_command('/python', RUN_PATH, 'continuous', selection) == (
        old + ['--turn-profile', selection])
    with pytest.raises(ValueError, match='Unsupported turn profile'):
        diagnostic.runtime_command('/python', RUN_PATH, 'continuous', 'custom')


def test_incremental_tail_keeps_partial_lines_and_never_reads_entire_log(tmp_path):
    path = tmp_path / 'events.jsonl'
    tail = diagnostic.JsonlTail(path)
    assert tail.poll() == [] and tail.caught_up
    path.write_bytes(b'{"kind":"one"}\n{"kind":')
    assert tail.poll() == [dict(kind='one')] and not tail.caught_up
    with path.open('ab') as stream:
        stream.write(b'"two"}\n' + b'{"kind":"three"}\n' * 40000)
    rows = tail.poll()
    assert rows[0] == dict(kind='two') and not tail.caught_up
    assert tail.offset < path.stat().st_size
    count = len(rows)
    while not tail.caught_up:
        count += len(tail.poll())
    assert count == 40001
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='truncated'):
        tail.poll()


def test_next_task_requires_matched_waiting_stop_and_complete_evidence():
    current = driver()
    current.evidence.brain(decision())
    current.evidence.bridge(event())
    current.status_event(status(request_id='old-request', stop_confirmed=True))
    assert current.advance(11., caught_up=True) is None
    current.status_event(status(request_id='request-1', stop_confirmed=False))
    assert current.advance(12., caught_up=True) is None
    current.status_event(status(request_id='request-1', stop_confirmed=True))
    assert current.advance(13., caught_up=False) is None
    assert current.advance(14., caught_up=True) == dict(command='task', task='FACE_LEFT')
    assert current.results[0]['status'] == 'passed'
    assert current.results[0]['metrics']['forward_distance_m'] == 1.21
    assert current.results[1]['status'] == 'running'


@pytest.mark.parametrize('variant', ['hold_only', 'other_request', 'missing_brain', 'unconfirmed_stop', 'cancelled'])
def test_motion_cannot_pass_with_hold_foreign_or_incomplete_evidence(variant):
    current = driver(('SHOW_FORWARD',))
    request = 'another-request' if variant == 'other_request' else 'request-1'
    behavior = 'HOLD' if variant == 'hold_only' else 'ADVANCE'
    if variant != 'missing_brain':
        current.evidence.brain(decision(request=request, behavior=behavior))
    current.evidence.bridge(event(request=request, behavior=behavior, stop=variant != 'unconfirmed_stop',
                                  state='CANCELLED' if variant == 'cancelled' else 'COMPLETED'))
    current.status_event(status(request_id='request-1', stop_confirmed=True))
    with pytest.raises(ValueError):
        current.advance(15., caught_up=True)
    assert not current.complete


@pytest.mark.parametrize('task,angle,passed', [('FACE_LEFT', 90.79, True), ('FACE_RIGHT', -99.05, False),
    ('FACE_LEFT', -90., False), ('FACE_RIGHT', -90., True), ('FACE_LEFT', float('nan'), False)])
def test_turn_uses_final_signed_error_not_executor_completed_flag(task, angle, passed):
    evidence = diagnostic.Evidence(RUN)
    behavior = diagnostic.EXPECTED[task]
    evidence.brain(decision(behavior=behavior))
    evidence.bridge(event(behavior=behavior, angle=math.radians(angle)))
    result = evidence.request(task, 'request-1')
    assert result['passed'] is passed
    if not passed:
        assert 'final_turn_error_exceeds_5deg' in result['failures']


def test_wave_requires_same_execution_joint_excursion_and_stopped_return():
    evidence = diagnostic.Evidence(RUN)
    evidence.brain(decision(behavior='WAVE_RIGHT'))
    evidence.bridge(event(behavior='WAVE_RIGHT', state='ACCEPTED', arm_position=0.))
    evidence.bridge(event(behavior='WAVE_RIGHT', state='STARTED', event_sequence=1, arm_position=.3))
    evidence.bridge(event(behavior='WAVE_RIGHT', event_sequence=2, arm_position=.04))
    result = evidence.request('GREET', 'request-1')
    assert result['passed'] and result['metrics']['measured_joint_excursion_rad'] == .3
    evidence.events['request-1'][-1]['arm'][diagnostic.ARM[0]]['velocity'] = .1
    assert not evidence.request('GREET', 'request-1')['passed']


def test_end_requires_its_own_brain_hold_and_allows_a_new_task_only_after_stop():
    current = driver(('END_INTERACTION', 'GREET'))
    current.evidence.brain(decision(behavior='HOLD'))
    current.evidence.bridge(event(behavior='HOLD'))
    current.status_event(status(request_id='prior-motion', stop_confirmed=True))
    assert current.advance(11., caught_up=True) is None
    current.status_event(status(request_id='request-1', stop_confirmed=True))
    assert current.advance(12., caught_up=True) == dict(command='task', task='GREET')


def test_failure_and_request_timeout_do_not_advance_or_retry():
    current = driver()
    current.evidence.brain(decision())
    current.evidence.bridge(event(state='FAILED'))
    with pytest.raises(ValueError, match='Actual execution failed'):
        current.advance(12., caught_up=True)
    current.fail_active('observed_failure', 12.)
    assert current.results[0]['status'] == 'failed' and current.results[1]['status'] == 'not_run'
    timed = driver()
    with pytest.raises(ValueError, match='Single request timeout'):
        timed.advance(131., caught_up=True)
    assert timed.next_index == 1


def test_foreign_run_or_reordered_bridge_events_rejected():
    evidence = diagnostic.Evidence(RUN)
    evidence.bridge(event())
    with pytest.raises(ValueError, match='Reordered'):
        evidence.bridge(event())
    foreign = event(event_sequence=1)
    foreign['event']['run_id'] = 'foreign'
    with pytest.raises(ValueError, match='Foreign execution'):
        evidence.bridge(foreign)


def test_operator_eof_uses_only_safe_stop_and_preserves_failed_summary(tmp_path, monkeypatch):
    commands = []

    class FakeChild:
        pid = 12345

        def __init__(self, command, **kwargs):
            self.exit_code = None
            self.stdout = io.StringIO('')
            self.stdin = self
            self.output = Path(command[command.index('--output') + 1])
            self.output.mkdir(parents=True)

        def write(self, text):
            value = json.loads(text)
            commands.append(value)
            assert value == {'command': 'stop'}
            self.exit_code = 0
            (self.output / 'desktop_session.json').write_text(json.dumps(dict(
                output=str(self.output), state='cancelled', resources_stopped=True, stop_confirmed=None)))

        def flush(self):
            pass

        def poll(self):
            return self.exit_code

        def send_signal(self, sig):
            raise AssertionError('Safe stdin stop was available')

    monkeypatch.setattr(diagnostic, 'ROOT', tmp_path)
    monkeypatch.setattr(diagnostic.sys, 'stdin', io.StringIO(''))
    monkeypatch.setattr(diagnostic.subprocess, 'Popen', FakeChild)
    output = tmp_path / 'diagnostic'
    assert diagnostic.main(['--execute', '--output', str(output), '--sequence', 'GREET,SHOW_FORWARD']) == 1
    result = json.loads((output / 'summary.json').read_text())
    assert commands == [{'command': 'stop'}]
    assert result['status'] == 'failed' and result['stop_requested']
    assert result['resources_stopped'] and not result['runtime_alive']
    assert all(request['status'] == 'not_run' for request in result['requests'])
    assert result['verified_directed_pair_count'] == 0
