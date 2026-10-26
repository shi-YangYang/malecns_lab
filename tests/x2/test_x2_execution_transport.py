"""Bounded execution mailboxes retain facts and archive full diagnostics; no ROS."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from malecns_lab.control.x2_behavior import ExecutionEvent, Feedback
from malecns_lab.control.x2_interaction import integral_action_interaction_profile
from malecns_lab.control.x2_skill_profile import integral_action_skill_profile
from malecns_lab.experiments.x2_behavior_bridge import (
    _archive_execution, _archive_summary, _execution_mailbox, RemoteExchange,
    publish_observation, wrap_execution, clean,
)
from malecns_lab.learning.x2_behavior_session import MAX_PACKET_BYTES, atomic, read_json, ClockExchange, host_event
from .test_x2_action_continuity import rig, step, submit


REG = dict(run_id='run', session_id='session')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def diagnostics():
    sample = dict(monotonic=400.123456789, phase='tracking', correction=1,
        error_rad=.180123456789, p=.360246913578, i=.799123456789, d=-.150123456789,
        raw_rad_s=1.009246913578, clipped_rad_s=.8, odom_sequence=12345,
        sample_at=400.123456789, sample_dt=.020123456789, integral_rule='same_direction_saturation_hold')
    stage = dict(phase='stopped', reason='measured_stop_confirmed', monotonic=400.123456789,
        correction=1, angle_rad=-1.3816839671800134, error_rad=-.1891123596148832,
        angular_speed_rad_s=.00123456789, commanded_velocity=[0.,0.,0.], pid=sample,
        nonzero_seconds=7.405214550000267, segment_deadline=405.123456789)
    return dict(phase='failed', corrections=1, error_rad=-.1891123596148832,
        controller=integral_action_skill_profile()['turn_execution']['feedback_controller'],
        motion_deadline=404.123456789, total_deadline=416.123456789,
        failure='integral_turn_corrections_exhausted', trace=[deepcopy(stage) for _ in range(64)],
        integral=dict(value=0., pid=dict(i=0., integral_rule='reset'), nonzero_seconds=7.405214550000267,
            segment_deadline=405.123456789, previous_stopped_error_rad=.2,
            sample_records_dropped=17, stage_records_dropped=3, samples=[dict(sample, odom_sequence=i) for i in range(512)]))


def full_event(status):
    return ExecutionEvent('run','session',14,17,'skill-17','request','sim','skills',status,'TURN_RIGHT',400.185646023,
        mc_request=dict(model_intensity=.5, resolved_velocity=[0.,0.,0.]),
        raw_feedback=dict(precision_turn=diagnostics(), status=status, reason='integral_turn_corrections_exhausted',
            angle_rad=-1.3816839671800134, stop_confirmed=status != 'UNKNOWN', turn_target_reached=True),
        stop_evidence=dict(stop_confirmed=status != 'UNKNOWN', base_stop_confirmed=True, angle_rad=-1.3816839671800134),
        protection_reason='original_protection' if status == 'UNKNOWN' else None).to_dict()


def recorder(tmp_path):
    def record(kind, **fields):
        with (tmp_path/'bridge_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(kind=kind, **fields), allow_nan=False)+'\n')
            stream.flush()
    return record


@pytest.mark.parametrize('status', ['COMPLETED','FAILED','UNKNOWN'])
def test_maximum_diagnostics_archive_once_and_fit_mailbox_and_observation(tmp_path, status):
    full = full_event(status); original = deepcopy(full)
    summary = dict(status=status, reason=full['raw_feedback']['reason'], terminal=True,
        execution_id='skill-17', precision_turn=deepcopy(full['raw_feedback']['precision_turn']),
        events=[deepcopy(full['raw_feedback']) for _ in range(3)], failures=[], angle_rad=-1.3816839671800134)
    before = dict(schema='fixture', **REG, event=full, core=summary)
    assert len(encoded(before)) > MAX_PACKET_BYTES
    log = recorder(tmp_path)
    compact = _archive_execution(full, log)
    packet = _execution_mailbox(REG, compact, summary)
    atomic(tmp_path/'execution_latest.json', packet)
    loaded = read_json(tmp_path/'execution_latest.json')
    delivered = host_event(loaded['event'], 'run', 'session', 10.)
    assert delivered.status == status and delivered.event_seq == 14
    assert delivered.stop_evidence == full['stop_evidence']
    assert delivered.protection_reason == full['protection_reason']
    assert {k:v for k,v in compact.items() if k != 'raw_feedback'} == {
        k:v for k,v in full.items() if k != 'raw_feedback'}
    p = compact['raw_feedback']['precision_turn']
    for key in ('controller','motion_deadline','total_deadline','corrections','failure','error_rad'):
        assert p[key] == full['raw_feedback']['precision_turn'][key]
    assert p['trace_transport_omitted_count'] == 64 and 'trace' not in p
    assert p['integral']['samples_transport_omitted_count'] == 512 and 'samples' not in p['integral']
    assert p['integral']['sample_records_dropped'] == 17 and p['integral']['stage_records_dropped'] == 3
    assert p['integral']['nonzero_seconds'] == 7.405214550000267
    assert 'events' not in loaded['core'] and loaded['core']['event_history_transport_omitted_count'] == 3
    assert full == original and len(summary['events']) == 3
    archived = json.loads((tmp_path/'bridge_events.jsonl').read_text())['event']
    reference = compact['raw_feedback']['diagnostics_archive']
    assert archived == full
    assert reference['sha256'] == hashlib.sha256(encoded(archived)).hexdigest()
    assert (reference['run_id'],reference['session_id'],reference['request_id'],reference['event_seq']) == ('run','session','request',14)
    assert 'bounded' in reference['scope']  # Does not claim the controller retained every sample.
    profile = integral_action_interaction_profile()
    host = ClockExchange('run','session',clock=lambda:10.,profile=profile)
    remote = RemoteExchange('run','session',clock=lambda:400.,profile=profile)
    query = host.request('GREET','request')
    response = remote.observe(query, Feedback(valid=True, age_s=0., source='fixture', standing=True), [compact])
    publish_observation(tmp_path, response, log)
    response = read_json(tmp_path/'observation_response.json')
    host.accept(response)
    assert ExecutionEvent.from_dict(response['events'][0]).status == status
    assert response['events'][0] == compact
    final = _archive_summary(dict(summary, events=15), REG, log)
    atomic(tmp_path/'bridge_summary.json', final)
    lines = [json.loads(line) for line in (tmp_path/'bridge_events.jsonl').read_text().splitlines()]
    assert lines[-1]['kind'] == 'bridge_summary'
    assert len(lines[-1]['summary']['precision_turn']['integral']['samples']) == 512
    assert len(lines[-1]['summary']['precision_turn']['trace']) == 64
    for name in ('execution_latest.json','observation_response.json','bridge_summary.json'):
        assert (tmp_path/name).stat().st_size < MAX_PACKET_BYTES


def test_real_executor_terminal_assignment_survives_maximum_diagnostic_delivery(tmp_path):
    r = rig(integral_action_skill_profile, integral_action_interaction_profile)
    assert submit(r,'TURN_RIGHT')
    r.a = -math.pi/2; step(r)
    r.a = -math.radians(79.)
    r.core.precision_corrections = 1
    r.core.integral_stopped_error = .3
    r.core.integral_samples = diagnostics()['integral']['samples']
    r.core.precision_trace = diagnostics()['trace']
    archived_events = []
    def record(raw):
        event = wrap_execution(REG, raw, len(archived_events), [], {}, None).to_dict()
        compact = _archive_execution(event, recorder(tmp_path))
        archived_events.append(compact)
        atomic(tmp_path/'execution_latest.json', _execution_mailbox(REG, compact, clean(r.core.summary())))
    r.core.record = record
    for _ in range(80):
        step(r)
        if r.core.terminal:
            break
    assert r.core.terminal and r.core.status == 'FAILED'
    assert r.core.reason == 'integral_turn_corrections_exhausted'
    assert not r.core.motion.failures and r.commands[-1] == (0.,0.,0.)
    terminal = ExecutionEvent.from_dict(read_json(tmp_path/'execution_latest.json')['event'])
    assert terminal.status == 'FAILED' and terminal.stop_evidence['stop_confirmed'] is True
    assert terminal.stop_evidence['angle_rad'] == pytest.approx(-math.radians(79.))
    full = json.loads((tmp_path/'bridge_events.jsonl').read_text())['event']
    assert len(full['raw_feedback']['precision_turn']['integral']['samples']) == 512
    assert len(full['raw_feedback']['precision_turn']['trace']) == 64


def test_failed_archive_never_returns_a_reference_or_writes_a_mailbox(tmp_path):
    def failed(*args, **kwargs):
        raise OSError('archive unavailable')
    with pytest.raises(OSError, match='archive unavailable'):
        event = _archive_execution(full_event('FAILED'), failed)
        atomic(tmp_path/'execution_latest.json', _execution_mailbox(REG,event,{}))
    assert not (tmp_path/'execution_latest.json').exists()


@pytest.mark.parametrize('partial', [b'{"closed_request_ack":', b'{"label":"\xe4'])
def test_incomplete_mailbox_reopens_new_version_without_caching_old_packet(tmp_path, monkeypatch, partial):
    from malecns_lab.learning import x2_behavior_session as transport
    path = tmp_path/'observation_request.json'
    path.write_bytes(partial)
    delays = []
    def complete(delay):
        delays.append(delay)
        atomic(path, {'closed_request_ack': None, 'sequence': 2})
    monkeypatch.setattr(transport.time, 'sleep', complete)
    assert read_json(path) == {'closed_request_ack': None, 'sequence': 2}
    assert delays == [.002]


def test_persistent_mailbox_corruption_is_bounded_and_names_the_file(tmp_path, monkeypatch):
    from malecns_lab.learning import x2_behavior_session as transport
    path = tmp_path/'decision.json'; path.write_bytes(b'{"decision":')
    delays = []
    monkeypatch.setattr(transport.time, 'sleep', delays.append)
    with pytest.raises(ValueError, match=r'Invalid transport JSON: decision.json; bytes=12'):
        read_json(path)
    assert delays == [.002, .002]


@pytest.mark.parametrize('kind', ['symlink', 'oversized', 'not_object'])
def test_invalid_mailbox_boundaries_fail_without_retry(tmp_path, monkeypatch, kind):
    from malecns_lab.learning import x2_behavior_session as transport
    path = tmp_path/'decision.json'
    if kind == 'symlink':
        target = tmp_path/'other.json'; target.write_text('{}'); path.symlink_to(target)
    else:
        path.write_bytes(b' '*(MAX_PACKET_BYTES+1) if kind == 'oversized' else b'[]')
    delays = []; monkeypatch.setattr(transport.time, 'sleep', delays.append)
    with pytest.raises(ValueError):
        read_json(path)
    assert not delays


def test_failed_sync_preserves_previous_packet_and_removes_unpublished_temp(tmp_path, monkeypatch):
    from malecns_lab.learning import x2_behavior_session as transport
    path = tmp_path/'task.json'; atomic(path, {'sequence': 1})
    monkeypatch.setattr(transport.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('sync failed')))
    with pytest.raises(OSError, match='sync failed'):
        atomic(path, {'sequence': 2})
    assert read_json(path) == {'sequence': 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ['task.json']
