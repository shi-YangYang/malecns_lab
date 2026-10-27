"""Bounded mailbox refresh fixtures; never start a brain, robot, or VM."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import Decision, DecisionGate, ExecutionEvent, Feedback, Observation
from malecns_lab.control.x2_interaction import interaction_profile, measured_action_interaction_profile
from malecns_lab.experiments.x2_behavior_bridge import RemoteExchange
from malecns_lab.learning import x2_behavior_session as transport


class Clock:
    value = 1000.
    def __call__(self): return self.value
    def sleep(self, seconds): self.value += seconds


def rig(tmp_path, monkeypatch, delays, *, age=.02, task='SHOW_FORWARD', corrupt=None, profile=None):
    clock = Clock()
    profile = measured_action_interaction_profile() if profile is None else profile
    host = transport.ClockExchange('run', 'session', clock, profile=profile)
    remote = RemoteExchange('run', 'session', lambda: clock()-900., profile=profile, continuous=True)
    session = object.__new__(transport.NeuralSession)
    session.policy = SimpleNamespace(set_runtime_context=lambda *args: None)
    session.gate = DecisionGate('run', 'session', 'policy', 'checkpoint', 'input')
    previous = Observation('run', 'session', 0, 'previous', 'SHOW_FORWARD', clock() - 1.,
        capabilities=host.capabilities)
    chosen = Decision('run', 'session', 1, 'previous', 0, 'before', 'after',
        (0., 1., 0., 0., 0.), (1., 0., 0.), (0.,), 'ADVANCE', 'START', .5,
        'policy', 'checkpoint', 'input', 600, 1100)
    session.gate.observe(previous)
    assert session.gate.admit(chosen, clock() - 1.).accepted
    def event(sequence, status, when):
        return ExecutionEvent('run', 'session', sequence, 1, 'skill-1', 'previous', 'offline_fixture',
            'fixture-skills', status, 'ADVANCE', when,
            stop_evidence={'stop_confirmed': status == 'COMPLETED'})
    session.gate.execution_event(event(6, 'ACCEPTED', clock() - .9))
    session.gate.execution_event(event(7, 'STARTED', clock() - .8))
    session.runtime_contexts = {}
    session.pending_neural_events = []
    session.consumed_event_sequence = 7
    writes, packets, records = [], [], []
    real_atomic = transport.atomic
    def send(path, query):
        assert path.name == 'observation_request.json'
        writes.append(deepcopy(query))
        number = len(writes)
        yaw = .2 + number * .01
        feedback = Feedback(valid=True, age_s=age, source='fixture', standing=True, yaw_rad=yaw)
        odom = dict(sequence=number, remote_received_monotonic=remote.clock()-age,
            message={'pose': {'pose': {'orientation': dict(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))}}})
        context = dict(protection_state='READY', event_sequence=8,
            facts=dict(failures=[], core_status='COMPLETED', source_registered=True, armed=True,
                stable_preparation_completed=True, fresh_body=True, ownership_supported=True))
        packet = remote.observe(query, feedback, [event(8, 'COMPLETED', remote.clock()).to_dict()],
            odom=odom, runtime_context=context)
        if corrupt is not None: corrupt(packet, number)
        packets.append(deepcopy(packet))
        real_atomic(tmp_path/'observation_response.json', packet)
        clock.value += delays[min(number-1, len(delays)-1)]
    monkeypatch.setattr(transport, 'atomic', send)
    kwargs = dict(after_event_seq=7, last_consumed_request_id='previous', request_sequence=2,
        closed_request_ack={'request_sequence': 1, 'request_id': 'previous'}, guard=lambda: None,
        record=lambda kind, **data: records.append(dict(kind=kind, **data)), sleep=clock.sleep,
        validate=session.validate_snapshot)
    def fetch(**changes):
        return transport.fresh_observation(host, tmp_path, {'task': task, 'request_id': 'request'}, **(kwargs | changes))
    fetch.session = session
    return host, clock, writes, packets, records, fetch


def test_observation_transient_transit_expiry_requests_new_nonce_without_consuming_old_events(tmp_path, monkeypatch):
    host, clock, writes, packets, records, fetch = rig(tmp_path, monkeypatch, [1.34, .03])
    observation, clocks, packet = fetch()
    assert len(writes) == 2 and writes[0]['nonce'] != writes[1]['nonce']
    for query in writes:
        assert query['after_event_seq'] == 7 and query['last_consumed_request_id'] == 'previous'
        assert query['request_sequence'] == 2 and query['closed_request_ack']['request_sequence'] == 1
    assert len(packet['events']) == 1
    assert ExecutionEvent.from_dict(packet['events'][0]).event_seq == 8
    # Neither the discarded response nor validation of the accepted one has
    # consumed the event; only the caller's real snapshot applies it later.
    assert fetch.session.gate.last_event_seq == fetch.session.consumed_event_sequence == 7
    assert fetch.session.gate.active['status'] == 'STARTED'
    assert fetch.session.runtime_contexts == {} and fetch.session.pending_neural_events == []
    assert observation.observation_seq == 2 and observation.request_changed
    assert observation.feedback.age_s == pytest.approx(.05)
    assert clocks['round_trip_upper_s'] == pytest.approx(.03)
    assert records[0]['round_trip_upper_s'] == pytest.approx(1.34) and records[0]['retry']
    assert host.pending is None and host.last_remote_sequence == 2
    assert clock() < 1002.
    host.request('SHOW_FORWARD', 'request')
    with pytest.raises(ValueError, match='nonce'): host.accept(packets[0])


def test_observation_refresh_retains_original_turn_goal(tmp_path, monkeypatch):
    host, _, writes, packets, _, fetch = rig(tmp_path, monkeypatch, [.8, .02], task='FACE_LEFT')
    observation, clocks, _ = fetch()
    first = packets[0]['heading_target']
    assert clocks['heading_target']['origin_odom'] == first['origin_odom']
    assert clocks['heading_target']['established_remote_sequence'] == writes[0]['sequence']
    assert host.heading_targets['request'] == transport.heading_goal(first)
    assert observation.feedback.yaw_rad == pytest.approx(.22)


def test_observation_repeated_expiry_keeps_original_total_deadline(tmp_path, monkeypatch):
    host, clock, writes, _, _, fetch = rig(tmp_path, monkeypatch, [1.1, 1.1])
    with pytest.raises(TimeoutError, match='timed out'): fetch()
    assert len(writes) == 2  # The second exchange does not receive a new 2s budget.
    assert clock() == pytest.approx(1002.2) and host.last_remote_sequence == 1


def test_observation_retry_count_is_bounded_even_when_time_remains(tmp_path, monkeypatch):
    _, clock, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.2], age=.6)
    with pytest.raises(TimeoutError, match='连续过期'): fetch()
    assert len(writes) == 3 and clock() == pytest.approx(1000.6)
    assert [r['retry'] for r in records] == [True, True, False]


@pytest.mark.parametrize('field,value,match', [
    ('run_id', 'foreign', 'identity'), ('remote_sequence', 0, 'replay'),
    ('capabilities', ['NOT_QUALIFIED'], 'capability'), ('request_changed', False, 'edge'),
])
def test_observation_invalid_response_is_not_retried_even_when_transit_expired(tmp_path, monkeypatch, field, value, match):
    _, _, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.8], corrupt=lambda p, _: p.update({field: value}))
    with pytest.raises(ValueError, match=match): fetch()
    assert len(writes) == 1 and not records


@pytest.mark.parametrize('capabilities', [('ADVANCE',), ('HOLD', 'HOLD'), ()])
def test_observation_no_wave_capabilities_must_be_unique_and_include_hold_before_retry(tmp_path, monkeypatch, capabilities):
    host, _, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.8], profile=interaction_profile(),
        corrupt=lambda packet, _: packet.update(capabilities=list(capabilities)))
    with pytest.raises(ValueError, match='unique and include HOLD'):
        fetch()
    assert len(writes) == 1 and not records
    assert host.pending is not None and host.last_remote_sequence == 0 and host.heading_targets == {}


@pytest.mark.parametrize('damage,match', [
    ('foreign_run', 'Foreign'), ('foreign_session', 'Foreign'), ('malformed_event', 'schema/kind'),
    ('unattributed_event', 'Unattributed'), ('duplicate_event', 'Duplicate/reordered'),
    ('missing_context', 'runtime'), ('false_protection', 'differs'), ('invalid_facts', 'facts'),
    ('missing_facts', 'facts'), ('invalid_watermark', 'runtime'), ('watermark_mismatch', 'watermark'),
])
def test_observation_invalid_events_and_runtime_context_do_not_enter_age_retry(tmp_path, monkeypatch, damage, match):
    def corrupt(packet, _):
        if damage == 'foreign_run': packet['events'][0]['run_id'] = 'other-run'
        elif damage == 'foreign_session': packet['events'][0]['session_id'] = 'other-session'
        elif damage == 'malformed_event': packet['events'] = [{'event_seq': 8}]
        elif damage == 'unattributed_event': packet['events'][0]['decision_seq'] = 99
        elif damage == 'duplicate_event': packet['events'].append(deepcopy(packet['events'][0]))
        elif damage == 'missing_context': del packet['runtime_context']
        elif damage == 'false_protection': packet['runtime_context']['protection_state'] = 'LATCHED'
        elif damage == 'invalid_facts': packet['runtime_context']['facts']['armed'] = 'yes'
        elif damage == 'missing_facts': del packet['runtime_context']['facts']['failures']
        elif damage == 'invalid_watermark': packet['runtime_context']['event_sequence'] = True
        else: packet['runtime_context']['event_sequence'] = 9
    host, _, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.8], corrupt=corrupt)
    gate_before = deepcopy(vars(fetch.session.gate))
    with pytest.raises(ValueError, match=match): fetch()
    assert len(writes) == 1 and not records
    assert vars(fetch.session.gate) == gate_before
    assert fetch.session.runtime_contexts == {} and fetch.session.pending_neural_events == []
    assert fetch.session.consumed_event_sequence == 7
    assert host.pending is not None and host.last_remote_sequence == 0 and host.heading_targets == {}


def test_observation_validation_time_counts_toward_age_and_shared_deadline(tmp_path, monkeypatch):
    host, clock, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.01])
    def slow_validate(observation, packet):
        fetch.session.validate_snapshot(observation, packet)
        clock.value += .8 if len(writes) == 1 else .01
    observation, clocks, _ = fetch(validate=slow_validate)
    assert len(writes) == 2 and records[0]['round_trip_upper_s'] == pytest.approx(.81)
    assert observation.feedback.age_s == pytest.approx(.04)
    assert clocks['round_trip_upper_s'] == pytest.approx(.02)
    def expired_validate(observation, packet):
        fetch.session.validate_snapshot(observation, packet)
        clock.value += 2.
    with pytest.raises(TimeoutError, match='timed out'):
        fetch(validate=expired_validate)
    assert host.pending is not None and host.last_remote_sequence == 2


def test_observation_stale_sensor_and_emergency_are_not_hidden_by_refresh(tmp_path, monkeypatch):
    def stale(packet, _): packet.update(source_age_s=.76); packet['feedback']['age_s'] = .76
    _, _, writes, _, records, fetch = rig(tmp_path, monkeypatch, [.01], corrupt=stale)
    with pytest.raises(ValueError, match='Remote original'): fetch()
    assert len(writes) == 1 and not records
    def stop(): raise InterruptedError('emergency')
    with pytest.raises(InterruptedError): fetch(guard=stop)
    assert len(writes) == 1


def test_observation_read_delay_counts_against_same_deadline(tmp_path, monkeypatch):
    _, clock, writes, _, _, fetch = rig(tmp_path, monkeypatch, [.01])
    read = transport.read_json
    def slow_read(path):
        packet = read(path); clock.value += 2.; return packet
    monkeypatch.setattr(transport, 'read_json', slow_read)
    with pytest.raises(TimeoutError, match='timed out'): fetch()
    assert len(writes) == 1
