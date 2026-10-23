"""Cross-clock transport and one persistent, neural-only behavior session.

Import is stdlib-only. Clock exchange never subtracts macOS from Linux time.
The coordinator exclusively owns locks, resource supervision and simulation.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import uuid

SCHEMA = 'malecns_lab.x2_behavior_transport.v1'
TASKS = ('GREET', 'FACE_LEFT', 'FACE_RIGHT', 'SHOW_FORWARD', 'END_INTERACTION')
CAPABILITIES = ('HOLD', 'ADVANCE', 'TURN_LEFT', 'TURN_RIGHT')
MAX_PACKET_BYTES = 256*1024
ROUND_TRIP_SECONDS = 2.
DECISION_SECONDS = 30.
SOURCE_AGE_SECONDS = .75
TASK_REQUEST_SECONDS = 120.
HEADING_OFFSETS = {'FACE_LEFT': .45, 'FACE_RIGHT': -.45}
HEADING_SOURCE = 'requested_relative_heading_from_actual_odom'


def wrap_heading(value):
    return math.atan2(math.sin(value), math.cos(value))


def heading_goal(target):
    """The immutable request goal, excluding the current measured body state."""
    return {k:v for k,v in target.items() if k not in
            ('current_odom', 'current_yaw_rad', 'heading_error_rad')}


def validate_heading_target(target, feedback, query, remote_sequence, remote_now):
    """Audit a geometric request against its original and current raw odometry."""
    if query['task'] not in HEADING_OFFSETS:
        require(target is None and feedback.heading_error_rad == 0., 'Unexpected heading target')
        return
    require(isinstance(target, dict) and target.get('source') == HEADING_SOURCE
            and all(target.get(k) == query[k] for k in ('run_id','session_id','request_id','task')),
            'Heading target attribution differs')
    def measured_yaw(raw, at, max_age):
        require(isinstance(raw, dict) and type(raw.get('sequence')) is int and raw['sequence'] > 0,
                'Missing original odometry sequence')
        received = raw.get('remote_received_monotonic')
        require(type(received) in (int,float) and math.isfinite(received)
                and 0 <= at-received <= max_age+1e-8, 'Original heading odometry expired')
        try:
            q = raw['message']['pose']['pose']['orientation']
            x,y,z,w = (q[k] for k in ('x','y','z','w'))
        except (KeyError,TypeError):
            raise ValueError('Missing original heading orientation') from None
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (x,y,z,w))
                and .9 <= math.sqrt(x*x+y*y+z*z+w*w) <= 1.1, 'Invalid original heading orientation')
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    established = target.get('established_remote_monotonic')
    sequence = target.get('established_remote_sequence')
    require(type(established) in (int,float) and math.isfinite(established) and established <= remote_now
            and type(sequence) is int and 0 < sequence <= remote_sequence, 'Invalid heading establishment')
    origin = measured_yaw(target.get('origin_odom'), established, SOURCE_AGE_SECONDS)
    current = measured_yaw(target.get('current_odom'), remote_now, feedback.age_s)
    require(target['current_odom']['sequence'] >= target['origin_odom']['sequence']
            and target['current_odom']['remote_received_monotonic'] >= target['origin_odom']['remote_received_monotonic'],
            'Heading odometry precedes request origin')
    expected = dict(relative_offset_rad=HEADING_OFFSETS[query['task']], origin_yaw_rad=origin,
        target_yaw_rad=wrap_heading(origin+HEADING_OFFSETS[query['task']]), current_yaw_rad=current)
    expected['heading_error_rad'] = wrap_heading(expected['target_yaw_rad']-current)
    require(all(type(target.get(k)) in (int,float) and math.isfinite(target[k])
                and abs(target[k]-v) <= 1e-8 for k,v in expected.items())
            and abs(feedback.yaw_rad-current) <= 1e-8
            and abs(feedback.heading_error_rad-expected['heading_error_rad']) <= 1e-8,
            'Heading target differs from original odometry')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atomic(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    require(len(content.encode()) <= MAX_PACKET_BYTES, 'Transport packet exceeds budget')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_PACKET_BYTES,
            'Missing, linked or oversized transport packet')
    value = json.loads(path.read_text())
    require(isinstance(value, dict), 'Transport object required')
    return value


def append(stream, kind, **fields):
    stream.write(json.dumps(dict(kind=kind, host_monotonic=time.monotonic(), **fields),
                            ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


class ClockExchange:
    """One-use nonce bounds transit by host RTT; source age remains remote."""
    def __init__(self, run_id, session_id, clock=time.monotonic):
        self.run_id, self.session_id, self.clock = run_id, session_id, clock
        self.pending = None
        self.sequence = 0
        self.last_remote_sequence = 0
        self.heading_targets = {}

    def request(self, task, request_id, after_event_seq=-1, *, last_consumed_request_id=None):
        require(task in TASKS and isinstance(request_id, str) and request_id, 'Invalid task request')
        require(last_consumed_request_id is None or isinstance(last_consumed_request_id, str)
                and last_consumed_request_id, 'Invalid consumed request')
        require(self.pending is None, 'Observation request already pending')
        self.sequence += 1
        self.pending = dict(schema=SCHEMA, kind='observe', run_id=self.run_id, session_id=self.session_id,
            nonce=uuid.uuid4().hex, sequence=self.sequence, request_id=request_id, task=task,
            after_event_seq=after_event_seq, last_consumed_request_id=last_consumed_request_id,
            host_sent_monotonic=self.clock())
        return dict(self.pending)

    def accept(self, packet):
        p, now = self.pending, self.clock()
        require(p is not None, 'Observation replay without pending nonce')
        require(packet.get('schema') == SCHEMA and packet.get('kind') == 'observation'
                and all(packet.get(k) == p[k] for k in ('run_id','session_id','nonce','sequence','request_id','task')),
                'Observation identity/nonce differs')
        elapsed = now-p['host_sent_monotonic']
        require(0 <= elapsed <= ROUND_TRIP_SECONDS, 'Observation round trip expired')
        require(type(packet.get('remote_sequence')) is int and packet['remote_sequence'] > self.last_remote_sequence,
                'Observation replay/reordering')
        age = packet.get('source_age_s')
        require(type(age) in (float, int) and math.isfinite(age) and 0 <= age <= SOURCE_AGE_SECONDS,
                'Remote original feedback expired')
        require(age+elapsed <= SOURCE_AGE_SECONDS, 'Feedback age plus transit bound expired')
        require(type(packet.get('request_changed')) is bool
                and packet['request_changed'] == (p['request_id'] != p['last_consumed_request_id']),
                'Request edge differs from last neural consumption')
        from ..control.x2_behavior import Feedback, Observation
        feedback = Feedback(**packet['feedback'])
        require(feedback.valid and abs(feedback.age_s-age) <= 1e-8, 'Source validity/age differs')
        require('heading_target' in packet, 'Missing heading target metadata')
        target = packet['heading_target']
        validate_heading_target(target, feedback, p, packet['remote_sequence'], packet['remote_issued_monotonic'])
        goal = heading_goal(target) if target is not None else None
        require(p['request_id'] not in self.heading_targets or self.heading_targets[p['request_id']] == goal,
                'Heading target moved within request')
        self.heading_targets[p['request_id']] = goal
        feedback = replace(feedback, age_s=age+elapsed)
        self.pending = None
        self.last_remote_sequence = packet['remote_sequence']
        observation = Observation(self.run_id, self.session_id, p['sequence'], p['request_id'], p['task'],
            now, feedback=feedback, capabilities=tuple(packet['capabilities']), request_changed=packet['request_changed'])
        require(set(str(x) for x in observation.capabilities) <= set(CAPABILITIES), 'Unqualified capability')
        evidence = dict(nonce=p['nonce'], host_sent_monotonic=p['host_sent_monotonic'],
            host_received_monotonic=now, round_trip_upper_s=elapsed, source_age_s=age,
            remote_issued_monotonic=packet['remote_issued_monotonic'], remote_sequence=packet['remote_sequence'],
            heading_target=target, response_record=deepcopy(packet.get('response_record')),
            feedback_sources=deepcopy(packet.get('feedback_sources')),
            sampled_remote_monotonic=packet.get('sampled_remote_monotonic'))
        return observation, evidence


def host_event(value, run_id, session_id, received):
    """Keep the original remote timestamp in evidence; gate sees host time only."""
    from ..control.x2_behavior import ExecutionEvent
    event = ExecutionEvent.from_dict(value)
    require((event.run_id,event.session_id) == (run_id,session_id), 'Foreign execution event')
    raw = dict(event.raw_feedback, remote_received_monotonic=event.received_monotonic,
               remote_clock_domain='linux_bridge_monotonic', host_received_monotonic=received)
    return replace(event, received_monotonic=received, raw_feedback=raw)


class NeuralSession:
    def __init__(self, policy, checkpoint_id, run_id, session_id, telemetry, record):
        from ..control.x2_behavior import DecisionGate
        self.policy, self.checkpoint_id, self.telemetry, self.record = policy, checkpoint_id, telemetry, record
        self.gate = DecisionGate(run_id, session_id, policy.policy_id, checkpoint_id, policy.encoder.input_map_id,
                                 max_age_s=DECISION_SECONDS)
        self.state = policy.initial_state(session_id, telemetry=telemetry)
        self.decision_seq = 0
        self.last_observation = None
        self.last_consumed_request_id = None
        self.pending_rejections = deque()
        self.execution_results = {}
        self.heading_targets = {}
        self.record('session_reset', reason='explicit_new_session', state_id=policy.state_identity(self.state)['state_id'])

    def observe(self, observation, *, usage='gate_refresh'):
        self.gate.observe(observation)
        self.last_observation = observation
        self.record('observation', observation=observation.to_dict(), usage=usage)
        if usage == 'gate_refresh':
            self.record_gate_refresh(observation, purpose='newer_observation', gate_observation_updated=True)

    def record_gate_refresh(self, observation, *, purpose, gate_observation_updated=False):
        self.record('gate_refresh', observation=observation.to_dict(),
            observation_seq=observation.observation_seq, request_id=observation.request_id,
            decision_seq=self.decision_seq, purpose=purpose, gate_observation_updated=gate_observation_updated)

    def heading_evidence(self, observation, target):
        self.heading_targets[observation.observation_seq] = target
        while len(self.heading_targets) > 32:
            del self.heading_targets[next(iter(self.heading_targets))]
        self.record('heading_target', observation_seq=observation.observation_seq,
            request_id=observation.request_id, target=target)

    def execution_event(self, event):
        """Gate-only polling records facts, but never consumes neural feedback."""
        from ..control.x2_behavior import ExecutionStatus
        self.gate.execution_event(event)
        self.execution_results[event.decision_seq] = event
        if event.finish_decision_seq is not None:
            self.execution_results[event.finish_decision_seq] = event
        if event.status == ExecutionStatus.REJECTED:
            self.pending_rejections.append(dict(source='official_sim_execution_rejection',
                request_id=event.request_id, execution_id=event.execution_id,
                decision_seq=event.decision_seq, behavior=event.behavior, event=event.to_dict()))

    def confirmed_finish(self, decision):
        from ..control.x2_behavior import ExecutionStatus
        event = self.execution_results.get(decision.decision_seq)
        return (event is not None and event.status in (ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED)
                and event.protection_reason is None and event.stop_evidence.get('stop_confirmed') is True)

    def neural_observation(self, observation):
        """Project one real refusal into the fixed feedback slot without editing active execution."""
        from ..control.x2_behavior import ExecutionStatus
        observation = replace(observation,
            request_changed=observation.request_id != self.last_consumed_request_id)
        if self.pending_rejections:
            rejected = self.pending_rejections[0]
            current = observation.feedback
            observation = replace(observation, feedback=replace(current,
                source=rejected['source'], status=ExecutionStatus.REJECTED,
                behavior=rejected['behavior'], request_id=rejected['request_id'],
                execution_id=rejected['execution_id']))
            self.record('rejection_neural_input', observation_seq=observation.observation_seq,
                rejected=rejected, current_feedback=current.__dict__,
                current_heading_target=self.heading_targets.get(observation.observation_seq),
                active_execution_id=self.gate.active['execution_id'] if self.gate.active else None)
        return observation

    def infer(self, observation, refresh=None):
        import torch
        observation = self.neural_observation(observation)
        pending_rejection = bool(self.pending_rejections)
        self.observe(observation, usage='neural_input')
        self.decision_seq += 1
        before = self.policy.state_identity(self.state)['state_id']
        self.telemetry.begin_decision(observation, self.decision_seq)
        started = time.monotonic()
        self.record('neural_input', status='attempted', observation=observation.to_dict(),
            observation_seq=observation.observation_seq, decision_seq=self.decision_seq,
            request_id=observation.request_id)
        with torch.no_grad():
            output = self.policy(observation, self.state, checkpoint_blocks=False, telemetry=self.telemetry)
        self.last_consumed_request_id = observation.request_id
        if pending_rejection:
            self.pending_rejections.popleft()
        # Advance even when a newer task invalidates the completed decision.
        self.state = output.state.detach()
        decision = self.policy.decision(observation, output, checkpoint_id=self.checkpoint_id,
                                        state_before_id=before, decision_seq=self.decision_seq)
        self.record('raw_decision', decision=decision.to_dict(), inference_seconds=time.monotonic()-started,
                    features_mv=output.features_mv.detach().cpu().tolist())
        if refresh is not None:
            newest = refresh()
            if newest is not None:
                self.observe(newest)
        admission = self.gate.admit(decision, time.monotonic())
        self.record('admission', decision_seq=decision.decision_seq, accepted=admission.accepted,
                    reason=admission.reason, active_execution_id=admission.active_execution_id)
        if not admission.accepted:
            self.pending_rejections.append(dict(source='host_decision_gate_rejection',
                request_id=decision.request_id, execution_id='host-rejected-'+str(decision.decision_seq),
                decision_seq=decision.decision_seq, behavior=decision.behavior, reason=admission.reason))
        return decision, admission


def load_candidate_policy(root, metadata, checkpoint_path):
    """Reconstruct the registered estimator, then perform unchanged strict restore."""
    from . import x2_inference as frozen
    from .x2_behavior_policy import X2BehaviorPolicy, backward_identity
    from .x2_behavior_training import restore_checkpoint
    identity = metadata['policy']
    backward = identity.get('backward')
    require('backward' not in identity or isinstance(backward, dict), 'Invalid candidate backward identity')
    gain = 1.0 if 'backward' not in identity else backward.get('spike_backward_gain')
    expected_backward = backward_identity(gain)
    require(backward is None or backward == expected_backward, 'Candidate backward identity differs')
    anchors = frozen.read_anchors(root)
    source = anchors['run']['policy']
    values = frozen.load_parameters(root/frozen.CHECKPOINT/'arrays.npz', anchors['checkpoint'], source)
    init = identity['initialization']
    policy = X2BehaviorPolicy.from_reference(root/frozen.REFERENCE, source, theta=values['theta'],
        initialization=init['theta'], seed=init['head_seed'], spike_backward_gain=gain)
    restored = restore_checkpoint(checkpoint_path, policy)
    require(restored['checkpoint_id'] == metadata['checkpoint_id'], 'Candidate changed')
    return policy


def run_worker(directory):
    """Only invoked by the resource-owning desktop parent with a fresh claim."""
    import fcntl
    directory = Path(directory).resolve()
    reg = read_json(directory/'behavior_registration.json')
    require(os.getppid() == reg['parent_pid'] and str(directory) == reg['output'], 'Supervising parent required')
    root = Path(reg['root'])
    with (root/'data/.continuous-training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('Parent must hold complete-brain lock')
    (directory/'behavior-worker.claim').open('x').write(str(os.getpid()))
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    import hashlib
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from .x2_behavior_telemetry import BehaviorTelemetry
    from .mps_memory import MPSMemorySampler
    sampler = MPSMemorySampler(directory/'behavior_mps_memory.csv', reg['rss_limit_bytes'])
    telemetry = session = None
    outcome = dict(status='failed', reason='initializing', decisions=0)
    started = time.monotonic()
    with (directory/'behavior_events.jsonl').open('x') as evidence:
        def record(kind, **fields):
            append(evidence, kind, **fields)
        def phase(name, **fields):
            sampler.sample(name)
            atomic(directory/'behavior_phase.json', dict(phase=name, **fields))
        def guard():
            require(os.getppid() == reg['parent_pid'], 'Parent lost')
            require(time.monotonic()-started < reg['worker_wall_seconds'], 'Worker wall budget expired')
            if (directory/'emergency.json').exists():
                raise InterruptedError('independent_emergency_stop')
        try:
            for name, expected in reg['source_sha256'].items():
                require(hashlib.sha256((root/'src/malecns_lab'/name).read_bytes()).hexdigest() == expected
                        and hashlib.sha256((directory/'sources/malecns_lab'/name).read_bytes()).hexdigest() == expected,
                        'Registered source changed')
            phase('load_complete_brain')
            meta = reg['checkpoint_metadata']
            policy = load_candidate_policy(root, meta, reg['checkpoint_path'])
            policy.eval()
            identity = dict(execution_id=reg['session_id'], checkpoint_id=meta['checkpoint_id'],
                            source_files_sha256=reg['source_sha256'])
            telemetry = BehaviorTelemetry(directory, policy, identity, reg['run_id'], reg['session_id'])
            require(telemetry.error is None, 'Telemetry unavailable: '+str(telemetry.error))
            session = NeuralSession(policy, meta['checkpoint_id'], reg['run_id'], reg['session_id'], telemetry, record)
            atomic(directory/'brain_ready.json', dict(policy_id=policy.policy_id, checkpoint_id=meta['checkpoint_id'],
                input_map_id=policy.encoder.input_map_id, run_id=reg['run_id'], session_id=reg['session_id']))
            phase('waiting_task', qualification_only=True)
            exchange = ClockExchange(reg['run_id'],reg['session_id'])
            last_event = -1
            last_task_sequence = 0
            task = None
            finished_request = None
            per_request_count = 0

            def current_task():
                if not (directory/'task.json').exists():
                    return None
                value = read_json(directory/'task.json')
                require(value['run_id'] == reg['run_id'] and value['session_id'] == reg['session_id']
                        and value['task'] in TASKS and type(value['sequence']) is int and value['sequence'] > 0,
                        'Invalid task mailbox')
                return value

            def snapshot(request, *, gate_purpose=None):
                nonlocal last_event
                query = exchange.request(request['task'], request['request_id'], last_event,
                    last_consumed_request_id=session.last_consumed_request_id)
                atomic(directory/'observation_request.json', query)
                end = time.monotonic()+ROUND_TRIP_SECONDS
                while time.monotonic() < end:
                    guard()
                    path = directory/'observation_response.json'
                    if path.exists():
                        packet = read_json(path)
                        if packet.get('nonce') == query['nonce']:
                            observation, clocks = exchange.accept(packet)
                            session.heading_evidence(observation, clocks['heading_target'])
                            for raw in packet['events']:
                                event = host_event(raw, reg['run_id'], reg['session_id'], time.monotonic())
                                session.execution_event(event)
                                last_event = event.event_seq
                                record('execution', event=event.to_dict())
                            record('observation_clocks', observation_seq=observation.observation_seq, **clocks)
                            if gate_purpose is not None:
                                session.record_gate_refresh(observation, purpose=gate_purpose)
                            return observation, clocks
                    time.sleep(.02)
                raise TimeoutError('Remote observation timed out')

            while session.decision_seq < reg['max_decisions']:
                guard()
                newest = current_task()
                if newest is None or newest['request_id'] == finished_request:
                    time.sleep(.05)
                    continue
                if task is None or newest['sequence'] > last_task_sequence:
                    task, last_task_sequence = newest, newest['sequence']
                    per_request_count = 0
                    record('task_consumed', request=task)
                request_age = time.monotonic()-task['host_received_monotonic']
                require(0 <= request_age <= TASK_REQUEST_SECONDS, 'Task request budget expired')
                observation, clocks = snapshot(task)
                # The executor advances independently while the unique brain is computing.
                phase('computing', task=task['task'], request_id=task['request_id'],
                      request_age_s=request_age, decision_seq=session.decision_seq+1)
                def refresh():
                    current = current_task()
                    # Drain real completion events before reserving the next
                    # decision. A snapshot alone does not replace the neural
                    # input or renew its timestamp.
                    replacement, _ = snapshot(current or task, gate_purpose='post_inference_event_refresh')
                    if current is not None and current['sequence'] > last_task_sequence:
                        return replacement
                    return None
                decision, admission = session.infer(observation, refresh)
                guard()
                require(telemetry.error is None, 'Telemetry failed: '+str(telemetry.error))
                per_request_count += 1
                outcome['decisions'] = session.decision_seq
                if admission.accepted:
                    atomic(directory/'decision.json', dict(schema=SCHEMA, kind='decision',
                        run_id=reg['run_id'], session_id=reg['session_id'], nonce=clocks['nonce'],
                        expected_active_execution_id=admission.active_execution_id,
                        decision=decision.to_dict(), host_inference_age_s=time.monotonic()-observation.received_monotonic))
                    phase('executing', task=task['task'], behavior=str(decision.behavior), behavior_phase=str(decision.phase),
                          request_id=task['request_id'], decision_seq=decision.decision_seq)
                    # Poll actual events to close the gate; a new task may interrupt this wait.
                    end = time.monotonic()+15.
                    while session.gate.active is not None and time.monotonic() < end:
                        guard()
                        latest = current_task()
                        if latest and latest['sequence'] > last_task_sequence:
                            break
                        time.sleep(.1)
                        snapshot(task, gate_purpose='execution_event_poll')
                        if session.gate.blocked_reason:
                            raise RuntimeError('Execution protection latched: '+session.gate.blocked_reason)
                    if session.gate.active is not None and time.monotonic() >= end:
                        raise TimeoutError('Execution completion unconfirmed')
                    if (str(decision.behavior) == 'HOLD' and str(decision.phase) == 'FINISH'
                            and session.gate.active is None and session.confirmed_finish(decision)):
                        finished_request = task['request_id']
                        record('request_waiting', request_id=finished_request, reason='brain_hold', task_success=None)
                        phase('waiting_task', reason='脑已选择待命', task=task['task'])
                else:
                    phase('rejected', reason=admission.reason, task=task['task'], behavior=str(decision.behavior))
                if per_request_count >= 8:
                    finished_request = task['request_id']
                    record('request_waiting', request_id=finished_request, reason='decision_budget_reached', task_success=None)
                    phase('waiting_task', reason='本次任务达到决策上限', task=task['task'])
            outcome.update(status='completed', reason='bounded_decision_budget')
        except BaseException as error:
            outcome.update(status='cancelled' if isinstance(error, InterruptedError) else 'failed',
                           reason=type(error).__name__+':'+str(error))
            try:
                record('worker_exit', **outcome)
            except Exception:
                pass
        finally:
            try:
                if session is not None:
                    np.savez_compressed(directory/'final-neural-state.npz', **session.policy.state_arrays(session.state))
                    outcome['final_state'] = session.policy.state_identity(session.state)
            except BaseException as error:
                outcome.update(status='failed', state_save_error=repr(error))
            if telemetry is not None:
                outcome['telemetry'] = telemetry.close(completed=outcome['status'] in ('completed','cancelled'))
            try:
                outcome['mps_memory'] = sampler.close()
            except BaseException as error:
                outcome.update(status='failed', memory_sampler_error=repr(error))
            outcome['wall_seconds'] = time.monotonic()-started
            atomic(directory/'behavior_worker_result.json', outcome)
    return 0 if outcome['status'] in ('completed','cancelled') else 1
