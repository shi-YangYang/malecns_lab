"""Simulation-only behavior transport. No teacher, task planner or brain here."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import time
import uuid

from ..learning.x2_behavior_session import (SCHEMA, TASKS, DECISION_SECONDS,
    SOURCE_AGE_SECONDS, HEADING_OFFSETS, HEADING_SOURCE, heading_goal, wrap_heading,
    validate_heading_target, atomic, read_json, require, validate_session_profile)
from ..control.x2_skill_profile import (candidate_skill_profile, validate_candidate_skill_profile,
    SIM_FIXED_ADVANCE_PROFILE_ID, SIM_QUARTER_TURN_ACTION_PROFILE_ID, FULL_ACTION_PROFILE_IDS)
from ..control.x2_skills import SOURCE
from ..control.x2_interaction import (interaction_profile, validate_interaction_profile,
    relative_heading_goal, relative_heading_error, profile_capabilities)


def _pump_callbacks(spin_once, node, core, *, clock=time.monotonic):
    """Bound callback work; let the existing tick handle an observed v9 crossing."""
    began = clock()
    for _ in range(64):
        if clock()-began >= .008:
            break
        spin_once(node, timeout_sec=.0001)
        command = core.command
        if (core.skill_profile_id == SIM_QUARTER_TURN_ACTION_PROFILE_ID
                and command is not None and command.behavior in ('TURN_LEFT', 'TURN_RIGHT')
                and not core.terminal and not core.cancel_requested and core.stop_at is None
                and core.motion.state in ('CLAIMING', 'FORWARD')
                and core.signed_angle * (1 if command.behavior == 'TURN_LEFT' else -1) >= core.turn_target_rad):
            # No parking/state mutation here. Tick still validates fresh MC/odom
            # and the active intent before it can acknowledge target crossing.
            break


def registered_skill_profile(directory, reg):
    """Bind both process registrations to this exact simulation-only profile."""
    host = read_json(Path(directory)/'behavior_registration.json')
    require(all(host.get(k) == reg.get(k) for k in ('run_id','session_id')), 'Skill profile session differs')
    require(host.get('qualification_only') is True and host.get('simulation_only') is True
            and host.get('wave_enabled') == reg.get('wave_enabled'), 'Host skill profile is not simulation qualification')
    profile = validate_candidate_skill_profile(reg.get('skill_profile'))
    require(validate_candidate_skill_profile(host.get('skill_profile')) == profile,
            'Host and bridge skill profiles differ')
    require(validate_interaction_profile(host.get('interaction_profile'))
            == validate_interaction_profile(reg.get('interaction_profile')),
            'Host and bridge interaction profiles differ')
    require(profile['wave_enabled'] == reg.get('wave_enabled')
            and reg['interaction_profile']['skill_profile_id'] == profile['profile_id']
            and reg['interaction_profile']['wave_enabled'] == profile['wave_enabled'], 'Registered capabilities differ')
    if profile['wave_enabled']:
        require(reg.get('simulation_only') is True, 'Wave is restricted to fixed simulation')
    return profile


def candidate_executor(reg, publish, *, record=None, clock=time.monotonic, preset=None, wave_poses=None):
    from ..control.x2_skills import SkillExecutor
    profile = validate_candidate_skill_profile(reg.get('skill_profile'))
    def disabled_preset(_):
        raise RuntimeError('Wave cancellation unqualified; preset requests disabled')
    enabled = profile_capabilities(reg.get('interaction_profile', interaction_profile()))
    require(('WAVE_RIGHT' in enabled) == profile['wave_enabled'], 'Executor capability profile differs')
    wave = profile['wave_enabled']
    fixed_advance = profile['profile_id'] in (SIM_FIXED_ADVANCE_PROFILE_ID, *FULL_ACTION_PROFILE_IDS)
    require(not fixed_advance or (reg.get('qualification_only') is True and reg.get('simulation_only') is True
            and reg['interaction_profile']['skill_profile_id'] == profile['profile_id']),
            'Fixed ADVANCE requires matching simulation qualification')
    require(not wave or preset is not None and wave_poses is not None and reg.get('simulation_only') is True,
            'Qualified simulation wave transport required')
    return SkillExecutor(publish,preset if wave else disabled_preset,enabled=enabled,wave_probe=False,record=record,clock=clock,
        command_seconds_by_behavior=profile['command_seconds_by_behavior'],skill_profile_id=profile['profile_id'],
        wave_qualification=profile.get('wave_contract'),
        execution_profile=profile if fixed_advance else None,
        simulation_only=reg.get('simulation_only'), qualification_only=reg.get('qualification_only'),
        wave_return_pose=wave_poses['terminal'] if wave else None,
        wave_first_pose=wave_poses['first'] if wave else None,
        wave_return_reference='official_animation_terminal_frame' if wave else None)


def observed_base_stop_basis(core, *, allow_unsent_wave=False):
    """Measured manual-SIM stop rule; never rewrite the owned-motion state."""
    if core.motion.state == 'STOPPED':
        return 'owned_executor_stop_window'
    m, o, stopped_at = core.motion.mc, core.motion.odom, core.motion.stop_requested
    idle = core.command is None and core.status == 'IDLE'
    finished = (core.command is not None and core.terminal and core.status in ('COMPLETED', 'CANCELLED')
        and (core.command.behavior == 'HOLD' or core.command.behavior == 'WAVE_RIGHT' and core.wave_return_confirmed))
    withdrawn = (core.command is not None and core.command.behavior == 'WAVE_RIGHT'
        and core.terminal and core.status == 'CANCELLED' and core.reason == 'wave_withdrawn_before_send'
        and core.wave_dispatch_at is None and core.preset_sent_at is None)
    unsent = (allow_unsent_wave and core.command is not None and core.command.behavior == 'WAVE_RIGHT'
        and core.status == 'ACCEPTED' and core.wave_dispatch_at is None and core.preset_sent_at is None)
    if (not core.motion.base_motion_requested and (idle or finished or withdrawn or unsent) and not core.motion.failures
            and m is not None and o is not None and m.source == '' and stopped_at is not None
            and min(m.received, o.received)-stopped_at >= 1. and core.standing_ready()):
        return 'official_unclaimed_stationary_standing_after_stop'
    return None


def cleanup_base_stop_basis(core):
    """Final cleanup evidence only; a recovered body does not clear protection."""
    ordinary = observed_base_stop_basis(core)
    if ordinary is not None:
        return ordinary
    from ..control.x2_skills import RIGHT_ARM
    now = core.clock()
    m, o, stopped_at = core.motion.mc, core.motion.odom, core.motion.stop_requested
    if (core.wave_qualified and core.command is not None and core.command.behavior == 'WAVE_RIGHT'
            and core.status in ('UNKNOWN', 'FAILED') and core.terminal and core.wave_return_confirmed
            and core.preset_sent_at is not None and not core.motion.base_motion_requested
            and core.motion.target == (0., 0., 0.) and core.motion.state == 'STOPPING'
            and m is not None and o is not None and m.source == '' and stopped_at is not None
            and min(m.received, o.received)-stopped_at >= 1.
            and core._wave_feedback_ready(now) and core.player[1] == 0
            and core.query_state == 1 and 0 <= now-core.query_at <= .5
            and core.wave_return_pose is not None
            and max(abs(core.joints[k][0]-core.wave_return_pose[k]) for k in RIGHT_ARM) <= .15
            and max(abs(core.joints[k][1]) for k in RIGHT_ARM) < .08
            and core.distance <= .02 and abs(core.signed_angle) <= .03 and core.standing_ready()):
        return 'cleanup_protected_wave_returned_unclaimed_stationary_standing'
    return None


class WaveStandPreparation:
    """One actual SD for an already brain-selected WAVE; no task dispatcher."""
    def __init__(self, client, request_factory, serialize, record, contract, *, clock=time.monotonic):
        self.client, self.request_factory, self.serialize, self.record = client, request_factory, serialize, record
        self.contract, self.clock = contract, clock
        self.future = self.request = self.started = self.responded = None

    def start(self, core):
        require(core.command is not None and core.command.behavior == 'WAVE_RIGHT'
                and core.status == 'ACCEPTED' and core.wave_dispatch_at is None
                and observed_base_stop_basis(core, allow_unsent_wave=True) and core.standing_ready()
                and core.player is not None and core.player[1] == 0
                and core.query_state == 1 and self.clock()-core.query_at <= .5,
                'Wave SD requires admitted WAVE, stopped body and idle preset player')
        require(self.request is None, 'Wave SD already active')
        self.request = dict(decision_seq=core.command.decision_seq, request_id=core.command.request_id,
                            execution_id=core.execution_id, behavior='WAVE_RIGHT')
        self.started, self.responded, self.future = self.clock(), None, None

    def poll(self, core):
        require(self.request is not None, 'No wave SD preparation')
        require(self.clock()-self.started < self.contract['timeout_seconds'], 'Wave SD preparation timed out')
        if self.future is None and self.responded is None:
            if not self.client.service_is_ready():
                return False
            message = self.request_factory()
            self.future = self.client.call_async(message)
            self.record('wave_sd_request', request=self.request, message=self.serialize(message))
        if self.future is not None and self.responded is None and self.future.done():
            response = self.future.result()
            self.record('wave_sd_response', request=self.request,
                        response=self.serialize(response) if response is not None else None)
            require(response is not None and response.response.header.code == 0
                    and response.response.status.value == 1, 'Official wave SD rejected or unsuccessful')
            self.responded = self.clock()
        m, o = core.motion.mc, core.motion.odom
        ready = (self.responded is not None and m is not None and o is not None
                 and min(m.received, o.received) >= self.responded+self.contract['post_response_standing_seconds']
                 and core.standing_ready())
        if ready:
            self.record('wave_sd_standing', request=self.request, responded_at=self.responded,
                        mc_received=m.received, odom_received=o.received)
            self.request = None
        return ready

    def cancel(self):
        if self.request is not None:
            if self.future is not None:
                self.future.cancel()
            self.record('wave_sd_wait_cancelled', request=self.request,
                        note='local wait only; a sent SD cannot be recalled; no wave dispatched')
        self.request = self.future = None


class SupervisorLease:
    """Observe host heartbeat advances using only the bridge's local clock."""
    def __init__(self, run_id, session_id, timeout=5., clock=time.monotonic):
        self.run_id, self.session_id, self.timeout, self.clock = run_id, session_id, timeout, clock
        self.sequence, self.received = -1, clock()

    def check(self, value):
        require((value.get('run_id'), value.get('session_id')) == (self.run_id, self.session_id)
                and type(value.get('sequence')) is int and value['sequence'] >= self.sequence,
                'Invalid supervisor heartbeat')
        now = self.clock()
        if value['sequence'] > self.sequence:
            self.sequence, self.received = value['sequence'], now
        require(0 <= now-self.received < self.timeout, 'Supervisor heartbeat expired')


class RemoteExchange:
    """Nonce time is signed by this process's own clock and never renewed."""
    def __init__(self, run_id, session_id, clock=time.monotonic, *, profile=None, continuous=False):
        self.profile = interaction_profile() if profile is None else profile
        self.capabilities = profile_capabilities(self.profile)
        self.run_id, self.session_id, self.clock = run_id, session_id, clock
        self.issued = {}
        self.last_query = 0
        self.last_request = None
        self.last_decision = -1
        self.used = set()
        self.heading_targets = {}
        self.continuous = continuous
        self.request_sequences = {}
        self.last_request_sequence = 0
        self.closed_request_ack = None

    def observe(self, query, feedback, events, *, odom=None, sampled_remote_monotonic=None,
                feedback_sources=None, runtime_context=None):
        require(query.get('schema') == SCHEMA and query.get('kind') == 'observe'
                and (query.get('run_id'),query.get('session_id')) == (self.run_id,self.session_id),
                'Foreign observation query')
        require(query.get('task') in TASKS and type(query.get('sequence')) is int
                and query['sequence'] > self.last_query and isinstance(query.get('nonce'),str)
                and len(query['nonce']) == 32 and query['nonce'] not in self.issued, 'Replayed observation query')
        if self.continuous:
            sequence = query.get('request_sequence')
            require(type(sequence) is int and sequence >= self.last_request_sequence and sequence > 0
                    and (sequence > self.last_request_sequence or query['request_id'] == self.last_request),
                    'Replayed or changed task request sequence')
            require('closed_request_ack' in query, 'Missing explicit closure ACK')
        now = self.clock()
        odom = deepcopy(odom)
        if sampled_remote_monotonic is not None:
            require(type(sampled_remote_monotonic) in (int,float)
                    and math.isfinite(sampled_remote_monotonic) and sampled_remote_monotonic <= now,
                    'Invalid body snapshot clock')
            feedback = replace(feedback, age_s=feedback.age_s+now-sampled_remote_monotonic)
        require(feedback.valid and feedback.age_s <= SOURCE_AGE_SECONDS, 'Original feedback unavailable')
        consumed = query.get('last_consumed_request_id')
        require(consumed is None or isinstance(consumed, str) and consumed, 'Invalid consumed request')
        changed = consumed != query['request_id']
        target = None
        if query['task'] in HEADING_OFFSETS:
            goal = self.heading_targets.get(query['request_id'])
            if goal is None:
                goal = dict(source=HEADING_SOURCE,run_id=self.run_id,session_id=self.session_id,
                    request_id=query['request_id'],task=query['task'],**relative_heading_goal(query['task'],feedback.yaw_rad,profile=self.profile),
                    established_remote_sequence=query['sequence'],established_remote_monotonic=now,origin_odom=odom)
            error = relative_heading_error(query['task'],goal,feedback.yaw_rad,profile=self.profile)
            feedback = replace(feedback,heading_error_rad=error)
            target = dict(goal,current_odom=odom,current_yaw_rad=feedback.yaw_rad,heading_error_rad=error)
        else:
            feedback = replace(feedback,heading_error_rad=relative_heading_error(query['task'],None,feedback.yaw_rad,profile=self.profile))
        validate_heading_target(target,feedback,query,query['sequence'],now,profile=self.profile)
        goal = heading_goal(target) if target is not None else None
        require(query['request_id'] not in self.heading_targets or self.heading_targets[query['request_id']] == goal,
                'Heading target moved within request')
        self.heading_targets[query['request_id']] = goal
        if self.continuous:
            self.request_sequences[query['request_id']] = query['request_sequence']
            self.last_request_sequence = query['request_sequence']
        packet = dict(schema=SCHEMA, kind='observation', run_id=self.run_id, session_id=self.session_id,
            nonce=query['nonce'], sequence=query['sequence'], remote_sequence=query['sequence'],
            request_id=query['request_id'], task=query['task'], remote_issued_monotonic=now,
            source_age_s=feedback.age_s, feedback=asdict(feedback), capabilities=list(self.capabilities),
            request_changed=changed, events=events,heading_target=target,
            feedback_sources=deepcopy(feedback_sources), sampled_remote_monotonic=sampled_remote_monotonic,
            host_sent_monotonic=query['host_sent_monotonic'],
            response_record=dict(file='bridge_events.jsonl',kind='observation_response',response_id=query['nonce']))
        if runtime_context is not None:
            packet['runtime_context'] = deepcopy(runtime_context)
        if self.continuous:
            packet.update(request_sequence=query['request_sequence'], closed_request_ack=deepcopy(query['closed_request_ack']))
        self.issued[query['nonce']] = deepcopy(packet)
        self.last_query, self.last_request = query['sequence'], query['request_id']
        # Bounded cache; an evicted nonce can never be resurrected from a file.
        while len(self.issued) > 32:
            expired = next(iter(self.issued))
            del self.issued[expired]
            if self.continuous:
                self.used.discard(expired)
        return packet

    def acknowledge_closed_history(self, query, gate, core, events, event_sequence):
        """Receipt cursor is not an ACK: only an explicit consumed closure retires."""
        require(self.continuous, 'History retirement requires continuous mode')
        ack = query['closed_request_ack']
        if ack is None:
            return
        require(isinstance(ack, dict) and set(ack) == {'event_seq', 'decision_seq', 'request_sequence', 'request_id'}
                and all(type(ack[k]) is int for k in ('event_seq', 'decision_seq', 'request_sequence'))
                and -1 <= ack['event_seq'] <= query['after_event_seq']
                and 0 < ack['request_sequence'] <= query['request_sequence']
                and isinstance(ack['request_id'], str), 'Invalid consumed closure ACK')
        if ack != self.closed_request_ack:
            require(self.closed_request_ack is None or
                    (ack['event_seq'] > self.closed_request_ack['event_seq']
                     and ack['request_sequence'] > self.closed_request_ack['request_sequence']), 'Reordered closure ACK')
            require(ack['event_seq'] == event_sequence-1
                    and self.request_sequences.get(ack['request_id']) == ack['request_sequence']
                    and (core.command is None or core.terminal), 'Closure ACK has unconsumed or active execution')
            gate.retire_history(ack['event_seq'], ack['decision_seq'])
            core.retire_events()
            events.clear()
            self.closed_request_ack = deepcopy(ack)
        retired = {request for request, sequence in self.request_sequences.items()
                   if sequence <= ack['request_sequence'] and request != query['request_id']}
        for request in retired:
            self.request_sequences.pop(request)
            self.heading_targets.pop(request, None)
        for nonce, packet in list(self.issued.items()):
            if packet['request_id'] in retired:
                del self.issued[nonce]
                self.used.discard(nonce)

    def consume(self, message):
        from ..control.x2_behavior import Decision, Observation, Feedback
        require(message.get('schema') == SCHEMA and message.get('kind') == 'decision'
                and (message.get('run_id'),message.get('session_id')) == (self.run_id,self.session_id),
                'Foreign decision envelope')
        nonce = message.get('nonce')
        require(nonce in self.issued and nonce not in self.used, 'Unknown/reused decision nonce')
        issued = self.issued[nonce]
        age = self.clock()-issued['remote_issued_monotonic']
        require(0 <= age <= DECISION_SECONDS, 'Decision expired on remote clock')
        decision = Decision.from_dict(message['decision'])
        require((decision.run_id,decision.session_id,decision.request_id,decision.observation_seq)
                == (self.run_id,self.session_id,issued['request_id'],issued['sequence'])
                and decision.request_id == self.last_request and decision.decision_seq > self.last_decision,
                'Decision attribution superseded or replayed')
        observation = Observation(self.run_id,self.session_id,issued['sequence'],issued['request_id'],
            issued['task'],issued['remote_issued_monotonic'],feedback=Feedback(**issued['feedback']),
            capabilities=tuple(issued['capabilities']),request_changed=issued['request_changed'])
        self.used.add(nonce)
        self.last_decision = decision.decision_seq
        return decision, observation, age


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    return str(value) if isinstance(value,float) and not math.isfinite(value) else value


class SensorEvidence:
    """Callback evidence, separate from the executor's accepted body state."""
    def __init__(self, record):
        self.record, self.latest, self.trigger = record, {}, None
        self.sequence = 0

    def update(self, sensor, message, update, core_sample, *, received, archive=True):
        self.sequence += 1
        snapshot = dict(snapshot_id=f'{sensor}-{self.sequence}', sensor=sensor,
            clock_domain='linux_bridge_monotonic', remote_received_monotonic=received,
            message=deepcopy(clean(message)), accepted=None, acceptance='pending', sequence=None)
        self.latest[sensor] = snapshot
        self.trigger = snapshot
        try:
            ok = update(received=received)
            sample = core_sample() if core_sample is not None else None
            if ok and core_sample is not None:
                require(sample is not None and sample.received == received,
                        'Accepted sensor differs from callback receipt')
            snapshot = dict(snapshot, accepted=bool(ok), acceptance='accepted' if ok else 'rejected',
                sequence=sample.sequence if ok and sample is not None else None)
        except BaseException as error:
            snapshot = dict(snapshot, acceptance='error', update_error=type(error).__name__+':'+str(error))
            raise
        finally:
            self.latest[sensor] = snapshot
            self.trigger = None
            if archive:
                self.record(sensor, **deepcopy(snapshot))
        return ok

    def accepted(self, sensor, sample):
        snapshot = self.latest.get(sensor)
        require(sample is not None and snapshot is not None and snapshot['accepted'] is True
                and snapshot['sequence'] == sample.sequence
                and snapshot['remote_received_monotonic'] == sample.received,
                'Feedback source missing or differs from accepted '+sensor)
        return deepcopy(snapshot)


def feedback_snapshot(core, sensors, clock=time.monotonic):
    from ..control.x2_skills import yaw
    from ..control.x2_behavior import Behavior, ExecutionStatus, Feedback
    now = clock()
    odom, mc = core.motion.odom, core.motion.mc
    require(odom is not None and mc is not None, 'Feedback missing')
    originals = dict(odom=sensors.accepted('odom', odom), mc=sensors.accepted('mc', mc))
    age = max(now-odom.received, now-mc.received)
    valid = 0 <= now-odom.received <= .25 and 0 <= now-mc.received <= .5
    body = Feedback(valid=valid, age_s=age, source='official_sim_odom_and_mc',
        status=ExecutionStatus(core.status), behavior=Behavior(core.command.behavior if core.command else 'HOLD'),
        execution_id=core.execution_id or '', request_id=core.command.request_id if core.command else '',
        standing=core.standing_ready(), yaw_rad=yaw(odom.quaternion), vx_m_s=odom.linear[0],
        wz_rad_s=odom.angular[2], heading_error_rad=0., mc_mode=mc.mode)
    return body, originals, now


def runtime_context_snapshot(core, body, event_sequence):
    """Same factual readiness projection as the archived collection_guard.v1.

    Called only after the bridge has confirmed stable preparation. Current
    standing is a separate model input, never substituted for this guard.
    """
    m = core.motion.mc
    facts = dict(source_registered=core.motion._source_registered, armed=core.motion.armed,
        stable_preparation_completed=True, fresh_body=body.valid,
        ownership_supported=bool(m and m.source in ('', SOURCE) and core.motion._source_registered),
        actual_mc_source=None if m is None else m.source, motion_state=core.motion.state,
        failures=list(core.motion.failures), core_status=core.status)
    if facts['failures'] or facts['core_status'] in ('UNKNOWN', 'FAILED'):
        state = 'LATCHED'
    elif not all(facts[k] for k in ('source_registered','armed','stable_preparation_completed',
                                    'fresh_body','ownership_supported')):
        state = 'RECOVERING'
    else:
        state = 'READY'
    return dict(protection_state=state, facts=facts, event_sequence=event_sequence,
                source='collection_guard.v1 factual projection; not a model decision')


def publish_observation(directory, packet, record):
    # Persist the exact response before exposing its overwritable mailbox.
    record('observation_response', response_id=packet['nonce'], response=deepcopy(packet))
    atomic(Path(directory)/'observation_response.json', packet)
    record('observation_sent', nonce=packet['nonce'], remote_sequence=packet['remote_sequence'],
        remote_issued_monotonic=packet['remote_issued_monotonic'], source_age_s=packet['source_age_s'],
        heading_target=packet['heading_target'], response_record=packet['response_record'])


def _diagnostic_reference(value, *, kind, run_id, session_id):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    return dict(file='bridge_events.jsonl', kind=kind, run_id=run_id, session_id=session_id,
        request_id=value.get('request_id'), execution_id=value.get('execution_id'),
        event_seq=value.get('event_seq'), sha256=hashlib.sha256(encoded).hexdigest(),
        encoding='json ensure_ascii=False separators=comma:colon; hash covers event or summary value',
        scope='recorded bounded diagnostics; original dropped-record counters still apply')


def _compact_diagnostics(value, archive):
    """Transport projection only; retain all scalar facts and controller budgets."""
    result = deepcopy({key:item for key,item in value.items()
                       if key != 'events' or not isinstance(item, list)})
    if isinstance(value.get('events'), list):
        result['event_history_transport_omitted_count'] = len(value['events'])
    precision = result.get('precision_turn')
    if isinstance(precision, dict):
        if isinstance(precision.get('trace'), list):
            precision['trace_transport_omitted_count'] = len(precision.pop('trace'))
        integral = precision.get('integral')
        if isinstance(integral, dict) and isinstance(integral.get('samples'), list):
            integral['samples_transport_omitted_count'] = len(integral.pop('samples'))
        precision['diagnostics_archive'] = deepcopy(archive)
    return result


def _archive_execution(event, record):
    # The reference becomes usable only after the full event was written/flushed.
    record('execution', event=event)
    reference = _diagnostic_reference(event, kind='execution', run_id=event['run_id'], session_id=event['session_id'])
    compact = deepcopy(event)
    compact['raw_feedback'] = _compact_diagnostics(event['raw_feedback'], reference)
    if isinstance(event['raw_feedback'].get('summary'), dict):
        compact['raw_feedback']['summary'] = _compact_diagnostics(event['raw_feedback']['summary'], reference)
    compact['raw_feedback']['diagnostics_archive'] = reference
    return compact


def _execution_mailbox(reg, event, core_summary):
    return dict(schema=SCHEMA, run_id=reg['run_id'], session_id=reg['session_id'], event=event,
        core=_compact_diagnostics(core_summary, event['raw_feedback']['diagnostics_archive']))


def _archive_summary(summary, reg, record):
    record('bridge_summary', run_id=reg['run_id'], session_id=reg['session_id'], summary=summary)
    reference = _diagnostic_reference(summary, kind='bridge_summary', run_id=reg['run_id'], session_id=reg['session_id'])
    return _compact_diagnostics(summary, reference)


def wrap_execution(reg, raw, event_seq, recent_requests, sensors, trigger=None, wave_owner=None):
    """Serialize original executor ownership and facts without renaming ACKs."""
    from ..control.x2_behavior import ExecutionEvent
    from ..control.x2_skills import SKILL_PACK_ID
    return ExecutionEvent(reg['run_id'],reg['session_id'],event_seq,raw['decision_seq'],raw['execution_id'],
        raw['request_id'],'sim',SKILL_PACK_ID,raw['status'],raw['behavior'],raw['received_monotonic'],
        mc_request=dict(recent_actual_requests=list(recent_requests),skill_profile_id=raw.get('skill_profile_id'),
            command_seconds=raw.get('command_seconds'),intensity=raw.get('intensity'),
            **{key:raw[key] for key in ('model_intensity','resolved_velocity','velocity_resolution') if key in raw},
            **({'wave_owner':deepcopy(wave_owner)} if raw['behavior'] == 'WAVE_RIGHT' and wave_owner else {})),
        raw_feedback=clean(deepcopy(dict(raw, sensors=sensors, triggering_snapshot=trigger))),
        stop_evidence=clean({k:v for k,v in raw.items() if k in ('stop_confirmed','base_stop_confirmed',
            'distance_m','forward_distance_m','lateral_distance_m','angle_rad','advance_direction_ok')}),
        protection_reason=raw.get('protection_reason'), fact=raw.get('fact','status'),
        finish_decision_seq=raw.get('finish_decision_seq'),finish_request_id=raw.get('finish_request_id'))


def rejected_execution(reg, decision, event_seq, core):
    """A real executor refusal owns the refused decision, never an old command."""
    from ..control.x2_behavior import ExecutionEvent, ExecutionStatus
    from ..control.x2_skills import SKILL_PACK_ID
    return ExecutionEvent(reg['run_id'],reg['session_id'],event_seq,decision.decision_seq,
        'rejected-'+str(decision.decision_seq),decision.request_id,'sim',SKILL_PACK_ID,
        ExecutionStatus.REJECTED,decision.behavior,time.monotonic(),
        mc_request=dict(skill_profile_id=core.skill_profile_id,command_seconds=None,intensity=decision.intensity),
        raw_feedback=dict(reason='executor_preconditions',summary=clean(core.summary()),
            summary_scope='executor_context_not_rejected_execution'))


class WaveSession:
    """One fresh old-SDK child per admitted wave; newer SDK only queries state.

    Ownership is local/exclusive and temporal. Global MC state and task_id=0 do
    not provide a strong task identifier. Closing a child is never cancellation.
    """
    def __init__(self, directory, reg, core, binding, record, query_client, request_factory,
                 serialize, *, clock=time.monotonic, boundary_factory=None):
        from .x2_preset_compat import PresetBoundary
        self.directory, self.reg, self.core = Path(directory), reg, core
        self.binding, self.record, self.clock = binding, record, clock
        self.client, self.request_factory, self.serialize = query_client, request_factory, serialize
        self.factory = boundary_factory or PresetBoundary
        self.boundary = self.owner = self.future = None
        self.query_sent = self.next_query = 0.
        self.failed = False
        self.actions = 0
        self.stand_preparation = None

    def _log(self, kind, **fields):
        self.record(kind, wave_owner=deepcopy(self.owner), **fields)

    def _drop_query(self):
        if self.future is not None:
            self.future.cancel()
            self.future = None

    def preset(self, command):
        from .x2_preset_compat import FIXED, CANCEL
        require(not self.failed, 'Wave transport is latched closed')
        if command == FIXED:
            require(self.boundary is None and (self.reg.get('session_profile') == 'continuous' or self.actions < 50)
                    and (self.stand_preparation is None or self.stand_preparation.request is None),
                    'Previous wave helper/preparation not reaped')
            self.actions += 1
            if self.stand_preparation is not None:
                self.stand_preparation.start(self.core)
            else:
                self._start_helper()
        else:
            require(command == CANCEL and self.boundary is not None, 'No exclusive active wave to cancel')
            self._drop_query()
            self.boundary.request(command)

    def _start_helper(self):
        child_run = 'skills-'+time.strftime('%Y%m%d-%H%M%S', time.gmtime())+'-'+uuid.uuid4().hex
        output = self.directory/'preset-actions'/child_run/'probe-00-wave_right'
        output.mkdir(parents=True,exist_ok=False)
        self.boundary = self.factory(output,self._log,binding=self.binding,clock=self.clock,allow_cancel=True)
        self.owner = dict(run_id=self.reg['run_id'],session_id=self.reg['session_id'],
            decision_seq=self.core.command.decision_seq,request_id=self.core.command.request_id,
            execution_id=self.core.execution_id,helper_run_id=child_run,helper_probe_id=output.name,
            helper_nonce=self.boundary.nonce,attribution=self.reg['skill_profile']['wave_contract']['attribution'])
        atomic(output/'owner.json',self.owner)
        self.boundary.start()

    def fail(self, reason):
        if not self.failed:
            self.failed = True
            self._log('wave_transport_failed',reason=reason)
            self.core.cancel(reason)

    def poll(self):
        from .x2_preset_compat import FIXED
        now = self.clock()
        if self.stand_preparation is not None and self.stand_preparation.request is not None:
            try:
                if self.core.cancel_requested or self.core.status == 'UNKNOWN' or self.core.terminal:
                    self.stand_preparation.cancel()
                    self.core.withdraw_unsent_wave()
                elif self.stand_preparation.poll(self.core):
                    self._start_helper()
            except Exception as error:
                self.stand_preparation.cancel()
                self.fail('wave_sd_preparation:'+type(error).__name__+':'+str(error))
        if self.boundary is not None and not self.failed:
            try:
                for event in self.boundary.poll():
                    kind = event['kind']
                    if kind == 'error':
                        self.fail(event['payload']['error'])
                        break
                    if kind in ('sent','cancel_sent'):
                        self._log('actual_preset_request',kind_of_request=kind,**event['payload'],
                                  helper_sent_at=event['monotonic'])
                        (self.core.preset_sent if kind == 'sent' else self.core.preset_cancel_sent)(event['monotonic'])
                    elif kind in ('response','cancel_response'):
                        self._log('actual_preset_response',kind_of_response=kind,**event['payload'])
                        r = event['payload']['message']['response']
                        (self.core.preset_ack if kind == 'response' else self.core.preset_cancel_ack)(
                            r['header']['code'],r['task_id'],r['state']['value'])
                if not self.failed and not self.boundary.sent:
                    if self.core.cancel_requested or self.core.status == 'UNKNOWN':
                        self.close_action()
                        self.core.withdraw_unsent_wave()
                    elif self.boundary.ready and self.core.standing_ready() and self.core.query_state == 1 and now-self.core.query_at <= .5:
                        self._drop_query()
                        self.core.preset_dispatch()
                        self.boundary.request(FIXED)
            except Exception as error:
                self.fail('wave_transport:'+type(error).__name__+':'+str(error))
        if self.future is not None:
            try:
                if self.future.done():
                    response = self.future.result()
                    self.future = None
                    self._log('preset_query',response=self.serialize(response) if response else None,
                              scope='global_player_state_not_task_identity')
                    if response is None:
                        raise RuntimeError('empty preset query')
                    r = response.response
                    self.core.preset_query(r.header.code,r.task_id,r.state.value,received=now)
                elif now-self.query_sent >= .5:
                    self._drop_query()
                    raise RuntimeError('preset query timeout')
            except Exception as error:
                self._drop_query()
                self.core.query_state = None
                if self.boundary is not None and self.boundary.sent:
                    self.fail('wave_query:'+str(error))
        active_wave = self.core.command is not None and self.core.command.behavior == 'WAVE_RIGHT' and not self.core.terminal
        may_query = not active_wave or self.core.wave_dispatch_at is None or self.core.task_id is not None
        if self.future is None and now >= self.next_query and may_query and self.client.service_is_ready():
            request = self.request_factory(self.core.task_id or 0)
            self.future = self.client.call_async(request)
            self.query_sent, self.next_query = now, now+.1
            self._log('preset_query_request',message=self.serialize(request))

    def close_action(self):
        if self.stand_preparation is not None:
            self.stand_preparation.cancel()
        if self.boundary is not None:
            boundary = self.boundary
            boundary.close()  # Reap before allowing another action; an exception keeps the slot closed.
            self.boundary = None
        self._drop_query()

    def reap_terminal(self):
        if self.boundary is not None and self.core.terminal:
            self.close_action()


def execute(directory):
    require(os.environ.get('ROS_DOMAIN_ID') == '47' and os.environ.get('ROS_LOCALHOST_ONLY') == '1',
            'Isolated simulation ROS environment required')
    directory = Path(directory)
    reg, ready = read_json(directory/'bridge_registration.json'), read_json(directory/'brain_ready.json')
    limits = validate_session_profile(reg)
    require(validate_session_profile(read_json(directory/'behavior_registration.json')) == limits,
            'Host and bridge session limits differ')
    require(ready.get('session_contract_id') == limits['session_contract_id'], 'Brain session limits differ')
    continuous = limits['session_profile'] == 'continuous'
    require(reg['qualification_only'] is True, 'Only bounded candidate qualification supported')
    require(all(reg[k] == ready[k] for k in ('run_id','session_id')), 'Bridge session differs')
    skill_profile = registered_skill_profile(directory,reg)
    require(ready.get('skill_profile_id') == skill_profile['profile_id'], 'Brain execution profile differs')
    capabilities = profile_capabilities(reg['interaction_profile'])
    wave_poses = binding = None
    if skill_profile['wave_enabled']:
        from . import x2_preset_identity as preset_identity
        from .x2_skills import official_wave_poses
        binding = preset_identity.verify_build(preset_identity.LINUX_ROOT)
        require(binding == skill_profile['wave_contract']['helper_binding']
                and binding == read_json(directory/'preset_compat/binding.json'), 'Installed wave contract differs')
        prepared = preset_identity.read(preset_identity.LINUX_ROOT/'prepared.json')
        for name in ('x2_preset_identity.py','x2_preset_compat.py','x2_preset_helper.py'):
            require(preset_identity.sha(Path(__file__).with_name(name)) == prepared['tool_files']['malecns_lab/experiments/'+name],
                    'Wave helper sources differ from inspected build')
        wave_poses = official_wave_poses(preset_identity.LINUX_ROOT.parent)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction, JointStateArray
    from aimdk_msgs.srv import SetMcInputSource, GetMcPresetMotionState, SetMcAction
    from nav_msgs.msg import Odometry
    from ..control.x2_skills import SOURCE, SKILL_PACK_ID, RIGHT_ARM
    from ..control.x2_behavior import DecisionGate
    started = time.monotonic()
    deadline = math.inf if reg['bridge_wall_seconds'] is None else started+reg['bridge_wall_seconds']
    interrupted, handlers = [], {}
    gate = DecisionGate(reg['run_id'],reg['session_id'],ready['policy_id'],ready['checkpoint_id'],ready['input_map_id'],
        sim_wave_profile_id=skill_profile['profile_id'] if skill_profile['wave_enabled'] else None,
        monotonic_execution_ids=continuous, single_action_requests=continuous)
    exchange = RemoteExchange(reg['run_id'],reg['session_id'],profile=reg['interaction_profile'], continuous=continuous)
    lease = SupervisorLease(reg['run_id'], reg['session_id'], limits['supervisor_timeout_seconds'])
    last_resource_check = -math.inf
    events, actual_requests = [], []
    event_sequence = 0
    core = node = wave = None
    failure = None
    last_decision = -1
    last_query = 0
    summary = {}
    stream = (directory/'bridge_events.jsonl').open('x')
    def log(kind, **fields):
        stream.write(json.dumps(clean(dict(kind=kind, remote_monotonic=time.monotonic(), **fields)),allow_nan=False)+'\n')
        stream.flush()
    sensors = SensorEvidence(log)
    def publish(command):
        msg = McLocomotionVelocity()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.source = SOURCE
        msg.forward_velocity,msg.lateral_velocity,msg.angular_velocity = command
        publisher.publish(msg)
        record = dict(topic='/aima/mc/locomotion/velocity', message=clean(message_to_ordereddict(msg)),
                      remote_sent_monotonic=time.monotonic())
        actual_requests.append(record)
        del actual_requests[:-4]
        log('actual_mc_request', **record)
    def execution(raw):
        nonlocal event_sequence
        require(len(events) < 512, 'Execution event budget exhausted')
        event = wrap_execution(reg,raw,event_sequence,actual_requests,sensors.latest,sensors.trigger,
            wave_owner=wave.owner if wave is not None and wave.owner and wave.owner['execution_id'] == raw['execution_id'] else None)
        gate.execution_event(event)
        event_sequence += 1
        compact = _archive_execution(event.to_dict(), log)
        events.append(compact)
        atomic(directory/'execution_latest.json', _execution_mailbox(reg, compact, clean(core.summary())))
    def on_odom(msg):
        received = time.monotonic()
        p,q,v,w = msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist.linear,msg.twist.twist.angular
        sensors.update('odom', message_to_ordereddict(msg),
            lambda **stamp:core.update_odom((p.x,p.y,p.z),(q.x,q.y,q.z,q.w),(v.x,v.y,v.z),(w.x,w.y,w.z),**stamp),
            lambda:core.motion.odom, received=received)
    def on_mc(msg):
        received = time.monotonic()
        m = msg.motion_status
        sensors.update('mc', message_to_ordereddict(msg),
            lambda **stamp:core.update_mc(msg.input_source.name,msg.action_info.action_desc,msg.body_status.value,
                msg.action_info.status.value,player_state=m.player_state.value,motion=m.motion,
                area=m.control_area.value,motion_type=m.type.value,**stamp),
            lambda:core.motion.mc, received=received)
    def on_joints(msg):
        received = time.monotonic()
        def update(**stamp):
            ok = core.update_joints({j.name:(j.position,j.velocity) for j in msg.joints},**stamp)
            if wave is not None and any(j.error_code for j in msg.joints if j.name in RIGHT_ARM):
                core.joints = None
                core.cancel('arm_joint_error')
                return False
            return ok
        sensors.update('arm', message_to_ordereddict(msg),update,None,received=received,
            archive=wave is not None and core.command is not None and core.command.behavior == 'WAVE_RIGHT')
    def spin():
        _pump_callbacks(rclpy.spin_once, node, core)
        if wave is not None:
            wave.poll()
        core.standing_ready()
        core.tick()
        if wave is not None:
            wave.reap_terminal()
    def register():
        client = node.create_client(SetMcInputSource,'/aimdk_5Fmsgs/srv/SetMcInputSource')
        end = min(deadline-15,started+12.)
        while not client.service_is_ready() and time.monotonic()<end and not interrupted:
            spin()
            time.sleep(.01)
        for action in (McInputAction.INPUTACTION_ADD,McInputAction.INPUTACTION_ENABLE):
            require(client.service_is_ready() and time.monotonic()<end and not interrupted,'Source registration unavailable')
            req = SetMcInputSource.Request()
            req.request.header.stamp = node.get_clock().now().to_msg()
            req.action.value = action
            req.input_source.name,req.input_source.priority,req.input_source.timeout = SOURCE,40,1000
            future = client.call_async(req)
            stop = min(end,time.monotonic()+2.)
            while not future.done() and time.monotonic()<stop and not interrupted:
                spin()
                time.sleep(.01)
            response = future.result() if future.done() else None
            log('registration',action=action,response=message_to_ordereddict(response) if response else None)
            if response and response.response.header.code == 0:
                core.motion.confirm_source_registration()
                return
        raise RuntimeError('Source registration failed')
    try:
        log('configuration',skill_profile=skill_profile)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        for sig in (signal.SIGTERM,signal.SIGINT):
            handlers[sig] = signal.signal(sig,lambda sig,frame:interrupted.append(sig))
        node = Node('malecns_x2_behavior_bridge')
        publisher = node.create_publisher(McLocomotionVelocity,'/aima/mc/locomotion/velocity',10)
        core = candidate_executor(reg,publish,record=execution,
            preset=lambda command:wave.preset(command),wave_poses=wave_poses)
        if binding is not None:
            query_client = node.create_client(GetMcPresetMotionState,'/aimdk_5Fmsgs/srv/GetMcPresetMotionState')
            def query_request(task_id):
                request = GetMcPresetMotionState.Request()
                request.request.header.stamp = node.get_clock().now().to_msg()
                request.request.task_id = task_id
                return request
            def wave_record(kind, **fields):
                if kind == 'actual_preset_request':
                    actual_requests.append(dict(service='/aimdk_5Fmsgs/srv/SetMcPresetMotion',**fields))
                    del actual_requests[:-4]
                log(kind,**fields)
            wave = WaveSession(directory,reg,core,binding,wave_record,query_client,query_request,message_to_ordereddict)
            if skill_profile['profile_id'] in FULL_ACTION_PROFILE_IDS:
                stand_client = node.create_client(SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')
                def stand_request():
                    require(wave.boundary is None, 'Active preset helper prevents SD preparation')
                    request = SetMcAction.Request()
                    request.header.stamp = node.get_clock().now().to_msg()
                    request.source = SOURCE
                    request.command.action_desc = 'STAND_DEFAULT'
                    return request
                wave.stand_preparation = WaveStandPreparation(stand_client,stand_request,
                    message_to_ordereddict,log,skill_profile['wave_preparation'])
            log('preset_parent_identity',binding=binding,identity=preset_identity.runtime_identity(preset_identity.LINUX_ROOT,'v1.1.0'))
        core.motion.arm()
        qos = QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry,'/aima/hal/odom/state',on_odom,qos)
        node.create_subscription(McCommonState,'/aima/mc/common/state',on_mc,qos)
        node.create_subscription(JointStateArray,'/aima/hal/joint/arm/state',on_joints,qos)
        register()
        while not core.standing_ready() and time.monotonic()<started+12. and not interrupted:
            spin()
            time.sleep(.02)
        require(core.standing_ready(),'Fresh stable standing not confirmed')
        atomic(directory/'bridge_ready.json',dict(run_id=reg['run_id'],session_id=reg['session_id'],
            capabilities=capabilities, qualification_only=True,wave_enabled=skill_profile['wave_enabled'],skill_profile=skill_profile,
            session_contract_id=limits['session_contract_id']))
        while time.monotonic()<deadline-14. and not interrupted:
            began = time.monotonic()
            if continuous and began-last_resource_check >= 1.:
                lease.check(read_json(directory/'supervisor_heartbeat.json'))
                require(shutil.disk_usage(directory).free >= limits['disk_reserve_bytes'], 'Bridge log disk reserve exhausted')
                last_resource_check = began
            if (directory/'emergency.json').exists():
                raise InterruptedError('independent_emergency_stop')
            spin()
            query_path,decision_path = directory/'observation_request.json',directory/'decision.json'
            if query_path.exists():
                query = read_json(query_path)
                if query['sequence'] > last_query:
                    current,originals,sampled_at = feedback_snapshot(core,sensors)
                    if current.valid:
                        packet = exchange.observe(query,current,[e for e in events if e['event_seq']>query['after_event_seq']],
                            odom=originals['odom'],sampled_remote_monotonic=sampled_at,feedback_sources=originals,
                            runtime_context=runtime_context_snapshot(core,current,event_sequence-1))
                        if continuous:
                            exchange.acknowledge_closed_history(query, gate, core, events, event_sequence)
                        publish_observation(directory,packet,log)
                        last_query = query['sequence']
            if decision_path.exists():
                message = read_json(decision_path)
                sequence = message['decision']['decision_seq']
                if sequence > last_decision:
                    decision,observation,remote_age = exchange.consume(message)
                    actual_active = gate.active['execution_id'] if gate.active else None
                    require(actual_active == message.get('expected_active_execution_id')
                            or gate.completed_action(decision.request_id) is not None,
                            'Execution ownership changed in transit; decision not executed')
                    gate.observe(observation)
                    admission = gate.admit(decision,time.monotonic())
                    log('remote_admission',decision=decision.to_dict(),accepted=admission.accepted,
                        reason=admission.reason,remote_nonce_age_s=remote_age,
                        host_inference_age_s=message['host_inference_age_s'])
                    if admission.reason == 'completed_action_request':
                        last_decision = sequence
                        continue  # Retain the raw output, but create no second skill or artificial event.
                    require(admission.accepted,'Remote decision gate: '+admission.reason)
                    wave_seconds = 32. if skill_profile['profile_id'] in FULL_ACTION_PROFILE_IDS else 26.
                    enough_wave_budget = str(decision.behavior) != 'WAVE_RIGHT' or str(decision.phase) != 'START' or deadline-time.monotonic() >= wave_seconds
                    accepted = enough_wave_budget and core.submit_decision(decision,observation.received_monotonic)
                    if not accepted:
                        # A locally admitted decision can still fail current MC preconditions.
                        # Preserve the actual rejection; never replace it with HOLD.
                        if gate.active and gate.active['execution_id'] is None:
                            require(len(events) < 512, 'Execution event budget exhausted')
                            event = rejected_execution(reg,decision,event_sequence,core)
                            gate.execution_event(event)
                            event_sequence += 1
                            events.append(_archive_execution(event.to_dict(), log))
                        else:
                            raise RuntimeError('In-flight decision rejected by executor')
                    last_decision = sequence
            if gate.blocked_reason:
                raise RuntimeError('Execution protection latched: '+gate.blocked_reason)
            time.sleep(max(0.,.02-(time.monotonic()-began)))
        if interrupted:
            raise InterruptedError('bridge_signal')
        failure = 'bridge_wall_budget'
    except BaseException as error:
        failure = type(error).__name__+':'+str(error)
        try:
            log('exception',reason=failure)
        except BaseException:
            pass
    finally:
        # Each cleanup step is independent of logging and every other step.
        if core is not None:
            try:
                if core.command and not core.terminal:
                    core.cancel(failure or 'bridge_exit')
                else:
                    core.motion.request_stop(failure or 'bridge_exit',failure=skill_profile['profile_id'] not in FULL_ACTION_PROFILE_IDS)
            except BaseException:
                pass
            end = min(deadline,time.monotonic()+12.)
            def stopped_basis():
                return (cleanup_base_stop_basis(core) if skill_profile['profile_id'] in FULL_ACTION_PROFILE_IDS
                        else 'owned_executor_stop_window' if core.motion.state == 'STOPPED' else None)
            while time.monotonic()<end and (stopped_basis() is None
                    or wave is not None and core.command is not None and core.command.behavior == 'WAVE_RIGHT' and not core.terminal):
                try:
                    spin()
                except BaseException:
                    try:
                        publish((0.,0.,0.))
                    except BaseException:
                        pass
                time.sleep(.02)
            try:
                publish((0.,0.,0.))
            except BaseException:
                pass
            summary = clean(core.summary())
            wave_stopped = (wave is None or core.command is None or core.command.behavior != 'WAVE_RIGHT'
                or core.wave_return_confirmed or core.terminal and core.wave_dispatch_at is None)
            summary.update(base_stop_confirmed=stopped_basis() is not None,base_stop_basis=stopped_basis(),wave_stop_confirmed=wave_stopped,
                           wave_cancellation_verified=False)
        if wave is not None:
            try:
                wave.close_action()
            except BaseException as error:
                summary['wave_stop_confirmed'] = False
                failure = (failure or '')+';helper_cleanup:'+str(error)
        summary.update(reason=failure,wall_seconds=time.monotonic()-started,qualification_only=True,
                       events=event_sequence, retained_events=len(events), session_profile=limits['session_profile'],
                       clock_domain='linux_bridge_monotonic',skill_profile=skill_profile)
        try:
            atomic(directory/'bridge_summary.json',_archive_summary(summary, reg, log))
        finally:
            try:
                stream.close()
            finally:
                if node is not None:
                    node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                for sig,handler in handlers.items():
                    signal.signal(sig,handler)
    return 0 if summary.get('base_stop_confirmed') and summary.get('wave_stop_confirmed',True) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--sim-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(mode='preview',ros_connected=False,brain_loaded=False,wave_enabled=False,
            skill_profile=candidate_skill_profile())))
        return 0
    if not args.sim_only or args.output is None:
        parser.error('execute requires --sim-only --output')
    return execute(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
