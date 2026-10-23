"""Simulation-only behavior transport. No teacher, task planner or brain here."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import signal
import time

from ..learning.x2_behavior_session import (SCHEMA, TASKS, CAPABILITIES, DECISION_SECONDS,
    SOURCE_AGE_SECONDS, HEADING_OFFSETS, HEADING_SOURCE, heading_goal, wrap_heading,
    validate_heading_target, atomic, read_json, require)
from ..control.x2_skill_profile import candidate_skill_profile, validate_candidate_skill_profile


def registered_skill_profile(directory, reg):
    """Bind both process registrations to this exact simulation-only profile."""
    host = read_json(Path(directory)/'behavior_registration.json')
    require(all(host.get(k) == reg.get(k) for k in ('run_id','session_id')), 'Skill profile session differs')
    require(host.get('qualification_only') is True and host.get('simulation_only') is True
            and host.get('wave_enabled') is False, 'Host skill profile is not simulation qualification')
    profile = validate_candidate_skill_profile(reg.get('skill_profile'))
    require(validate_candidate_skill_profile(host.get('skill_profile')) == profile,
            'Host and bridge skill profiles differ')
    return profile


def candidate_executor(reg, publish, *, record=None, clock=time.monotonic):
    from ..control.x2_skills import SkillExecutor
    profile = validate_candidate_skill_profile(reg.get('skill_profile'))
    def disabled_preset(_):
        raise RuntimeError('Wave cancellation unqualified; preset requests disabled')
    return SkillExecutor(publish,disabled_preset,enabled=CAPABILITIES,wave_probe=False,record=record,clock=clock,
        command_seconds_by_behavior=profile['command_seconds_by_behavior'],skill_profile_id=profile['profile_id'])


class RemoteExchange:
    """Nonce time is signed by this process's own clock and never renewed."""
    def __init__(self, run_id, session_id, clock=time.monotonic):
        self.run_id, self.session_id, self.clock = run_id, session_id, clock
        self.issued = {}
        self.last_query = 0
        self.last_request = None
        self.last_decision = -1
        self.used = set()
        self.heading_targets = {}

    def observe(self, query, feedback, events, *, odom=None, sampled_remote_monotonic=None,
                feedback_sources=None):
        require(query.get('schema') == SCHEMA and query.get('kind') == 'observe'
                and (query.get('run_id'),query.get('session_id')) == (self.run_id,self.session_id),
                'Foreign observation query')
        require(query.get('task') in TASKS and type(query.get('sequence')) is int
                and query['sequence'] > self.last_query and isinstance(query.get('nonce'),str)
                and len(query['nonce']) == 32 and query['nonce'] not in self.issued, 'Replayed observation query')
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
                    request_id=query['request_id'],task=query['task'],relative_offset_rad=HEADING_OFFSETS[query['task']],
                    origin_yaw_rad=feedback.yaw_rad,target_yaw_rad=wrap_heading(feedback.yaw_rad+HEADING_OFFSETS[query['task']]),
                    established_remote_sequence=query['sequence'],established_remote_monotonic=now,origin_odom=odom)
            error = wrap_heading(goal['target_yaw_rad']-feedback.yaw_rad)
            feedback = replace(feedback,heading_error_rad=error)
            target = dict(goal,current_odom=odom,current_yaw_rad=feedback.yaw_rad,heading_error_rad=error)
        else:
            feedback = replace(feedback,heading_error_rad=0.)
        validate_heading_target(target,feedback,query,query['sequence'],now)
        goal = heading_goal(target) if target is not None else None
        require(query['request_id'] not in self.heading_targets or self.heading_targets[query['request_id']] == goal,
                'Heading target moved within request')
        self.heading_targets[query['request_id']] = goal
        packet = dict(schema=SCHEMA, kind='observation', run_id=self.run_id, session_id=self.session_id,
            nonce=query['nonce'], sequence=query['sequence'], remote_sequence=query['sequence'],
            request_id=query['request_id'], task=query['task'], remote_issued_monotonic=now,
            source_age_s=feedback.age_s, feedback=asdict(feedback), capabilities=list(CAPABILITIES),
            request_changed=changed, events=events,heading_target=target,
            feedback_sources=deepcopy(feedback_sources), sampled_remote_monotonic=sampled_remote_monotonic,
            host_sent_monotonic=query['host_sent_monotonic'],
            response_record=dict(file='bridge_events.jsonl',kind='observation_response',response_id=query['nonce']))
        self.issued[query['nonce']] = deepcopy(packet)
        self.last_query, self.last_request = query['sequence'], query['request_id']
        # Bounded cache; an evicted nonce can never be resurrected from a file.
        while len(self.issued) > 32:
            del self.issued[next(iter(self.issued))]
        return packet

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


def publish_observation(directory, packet, record):
    # Persist the exact response before exposing its overwritable mailbox.
    record('observation_response', response_id=packet['nonce'], response=deepcopy(packet))
    atomic(Path(directory)/'observation_response.json', packet)
    record('observation_sent', nonce=packet['nonce'], remote_sequence=packet['remote_sequence'],
        remote_issued_monotonic=packet['remote_issued_monotonic'], source_age_s=packet['source_age_s'],
        heading_target=packet['heading_target'], response_record=packet['response_record'])


def wrap_execution(reg, raw, event_seq, recent_requests, sensors, trigger=None):
    """Serialize original executor ownership and facts without renaming ACKs."""
    from ..control.x2_behavior import ExecutionEvent
    from ..control.x2_skills import SKILL_PACK_ID
    return ExecutionEvent(reg['run_id'],reg['session_id'],event_seq,raw['decision_seq'],raw['execution_id'],
        raw['request_id'],'sim',SKILL_PACK_ID,raw['status'],raw['behavior'],raw['received_monotonic'],
        mc_request=dict(recent_actual_requests=list(recent_requests),skill_profile_id=raw.get('skill_profile_id'),
            command_seconds=raw.get('command_seconds'),intensity=raw.get('intensity')),
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


def execute(directory):
    require(os.environ.get('ROS_DOMAIN_ID') == '47' and os.environ.get('ROS_LOCALHOST_ONLY') == '1',
            'Isolated simulation ROS environment required')
    directory = Path(directory)
    reg, ready = read_json(directory/'bridge_registration.json'), read_json(directory/'brain_ready.json')
    require(reg['qualification_only'] is True and reg['wave_enabled'] is False, 'Only bounded candidate qualification supported')
    require(all(reg[k] == ready[k] for k in ('run_id','session_id')), 'Bridge session differs')
    skill_profile = registered_skill_profile(directory,reg)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction, JointStateArray
    from aimdk_msgs.srv import SetMcInputSource
    from nav_msgs.msg import Odometry
    from ..control.x2_skills import SOURCE, SKILL_PACK_ID
    from ..control.x2_behavior import DecisionGate
    started = time.monotonic()
    deadline = started+reg['bridge_wall_seconds']
    interrupted, handlers = [], {}
    gate = DecisionGate(reg['run_id'],reg['session_id'],ready['policy_id'],ready['checkpoint_id'],ready['input_map_id'])
    exchange = RemoteExchange(reg['run_id'],reg['session_id'])
    events, actual_requests = [], []
    core = node = None
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
        require(len(events) < 512, 'Execution event budget exhausted')
        event = wrap_execution(reg,raw,len(events),actual_requests,sensors.latest,sensors.trigger)
        gate.execution_event(event)
        events.append(event.to_dict())
        log('execution', event=event.to_dict())
        atomic(directory/'execution_latest.json',dict(schema=SCHEMA,run_id=reg['run_id'],session_id=reg['session_id'],
            event=event.to_dict(), core=clean(core.summary())))
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
        sensors.update('arm', message_to_ordereddict(msg),
            lambda **stamp:core.update_joints({j.name:(j.position,j.velocity) for j in msg.joints},**stamp),
            None, received=received, archive=False)
    def spin():
        began = time.monotonic()
        for _ in range(64):
            if time.monotonic()-began >= .008:
                break
            rclpy.spin_once(node,timeout_sec=.0001)
        core.standing_ready()
        core.tick()
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
        core = candidate_executor(reg,publish,record=execution)
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
            capabilities=CAPABILITIES, qualification_only=True,wave_enabled=False,skill_profile=skill_profile))
        while time.monotonic()<deadline-14. and not interrupted:
            began = time.monotonic()
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
                            odom=originals['odom'],sampled_remote_monotonic=sampled_at,feedback_sources=originals)
                        publish_observation(directory,packet,log)
                        last_query = query['sequence']
            if decision_path.exists():
                message = read_json(decision_path)
                sequence = message['decision']['decision_seq']
                if sequence > last_decision:
                    decision,observation,remote_age = exchange.consume(message)
                    actual_active = gate.active['execution_id'] if gate.active else None
                    require(actual_active == message.get('expected_active_execution_id'),
                            'Execution ownership changed in transit; decision not executed')
                    gate.observe(observation)
                    admission = gate.admit(decision,time.monotonic())
                    log('remote_admission',decision=decision.to_dict(),accepted=admission.accepted,
                        reason=admission.reason,remote_nonce_age_s=remote_age,
                        host_inference_age_s=message['host_inference_age_s'])
                    require(admission.accepted,'Remote decision gate: '+admission.reason)
                    accepted = core.submit_decision(decision,observation.received_monotonic)
                    if not accepted:
                        # A locally admitted decision can still fail current MC preconditions.
                        # Preserve the actual rejection; never replace it with HOLD.
                        if gate.active and gate.active['execution_id'] is None:
                            event = rejected_execution(reg,decision,len(events),core)
                            gate.execution_event(event)
                            events.append(event.to_dict())
                            log('execution',event=event.to_dict())
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
                    core.motion.request_stop(failure or 'bridge_exit',failure=True)
            except BaseException:
                pass
            end = min(deadline,time.monotonic()+12.)
            while time.monotonic()<end and core.motion.state != 'STOPPED':
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
            summary.update(base_stop_confirmed=core.motion.state=='STOPPED',wave_cancellation_verified=False)
        summary.update(reason=failure,wall_seconds=time.monotonic()-started,qualification_only=True,
                       events=len(events),clock_domain='linux_bridge_monotonic',skill_profile=skill_profile)
        try:
            atomic(directory/'bridge_summary.json',summary)
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
    return 0 if summary.get('base_stop_confirmed') else 1


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
