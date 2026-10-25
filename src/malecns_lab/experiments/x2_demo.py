"""Manual, simulation-only official primitives using the existing safety executor."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from .x2_behavior_bridge import SensorEvidence, WaveSession, candidate_executor, clean
from ..control.x2_skills import SkillCommand, VelocityCore, SOURCE, RIGHT_ARM
from ..control.x2_skill_profile import candidate_skill_profile
from ..control.x2_interaction import interaction_profile
from ..learning.x2_behavior_session import atomic, read_json, require

TASK_SKILLS = {'SHOW_FORWARD': 'ADVANCE', 'FACE_LEFT': 'TURN_LEFT',
               'FACE_RIGHT': 'TURN_RIGHT', 'GREET': 'WAVE_RIGHT'}
DEMO_MOTION = dict(profile_id='malecns.manual-demo.turn04-feedback035.v1',
    advance_velocity_m_s=.2, advance_seconds=1., turn_velocity_rad_s=.4, turn_max_seconds=2.,
    turn_stop_angle_rad=.35, spatial_fault_angle_rad=.65,
    qualification='unqualified_manual_sim_candidate; existing_wave_binding_unchanged',
    evidence=dict(official_speed_state='mc/bin/mc_param/src/robot/lx2501_3_t2d5/speed_state.yaml',
                  official_locomotion='aimdk/docs/cn/_build/html/zh-cn/dev/Interface/control_mod/locomotion.html',
                  actual_bounds_run='desktop-20261025-033020-1c020d52', angular_bounds_rad_s=[-1.,1.]))
WAVE_PREPARATION = dict(mode='single_official_SD', max_requests_per_greet=1, timeout_seconds=6.,
    post_response_standing_seconds=1., claim='preparation_only; no_verified_yaw_reanchoring')


class DemoVelocityCore(VelocityCore):
    """Only manual demo turning changes; TTL, source and stop protections remain."""
    def __init__(self, publish, *, clock=time.monotonic):
        self.base_motion_requested = False
        super().__init__(publish, clock=clock)

    def request_velocity(self, behavior, intensity, ttl):
        # A request permanently disables the empty-source exception for this
        # executor, even if the request is subsequently rejected or cancelled.
        self.base_motion_requested = True
        return super().request_velocity(behavior, intensity, ttl)

    def resolve_velocity(self, behavior, intensity):
        if behavior in ('TURN_LEFT', 'TURN_RIGHT'):
            return (0., 0., (.4 if behavior == 'TURN_LEFT' else -.4)*intensity)
        return super().resolve_velocity(behavior, intensity)


def demo_executor(reg, publish, *, clock=time.monotonic, **kwargs):
    require(reg['demo_motion'] == DEMO_MOTION, 'Manual motion contract differs')
    core = candidate_executor(reg, publish, clock=clock, **kwargs)
    core.motion = DemoVelocityCore(publish, clock=clock)
    return core


def base_stop_basis(core):
    """Separate observed unclaimed idle from the executor's owned-motion stop.

    Never rewrite motion.state or accept this exception after a base-motion request.
    Both real sensors must advance a full second beyond the stop request while
    the existing stable-standing predicate continues to enforce fresh body data.
    """
    if core.motion.state == 'STOPPED':
        return 'owned_executor_stop_window'
    m, o, stopped_at = core.motion.mc, core.motion.odom, core.motion.stop_requested
    idle = core.command is None and core.status == 'IDLE'
    finished_stationary = (core.command is not None and core.terminal and core.status in ('COMPLETED', 'CANCELLED')
        and (core.command.behavior == 'HOLD' or core.command.behavior == 'WAVE_RIGHT' and core.wave_return_confirmed))
    if (not getattr(core.motion, 'base_motion_requested', True) and (idle or finished_stationary) and not core.motion.failures
            and m is not None and o is not None and m.source == '' and stopped_at is not None
            and min(m.received, o.received)-stopped_at >= 1. and core.standing_ready()):
        return ('official_unclaimed_idle_standing_after_stop' if idle else
                'official_unclaimed_completed_stationary_skill_after_stop')
    return None


class StandPreparation:
    """One nonblocking SD call; END cancels the wait, never dispatches a wave."""
    def __init__(self, client, request_factory, serialize, record, *, clock=time.monotonic):
        self.client, self.request_factory, self.serialize, self.record = client, request_factory, serialize, record
        self.clock = clock
        self.future = None
        self.started = self.responded = None
        self.request = None

    def start(self, core, request):
        require((core.command is None or core.terminal) and base_stop_basis(core) and core.standing_ready()
                and core.player is not None and core.player[1] == 0
                and core.query_state == 1 and self.clock()-core.query_at <= .5,
                'SD preparation requires stopped body and idle preset player')
        require(self.request is None, 'SD preparation already active')
        self.request = dict(request)
        self.started, self.responded, self.future = self.clock(), None, None

    def poll(self, core):
        require(self.request is not None, 'No SD preparation')
        now = self.clock()
        require(now-self.started < WAVE_PREPARATION['timeout_seconds'], 'SD preparation timed out')
        if self.future is None and self.responded is None:
            if not self.client.service_is_ready():
                return False
            message = self.request_factory()
            self.future = self.client.call_async(message)
            self.record('manual_sd_request', request=self.request, message=self.serialize(message))
        if self.future is not None and self.responded is None and self.future.done():
            response = self.future.result()
            self.record('manual_sd_response', request=self.request,
                        response=self.serialize(response) if response is not None else None)
            require(response is not None and response.response.header.code == 0
                    and response.response.status.value == 1, 'Official SD rejected or unsuccessful')
            self.responded = self.clock()
        m, o = core.motion.mc, core.motion.odom
        ready = (self.responded is not None and m is not None and o is not None
                 and min(m.received, o.received) >= self.responded+1. and core.standing_ready())
        if ready:
            self.record('manual_sd_standing', request=self.request, responded_at=self.responded,
                        mc_received=m.received, odom_received=o.received)
            self.request = None
        return ready

    def cancel(self):
        if self.request is not None:
            if self.future is not None:
                self.future.cancel()
            self.record('manual_sd_wait_cancelled', request=self.request,
                        note='local_wait_only; an already-sent SD cannot be recalled; no wave dispatched')
        self.request = self.future = None


class ManualDispatcher:
    """No learned decisions: fixed user-request mapping, no queue and no fallback."""
    def __init__(self, core, *, clock=time.monotonic, deadline=math.inf, reset_after_end=None, wave_preparation=None):
        self.core, self.clock, self.deadline = core, clock, deadline
        self.reset_after_end = reset_after_end
        self.sequence = 0
        self.task = None
        self.ending = False
        self.end_reset = False
        self.reason = None
        self.turn_target_stop = None
        self.wave_preparation, self.pending_wave = wave_preparation, None

    @property
    def busy(self):
        return self.ending or self.pending_wave is not None or self.core.command is not None and not self.core.terminal

    def submit(self, request):
        require(type(request['sequence']) is int and request['sequence'] > self.sequence, 'Replayed manual command')
        require(request['task'] in (*TASK_SKILLS, 'END_INTERACTION'), 'Unsupported manual task')
        self.sequence, self.task = request['sequence'], request['task']
        if self.task == 'END_INTERACTION':
            if self.wave_preparation is not None:
                self.wave_preparation.cancel()
            self.pending_wave = None
            self.ending = True
            self.end_reset = False
            if self.core.command is not None and not self.core.terminal:
                self.core.cancel('manual_end_interaction', protection=False)
            else:
                self.core.motion.request_stop('manual_end_interaction')
            return True
        if self.busy:
            self.reason = 'busy'
            return False
        behavior = TASK_SKILLS[self.task]
        require(self.deadline-self.clock() >= (34. if behavior == 'WAVE_RIGHT' else 18.), 'Insufficient action/stop budget')
        if behavior == 'WAVE_RIGHT' and self.wave_preparation is not None:
            self.wave_preparation.start(self.core, request)
            self.pending_wave = dict(request)
            self.reason = None
            return True
        accepted = self.core.submit(SkillCommand(self.sequence, request['request_id'], behavior, 'START', 1., self.clock()))
        if accepted:
            self.turn_target_stop = None
        self.reason = None if accepted else 'executor_preconditions'
        return accepted

    def state(self):
        core = self.core
        require(not core.motion.failures and core.status != 'UNKNOWN', 'Execution protection: '+str(core.motion.failures or core.reason))
        if self.pending_wave is not None and self.wave_preparation.poll(core):
            request, self.pending_wave = self.pending_wave, None
            require(core.submit(SkillCommand(request['sequence'], request['request_id'], 'WAVE_RIGHT', 'START', 1., self.clock())),
                    'Wave preconditions changed after SD preparation')
            self.turn_target_stop = None
        if core.command is not None and not core.terminal:
            if core.distance > .5 or abs(core.signed_angle) > .65:
                core.cancel('manual_spatial_limit')
                raise RuntimeError('Manual action exceeded spatial limit')
            direction = {'TURN_LEFT': 1, 'TURN_RIGHT': -1}.get(core.command.behavior)
            if direction is not None and core.stop_at is None and core.signed_angle*direction >= .35:
                self.turn_target_stop = dict(remote_monotonic=self.clock(), threshold_rad=.35,
                    measured_angle_rad=core.signed_angle, resolved_velocity=list(core.resolved_velocity))
                core.motion.request_stop('manual_turn_target')
                # The original tick consumes this stop marker; it never renews
                # velocity and will preserve the actual direction/stop scoring.
                core.stop_at = self.clock()
        if self.ending:
            if (core.command is None or core.terminal) and base_stop_basis(core) and core.standing_ready():
                if self.reset_after_end is not None and not self.end_reset:
                    # Preserve the old executor's actual terminal record (an early
                    # cancel can be FAILED/no motion). A new stopped executor must
                    # reacquire fresh standing feedback before admitting another task.
                    self.core = core = self.reset_after_end()
                    self.end_reset = True
                else:
                    self.ending = False
                    self.reason = 'end_confirmed'
        elif core.status == 'FAILED':
            raise RuntimeError('Action failed: '+core.reason)
        ready = not self.busy and core.standing_ready()
        phase = 'stopping_action' if self.ending else 'executing' if not ready else 'waiting_task'
        message = ('正在结束互动' if self.ending else '正在准备挥手' if self.pending_wave is not None else '正在执行动作' if not ready else
                   '动作未获准，请重试' if self.reason == 'executor_preconditions' else
                   '互动已结束' if self.reason == 'end_confirmed' else '请选择动作')
        return dict(sequence=self.sequence, task=self.task, busy=not ready, phase=phase, message=message,
                    measured_motion=core.moved, execution=clean(dict(core.summary(), demo_motion=DEMO_MOTION,
                        resolved_velocity=core.resolved_velocity, turn_target_stop=self.turn_target_stop,
                        base_motion_requested=core.motion.base_motion_requested, base_stop_basis=base_stop_basis(core))), rejection=self.reason,
                    wave_preparing=self.pending_wave is not None)


def execute(directory):
    require(os.environ.get('ROS_DOMAIN_ID') == '47' and os.environ.get('ROS_LOCALHOST_ONLY') == '1', 'Isolated SIM ROS required')
    directory = Path(directory)
    reg = read_json(directory/'demo_registration.json')
    require(reg.get('mode') == 'manual_demo' and reg.get('brain_controlled') is False
            and reg.get('simulation_only') is True and reg.get('qualification_only') is True, 'Manual SIM registration required')
    require(reg['skill_profile'] == candidate_skill_profile(wave_enabled=True)
            and reg['interaction_profile'] == interaction_profile(wave_enabled=True)
            and reg['intensity'] == 1. and reg['demo_motion'] == DEMO_MOTION
            and reg['wave_preparation'] == WAVE_PREPARATION, 'Manual skill contract differs')
    from . import x2_preset_identity as identity
    from .x2_skills import official_wave_poses
    binding = identity.verify_build(identity.LINUX_ROOT)
    require(binding == reg['skill_profile']['wave_contract']['helper_binding']
            and binding == read_json(directory/'preset_compat/binding.json'), 'Wave helper binding differs')
    prepared = identity.read(identity.LINUX_ROOT/'prepared.json')
    for name in ('x2_preset_identity.py', 'x2_preset_compat.py', 'x2_preset_helper.py'):
        require(identity.sha(Path(__file__).with_name(name)) == prepared['tool_files']['malecns_lab/experiments/'+name], 'Helper source differs')
    poses = official_wave_poses(identity.LINUX_ROOT.parent)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from aimdk_msgs.msg import McCommonState, McLocomotionVelocity, McInputAction, JointStateArray
    from aimdk_msgs.srv import SetMcInputSource, GetMcPresetMotionState, SetMcAction
    from nav_msgs.msg import Odometry
    started = time.monotonic()
    seconds = read_json(directory/'demo_remaining.json')['seconds']
    require(30. < seconds <= 900., 'Invalid remaining session budget')
    deadline = started+seconds
    interrupted, handlers, failure = [], {}, None
    core = node = wave = None
    last_state = None
    summary = {}
    stream = (directory/'demo_events.jsonl').open('x')
    def log(kind, **fields):
        stream.write(json.dumps(clean(dict(kind=kind, remote_monotonic=time.monotonic(), **fields)), allow_nan=False)+'\n')
        stream.flush()
    sensors = SensorEvidence(log)
    def publish(command):
        msg = McLocomotionVelocity()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.source = SOURCE
        msg.forward_velocity, msg.lateral_velocity, msg.angular_velocity = command
        publisher.publish(msg)
        log('actual_mc_request', message=message_to_ordereddict(msg))
    def on_odom(msg):
        p, q, v, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear, msg.twist.twist.angular
        sensors.update('odom', message_to_ordereddict(msg),
                       lambda **stamp: core.update_odom((p.x,p.y,p.z), (q.x,q.y,q.z,q.w), (v.x,v.y,v.z), (w.x,w.y,w.z), **stamp),
                       lambda: core.motion.odom, received=time.monotonic())
    def on_mc(msg):
        m = msg.motion_status
        sensors.update('mc', message_to_ordereddict(msg),
                       lambda **stamp: core.update_mc(msg.input_source.name, msg.action_info.action_desc, msg.body_status.value,
                           msg.action_info.status.value, player_state=m.player_state.value, motion=m.motion,
                           area=m.control_area.value, motion_type=m.type.value, **stamp),
                       lambda: core.motion.mc, received=time.monotonic())
    def on_joints(msg):
        def update(**stamp):
            ok = core.update_joints({j.name:(j.position,j.velocity) for j in msg.joints}, **stamp)
            if any(j.error_code for j in msg.joints if j.name in RIGHT_ARM):
                core.joints = None
                core.cancel('arm_joint_error')
                return False
            return ok
        sensors.update('arm', message_to_ordereddict(msg), update, None, received=time.monotonic(),
                       archive=core.command is not None and core.command.behavior == 'WAVE_RIGHT')
    def spin():
        began = time.monotonic()
        for _ in range(64):
            if time.monotonic()-began >= .008:
                break
            rclpy.spin_once(node, timeout_sec=.0001)
        if wave is not None:
            wave.poll()
        core.standing_ready()
        core.tick()
        if wave is not None:
            wave.reap_terminal()
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda sig, frame: interrupted.append(sig))
        node = Node('malecns_x2_manual_demo')
        publisher = node.create_publisher(McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        core = demo_executor(reg, publish, record=lambda event: log('execution', event=event, demo_motion=DEMO_MOTION),
                                  preset=lambda command: wave.preset(command), wave_poses=poses)
        client = node.create_client(GetMcPresetMotionState, '/aimdk_5Fmsgs/srv/GetMcPresetMotionState')
        def query(task_id):
            req = GetMcPresetMotionState.Request()
            req.request.header.stamp = node.get_clock().now().to_msg()
            req.request.task_id = task_id
            return req
        wave = WaveSession(directory, reg, core, binding, log, client, query, message_to_ordereddict)
        stand_client = node.create_client(SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')
        def stand_request():
            require(wave.boundary is None, 'Active preset helper prevents SD preparation')
            request = SetMcAction.Request()
            request.header.stamp = node.get_clock().now().to_msg()
            request.source = SOURCE
            request.command.action_desc = 'STAND_DEFAULT'
            return request
        preparation = StandPreparation(stand_client, stand_request, message_to_ordereddict, log)
        core.motion.arm()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(Odometry, '/aima/hal/odom/state', on_odom, qos)
        node.create_subscription(McCommonState, '/aima/mc/common/state', on_mc, qos)
        node.create_subscription(JointStateArray, '/aima/hal/joint/arm/state', on_joints, qos)
        source = node.create_client(SetMcInputSource, '/aimdk_5Fmsgs/srv/SetMcInputSource')
        end = min(deadline-15., started+12.)
        while not source.service_is_ready() and time.monotonic() < end and not interrupted:
            spin()
            time.sleep(.01)
        registered = False
        for action in (McInputAction.INPUTACTION_ADD, McInputAction.INPUTACTION_ENABLE):
            require(source.service_is_ready() and time.monotonic() < end and not interrupted, 'Source registration unavailable')
            req = SetMcInputSource.Request()
            req.request.header.stamp = node.get_clock().now().to_msg()
            req.action.value = action
            req.input_source.name, req.input_source.priority, req.input_source.timeout = SOURCE, 40, 1000
            future = source.call_async(req)
            response_end = min(end, time.monotonic()+2.)
            while not future.done() and time.monotonic() < response_end and not interrupted:
                spin()
                time.sleep(.01)
            response = future.result() if future.done() else None
            log('registration', action=action, response=message_to_ordereddict(response) if response else None)
            if response and response.response.header.code == 0:
                core.motion.confirm_source_registration()
                registered = True
                break
        require(registered, 'Source registration failed')
        while not core.standing_ready() and time.monotonic() < started+15. and not interrupted:
            spin()
            time.sleep(.02)
        require(core.standing_ready() and not core.motion.failures and not interrupted, 'Fresh stable standing not confirmed')
        def reset_after_end():
            nonlocal core, wave
            log('end_stop_evidence', execution=clean(core.summary()))
            wave.close_action()
            core = demo_executor(reg, publish, record=lambda event: log('execution', event=event, demo_motion=DEMO_MOTION),
                                      preset=lambda command: wave.preset(command), wave_poses=poses)
            wave = WaveSession(directory, reg, core, binding, log, client, query, message_to_ordereddict)
            core.motion.arm()
            core.motion.confirm_source_registration()
            return core
        dispatcher = ManualDispatcher(core, deadline=deadline-14., reset_after_end=reset_after_end,
                                      wave_preparation=preparation)
        atomic(directory/'demo_ready.json', dict(run_id=reg['run_id'], session_id=reg['session_id']))
        while time.monotonic() < deadline-14. and not interrupted:
            began = time.monotonic()
            if (directory/'demo_emergency.json').exists():
                raise InterruptedError('independent_emergency_stop')
            spin()
            path = directory/'demo_command.json'
            if path.exists():
                request = read_json(path)
                require((request['run_id'], request['session_id']) == (reg['run_id'], reg['session_id'])
                        and request['mode'] == 'manual_demo' and request['brain_controlled'] is False, 'Foreign manual request')
                if request['sequence'] > dispatcher.sequence:
                    accepted = dispatcher.submit(request)
                    log('manual_admission', request=request, accepted=accepted, reason=dispatcher.reason)
            state = dispatcher.state()
            # Persist state changes only; raw sensor/command evidence stays in JSONL.
            signature = (state['sequence'], state['phase'], state['busy'], core.execution_id, core.status,
                         state['rejection'], core.stop_at, state['wave_preparing'])
            if signature != last_state:
                last_state = signature
                state.update(run_id=reg['run_id'], session_id=reg['session_id'], mode='manual_demo', brain_controlled=False)
                atomic(directory/'demo_state.json', state)
                log('manual_state', state=state)
            time.sleep(max(0., .02-(time.monotonic()-began)))
        failure = 'manual_stop' if interrupted else 'session_budget'
    except BaseException as error:
        failure = type(error).__name__+':'+str(error)
        try:
            log('exception', reason=failure)
        except BaseException:
            pass
    finally:
        if 'preparation' in locals():
            try:
                preparation.cancel()
            except BaseException:
                pass
        if core is not None:
            try:
                if core.command and not core.terminal:
                    core.cancel(failure or 'demo_exit', protection=False)
                else:
                    core.motion.request_stop(failure or 'demo_exit')
            except BaseException:
                pass
            end = min(deadline, time.monotonic()+12.)
            while time.monotonic() < end and (base_stop_basis(core) is None or
                    core.command is not None and core.command.behavior == 'WAVE_RIGHT' and not core.terminal):
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
            summary = clean(dict(core.summary(), demo_motion=DEMO_MOTION, resolved_velocity=core.resolved_velocity,
                                 base_motion_requested=core.motion.base_motion_requested,
                                 turn_target_stop=dispatcher.turn_target_stop if 'dispatcher' in locals() else None))
            stop_basis = base_stop_basis(core)
            summary.update(base_stop_confirmed=stop_basis is not None, base_stop_basis=stop_basis,
                wave_stop_confirmed=core.command is None or core.command.behavior != 'WAVE_RIGHT'
                    or core.wave_return_confirmed or core.terminal and core.wave_dispatch_at is None)
        if wave is not None:
            try:
                wave.close_action()
            except BaseException as error:
                summary['wave_stop_confirmed'] = False
                failure = (failure or '')+';helper_cleanup:'+str(error)
        summary.update(reason=failure, wall_seconds=time.monotonic()-started, mode='manual_demo', brain_controlled=False,
                       wave_preparation=WAVE_PREPARATION)
        try:
            atomic(directory/'demo_summary.json', summary)
        finally:
            stream.close()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return 0 if summary.get('base_stop_confirmed') and summary.get('wave_stop_confirmed') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sim-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute or not args.sim_only or args.output is None:
        parser.error('requires --execute --sim-only --output')
    return execute(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
