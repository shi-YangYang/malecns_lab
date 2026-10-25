"""Explicit manual official-skill demonstration; no model or brain is loaded."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import re
import shlex
import signal
import sys
import threading
import time
import uuid

from . import desktop
from .behavior import Commands as BehaviorCommands, BehaviorRuntime, TRANSPORT as BEHAVIOR_TRANSPORT
from .skills import SkillRuntime, ASSETS as SHARED_ASSETS, numerical_exclusion
from ...experiments.x2_demo import DEMO_MOTION, WAVE_PREPARATION
from ...control.x2_skill_profile import candidate_skill_profile
from ...control.x2_interaction import interaction_profile
from ...learning.x2_behavior_session import TASKS, atomic, read_json, require

ROOT = desktop.ROOT
WALL_SECONDS = 900.
TRANSPORT = (*BEHAVIOR_TRANSPORT, 'experiments/x2_demo.py')
ASSETS = (*SHARED_ASSETS, DEMO_MOTION['evidence']['official_speed_state'],
          DEMO_MOTION['evidence']['official_locomotion'])


class Commands(BehaviorCommands):
    """One reserved action slot, with an independent END and emergency channel."""
    def __init__(self, stream):
        super().__init__(stream)
        self.lock = threading.Lock()
        self.ready = self.busy = False
        self.sequence = 0
        self.next_action = None
        self.notifications = queue.Queue(maxsize=32)

    def task(self, task):
        with self.lock:
            reason = None
            if task != 'END_INTERACTION' and (not self.ready or self.busy):
                reason = 'busy' if self.busy else 'not_ready'
            if reason:
                self.notifications.put_nowait(dict(task=task, reason=reason))
                return False
            self.sequence += 1
            self.busy = True
            # END replaces an as-yet-unsent ordinary action; never waits behind it.
            self.next_action = dict(sequence=self.sequence, task=task, request_id=uuid.uuid4().hex)
            return True

    def take(self):
        with self.lock:
            value, self.next_action = self.next_action, None
            return value

    def remote_state(self, state):
        with self.lock:
            self.ready = True
            if state['sequence'] >= self.sequence:
                self.busy = state['busy']

    def read(self):
        try:
            while not self.stop.is_set():
                line = self.stream.readline(4097)
                if not line:
                    self.request_stop('desktop_control_pipe_closed')
                    return
                require(len(line) <= 4096 and line.endswith('\n'), 'Invalid desktop command length')
                value = json.loads(line)
                if value == {'command': 'stop'}:
                    self.request_stop('user_emergency_stop')
                elif value == {'command': 'continue'}:
                    self.pending.put_nowait('continue')
                elif (set(value) == {'command', 'task'} and value['command'] == 'task'
                      and value['task'] in TASKS):
                    self.task(value['task'])
                elif (set(value) == {'command', 'run_id', 'nonce'} and value['command'] == 'clock'
                      and value['run_id'] == self.clock_run_id and isinstance(value['nonce'], str)
                      and re.fullmatch('[a-f0-9]{32}', value['nonce'])):
                    try:
                        self.clock_requests.put_nowait(value['nonce'])
                    except queue.Full:
                        pass
                else:
                    raise ValueError('Unknown desktop command')
        except (OSError, ValueError, TypeError, queue.Full):
            self.request_stop('desktop_command_error')


class DemoRuntime(SkillRuntime):
    def __init__(self, output, root, commands, emit):
        super().__init__(output, root, commands, emit, ui_timeout=300., wall_timeout=480.)
        self.run_id, self.session_id = output.name, uuid.uuid4().hex
        self.skill_profile = candidate_skill_profile(wave_enabled=True)
        self.interaction = interaction_profile(wave_enabled=True)
        self.last_remote_state = None
        self.bridge_started = False
        self.remote_summary = None

    def deadline(self, cleanup=False):
        return self.started + WALL_SECONDS - (0. if cleanup else 35.)

    def status(self, state, message, **fields):
        if state == "awaiting_simulation":
            message = "请先 Pause、Reset，确认直立暂停后 Run，再点击开始"
        super().status(state, message, **{**fields, 'mode': 'manual_demo', 'brain_controlled': False,
                                       'host_clock_id': self.commands.clock_id})

    def guard(self):
        super().guard()
        if self.commands.consume_continue():
            require(self.state == 'awaiting_simulation', 'Continue is only valid during UI preparation')
            atomic(self.output/'continue.json', {'command': 'continue'})
        if self.bridge_started:
            request = self.commands.take()
            if request is not None:
                packet = dict(request, run_id=self.run_id, session_id=self.session_id,
                              mode='manual_demo', brain_controlled=False)
                atomic(self.output/'demo_command.json', packet)
                with (self.output/'manual_requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(packet)+'\n')
                self.status('running', '正在结束互动' if request['task'] == 'END_INTERACTION' else '正在执行动作',
                            phase='stopping_action' if request['task'] == 'END_INTERACTION' else 'executing',
                            request_pending=True, task=request['task'])
        while not self.commands.notifications.empty():
            rejected = self.commands.notifications.get_nowait()
            self.status(self.state, '当前动作尚未结束，请稍后再试' if rejected['reason'] == 'busy' else '仿真尚未准备好',
                        phase='executing' if self.commands.busy else 'preparing',
                        request_pending=self.commands.busy, task_rejected=rejected)

    def prepare_transport(self):
        manifest = {}
        for name in TRANSPORT:
            content = (self.root/'src/malecns_lab'/name).read_bytes()
            target = self.output/'transport_src/malecns_lab'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        (self.output/'host_source.py').write_bytes(Path(__file__).read_bytes())
        self.protocol.atomic(self.output/'transport_manifest.json', manifest)
        self.protocol.atomic(self.output/'official_assets.json', dict(sha256={
            name: hashlib.sha256((self.root/'data/x2_sim/packages'/name).read_bytes()).hexdigest() for name in ASSETS}))
        BehaviorRuntime.prepare_wave_qualification(self)
        self.protocol.atomic(self.output/'demo_registration.json', dict(
            run_id=self.run_id, session_id=self.session_id, mode='manual_demo', brain_controlled=False,
            simulation_only=True, qualification_only=True, skill_profile=self.skill_profile,
            interaction_profile=self.interaction, wall_seconds=WALL_SECONDS,
            intensity=1., forward_velocity_m_s=.2, turn_velocity_rad_s=.4, demo_motion=DEMO_MOTION,
            wave_preparation=WAVE_PREPARATION))

    def start_demo_bridge(self):
        # The remote budget is the remaining session budget, never a new 900 s window.
        atomic(self.output/'demo_remaining.json', dict(seconds=max(0., self.deadline()-time.monotonic())))
        command = ('source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && '
                   'PYTHONPATH='+shlex.quote(self.remote+'/transport_src')+':$PYTHONPATH '
                   'ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_demo '
                   '--execute --sim-only --output '+shlex.quote(self.remote))
        self.bridge = self.spawn(self.docker+['exec', desktop.CONTAINER, 'bash', '-lc', command], 'demo-bridge.log')
        end = min(self.deadline(), time.monotonic()+18.)
        while not (self.output/'demo_ready.json').exists() and time.monotonic() < end:
            self.guard()
            require(self.bridge.poll() is None, 'Demo bridge startup failed; see demo-bridge.log')
            time.sleep(.05)
        ready = read_json(self.output/'demo_ready.json')
        require((ready['run_id'], ready['session_id']) == (self.run_id, self.session_id), 'Foreign demo readiness')
        self.bridge_started = True
        self.status('running', '请选择动作', phase='waiting_task', request_pending=False)
        self.commands.remote_state(dict(sequence=0, busy=False))

    def shutdown_demo(self, reason):
        try:
            atomic(self.output/'demo_emergency.json', dict(reason=reason))
        except BaseException as error:
            self.errors.append('emergency_write:'+repr(error))
        if self.bridge is not None and self.bridge.poll() is None:
            try:
                self.command(self.docker+['exec', desktop.CONTAINER, 'pkill', '-TERM', '-f',
                             '^python3 -m malecns_lab.experiments.x2_demo '], timeout=2., cleanup=True, check=False)
            except BaseException as error:
                self.errors.append('bridge_signal:'+repr(error))
            end = min(self.deadline(True)-17., time.monotonic()+12.)
            while self.bridge.poll() is None and time.monotonic() < end:
                time.sleep(.05)
        try:
            result = read_json(self.output/'demo_summary.json')
            self.remote_summary = result
            self.stop_confirmed = result['base_stop_confirmed'] and result['wave_stop_confirmed']
        except BaseException:
            self.stop_confirmed = False if self.bridge is not None else None
        self.cleanup()

    def execute_demo(self):
        self.log_directory = self.output
        outcome = dict(state='failed', reason='startup_incomplete', mode='manual_demo', brain_controlled=False)
        try:
            self.preflight()
            self.prepare_transport()
            self.start_simulator()
            self.start_demo_bridge()
            while True:
                self.guard()
                require(self.bridge.poll() is None, 'Demo bridge ended; see demo_summary.json')
                path = self.output/'demo_state.json'
                if path.exists():
                    state = read_json(path)
                    if state != self.last_remote_state:
                        require((state['run_id'], state['session_id']) == (self.run_id, self.session_id), 'Foreign demo state')
                        self.last_remote_state = state
                        self.commands.remote_state(state)
                        self.motion_started |= state.get('measured_motion', False)
                        # An old idle mailbox cannot clear a newly reserved task slot.
                        if state['sequence'] >= self.commands.sequence:
                            self.status('running', state['message'], phase=state['phase'], request_pending=state['busy'],
                                        task=state.get('task'), execution=state.get('execution'), sequence=state['sequence'])
                time.sleep(.05)
        except BaseException as error:
            outcome.update(state='cancelled' if isinstance(error, desktop.StopRequested) else 'failed',
                           reason=type(error).__name__+':'+str(error))
        finally:
            try:
                self.shutdown_demo(outcome['reason'])
            except BaseException as error:
                self.errors.append('shutdown:'+repr(error))
                self.cleanup()
            if self.default_context is not None:
                try:
                    after = self.command(['docker', 'context', 'show'], timeout=2., cleanup=True).stdout.strip()
                    require(after == self.default_context, 'default_context_changed')
                except BaseException as error:
                    self.errors.append('context_check:'+repr(error))
            if self.errors or not self.resources_stopped or self.bridge is not None and not self.stop_confirmed:
                outcome['state'] = 'failed'
            if self.remote_summary and (self.remote_summary.get('failures') or self.remote_summary.get('status') == 'UNKNOWN'):
                outcome['state'] = 'failed'
                outcome['remote_failure'] = self.remote_summary
            outcome.update(resources_stopped=self.resources_stopped, stop_confirmed=self.stop_confirmed,
                           errors=self.errors, wall_seconds=time.monotonic()-self.started)
            atomic(self.output/'demo_host_summary.json', outcome)
            self.status(outcome['state'], '演示已结束' if outcome['state'] != 'failed' else '演示已停止，请查看记录',
                        reason=outcome['reason'], request_pending=False)
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sim-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(mode='manual_demo', brain_controlled=False, brain_loaded=False,
                              simulator_started=False, tasks=TASKS, wall_seconds=WALL_SECONDS)))
        return 0
    if not args.sim_only or args.output is None:
        parser.error('--execute requires --sim-only --output')
    directory = desktop.output_directory(args.output)
    commands = Commands(sys.stdin)
    handlers = {sig: signal.signal(sig, lambda sig, frame: commands.request_stop('signal:'+str(sig)))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    runtime = None
    try:
        with desktop.desktop_lock(ROOT), numerical_exclusion(ROOT):
            directory.mkdir(parents=True, exist_ok=False)
            with (directory/'lifecycle.jsonl').open('x') as stream:
                events = desktop.Events(sys.stdout, stream, commands.request_stop)
                commands.start_clock_replies(directory.name, events.emit)
                commands.start()
                runtime = DemoRuntime(directory, ROOT, commands, events.emit)
                result = runtime.execute_demo()
                events.drain()
                return 0 if result['state'] == 'cancelled' else 1
    except BaseException as error:
        print(json.dumps(dict(schema=desktop.SCHEMA, type='status', state='failed', message='无法准备演示：'+str(error),
                              mode='manual_demo', brain_controlled=False, output=str(directory),
                              run_id=runtime.run_id if runtime else None,
                              simulator_ready=runtime.simulator_ready if runtime else False,
                              motion_started=runtime.motion_started if runtime else False,
                              stop_confirmed=runtime.stop_confirmed if runtime else None,
                              resources_stopped=runtime.resources_stopped if runtime else True), ensure_ascii=False), flush=True)
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
