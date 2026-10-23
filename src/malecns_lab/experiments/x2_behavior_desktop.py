"""Bounded desktop qualification of a fixed new X2 behavior checkpoint.

No model, VM or ROS is started by import/preview. Default desktop startup stays
idle. An operator selects a real candidate in host-only configuration; the
renderer sends only five task enums and independent emergency stop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import sys
import threading
import time
import uuid

from . import x2_desktop as legacy
from .x2_skill_host import SkillRuntime, ASSETS, numerical_exclusion
from ..control.x2_skill_profile import candidate_skill_profile, validate_candidate_skill_profile
from ..learning.x2_behavior_session import TASKS, atomic, read_json, require

ROOT = legacy.ROOT
WALL_SECONDS = 600.
VM_SECONDS = 240.
CLEANUP_SECONDS = 35.
MAX_DECISIONS = 50
TRANSPORT = tuple(dict.fromkeys((*legacy.TRANSPORT,'control/x2_behavior.py','control/x2_skills.py',
    'control/x2_skill_profile.py',
    'learning/__init__.py','learning/x2_behavior_session.py','experiments/x2_behavior_bridge.py')))


class Commands(legacy.Commands):
    def __init__(self, stream):
        super().__init__(stream)
        self.tasks = queue.Queue(maxsize=50)

    def read(self):
        try:
            while True:
                line = self.stream.readline(4097)
                if not line:
                    self.request_stop('desktop_control_pipe_closed')
                    return
                require(len(line) <= 4096 and line.endswith('\n'), 'Invalid command size')
                value = json.loads(line)
                if value == {'command':'continue'}:
                    self.pending.put_nowait('continue')
                elif value == {'command':'stop'}:
                    self.request_stop('user_emergency_stop')
                elif isinstance(value,dict) and set(value) == {'command','task'} and value['command'] == 'task' and value['task'] in TASKS:
                    self.tasks.put_nowait((value['task'],time.monotonic()))
                else:
                    raise ValueError('Invalid desktop command')
        except BaseException as error:
            self.request_stop('desktop_command_error:'+str(error))


def candidate(root, override=None):
    """Resolve only host configuration/CLI, never a renderer path."""
    root = Path(root).resolve()
    if override is None:
        config = read_json(root/'data/x2_behavior_desktop.json')
        require(set(config) == {'checkpoint','mode'} and config['mode'] == 'qualification',
                'Desktop config must explicitly select qualification mode')
        override = config['checkpoint']
    path = Path(override)
    require(not path.is_absolute() or path.is_relative_to(root/'data'), 'Checkpoint must be inside project data')
    path = path if path.is_absolute() else root/path
    canonical = path.resolve(strict=True)
    require(canonical.is_relative_to((root/'data').resolve()) and canonical.is_dir() and canonical == path.absolute(),
            'Checkpoint path escaped data or used a symlink')
    from ..learning.x2_behavior_training import inspect_checkpoint
    meta = inspect_checkpoint(canonical)
    require(meta['policy']['schema'] == 'malecns_lab.x2_behavior_policy.v1', 'New behavior checkpoint required')
    return canonical,meta


class BehaviorRuntime(SkillRuntime):
    def __init__(self, output, root, commands, emit, checkpoint_path, metadata):
        super().__init__(output,root,commands,emit)
        self.checkpoint_path,self.metadata = checkpoint_path,metadata
        self.run_id,self.session_id = output.name,'session-'+uuid.uuid4().hex
        self.task_sequence = 0
        self.tasks_open = False
        self.last_phase_value = None
        self.rss_limit = 0
        self.resource_stream = None
        self.peak_rss = 0

    def deadline(self, cleanup=False):
        end = self.started+WALL_SECONDS
        if self.vm_started is not None:
            end = min(end,self.vm_started+VM_SECONDS)
        return end-(0. if cleanup else CLEANUP_SECONDS)

    def guard(self):
        super().guard()
        import psutil
        if self.host is not None and self.host.poll() is None:
            rss = psutil.Process(self.host.pid).memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            if self.resource_stream is not None:
                self.resource_stream.write(f'{time.monotonic()-self.started},{rss},{psutil.virtual_memory().available}\n')
                self.resource_stream.flush()
            require(rss <= self.rss_limit,'Brain RSS budget exceeded')
        if self.commands.consume_continue():
            require(self.state == 'awaiting_simulation','Unexpected continue command')
            atomic(self.output/'continue.json',{'command':'continue'})
        while True:
            try:
                task,received = self.commands.tasks.get_nowait()
            except queue.Empty:
                break
            require(self.tasks_open,'Tasks are not enabled before bridge readiness')
            require(self.task_sequence < 50,'Task request budget reached')
            self.task_sequence += 1
            request = dict(run_id=self.run_id,session_id=self.session_id,sequence=self.task_sequence,
                request_id='request-'+uuid.uuid4().hex,task=task,host_received_monotonic=received)
            atomic(self.output/'task.json',request)
            with (self.output/'task_requests.jsonl').open('a') as stream:
                stream.write(json.dumps(request)+'\n')
            self.status('running','任务已送入脑模型',task=task,request_id=request['request_id'],qualification_only=True)

    def prepare(self):
        from ..learning.x2_behavior_training import source_inventory
        from ..learning.resource_guard import resource_budget
        import psutil
        resources = resource_budget(psutil.virtual_memory().available,rss_bytes=6*legacy.GIB,reserve_bytes=legacy.GIB)
        require(resources['can_launch'], 'Insufficient system reserve')
        self.rss_limit = resources['effective_rss_limit_bytes']
        self.resource_stream = (self.output/'behavior_resources.csv').open('x')
        self.resource_stream.write('wall_seconds,brain_rss_bytes,system_available_bytes\n')
        self.logs.append(self.resource_stream)
        sources = source_inventory(self.root)
        for name in (*TRANSPORT,'learning/x2_behavior_telemetry.py','learning/x2_telemetry.py',
                     'experiments/x2_behavior_desktop.py','experiments/x2_skill_host.py','experiments/x2_desktop.py'):
            sources[name] = hashlib.sha256((self.root/'src/malecns_lab'/name).read_bytes()).hexdigest()
        for name,expected in sources.items():
            content = (self.root/'src/malecns_lab'/name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == expected,'Source snapshot changed')
            target = self.output/'sources/malecns_lab'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
        for name in TRANSPORT:
            target = self.output/'transport_src/malecns_lab'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((self.root/'src/malecns_lab'/name).read_bytes())
        atomic(self.output/'transport_manifest.json',{name:sources[name] for name in TRANSPORT})
        atomic(self.output/'official_assets.json',dict(sha256={name:hashlib.sha256((self.root/'data/x2_sim/packages'/name).read_bytes()).hexdigest() for name in ASSETS}))
        identity = dict(run_id=self.run_id,session_id=self.session_id)
        skill_profile = candidate_skill_profile()
        registration = dict(**identity,root=str(self.root),output=str(self.output),parent_pid=os.getpid(),
            checkpoint_path=str(self.checkpoint_path),checkpoint_metadata={'checkpoint_id':self.metadata['checkpoint_id'],
                'policy':{key:self.metadata['policy'][key] for key in ('initialization','backward')
                          if key in self.metadata['policy']}},
            source_sha256=sources,rss_limit_bytes=self.rss_limit,system_reserve_bytes=legacy.GIB,
            worker_wall_seconds=WALL_SECONDS-CLEANUP_SECONDS,max_decisions=MAX_DECISIONS,
            wall_seconds=WALL_SECONDS,vm_seconds=VM_SECONDS,cleanup_seconds=CLEANUP_SECONDS,
            qualification_only=True,wave_enabled=False,simulation_only=True,skill_profile=skill_profile)
        atomic(self.output/'behavior_registration.json',registration)
        atomic(self.output/'bridge_registration.json',dict(**identity,qualification_only=True,wave_enabled=False,
            bridge_wall_seconds=VM_SECONDS-CLEANUP_SECONDS,skill_profile=skill_profile))
        self.status('preparing','正在加载候选脑模型',qualification_only=True,checkpoint_id=self.metadata['checkpoint_id'])
        environment = dict(self.environment,PYTHONPATH=str(self.root/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
        self.host = self.spawn([str(self.root/'.venv-train/bin/python'),'-u','-m',
            'malecns_lab.experiments.x2_behavior_desktop','--worker',str(self.output)],'brain-worker.log',environment)
        while not (self.output/'brain_ready.json').exists():
            self.guard()
            require(self.host.poll() is None,'Candidate brain initialization failed; see brain-worker.log')
            time.sleep(.1)
        ready = read_json(self.output/'brain_ready.json')
        require(ready['checkpoint_id'] == self.metadata['checkpoint_id'],'Worker selected another checkpoint')

    def start_behavior_bridge(self):
        command = ('source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && '
            'PYTHONPATH='+shlex.quote(self.remote+'/transport_src')+':$PYTHONPATH '
            'ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_behavior_bridge '
            '--execute --sim-only --output '+shlex.quote(self.remote))
        self.bridge = self.spawn(self.docker+['exec',legacy.CONTAINER,'bash','-lc',command],'behavior-bridge.log')
        end = min(self.deadline(),time.monotonic()+15.)
        while not (self.output/'bridge_ready.json').exists() and time.monotonic()<end:
            self.guard()
            require(self.bridge.poll() is None,'Behavior bridge startup failed')
            time.sleep(.1)
        require((self.output/'bridge_ready.json').exists(),'Bridge readiness expired')
        validate_candidate_skill_profile(read_json(self.output/'bridge_ready.json').get('skill_profile'))
        self.tasks_open = True
        self.status('running','请选择互动任务',qualification_only=True,wave_enabled=False,phase='waiting_task')

    def phase_status(self, phase):
        label = {'computing':'脑模型正在计算','executing':'正在执行脑输出','rejected':'脑输出被拒绝',
                 'waiting_task':'等待互动任务'}.get(phase['phase'],'正在准备')
        self.status('running',label,**{**phase,'qualification_only':True})

    def shutdown_behavior(self, reason):
        self.tasks_open = False
        try:
            atomic(self.output/'emergency.json',dict(reason=reason,run_id=self.run_id,session_id=self.session_id,
                host_monotonic=time.monotonic(),independent_protection=True))
        except BaseException as error:
            self.errors.append('emergency_write:'+repr(error))
        if self.bridge is not None and self.bridge.poll() is None:
            try:
                # Signal only our fixed entrypoint in our exclusively owned container.
                self.command(self.docker+['exec',legacy.CONTAINER,'pkill','-TERM','-f',
                    '^python3 -m malecns_lab.experiments.x2_behavior_bridge '],timeout=2.,cleanup=True,check=False)
            except BaseException as error:
                self.errors.append('bridge_signal:'+repr(error))
            end = min(self.deadline(True)-17.,time.monotonic()+12.)
            while self.bridge.poll() is None and time.monotonic()<end:
                time.sleep(.1)
        try:
            result = read_json(self.output/'bridge_summary.json')
            self.stop_confirmed = result.get('base_stop_confirmed') is True
        except BaseException:
            self.stop_confirmed = False if self.bridge is not None else None
        # Cleanup independently closes owned container, VM and child even when logging fails.
        self.cleanup()

    def execute_behavior(self):
        self.log_directory = self.output
        outcome = dict(state='failed',reason='startup_incomplete')
        try:
            self.preflight()
            self.prepare()
            self.start_simulator()
            self.start_behavior_bridge()
            while True:
                self.guard()
                require(self.bridge.poll() is None,'Behavior bridge exited')
                if self.host.poll() is not None:
                    result = read_json(self.output/'behavior_worker_result.json')
                    require(result['status'] == 'completed','Brain worker: '+result['reason'])
                    outcome.update(state='completed',reason='bounded_candidate_run_ended')
                    break
                path = self.output/'behavior_phase.json'
                if path.exists():
                    phase = read_json(path)
                    if phase != self.last_phase_value:
                        self.last_phase_value = phase
                        self.phase_status(phase)
                latest = self.output/'execution_latest.json'
                if latest.exists():
                    event = read_json(latest)['event']
                    self.motion_started = self.motion_started or (event['behavior'] != 'HOLD' and event['status'] in ('STARTED','RUNNING','COMPLETED'))
                time.sleep(.1)
        except BaseException as error:
            outcome.update(state='cancelled' if isinstance(error,legacy.StopRequested) else 'failed',
                           reason=type(error).__name__+':'+str(error))
        finally:
            try:
                self.status('stopping','正在停止并关闭仿真',qualification_only=True)
            except BaseException:
                pass
            try:
                self.shutdown_behavior(outcome['reason'])
            except BaseException as error:
                self.errors.append('shutdown:'+repr(error))
                self.resources_stopped = False
            try:
                context = self.command(['docker','context','show'],timeout=2.,cleanup=True,check=False).stdout.strip()
                if self.default_context is not None and context != self.default_context:
                    self.errors.append('default_context_changed')
            except BaseException as error:
                self.errors.append('context_check:'+repr(error))
            if self.errors or not self.resources_stopped or (self.bridge is not None and self.stop_confirmed is not True):
                outcome['state'] = 'failed'
            outcome.update(resources_stopped=self.resources_stopped,stop_confirmed=self.stop_confirmed,errors=self.errors,
                qualification_only=True,behavior_pass=None,wave_enabled=False,wall_seconds=time.monotonic()-self.started,
                peak_observed_rss_bytes=self.peak_rss,skill_profile=candidate_skill_profile())
            try:
                atomic(self.output/'behavior_host_summary.json',outcome)
            except BaseException as error:
                self.errors.append('summary_write:'+repr(error))
                outcome['state'] = 'failed'
            self.status(outcome['state'],'验证已结束' if outcome['state'] != 'failed' else '验证失败，请查看记录',
                        qualification_only=True,reason=outcome['reason'])
            if self.errors and outcome['state'] != 'failed':
                outcome['state'] = 'failed'
                try:
                    atomic(self.output/'behavior_host_summary.json',outcome)
                except BaseException:
                    pass
        return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--sim-only',action='store_true')
    p.add_argument('--qualification',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        from ..learning.x2_behavior_session import run_worker
        return run_worker(args.worker)
    if not args.execute:
        print(json.dumps(dict(mode='preview',brain_loaded=False,simulator_started=False,wave_enabled=False,
            tasks=TASKS,wall_seconds=WALL_SECONDS,vm_seconds=VM_SECONDS,max_decisions=MAX_DECISIONS,
            skill_profile=candidate_skill_profile())))
        return 0
    if not args.sim_only or not args.qualification or args.output is None:
        p.error('execute requires --sim-only --qualification --output')
    directory = legacy.output_directory(args.output)
    commands = Commands(sys.stdin)
    runtime = None
    handlers = {sig:signal.signal(sig,lambda sig,frame:commands.request_stop('signal:'+str(sig)))
                for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        # No candidate, no locks/VM/brain. A fixed candidate may qualify in sim
        # without claiming the yet-unpassed product gates P3/P5.
        checkpoint_path,metadata = candidate(ROOT,args.checkpoint)
        with legacy.desktop_lock(ROOT),numerical_exclusion(ROOT):
            directory.mkdir(parents=True,exist_ok=False)
            with (directory/'lifecycle.jsonl').open('x') as stream:
                def emit(value):
                    line = json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
                    stream.write(line)
                    stream.flush()
                    print(line,end='',flush=True)
                commands.start()
                runtime = BehaviorRuntime(directory,ROOT,commands,emit,checkpoint_path,metadata)
                result = runtime.execute_behavior()
        return 0 if result['state'] in ('completed','cancelled') else 1
    except BaseException as error:
        # This outer error happens before Runtime owns any resources (or after
        # its mandatory finally). Do not claim teardown if it failed there.
        sealed = directory/'desktop_session.json'
        if not sealed.exists():
            print(json.dumps(dict(schema=legacy.SCHEMA,type='status',state='failed',message='无法准备新脑模型：'+str(error),
                output=str(directory),run_id=None,simulator_ready=False,motion_started=False,stop_confirmed=None,
                resources_stopped=True if runtime is None else runtime.resources_stopped),ensure_ascii=False),flush=True)
        return 1
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    raise SystemExit(main())
