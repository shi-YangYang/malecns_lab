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
import shutil
import sys
import threading
import time
import uuid

from . import desktop as legacy
from . import sim_configuration
from .skills import SkillRuntime, ASSETS, AFTER_UI_RESERVE, numerical_exclusion
from ...control.x2_behavior import ExecutionEvent, ExecutionStatus
from ...control.x2_skill_profile import (candidate_skill_profile, validate_candidate_skill_profile,
    skill_profile_by_id, SIM_FIXED_ADVANCE_PROFILE_ID, FULL_ACTION_PROFILE_IDS)
from ...control.x2_interaction import interaction_profile, validate_interaction_profile
from ...learning.x2_behavior_session import (TASKS, atomic, read_json, require,
    checkpoint_registration_metadata, SESSION_PROFILES, session_profile_limits, TASK_REQUEST_SECONDS,
    SCHEMA as TRANSPORT_SCHEMA)
from ...learning.x2_frozen_history_policy import TURN_PROFILES, WAVE_PROFILES

ROOT = legacy.ROOT
WALL_SECONDS = 900.
VM_SECONDS = 480.
CLEANUP_SECONDS = 35.
MAX_DECISIONS = 160
MIN_BRAIN_BUDGET = 2 * legacy.GIB
TRANSPORT = tuple(dict.fromkeys((*legacy.TRANSPORT,'control/x2_behavior.py','control/x2_skills.py',
    'control/x2_skill_profile.py','control/x2_interaction.py',
    'learning/__init__.py','learning/x2_behavior_session.py','experiments/x2_behavior_bridge.py',
    'experiments/x2_skills.py','experiments/x2_preset_identity.py','experiments/x2_preset_compat.py',
    'experiments/x2_preset_helper.py','experiments/x2_sim_startup.py')))


def concise_failure(reason):
    """Keep the actionable exception line in the UI; full detail stays in logs."""
    text = next((line.strip() for line in reversed(str(reason).splitlines()) if line.strip()), '原因未记录')
    if 'MPS backend out of memory' in text:
        return '脑模型加载或计算时可用内存不足，请释放内存后重试'
    if 'Invalid transport JSON:' in text:
        filename = text.split('Invalid transport JSON:', 1)[1].split(';', 1)[0].strip()
        return '控制通信文件读取失败（'+filename+'），已停止本次运行'
    for prefix in ('ValueError:', 'RuntimeError:', 'TimeoutError:', 'InterruptedError:'):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = {'Fresh stable standing not confirmed': '未能确认机器人稳定站立',
            'Behavior bridge startup failed': '机器人控制连接初始化失败',
            'Bridge readiness expired': '等待机器人控制连接就绪超时'}.get(text, text)
    return text[:240]


def brain_resource_budget(available):
    from ...learning.resource_guard import resource_budget
    resources = resource_budget(available, rss_bytes=6*legacy.GIB, reserve_bytes=legacy.GIB)
    # Observed fixed-brain driver peak is ~1.42 GiB. Do not launch it with
    # an arbitrarily small positive budget or disable the allocator guard.
    require(resources['effective_rss_limit_bytes'] >= MIN_BRAIN_BUDGET,
            f'可用内存不足：当前 {available/legacy.GIB:.2f} GiB；'
            '启动至少需要 3 GiB（脑模型预算 2 GiB，系统预留 1 GiB），请释放内存后重试')
    return resources


class Commands(legacy.Commands):
    def __init__(self, stream):
        super().__init__(stream)
        self.tasks = queue.Queue(maxsize=50)
        self.clock_requests = queue.Queue(maxsize=4)
        self.clock_id = uuid.uuid4().hex
        self.clock_run_id = None
        self.clock_emit = None
        self.task_overflow_count = 0
        self.last_overflow_task = None
        self.priority_end = queue.Queue(maxsize=1)

    def start_clock_replies(self, run_id, emit):
        """Clock output cannot hold up the command reader or emergency stop."""
        self.clock_run_id, self.clock_emit = run_id, emit
        threading.Thread(target=self._clock_replies, daemon=True).start()

    def _clock_replies(self):
        info = time.get_clock_info('monotonic')
        while not self.stop.is_set():
            try:
                nonce = self.clock_requests.get(timeout=.1)
            except queue.Empty:
                continue
            if self.stop.is_set():
                return
            try:
                self.clock_emit(dict(schema='malecns_lab.desktop_clock.v1', type='clock',
                    run_id=self.clock_run_id, nonce=nonce, clock_id=self.clock_id,
                    python_monotonic_ns=str(time.monotonic_ns()),
                    clock_source=dict(api='time.monotonic_ns', implementation=info.implementation,
                                      monotonic=info.monotonic, adjustable=info.adjustable)))
            except (OSError, ValueError):
                return  # The bounded client request expires; no control command is synthesized.

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
                elif (isinstance(value,dict) and set(value) == {'command','run_id','nonce'}
                      and value['command'] == 'clock' and value['run_id'] == self.clock_run_id
                      and isinstance(value['nonce'],str) and len(value['nonce']) == 32
                      and all(c in '0123456789abcdef' for c in value['nonce'])):
                    try:
                        self.clock_requests.put_nowait(value['nonce'])
                    except queue.Full:
                        pass  # Drop clock probes, never wait ahead of stop/task input.
                elif isinstance(value,dict) and set(value) == {'command','task'} and value['command'] == 'task' and value['task'] in TASKS:
                    try:
                        self.tasks.put_nowait((value['task'],time.monotonic()))
                    except queue.Full:
                        if value['task'] == 'END_INTERACTION':
                            try:
                                self.priority_end.put_nowait((value['task'], time.monotonic()))
                                continue
                            except queue.Full:
                                pass
                        self.last_overflow_task = value['task']
                        self.task_overflow_count += 1
                else:
                    raise ValueError('Invalid desktop command')
        except BaseException as error:
            self.request_stop('desktop_command_error:'+str(error))


def candidate(root, override=None, *, turn_profile='measured', wave_profile='baseline'):
    """Resolve only host configuration/CLI, never a renderer path."""
    require(turn_profile in TURN_PROFILES, 'Unsupported frozen turn profile')
    require(wave_profile in WAVE_PROFILES, 'Unsupported frozen wave profile')
    require(wave_profile == 'baseline' or turn_profile in ('measured', 'quarter-turn', 'integral', 'compensated'), 'Wave balance-sway requires measured, quarter-turn, integral or compensated profile')
    require(turn_profile != 'quarter-turn' or wave_profile == 'balance-sway', 'Quarter-turn requires balance-sway wave profile')
    require(turn_profile != 'integral' or wave_profile == 'balance-sway', 'Integral requires balance-sway wave profile')
    require(turn_profile != 'compensated' or wave_profile == 'balance-sway', 'Compensated requires balance-sway wave profile')
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
    from ...learning.x2_frozen_history_policy import inspect_frozen_history_candidate, CANDIDATE_RELATIVE
    if canonical == root/CANDIDATE_RELATIVE:
        return canonical, inspect_frozen_history_candidate(root, canonical, turn_profile=turn_profile, wave_profile=wave_profile)
    require(turn_profile == 'measured', 'Selected turn profile requires the fixed B candidate')
    require(wave_profile == 'baseline', 'Selected wave profile requires the fixed B candidate')
    from ...learning.x2_behavior_training import inspect_checkpoint
    from ...learning.x2_interaction_policy import SCHEMA as INTERACTION_SCHEMA
    from ...learning.x2_interaction_paired_policy import SCHEMA as PAIRED_SCHEMA
    from ...learning.x2_sim_execution_policy import SCHEMA as EXECUTION_SCHEMA, validate_execution_checkpoint
    meta = inspect_checkpoint(canonical)
    require(meta['policy']['schema'] in ('malecns_lab.x2_behavior_policy.v1', INTERACTION_SCHEMA, PAIRED_SCHEMA, EXECUTION_SCHEMA),
            'New behavior checkpoint required')
    if meta['policy']['schema'] == EXECUTION_SCHEMA:
        validate_execution_checkpoint(root, canonical, meta)
    else:
        require(meta['policy'].get('interaction_profile', interaction_profile())['skill_profile_id'] != SIM_FIXED_ADVANCE_PROFILE_ID,
                'Fixed ADVANCE requires an execution-derived policy')
    return canonical,meta


class BehaviorRuntime(SkillRuntime):
    def __init__(self, output, root, commands, emit, checkpoint_path, metadata, *, session_profile='qualification', sim_profile='official', automatic_start=False):
        require(sim_profile in sim_configuration.PROFILES, 'Unsupported simulation profile')
        require(not automatic_start or sim_profile == 'official', 'Automatic startup requires the official simulation profile')
        self.automatic_start = automatic_start
        self.sim_profile = sim_profile
        self.simulation_configuration = None
        self.simulation_files_verified = False
        self.session_limits = session_profile_limits(session_profile)
        self.continuous = session_profile == 'continuous'
        self.session_running = False
        self.cleanup_deadline = None
        self.last_resource_check = -float('inf')
        self.heartbeat_sequence = 0
        self.brain_ready = False
        super().__init__(output,root,commands,emit,ui_timeout=300.,wall_timeout=480.)
        self.checkpoint_path,self.metadata = checkpoint_path,metadata
        self.run_id,self.session_id = output.name,'session-'+uuid.uuid4().hex
        self.interaction = validate_interaction_profile(metadata['policy'].get('interaction_profile', interaction_profile()))
        self.skill_profile = skill_profile_by_id(self.interaction['skill_profile_id'])
        if self.skill_profile['profile_id'] in (SIM_FIXED_ADVANCE_PROFILE_ID, *FULL_ACTION_PROFILE_IDS):
            require(metadata['policy'].get('execution_profile') == self.skill_profile,
                    'Candidate execution profile differs')
        self.task_sequence = 0
        self.tasks_open = False
        self.action_busy = self.request_pending = False
        self.active_request_id = self.active_task = None
        self.active_request_deadline = None
        self.reported_task_overflows = 0
        self.last_phase_value = None
        self.current_phase = 'initializing'
        self.rss_limit = 0
        self.resource_stream = None
        self.peak_rss = 0
        self.execution_failure = None
        self.last_execution_event_seq = -1
        self.secondary_errors = []

    def capture_execution_failure(self):
        """Latch only an owned, validated pre-shutdown execution protection fact."""
        path = self.output/'execution_latest.json'
        if not path.exists():
            return
        try:
            packet = read_json(path)
            require(packet.get('schema') == TRANSPORT_SCHEMA, 'Execution transport schema differs')
            require((packet.get('run_id'), packet.get('session_id')) == (self.run_id, self.session_id),
                    'Foreign execution transport')
            event = ExecutionEvent.from_dict(packet['event'])
            require((event.run_id, event.session_id) == (self.run_id, self.session_id),
                    'Foreign execution event')
            require(event.backend == 'sim', 'Execution backend differs')
            if event.event_seq <= self.last_execution_event_seq:
                return
            self.last_execution_event_seq = event.event_seq
            if (self.execution_failure is None and event.status in (ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN)
                    and event.protection_reason and event.protection_reason.strip()):
                self.execution_failure = dict(source=path.name, run_id=event.run_id, session_id=event.session_id,
                    event_seq=event.event_seq, decision_seq=event.decision_seq, execution_id=event.execution_id,
                    request_id=event.request_id, behavior=event.behavior.value, status=event.status.value,
                    protection_reason=event.protection_reason, remote_received_monotonic=event.received_monotonic,
                    observed_host_monotonic=time.monotonic(), stop_evidence=event.stop_evidence)
        except (OSError, ValueError, TypeError, KeyError) as error:
            rejected = 'execution_evidence_rejected:'+type(error).__name__+':'+str(error)
            if rejected not in self.secondary_errors:
                self.secondary_errors.append(rejected)

    def collect_secondary_errors(self):
        """Child summaries can explain exit/cleanup, but never establish the first cause."""
        for filename, failed in (('behavior_worker_result.json', ('failed',)),
                                 ('bridge_summary.json', ('FAILED', 'UNKNOWN'))):
            path = self.output/filename
            if not path.exists():
                continue
            try:
                result = read_json(path)
                bridge_error = (filename == 'bridge_summary.json' and result.get('reason')
                    and result['reason'] not in ('InterruptedError:independent_emergency_stop', 'InterruptedError:bridge_signal'))
                if (result.get('status') in failed or bridge_error) and isinstance(result.get('reason'), str):
                    error = filename+':'+result['reason']
                    if error not in self.secondary_errors:
                        self.secondary_errors.append(error)
            except (OSError, ValueError) as error:
                self.secondary_errors.append(filename+':'+type(error).__name__+':'+str(error))
        self.secondary_errors.extend(error for error in self.errors if error not in self.secondary_errors)

    def child_exit_reason(self, filename, fallback):
        """Read the owned child's cause before cleanup can add cancellations."""
        path = self.output/filename
        if not path.exists():
            return fallback
        try:
            result = read_json(path)
            require(all(result.get(k, expected) == expected for k, expected in
                        (('run_id', self.run_id), ('session_id', self.session_id))), 'Foreign child summary')
            reason = result.get('reason')
            return reason if isinstance(reason, str) and reason.strip() else fallback
        except (OSError, ValueError, TypeError) as error:
            self.secondary_errors.append(filename+':'+type(error).__name__+':'+str(error))
            return fallback

    def status(self, state, message, **fields):
        if 'phase' in fields:
            self.current_phase = fields['phase']
        if state in ('completed', 'cancelled', 'failed'):
            self.action_busy = self.request_pending = False
            self.active_request_id = self.active_task = None
            self.active_request_deadline = None
        alive = self.host is not None and self.host.poll() is None
        super().status(state, message, **{**fields, 'mode': 'brain_control',
            'sim_profile': self.sim_profile,
            'simulation_configuration_id': (self.simulation_configuration or {}).get('configuration_id'),
            'session_profile': self.session_limits['session_profile'],
            'action_busy': self.action_busy, 'request_pending': self.request_pending,
            'active_request_id': self.active_request_id, 'active_task': self.active_task,
            'brain_controlled': self.brain_ready and alive and state not in ('completed','cancelled','failed')})

    def reject_task(self, task, reason, *, count=1):
        record = dict(reason=reason, rejected_task=task, active_request_id=self.active_request_id,
                      host_monotonic=time.monotonic(), count=count)
        with (self.output/'task_rejections.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        label = {'action_busy':'当前动作尚未结束', 'end_pending':'正在结束互动',
                 'tasks_not_ready':'当前尚不能接收任务', 'task_budget':'本次任务次数已达上限',
                 'task_queue_full':'请求过于频繁，请等待当前动作结束'}[reason]
        self.status(self.state, label, phase=self.current_phase,
                    task=self.active_task, notice='request_rejected', **record)
        return False

    def accept_task(self, task, received):
        """Single owned request slot; ordinary busy input never overwrites it."""
        if not self.tasks_open:
            return self.reject_task(task, 'tasks_not_ready')
        if self.action_busy and (task != 'END_INTERACTION' or self.active_task == 'END_INTERACTION'):
            return self.reject_task(task, 'end_pending' if task == 'END_INTERACTION' else 'action_busy')
        if self.session_limits['max_requests'] is not None and self.task_sequence >= self.session_limits['max_requests'] and task != 'END_INTERACTION':
            return self.reject_task(task, 'task_budget')
        self.task_sequence += 1
        request = dict(run_id=self.run_id,session_id=self.session_id,sequence=self.task_sequence,
            request_id='request-'+uuid.uuid4().hex,task=task,host_received_monotonic=received)
        self.action_busy = self.request_pending = True
        self.active_request_id, self.active_task = request['request_id'], task
        self.active_request_deadline = received + TASK_REQUEST_SECONDS
        atomic(self.output/'task.json',request)
        with (self.output/'task_requests.jsonl').open('a') as stream:
            stream.write(json.dumps(request)+'\n')
        self.status('running','任务已接收，等待脑计算',phase='queued',task=task,
            request_id=request['request_id'],qualification_only=True)
        return True

    def deadline(self, cleanup=False):
        if self.continuous:
            if self.cleanup_deadline is not None:
                return self.cleanup_deadline
            if self.session_running:
                return float('inf')
        end = self.started+WALL_SECONDS
        if self.vm_started is not None:
            end = min(end,self.vm_started+VM_SECONDS)
        return end-(0. if cleanup else CLEANUP_SECONDS)

    def budget_record(self):
        if not self.continuous:
            return super().budget_record()
        return dict(session_profile='continuous', ui_timeout_seconds=self.ui_timeout,
            wall_timeout_seconds=None, wall_deadline_monotonic=None, work_deadline_monotonic=None,
            cleanup_reserve_seconds=CLEANUP_SECONDS, startup_deadline_monotonic=self.started+WALL_SECONDS,
            cleanup_deadline_monotonic=self.cleanup_deadline, after_ui_reserve_seconds=AFTER_UI_RESERVE)

    def guard(self):
        super().guard()
        now = time.monotonic()
        if self.continuous and now-self.last_resource_check >= 1.:
            require(shutil.disk_usage(self.output).free >= self.session_limits['disk_reserve_bytes'],
                    'Behavior log disk reserve exhausted')
            self.heartbeat_sequence += 1
            atomic(self.output/'supervisor_heartbeat.json', dict(run_id=self.run_id, session_id=self.session_id,
                sequence=self.heartbeat_sequence, host_monotonic=now))
            self.last_resource_check = now
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
        try:
            task, received = self.commands.priority_end.get_nowait()
        except queue.Empty:
            pass
        else:
            self.accept_task(task, received)
        while True:
            try:
                task,received = self.commands.tasks.get_nowait()
            except queue.Empty:
                break
            self.accept_task(task, received)
        if self.commands.task_overflow_count > self.reported_task_overflows:
            count = self.commands.task_overflow_count - self.reported_task_overflows
            self.reported_task_overflows += count
            self.reject_task(self.commands.last_overflow_task, 'task_queue_full', count=count)
        self.guard_request_deadline(time.monotonic())

    def guard_request_deadline(self, now):
        if self.continuous and self.action_busy and self.active_request_deadline is not None:
            require(now < self.active_request_deadline, 'Request 120 s budget expired')

    def prepare_wave_qualification(self):
        """Pin prior primitive evidence and installed artifacts before loading brain/VM."""
        contract = self.skill_profile['wave_contract']
        evidence_root = self.root/'data/x2_sim/agibot/data/var'
        archive = self.output/'wave_qualification'
        archive.mkdir(exist_ok=False)
        for run, files in contract['qualification_evidence'].items():
            for relative, expected in files.items():
                content = (evidence_root/run/relative).read_bytes()
                require(hashlib.sha256(content).hexdigest() == expected, 'Fixed wave evidence changed: '+run+'/'+relative)
                target = archive/run/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(content)
        self.prepare_preset_compat()
        binding = read_json(self.output/'preset_compat/binding.json')
        require(binding == contract['helper_binding'], 'Wave qualification differs from installed helper/MC')
        atomic(archive/'contract.json',contract)

    def prepare(self):
        from ...learning.x2_behavior_training import source_inventory
        import psutil
        resources = brain_resource_budget(psutil.virtual_memory().available)
        self.simulation_configuration = sim_configuration.prepare_configuration(
            self.root, self.output, self.remote, self.sim_profile)
        if self.skill_profile['wave_enabled']:
            self.prepare_wave_qualification()
        resources = brain_resource_budget(psutil.virtual_memory().available)
        self.rss_limit = resources['effective_rss_limit_bytes']
        self.resource_stream = (self.output/'behavior_resources.csv').open('x')
        self.resource_stream.write('wall_seconds,brain_rss_bytes,system_available_bytes\n')
        self.logs.append(self.resource_stream)
        sources = source_inventory(self.root)
        for name in (*TRANSPORT,'learning/x2_behavior_telemetry.py','learning/x2_telemetry.py',
                     'learning/x2_interaction_policy.py','learning/x2_interaction_paired_policy.py',
                     'learning/x2_sim_execution_policy.py','learning/x2_frozen_history_policy.py','senses/x2_relative_heading.py',
                     'runtime/__init__.py','runtime/x2/__init__.py',
                     'runtime/x2/behavior.py','runtime/x2/skills.py','runtime/x2/desktop.py','runtime/x2/sim_configuration.py'):
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
        identity = dict(run_id=self.run_id,session_id=self.session_id, sim_profile=self.sim_profile,
                        simulation_configuration_id=(self.simulation_configuration or {}).get('configuration_id'))
        skill_profile = self.skill_profile
        registration = dict(**identity,root=str(self.root),output=str(self.output),parent_pid=os.getpid(),
            checkpoint_path=str(self.checkpoint_path),checkpoint_metadata=checkpoint_registration_metadata(self.metadata),
            source_sha256=sources,rss_limit_bytes=self.rss_limit,system_reserve_bytes=legacy.GIB,
            **self.session_limits,
            qualification_only=True,wave_enabled=skill_profile['wave_enabled'],simulation_only=True,skill_profile=skill_profile,
            interaction_profile=self.interaction)
        atomic(self.output/'behavior_registration.json',registration)
        atomic(self.output/'bridge_registration.json',dict(**identity,qualification_only=True,simulation_only=True,wave_enabled=skill_profile['wave_enabled'],
            **self.session_limits,skill_profile=skill_profile,
            interaction_profile=self.interaction))
        self.status('preparing','正在加载候选脑模型',qualification_only=True,checkpoint_id=self.metadata['checkpoint_id'])
        environment = dict(self.environment,PYTHONPATH=str(self.root/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0')
        self.host = self.spawn([str(self.root/'.venv-train/bin/python'),'-u','-m',
            'malecns_lab.runtime.x2.behavior','--worker',str(self.output)],'brain-worker.log',environment)
        while not (self.output/'brain_ready.json').exists():
            self.guard()
            if self.host.poll() is not None:
                raise RuntimeError(self.child_exit_reason('behavior_worker_result.json',
                    'Candidate brain initialization failed; see brain-worker.log'))
            time.sleep(.1)
        ready = read_json(self.output/'brain_ready.json')
        require(ready['checkpoint_id'] == self.metadata['checkpoint_id'],'Worker selected another checkpoint')
        require(ready.get('skill_profile_id') == self.skill_profile['profile_id'], 'Worker execution profile differs')
        require(ready.get('session_contract_id') == self.session_limits['session_contract_id'], 'Worker session profile differs')
        self.brain_ready = True

    def verify_simulator_configuration(self):
        if self.sim_profile != 'official':
            require(self.simulation_configuration is not None, 'Simulation configuration not prepared')
            self.command(self.docker+['exec',legacy.CONTAINER,'python3','-c',
                sim_configuration.verification_code(self.simulation_configuration)])
            self.simulation_files_verified = True

    def simulator_launch_shell(self):
        if self.sim_profile == 'official':
            return super().simulator_launch_shell()
        require(self.simulation_configuration is not None and self.simulation_files_verified,
                'Derived simulation files not verified')
        return sim_configuration.launch_shell(self.simulation_configuration, self.remote)

    def prepare_simulator_ui(self):
        if not self.automatic_start:
            return False
        self.status('preparing', '正在复位仿真', stage='automatic_simulation_reset')
        command = ('source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && '
            'PYTHONPATH='+shlex.quote(self.remote+'/transport_src')+':$PYTHONPATH '
            'ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 timeout --kill-after=1s 15s '
            'python3 -m malecns_lab.experiments.x2_sim_startup --execute --sim-only --output '+shlex.quote(self.remote))
        try:
            result = self.command(self.docker+['exec',legacy.CONTAINER,'bash','-lc',command],
                                  timeout=17.,check=False,record='sim-startup-command.json')
            receipt = read_json(self.output/'sim_startup.json') if (self.output/'sim_startup.json').exists() else {}
            require(result.returncode == 0 and receipt.get('status') == 'ready',
                    '仿真自动准备失败：'+concise_failure(receipt.get('reason') or result.stderr))
            require((receipt.get('schema'),receipt.get('run_id'),receipt.get('session_id')) ==
                    ('malecns_lab.sim_startup.v1',self.run_id,self.session_id), 'Foreign simulation startup receipt')
            self.guard()
        except BaseException as error:
            # Killing docker exec's local client does not kill its container
            # process. Publish the owned cancellation before broader cleanup.
            try:
                atomic(self.output/'sim_startup_cancel.json', dict(run_id=self.run_id,
                    session_id=self.session_id, reason=type(error).__name__+':'+str(error),
                    host_monotonic=time.monotonic()))
            except BaseException as write_error:
                self.errors.append('startup_cancel_write:'+repr(write_error))
            raise
        return True

    def start_behavior_bridge(self):
        if self.automatic_start:
            self.status('preparing', '正在确认机器人站稳', stage='confirm_standing')
        command = ('source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && '
            'PYTHONPATH='+shlex.quote(self.remote+'/transport_src')+':$PYTHONPATH '
            'ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_behavior_bridge '
            '--execute --sim-only --output '+shlex.quote(self.remote))
        self.bridge = self.spawn(self.docker+['exec',legacy.CONTAINER,'bash','-lc',command],'behavior-bridge.log')
        end = min(self.deadline(),time.monotonic()+30.)
        while not (self.output/'bridge_ready.json').exists() and time.monotonic()<end:
            self.guard()
            if self.bridge.poll() is not None:
                raise RuntimeError(self.child_exit_reason('bridge_summary.json', '机器人控制连接初始化失败：控制进程已退出'))
            time.sleep(.1)
        require((self.output/'bridge_ready.json').exists(),'等待机器人稳定站立和控制连接就绪超时')
        ready = read_json(self.output/'bridge_ready.json')
        require(validate_candidate_skill_profile(ready.get('skill_profile')) == self.skill_profile
                and ready.get('capabilities') == self.interaction['capabilities']
                and ready.get('wave_enabled') == self.interaction['wave_enabled'], 'Bridge capabilities differ from candidate')
        require(ready.get('session_contract_id') == self.session_limits['session_contract_id'], 'Bridge session profile differs')
        self.session_running = True
        self.tasks_open = True
        self.status('running','请选择互动任务',qualification_only=True,wave_enabled=self.interaction['wave_enabled'],phase='waiting_task')

    def phase_status(self, phase):
        request_id = phase.get('request_id')
        if self.active_request_id is not None:
            if request_id != self.active_request_id:
                return False
            if phase['phase'] == 'request_failed':
                self.tasks_open = False
                raise RuntimeError('Brain request failed: '+str(phase.get('reason', 'unknown')))
            self.request_pending = False
            if phase['phase'] == 'waiting_task' and phase.get('stop_confirmed') is True:
                self.action_busy = False
                self.active_request_id = self.active_task = None
                self.active_request_deadline = None
        elif request_id is not None or phase['phase'] != 'waiting_task':
            return False
        label = {'computing':'脑模型正在计算','executing':'正在执行脑输出','rejected':'脑输出被拒绝',
                 'waiting_task':'等待互动任务'}.get(phase['phase'],'正在准备')
        if phase['phase'] == 'rejected':
            reason = {'invalid_hold_phase':'模型选择待命，但阶段无效',
                      'unavailable_skill':'模型选择的动作尚未开放',
                      'no_active_skill':'没有可继续或结束的动作',
                      'active_skill_mismatch':'模型输出与当前动作不一致',
                      'skill_already_active':'上一动作尚未结束',
                      'awaiting_execution_acceptance':'动作尚未确认接收',
                      'superseded_observation':'任务已更新，旧输出未执行',
                      'superseded_request':'任务已更新，旧输出未执行',
                      'expired_observation':'计算结果已超时，未执行'}.get(phase.get('reason'))
            if reason:
                label = reason
        elif phase['phase'] == 'waiting_task' and phase.get('reason'):
            label = phase['reason']
        self.status('running',label,**{**phase,'qualification_only':True})
        return True

    def shutdown_behavior(self, reason):
        if getattr(self, 'continuous', False) and self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic()+CLEANUP_SECONDS
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
            self.stop_confirmed = (result.get('base_stop_confirmed') is True
                and (not self.interaction['wave_enabled'] or result.get('wave_stop_confirmed') is True))
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
                self.capture_execution_failure()
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
            # Read the worker/bridge-exit race once before requesting cleanup. A
            # later cancellation event must not replace the pre-shutdown cause.
            if (isinstance(error, ValueError) and
                    (str(error) == 'Behavior bridge exited' or str(error).startswith('Brain worker: '))):
                self.capture_execution_failure()
            outcome.update(state='cancelled' if isinstance(error,legacy.StopRequested) else 'failed',
                           reason=type(error).__name__+':'+str(error))
            if str(error) == 'Behavior bridge exited':
                reason = self.child_exit_reason('bridge_summary.json', outcome['reason'])
                if reason != outcome['reason']:
                    self.secondary_errors.append(outcome['reason'])
                    outcome['reason'] = reason
        finally:
            if self.execution_failure is not None:
                if outcome['state'] != 'completed':
                    self.secondary_errors.append(outcome['reason'])
                outcome.update(state='failed', reason='Execution protection: '+self.execution_failure['protection_reason'])
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
            self.collect_secondary_errors()
            outcome.update(resources_stopped=self.resources_stopped,stop_confirmed=self.stop_confirmed,errors=self.errors,
                execution_failure=self.execution_failure, secondary_errors=self.secondary_errors,
                sim_profile=self.sim_profile, simulation_configuration=self.simulation_configuration,
                simulation_files_verified=self.simulation_files_verified,
                session_profile=self.session_limits['session_profile'], session_contract_id=self.session_limits['session_contract_id'],
                qualification_only=True,behavior_pass=None,wave_enabled=self.interaction['wave_enabled'],wall_seconds=time.monotonic()-self.started,
                peak_observed_rss_bytes=self.peak_rss,skill_profile=self.skill_profile)
            try:
                atomic(self.output/'behavior_host_summary.json',outcome)
            except BaseException as error:
                self.errors.append('summary_write:'+repr(error))
                self.secondary_errors.append(self.errors[-1])
                outcome['state'] = 'failed'
            message = '验证已结束' if outcome['state'] != 'failed' else '验证失败：'+concise_failure(outcome['reason'])
            if self.execution_failure is not None:
                message = '动作保护停止：'+self.execution_failure['protection_reason']
            self.status(outcome['state'],message, qualification_only=True,reason=outcome['reason'],
                        execution_failure=self.execution_failure, secondary_errors=self.secondary_errors)
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
    p.add_argument('--automatic-start',action='store_true',help='Prepare the owned official simulator before desktop continuation')
    p.add_argument('--session-profile', choices=SESSION_PROFILES, default='qualification')
    p.add_argument('--turn-profile', choices=TURN_PROFILES, default='measured')
    p.add_argument('--wave-profile', choices=WAVE_PROFILES, default='baseline')
    p.add_argument('--sim-profile', choices=sim_configuration.PROFILES, default='official')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        from ...learning.x2_behavior_session import run_worker
        return run_worker(args.worker)
    require(args.wave_profile == 'baseline' or args.turn_profile in ('measured', 'quarter-turn', 'integral', 'compensated'),
            'Wave balance-sway requires measured, quarter-turn, integral or compensated profile')
    require(args.turn_profile != 'quarter-turn' or args.wave_profile == 'balance-sway',
            'Quarter-turn requires balance-sway wave profile')
    require(args.turn_profile != 'integral' or args.wave_profile == 'balance-sway',
            'Integral requires balance-sway wave profile')
    require(args.turn_profile != 'compensated' or args.wave_profile == 'balance-sway',
            'Compensated requires balance-sway wave profile')
    if not args.execute:
        print(json.dumps(dict(mode='preview',brain_loaded=False,simulator_started=False,wave_enabled=False,
            tasks=TASKS, turn_profile=args.turn_profile, wave_profile=args.wave_profile,
            sim_profile=args.sim_profile,
            **session_profile_limits(args.session_profile),
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
        checkpoint_path,metadata = candidate(ROOT,args.checkpoint,turn_profile=args.turn_profile,wave_profile=args.wave_profile)
        with legacy.desktop_lock(ROOT),numerical_exclusion(ROOT):
            directory.mkdir(parents=True,exist_ok=False)
            with (directory/'lifecycle.jsonl').open('x') as stream:
                output_lock = threading.Lock()
                def emit(value):
                    if value.get('type') == 'status':
                        value = dict(value, host_clock_id=commands.clock_id)
                    line = json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
                    with output_lock:
                        stream.write(line)
                        stream.flush()
                        print(line,end='',flush=True)
                commands.start_clock_replies(directory.name,emit)
                commands.start()
                runtime = BehaviorRuntime(directory,ROOT,commands,emit,checkpoint_path,metadata,
                    session_profile=args.session_profile,sim_profile=args.sim_profile,automatic_start=args.automatic_start)
                result = runtime.execute_behavior()
        return 0 if result['state'] in ('completed','cancelled') else 1
    except BaseException as error:
        # This outer error happens before Runtime owns any resources (or after
        # its mandatory finally). Do not claim teardown if it failed there.
        sealed = directory/'desktop_session.json'
        if not sealed.exists():
            print(json.dumps(dict(schema=legacy.SCHEMA,type='status',state='failed',message='无法准备新脑模型：'+str(error),
                output=str(directory),run_id=None,simulator_ready=False,motion_started=False,stop_confirmed=None,
                mode='brain_control',brain_controlled=False,
                resources_stopped=True if runtime is None else runtime.resources_stopped),ensure_ascii=False),flush=True)
        return 1
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    raise SystemExit(main())
