"""CPU-only compatibility ownership/IPC/real runner wiring. No ROS or devices."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from malecns_lab.experiments import x2_preset_compat as ipc
from malecns_lab.experiments import x2_preset_identity as identity
from malecns_lab.experiments import x2_skills as runner
from malecns_lab.runtime.x2 import skills as host

NONCE = 'a' * 32
RUN = 'skills-20261023-120000-aaaaaaaa'
PROBE = 'probe-00-wave_right'
KEY = dict(nonce=NONCE, run=RUN, probe=PROBE)
RAW_REQUEST = dict(header={'stamp': {'sec': 0, 'nanosec': 0}}, area={'value': 2},
    motion={'value': 1002}, interrupt=False, ani_path='', play_timestamp=0)
RAW_RESPONSE = dict(response=dict(header=dict(stamp=dict(sec=0, nanosec=0), code=0),
                                 task_id=9, state={'value': 400}))


def test_strict_protocol_and_clean_environment(monkeypatch):
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH', 'BASH_ENV', 'LD_PRELOAD', 'FASTRTPS_DEFAULT_PROFILES_FILE'):
        monkeypatch.setenv(key, '/new-sdk-or-hook')
    command, env = ipc.helper_command(NONCE, RUN, PROBE)
    assert command[:4] == ['/bin/bash', '--noprofile', '--norc', '-ec']
    assert identity.DIRECTORY + '/install/local_setup.bash' in command[-1]
    assert not any('/new-sdk-or-hook' in v for v in env.values())
    assert not any(k in env for k in ('BASH_ENV', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH'))
    assert env['ROS_DOMAIN_ID'] == '47' and env['RMW_IMPLEMENTATION'] == 'rmw_fastrtps_cpp'
    line = ipc.encode('request', {}, **KEY, seq=0)
    assert ipc.decode(line, **KEY, seq=0)['payload'] == {}
    invalid = [line.replace(b'"payload":{}', b'"payload":{"motion":1002}'),
        line.replace(b'"seq":0', b'"seq":0,"seq":0'),
        line.replace(b'"seq":0', b'"seq":true'),
        line.replace(b'"nonce":"a', b'"nonce":"b'),
        line[:-1], b'x' * (ipc.MAX_LINE + 1) + b'\n',
        line.replace(b'"payload":{}', b'"payload":{},"extra":1')]
    for bad in invalid:
        with pytest.raises(ValueError):
            ipc.decode(bad, **KEY, seq=0)
    with pytest.raises(ValueError):
        ipc.raw_request({**RAW_REQUEST, 'input_source': {}})
    for field, value in (('motion', {'value': 1001}), ('interrupt', True), ('play_timestamp', 1)):
        with pytest.raises(ValueError):
            ipc.raw_request({**RAW_REQUEST, field: value})


class PipeProcess:
    def __init__(self):
        self.parent_input, child_input = os.pipe()
        output, self.child_output = os.pipe()
        error, self.child_error = os.pipe()
        self.stdin = os.fdopen(child_input, 'wb', buffering=0)
        self.stdout = os.fdopen(output, 'rb', buffering=0)
        self.stderr = os.fdopen(error, 'rb', buffering=0)
        self.returncode = None
        self.terminated = self.waited = 0
        self.pid = 999999
    def poll(self): return self.returncode
    def terminate(self): self.terminated += 1; self.returncode = -15
    def wait(self, timeout): self.waited += 1; return self.returncode
    def release(self):
        for fd in (self.parent_input, self.child_output, self.child_error): os.close(fd)


def boundary(tmp_path, monkeypatch):
    output = tmp_path / RUN / PROBE
    output.mkdir(parents=True)
    process, records, now = PipeProcess(), [], [10.]
    monkeypatch.setattr(ipc.subprocess, 'Popen', lambda *a, **kw: process)
    b = ipc.PresetBoundary(output, lambda kind, **data: records.append((kind, data)),
                           binding={}, clock=lambda: now[0])
    monkeypatch.setattr(b, '_ready', lambda payload: None)
    b.start()
    def send(kind, payload, seq):
        os.write(process.child_output, ipc.encode(kind, payload,
            nonce=b.nonce, run=RUN, probe=PROBE, seq=seq))
    ready = {'binding': dict(schema='', zip_sha256='', mc_sha256='', mc_contract_sha256={}, prepared_sha256='', build_sha256='', inspection_sha256=''),
             'identity': dict(version='', prefix='', fields={}, modules={}, libraries={})}
    send('ready', ready, 0)
    assert b.poll()[0]['kind'] == 'ready'
    return b, process, send, records, now


def test_nonblocking_single_request_response_and_reap(tmp_path, monkeypatch):
    b, process, send, records, now = boundary(tmp_path, monkeypatch)
    try:
        assert b.poll() == []  # Empty pipes never wait.
        b.request(ipc.FIXED)
        request = os.read(process.parent_input, ipc.MAX_LINE)
        assert ipc.decode(request, nonce=b.nonce, run=RUN, probe=PROBE, seq=0)['kind'] == 'request'
        with pytest.raises(RuntimeError): b.request(ipc.FIXED)
        send('sent', {'message': RAW_REQUEST, 'cdr_hex': '0102'}, 1)
        send('response', {'message': RAW_RESPONSE, 'cdr_hex': '0304'}, 2)
        assert [e['kind'] for e in b.poll()] == ['sent', 'response']
        assert b.acknowledged
        process.returncode = 0
        assert b.poll() == []
        b.close()
        assert records[-1][0] == 'preset_helper_exit'
        assert records[-1][1]['reaped'] and not records[-1][1]['cancellation_verified']
    finally:
        process.release()


@pytest.mark.parametrize('failure', ['timeout', 'duplicate', 'exit', 'invalid', 'stderr'])
def test_unknown_failures_do_not_retry_and_close(tmp_path, monkeypatch, failure):
    b, process, send, records, now = boundary(tmp_path, monkeypatch)
    try:
        b.request(ipc.FIXED)
        if failure == 'timeout': now[0] += 2.
        elif failure == 'duplicate':
            send('sent', {'message': RAW_REQUEST, 'cdr_hex': '0102'}, 1)
            send('sent', {'message': RAW_REQUEST, 'cdr_hex': '0102'}, 1)
        elif failure == 'exit': process.returncode = 1
        elif failure == 'invalid': os.write(process.child_output, b'{"invalid":true}\n')
        else:
            b.stderr_size = ipc.MAX_STDERR
            os.write(process.child_error, b'x')
        events = b.poll()
        assert events[-1]['kind'] == 'error' and b.failed
        with pytest.raises(RuntimeError): b.request(ipc.FIXED)
        assert b.poll() == []
        b.close()
        assert process.returncode is not None
        assert any(k == 'preset_helper_unknown' for k, v in records)
    finally:
        process.release()


def test_source_fields_match_ros_generated_spelling(tmp_path):
    path = tmp_path / 'Example.msg'
    path.write_text('float64 velocity\nbool active\nJointState[] joints\nuint64 SOME_CONSTANT=2\n')
    assert identity.source_fields(path, 'Example') == dict(velocity='float64', active='bool', joints='sequence<aimdk_msgs/msg/JointState>')
    assert identity._type('double') == 'float64'
    assert identity._type('boolean') == 'bool'
    assert identity._type('sequence<aimdk_msgs/JointState>') == 'sequence<aimdk_msgs/msg/JointState>'


def test_missing_build_fails_before_ros_or_host_start(tmp_path, monkeypatch):
    def missing(*args, **kw): raise ValueError('missing inspected build')
    monkeypatch.setattr(identity, 'verify_build', missing)
    args = NS(command_seconds=1., skill='WAVE_RIGHT')
    with pytest.raises(ValueError, match='missing inspected build'):
        runner.execute(args)
    runtime = host.SkillRuntime(tmp_path, tmp_path, host.desktop.Commands(io.StringIO()), lambda value: None)
    with pytest.raises(ValueError, match='missing inspected build'):
        runtime.prepare_preset_compat()


@pytest.mark.parametrize('mode', ['wave_ack', 'wave_error', 'nonwave'])
def test_real_runner_callback_ack_unknown_and_nonwave_no_helper(tmp_path, monkeypatch, mode):
    from malecns_lab.control import x2_skills as control
    cores, boundaries, services, published = [], [], [], []
    clock = [100.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + max(seconds, .02)))
    monkeypatch.setattr(identity, 'verify_build', lambda *a, **kw: {})
    monkeypatch.setattr(identity, 'runtime_identity', lambda *a, **kw: {'new': True})
    monkeypatch.setattr(runner, 'official_wave_poses', lambda *a: dict(first={}, terminal={}))

    class Core:
        def __init__(self, publish, preset, **kwargs):
            self.preset, self.command_seconds = preset, kwargs['command_seconds']
            self.command = None
            self.terminal = False
            self.cancel_requested = False
            self.started_at = None
            self.query_state, self.query_at, self.joints, self.task_id = 1, clock[0], {}, None
            self.motion = NS(arm=lambda: None, confirm_source_registration=lambda: None, state='READY')
            self.status, self.reason = 'IDLE', ''
            self.acks = []
            cores.append(self)
        def standing_ready(self): return True
        def tick(self):
            if self.command and mode == 'nonwave': self.terminal = True; self.status = 'COMPLETED'
            return {}
        def submit(self, command):
            self.command, self.started_at = command, clock[0]
            if command.behavior == 'WAVE_RIGHT': self.preset(dict(ipc.FIXED))
            return True
        def cancel(self, reason): self.status = 'UNKNOWN'; self.reason = reason; self.terminal = True
        def preset_sent(self, received): pass
        def preset_ack(self, code, task, state):
            self.acks.append((code, task, state)); self.task_id = task
            # Stop this fake core on ACK; real core completion is separately
            # tested and unchanged. This only exercises runner callback wiring.
            self.status = 'ACCEPTED'; self.terminal = True
        def preset_query(self, *args, **kw): self.query_at = clock[0]
        def summary(self): return dict(status=self.status, reason=self.reason)
    monkeypatch.setattr(control, 'SkillExecutor', Core)

    class Boundary:
        def __init__(self, *a, **kw): self.ready = True; self.requests = []; self.closed = False; boundaries.append(self)
        def start(self): pass
        def request(self, command): self.requests.append(command)
        def poll(self):
            if not self.requests: return []
            if mode == 'wave_error': return [dict(kind='error', payload={'error': 'preset_ack_timeout'})]
            return [dict(kind='sent', monotonic=clock[0], payload=dict(message=RAW_REQUEST, cdr_hex='0102')),
                    dict(kind='response', payload=dict(message=RAW_RESPONSE, cdr_hex='0304'))]
        def close(self): self.closed = True
    monkeypatch.setattr(ipc, 'PresetBoundary', Boundary)

    class Request:
        def __init__(self):
            self.request = NS(header=NS(stamp=None), task_id=0)
            self.action = NS(value=None)
            self.input_source = NS(name=None, priority=None, timeout=None)
    class Node:
        def __init__(self, *args): pass
        def create_publisher(self, *args): return NS(publish=published.append)
        def create_client(self, service, name):
            services.append(name)
            return NS(service_is_ready=lambda: True, call_async=lambda request:
                NS(done=lambda: True, cancel=lambda: None, result=lambda:
                    NS(response=NS(header=NS(code=0), task_id=0, state=NS(value=1)))))
        def create_subscription(self, *args): pass
        def get_clock(self): return NS(now=lambda: NS(to_msg=lambda: None))
        def destroy_node(self): pass
    modules = {
        'rclpy': NS(init=lambda **kw: None, spin_once=lambda *a, **kw: None, ok=lambda: True, shutdown=lambda: None),
        'rclpy.node': NS(Node=Node),
        'rclpy.qos': NS(QoSProfile=lambda **kw: None, ReliabilityPolicy=NS(BEST_EFFORT=1), DurabilityPolicy=NS(VOLATILE=1)),
        'rclpy.signals': NS(SignalHandlerOptions=NS(NO=0)),
        'rosidl_runtime_py.convert': NS(message_to_ordereddict=lambda value: {}),
        'aimdk_msgs.msg': NS(McCommonState=object, JointStateArray=object, McLocomotionVelocity=lambda: NS(header=NS(stamp=None)), McInputAction=NS(INPUTACTION_ADD=1, INPUTACTION_ENABLE=2)),
        # No SetMcPresetMotion at all: parent cannot accidentally use new Set.
        'aimdk_msgs.srv': NS(SetMcInputSource=NS(Request=Request), GetMcPresetMotionState=NS(Request=Request)),
        'nav_msgs.msg': NS(Odometry=object),
    }
    for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
    args = NS(command_seconds=1., skill='HOLD' if mode == 'nonwave' else 'WAVE_RIGHT',
              output=tmp_path / mode, wave_probe=True, cancel_after=None)
    runner.execute(args)
    assert not any(name.endswith('/SetMcPresetMotion') for name in services)
    if mode == 'nonwave':
        assert boundaries == []
    else:
        assert len(boundaries) == 1 and boundaries[0].closed
        assert boundaries[0].requests == [ipc.FIXED]
        if mode == 'wave_ack': assert cores[0].acks == [(0, 9, 400)]
        else: assert cores[0].status == 'UNKNOWN' and not cores[0].acks
    config = [json.loads(line) for line in (args.output/'events.jsonl').read_text().splitlines()]
    assert next(e for e in config if e['kind']=='configuration')['wave_seconds'] == 12.
    assert not next(e for e in config if e['kind']=='configuration')['cancellation_verified']

@pytest.mark.parametrize('mode', ['natural', 'duplicate', 'cancel'])
def test_helper_actual_send_site_uses_old_six_fields_once(tmp_path, monkeypatch, mode):
    from malecns_lab.experiments import x2_preset_helper as helper
    prepared = dict(tool_files={'malecns_lab/experiments/'+name: 'hash' for name in
        ('x2_preset_identity.py', 'x2_preset_compat.py', 'x2_preset_helper.py')})
    binding = dict(schema='', zip_sha256='', mc_sha256='', mc_contract_sha256={}, prepared_sha256='', build_sha256='', inspection_sha256='')
    monkeypatch.setattr(identity, 'verify_build', lambda *a, **kw: binding)
    monkeypatch.setattr(identity, 'read', lambda *a: prepared)
    monkeypatch.setattr(identity, 'sha', lambda *a: 'hash')
    monkeypatch.setattr(identity, 'runtime_identity', lambda *a: dict(version='', prefix='', fields={}, modules={}, libraries={}))
    calls, read_calls, cancel_reads = [], [], []
    output = io.BytesIO()
    monkeypatch.setattr(helper.sys, 'stdin', NS(fileno=lambda: 90))
    monkeypatch.setattr(helper.sys, 'stdout', NS(buffer=output))
    monkeypatch.setattr(helper.os, 'set_blocking', lambda *a: None)
    def receive(*args):
        read_calls.append(True)
        if len(read_calls) == 1 or mode == 'duplicate':
            return ipc.encode('request', {}, **KEY, seq=0)
        if mode == 'cancel' and len(output.getvalue().splitlines()) == 3 and not cancel_reads:
            cancel_reads.append(True)
            return ipc.encode('cancel', {}, **KEY, seq=1)
        raise BlockingIOError()
    monkeypatch.setattr(helper.os, 'read', receive)
    class Request:
        def __init__(self):
            self.header = NS(stamp=None); self.area = NS(value=0); self.motion = NS(value=0)
            self.interrupt = True; self.ani_path = 'unset'; self.play_timestamp = 1
    response = NS(response=NS(header=NS(code=0), state=NS(value=400)))
    def call(request):
        calls.append(request)
        return NS(done=lambda: True, result=lambda: response)
    class Node:
        def __init__(self, *a): pass
        def create_client(self, service, name):
            assert name == ipc.SERVICE
            return NS(service_is_ready=lambda: True, call_async=call)
        def get_clock(self): return NS(now=lambda: NS(to_msg=lambda: {'sec': 0, 'nanosec': 0}))
        def destroy_node(self): pass
    def raw(message):
        if message is response: return RAW_RESPONSE
        assert vars(message).keys() == RAW_REQUEST.keys()
        cancel = len(calls) == 1 and mode == 'cancel'
        assert (message.motion.value, message.area.value, message.interrupt, message.play_timestamp, message.ani_path) == (0 if cancel else 1002, 2, cancel, 0, '')
        return {**RAW_REQUEST, 'motion': {'value': message.motion.value}, 'interrupt': message.interrupt}
    for name, module in {
        'rclpy': NS(init=lambda: None, shutdown=lambda: None, spin_once=lambda *a, **kw: None),
        'rclpy.node': NS(Node=Node), 'rclpy.serialization': NS(serialize_message=lambda value: b'\x01\x02'),
        'rosidl_runtime_py.convert': NS(message_to_ordereddict=raw),
        'aimdk_msgs.srv': NS(SetMcPresetMotion=NS(Request=Request)),
    }.items(): monkeypatch.setitem(sys.modules, name, module)
    code = helper.run(NS(**KEY, allow_cancel=mode == 'cancel'))
    assert code == int(mode == 'duplicate')
    assert len(calls) == (2 if mode == 'cancel' else 1)
    messages = [ipc.decode(line+b'\n', **KEY, seq=n) for n, line in enumerate(output.getvalue().splitlines())]
    assert [m['kind'] for m in messages] == (['ready', 'sent', 'error'] if mode == 'duplicate' else
        ['ready', 'sent', 'response', 'cancel_sent', 'cancel_response'] if mode == 'cancel' else ['ready', 'sent', 'response'])


def test_two_step_boundary_requires_wave_ack_and_never_retries_cancel(tmp_path, monkeypatch):
    b, process, send, records, now = boundary(tmp_path, monkeypatch)
    b.allow_cancel = True
    zero = {**RAW_REQUEST, 'motion': {'value': 0}, 'interrupt': True}
    try:
        with pytest.raises(RuntimeError): b.request(ipc.CANCEL)
        b.request(ipc.FIXED)
        os.read(process.parent_input, ipc.MAX_LINE)
        with pytest.raises(RuntimeError): b.request(ipc.CANCEL)
        send('sent', {'message': RAW_REQUEST, 'cdr_hex': '0102'}, 1)
        send('response', {'message': RAW_RESPONSE, 'cdr_hex': '0304'}, 2)
        b.poll()
        b.request(ipc.CANCEL)
        message = os.read(process.parent_input, ipc.MAX_LINE)
        assert ipc.decode(message, nonce=b.nonce, run=RUN, probe=PROBE, seq=1)['kind'] == 'cancel'
        with pytest.raises(RuntimeError): b.request(ipc.CANCEL)
        send('cancel_sent', {'message': zero, 'cdr_hex': '0506'}, 3)
        send('cancel_response', {'message': RAW_RESPONSE, 'cdr_hex': '0708'}, 4)
        assert [v['kind'] for v in b.poll()] == ['cancel_sent', 'cancel_response']
        assert b.cancel_acknowledged
        process.returncode = 0
        assert b.poll() == []
        with pytest.raises(ValueError):
            ipc.encode('cancel_sent', {'message': RAW_REQUEST, 'cdr_hex': '0506'}, **KEY, seq=3)
    finally:
        b.close()
        process.release()


def test_cancel_ack_timeout_closes_boundary_without_resend(tmp_path, monkeypatch):
    b, process, send, records, now = boundary(tmp_path, monkeypatch)
    b.allow_cancel = True
    try:
        b.request(ipc.FIXED)
        send('sent', {'message': RAW_REQUEST, 'cdr_hex': '0102'}, 1)
        send('response', {'message': RAW_RESPONSE, 'cdr_hex': '0304'}, 2)
        b.poll()
        b.request(ipc.CANCEL)
        now[0] += 2.
        assert b.poll()[-1]['payload']['error'] == 'preset_cancel_ack_timeout'
        with pytest.raises(RuntimeError): b.request(ipc.CANCEL)
        assert b.failed
    finally:
        b.close()
        process.release()


def test_fixed_contract_requires_every_official_binary_and_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        identity.verify_mc_contract(tmp_path)
    assert len(identity.MC_CONTRACT) == 6
    assert identity.DIRECTORY != 'aimdk-preset-v1_0_0'  # Preserve previous evidence workspace.


@pytest.mark.parametrize('entry', [runner.main, host.main])
def test_cancel_probe_cannot_target_other_skills(entry):
    with pytest.raises(SystemExit) as error:
        entry(['--execute', '--sim-only', '--output', '/tmp/unused-probe', '--skill', 'HOLD', '--wave-cancel-probe'])
    assert error.value.code == 2


def test_wave_reference_reads_first_and_terminal_separately_and_checks_hash(tmp_path, monkeypatch):
    from malecns_lab.control.x2_skills import RIGHT_ARM
    relative = 'mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation/SM_wave_right_hand.csv'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(','.join(['timeMS'] + ['command_pos::' + k for k in RIGHT_ARM]) + '\n'
                    + '0,' + ','.join(['0.1'] * len(RIGHT_ARM)) + '\n'
                    + '4280,' + ','.join(['0.4'] * len(RIGHT_ARM)) + '\n')
    monkeypatch.setitem(identity.MC_CONTRACT, relative, identity.sha(path))
    poses = runner.official_wave_poses(tmp_path)
    assert poses['first'] == dict.fromkeys(RIGHT_ARM, .1)
    assert poses['terminal'] == dict.fromkeys(RIGHT_ARM, .4)
    path.write_text(path.read_text().replace('0.4', '0.5'))
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.official_wave_poses(tmp_path)
