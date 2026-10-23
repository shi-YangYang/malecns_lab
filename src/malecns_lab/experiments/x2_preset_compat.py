"""Nonblocking, single-request boundary to the official old-message process."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time

from . import x2_preset_identity as identity

MAX_LINE = 32768
MAX_STDERR = 65536
FIXED = {'motion': 1002, 'area': 2, 'interrupt': False}
SERVICE = '/aimdk_5Fmsgs/srv/SetMcPresetMotion'


def isolated_environment():
    # Construct rather than filter: no profile, BASH_ENV, DDS XML, Python hooks,
    # preload variables or newer SDK overlays can cross this boundary.
    return dict(PATH='/usr/bin:/bin', HOME='/tmp', LANG='C.UTF-8',
                ROS_DOMAIN_ID='47', ROS_LOCALHOST_ONLY='1', RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
                PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('Unexpected IPC keys')


def integer(value):
    if type(value) is not int:
        raise ValueError('Expected integer')


def stamp(value):
    exact(value, ('sec', 'nanosec'))
    integer(value['sec']); integer(value['nanosec'])
    if not 0 <= value['nanosec'] < 1000000000:
        raise ValueError('Invalid timestamp')


def raw_request(value):
    exact(value, ('header', 'area', 'motion', 'interrupt', 'ani_path', 'play_timestamp'))
    exact(value['header'], ('stamp',)); stamp(value['header']['stamp'])
    for key, number in (('area', 2), ('motion', 1002)):
        exact(value[key], ('value',))
        integer(value[key]['value'])
        if value[key]['value'] != number:
            raise ValueError('Unexpected preset')
    if value['interrupt'] is not False or value['ani_path'] != '' or type(value['play_timestamp']) is not int or value['play_timestamp'] != 0:
        raise ValueError('Unexpected preset parameters')


def raw_response(value):
    exact(value, ('response',))
    response = value['response']
    exact(response, ('header', 'task_id', 'state'))
    exact(response['header'], ('stamp', 'code')); stamp(response['header']['stamp'])
    exact(response['state'], ('value',))
    for number in (response['header']['code'], response['task_id'], response['state']['value']):
        integer(number)
    if not 0 <= response['task_id'] < 2**64:
        raise ValueError('Invalid task id')


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key')
        value[key] = item
    return value


def decode(line, *, nonce, run, probe, seq):
    if len(line) > MAX_LINE or not line.endswith(b'\n'):
        raise ValueError('Overlong or unterminated IPC')
    value = json.loads(line, object_pairs_hook=_unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    exact(value, ('kind', 'nonce', 'run', 'probe', 'seq', 'monotonic', 'wall_time', 'payload'))
    if (value['nonce'], value['run'], value['probe'], value['seq']) != (nonce, run, probe, seq) or type(value['seq']) is not int:
        raise ValueError('IPC identity/sequence mismatch')
    for key in ('monotonic', 'wall_time'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError('Invalid local IPC clock')
    payload, kind = value['payload'], value['kind']
    if kind == 'request':
        exact(payload, ())
    elif kind == 'ready':
        exact(payload, ('binding', 'identity'))
        exact(payload['binding'], ('schema', 'zip_sha256', 'mc_sha256', 'prepared_sha256', 'build_sha256', 'inspection_sha256'))
        exact(payload['identity'], ('version', 'prefix', 'fields', 'modules', 'libraries'))
    elif kind in ('sent', 'response'):
        exact(payload, ('message', 'cdr_hex'))
        (raw_request if kind == 'sent' else raw_response)(payload['message'])
        if not isinstance(payload['cdr_hex'], str) or not re.fullmatch(r'(?:[0-9a-f]{2}){1,4096}', payload['cdr_hex']):
            raise ValueError('Invalid serialized message')
    elif kind == 'error':
        exact(payload, ('error',))
        if not isinstance(payload['error'], str) or not 0 < len(payload['error']) <= 2048:
            raise ValueError('Invalid IPC error')
    else:
        raise ValueError('Unknown IPC kind')
    return value


def encode(kind, payload, *, nonce, run, probe, seq):
    value = dict(kind=kind, nonce=nonce, run=run, probe=probe, seq=seq,
                 monotonic=time.monotonic(), wall_time=time.time(), payload=payload)
    line = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
    decode(line, nonce=nonce, run=run, probe=probe, seq=seq)
    return line


def helper_command(nonce, run, probe):
    if not re.fullmatch('[a-f0-9]{32}', nonce) or not re.fullmatch(r'skills-\d{8}-\d{6}-[a-f0-9]{8,32}', run) or not re.fullmatch(r'probe-\d{2}-wave_right', probe):
        raise ValueError('Invalid probe identity')
    # Only the already-archived transport source directory is passed as Python
    # code; requests cannot supply paths, commands, service names or parameters.
    transport = Path(__file__).resolve().parents[2]
    env = isolated_environment()
    env['PYTHONPATH'] = str(transport)
    command = ['/bin/bash', '--noprofile', '--norc', '-ec',
        'source /opt/ros/humble/setup.bash; source /workspace/aimdk-preset-v1_0_0/install/local_setup.bash; '
        'exec /usr/bin/python3 -m malecns_lab.experiments.x2_preset_helper '
        '--nonce ' + nonce + ' --run ' + run + ' --probe ' + probe]
    return command, env


class PresetBoundary:
    def __init__(self, output, record, *, binding, root=identity.LINUX_ROOT, clock=time.monotonic):
        self.output, self.record, self.binding, self.root, self.clock = Path(output), record, binding, root, clock
        self.run, self.probe = self.output.parent.name, self.output.name
        self.nonce = secrets.token_hex(16)
        self.process = None
        self.buffer = bytearray()
        self.stderr_size = 0
        self.seq = 0
        self.ready = self.sent = self.acknowledged = self.failed = False
        self.requested_at = None
        self.started = clock()
        self.stderr = (self.output / 'preset-helper.stderr').open('xb')

    def start(self):
        if self.process is not None:
            raise ValueError('Helper already started')
        command, env = helper_command(self.nonce, self.run, self.probe)
        self.process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
        for stream in (self.process.stdout, self.process.stderr, self.process.stdin):
            os.set_blocking(stream.fileno(), False)
        self.record('preset_helper_start', nonce=self.nonce, binding=self.binding,
                    helper_sha256=identity.sha(Path(__file__).with_name('x2_preset_helper.py')))

    def request(self, command):
        if command != FIXED or not self.ready or self.sent or self.failed:
            raise RuntimeError('Preset helper not ready or duplicate/invalid request')
        line = encode('request', {}, nonce=self.nonce, run=self.run, probe=self.probe, seq=0)
        self.sent = True  # A partial write is unknown and must never retry.
        self.requested_at = self.clock()
        if os.write(self.process.stdin.fileno(), line) != len(line):
            raise RuntimeError('Partial preset IPC write')
        self.record('preset_ipc_tx', message=json.loads(line))

    def _ready(self, payload):
        if payload['binding'] != self.binding:
            raise ValueError('Child build binding mismatch')
        actual = payload['identity']
        inspection = identity.read(self.root / 'inspection.json')['identity']
        files = identity.read(self.root / 'build.json')['install_files']
        if (actual['version'] != 'v1.0.0' or actual['prefix'] != str(self.root / 'install')
                or actual['fields'] != inspection['fields']):
            raise ValueError('Child type/layout identity mismatch')
        for group in ('modules', 'libraries'):
            if not isinstance(actual[group], dict) or not actual[group]:
                raise ValueError('Missing child loaded artifacts')
            if any(files.get(path) != digest for path, digest in actual[group].items()):
                raise ValueError('Child loaded artifact mismatch')

    def poll(self):
        """Bounded nonblocking drain; one callback tick never waits for ROS/child."""
        if self.failed:
            return []
        now = self.clock()
        if self.sent and not self.acknowledged and now - self.requested_at >= 2.:
            return self.fail('preset_ack_timeout')
        if not self.ready and now - self.started >= 12.:
            return self.fail('preset_helper_ready_timeout')
        events = []
        try:
            for stream, channel in ((self.process.stderr, 'stderr'), (self.process.stdout, 'stdout')):
                try:
                    chunk = os.read(stream.fileno(), MAX_LINE + 1)
                except BlockingIOError:
                    continue
                if channel == 'stderr':
                    self.stderr_size += len(chunk)
                    if self.stderr_size > MAX_STDERR:
                        raise ValueError('Helper stderr budget exceeded')
                    self.stderr.write(chunk)
                else:
                    self.buffer.extend(chunk)
                    if len(self.buffer) > MAX_LINE:
                        raise ValueError('Helper stdout budget exceeded')
            while b'\n' in self.buffer:
                line, _, tail = self.buffer.partition(b'\n')
                self.buffer[:] = tail
                value = decode(bytes(line)+b'\n', nonce=self.nonce, run=self.run, probe=self.probe, seq=self.seq)
                kind = value['kind']
                self.record('preset_ipc_rx', message=value)
                if kind == 'error':
                    raise ValueError('Child error: ' + value['payload']['error'])
                expected = ('ready', 'sent', 'response')[self.seq] if self.seq < 3 else None
                if kind != expected or (self.seq > 0 and not self.sent):
                    raise ValueError('Unexpected IPC state')
                if kind == 'ready':
                    self._ready(value['payload'])
                    self.ready = True
                elif kind == 'response':
                    self.acknowledged = True
                self.seq += 1
                events.append(value)
            code = self.process.poll()
            if code is not None and (code != 0 or not self.acknowledged or self.buffer):
                raise ValueError('Helper exited without a complete successful exchange: ' + str(code))
            return events
        except (ValueError, OSError, IndexError, TypeError, KeyError) as error:
            return self.fail('preset_helper_protocol:' + str(error))

    def fail(self, reason):
        self.failed = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()  # Nonblocking; final close reaps with a fixed bound.
        self.record('preset_helper_unknown', reason=reason, cancellation_verified=False)
        return [dict(kind='error', payload={'error': reason})]

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=.5)
                self.record('preset_helper_exit', returncode=self.process.returncode,
                            reaped=True, cancellation_verified=False)
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    stream.close()
        finally:
            self.stderr.close()
