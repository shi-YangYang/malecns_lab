"""Bounded, read-only MC debug subscriber; importing it does not load ROS."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import time

SCHEMA = 'malecns_lab.x2_mc_debug_capture.v1'
TOPIC = '/aima/mc/rl/debug'
MC_LIB = Path('/workspace/mc/lib')
MAX_SECONDS = 120.
MAX_MESSAGES = 6000
MAX_CDR_BYTES = 65536
MAX_VECTOR = 4096
MAX_LOG_BYTES = 30 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
WIRE_SCHEMA = {
    'type': 'aimdk_msgs/msg/McRLDebug',
    'fields': [['header', 'aimdk_msgs/msg/MessageHeader'],
               ['obs_vec', 'float32[]'], ['action_vec', 'float32[]']],
    'MessageHeader': [['stamp', 'builtin_interfaces/msg/Time'], ['frame_id', 'string'],
                      ['sequence', 'uint32'], ['meas_stamp', 'builtin_interfaces/msg/Time']],
    'Time': [['sec', 'int32'], ['nanosec', 'uint32']],
    'encoding': 'CDR v1, big or little endian; alignment relative to payload after 4-byte encapsulation',
    'interpretation': 'raw vectors only; no assumed history ordering or action-joint mapping',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_cdr(raw):
    """Decode precisely MessageHeader + two float32 sequences, or reject."""
    require(isinstance(raw, bytes) and 4 <= len(raw) <= MAX_CDR_BYTES, 'Invalid CDR byte budget')
    require(raw[:2] in (b'\x00\x00', b'\x00\x01') and raw[2:4] == b'\x00\x00',
            'Unsupported CDR encapsulation/options')
    order = '<' if raw[1] else '>'
    offset = 4

    def take(count):
        nonlocal offset
        require(0 <= count <= len(raw) - offset, 'Truncated CDR payload')
        result = raw[offset:offset + count]
        offset += count
        return result

    def scalar(code):
        # All primitives in this fixed schema are four bytes. CDR padding is
        # not required to contain zero bytes, so only bounds are checked.
        take((-(offset - 4)) % 4)
        return struct.unpack(order + code, take(4))[0]

    def timestamp():
        result = dict(sec=scalar('i'), nanosec=scalar('I'))
        require(result['nanosec'] < 1_000_000_000, 'Invalid Time nanosec')
        return result

    stamp = timestamp()
    length = scalar('I')
    require(1 <= length <= 4096, 'Invalid frame_id string length')
    frame_bytes = take(length)
    require(frame_bytes[-1:] == b'\x00' and b'\x00' not in frame_bytes[:-1], 'Invalid frame_id terminator')
    frame_id = frame_bytes[:-1].decode('utf-8', errors='strict')
    sequence = scalar('I')
    meas_stamp = timestamp()

    def vector(name):
        count = scalar('I')
        require(count <= MAX_VECTOR, name + ' exceeds vector budget')
        values = list(struct.unpack(order + str(count) + 'f', take(count * 4)))
        require(all(math.isfinite(value) for value in values), name + ' contains nonfinite values')
        return values

    obs = vector('obs_vec')
    actions = vector('action_vec')
    require(offset == len(raw), 'Unexpected trailing CDR bytes')
    return dict(header=dict(stamp=stamp, frame_id=frame_id, sequence=sequence, meas_stamp=meas_stamp),
                obs_vec=obs, action_vec=actions)


class _TypeSupport(C.Structure):
    _fields_ = [('identifier', C.c_char_p), ('data', C.c_void_p), ('function', C.c_void_p)]


class _Member(C.Structure):
    # ROS 2 Humble introspection_cpp ABI, used only against the MC-owned library.
    _fields_ = [('name', C.c_char_p), ('type_id', C.c_uint8), ('string_bound', C.c_size_t),
                ('members', C.c_void_p), ('is_array', C.c_bool), ('array_size', C.c_size_t),
                ('is_upper_bound', C.c_bool), ('offset', C.c_uint32), ('default', C.c_void_p),
                *[(name, C.c_void_p) for name in ('size_fn', 'get_const_fn', 'get_fn',
                                                 'fetch_fn', 'assign_fn', 'resize_fn')]]


class _Members(C.Structure):
    _fields_ = [('namespace', C.c_char_p), ('name', C.c_char_p), ('count', C.c_uint32),
                ('size', C.c_size_t), ('members', C.POINTER(_Member)),
                ('init', C.c_void_p), ('fini', C.c_void_p)]


def _inspect_schema(pointer, depth=0):
    require(bool(pointer) and depth <= 2, 'Invalid introspection nesting')
    handle = C.cast(pointer, C.POINTER(_TypeSupport)).contents
    require(handle.identifier == b'rosidl_typesupport_introspection_cpp' and bool(handle.data),
            'MC introspection backend differs')
    members = C.cast(handle.data, C.POINTER(_Members)).contents
    require(0 < members.count <= 4 and members.size <= 256 and bool(members.members),
            'MC introspection layout differs')
    fields = []
    for member in members.members[:members.count]:
        field = dict(name=member.name.decode(), type_id=member.type_id, offset=member.offset,
                     is_array=member.is_array, array_size=member.array_size,
                     is_upper_bound=member.is_upper_bound, string_bound=member.string_bound)
        if member.type_id == 18:  # ROS_TYPE_MESSAGE
            field['message'] = _inspect_schema(member.members, depth + 1)
        else:
            require(not member.members, 'Unexpected nested primitive type')
        fields.append(field)
    return dict(namespace=members.namespace.decode(), name=members.name.decode(),
                size=members.size, fields=fields)


def _validate_schema(value):
    def field(name, type_id, offset, *, array=False, message=None):
        result = dict(name=name, type_id=type_id, offset=offset, is_array=array,
                      array_size=0, is_upper_bound=False, string_bound=0)
        if message is not None:
            result['message'] = message
        return result

    # ROS_TYPE_FLOAT=1, UINT32=12, INT32=13, STRING=16, MESSAGE=18.
    stamp = dict(namespace='builtin_interfaces::msg', name='Time', size=8,
                 fields=[field('sec', 13, 0), field('nanosec', 12, 4)])
    header = dict(namespace='aimdk_msgs::msg', name='MessageHeader', size=56,
                  fields=[field('stamp', 18, 0, message=stamp), field('frame_id', 16, 8),
                          field('sequence', 12, 40), field('meas_stamp', 18, 44, message=stamp)])
    expected = dict(namespace='aimdk_msgs::msg', name='McRLDebug', size=104,
                    fields=[field('header', 18, 0, message=header),
                            field('obs_vec', 1, 56, array=True), field('action_vec', 1, 80, array=True)])
    require(value == expected, 'MC McRLDebug/MessageHeader/Time schema differs from the decoder')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _loaded_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'libaimdk_msgs__' in line:
            path = line.split(maxsplit=5)[-1]
            require(path.startswith('/') and not path.endswith(' (deleted)'), 'Invalid loaded MC library')
            paths.add(Path(path).resolve())
    require(paths and all(path.parent == MC_LIB.resolve() for path in paths),
            'Mixed SDK/MC message libraries; isolate this diagnostic process')
    return {str(path): _sha(path) for path in sorted(paths)}


def _raw_message_type():
    require(C.sizeof(C.c_void_p) == 8, 'Only the inspected 64-bit MC ABI is supported')
    libraries = {name: C.CDLL(str(MC_LIB / ('libaimdk_msgs__' + name + '.so')))
                 for name in ('rosidl_typesupport_c', 'rosidl_generator_c', 'rosidl_generator_py',
                              'rosidl_typesupport_introspection_cpp')}
    get_schema = getattr(libraries['rosidl_typesupport_introspection_cpp'],
        'rosidl_typesupport_introspection_cpp__get_message_type_support_handle__aimdk_msgs__msg__McRLDebug')
    get_schema.argtypes = []; get_schema.restype = C.c_void_p
    schema = _inspect_schema(get_schema())
    _validate_schema(schema)
    get_support = getattr(libraries['rosidl_typesupport_c'],
        'rosidl_typesupport_c__get_message_type_support_handle__aimdk_msgs__msg__McRLDebug')
    get_support.argtypes = []; get_support.restype = C.c_void_p
    capsule = C.pythonapi.PyCapsule_New
    capsule.argtypes = (C.c_void_p, C.c_char_p, C.c_void_p)
    capsule.restype = C.py_object

    def native(library, symbol):
        return capsule(C.cast(getattr(libraries[library], symbol), C.c_void_p), None, None)

    class Meta(type):
        _TYPE_SUPPORT = capsule(get_support(), None, None)
        _CREATE_ROS_MESSAGE = native('rosidl_generator_c', 'aimdk_msgs__msg__McRLDebug__create')
        _DESTROY_ROS_MESSAGE = native('rosidl_generator_c', 'aimdk_msgs__msg__McRLDebug__destroy')
        _CONVERT_FROM_PY = native('rosidl_generator_py', 'aimdk_msgs__msg__mc_rl_debug__convert_from_py')
        _CONVERT_TO_PY = native('rosidl_generator_py', 'aimdk_msgs__msg__mc_rl_debug__convert_to_py')

    class McRLDebug(metaclass=Meta):
        pass

    return McRLDebug, schema, libraries  # Keep every native library alive until shutdown.


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()


def _write_metadata(path, value):
    data = _json_bytes(value)
    require(len(data) <= MAX_METADATA_BYTES, 'Metadata exceeds file budget')
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def run(args):
    output = Path(args.output).absolute()
    stop_file = getattr(args, 'stop_file', None)
    stop_file = Path(stop_file).absolute() if stop_file is not None else None
    if stop_file is not None:
        require(stop_file.name == 'bridge_summary.json' and stop_file.parent == output.parent,
                'Stop file must be bridge_summary.json in the capture run directory')
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    summary = dict(schema=SCHEMA, topic=TOPIC, status='running', reason=None,
        started_monotonic=started, seconds_limit=args.seconds, message_limit=MAX_MESSAGES,
        stop_file=str(stop_file) if stop_file is not None else None,
        log_byte_limit=MAX_LOG_BYTES, received_messages=0, recorded_messages=0, written_messages=0, invalid_messages=0,
        log_bytes=0, obs_lengths={}, action_lengths={}, libraries_sha256={}, errors=[])
    stop_reason = None
    node = None
    ros = None
    old_signals = {}
    libraries = None

    def stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_signals[signum] = signal.signal(signum, stop)
    try:
        with (output / 'messages.jsonl').open('xb', buffering=0) as stream:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            ros = rclpy
            message_type, native_schema, libraries = _raw_message_type()
            summary['libraries_sha256'] = _loaded_libraries()
            ros.init(args=[])
            node = Node('malecns_mc_debug_' + str(os.getpid()), enable_rosout=False, start_parameter_services=False)

            def receive(raw):
                nonlocal stop_reason
                if stop_reason is not None:
                    return
                if time.monotonic() >= started + args.seconds:
                    stop_reason = 'seconds_limit'; return
                summary['received_messages'] += 1
                record = dict(received_monotonic_ns=time.monotonic_ns(),
                              capture_seq=summary['received_messages'], cdr_bytes=len(raw))
                try:
                    record['message'] = decode_cdr(raw)
                except (ValueError, UnicodeError, struct.error) as error:
                    summary['invalid_messages'] += 1
                    record.update(error=type(error).__name__ + ':' + str(error),
                                  cdr_hex=raw[:MAX_CDR_BYTES].hex(),
                                  cdr_truncated=len(raw) > MAX_CDR_BYTES)
                    stop_reason = 'invalid_cdr'
                line = _json_bytes(record)
                if summary['log_bytes'] + len(line) > MAX_LOG_BYTES:
                    stop_reason = 'log_byte_limit'; return
                stream.write(line)
                summary['recorded_messages'] += 1
                summary['log_bytes'] += len(line)
                if 'message' in record:
                    summary['written_messages'] += 1
                    for field, counts in (('obs_vec', 'obs_lengths'), ('action_vec', 'action_lengths')):
                        key = str(len(record['message'][field]))
                        summary[counts][key] = summary[counts].get(key, 0) + 1
                if summary['received_messages'] >= MAX_MESSAGES and stop_reason is None:
                    stop_reason = 'message_limit'

            subscription = node.create_subscription(message_type, TOPIC, receive, qos_profile_sensor_data, raw=True)
            summary['libraries_sha256'] = _loaded_libraries()
            _write_metadata(output / 'metadata.json', dict(schema=SCHEMA, wire_schema=WIRE_SCHEMA,
                native_schema=native_schema, libraries_sha256=summary['libraries_sha256'],
                script_sha256=_sha(__file__), topic=TOPIC, raw_subscription=True,
                ros_domain_id=47, ros_localhost_only=True, rmw_implementation='rmw_fastrtps_cpp',
                seconds_limit=args.seconds, message_limit=MAX_MESSAGES, cdr_byte_limit=MAX_CDR_BYTES,
                vector_length_limit=MAX_VECTOR, log_byte_limit=MAX_LOG_BYTES,
                stop_file=summary['stop_file'],
                metadata_file_byte_limit=MAX_METADATA_BYTES, file_count=3,
                qos='BEST_EFFORT/VOLATILE/KEEP_LAST depth=5; DDS drops are not measured'))
            print(json.dumps(dict(state='ready', output=str(output), topic=TOPIC)), flush=True)
            while stop_reason is None and ros.ok() and time.monotonic() < started + args.seconds:
                if stop_file is not None and stop_file.is_file():
                    stop_reason = 'stop_file'
                    break
                ros.spin_once(node, timeout_sec=min(.1, max(0., started + args.seconds - time.monotonic())))
            stop_reason = stop_reason or ('seconds_limit' if ros.ok() else 'ros_shutdown')
            os.fsync(stream.fileno())
            summary.update(status='failed' if stop_reason == 'invalid_cdr' else 'sealed', reason=stop_reason)
    except Exception as error:
        summary.update(status='failed', reason=type(error).__name__ + ':' + str(error))
        summary['errors'].append(summary['reason'])
    finally:
        for name, cleanup in (('destroy_node', lambda: node.destroy_node() if node is not None else None),
                              ('ros_shutdown', lambda: ros.shutdown() if ros is not None and ros.ok() else None)):
            try:
                cleanup()
            except Exception as error:
                summary['errors'].append(name + ':' + type(error).__name__ + ':' + str(error))
        summary.update(elapsed_seconds=time.monotonic() - started, sealed=True,
                       unrecorded_messages=summary['received_messages'] - summary['recorded_messages'])
        _write_metadata(output / 'summary.json', summary)
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
        print(json.dumps(dict(state=summary['status'], output=str(output),
                              messages=summary['written_messages'], reason=summary['reason'])), flush=True)
    return 1 if summary['status'] == 'failed' or summary['errors'] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-only', action='store_true', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path,
                        help='Read-only stop marker: bridge_summary.json in the capture run directory')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= MAX_SECONDS
            or os.environ.get('ROS_DOMAIN_ID') != '47' or os.environ.get('ROS_LOCALHOST_ONLY') != '1'
            or os.environ.get('RMW_IMPLEMENTATION') != 'rmw_fastrtps_cpp'
            or os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)[0] != str(MC_LIB)
            or sys.platform != 'linux'):
        parser.error('Require Linux, explicit simulation, 0 < seconds <= 120, domain 47, localhost, FastRTPS and MC library prefix')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
