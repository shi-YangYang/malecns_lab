"""Old official messages only: one fixed preset request, bounded stdin/stdout IPC."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import sys
import time

from . import x2_preset_identity as identity
from .x2_preset_compat import MAX_LINE, SERVICE, decode, encode, raw_request


def run(args):
    seq = 0
    node = None
    initialized = False
    started = time.monotonic()
    buffer = bytearray()

    def emit(kind, payload):
        nonlocal seq
        line = encode(kind, payload, nonce=args.nonce, run=args.run, probe=args.probe, seq=seq)
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        seq += 1

    def incoming():
        try:
            chunk = os.read(sys.stdin.fileno(), MAX_LINE + 1)
        except BlockingIOError:
            return None
        if not chunk:
            raise ValueError('Parent pipe closed')
        buffer.extend(chunk)
        if len(buffer) > MAX_LINE:
            raise ValueError('IPC input exceeds budget')
        if b'\n' not in buffer:
            return None
        line, _, rest = buffer.partition(b'\n')
        if rest:
            raise ValueError('Duplicate/trailing IPC input')
        buffer.clear()
        return decode(bytes(line)+b'\n', nonce=args.nonce, run=args.run, probe=args.probe, seq=0)

    try:
        root = identity.LINUX_ROOT
        binding = identity.verify_build(root)
        prepared = identity.read(root / 'prepared.json')
        for name in ('x2_preset_identity.py', 'x2_preset_compat.py', 'x2_preset_helper.py'):
            if identity.sha(Path(__file__).with_name(name)) != prepared['tool_files']['malecns_lab/experiments/' + name]:
                raise ValueError('Helper source differs from inspected build tools')
        actual = identity.runtime_identity(root, 'v1.0.0')
        import rclpy
        from rclpy.node import Node
        from rclpy.serialization import serialize_message
        from rosidl_runtime_py.convert import message_to_ordereddict
        from aimdk_msgs.srv import SetMcPresetMotion
        rclpy.init()
        initialized = True
        node = Node('malecns_preset_compat_' + args.nonce[:8])
        client = node.create_client(SetMcPresetMotion, SERVICE)
        # After client construction, assert the actual loaded middleware types
        # still live exclusively in the old install prefix.
        actual = identity.runtime_identity(root, 'v1.0.0')
        os.set_blocking(sys.stdin.fileno(), False)
        while not client.service_is_ready() and time.monotonic() < started + 12.:
            if incoming() is not None:
                raise ValueError('Request before ready')
            rclpy.spin_once(node, timeout_sec=.01)
        if not client.service_is_ready():
            raise TimeoutError('preset_service_unavailable')
        emit('ready', dict(binding=binding, identity=actual))
        request_message = None
        while time.monotonic() < started + 12.:
            request_message = incoming()
            if request_message is not None:
                break
            rclpy.spin_once(node, timeout_sec=.01)
        if request_message is None or request_message['kind'] != 'request':
            raise ValueError('Missing single request')
        request = SetMcPresetMotion.Request()
        request.header.stamp = node.get_clock().now().to_msg()
        request.area.value, request.motion.value = 2, 1002
        request.interrupt, request.ani_path, request.play_timestamp = False, '', 0
        raw = message_to_ordereddict(request)
        raw_request(raw)
        cdr_hex = serialize_message(request).hex()
        future = client.call_async(request)  # The only action send in this process.
        sent = time.monotonic()
        emit('sent', dict(message=raw, cdr_hex=cdr_hex))
        while not future.done() and time.monotonic() - sent < 2.:
            if incoming() is not None:
                raise ValueError('Duplicate request')
            rclpy.spin_once(node, timeout_sec=.005)
        if not future.done() or time.monotonic() - sent >= 2.:
            raise TimeoutError('preset_ack_timeout')
        # Drain once even for an immediately completed fake/real response.
        if incoming() is not None:
            raise ValueError('Duplicate request')
        response = future.result()
        if response is None:
            raise ValueError('Empty preset response')
        emit('response', dict(message=message_to_ordereddict(response), cdr_hex=serialize_message(response).hex()))
        return 0
    except BaseException as error:
        try:
            emit('error', {'error': (type(error).__name__ + ':' + str(error))[:2048]})
        except BaseException:
            pass
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if initialized:
            rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nonce', 'run', 'probe'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    if (not re.fullmatch('[a-f0-9]{32}', args.nonce)
            or not re.fullmatch(r'skills-\d{8}-\d{6}-[a-f0-9]{8,32}', args.run)
            or not re.fullmatch(r'probe-\d{2}-wave_right', args.probe)
            or os.environ.get('ROS_DOMAIN_ID') != '47' or os.environ.get('ROS_LOCALHOST_ONLY') != '1'
            or os.environ.get('RMW_IMPLEMENTATION') != 'rmw_fastrtps_cpp'):
        parser.error('Invalid fixed simulation identity/environment')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
