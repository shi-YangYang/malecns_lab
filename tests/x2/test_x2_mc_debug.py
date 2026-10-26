"""Offline wire/capture checks; these tests never load ROS or native MC libs."""
import json
import math
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from malecns_lab.experiments.x2_mc_debug import MAX_CDR_BYTES, MAX_VECTOR, decode_cdr


def encoded(*, endian='<', frame='pelvis', obs=None, actions=None):
    payload = bytearray()

    def scalar(code, value):
        payload.extend(b'\xa5' * (-len(payload) % 4))
        payload.extend(struct.pack(endian + code, value))

    scalar('i', -3); scalar('I', 123456789)
    string = frame.encode() + b'\0'
    scalar('I', len(string)); payload.extend(string)
    scalar('I', 0xffffffff)
    scalar('i', 42); scalar('I', 999999999)
    for values in ([i / 8. for i in range(111)] if obs is None else obs,
                   [-.25, .5, 1.] if actions is None else actions):
        scalar('I', len(values))
        for value in values:
            scalar('f', value)
    return (b'\0\1\0\0' if endian == '<' else b'\0\0\0\0') + payload


@pytest.mark.parametrize('endian,frame', [('<', 'pelvis'), ('>', 'a'), ('<', ''), ('>', '骨盆')])
def test_exact_header_alignment_and_complete_vectors(endian, frame):
    result = decode_cdr(encoded(endian=endian, frame=frame))
    assert result['header'] == dict(stamp=dict(sec=-3, nanosec=123456789), frame_id=frame,
                                   sequence=0xffffffff, meas_stamp=dict(sec=42, nanosec=999999999))
    assert result['obs_vec'] == [i / 8. for i in range(111)]
    assert result['action_vec'] == [-.25, .5, 1.]
    assert decode_cdr(encoded(obs=[], actions=[]))['obs_vec'] == []


@pytest.mark.parametrize('raw', [b'', b'\0\7\0\0', b'\0\1\0\1' + encoded()[4:],
    encoded()[:-1], encoded() + b'\0', b'x' * (MAX_CDR_BYTES + 1),
    encoded(obs=[math.nan]), encoded(actions=[math.inf]), encoded(obs=[0.] * (MAX_VECTOR + 1))])
def test_rejects_unmatched_or_unbounded_wire_data(raw):
    with pytest.raises(ValueError):
        decode_cdr(raw)


def test_rejects_invalid_header_time_and_string_without_decoding_arrays():
    bad_time = bytearray(encoded()); struct.pack_into('<I', bad_time, 8, 1_000_000_000)
    bad_string = bytearray(encoded()); bad_string[22] = 65  # The frame_id terminating NUL.
    for raw in (bytes(bad_time), bytes(bad_string)):
        with pytest.raises(ValueError):
            decode_cdr(raw)


def test_stop_file_seals_capture_without_waiting_for_seconds_limit(tmp_path, monkeypatch):
    from malecns_lab.experiments import x2_mc_debug as capture

    output = tmp_path / 'mc_debug_left'
    stop_file = tmp_path / 'bridge_summary.json'
    state = dict(ok=True, spins=0, destroyed=False)
    ros = ModuleType('rclpy')
    node_module = ModuleType('rclpy.node')
    qos_module = ModuleType('rclpy.qos')

    class FakeNode:
        def __init__(self, *args, **kwargs):
            pass

        def create_subscription(self, message_type, topic, receive, qos, **kwargs):
            state['receive'] = receive

        def destroy_node(self):
            state['destroyed'] = True

    def spin_once(node, timeout_sec):
        state['spins'] += 1
        assert state['spins'] == 1, 'Capture continued after the bridge wrote its summary'
        assert 0 <= timeout_sec <= .1
        state['receive'](encoded())
        stop_file.write_text('{"status":"FAILED"}\n')

    ros.init = lambda **kwargs: None
    ros.ok = lambda: state['ok']
    ros.spin_once = spin_once
    ros.shutdown = lambda: state.update(ok=False)
    node_module.Node = FakeNode
    qos_module.qos_profile_sensor_data = object()
    for name, module in [('rclpy', ros), ('rclpy.node', node_module), ('rclpy.qos', qos_module)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(capture, '_raw_message_type', lambda: (object, {}, {}))
    monkeypatch.setattr(capture, '_loaded_libraries', lambda: {})

    assert capture.run(SimpleNamespace(output=output, stop_file=stop_file, seconds=60.)) == 0
    summary = json.loads((output / 'summary.json').read_text())
    metadata = json.loads((output / 'metadata.json').read_text())
    assert summary['status'] == 'sealed' and summary['reason'] == 'stop_file'
    assert summary['stop_file'] == metadata['stop_file'] == str(stop_file)
    assert summary['written_messages'] == 1 and summary['errors'] == []
    assert len((output / 'messages.jsonl').read_text().splitlines()) == 1
    assert stop_file.read_text() == '{"status":"FAILED"}\n'
    assert state['destroyed'] and not state['ok']
