"""CPU protocol checks; no simulator, model, or runtime execution."""
import io
import json
import queue
import threading

from malecns_lab.runtime.x2.behavior import Commands


def test_clock_reply_is_scoped_and_blocked_output_cannot_delay_emergency_stop():
    inputs, replies = queue.Queue(), []
    entered, release = threading.Event(), threading.Event()

    class Input:
        def readline(self, _limit):
            return inputs.get(timeout=2)

    def emit(value):
        replies.append(value)
        entered.set()
        assert release.wait(2)

    commands = Commands(Input())
    commands.start_clock_replies('run', emit)
    commands.start()
    try:
        inputs.put(json.dumps(dict(command='clock', run_id='run', nonce='a'*32))+'\n')
        assert entered.wait(1)
        response = replies[0]
        assert response['schema'] == 'malecns_lab.desktop_clock.v1'
        assert response['type'] == 'clock' and response['run_id'] == 'run'
        assert response['nonce'] == 'a'*32 and response['clock_id'] == commands.clock_id
        assert int(response['python_monotonic_ns']) > 0
        assert response['clock_source']['api'] == 'time.monotonic_ns'
        inputs.put('{"command":"stop"}\n')
        assert commands.stop.wait(.5)
        assert commands.reason == 'user_emergency_stop'
        assert commands.tasks.empty() and commands.pending.empty()
    finally:
        release.set()
        inputs.put('')


def test_clock_queue_is_bounded_and_rejects_foreign_or_missing_nonce():
    for request in (dict(command='clock', run_id='other', nonce='a'*32),
                    dict(command='clock', run_id='run'),
                    dict(command='clock', run_id='run', nonce='bad')):
        commands = Commands(io.StringIO(json.dumps(request)+'\n'))
        commands.clock_run_id = 'run'
        commands.read()
        assert commands.stop.is_set()
        assert commands.clock_requests.empty() and commands.tasks.empty()
    command = json.dumps(dict(command='clock', run_id='run', nonce='b'*32))+'\n'
    commands = Commands(io.StringIO(command*10+'{"command":"stop"}\n'))
    commands.clock_run_id = 'run'
    commands.read()
    assert commands.clock_requests.qsize() == 4
    assert commands.reason == 'user_emergency_stop'
