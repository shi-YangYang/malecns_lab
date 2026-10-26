"""Startup sequencing/failure tests only; never open a display, ROS or simulator."""
from types import SimpleNamespace
import json

import pytest

from malecns_lab.experiments import x2_sim_startup as startup
from malecns_lab.experiments.x2_sim_startup import Cancellation, prepare, reset_upright
from malecns_lab.runtime.x2.behavior import BehaviorRuntime, concise_failure
from malecns_lab.runtime.x2.desktop import StopRequested

UPRIGHT = (0., 0., .68, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.)


def observation(frozen, pose=UPRIGHT):
    return dict(samples=50, frozen=frozen, first=pose, last=pose)


def run(observations):
    keys, stages = [], []
    samples = iter(observations)
    prepare(SimpleNamespace(press=keys.append), SimpleNamespace(observe=lambda: next(samples)),
            lambda stage, value: stages.append((stage, value)))
    return keys, stages


@pytest.mark.parametrize('initially_paused', [False, True])
def test_startup_resets_paused_sim_and_checks_resume(initially_paused):
    keys, stages = run([observation(initially_paused), observation(True), observation(True), observation(False)])
    assert keys == ([] if initially_paused else ['space']) + ['BackSpace', 'space']
    assert [stage for stage, _ in stages] == ['initial', 'paused', 'reset', 'resumed']


@pytest.mark.parametrize('samples, message', [
    ([observation(False), observation(False)], 'did not pause'),
    ([observation(True), observation(True), observation(False)], 'reset upright'),
    ([observation(True), observation(True), observation(True, (0., 0., .2, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.))], 'reset upright'),
    ([observation(True), observation(True), observation(True), observation(True)], 'did not resume'),
])
def test_failed_feedback_cannot_report_ready(samples, message):
    with pytest.raises(RuntimeError, match=message):
        run(samples)


def test_no_feedback_cannot_trigger_reset():
    keys = []
    def missing():
        raise RuntimeError('No fresh simulation odometry')
    with pytest.raises(RuntimeError, match='No fresh'):
        prepare(SimpleNamespace(press=keys.append), SimpleNamespace(observe=missing), lambda *args: None)
    assert keys == []
    assert reset_upright(UPRIGHT)
    assert not reset_upright((float('nan'), *UPRIGHT[1:]))


def test_manual_runtime_does_not_automate_simulator():
    assert BehaviorRuntime.prepare_simulator_ui(SimpleNamespace(automatic_start=False)) is False


@pytest.mark.parametrize('filename', ['emergency.json', 'sim_startup_cancel.json'])
@pytest.mark.parametrize('stage, expected_keys', [('initial', []), ('paused', ['space']), ('reset', ['space', 'BackSpace'])])
def test_cancellation_prevents_next_shortcut(tmp_path, filename, stage, expected_keys):
    keys = []
    samples = iter([observation(False), observation(True), observation(True), observation(False)])
    check = Cancellation(tmp_path, 'run-a', 'session-a')
    def record(current, value):
        if current == stage:
            (tmp_path/filename).write_text(json.dumps(dict(run_id='run-a', session_id='session-a')))
    with pytest.raises(InterruptedError, match='cancelled'):
        prepare(SimpleNamespace(press=keys.append), SimpleNamespace(observe=lambda: next(samples)), record, check)
    assert keys == expected_keys


@pytest.mark.parametrize('value', ['broken json', json.dumps(dict(run_id='foreign', session_id='session-a'))])
def test_untrusted_cancellation_fails_closed(tmp_path, value):
    (tmp_path/'sim_startup_cancel.json').write_text(value)
    with pytest.raises(RuntimeError):
        Cancellation(tmp_path, 'run-a', 'session-a')()


@pytest.mark.parametrize('cancel_after_spins', [1, 12])
def test_observation_checks_cancel_during_drain_and_sample_windows(monkeypatch, cancel_after_spins):
    now, spins = [0.], []
    monkeypatch.setattr(startup.time, 'monotonic', lambda: now[0])
    def spin(node, timeout_sec):
        assert timeout_sec == .03
        spins.append(node)
        now[0] += .03
    def check():
        if len(spins) >= cancel_after_spins:
            raise InterruptedError('cancelled')
    feedback = SimpleNamespace(ros=SimpleNamespace(spin_once=spin), node=object(),
                               samples=[], check_cancelled=check)
    with pytest.raises(InterruptedError, match='cancelled'):
        startup.Odometry.observe(feedback)
    assert len(spins) == cancel_after_spins


def test_window_checks_cancellation_before_touching_display():
    def check():
        raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError, match='cancelled'):
        startup.SimulatorWindow.press(SimpleNamespace(check_cancelled=check), 'space')


@pytest.mark.parametrize('error', [StopRequested('user_stop'), TimeoutError('command timeout')])
def test_runtime_publishes_owned_cancellation_on_command_interruption(tmp_path, error):
    def command(*args, **kwargs):
        assert kwargs['timeout'] == 17.
        assert 'timeout --kill-after=1s 15s' in args[0][-1]
        raise error
    runtime = SimpleNamespace(automatic_start=True, output=tmp_path, remote='/agibot/data/var/run-a',
        run_id='run-a', session_id='session-a', docker=['docker'], errors=[],
        status=lambda *args, **kwargs: None, command=command)
    with pytest.raises(type(error)):
        BehaviorRuntime.prepare_simulator_ui(runtime)
    value = json.loads((tmp_path/'sim_startup_cancel.json').read_text())
    assert (value['run_id'], value['session_id']) == ('run-a', 'session-a')
    assert not runtime.errors


def test_failure_message_keeps_specific_reason_without_traceback():
    assert concise_failure('Traceback (most recent call last):\n  file.py:1\nRuntimeError: bad window') == 'bad window'
    assert concise_failure('ValueError:Fresh stable standing not confirmed') == '未能确认机器人稳定站立'
    assert concise_failure('ValueError:仿真自动准备失败：No fresh simulation odometry') == '仿真自动准备失败：No fresh simulation odometry'


@pytest.mark.parametrize('executable, argv, translated', [
    ('/workspace/sim_mujoco/bin/aima-sim-app', ['./aima-sim-app'], False),
    ('/mnt/rosetta/rosetta', ['./aima-sim-app'], True),
    ('/mnt/rosetta/rosetta', ['/workspace/sim_mujoco/bin/aima-sim-app'], True),
    ('/mnt/rosetta/rosetta', ['/mnt/rosetta/rosetta', './aima-sim-app'], True),
])
def test_process_identity_accepts_exact_official_program(executable, argv, translated):
    identity = startup.simulator_process_identity(executable, '/workspace/sim_mujoco/bin', argv)
    assert identity['program'] == '/workspace/sim_mujoco/bin/aima-sim-app'
    assert identity['executable'] == executable
    assert identity['translated'] is translated
    if translated:
        assert identity['argv_prefix'] == argv


@pytest.mark.parametrize('executable, cwd, argv', [
    ('/mnt/rosetta/rosetta', '/tmp', ['./aima-sim-app']),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', ['python3', './aima-sim-app']),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', ['/mnt/rosetta/rosetta', 'wrong', './aima-sim-app']),
    ('/tmp/rosetta', '/workspace/sim_mujoco/bin', ['./aima-sim-app']),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', []),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', ['/mnt/rosetta/rosetta']),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', ['/tmp/prefix-workspace/sim_mujoco/bin/aima-sim-app']),
    ('/mnt/rosetta/rosetta', '/workspace/sim_mujoco/bin', ['./aima-sim-app-fake']),
])
def test_process_identity_rejects_wrong_launcher_or_program(executable, cwd, argv):
    with pytest.raises(RuntimeError):
        startup.simulator_process_identity(executable, cwd, argv)


def discovery_feedback(monkeypatch, first_at, *, single=False, cancel_at=None):
    now, emitted = [0.], [False]
    feedback = SimpleNamespace(node=object(), samples=[], awaiting_first_feedback=True)
    monkeypatch.setattr(startup.time, 'monotonic', lambda: now[0])
    def spin(node, timeout_sec):
        assert 0 < timeout_sec <= .03
        now[0] += timeout_sec
        if first_at is not None and now[0] >= first_at and (not single or not emitted[0]):
            feedback.samples.append((now[0],))
            emitted[0] = True
    def check():
        if cancel_at is not None and now[0] >= cancel_at:
            raise InterruptedError('cancelled')
    feedback.ros, feedback.check_cancelled = SimpleNamespace(spin_once=spin), check
    return feedback, now


def test_initial_discovery_waits_for_delayed_feedback_then_uses_full_fresh_window(monkeypatch):
    feedback, now = discovery_feedback(monkeypatch, 1.5)
    observed = startup.Odometry.observe(feedback)
    assert 2.4 <= now[0] < 2.5
    assert observed['samples'] >= 20
    assert observed['first'][0] >= 1.8  # Discovery/drain values were discarded.
    assert observed['last'][0] - observed['first'][0] >= .57
    assert not feedback.awaiting_first_feedback


def test_initial_discovery_without_feedback_stops_with_fixed_deadline(monkeypatch):
    feedback, now = discovery_feedback(monkeypatch, None)
    with pytest.raises(RuntimeError, match='initial discovery'):
        startup.Odometry.observe(feedback)
    assert 5. <= now[0] < 5.04


def test_initial_discovery_does_not_accept_one_stale_sample(monkeypatch):
    feedback, now = discovery_feedback(monkeypatch, 1.5, single=True)
    with pytest.raises(RuntimeError, match='No fresh simulation odometry'):
        startup.Odometry.observe(feedback)
    assert feedback.samples == []
    assert 2.4 <= now[0] < 2.5


def test_initial_discovery_can_be_cancelled_between_short_spins(monkeypatch):
    feedback, now = discovery_feedback(monkeypatch, None, cancel_at=.06)
    with pytest.raises(InterruptedError, match='cancelled'):
        startup.Odometry.observe(feedback)
    assert now[0] == .06
