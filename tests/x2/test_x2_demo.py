"""Focused manual-demo IPC/executor checks; no ROS, VM, model or hardware."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.runtime.x2.demo import Commands, DemoRuntime, TRANSPORT
from malecns_lab.experiments.x2_demo import ManualDispatcher, TASK_SKILLS, base_stop_basis, demo_executor, DEMO_MOTION, StandPreparation
from malecns_lab.control.x2_skill_profile import candidate_skill_profile
from malecns_lab.control.x2_interaction import interaction_profile
from malecns_lab.control.x2_skills import RIGHT_ARM, SOURCE
from .test_x2_skills import Rig


def rig(source=SOURCE):
    r = Rig(qualified=True)
    r.source = source
    r.core = demo_executor(dict(skill_profile=candidate_skill_profile(wave_enabled=True),
        interaction_profile=interaction_profile(wave_enabled=True), simulation_only=True, qualification_only=True,
        demo_motion=DEMO_MOTION),
        r.commands.append, clock=lambda:r.t, preset=r.presets.append,
        wave_poses=dict(first=dict.fromkeys(RIGHT_ARM,0.), terminal=dict.fromkeys(RIGHT_ARM,0.)))
    r.core.motion.arm()
    r.core.motion.confirm_source_registration()
    for _ in range(65):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    return r


def test_idle_empty_source_stop_uses_full_fresh_window_but_never_relaxes_motion_ownership():
    r = rig(source='')
    assert r.core.motion.state == 'STOPPING'  # original owned-source algorithm stays intact
    r.core.motion.request_stop('manual_emergency')
    for _ in range(40):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    assert base_stop_basis(r.core) is None  # less than one second after stop
    for _ in range(25):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    assert base_stop_basis(r.core) == 'official_unclaimed_idle_standing_after_stop'
    assert r.core.motion.state == 'STOPPING'  # measured proof, never a fabricated state transition
    assert not r.core.motion.failures and all(command == (0.,0.,0.) for command in r.commands)
    r.t += .6
    r.core.tick()
    assert base_stop_basis(r.core) is None  # stale data cannot preserve stopped
    r.source = 'foreign'
    for _ in range(65):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    assert base_stop_basis(r.core) is None
    r.source = ''
    for _ in range(65):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    assert dispatch.submit(request(1,'SHOW_FORWARD'))
    r.source = SOURCE
    r.feed()
    r.core.tick()
    r.core.motion.request_stop('manual_end')
    r.source = ''
    for _ in range(65):
        r.feed()
        r.core.tick()
    assert base_stop_basis(r.core) is None
    assert r.core.motion.ready_reason(r.t) is not None


def test_completed_wave_without_base_request_can_end_with_fresh_empty_source_but_motion_cannot():
    r = rig(source='')
    r.core.command = SimpleNamespace(behavior='WAVE_RIGHT')
    r.core.terminal = True
    r.core.status = 'COMPLETED'
    r.core.wave_return_confirmed = True
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    assert dispatch.submit(request(1,'END_INTERACTION'))
    assert base_stop_basis(r.core) is None
    for _ in range(65):
        r.feed();r.core.standing_ready();r.core.motion.tick()
    assert base_stop_basis(r.core) == 'official_unclaimed_completed_stationary_skill_after_stop'
    assert r.core.motion.state == 'STOPPING'
    # Exercise END's exact completion predicate without constructing fictitious
    # wave request/event metadata for core.summary().
    assert (r.core.command is None or r.core.terminal) and base_stop_basis(r.core) and r.core.standing_ready()
    r.core.wave_return_confirmed = False
    assert base_stop_basis(r.core) is None
    r.core.wave_return_confirmed = True
    r.core.motion.base_motion_requested = True
    assert base_stop_basis(r.core) is None


def request(seq, task):
    return dict(sequence=seq, task=task, request_id='request-'+str(seq))


def test_host_reserves_action_rejects_busy_and_end_preempts_without_queue():
    commands = Commands(io.StringIO(''))
    assert not commands.task('GREET')
    commands.remote_state(dict(sequence=0, busy=False))
    assert commands.task('SHOW_FORWARD')
    assert not commands.task('FACE_LEFT')
    commands.remote_state(dict(sequence=0, busy=False))
    assert commands.busy  # stale idle cannot reopen a newly reserved slot
    assert commands.task('END_INTERACTION')
    assert commands.take()['task'] == 'END_INTERACTION'
    assert commands.take() is None
    commands.remote_state(dict(sequence=2, busy=False))
    assert commands.task('GREET')


@pytest.mark.parametrize('line,reason', [('', 'desktop_control_pipe_closed'),
    ('{"command":"stop"}\n', 'user_emergency_stop'), ('{"command":"unknown"}\n','desktop_command_error')])
def test_input_stop_eof_invalid_fail_closed(line, reason):
    commands = Commands(io.StringIO(line))
    commands.read()
    assert commands.stop.is_set() and commands.reason == reason


def test_clock_command_matches_existing_desktop_and_reply_queue():
    nonce = 'a'*32
    commands = Commands(io.StringIO(json.dumps(dict(command='clock',run_id='run',nonce=nonce))+'\n'))
    commands.clock_run_id = 'run'
    commands.read()
    assert commands.clock_requests.get_nowait() == nonce
    assert commands.reason == 'desktop_control_pipe_closed'


def test_outer_error_emits_complete_desktop_status(tmp_path, monkeypatch, capsys):
    import malecns_lab.runtime.x2.demo as demo
    monkeypatch.setattr(demo.desktop, 'output_directory', lambda path: tmp_path/'demo')
    def fail(root):
        raise RuntimeError('lock unavailable')
    monkeypatch.setattr(demo.desktop, 'desktop_lock', fail)
    assert demo.main(['--execute','--sim-only','--output',str(tmp_path/'demo')]) == 1
    event = json.loads(capsys.readouterr().out)
    assert event['simulator_ready'] is False and event['motion_started'] is False
    assert event['stop_confirmed'] is None and event['resources_stopped'] is True
    assert event['mode'] == 'manual_demo' and event['brain_controlled'] is False


@pytest.mark.parametrize('task,axis,sign,seconds', [('SHOW_FORWARD',0,1,1.), ('FACE_LEFT',2,1,2.), ('FACE_RIGHT',2,-1,2.)])
def test_manual_motion_actual_executor_mapping_completion_and_repeated_task(task, axis, sign, seconds):
    r = rig()
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    assert dispatch.submit(request(1,task))
    assert r.core.command.behavior == TASK_SKILLS[task]
    assert r.core.active_command_seconds == seconds
    assert r.core.resolved_velocity[axis] == sign*(.2 if axis == 0 else .4)
    assert dispatch.state()['busy']
    assert not dispatch.submit(request(2,'GREET'))
    while r.t < r.core.deadline-.025:
        if axis == 0:
            r.x += .004
        else:
            r.a += sign*.004
        r.feed(.2 if axis == 0 else 0., sign*.2 if axis == 2 else 0.)
        r.core.tick()
    for _ in range(120):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    state = dispatch.state()
    assert state['phase'] == 'waiting_task' and not state['busy']
    assert state['execution']['status'] == 'COMPLETED'
    assert state['execution']['measured_motion']
    assert dispatch.submit(request(3,task))


@pytest.mark.parametrize('task,sign', [('FACE_LEFT',1),('FACE_RIGHT',-1)])
def test_manual_turn_feedback_stop_never_restarts_or_weakens_spatial_fault(task,sign):
    r = rig()
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    assert dispatch.submit(request(1,task))
    deadline = r.core.deadline
    assert r.core.resolved_velocity == (0.,0.,sign*.4)
    while r.t < deadline-.2 and abs(r.a) < .36:
        r.a += sign*.008
        r.feed(wz=sign*.4)
        r.core.tick()
        dispatch.state()
    assert r.core.stop_at is not None and r.t < deadline
    assert not r.core.cancel_requested and r.core.motion.reason == 'manual_turn_target'
    before = len(r.commands)
    for _ in range(130):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
        dispatch.state()
    assert all(command == (0.,0.,0.) for command in r.commands[before:])
    assert r.core.status == 'COMPLETED' and r.core.motion.state == 'STOPPED'
    assert r.core.signed_angle*sign >= .35
    assert dispatch.submit(request(2,task))
    r.core.signed_angle = sign*.66
    with pytest.raises(RuntimeError,match='spatial limit'):
        dispatch.state()
    assert 'manual_spatial_limit' in r.core.motion.failures


def test_end_is_independent_and_waits_for_actual_stop_before_rearming():
    r = rig()
    resets = []
    def reset():
        resets.append(r.core.summary())
        replacement = rig()
        return replacement.core
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t, reset_after_end=reset)
    assert dispatch.submit(request(1,'SHOW_FORWARD'))
    assert dispatch.submit(request(2,'END_INTERACTION'))
    assert r.core.cancel_requested
    assert dispatch.state()['phase'] == 'stopping_action'
    assert not resets
    for _ in range(120):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    assert dispatch.state()['phase'] == 'stopping_action'
    assert resets[0]['motion_state'] == 'STOPPED'
    # The real terminal record is preserved even if no displacement preceded END.
    assert resets[0]['status'] == 'FAILED' and not resets[0]['measured_motion']
    assert dispatch.state()['phase'] == 'waiting_task'


def test_wave_uses_qualified_official_1002_and_ack_does_not_finish():
    r = rig()
    r.core.preset_query(0,0,1,received=r.t)
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    assert dispatch.submit(request(1,'GREET'))
    assert r.presets == [dict(motion=1002,area=2,interrupt=False)]
    assert r.core.wave_qualified and not r.core.terminal
    r.core.preset_dispatch()
    r.core.preset_sent(r.t)
    r.core.preset_ack(0,0,400)
    assert dispatch.state()['busy'] and not r.core.moved and not r.core.terminal
    assert dispatch.submit(request(2,'END_INTERACTION'))
    assert dispatch.state()['phase'] == 'stopping_action'


def test_sd_once_requires_success_and_post_response_standing_and_end_cancels_wait():
    class Future:
        complete = False
        cancelled = False
        def done(self): return self.complete
        def result(self): return SimpleNamespace(response=SimpleNamespace(header=SimpleNamespace(code=0), status=SimpleNamespace(value=1)))
        def cancel(self): self.cancelled = True
    class Client:
        def __init__(self): self.sent = []; self.future = Future()
        def service_is_ready(self): return True
        def call_async(self, request): self.sent.append(request); return self.future
    r = rig()
    r.core.preset_query(0,0,1,received=r.t)
    client = Client()
    preparation = StandPreparation(client, lambda: 'SD', lambda value: str(value), lambda *a,**kw: None, clock=lambda:r.t)
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t, wave_preparation=preparation)
    assert dispatch.submit(request(1,'GREET'))
    assert dispatch.state()['message'] == '正在准备挥手'
    for _ in range(3): dispatch.state()
    assert client.sent == ['SD'] and r.presets == []
    assert dispatch.submit(request(2,'END_INTERACTION'))
    assert client.future.cancelled and dispatch.pending_wave is None
    for _ in range(65):
        r.feed();r.core.standing_ready();r.core.tick();dispatch.state()
    assert dispatch.state()['phase'] == 'waiting_task' and r.presets == []
    client.future = Future()
    r.core.preset_query(0,0,1,received=r.t)
    assert dispatch.submit(request(3,'GREET'))
    dispatch.state()
    client.future.complete = True
    dispatch.state()
    assert r.presets == []  # ACK alone does not start the wave
    for _ in range(40):
        r.feed();r.core.standing_ready();r.core.tick();dispatch.state()
    assert r.presets == []
    for _ in range(25):
        r.feed();r.core.preset_query(0,0,1,received=r.t);r.core.standing_ready();r.core.tick();dispatch.state()
    assert client.sent == ['SD','SD']
    assert r.presets == [dict(motion=1002,area=2,interrupt=False)]


def test_protection_is_not_hidden_by_success_or_end():
    r = rig()
    dispatch = ManualDispatcher(r.core, clock=lambda:r.t)
    r.core.motion.failures.append('arm_joint_error')
    with pytest.raises(ValueError, match='arm_joint_error'):
        dispatch.state()


def test_transport_manifest_single_publication_and_mode_status(tmp_path, monkeypatch):
    # Actual prepare_transport, with only expensive asset/helper verification stubbed.
    import malecns_lab.runtime.x2.demo as demo
    root = Path(__file__).resolve().parents[2]
    output = tmp_path/'desktop-20261025-000000-12345678'
    output.mkdir()
    commands = Commands(io.StringIO(''))
    events = []
    runtime = DemoRuntime(output,root,commands,events.append)
    monkeypatch.setattr(demo, 'ASSETS', ())
    monkeypatch.setattr(demo.BehaviorRuntime, 'prepare_wave_qualification', lambda self: None)
    runtime.prepare_transport()
    manifest = json.loads((output/'transport_manifest.json').read_text())
    assert set(manifest) == set(TRANSPORT)
    assert (output/'host_source.py').read_bytes() == Path(demo.__file__).read_bytes()
    assert json.loads((output/'demo_registration.json').read_text())['brain_controlled'] is False
    assert not (output/'brain_ready.json').exists()
    runtime.status('running','请选择动作',phase='waiting_task',request_pending=False)
    assert events[-1]['mode'] == 'manual_demo' and events[-1]['brain_controlled'] is False
    assert events[-1]['budget']['wall_timeout_seconds'] == 900.
