"""Candidate duration wiring with real control code and CPU-only transport doubles."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace as NS

import pytest

from malecns_lab.control.x2 import SOURCE
from malecns_lab.control.x2_skills import SkillCommand, SkillExecutor, SKILL_PACK_ID
from malecns_lab.control.x2_skill_profile import candidate_skill_profile, PROFILE_ID
from malecns_lab.experiments import x2_behavior_bridge as bridge, x2_behavior_desktop as desktop
from test_x2_skills import Rig


def profile_rig(profile=None, duration=None):
    r = Rig.__new__(Rig)
    r.t = 1.
    r.commands, r.presets = [], []
    r.core = (bridge.candidate_executor({'skill_profile': candidate_skill_profile()}, r.commands.append,
        clock=lambda:r.t) if profile is None and duration is None else
        SkillExecutor(r.commands.append, r.presets.append, clock=lambda:r.t,
            enabled=('HOLD','ADVANCE','TURN_LEFT','TURN_RIGHT'),
            command_seconds=1. if duration is None else duration, command_seconds_by_behavior=profile))
    r.core.motion.arm()
    r.core.motion.confirm_source_registration()
    r.x = r.y = r.a = r.joint = 0.
    r.source, r.player = SOURCE, 0
    for _ in range(65):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    return r


@pytest.mark.parametrize('behavior,seconds,axis,sign', [
    ('ADVANCE',1.,0,1), ('TURN_LEFT',2.,2,1), ('TURN_RIGHT',2.,2,-1)])
def test_candidate_actual_deadlines_and_records_do_not_follow_continue(behavior,seconds,axis,sign):
    r = profile_rig()
    intensity = .37
    assert r.core.submit(SkillCommand(0,'request',behavior,'START',intensity,r.t))
    started, deadline = r.t, r.t+seconds
    assert r.core.deadline == r.core.motion._expires == deadline
    assert r.core.command_seconds == 1.  # Legacy default is not the active per-skill limit.
    for _ in range(20):
        r.feed()
        r.core.tick()
    assert r.core.submit(SkillCommand(1,'request',behavior,'CONTINUE',.91,r.t))
    assert r.core.deadline == r.core.motion._expires == deadline
    assert r.core.command.intensity == intensity
    while r.t < deadline-.03:
        r.feed()
        r.core.tick()
    assert r.commands[-1][axis] == pytest.approx(sign*.2*intensity)
    r.t = deadline
    r.feed()
    r.core.tick()
    assert r.commands[-1] == (0.,0.,0.) and r.core.stop_at >= deadline
    raw = r.core.events[0]
    event = bridge.wrap_execution({'run_id':'run','session_id':'session'},raw,0,[],{})
    assert event.skill_pack_id == SKILL_PACK_ID and event.skill_pack_id != PROFILE_ID
    assert event.mc_request == dict(recent_actual_requests=[],skill_profile_id=PROFILE_ID,
                                   command_seconds=seconds,intensity=intensity)
    for record in (raw,r.core.summary()):
        assert record['skill_profile_id'] == PROFILE_ID and record['command_seconds'] == seconds
        assert record['intensity'] == intensity and record['deadline'] == deadline
        assert record['started_at'] == started


def test_caller_duration_mutation_cannot_renew_or_change_an_admitted_action():
    values = dict(ADVANCE=1.,TURN_LEFT=2.,TURN_RIGHT=2.)
    r = profile_rig(values)
    assert r.submit('TURN_LEFT')
    deadline = r.core.deadline
    values['TURN_LEFT'] = .1
    r.core.command_seconds = 2.  # Active limit was already latched.
    with pytest.raises(TypeError):
        r.core.command_seconds_by_behavior['TURN_LEFT'] = 1.
    r.feed()
    assert r.submit('TURN_LEFT',seq=1,phase='CONTINUE')
    assert r.core.deadline == r.core.motion._expires == deadline
    assert r.core.active_command_seconds == 2.
    assert r.core.summary()['command_seconds_by_behavior']['TURN_LEFT'] == 2.


def test_old_uniform_default_and_single_skill_explicit_duration_remain():
    for duration in (1.,2.):
        r = profile_rig(duration=duration)
        assert r.submit('TURN_LEFT')
        assert r.core.deadline-r.t == pytest.approx(duration)
        assert r.core.motion._expires == r.core.deadline
        assert r.core.summary()['command_seconds'] == duration
        assert r.core.skill_profile_id is None
    hold = profile_rig()
    assert hold.submit('HOLD',phase='FINISH')
    assert hold.core.terminal and hold.core.deadline == hold.t
    assert hold.core.summary()['command_seconds'] is None
    wave = profile_rig()
    assert not wave.submit('WAVE_RIGHT') and not wave.presets


def test_invalid_per_behavior_budgets_are_rejected_before_control_is_constructed():
    valid = dict(ADVANCE=1.,TURN_LEFT=2.,TURN_RIGHT=2.)
    cases = [{}, {k:v for k,v in valid.items() if k != 'TURN_RIGHT'},
             dict(valid,HOLD=1.), [], *[dict(valid,TURN_LEFT=x) for x in
             (0.,-1.,2.001,float('nan'),float('inf'),True,'2',None)]]
    for config in cases:
        with pytest.raises(ValueError):
            SkillExecutor(lambda *_:pytest.fail('publish'),lambda *_:pytest.fail('preset'),
                command_seconds_by_behavior=config)


def test_production_host_registration_and_bridge_constructor_share_fixed_profile(tmp_path,monkeypatch):
    from malecns_lab.learning import resource_guard, x2_behavior_training
    import psutil
    # Only replace resource queries, frozen numerical source inventory and spawn;
    # the production preparation writes both registrations and transport bytes.
    monkeypatch.setattr(x2_behavior_training,'source_inventory',lambda root:{})
    monkeypatch.setattr(resource_guard,'resource_budget',lambda *args,**kwargs:
        dict(can_launch=True,effective_rss_limit_bytes=1024**3))
    monkeypatch.setattr(psutil,'virtual_memory',lambda:NS(available=16*1024**3))
    monkeypatch.setattr(desktop,'ASSETS',())
    runtime = desktop.BehaviorRuntime(tmp_path,desktop.ROOT,desktop.Commands(io.StringIO()),lambda value:None,
        tmp_path/'candidate',dict(checkpoint_id='checkpoint',policy={}))
    ready = dict(checkpoint_id='checkpoint',run_id=runtime.run_id,session_id=runtime.session_id)
    desktop.atomic(tmp_path/'brain_ready.json',ready)
    runtime.spawn = lambda *args:NS(poll=lambda:None)
    try:
        runtime.prepare()
    finally:
        for stream in runtime.logs:
            stream.close()
    host = json.loads((tmp_path/'behavior_registration.json').read_text())
    reg = json.loads((tmp_path/'bridge_registration.json').read_text())
    assert host['skill_profile'] == reg['skill_profile'] == candidate_skill_profile()
    assert 'control/x2_skill_profile.py' in host['source_sha256']
    assert (tmp_path/'transport_src/malecns_lab/control/x2_skill_profile.py').is_file()
    assert bridge.registered_skill_profile(tmp_path,reg) == candidate_skill_profile()
    core = bridge.candidate_executor(reg,lambda *_:pytest.fail('unexpected publish'))
    assert dict(core.command_seconds_by_behavior) == host['skill_profile']['command_seconds_by_behavior']
    assert core.skill_profile_id == PROFILE_ID and not core.wave_probe
    # A refused decision has no executed budget, even if an older command exists.
    r = profile_rig()
    assert r.submit('TURN_LEFT')
    rejected = bridge.rejected_execution(reg,NS(decision_seq=5,request_id='rejected',
        behavior='ADVANCE',intensity=.23),0,r.core)
    assert rejected.mc_request == dict(skill_profile_id=PROFILE_ID,command_seconds=None,intensity=.23)
    assert rejected.raw_feedback['summary_scope'] == 'executor_context_not_rejected_execution'
    assert rejected.raw_feedback['summary']['command_seconds'] == 2.
    monkeypatch.setenv('ROS_DOMAIN_ID','47')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY','1')
    # Invalid profile must fail before importing ROS, constructing a Node, or publishing.
    import builtins
    original_import = builtins.__import__
    def guarded_import(name,*args,**kwargs):
        assert not name.startswith('rclpy'), 'profile checked after ROS import'
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    cases = []
    for key,value in [('profile_id','unknown'),('skill_pack_id','other'),('wave_enabled',True),
                      ('command_seconds_by_behavior',dict(ADVANCE=1.,TURN_LEFT=1.,TURN_RIGHT=2.))]:
        changed = deepcopy(reg)
        changed['skill_profile'][key] = value
        cases.append(changed)
    missing = deepcopy(reg)
    del missing['skill_profile']
    cases.append(missing)
    for changed in cases:
        desktop.atomic(tmp_path/'bridge_registration.json',changed)
        with pytest.raises(ValueError,match='skill profile'):
            bridge.execute(tmp_path)
    desktop.atomic(tmp_path/'bridge_registration.json',reg)
    host['skill_profile']['command_seconds_by_behavior']['TURN_RIGHT'] = 1.
    desktop.atomic(tmp_path/'behavior_registration.json',host)
    with pytest.raises(ValueError,match='skill profile'):
        bridge.execute(tmp_path)
