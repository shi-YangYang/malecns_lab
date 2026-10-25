"""Fixed simulation wave integration; fake feedback/IPC only, never ROS or a VM."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import DecisionGate, ExecutionStatus as S
from malecns_lab.control.x2_skill_profile import candidate_skill_profile, SIM_WAVE_PROFILE_ID, sim_wave_contract, validate_candidate_skill_profile
from malecns_lab.control.x2_interaction import interaction_profile, validate_interaction_profile
from malecns_lab.experiments.x2_behavior_bridge import WaveSession, candidate_executor, RemoteExchange
from malecns_lab.learning.x2_behavior_session import ClockExchange
from .test_x2_skills import Rig
from .test_x2_behavior import observation, decision, event


def qualified_reg():
    return dict(run_id='run',session_id='session',simulation_only=True,qualification_only=True,wave_enabled=True,
        skill_profile=candidate_skill_profile(wave_enabled=True),interaction_profile=interaction_profile(wave_enabled=True))


def start(r, seq=0):
    r.feed()
    assert r.core.preset_query(0,0,1,received=r.t)
    assert r.submit('WAVE_RIGHT',seq=seq)
    r.core.preset_dispatch()
    assert r.core.preset_sent(r.t)
    assert r.core.preset_ack(0,0,400)
    r.player,r.joint=2,.3
    r.feed()
    r.core.preset_query(0,0,400,received=r.t)
    r.core.tick()
    assert r.core.status=='STARTED'


def recover(r, *, cancel=False):
    if cancel:
        assert r.core.preset_cancel_sent(r.t)
        assert r.core.preset_cancel_ack(0,1,400)
        r.player=3
        r.feed()
        r.core.preset_query(0,0,400,received=r.t)
        r.core.tick()
    r.player,r.joint=0,0.
    for _ in range(125):
        r.feed()
        r.core.preset_query(0,0,1,received=r.t)
        r.core.tick()


def test_new_profile_is_explicit_and_does_not_upgrade_default_capabilities():
    old=interaction_profile()
    assert not old['wave_enabled'] and 'WAVE_RIGHT' not in old['capabilities']
    new=interaction_profile(wave_enabled=True)
    assert new['profile_id']!=old['profile_id'] and validate_interaction_profile(new)==new
    assert validate_candidate_skill_profile(candidate_skill_profile(wave_enabled=True))['wave_contract']==sim_wave_contract()
    changed=deepcopy(old);changed['wave_enabled']=True
    with pytest.raises(ValueError,match='profile differs'):
        validate_interaction_profile(changed)
    with pytest.raises(ValueError,match='transport required'):
        candidate_executor(qualified_reg(),lambda _:None)
    assert 'WAVE_RIGHT' not in ClockExchange('run','session').capabilities
    assert ClockExchange('run','session',profile=new).capabilities==RemoteExchange('run','session',profile=new).capabilities


def test_multiple_natural_waves_reset_action_evidence_and_return_terminal_pose():
    r=Rig(qualified=True)
    for seq in (0,1):
        start(r,seq)
        assert r.core.probe_cancel_at is None and not r.core.saw_cancel_interrupting
        recover(r)
        assert r.core.status=='COMPLETED' and r.core.wave_return_confirmed
        assert r.core.events[-1]['stop_confirmed']
        assert r.core.wave_return_reference=='official_animation_terminal_frame'
    assert len(r.presets)==2


def test_brain_finish_cancels_once_and_requires_cancel_feedback():
    r=Rig(qualified=True);start(r)
    assert r.submit('HOLD',seq=1,phase='FINISH')
    for _ in range(5):
        r.feed();r.core.tick()
    assert [p['motion'] for p in r.presets]==[1002,0]
    assert r.core.status=='STARTED' and not r.core.terminal
    recover(r,cancel=True)
    assert r.core.status=='CANCELLED' and r.core.events[-1]['cancellation_confirmed']
    assert r.core.wave_return_reference=='official_animation_first_frame'
    assert r.core.events[-1]['finish_decision_seq']==1
    assert all(e['fact'] not in ('probe_cancel_requested','probe_cancel_ack') for e in r.core.events)
    start(r,2)
    assert r.core.probe_cancel_ack is None and r.core.probe_cancel_at is None


def test_late_finish_reports_natural_completion_without_zero():
    r=Rig(qualified=True);start(r)
    r.player,r.joint=0,0.;r.feed();r.core.preset_query(0,0,1,received=r.t)
    assert r.submit('HOLD',seq=1,phase='FINISH')
    recover(r)
    assert r.core.status=='COMPLETED' and [p['motion'] for p in r.presets]==[1002]
    assert not r.core.events[-1]['cancellation_confirmed']


def test_emergency_return_never_clears_protection():
    r=Rig(qualified=True);start(r)
    r.core.cancel('user_emergency_stop')
    assert r.core.status=='UNKNOWN' and [p['motion'] for p in r.presets]==[1002,0]
    recover(r,cancel=True)
    assert r.core.status=='UNKNOWN' and r.core.wave_return_confirmed and r.core.terminal
    assert not r.submit('HOLD',seq=1,phase='FINISH')


def test_cancel_ack_and_interrupting_are_both_required():
    r=Rig(qualified=True);start(r)
    assert r.submit('HOLD',seq=1,phase='FINISH')
    recover(r)  # Idle/pose alone do not prove the requested cancellation.
    assert not r.core.terminal
    for _ in range(650):
        r.feed();r.core.preset_query(0,0,1,received=r.t);r.core.tick()
    assert r.core.status=='UNKNOWN' and not r.core.wave_return_confirmed


def test_zero_ack_requires_explicit_sim_gate_and_local_execution_owner():
    owner=dict(run_id='run',session_id='session',decision_seq=0,request_id='request',execution_id='exec',helper_nonce='a'*32)
    ack=event(S.ACCEPTED,1,backend='sim',fact='preset_ack',
        mc_request=dict(skill_profile_id=SIM_WAVE_PROFILE_ID,wave_owner=owner),
        raw_feedback=dict(task_id=0,query_state=400,preset_send_confirmed=True))
    for enabled in (False,True):
        g=DecisionGate('run','session','policy','checkpoint','input',sim_wave_profile_id=SIM_WAVE_PROFILE_ID if enabled else None)
        g.observe(observation());assert g.admit(decision(),11.).accepted
        g.execution_event(event(S.ACCEPTED,fact='skill_admitted'))
        if enabled:
            wrong=replace(ack,mc_request={**ack.mc_request,'wave_owner':{**owner,'execution_id':'other'}})
            with pytest.raises(ValueError,match='exclusive request'):
                g.execution_event(wrong)
            g.execution_event(ack)
        else:
            with pytest.raises(ValueError,match='task attribution'):
                g.execution_event(ack)


class Boundary:
    def __init__(self, output, record, **kwargs):
        self.nonce=output.parent.name.rsplit('-',1)[-1]
        self.ready=True;self.sent=False;self.closed=False;self.requests=[];self.events=[]
    def start(self):pass
    def request(self,command):
        self.sent=True;self.requests.append(command)
    def poll(self):
        result,self.events=self.events,[]
        return result
    def close(self):self.closed=True


def transport(tmp_path,r):
    return WaveSession(tmp_path,qualified_reg(),r.core,{},lambda *a,**k:None,
        SimpleNamespace(service_is_ready=lambda:False),lambda _:None,lambda x:x,
        clock=lambda:r.t,boundary_factory=Boundary)


def test_preparation_withdraws_without_sending_and_reaps_child(tmp_path):
    r=Rig(qualified=True);session=transport(tmp_path,r);r.core.preset=session.preset
    r.core.preset_query(0,0,1,received=r.t);assert r.submit('WAVE_RIGHT')
    child=session.boundary
    assert r.submit('HOLD',seq=1,phase='FINISH')
    session.poll()
    assert child.closed and not child.sent and session.boundary is None
    assert r.core.status=='CANCELLED' and r.core.events[-1]['preset_never_sent']
    assert not r.core.events[-1]['cancellation_confirmed']


def test_per_action_helper_owner_is_new_and_protocol_failure_never_retries(tmp_path):
    r=Rig(qualified=True);session=transport(tmp_path,r);r.core.preset=session.preset
    r.core.preset_query(0,0,1,received=r.t);assert r.submit('WAVE_RIGHT')
    first=session.boundary;owner=deepcopy(session.owner)
    session.poll();assert first.sent and r.core.wave_dispatch_at is not None
    assert len(first.requests)==1
    session.poll();assert len(first.requests)==1
    first.events=[dict(kind='error',payload={'error':'ack_timeout'})]
    session.poll()
    assert session.failed and r.core.status=='UNKNOWN'
    assert len(first.requests)==1
    session.close_action();assert first.closed
    with pytest.raises(ValueError,match='latched closed'):
        session.preset(dict(motion=1002,area=2,interrupt=False))
    assert owner['helper_nonce'] and owner['decision_seq']==0


def test_new_helper_per_completed_wave_and_no_owner_reuse(tmp_path):
    r=Rig(qualified=True);session=transport(tmp_path,r);r.core.preset=session.preset
    owners=[];children=[]
    for seq in (0,1):
        r.feed();r.core.preset_query(0,0,1,received=r.t)
        assert r.submit('WAVE_RIGHT',seq=seq)
        owners.append(deepcopy(session.owner));child=session.boundary;children.append(child)
        session.poll()
        assert child.sent
        child.events=[dict(kind='sent',monotonic=r.t,payload={}),
            dict(kind='response',payload=dict(message=dict(response=dict(header=dict(code=0),task_id=0,state=dict(value=400)))))]
        session.poll()
        r.player,r.joint=2,.3;r.feed();r.core.preset_query(0,0,400,received=r.t);r.core.tick()
        recover(r);session.reap_terminal()
        assert child.closed and session.boundary is None
    assert owners[0]['helper_nonce']!=owners[1]['helper_nonce']
    assert owners[0]['execution_id']!=owners[1]['execution_id']
    assert all(len(child.requests)==1 for child in children)


def test_ambiguous_send_never_claims_unsent_or_cancels_without_ack(tmp_path):
    r=Rig(qualified=True);session=transport(tmp_path,r);r.core.preset=session.preset
    r.core.preset_query(0,0,1,received=r.t);assert r.submit('WAVE_RIGHT')
    child=session.boundary
    session.poll()
    assert child.sent
    r.core.cancel('user_emergency_stop')
    session.poll()
    assert r.core.status=='UNKNOWN' and len(child.requests)==1
    with pytest.raises(RuntimeError,match='possibly sent'):
        r.core.withdraw_unsent_wave()
    assert not r.core.terminal and not r.core.wave_return_confirmed


def test_late_ack_during_emergency_only_enables_constrained_single_stop():
    r=Rig(qualified=True)
    r.core.preset_query(0,0,1,received=r.t);assert r.submit('WAVE_RIGHT')
    r.core.preset_dispatch();r.core.preset_sent(r.t)
    r.core.cancel('user_emergency_stop')
    assert [p['motion'] for p in r.presets]==[1002]
    assert r.core.preset_ack(0,0,400)
    r.player,r.joint=2,.3;r.feed();r.core.preset_query(0,0,400,received=r.t);r.core.tick()
    assert [p['motion'] for p in r.presets]==[1002,0]
    assert r.core.status=='UNKNOWN'
    recover(r,cancel=True)
    assert r.core.status=='UNKNOWN' and r.core.wave_return_confirmed
