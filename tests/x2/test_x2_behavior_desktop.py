"""Protocol/IPC-boundary fixtures only; never construct a brain or start ROS/VM."""
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import (Behavior, Decision, DecisionGate, ExecutionEvent,
    ExecutionStatus, Feedback, Observation, Phase)
from malecns_lab.experiments.x2_behavior_bridge import RemoteExchange, wrap_execution, rejected_execution
from malecns_lab.runtime.x2.behavior import Commands, BehaviorRuntime, candidate
from malecns_lab.learning.x2_behavior_session import ClockExchange, NeuralSession, SCHEMA, host_event, load_candidate_policy


class Clock:
    def __init__(self,value): self.value=value
    def __call__(self): return self.value


def setup_exchange():
    hostclock,remoteclock=Clock(98765.),Clock(12.)
    host,remote=ClockExchange('run','session',hostclock),RemoteExchange('run','session',remoteclock)
    query=host.request('GREET','request')
    feedback=Feedback(valid=True,age_s=.03,source='official_sim_odom_and_mc',standing=True,mc_mode='STAND_DEFAULT')
    packet=remote.observe(query,feedback,[])
    hostclock.value+=.1
    observation,evidence=host.accept(packet)
    return host,remote,hostclock,remoteclock,packet,observation,evidence


def decision(observation,sequence=1,behavior='HOLD',phase='FINISH'):
    behaviors=('HOLD','ADVANCE','TURN_LEFT','TURN_RIGHT','WAVE_RIGHT')
    phases=('START','CONTINUE','FINISH')
    return Decision(observation.run_id,observation.session_id,sequence,observation.request_id,
        observation.observation_seq,'state-before','state-after',tuple(2. if b==behavior else 0. for b in behaviors),
        tuple(2. if p==phase else 0. for p in phases),(0.,),behavior,phase,.5,
        'policy','checkpoint','input',600,1100)


def test_double_clock_nonce_round_trip_and_age_are_not_cross_subtracted():
    host,remote,hclock,rclock,packet,obs,clocks=setup_exchange()
    assert obs.received_monotonic == hclock.value
    assert obs.feedback.age_s == pytest.approx(.13)
    assert clocks['remote_issued_monotonic'] == 12.
    d=decision(obs)
    rclock.value+=2.
    got,remote_obs,age=remote.consume(dict(schema=SCHEMA,kind='decision',run_id='run',session_id='session',
        nonce=packet['nonce'],decision=d.to_dict()))
    assert got==d and remote_obs.received_monotonic==12. and age==2.
    with pytest.raises(ValueError,match='replay|Replay'):
        host.accept(packet)
    with pytest.raises(ValueError,match='reused'):
        remote.consume(dict(schema=SCHEMA,kind='decision',run_id='run',session_id='session',nonce=packet['nonce'],decision=d.to_dict()))


def test_re_reading_old_remote_file_cannot_renew_decision_or_observation():
    host,remote,hclock,rclock,packet,obs,_=setup_exchange()
    rclock.value=43.
    with pytest.raises(ValueError,match='expired'):
        remote.consume(dict(schema=SCHEMA,kind='decision',run_id='run',session_id='session',nonce=packet['nonce'],decision=decision(obs).to_dict()))
    host.request('GREET','request')
    with pytest.raises(ValueError,match='nonce'):
        host.accept(packet)
    fresh=remote.observe(host.pending,Feedback(valid=True,source='official',age_s=.01),[])
    hclock.value+=2.01
    with pytest.raises(ValueError,match='round trip'):
        host.accept(fresh)


def test_remote_original_age_and_unqualified_wave_are_rejected():
    host,remote,hclock,rclock,packet,obs,_=setup_exchange()
    query=host.request('GREET','request')
    fresh=remote.observe(query,Feedback(valid=True,source='official',age_s=.74),[])
    hclock.value+=.1
    with pytest.raises(ValueError,match='age plus transit'):
        host.accept(fresh)
    fresh['source_age_s']=fresh['feedback']['age_s']=.01
    fresh['capabilities'].append('WAVE_RIGHT')
    with pytest.raises(ValueError,match='capability'):
        host.accept(fresh)


def test_execution_clock_conversion_keeps_remote_facts_and_gate_ownership():
    _,_,hclock,_,_,obs,_=setup_exchange()
    gate=DecisionGate('run','session','policy','checkpoint','input')
    gate.observe(obs)
    assert gate.admit(decision(obs),hclock()).accepted
    for i,status in enumerate(('ACCEPTED','STARTED','COMPLETED')):
        event=ExecutionEvent('run','session',i,1,'skill-1','request','sim','skills',status,'HOLD',12.+i,
            raw_feedback={'source_age_s':.01},stop_evidence={'stop_confirmed':status=='COMPLETED'},
            fact='skill_admitted' if i==0 else 'status')
        local=host_event(event.to_dict(),'run','session',hclock()+.1*i)
        assert local.raw_feedback['remote_received_monotonic']==12.+i
        assert local.received_monotonic==hclock()+.1*i
        gate.execution_event(local)
    assert gate.active is None
    with pytest.raises(ValueError,match='Foreign'):
        host_event(event.to_dict(),'other','session',hclock())


def test_old_task_raw_output_is_archived_and_state_never_rolled_back():
    import torch
    class State:
        def __init__(self,number): self.number=number
        def detach(self): return self
    class FixturePolicy:
        policy_id='policy'
        encoder=SimpleNamespace(input_map_id='input')
        initializations=0
        def initial_state(self,*args,**kwargs): self.initializations+=1; return State(0)
        def state_identity(self,state): return {'state_id':'state-'+str(state.number)}
        def __call__(self,obs,state,**kwargs):
            return SimpleNamespace(state=State(state.number+1),features_mv=torch.zeros(2628))
        def decision(self,obs,output,**kwargs):
            return replace(decision(obs,kwargs['decision_seq']),state_before_id=kwargs['state_before_id'],state_after_id='state-'+str(output.state.number))
    observed=[]
    telemetry=SimpleNamespace(begin_decision=lambda *args:None)
    policy=FixturePolicy()
    session=NeuralSession(policy,'checkpoint','run','session',telemetry,lambda kind,**fields:observed.append((kind,fields)))
    import time
    first=Observation('run','session',1,'old','GREET',time.monotonic())
    newer=Observation('run','session',2,'new','END_INTERACTION',time.monotonic())
    d,admission=session.infer(first,lambda:newer)
    assert not admission.accepted and admission.reason=='superseded_observation'
    assert session.state.number==1 and policy.initializations==1
    assert [kind for kind,_ in observed[-3:]]==['observation','gate_refresh','admission']
    assert any(kind=='raw_decision' and row['decision']['request_id']=='old' for kind,row in observed)
    third=replace(newer,observation_seq=3,received_monotonic=time.monotonic())
    session.infer(third)
    assert session.state.number==2 and policy.initializations==1


def test_desktop_commands_accept_only_task_enum_not_actions_or_paths():
    c=Commands(io.StringIO('{"command":"task","task":"END_INTERACTION"}\n{"command":"continue"}\n'))
    c.read()
    assert c.tasks.get_nowait()[0]=='END_INTERACTION' and c.consume_continue()
    for value in ({'command':'task','task':'WAVE_RIGHT'},{'command':'task','task':'GREET','checkpoint':'../bad'},
                  {'command':'mc','velocity':1}):
        invalid=Commands(io.StringIO(json.dumps(value)+'\n'))
        invalid.read()
        assert invalid.stop.is_set() and invalid.tasks.empty()


def test_preview_has_no_heavy_imports_or_startup():
    for module in ('malecns_lab.runtime.x2.behavior','malecns_lab.experiments.x2_behavior_bridge'):
        code=(f'import sys; from {module} import main; main([]); '
              'assert not any(x in sys.modules for x in ("torch","numpy","rclpy"))')
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        assert json.loads(result.stdout)['mode']=='preview'


def test_missing_or_external_candidate_fails_before_any_runtime(tmp_path):
    (tmp_path/'data').mkdir()
    with pytest.raises(ValueError,match='Missing'):
        candidate(tmp_path)
    outside=tmp_path/'outside'; outside.mkdir()
    with pytest.raises(ValueError,match='inside'):
        candidate(tmp_path,outside)


def test_emergency_write_failure_does_not_skip_owned_resource_cleanup(tmp_path,monkeypatch):
    from malecns_lab.runtime.x2 import behavior as module
    runtime=object.__new__(BehaviorRuntime)
    runtime.output=tmp_path; runtime.errors=[]; runtime.run_id='run'; runtime.session_id='session'; runtime.bridge=None
    cleaned=[]
    runtime.cleanup=lambda:cleaned.append(True)
    monkeypatch.setattr(module,'atomic',lambda *args:(_ for _ in ()).throw(OSError('disk full')))
    runtime.shutdown_behavior('emergency')
    assert cleaned==[True] and runtime.stop_confirmed is None and runtime.errors

def test_executor_fact_envelope_keeps_original_and_finish_decisions_separate():
    raw=dict(decision_seq=4,execution_id='skill-4',request_id='first',behavior='TURN_LEFT',
        received_monotonic=42.,status='STARTED',reason='brain_finish',fact='finish_requested',
        finish_decision_seq=5,finish_request_id='end',angle_rad=.1,base_stop_confirmed=False)
    event=wrap_execution({'run_id':'run','session_id':'session'},raw,7,[{'actual_velocity':[0,0,.1]}],{'odom':{'x':1}})
    assert event.decision_seq==4 and event.request_id=='first'
    assert event.finish_decision_seq==5 and event.finish_request_id=='end'
    assert event.fact=='finish_requested' and event.status==ExecutionStatus.STARTED
    assert event.raw_feedback['sensors']['odom']=={'x':1}
    assert event.mc_request['recent_actual_requests']==[{'actual_velocity':[0,0,.1]}]
    assert event.stop_evidence['base_stop_confirmed'] is False


def test_worker_phase_fields_do_not_collide_with_coordinator_status_fields():
    runtime=object.__new__(BehaviorRuntime)
    rows=[]
    runtime.status=lambda state,message,**fields:rows.append((state,message,fields))
    runtime.phase_status({'phase':'waiting_task','qualification_only':True})
    runtime.phase_status({'phase':'executing','behavior_phase':'START','task':'FACE_LEFT'})
    assert rows[0][2]=={'phase':'waiting_task','qualification_only':True}
    assert rows[1][2]['behavior_phase']=='START'


class FixtureState:
    def __init__(self,number): self.number=number
    def detach(self): return self


class FixturePolicy:
    """Explicit CPU protocol fixture: no graph, learned behavior or hardware."""
    policy_id='policy'
    encoder=SimpleNamespace(input_map_id='input')
    def __init__(self):
        self.inputs=[]
        self.behavior,self.phase='HOLD','FINISH'
    def initial_state(self,*args,**kwargs): return FixtureState(0)
    def state_identity(self,state): return {'state_id':'state-'+str(state.number)}
    def __call__(self,observation,state,**kwargs):
        import torch
        self.inputs.append(observation)
        return SimpleNamespace(state=FixtureState(state.number+1),features_mv=torch.zeros(2628))
    def decision(self,observation,output,**kwargs):
        return replace(decision(observation,kwargs['decision_seq'],self.behavior,self.phase),
            state_before_id=kwargs['state_before_id'],state_after_id='state-'+str(output.state.number))


def fixture_session():
    records=[]
    policy=FixturePolicy()
    session=NeuralSession(policy,'checkpoint','run','session',SimpleNamespace(begin_decision=lambda *args:None),
        lambda kind,**fields:records.append((kind,fields)))
    return session,policy,records


def test_gate_refresh_does_not_consume_repeated_task_request_edge():
    import time
    host=ClockExchange('run','session')
    remote=RemoteExchange('run','session',Clock(12.))
    session,policy,_=fixture_session()
    feedback=Feedback(valid=True,source='official',standing=True)
    def snapshot(request_id):
        query=host.request('GREET',request_id,last_consumed_request_id=session.last_consumed_request_id)
        return host.accept(remote.observe(query,feedback,[]))[0]
    old=snapshot('old')
    _,admission=session.infer(old,lambda:snapshot('new'))
    assert not admission.accepted and session.last_consumed_request_id=='old'
    # Arbitrarily many gate-only refreshes cannot consume the same-task new edge.
    assert snapshot('new').request_changed
    fresh=snapshot('new')
    session.infer(fresh)
    assert [o.request_changed for o in policy.inputs]==[True,True]
    assert session.last_consumed_request_id=='new'
    again=snapshot('new')
    assert not again.request_changed
    session.infer(again)
    assert policy.inputs[-1].request_changed is False
    assert policy.inputs[-1].received_monotonic<=time.monotonic()


def test_actual_executor_refusal_reaches_brain_once_with_its_own_attribution():
    import time
    from malecns_lab.control.x2_skills import SkillExecutor
    session,policy,records=fixture_session()
    policy.behavior,policy.phase='ADVANCE','START'
    feedback=Feedback(valid=True,source='official',standing=True,yaw_rad=.4,vx_m_s=.01,
        wz_rad_s=.02,mc_mode='STAND_DEFAULT')
    obs=Observation('run','session',1,'old','GREET',time.monotonic(),feedback=feedback,
        capabilities=('HOLD','ADVANCE'))
    chosen,admission=session.infer(obs)
    assert admission.accepted
    core=SkillExecutor(lambda *_:None,lambda *_:None)
    core.standing_ready=lambda:False
    assert not core.submit_decision(chosen,obs.received_monotonic)
    assert core.status=='IDLE' and core.command is None
    event=rejected_execution({'run_id':'run','session_id':'session'},chosen,0,core)
    session.execution_event(event)
    assert not session.confirmed_finish(chosen)
    assert session.gate.active is None and len(session.pending_rejections)==1
    refreshed=replace(obs,observation_seq=2,request_id='new',received_monotonic=time.monotonic())
    session.observe(refreshed)
    assert len(session.pending_rejections)==1  # Polling does not consume.
    policy.behavior,policy.phase='HOLD','FINISH'
    session.infer(replace(refreshed,observation_seq=3,received_monotonic=time.monotonic()))
    actual=policy.inputs[-1]
    assert actual.request_id=='new' and actual.request_changed
    assert actual.feedback.status==ExecutionStatus.REJECTED and actual.feedback.behavior==Behavior.ADVANCE
    assert actual.feedback.request_id=='old' and actual.feedback.execution_id==event.execution_id
    assert (actual.feedback.yaw_rad,actual.feedback.vx_m_s,actual.feedback.wz_rad_s)==(.4,.01,.02)
    projection=next(row for kind,row in records if kind=='rejection_neural_input')
    assert projection['rejected']['decision_seq']==chosen.decision_seq
    assert projection['current_feedback']['status']==ExecutionStatus.IDLE
    assert len(session.pending_rejections)==0
    session.infer(replace(refreshed,observation_seq=4,received_monotonic=time.monotonic()))
    assert policy.inputs[-1].feedback.status==ExecutionStatus.IDLE


def test_host_rejection_preserves_actual_active_skill_and_body_evidence():
    import time
    session,policy,records=fixture_session()
    policy.behavior,policy.phase='TURN_LEFT','START'
    obs=Observation('run','session',1,'old','FACE_LEFT',time.monotonic(),
        feedback=Feedback(valid=True,source='official',standing=True),
        capabilities=('HOLD','TURN_LEFT','TURN_RIGHT'))
    initial,admission=session.infer(obs)
    assert admission.accepted
    for sequence,status in enumerate(('ACCEPTED','STARTED')):
        session.execution_event(ExecutionEvent('run','session',sequence,initial.decision_seq,
            'active-left','old','sim','skills',status,'TURN_LEFT',time.monotonic()))
    active=Feedback(valid=True,source='official',status='RUNNING',behavior='TURN_LEFT',
        execution_id='active-left',request_id='old',yaw_rad=.3,wz_rad_s=.1,mc_mode='WALK')
    newer=replace(obs,observation_seq=2,request_id='new',task='FACE_RIGHT',
        received_monotonic=time.monotonic(),feedback=active)
    policy.behavior='TURN_RIGHT'
    refused,admission=session.infer(newer)
    assert not admission.accepted and admission.reason=='skill_already_active'
    active_before=dict(session.gate.active)
    policy.behavior,policy.phase='TURN_LEFT','CONTINUE'
    session.infer(replace(newer,observation_seq=3,received_monotonic=time.monotonic()))
    seen=policy.inputs[-1].feedback
    assert (seen.status,seen.behavior,seen.request_id)==(ExecutionStatus.REJECTED,Behavior.TURN_RIGHT,'new')
    assert seen.execution_id=='host-rejected-'+str(refused.decision_seq)
    assert (seen.yaw_rad,seen.wz_rad_s,seen.mc_mode)==(.3,.1,'WALK')
    assert session.gate.active==active_before
    projection=next(row for kind,row in records if kind=='rejection_neural_input')
    assert projection['current_feedback']==active.__dict__
    assert projection['active_execution_id']=='active-left'
    assert projection['rejected']['decision_seq']==refused.decision_seq


@pytest.mark.parametrize('gain',[None,1.0,0.1])
def test_candidate_uses_real_loader_and_registered_backward_without_full_brain(tmp_path,monkeypatch,gain):
    from malecns_lab.learning import x2_inference as loader, x2_behavior_training as training
    from malecns_lab.learning.x2_behavior_policy import X2BehaviorPolicy,backward_identity
    calls=[]
    source={'policy_id':'historical'}
    theta=object()
    fixture=object()
    identity={'initialization':{'theta':'migrated','head_seed':2401}}
    if gain is not None: identity['backward']=backward_identity(gain)
    meta={'checkpoint_id':'selected','policy':identity}
    monkeypatch.setattr(loader,'read_anchors',lambda root:(calls.append(('anchors',root)) or
        {'run':{'policy':source},'checkpoint':{'checkpoint_id':'historical-checkpoint'}}))
    def parameters(path,checkpoint,policy):
        calls.append(('parameters',path,checkpoint,policy))
        return {'theta':theta}
    def construct(path,policy,**kwargs):
        calls.append(('construct',path,policy,kwargs))
        return fixture
    def restore(path,policy):
        calls.append(('strict_restore',path,policy))
        return meta
    monkeypatch.setattr(loader,'load_parameters',parameters)
    monkeypatch.setattr(X2BehaviorPolicy,'from_reference',construct)
    monkeypatch.setattr(training,'restore_checkpoint',restore)
    assert load_candidate_policy(tmp_path,meta,tmp_path/'candidate') is fixture
    assert calls[1][1]==tmp_path/loader.CHECKPOINT/'arrays.npz'
    assert calls[2][1:]==(tmp_path/loader.REFERENCE,source,
        {'theta':theta,'initialization':'migrated','seed':2401,'spike_backward_gain':1.0 if gain is None else gain})
    assert calls[-1]==('strict_restore',tmp_path/'candidate',fixture)


def test_candidate_backward_tampering_fails_before_graph_and_restore_stays_strict(tmp_path,monkeypatch):
    from malecns_lab.learning import x2_inference as loader, x2_behavior_training as training
    from malecns_lab.learning.x2_behavior_policy import backward_identity
    identity={'initialization':{'theta':'migrated','head_seed':2401},'backward':backward_identity(.1)}
    identity['backward']['algorithm']='unregistered'
    monkeypatch.setattr(loader,'read_anchors',lambda *_:pytest.fail('Must reject before model assets'))
    with pytest.raises(ValueError,match='backward identity'):
        load_candidate_policy(tmp_path,{'policy':identity},tmp_path/'candidate')
    with pytest.raises(ValueError,match='backward identity'):
        load_candidate_policy(tmp_path,{'policy':{'backward':None}},tmp_path/'candidate')
    monkeypatch.setattr(training,'inspect_checkpoint',lambda *_:{'policy':{'backward':backward_identity(.1)}})
    with pytest.raises(ValueError,match='identity differs'):
        training.restore_checkpoint(tmp_path,SimpleNamespace(identity={'backward':backward_identity(1.)}))
    sources=training.source_inventory(Path(__file__).resolve().parents[2])
    assert sources['learning/x2_inference.py']==loader.sha256(Path(loader.__file__))


def heading_snapshot(host, remote, task, request_id, yaw, *, status='IDLE', consumed=None):
    """Explicit CPU odometry fixture, not simulator evidence."""
    query=host.request(task,request_id,last_consumed_request_id=consumed)
    raw=dict(sequence=query['sequence'],remote_received_monotonic=remote.clock()-.01,
        message={'pose':{'pose':{'orientation':dict(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))}}})
    feedback=Feedback(valid=True,age_s=.01,source='official_sim_odom_and_mc',standing=True,
        yaw_rad=yaw,status=status,behavior='TURN_LEFT' if status!='IDLE' else 'HOLD',
        request_id=request_id if status!='IDLE' else '',execution_id='actual-turn' if status!='IDLE' else '')
    packet=remote.observe(query,feedback,[],odom=raw,sampled_remote_monotonic=remote.clock())
    observation,evidence=host.accept(packet)
    return observation,evidence,packet


@pytest.mark.parametrize('task,origin,moved,offset',[
    ('FACE_LEFT',3.,-3.08,.45),('FACE_RIGHT',-3.,3.08,-.45)])
def test_heading_goal_is_fixed_wrapped_and_completion_does_not_zero_error(task,origin,moved,offset):
    host=ClockExchange('run','session',Clock(9000.))
    remote=RemoteExchange('run','session',Clock(12.))
    first,clocks,packet=heading_snapshot(host,remote,task,'request',origin)
    original=clocks['heading_target']
    assert first.feedback.heading_error_rad==pytest.approx(offset)
    assert original['established_remote_sequence']==1 and original['established_remote_monotonic']==12.
    assert original['origin_yaw_rad']==pytest.approx(origin)
    assert original['target_yaw_rad']==pytest.approx(math.atan2(math.sin(origin+offset),math.cos(origin+offset)))
    assert clocks['host_received_monotonic']==9000.  # Never subtract remote and host clocks.
    for status in ('IDLE','RUNNING','COMPLETED'):
        remote.clock.value+=.1
        current,clocks,packet=heading_snapshot(host,remote,task,'request',moved,status=status)
        target=clocks['heading_target']
        assert target['target_yaw_rad']==original['target_yaw_rad']
        assert target['origin_odom']==original['origin_odom']
        assert target['established_remote_sequence']==1 and current.request_changed
        expected=math.atan2(math.sin(original['target_yaw_rad']-moved),math.cos(original['target_yaw_rad']-moved))
        assert current.feedback.heading_error_rad==pytest.approx(expected)
        assert abs(current.feedback.heading_error_rad)>.2
    remote.clock.value+=.1
    again,clocks,_=heading_snapshot(host,remote,task,'new-request',moved,consumed='request')
    assert again.feedback.heading_error_rad==pytest.approx(offset)
    assert clocks['heading_target']['origin_yaw_rad']==pytest.approx(moved)
    assert clocks['heading_target']['established_remote_sequence']==again.observation_seq
    remote.clock.value+=.1
    end,clocks,_=heading_snapshot(host,remote,'END_INTERACTION','end',moved)
    assert clocks['heading_target'] is None and end.feedback.heading_error_rad==0.


@pytest.mark.parametrize('invalid', ['invalid','expired','missing_odom'])
def test_heading_target_is_not_established_from_unavailable_odometry(invalid):
    host=ClockExchange('run','session',Clock(9000.))
    remote=RemoteExchange('run','session',Clock(12.))
    query=host.request('FACE_LEFT','request')
    feedback=Feedback(valid=invalid!='invalid',age_s=.76 if invalid=='expired' else .01,
        source='official_sim_odom_and_mc',yaw_rad=.2)
    with pytest.raises(ValueError,match='unavailable|odometry'):
        remote.observe(query,feedback,[])
    assert not remote.heading_targets and not remote.issued and remote.last_query==0


def test_host_rejects_self_consistent_target_that_moves_within_request():
    from copy import deepcopy
    host=ClockExchange('run','session',Clock(9000.))
    remote=RemoteExchange('run','session',Clock(12.))
    heading_snapshot(host,remote,'FACE_LEFT','request',0.)
    query=host.request('FACE_LEFT','request')
    raw=dict(sequence=2,remote_received_monotonic=11.99,
        message={'pose':{'pose':{'orientation':dict(x=0.,y=0.,z=math.sin(.1/2),w=math.cos(.1/2))}}})
    packet=deepcopy(remote.observe(query,Feedback(valid=True,age_s=.01,source='official',yaw_rad=.1),[],odom=raw))
    goal=packet['heading_target']
    goal.update(origin_yaw_rad=.1,target_yaw_rad=.55,heading_error_rad=.45,origin_odom=raw,
        established_remote_sequence=2)
    packet['feedback']['heading_error_rad']=.45
    with pytest.raises(ValueError,match='moved within request'):
        host.accept(packet)


def test_rejection_projection_keeps_new_request_goal_and_current_body_once():
    host=ClockExchange('run','session')
    remote=RemoteExchange('run','session',Clock(12.))
    session,policy,records=fixture_session()
    session.pending_rejections.append(dict(source='host_decision_gate_rejection',request_id='old-request',
        execution_id='host-rejected-1',decision_seq=1,behavior=Behavior.TURN_LEFT,reason='fixture-refusal'))
    for _ in range(2):
        obs,clocks,_=heading_snapshot(host,remote,'FACE_RIGHT','new-request',.3)
        session.heading_evidence(obs,clocks['heading_target'])
        session.observe(obs)
    assert len(session.pending_rejections)==1 and obs.request_changed
    obs,clocks,_=heading_snapshot(host,remote,'FACE_RIGHT','new-request',.3)
    session.heading_evidence(obs,clocks['heading_target'])
    session.infer(obs)
    actual=policy.inputs[-1]
    assert actual.request_id=='new-request' and actual.feedback.request_id=='old-request'
    assert actual.feedback.yaw_rad==.3 and actual.feedback.heading_error_rad==pytest.approx(-.45)
    projection=next(row for kind,row in records if kind=='rejection_neural_input')
    assert projection['current_heading_target']['request_id']=='new-request'
    assert projection['current_heading_target']==clocks['heading_target']
    assert projection['current_feedback']['heading_error_rad']==pytest.approx(-.45)
    assert len(session.pending_rejections)==0 and session.last_consumed_request_id=='new-request'


def provenance_fixture():
    """Real executor, bounded CPU clock/message fixtures; no simulator."""
    from .test_x2_skills import Rig
    from malecns_lab.experiments.x2_behavior_bridge import SensorEvidence
    rig=Rig()
    records,events=[],[]
    sensors=SensorEvidence(lambda kind,**fields:records.append((kind,fields)))
    rig.core.record=lambda raw:events.append(wrap_execution(
        {'run_id':'run','session_id':'session'},raw,len(events),[],sensors.latest,sensors.trigger))
    def feed(sensor, *, x=0., player=0, valid=True):
        rig.t+=.01
        if sensor=='odom':
            quaternion=(0.,0.,0.,1. if valid else 0.)
            message=dict(pose=dict(pose=dict(position=dict(x=x,y=0.,z=.65),
                orientation=dict(zip(('x','y','z','w'),quaternion)))),
                twist=dict(twist=dict(linear=dict(x=0.,y=0.,z=0.),angular=dict(x=0.,y=0.,z=0.))))
            update=lambda **stamp:rig.core.update_odom((x,0.,.65),quaternion,(0.,0.,0.),(0.,0.,0.),**stamp)
        else:
            status=100 if valid else 'invalid'
            message=dict(input_source=dict(name=rig.source),body_status=dict(value=1),
                action_info=dict(action_desc='STAND_DEFAULT',status=dict(value=status)),
                motion_status=dict(player_state=dict(value=player),motion='foreign' if player else ''))
            update=lambda **stamp:rig.core.update_mc(rig.source,'STAND_DEFAULT',1,status,
                player_state=player,motion='foreign' if player else '',**stamp)
        ok=sensors.update(sensor,message,update,lambda:getattr(rig.core.motion,sensor),received=rig.t)
        return ok,message
    feed('odom')
    feed('mc')
    return rig,sensors,records,events,feed


@pytest.mark.parametrize('sensor,status',[('odom','STARTED'),('mc','UNKNOWN')])
def test_provenance_synchronous_executor_event_keeps_trigger_before_acceptance(sensor,status):
    rig,sensors,records,events,feed=provenance_fixture()
    assert rig.submit('ADVANCE')
    events.clear()
    previous=sensors.latest[sensor]
    ok,message=feed(sensor,x=.03,player=2)
    assert ok and events[-1].status==status
    raw=events[-1].raw_feedback
    trigger=raw['triggering_snapshot']
    assert trigger['snapshot_id']!=previous['snapshot_id']
    assert trigger['message']==message and trigger['remote_received_monotonic']==rig.t
    assert trigger['accepted'] is None and trigger['acceptance']=='pending'
    assert raw['sensors'][sensor]==trigger
    other=raw['sensors']['mc' if sensor=='odom' else 'odom']
    assert other['remote_received_monotonic']<trigger['remote_received_monotonic']
    result=records[-1][1]
    assert result['snapshot_id']==trigger['snapshot_id'] and result['accepted'] is True
    assert result['sequence']==getattr(rig.core.motion,sensor).sequence
    message.clear()
    assert trigger['message'] and sensors.latest[sensor]['message']
    assert trigger['accepted'] is None  # No retroactive event mutation.


@pytest.mark.parametrize('sensor',['odom','mc'])
def test_provenance_feedback_requires_exact_accepted_core_sample(sensor):
    from malecns_lab.experiments.x2_behavior_bridge import feedback_snapshot
    rig,sensors,records,_,feed=provenance_fixture()
    body,originals,_=feedback_snapshot(rig.core,sensors,lambda:rig.t)
    assert body.valid and set(originals)=={'odom','mc'}
    current=sensors.latest[sensor]
    for mismatch in (dict(current,sequence=current['sequence']+1),
                     dict(current,remote_received_monotonic=current['remote_received_monotonic']-.001),
                     dict(current,accepted=False)):
        sensors.latest[sensor]=mismatch
        with pytest.raises(ValueError,match='source missing or differs'):
            feedback_snapshot(rig.core,sensors,lambda:rig.t)
    sensors.latest[sensor]=current
    ok,_=feed(sensor,valid=False)
    assert not ok and getattr(rig.core.motion,sensor) is None
    assert records[-1][1]['accepted'] is False and records[-1][1]['sequence'] is None
    with pytest.raises(ValueError,match='Feedback missing'):
        feedback_snapshot(rig.core,sensors,lambda:rig.t)
    assert originals[sensor]['accepted'] is True  # Archived prior sample stays unchanged.


def test_provenance_non_heading_response_is_archived_before_overwrite(tmp_path,monkeypatch):
    from malecns_lab.experiments import x2_behavior_bridge as bridge
    rig,sensors,_,_,_=provenance_fixture()
    body,originals,sampled=bridge.feedback_snapshot(rig.core,sensors,lambda:rig.t)
    host=ClockExchange('run','session',Clock(10000.))
    remote=RemoteExchange('run','session',Clock(rig.t))
    records=[]
    real_atomic=bridge.atomic
    def mailbox(path,packet):
        assert records[-1][0]=='observation_response'
        assert records[-1][1]['response']==packet
        real_atomic(path,packet)
    monkeypatch.setattr(bridge,'atomic',mailbox)
    for request in ('first','second'):
        query=host.request('GREET',request)
        packet=remote.observe(query,body,[],odom=originals['odom'],
            sampled_remote_monotonic=sampled,feedback_sources=originals)
        bridge.publish_observation(tmp_path,packet,lambda kind,**fields:records.append((kind,fields)))
        observation,clocks=host.accept(packet)
        assert packet['heading_target'] is None and packet['feedback']==observation.to_dict()['feedback']
        assert clocks['feedback_sources']==originals and clocks['sampled_remote_monotonic']==sampled
        assert clocks['response_record']['response_id']==query['nonce']
    archived=[row['response'] for kind,row in records if kind=='observation_response']
    assert [row['request_id'] for row in archived]==['first','second']
    assert json.loads((tmp_path/'observation_response.json').read_text())['nonce']==archived[1]['nonce']
    assert archived[0]['feedback_sources']==originals
    packet['feedback_sources']['odom']['message'].clear()
    assert archived[1]['feedback_sources']['odom']['message']


def test_provenance_neural_attempt_after_projection_and_gate_only_records(monkeypatch):
    import time
    session,policy,records=fixture_session()
    rejection=dict(source='host_decision_gate_rejection',request_id='refused',
        execution_id='host-rejected-1',decision_seq=1,behavior=Behavior.ADVANCE,reason='fixture')
    session.pending_rejections.append(rejection)
    first=Observation('run','session',1,'first','GREET',time.monotonic())
    newer=replace(first,observation_seq=2,request_id='newer',received_monotonic=time.monotonic())
    session.infer(first,lambda:newer)
    attempt=next(row for kind,row in records if kind=='neural_input')
    assert attempt['status']=='attempted' and attempt['decision_seq']==1
    assert attempt['observation_seq']==1 and attempt['request_id']=='first'
    assert attempt['observation']==policy.inputs[0].to_dict()
    assert attempt['observation']['feedback']['status']=='REJECTED'
    assert [kind for kind,_ in records].index('rejection_neural_input')<[kind for kind,_ in records].index('neural_input')
    assert any(kind=='raw_decision' and row['decision']['observation_seq']==1 for kind,row in records)
    gate=next(row for kind,row in records if kind=='gate_refresh')
    assert gate['observation_seq']==2 and gate['gate_observation_updated'] is True
    session.record_gate_refresh(newer,purpose='execution_event_poll')
    assert records[-1][1]['gate_observation_updated'] is False and len(policy.inputs)==1
    assert len([1 for kind,_ in records if kind=='neural_input'])==1
    assert len([1 for kind,_ in records if kind=='observation'])==2

    aborted,_,attempts=fixture_session()
    aborted.pending_rejections.append(rejection)
    def interrupt(*args,**kwargs):
        raise InterruptedError('CPU fixture interruption')
    monkeypatch.setattr(FixturePolicy,'__call__',interrupt)
    with pytest.raises(InterruptedError):
        aborted.infer(first)
    assert any(kind=='neural_input' and row['status']=='attempted' for kind,row in attempts)
    assert not any(kind=='raw_decision' for kind,_ in attempts)
    assert aborted.last_consumed_request_id is None and len(aborted.pending_rejections)==1
