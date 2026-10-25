"""Full brain-output execution adapter with fake sensors/SDK only; no brain/VM."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2_behavior import DecisionGate
from malecns_lab.control.x2_interaction import (
    frozen_full_interaction_profile, frozen_history_interaction_profile,
    validate_interaction_profile,
)
from malecns_lab.control.x2_skill_profile import (
    SIM_FROZEN_FULL_PROFILE_ID, frozen_full_skill_profile,
    fixed_advance_skill_profile, candidate_skill_profile, validate_candidate_skill_profile,
)
from malecns_lab.control.x2_skills import SkillCommand, RIGHT_ARM, VelocityCore
from malecns_lab.experiments.x2_behavior_bridge import (
    candidate_executor, WaveSession, WaveStandPreparation, runtime_context_snapshot, observed_base_stop_basis,
)
from malecns_lab.learning.x2_frozen_history_policy import CANDIDATE_RELATIVE, inspect_frozen_history_candidate
from malecns_lab.runtime.x2.behavior import MAX_DECISIONS
from .test_x2_skills import Rig


def registration():
    return dict(run_id='run', session_id='session', simulation_only=True,
        qualification_only=True, wave_enabled=True, skill_profile=frozen_full_skill_profile(),
        interaction_profile=frozen_full_interaction_profile())


def rig():
    r = Rig(qualified=True)
    r.core = candidate_executor(registration(), r.commands.append, clock=lambda:r.t,
        preset=r.presets.append, wave_poses=dict(first=dict.fromkeys(RIGHT_ARM, 0.),
                                               terminal=dict.fromkeys(RIGHT_ARM, 0.)))
    r.core.motion.arm(); r.core.motion.confirm_source_registration()
    for _ in range(65):
        r.feed(); r.core.standing_ready(); r.core.tick()
    return r


def test_full_profile_preserves_brain_contract_and_binds_fixed_motion_without_task_map():
    p = frozen_full_skill_profile(); interaction = frozen_full_interaction_profile()
    assert validate_candidate_skill_profile(p) == p
    assert validate_interaction_profile(interaction) == interaction
    assert interaction['request_limits'] == dict(max_motion_starts=4, max_decisions=32)
    assert MAX_DECISIONS == 160
    assert not frozen_history_interaction_profile()['wave_enabled']
    assert 'request_limits' not in frozen_history_interaction_profile()
    assert not fixed_advance_skill_profile()['wave_enabled']
    assert candidate_skill_profile(wave_enabled=True)['wave_contract'] == p['wave_contract']
    metadata = inspect_frozen_history_candidate(Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[2]/CANDIDATE_RELATIVE)
    assert metadata['policy']['interaction_profile'] == interaction
    assert metadata['policy']['execution_profile'] == p and not metadata['policy']['can_deploy']
    DecisionGate('run','session','policy','checkpoint','input',sim_wave_profile_id=SIM_FROZEN_FULL_PROFILE_ID)
    with pytest.raises(ValueError):
        DecisionGate('run','session','policy','checkpoint','input',sim_wave_profile_id='invented')
    forged = deepcopy(p); forged['turn_execution']['left_rad_s'] = .5
    with pytest.raises(ValueError):
        validate_candidate_skill_profile(forged)
    assert VelocityCore(lambda _:None).resolve_velocity('TURN_LEFT', .5) == (0.,0.,.1)
    for behavior, velocity in [('ADVANCE',(.2,0.,0.)), ('TURN_LEFT',(0.,0.,.4)), ('TURN_RIGHT',(0.,0.,-.4))]:
        r = rig()
        assert r.core.submit(SkillCommand(0,'brain-request',behavior,'START',.5,r.t))
        assert r.core.resolved_velocity == velocity
        assert r.core.events[0]['model_intensity'] == .5
        deadline = r.core.deadline
        assert r.core.submit(SkillCommand(1,'brain-request',behavior,'CONTINUE',.5,r.t))
        assert r.core.deadline == deadline
        assert r.core.active_command_seconds == (1. if behavior == 'ADVANCE' else 2.)
        assert runtime_context_snapshot(r.core, SimpleNamespace(valid=True), -1)['protection_state'] == 'READY'
        if behavior != 'ADVANCE':
            r.a += .36 if behavior == 'TURN_LEFT' else -.36
            r.feed(wz=velocity[2]); r.core.tick()
            assert r.core.stop_at is not None and r.commands[-1] == (0.,0.,0.)
            assert not r.core.terminal  # A stop request is not measured completion.


def test_brain_selected_wave_waits_for_single_sd_and_end_withdraws_unsent_wave(tmp_path):
    for cancel in (False, True):
        r = rig(); calls = []; helpers = []; records = []
        class Future:
            def done(self): return True
            def result(self):
                return SimpleNamespace(response=SimpleNamespace(header=SimpleNamespace(code=0), status=SimpleNamespace(value=1)))
            def cancel(self): calls.append('cancel_wait')
        class Client:
            def service_is_ready(self): return True
            def call_async(self, message): calls.append(message); return Future()
        class Boundary:
            def __init__(self, *args, **kwargs):
                self.nonce='nonce'; self.ready=True; self.sent=False; self.requests=[]; helpers.append(self)
            def start(self): pass
            def poll(self): return []
            def close(self): pass
            def request(self, message): self.requests.append(message); self.sent=True
        wave = WaveSession(tmp_path/str(cancel), registration(), r.core, {},
            lambda kind,**fields:records.append((kind,fields)),
            SimpleNamespace(service_is_ready=lambda:False), lambda _:None, lambda _: {},
            clock=lambda:r.t, boundary_factory=Boundary)
        wave.stand_preparation = WaveStandPreparation(Client(), lambda:'actual_SD_request', lambda _: {},
            lambda kind,**fields:records.append((kind,fields)), frozen_full_skill_profile()['wave_preparation'], clock=lambda:r.t)
        r.core.preset = wave.preset
        r.core.preset_query(0,0,1,received=r.t)
        assert r.core.submit(SkillCommand(0,'brain-wave','WAVE_RIGHT','START',.5,r.t))
        wave.poll()
        assert calls == ['actual_SD_request'] and helpers == []
        if cancel:
            assert r.core.submit(SkillCommand(1,'end-request','HOLD','FINISH',.5,r.t))
            wave.poll()
            assert helpers == [] and r.core.terminal and r.core.status == 'CANCELLED'
            assert r.core.events[-1]['preset_never_sent']
            r.source = ''
            r.core.motion.request_stop('test_exit', failure=False)
            r.feed(); r.core.tick()
            assert observed_base_stop_basis(r.core) is None
            for _ in range(55):
                r.feed(); r.core.tick()
            assert observed_base_stop_basis(r.core) == 'official_unclaimed_stationary_standing_after_stop'
            assert not r.core.wave_return_confirmed
        else:
            for _ in range(55):
                r.feed();r.core.preset_query(0,0,1,received=r.t);wave.poll();r.core.tick()
            assert calls == ['actual_SD_request'] and len(helpers) == 1
            assert helpers[0].requests == [dict(motion=1002, area=2, interrupt=False)]
            assert any(kind=='wave_sd_standing' for kind,_ in records)
            assert r.core.command.intensity == .5 and r.core.command.behavior == 'WAVE_RIGHT'
