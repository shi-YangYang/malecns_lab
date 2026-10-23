"""Control safety tests use only fake feedback/clock and no ROS/device IO."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from malecns_lab.control.x2 import SOURCE
from malecns_lab.control.x2_skills import SkillExecutor, SkillCommand, RIGHT_ARM


class Rig:
    def __init__(self, wave=False, source=SOURCE, initial_yaw=0.):
        self.t = 1.
        self.commands, self.presets = [], []
        self.core = SkillExecutor(self.commands.append, self.presets.append, clock=lambda:self.t,
                                  enabled=("HOLD","ADVANCE","TURN_LEFT","TURN_RIGHT","WAVE_RIGHT"), wave_probe=wave)
        self.core.motion.arm()
        self.core.motion.confirm_source_registration()
        self.x = self.y = 0.
        self.a = initial_yaw
        self.source = source
        self.player = 0
        self.joint = 0.
        for _ in range(65):
            self.feed()
            self.core.standing_ready()
            self.core.tick()

    def feed(self, speed=0., wz=0.):
        self.t += .02
        self.core.update_odom((self.x,self.y,.65), (0.,0.,math.sin(self.a/2),math.cos(self.a/2)), (speed,0.,0.), (0.,0.,wz), received=self.t)
        self.core.update_mc(self.source, "STAND_DEFAULT", 1, 100, player_state=self.player,
                            motion="SM_wave_right_hand" if self.player else "", area=2 if self.player else 0,
                            motion_type=1 if self.player else 0, received=self.t)
        self.core.update_joints({k:(self.joint,0.) for k in RIGHT_ARM}, received=self.t)

    def submit(self, behavior, seq=0, phase="START", observed=None):
        return self.core.submit(SkillCommand(seq,"request",behavior,phase,1.,self.t if observed is None else observed))


@pytest.mark.parametrize("behavior,sign", [("ADVANCE",0),("TURN_LEFT",1),("TURN_RIGHT",-1)])
def test_bounded_velocity_actual_motion_and_stop(behavior, sign):
    r=Rig()
    assert r.submit(behavior)
    assert not r.submit(behavior)  # replay
    for _ in range(52):
        if sign:
            r.a += sign*.004
        else:
            r.x += .004
        r.feed(.2 if not sign else 0, sign*.2)
        r.core.tick()
    assert r.commands[-1] == (0.,0.,0.)
    for _ in range(65):
        r.feed()
        r.core.tick()
    assert r.core.status == "COMPLETED"
    assert r.core.motion.state == "STOPPED"
    assert any(c[2]*sign > 0 if sign else c[0]>0 for c in r.commands)


def test_stale_command_and_feedback_never_move():
    r=Rig()
    assert not r.submit("ADVANCE",observed=r.t-31)
    assert r.submit("ADVANCE",seq=1)
    r.t += .3
    r.core.tick()
    assert r.commands[-1] == (0.,0.,0.)
    assert "odom_unavailable" in r.core.motion.failures


def test_empty_source_claim_is_bounded_and_not_repeated():
    r=Rig(source="")
    assert r.submit("TURN_LEFT")
    for _ in range(30):
        r.feed()
        r.core.tick()
    assert "claim_timeout" in r.core.motion.failures
    assert r.commands[-1] == (0.,0.,0.)
    assert not r.submit("TURN_LEFT",seq=1)


def test_angular_velocity_prevents_false_stop():
    r=Rig()
    assert r.submit("TURN_LEFT")
    for _ in range(60):
        r.a += .004
        r.feed(wz=.2)
        r.core.tick()
    for _ in range(65):
        r.feed(wz=.04)
        r.core.tick()
    assert r.core.status != "COMPLETED"
    assert r.core.motion.state != "STOPPED"


def test_wave_disabled_without_probe_and_ack_is_not_completion():
    r=Rig()
    r.core.preset_query(0,0,1,received=r.t)
    assert not r.submit("WAVE_RIGHT")
    r=Rig(wave=True)
    r.core.preset_query(0,0,1,received=r.t)
    assert r.submit("WAVE_RIGHT")
    assert r.presets == [dict(motion=1002,area=2,interrupt=False)]
    assert r.core.preset_ack(0,123,400)
    r.feed()
    r.core.preset_query(0,123,1,received=r.t)
    r.core.tick()
    assert r.core.status == "ACCEPTED"
    assert not r.core.terminal


def test_wave_needs_actual_play_and_return_and_correct_task():
    r=Rig(wave=True)
    r.core.preset_query(0,0,1,received=r.t)
    assert r.submit("WAVE_RIGHT")
    r.core.preset_ack(0,123,400)
    r.player=2
    r.joint=.3
    r.feed()
    r.core.preset_query(0,123,400,received=r.t)
    r.core.tick()
    assert r.core.status == "STARTED"
    r.player=0
    r.joint=0
    for _ in range(125):
        r.feed()
        r.core.preset_query(0,123,1,received=r.t)
        r.core.tick()
    assert r.core.status == "COMPLETED"
    assert r.core.terminal
    assert not r.core.summary()["wave_cancellation_verified"]


def test_wave_cancel_and_misattribution_are_unknown_not_success():
    r=Rig(wave=True)
    r.core.preset_query(0,0,1,received=r.t)
    assert r.submit("WAVE_RIGHT")
    r.core.preset_ack(0,123,400)
    r.feed()
    assert not r.core.preset_query(0,456,400,received=r.t)
    assert r.core.status == "UNKNOWN"
    r.core.cancel()
    assert r.core.status == "UNKNOWN"
    assert r.presets == [dict(motion=1002,area=2,interrupt=False)]
    assert not r.core.summary()["wave_cancellation_verified"]


def test_import_preview_does_not_load_device_or_brain(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root/"src"))
    code='import sys; from malecns_lab.experiments.x2_skills import main; main([]); assert not any(x in sys.modules for x in ("rclpy", "torch", "numpy", "malecns_lab.control.x2_skills"))'
    result=subprocess.run([sys.executable,"-c",code],env=env,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "preview"
    assert list(tmp_path.iterdir()) == []


def test_normal_wave_finish_waits_without_claiming_cancel():
    r=Rig(wave=True)
    r.core.preset_query(0,0,1,received=r.t)
    assert r.submit("WAVE_RIGHT")
    r.core.preset_ack(0,123,400)
    assert r.submit("HOLD",seq=1,phase="FINISH")
    assert r.core.status == "ACCEPTED"  # A FINISH request is not observed motion.
    assert r.core.reason == "finish_pending_natural_completion"
    assert not r.core.terminal
    assert len(r.presets) == 1


@pytest.mark.parametrize("heading,dx,dy,expected", [
    (0., -.004, 0., "FAILED"), (0., 0., .004, "FAILED"),
    (math.pi/2, 0., .004, "COMPLETED"),
])
def test_advance_uses_initial_body_forward_projection(heading, dx, dy, expected):
    r = Rig(initial_yaw=heading)
    assert r.submit("ADVANCE")
    for _ in range(52):
        r.x += dx
        r.y += dy
        r.feed(speed=.2)
        r.core.tick()
    for _ in range(65):
        r.feed()
        r.core.tick()
    assert r.core.status == expected
    evidence = r.core.events[-1]
    assert evidence["distance_m"] == pytest.approx(.208)
    assert evidence["start_yaw_rad"] == pytest.approx(heading)
    assert evidence["forward_distance_m"] == pytest.approx(52*(dx*math.cos(heading)+dy*math.sin(heading)))
    assert evidence["lateral_distance_m"] == pytest.approx(52*(-dx*math.sin(heading)+dy*math.cos(heading)))
    assert evidence["advance_direction_ok"] is (expected == "COMPLETED")


@pytest.mark.parametrize("player", [1, 2, 3, 4, 99])
def test_external_preset_during_velocity_zeroes_and_latches_unknown(player):
    r = Rig()
    assert r.submit("TURN_LEFT")
    r.core.tick()
    assert r.commands[-1][2] > 0
    r.player = player
    r.feed()
    r.core.tick()
    assert r.commands[-1] == (0., 0., 0.)
    assert r.core.status == "UNKNOWN"
    assert "external_preset_during_velocity" in r.core.motion.failures
    for _ in range(65):
        r.feed()
        r.core.tick()
    assert r.core.status == "UNKNOWN"
    assert not r.core.events[-1]["stop_confirmed"]
    assert not r.submit("TURN_RIGHT", seq=1)


def protocol_decision(behavior, phase="START", seq=0, request="request", observation_seq=0):
    from malecns_lab.control.x2_behavior import Behavior, Decision, Phase
    logits = tuple(float(b == behavior) for b in Behavior)
    phases = tuple(float(p == phase) for p in Phase)
    return Decision("run", "session", seq, request, observation_seq, "before", "after", logits,
                    phases, (0.,), behavior, phase, .5, "policy", "checkpoint", "input", 600, 1100)


def connect_gate(r):
    from malecns_lab.control.x2_behavior import Behavior, DecisionGate, ExecutionEvent, Observation, Task
    g = DecisionGate("run", "session", "policy", "checkpoint", "input")
    g.observe(Observation("run", "session", 0, "request", Task.GREET, r.t, capabilities=tuple(Behavior)))
    delivered = []
    def deliver(raw):
        event = ExecutionEvent("run", "session", raw["event_seq"], raw["decision_seq"], raw["execution_id"],
            raw["request_id"], "offline_fixture", "pack", raw["status"], raw["behavior"], raw["received_monotonic"],
            raw_feedback=raw, stop_evidence={k: raw[k] for k in ("stop_confirmed", "base_stop_confirmed") if k in raw},
            protection_reason=raw.get("protection_reason"), fact=raw["fact"],
            finish_decision_seq=raw["finish_decision_seq"], finish_request_id=raw["finish_request_id"])
        g.execution_event(event)
        delivered.append(event)
    r.core.record = deliver
    return g, delivered


def test_idle_hold_actual_executor_lifecycle_passes_gate():
    r = Rig()
    g, events = connect_gate(r)
    d = protocol_decision("HOLD", "FINISH")
    assert g.admit(d, r.t).accepted
    assert g.active is not None
    assert r.core.submit_decision(d, r.t)
    assert [e.status for e in events] == ["ACCEPTED", "STARTED", "COMPLETED"]
    assert events[-1].stop_evidence["stop_confirmed"]
    assert g.active is None


def test_wave_ack_and_finish_facts_keep_origin_and_do_not_invent_start():
    from dataclasses import replace
    from malecns_lab.control.x2_behavior import Task
    r = Rig(wave=True)
    g, events = connect_gate(r)
    r.core.preset_query(0, 0, 1, received=r.t)
    d = protocol_decision("WAVE_RIGHT")
    assert g.admit(d, r.t).accepted and r.core.submit_decision(d, r.t)
    assert r.core.preset_ack(0, 123, 400)
    assert [e.fact for e in events] == ["skill_admitted", "preset_ack"]
    assert all(e.status == "ACCEPTED" for e in events)
    with pytest.raises(ValueError, match="Duplicate or invalid official preset ACK"):
        g.execution_event(replace(events[-1], event_seq=2))
    assert g.last_event_seq == 1
    g.observe(replace(g.observation, observation_seq=1, request_id="end", task=Task.END_INTERACTION))
    finish = protocol_decision("HOLD", "FINISH", 1, "end", 1)
    assert g.admit(finish, r.t).accepted and r.core.submit_decision(finish, r.t)
    assert events[-1].status == "ACCEPTED" and events[-1].fact == "finish_requested"
    assert events[-1].decision_seq == 0 and events[-1].request_id == "request"
    assert events[-1].finish_decision_seq == 1 and events[-1].finish_request_id == "end"
    r.player, r.joint = 2, .3
    r.feed()
    r.core.preset_query(0, 123, 400, received=r.t)
    r.core.tick()
    assert events[-1].status == "STARTED"
    r.player, r.joint = 0, 0.
    for _ in range(125):
        r.feed()
        r.core.preset_query(0, 123, 1, received=r.t)
        r.core.tick()
    assert events[-1].status == "COMPLETED" and events[-1].finish_decision_seq == 1
    assert g.active is None


def test_unknown_is_not_an_idle_gate_or_an_executor_restart():
    r = Rig()
    g, events = connect_gate(r)
    d = protocol_decision("TURN_LEFT")
    assert g.admit(d, r.t).accepted and r.core.submit_decision(d, r.t)
    r.player = 2
    r.feed()
    for _ in range(65):
        r.feed()
        r.core.tick()
    assert events[-1].status == "UNKNOWN" and not events[-1].stop_evidence["stop_confirmed"]
    assert g.active is not None and g.blocked_reason
    assert g.admit(protocol_decision("TURN_RIGHT", seq=1), r.t).reason == "execution_protection_latched"
    assert not r.submit("TURN_RIGHT", seq=1)


def test_stop_timeout_without_other_guard_failure_still_latches_executor():
    r = Rig()
    assert r.submit("TURN_LEFT")
    for _ in range(675):
        r.a += .0008
        r.feed(wz=.04)
        r.core.tick()
    assert r.core.status == "UNKNOWN" and r.core.terminal
    assert r.core.reason == "stop_timeout" and not r.core.motion.failures
    for _ in range(65):
        r.feed()
        r.core.standing_ready()
        r.core.tick()
    assert not r.submit("TURN_RIGHT", seq=1)


def test_host_preview_inert_and_assets_match_local_paths(tmp_path):
    from malecns_lab.experiments.x2_skill_host import ASSETS, ROOT, output_directory
    for asset in ASSETS:
        assert (ROOT / "data/x2_sim/packages" / asset).is_file()
    with pytest.raises(ValueError):
        output_directory(tmp_path/"skills-20261023-120000-1234abcd")
    result=subprocess.run([sys.executable,"-m","malecns_lab.experiments.x2_skill_host"],
        env=dict(os.environ,PYTHONPATH=str(ROOT/"src")),cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)["simulator_started"] is False
    assert list(tmp_path.iterdir()) == []


def test_host_failure_still_cleans_owned_resources(tmp_path):
    from malecns_lab.experiments.x2_skill_host import SkillRuntime
    from malecns_lab.experiments.x2_desktop import Commands
    import io
    import subprocess
    class FakeRuntime(SkillRuntime):
        def preflight(self):
            self.default_context="desktop-linux"
        def prepare_transport(self):
            pass
        def start_simulator(self):
            self.owns_vm=True
            raise RuntimeError("deliberate_start_failure")
        def cleanup(self):
            self.owns_vm=False
            self.resources_stopped=True
        def command(self,*args,**kwargs):
            return subprocess.CompletedProcess([],0,"desktop-linux\n","")
    r=FakeRuntime(tmp_path,tmp_path,Commands(io.StringIO()),lambda event:None)
    result=r.execute_skills(["HOLD"],False,None)
    assert result["state"] == "failed"
    assert result["resources_stopped"]
    assert "deliberate_start_failure" in result["reason"]
    assert (tmp_path/"host_summary.json").is_file()


@pytest.mark.parametrize("failed_stage", [None, "container", "vm", "host", "probe"])
def test_host_cleanup_survives_logging_and_independent_cleanup_failures(tmp_path, failed_stage):
    import io
    from malecns_lab.experiments.x2_skill_host import SkillRuntime
    from malecns_lab.experiments.x2_desktop import Commands
    calls = []
    class Child:
        exited = False
        def poll(self):
            return 0 if self.exited else None
    class FakeRuntime(SkillRuntime):
        def command(self, args, **kwargs):
            calls.append(args)
            if "--host" in args:
                if "stop" in args and failed_stage == "container":
                    raise OSError("container stop failed")
                output = "running" if failed_stage == "container" else "exited"
            else:
                if args[-1] == "stop" and failed_stage == "vm":
                    raise OSError("vm stop failed")
                output = "running" if failed_stage == "vm" else "lighter is not running."
            return subprocess.CompletedProcess(args, 0, output, "")
        def stop_host(self):
            calls.append(["host_cleanup"])
            if failed_stage == "host":
                raise RuntimeError("host cleanup failed")
            self.host.exited = True
        def _kill_command(self, child):
            calls.append(["probe_cleanup"])
            if failed_stage == "probe":
                raise RuntimeError("probe cleanup failed")
            child.exited = True
    def broken_emit(value):
        raise OSError("lifecycle log unavailable")
    r = FakeRuntime(tmp_path, tmp_path, Commands(io.StringIO()), broken_emit)
    r.owns_vm = r.owns_container = True
    r.host, r.bridge = Child(), Child()
    r.cleanup()
    assert any("stop" in c and "--host" in c for c in calls)
    assert any(c[-1] == "stop" and "--host" not in c for c in calls)
    assert ["host_cleanup"] in calls and ["probe_cleanup"] in calls
    assert r.resources_stopped is (failed_stage is None)
    assert r.owns_container is (failed_stage == "container")
    assert r.owns_vm is (failed_stage == "vm")
    assert any("lifecycle log unavailable" in e for e in r.errors)


def test_host_output_failure_returns_failed_after_verified_cleanup(tmp_path, monkeypatch):
    import io
    from malecns_lab.experiments.x2_skill_host import SkillRuntime
    from malecns_lab.experiments.x2_desktop import Commands
    class FakeRuntime(SkillRuntime):
        def preflight(self):
            self.default_context = "desktop-linux"
        def prepare_transport(self):
            pass
        def start_simulator(self):
            self.owns_vm = self.owns_container = True
            raise RuntimeError("startup failed")
        def command(self, args, **kwargs):
            output = "desktop-linux" if args == ["docker", "context", "show"] else (
                "exited" if "--host" in args else "lighter is not running.")
            return subprocess.CompletedProcess(args, 0, output, "")
    def broken_output(*args):
        raise OSError("disk or pipe failed")
    r = FakeRuntime(tmp_path, tmp_path, Commands(io.StringIO()), broken_output)
    monkeypatch.setattr(r.protocol, "atomic", broken_output)
    result = r.execute_skills(["HOLD"], False, None)
    assert result["state"] == "failed" and result["resources_stopped"]
    assert any("status_output:" in e for e in result["errors"])
    assert any("summary_write:" in e for e in result["errors"])


def test_host_cleanup_does_not_stop_unowned_resources(tmp_path):
    import io
    from malecns_lab.experiments.x2_skill_host import SkillRuntime
    from malecns_lab.experiments.x2_desktop import Commands
    r = SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda event: None)
    r.foreign_resources = True
    r.command = lambda *args, **kwargs: pytest.fail("must not stop or inspect unowned resources")
    r.cleanup()
    assert not r.resources_stopped
