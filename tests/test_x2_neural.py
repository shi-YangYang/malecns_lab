"""Protocol/scheduler checks with fake clock and feedback, never a brain or ROS."""
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from malecns_lab.control.x2 import SOURCE, X2MotionAdapter
from malecns_lab.control.x2_protocol import (
    LoopController, StandObserver, atomic, read, seal, verify, intent, angle,
    MAX_ROUNDS, BRAIN_TIMEOUT, OBSERVATION_TIMEOUT, POSE_TIMEOUT,
)


class Clock:
    now = 1.
    def __call__(self):
        return self.now


class Rig:
    def __init__(self, directory, source=""):
        self.clock = Clock()
        self.published = []
        self.core = X2MotionAdapter(self.published.append, clock=self.clock, allow_empty_source_claim=True)
        self.core.confirm_source_registration()
        self.core.arm()
        self.loop = LoopController(self.core, directory, "run", execution_id="execution", clock=self.clock)
        self.source = source
        self.position = [0., 0., .65]

    def sample(self, *, dt=.1, speed=0., source=None, mode="STAND_DEFAULT", action=100, schedule=True,
               angular=(0.,0.,0.), quaternion=(0.,0.,0.,1.)):
        self.clock.now += dt
        self.core.update_odom(self.position, quaternion, (speed,0.,0.), angular)
        self.loop.feedback()
        self.core.update_mc(self.source if source is None else source, mode, 1, action)
        self.loop.feedback()
        if schedule:
            self.loop.update()
        self.core.tick()

    def ready(self):
        for _ in range(30):
            self.sample()
        assert self.loop.phase == "thinking"

    def decision(self, vx=.1, **changes):
        value = {"kind": "decision", "run_id": "run", "sequence": self.loop.sequence,
                 "observation_hash": self.loop.observation["message_hash"], "raw_action": [vx,.2,-.3],
                 "intent": "FORWARD" if vx > .03 else "STOP", "frame_index": self.loop.sequence,
                 "input_sha256": str(self.loop.sequence)*64, "execution_id": "execution"}
        value.update(changes)
        return seal(value)

    def post(self, response=None):
        atomic(self.loop.directory / f"decision-{self.loop.sequence:03d}.json", response or self.decision())
        self.sample()

    def finish_pulse(self, *, angular=(0.,0.,0.)):
        self.source = SOURCE
        for _ in range(11):
            self.position[0] += .04
            self.sample(speed=.4, mode="LOCOMOTION_DEFAULT")
        for _ in range(30):
            self.sample(angular=angular)


def test_atomic_refuses_replacement_and_seals(tmp_path):
    value = seal({"kind":"decision", "run_id":"r", "sequence":1})
    atomic(tmp_path/"one.json", value)
    assert verify(read(tmp_path/"one.json"), "decision", "r", 1) == value
    with pytest.raises(ValueError):
        atomic(tmp_path/"one.json", value)
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("change", [{"run_id":"old"}, {"sequence":2}, {"sequence":True}, {"kind":"observation"}, {"message_hash":"bad"}])
def test_message_identity_and_order_rejected(change):
    value = seal({"kind":"decision", "run_id":"r", "sequence":1})
    value.update(change)
    with pytest.raises(ValueError):
        verify(value, "decision", "r", 1)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '[]'])
def test_invalid_json_rejected(tmp_path, text):
    (tmp_path/"bad").write_text(text)
    with pytest.raises(ValueError):
        read(tmp_path/"bad")


@pytest.mark.parametrize("vx,expected", [(-1.,"STOP"), (.03,"STOP"), (.030001,"FORWARD")])
def test_frozen_gate(vx, expected):
    assert intent([vx,100.,-10.]) == expected


@pytest.mark.parametrize("action", [[math.nan,0,0], [math.inf,0,0], [True,0,0], [0,0], "abc"])
def test_nonfinite_or_non_axis_action_rejected(action):
    with pytest.raises(ValueError):
        intent(action)


def test_initial_empty_source_never_claims_before_neural_forward(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    assert rig.core.state == "STOPPING"
    assert all(v == (0.,0.,0.) for v in rig.published)
    observation = rig.loop.observation
    assert observation["stationary_evidence"] == "initial_stand_observer"
    assert observation["target_world_m"] == [2.,0.,.65]
    rig.post()
    assert rig.core.state == "CLAIMING"
    assert rig.published[-1] == (.2,0.,0.)


def test_duplicate_ticks_do_not_create_initial_stand(tmp_path):
    rig = Rig(tmp_path)
    rig.sample()
    for _ in range(20):
        rig.clock.now += .01
        rig.loop.update()
    assert rig.loop.sequence == 0


def test_transient_speed_resets_both_stream_barrier(tmp_path):
    rig = Rig(tmp_path)
    for _ in range(8):
        rig.sample()
    rig.clock.now += .01
    rig.core.update_odom(rig.position, (0.,0.,0.,1.), (.03,0.,0.), (0.,0.,0.))
    rig.loop.feedback()
    rig.clock.now += .01
    rig.core.update_odom(rig.position, (0.,0.,0.,1.), (0.,0.,0.), (0.,0.,0.))
    rig.loop.feedback()
    for _ in range(8):
        rig.sample()
    assert rig.loop.sequence == 0
    for _ in range(5):
        rig.sample()
    assert rig.loop.phase == "pose_settling"
    for _ in range(15):
        rig.sample()
    assert rig.loop.sequence == 1


def test_initial_pose_wait_requires_small_full_angular_norm_and_records_evidence(tmp_path):
    rig = Rig(tmp_path)
    for _ in range(30):
        rig.sample(angular=(.003,.004,.001))
    assert rig.loop.phase == "pose_settling" and rig.loop.sequence == 0
    assert all(v == (0.,0.,0.) for v in rig.published)
    for _ in range(15):
        rig.sample(angular=(.001,.001,.001))
    evidence = rig.loop.observation["pose_stability"]
    assert evidence["duration_seconds"] >= 1.
    assert evidence["max_angular_speed_rad_s"] == pytest.approx(math.sqrt(3)*.001)
    assert evidence["max_orientation_drift_rad"] == 0.
    assert all(n > o for n, o in zip(evidence["end_sequences"], evidence["start_sequences"]))
    assert evidence["timeout_seconds"] == 40.


@pytest.mark.parametrize("bad_angular,bad_quaternion,reason", [
    ((.006,0.,0.), (0.,0.,0.,1.), "angular_speed"),
    ((0.,0.,0.), (math.sin(.003),0.,0.,math.cos(.003)), "pose_drift"),
])
def test_pose_transient_resets_before_slower_mc_then_requires_two_new_streams(
        tmp_path, bad_angular, bad_quaternion, reason):
    rig = Rig(tmp_path)
    while rig.loop.phase == "initial_stand":
        rig.sample()
    for _ in range(8):
        rig.sample()
    rig.clock.now += .01
    rig.core.update_odom(rig.position, bad_quaternion, (0.,0.,0.), bad_angular)
    rig.loop.feedback()
    pose = rig.loop.pose_observer
    assert pose.start is None and pose.reset_reason == reason
    barrier_mc = rig.core.mc.sequence
    rig.clock.now += .01
    rig.core.update_odom(rig.position, (0.,0.,0.,1.), (0.,0.,0.), (0.,0.,0.))
    rig.loop.feedback()
    assert rig.core.mc.sequence == barrier_mc and pose.start is None
    for _ in range(8):
        rig.sample()
    assert rig.loop.sequence == 0
    for _ in range(5):
        rig.sample()
    assert rig.loop.sequence == 1


def test_pose_duplicate_ticks_and_missing_stream_cannot_advance_window(tmp_path):
    rig = Rig(tmp_path)
    while rig.loop.phase == "initial_stand":
        rig.sample()
    rig.sample()
    pose = rig.loop.pose_observer
    first_end = pose.last_time
    for _ in range(15):
        rig.clock.now += .01
        rig.loop.update()
    assert pose.last_time == first_end and not pose.qualified
    for _ in range(20):
        rig.clock.now += .05
        rig.core.update_odom(rig.position, (0.,0.,0.,1.), (0.,0.,0.), (0.,0.,0.))
        rig.loop.feedback()
        rig.loop.update()
        rig.core.tick()
    assert rig.loop.sequence == 0 and pose.start is None
    for _ in range(15):
        rig.sample()
    assert rig.loop.sequence == 1


def test_pose_wait_timeout_fails_at_frozen_budget_and_stays_zero(tmp_path):
    rig = Rig(tmp_path)
    while rig.loop.phase == "initial_stand":
        rig.sample(angular=(0.,0.,.015))
    began = rig.loop.pose_wait_started
    while rig.loop.failure is None:
        rig.sample(angular=(0.,0.,.015))
    assert rig.clock()-began == pytest.approx(40., abs=.11)
    assert "pose_settling_timeout" in rig.loop.failure
    assert rig.loop.sequence == 0 and all(v == (0.,0.,0.) for v in rig.published)


@pytest.mark.parametrize("failure", ["host", "total", "mc"])
def test_stopped_ack_precedes_pose_wait_and_failures_still_abort(tmp_path, failure):
    rig = Rig(tmp_path)
    rig.ready()
    rig.post()
    rig.finish_pulse(angular=(0.,0.,.01475075))
    assert rig.loop.phase == "pose_settling" and rig.loop.sequence == 1
    ack = read(tmp_path/"ack-001.json")
    assert ack["stopped"] and not ack["final"]
    assert not (tmp_path/"observation-002.json").exists()
    assert rig.loop.pose_wait_started == ack["round"]["stop_confirmed_monotonic"]
    assert rig.loop.pose_observer.reset_pair[0] >= ack["round"]["final_odom"]["sequence"]
    if failure == "host":
        atomic(tmp_path/"host_abort.json", {"reason":"rss_limit"})
        rig.sample()
        assert rig.loop.failure == "host_failure:rss_limit"
    elif failure == "total":
        rig.loop.deadline = rig.clock()+.05
        rig.sample()
        assert rig.loop.failure == "ros_total_timeout" and rig.loop.done
    else:
        rig.sample(source="competing_source")
        assert rig.loop.failure == "other_input_source"
    assert rig.published[-1] == (0.,0.,0.) and not (tmp_path/"observation-002.json").exists()


def test_pose_window_does_not_relax_original_response_drift_bound(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    atomic(tmp_path/"decision-001.json", rig.decision())
    rig.sample(quaternion=(0.,0.,math.sin(.016),math.cos(.016)))
    assert "Observation drift exceeds bound" in rig.loop.failure
    assert all(v == (0.,0.,0.) for v in rig.published)


def test_foreign_source_fails_without_claim(tmp_path):
    rig = Rig(tmp_path)
    rig.sample(source="someone_else")
    assert rig.loop.failure == "other_input_source"
    assert all(v == (0.,0.,0.) for v in rig.published)


@pytest.mark.parametrize("changes", [{"run_id":"other"}, {"sequence":2}, {"observation_hash":"old"},
    {"execution_id":"different"}, {"intent":"STOP"}, {"frame_index":0}])
def test_decision_binding_rejection_stays_zero(tmp_path, changes):
    rig = Rig(tmp_path)
    rig.ready()
    rig.post(rig.decision(**changes))
    assert rig.loop.failure
    assert all(v == (0.,0.,0.) for v in rig.published)


def test_drift_after_brain_thought_rejects(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    rig.position[0] = .021
    rig.post()
    assert rig.loop.failure
    assert rig.published[-1] == (0.,0.,0.)


def test_angle_is_full_orientation_and_quaternion_sign_invariant():
    assert angle([0,0,0,1], [0,0,0,-1]) == 0
    assert angle([0,0,0,1], [math.sin(.02),0,0,math.cos(.02)]) == pytest.approx(.04)


def test_brain_timeout_clears_intent(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    for _ in range(310):
        rig.sample()
    assert "thinking_timeout" in rig.loop.failure
    assert rig.published[-1] == (0.,0.,0.)


def test_future_decision_is_rejected(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    atomic(tmp_path/"decision-002.json", rig.decision(sequence=2))
    rig.sample()
    assert "out-of-order" in rig.loop.failure


def test_sticky_core_failure_prevents_new_round(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    rig.core.failures.append("prior_transient")
    rig.sample()
    assert rig.loop.failure == "motion_core:prior_transient"


def test_host_failure_is_checked_during_forward(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    rig.post()
    atomic(tmp_path/"result.json", {"status":"failed", "reason":"rss_limit"})
    rig.sample(source=SOURCE)
    assert rig.loop.failure == "host_failure:rss_limit"
    assert rig.published[-1] == (0.,0.,0.)


def test_all_stop_never_claims_and_cannot_pass_engineering_acceptance(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    for _ in range(2):
        rig.post(rig.decision(vx=0.))
        for _ in range(15):
            rig.sample()
    assert rig.loop.phase == "brain_seal"
    assert all(v == (0.,0.,0.) for v in rig.published)
    assert rig.loop.rounds[-1]["stop_kind"] == "initial_empty_source_stand"
    atomic(tmp_path/"host_ack.json", seal({"kind":"host_ack", "run_id":"run", "sequence":2,
          "status":"completed", "supervisor_status":"completed", "parameters_unchanged":True}))
    rig.sample()
    assert rig.loop.done
    assert rig.loop.summary()["success"] is False


def test_two_actual_pulses_wait_for_host_supervisor_ack(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    for index in range(2):
        rig.post()
        rig.finish_pulse()
        assert len(rig.loop.rounds) == index+1
    assert rig.loop.phase == "brain_seal" and not rig.loop.done
    atomic(tmp_path/"worker_result.json", {"status":"completed", "run_id":"run", "completed_frames":2, "parameters_unchanged":True})
    rig.sample()
    assert not rig.loop.done
    atomic(tmp_path/"host_ack.json", seal({"kind":"host_ack", "run_id":"run", "sequence":2,
          "status":"completed", "supervisor_status":"completed", "parameters_unchanged":True}))
    rig.sample()
    result = rig.loop.summary()
    assert result["success"] and result["new_observation_after_motion"] and result["final_stopped"]
    assert all(r["stop_kind"] == "owned_core_STOPPED" for r in result["rounds"])
    assert max(r["stop_confirmed_monotonic"]-r["pulse_started_monotonic"] for r in result["rounds"]) > 1.
    ack = read(tmp_path/"ack-002.json")
    assert ack["stopped"] is True and ack["final"] is True
    assert not (tmp_path/"observation-003.json").exists()


def test_consumed_decision_tampering_is_rejected(tmp_path):
    rig = Rig(tmp_path)
    rig.ready()
    rig.post(rig.decision(vx=0.))
    rig.sample()
    path = tmp_path/"decision-001.json"
    value = read(path)
    value["input_sha256"] = "f"*64
    value.pop("message_hash")
    path.write_text(json.dumps(seal(value)))
    rig.sample()
    assert "Consumed decision changed" in rig.loop.failure


@pytest.mark.parametrize("module", ["malecns_lab.experiments.x2_bridge", "malecns_lab.experiments.x2_neural"])
def test_default_preview_imports_no_ros_torch_or_numpy(module):
    script = f'''import sys
import {module} as module
assert module.main([]) == 0
assert not any(n in sys.modules for n in ("torch", "rclpy", "numpy"))
'''
    result = subprocess.run([sys.executable,"-c",script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["rounds"] == MAX_ROUNDS == 2
    assert preview["brain_timeout_seconds"] == BRAIN_TIMEOUT == 30.
    assert preview["observation_timeout_seconds"] == OBSERVATION_TIMEOUT == 45.
    assert preview["pose_timeout_seconds"] == POSE_TIMEOUT == 40.


@pytest.mark.parametrize("field,value", [
    ("rounds", 3), ("brain_timeout_seconds", 45.),
    ("observation_timeout_seconds", 30.), ("pose_timeout_seconds", 25.), ("ready_rounds", 3),
])
def test_bridge_rejects_different_registered_budgets_before_ros(tmp_path, field, value):
    from malecns_lab.experiments.x2_bridge import execute
    registration = {"run_id": "run", "registration_id": "registration", "rounds": 2,
                    "brain_timeout_seconds": 30., "observation_timeout_seconds": 45., "pose_timeout_seconds": 40.}
    ready = {"kind": "ready", "run_id": "run", "sequence": 0,
             "registration_id": "registration", "rounds": 2}
    if field == "ready_rounds":
        ready["rounds"] = value
    else:
        registration[field] = value
    atomic(tmp_path/"registration.json", registration)
    atomic(tmp_path/"ready.json", seal(ready))
    with pytest.raises(ValueError, match="Registered loop budgets differ"):
        execute(tmp_path)
    assert not (tmp_path/"bridge.claim").exists()


def test_transport_sources_parse_as_python310():
    import ast
    root = Path(__file__).resolve().parents[1]/"src/malecns_lab"
    for name in ("control/x2_protocol.py", "experiments/x2_bridge.py"):
        ast.parse((root/name).read_text(), feature_version=(3,10))


def test_failed_supervisor_publishes_abort_without_host_ack(tmp_path, monkeypatch):
    import signal
    from malecns_lab.learning import x2_session as session
    from malecns_lab.learning import resource_guard
    from contextlib import nullcontext
    root = tmp_path/"project"
    (root/"data").mkdir(parents=True)
    for name in session.WRAPPERS:
        path = root/"src/malecns_lab"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture only\n")
    monkeypatch.setattr(session.base, "numerical_lock", lambda root: nullcontext())
    monkeypatch.setattr(session.base, "simulator_preflight", lambda root: {"gpu_stopped": True})
    monkeypatch.setattr(session.base, "read_anchors", lambda root: {"run": {"implementation_sha256": {}}})
    monkeypatch.setattr(session.base, "export_sources", lambda *args: {})
    monkeypatch.setattr(session.base, "runtime", lambda *args: {})
    monkeypatch.setattr(resource_guard, "supervise", lambda *a, **k: {"status":"failed", "returncode":-15, "reason":"rss_limit"})
    previous = signal.getsignal(signal.SIGTERM)
    output = root/"data/test-run"
    result = session.execute(output, root=root)
    assert result["status"] == "failed" and "rss_limit" in result["reason"]
    registration = read(output/"registration.json")
    assert registration["rounds"] == 2
    assert registration["brain_timeout_seconds"] == 30.
    assert registration["observation_timeout_seconds"] == 45.
    assert registration["pose_timeout_seconds"] == 40.
    assert (output/"host_abort.json").exists() and not (output/"host_ack.json").exists()
    assert signal.getsignal(signal.SIGTERM) == previous
