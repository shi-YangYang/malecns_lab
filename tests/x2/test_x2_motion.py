"""Offline fake-clock checks. Never load ROS or operate simulation/hardware."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from malecns_lab.control.x2 import FORWARD, SOURCE, ZERO, X2MotionAdapter


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds=0.1):
        self.now += seconds


def setup_adapter(**options):
    clock = Clock()
    sent = []
    core = X2MotionAdapter(sent.append, clock=clock, **options)
    return core, clock, sent


def feedback(core, *, source=SOURCE, mode="STAND_DEFAULT", speed=0.0, x=0.0,
             quaternion=(0, 0, 0, 1), body_status=1, action_status=100):
    assert core.update_odom((x, 0, .65), quaternion, (speed, 0, 0), (0, 0, 0))
    assert core.update_mc(source, mode, body_status, action_status)


def running():
    core, clock, sent = setup_adapter()
    core.arm()
    feedback(core)
    assert core.request_forward(1.0)
    core.tick()
    assert sent[-1] == FORWARD
    return core, clock, sent


def stable_for(core, clock, seconds=1.2, **kwargs):
    for _ in range(round(seconds/.1)):
        clock.advance()
        feedback(core, **kwargs)
        core.tick()


def test_default_disarmed_has_no_io_or_queued_intent():
    core, clock, sent = setup_adapter()
    feedback(core)
    assert not core.request_forward(1.0)
    core.tick()
    assert sent == []
    core.arm()
    clock.advance()
    feedback(core)
    core.tick()
    assert sent == [ZERO]
    assert core.state != "FORWARD"


@pytest.mark.parametrize("missing", ["odom", "mc", "source", "body", "tilt", "mode", "status"])
def test_forward_requires_all_gates(missing):
    core, clock, sent = setup_adapter()
    core.arm()
    feedback(core, source="another" if missing == "source" else SOURCE,
             body_status=2 if missing == "body" else 1,
             quaternion=(math.sin(.4), 0, 0, math.cos(.4)) if missing == "tilt" else (0,0,0,1),
             mode="PASSIVE_DEFAULT" if missing == "mode" else "STAND_DEFAULT")
    if missing == "odom":
        core.odom = None
    if missing == "mc":
        core.mc = None
    if missing == "status":
        clock.advance(.01)
        core.update_mc(SOURCE, "STAND_DEFAULT", 1, 200)
    assert not core.request_forward(1)
    core.tick()
    assert sent[-1] == ZERO


@pytest.mark.parametrize("fault", ["odom_age", "mc_age", "ownership", "upright", "mode", "ttl"])
def test_motion_faults_zero_and_do_not_resume(fault):
    core, clock, sent = running()
    if fault == "odom_age":
        clock.advance(.3)
        core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100)
    elif fault == "mc_age":
        clock.advance(.6)
        core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0))
    else:
        clock.advance(1.1 if fault == "ttl" else .1)
        feedback(core, source="other" if fault == "ownership" else SOURCE,
                 body_status=2 if fault == "upright" else 1,
                 mode="PASSIVE_DEFAULT" if fault == "mode" else "STAND_DEFAULT")
    core.tick()
    assert sent[-1] == ZERO
    assert core.failures
    clock.advance()
    feedback(core)
    core.tick()
    assert sent[-1] == ZERO
    assert core.state != "FORWARD"


@pytest.mark.parametrize("ttl", [0, -1, 5.01, math.nan, math.inf, "bad"])
def test_invalid_ttl_rejects_and_clears(ttl):
    core, clock, sent = running()
    assert not core.request_forward(ttl)
    core.tick()
    assert sent[-1] == ZERO


def test_five_second_limit_survives_renewals_and_restart_requires_stop():
    core, clock, sent = setup_adapter()
    core.arm()
    feedback(core)
    assert core.request_forward(5)
    for _ in range(49):
        clock.advance()
        feedback(core)
        assert core.request_forward(5)
        core.tick()
    clock.advance(.11)
    feedback(core)
    assert not core.request_forward(5)
    core.tick()
    assert sent[-1] == ZERO
    assert not core.request_forward(5)
    stable_for(core, clock)
    assert core.state == "STOPPED"
    assert core.request_forward(.2)


@pytest.mark.parametrize("axis,value", [("vx", -.2), ("vx", .3), ("vy", .1), ("vz", .1),
                                         ("wx", .1), ("wy", .1), ("wz", .1), ("vx", math.nan)])
def test_body_targets_do_not_silently_drop_axes(axis, value):
    core, clock, sent = running()
    target = dict.fromkeys(("vx", "vy", "vz", "wx", "wy", "wz"), 0.0)
    target[axis] = value
    assert not core.request_body(SimpleNamespace(**target))
    core.tick()
    assert sent[-1] == ZERO


@pytest.mark.parametrize("topic", ["odom", "mc"])
@pytest.mark.parametrize("stamp", ["future", "reverse", "same", "nan", "inf"])
def test_invalid_receipt_time_zeros(topic, stamp):
    core, clock, sent = running()
    now = clock.now
    clock.advance()
    received = {"future": clock.now+1, "reverse": now-.1, "same": now,
                "nan": math.nan, "inf": math.inf}[stamp]
    if topic == "odom":
        assert not core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0), received=received)
    else:
        assert not core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100, received=received)
    core.tick()
    assert sent[-1] == ZERO


@pytest.mark.parametrize("bad", ["position", "linear", "angular", "quaternion"])
def test_invalid_odom_values_zeros(bad):
    core, clock, sent = running()
    clock.advance()
    args = {"position": (0,0,.65), "quaternion": (0,0,0,1), "linear": (0,0,0), "angular": (0,0,0)}
    args[bad] = (0,0,0,0) if bad == "quaternion" else (math.nan,0,0)
    assert not core.update_odom(**args)
    core.tick()
    assert sent[-1] == ZERO


def test_clock_reversal_cancels_forward():
    core, clock, sent = running()
    clock.now -= 1
    core.tick()
    assert sent[-1] == ZERO
    assert "invalid_clock" in core.failures


def test_stop_needs_both_post_request_streams_not_repeated_ticks():
    core, clock, sent = running()
    core.request_stop()
    for _ in range(12):
        clock.advance()
        core.tick()
    assert core.state != "STOPPED"
    for _ in range(12):
        clock.advance()
        core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0))
        core.tick()
    assert core.state != "STOPPED"
    # The first recovered MC sample resets the gap; confirmation starts later.
    stable_for(core, clock, 1.3)
    assert core.state == "STOPPED"
    assert all(item == ZERO for item in sent[1:])


@pytest.mark.parametrize("reset", ["locomotion", "gap", "speed", "position", "body", "ownership"])
def test_stop_window_resets_on_discontinuity(reset):
    core, clock, sent = running()
    core.request_stop()
    stable_for(core, clock, .8)
    clock.advance(.7 if reset == "gap" else .1)
    feedback(core, mode="LOCOMOTION_DEFAULT" if reset == "locomotion" else "STAND_DEFAULT",
             speed=.05 if reset == "speed" else 0,
             x=.3 if reset == "position" else 0,
             body_status=2 if reset == "body" else 1,
             source="other" if reset == "ownership" else SOURCE)
    core.tick()
    assert core.state != "STOPPED"
    stable_for(core, clock, .5)
    assert core.state != "STOPPED"
    stable_for(core, clock, 1.2)
    assert core.state == "STOPPED"


def test_fast_odom_speed_reset_cannot_reuse_slow_mc_or_backdated_new_mc():
    core, clock, sent = running()
    core.request_stop()
    stable_for(core, clock, .8)
    clock.advance(.05)
    reset_at = clock.now
    core.update_odom((0,0,.65), (0,0,0,1), (.03,0,0), (0,0,0))
    clock.advance(.01)
    core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0))
    assert core.tick()["stop_window_start"] is None
    clock.advance(.01)
    # A new sequence alone is insufficient if its receipt predates the reset.
    assert core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100, received=reset_at-.01)
    assert core.tick()["stop_window_start"] is None
    clock.advance(.03)
    core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100)
    window_start = core.tick()["stop_window_start"]
    assert window_start > reset_at
    stable_for(core, clock, .9)
    assert core.state != "STOPPED"
    clock.advance(.07)
    feedback(core)
    core.tick()
    assert core.state == "STOPPED"
    assert all(command == ZERO for command in sent[1:])


def test_position_reset_cannot_reopen_on_triggering_pair_or_one_new_stream():
    core, clock, sent = running()
    core.request_stop()
    stable_for(core, clock, .8)
    clock.advance(.1)
    feedback(core, x=.3)
    reset_at = clock.now
    assert core.tick()["stop_window_start"] is None
    clock.advance(.05)
    core.update_odom((.3,0,.65), (0,0,0,1), (0,0,0), (0,0,0))
    assert core.tick()["stop_window_start"] is None
    clock.advance(.05)
    core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100)
    assert core.tick()["stop_window_start"] > reset_at
    stable_for(core, clock, x=.3)
    assert core.state == "STOPPED"


@pytest.mark.parametrize("topic", ["odom", "mc"])
def test_feedback_gap_requires_both_streams_after_the_gap_sample(topic):
    core, clock, sent = running()
    core.request_stop()
    stable_for(core, clock, .8)
    update = {
        "odom": lambda: core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0)),
        "mc": lambda: core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100),
    }
    other = "mc" if topic == "odom" else "odom"
    for _ in range(1 if topic == "odom" else 2):
        clock.advance(.2)
        update[other]()
    clock.advance(.06 if topic == "odom" else .11)
    reset_at = clock.now
    update[topic]()
    assert core.tick()["stop_window_start"] is None
    clock.advance(.01)
    update[other]()
    # Even the new sample which revealed the gap belongs to the reset barrier.
    assert core.tick()["stop_window_start"] is None
    clock.advance(.01)
    update[topic]()
    assert core.tick()["stop_window_start"] > reset_at
    stable_for(core, clock)
    assert core.state == "STOPPED"


def test_low_velocity_in_locomotion_is_not_stopped():
    core, clock, sent = running()
    core.request_stop()
    stable_for(core, clock, 2, mode="LOCOMOTION_DEFAULT")
    assert core.state != "STOPPED"


def test_publisher_exception_attempts_zero_and_is_visible():
    core, clock, sent = running()
    attempts = []
    def broken(command):
        attempts.append(command)
        if command == FORWARD:
            raise RuntimeError("transport failed")
    core.publish = broken
    result = core.tick()
    assert attempts == [FORWARD, ZERO]
    assert not result["attempts"][0]["ok"]
    assert result["attempts"][1]["ok"]
    assert core.state == "STOPPING"
    stable_for(core, clock)
    assert core.state == "STOPPED"
    assert "publish_error" in core.failures


def test_cli_default_and_help_do_not_load_ros_and_execute_guard_precedes_import(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "ROS_DOMAIN_ID": "0"}
    code = "import sys; from malecns_lab.experiments.x2 import main; main([]); assert 'rclpy' not in sys.modules; assert 'malecns_lab.control.x2' not in sys.modules"
    preview = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["mode"] == "preview"
    help_result = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2", "--help"], env=env, capture_output=True, text=True)
    assert help_result.returncode == 0
    guarded = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2", "--execute", "--output", str(tmp_path / "evidence")], env=env, capture_output=True, text=True)
    assert guarded.returncode != 0
    assert "requires --sim-only" in guarded.stderr
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("fault", ["source", "tilt", "odom_gap", "mc_gap"])
def test_fault_cannot_be_hidden_by_recovery_before_next_tick(fault):
    core, clock, sent = running()
    clock.advance(.6 if fault.endswith("gap") else .1)
    feedback(core, source="other" if fault == "source" else SOURCE,
             quaternion=(math.sin(.4),0,0,math.cos(.4)) if fault == "tilt" else (0,0,0,1))
    clock.advance(.001)
    feedback(core)
    core.tick()
    assert core.state != "FORWARD"
    assert sent[-1] == ZERO
    assert core.failures


def test_planned_deadline_stop_is_not_expiry_failure():
    core, clock, sent = running()
    deadline = core.intent_deadline
    clock.now = deadline
    core.request_stop("planned_stop")
    feedback(core)
    core.tick()
    assert sent[-1] == ZERO
    assert not core.failures


def test_evidence_failure_is_latched_without_interrupting_zero_cleanup(tmp_path, capsys):
    from malecns_lab.experiments.x2 import _EvidenceLog

    class BrokenStream:
        def __init__(self):
            self.writes = 0
            self.closed = False
        def write(self, text):
            self.writes += 1
            raise OSError("disk full")
        def close(self):
            self.closed = True
            raise OSError("close failed")

    stream = BrokenStream()
    evidence = _EvidenceLog(stream)
    assert not evidence.record({"kind": "tick"})
    assert not evidence.record({"kind": "tick"})
    assert stream.writes == 1
    core, clock, sent = running()
    core.request_stop("evidence_write_error", failure=True)
    stable_for(core, clock)
    assert core.state == "STOPPED"
    assert sent[-1] == ZERO
    assert "evidence_write_error" in core.failures
    assert evidence.summary(tmp_path / "summary.json", {"success": False})
    evidence.close()
    assert stream.closed
    assert capsys.readouterr().err.count("disk full") == 1


def test_summary_failure_does_not_throw_or_prevent_close(tmp_path):
    from io import StringIO
    from malecns_lab.experiments.x2 import _EvidenceLog
    stream = StringIO()
    evidence = _EvidenceLog(stream)
    assert not evidence.summary(tmp_path / "missing" / "summary.json", {"success": True})
    assert evidence.failure.startswith("evidence_write_error")
    evidence.close()
    assert stream.closed


def claiming(ttl=2.0):
    core, clock, sent = setup_adapter(allow_empty_source_claim=True)
    core.confirm_source_registration()
    core.arm()
    feedback(core, source="")
    assert core.ready_reason() == "ownership_lost"
    assert core.ready_reason(allow_empty_source=True) is None
    assert core.request_forward(ttl)
    assert core.state == "CLAIMING"
    core.tick()
    assert sent[-1] == FORWARD
    return core, clock, sent


@pytest.mark.parametrize("missing", ["opt_in", "registration", "arm", "odom", "mc", "source", "mode", "status", "body", "tilt", "freshness"])
def test_empty_source_claim_requires_explicit_opt_in_registration_and_feedback(missing):
    core, clock, sent = setup_adapter(allow_empty_source_claim=missing != "opt_in")
    if missing != "registration":
        core.confirm_source_registration()
    if missing != "arm":
        core.arm()
    feedback(core, source="another" if missing == "source" else "",
             mode="LOCOMOTION_DEFAULT" if missing == "mode" else "STAND_DEFAULT",
             body_status=2 if missing == "body" else 1,
             quaternion=(math.sin(.4), 0, 0, math.cos(.4)) if missing == "tilt" else (0,0,0,1))
    if missing == "odom":
        core.odom = None
    elif missing == "mc":
        core.mc = None
    elif missing == "status":
        clock.advance(.01)
        core.update_mc("", "STAND_DEFAULT", 1, 200)
    elif missing == "freshness":
        clock.advance(.6)
    assert not core.request_forward(2)
    core.tick()
    assert not sent or sent[-1] == ZERO
    clock.advance()
    feedback(core)
    core.tick()
    assert not sent or sent[-1] == ZERO


def test_registration_and_ready_feedback_alone_never_start_claim():
    core, clock, sent = setup_adapter(allow_empty_source_claim=True)
    core.confirm_source_registration()
    core.arm()
    stable_for(core, clock, source="")
    assert core.state == "STOPPING"
    assert sent and all(command == ZERO for command in sent)


def test_claim_requires_new_owner_ack_and_preserves_original_intent_deadline():
    core, clock, sent = claiming()
    deadline = core.intent_deadline
    assert core.ownership_confirmed_at is None
    clock.advance(.1)
    feedback(core, source="")
    core.tick()
    assert core.state == "CLAIMING"
    clock.advance(.1)
    feedback(core, source=SOURCE, mode="LOCOMOTION_DEFAULT")
    assert core.state == "FORWARD"
    assert core.ownership_confirmed_at == clock.now
    assert core.intent_deadline == deadline
    core.tick()
    assert sent[-1] == FORWARD
    assert not core.failures


def test_old_owner_sample_cannot_ack_claim():
    core, clock, sent = claiming()
    clock.advance(.1)
    assert not core.update_mc(SOURCE, "STAND_DEFAULT", 1, 100, received=10.0)
    core.tick()
    assert sent[-1] == ZERO
    assert core.ownership_confirmed_at is None


@pytest.mark.parametrize("fault", ["timeout", "competitor", "empty_locomotion", "tilt", "body", "odom_gap", "mc_gap"])
def test_claim_failure_cancels_and_recovery_does_not_retry(fault):
    core, clock, sent = claiming()
    if fault == "timeout":
        for _ in range(4):
            clock.advance(.1)
            feedback(core, source="")
            core.tick()
        clock.advance(.11)
        feedback(core, source=SOURCE)
    elif fault == "mc_gap":
        for _ in range(6):
            clock.advance(.1)
            core.update_odom((0,0,.65), (0,0,0,1), (0,0,0), (0,0,0))
        core.update_mc("", "STAND_DEFAULT", 1, 100)
    else:
        clock.advance(.3 if fault == "odom_gap" else .1)
        feedback(core, source="other" if fault == "competitor" else "",
                 mode="LOCOMOTION_DEFAULT" if fault == "empty_locomotion" else "STAND_DEFAULT",
                 body_status=2 if fault == "body" else 1,
                 quaternion=(math.sin(.4),0,0,math.cos(.4)) if fault == "tilt" else (0,0,0,1))
    # Even recovery before the publisher tick cannot hide an intervening fault.
    clock.advance(.001)
    feedback(core)
    core.tick()
    assert sent[-1] == ZERO
    assert core.state not in ("CLAIMING", "FORWARD")
    assert core.failures
    assert core.ownership_confirmed_at is None
    stable_for(core, clock)
    assert core.state == "STOPPED"
    clock.advance()
    feedback(core, source="")
    assert not core.request_forward(2)
    core.tick()
    assert sent[-1] == ZERO


def test_claim_renewal_cannot_extend_half_second_deadline():
    core, clock, sent = claiming()
    for _ in range(4):
        clock.advance(.1)
        feedback(core, source="")
        assert core.request_forward(5)
        core.tick()
    clock.advance(.11)
    # Current feedback is fresh; renewal itself must enforce the claim cap.
    assert not core.request_forward(5)
    core.tick()
    assert sent[-1] == ZERO
    assert "claim_timeout" in core.failures


def test_claim_cannot_renew_an_already_expired_intent():
    core, clock, sent = claiming(.1)
    clock.advance(.11)
    assert not core.request_forward(2)
    core.tick()
    assert sent[-1] == ZERO
    assert "intent_expired" in core.failures


def test_claim_time_counts_toward_five_second_cap_after_ack_and_renewals():
    core, clock, sent = claiming(5)
    started = clock.now
    clock.advance(.1)
    feedback(core)
    assert core.state == "FORWARD"
    for _ in range(48):
        clock.advance(.1)
        feedback(core)
        assert core.request_forward(5)
        assert core.intent_deadline == started + 5
        core.tick()
    clock.advance(.11)
    feedback(core)
    assert not core.request_forward(5)
    core.tick()
    assert sent[-1] == ZERO


@pytest.mark.parametrize("claim", [False, True])
@pytest.mark.parametrize("mode", ["STAND_DEFAULT", "LOCOMOTION_DEFAULT"])
def test_new_owned_startup_transition_recovers_running_without_extending_intent(claim, mode):
    core, clock, sent = claiming() if claim else running()
    deadline = core.intent_deadline
    clock.advance(.1)
    feedback(core, action_status=200, mode=mode)
    core.tick()
    assert core.state == "FORWARD"
    assert sent[-1] == FORWARD
    if claim:
        assert core.ownership_confirmed_at == clock.now
    clock.advance(.1)
    feedback(core, mode=mode)
    core.tick()
    for _ in range(4):
        clock.advance(.1)
        feedback(core, mode=mode)
        core.tick()
    assert core.state == "FORWARD"
    assert core.intent_deadline == deadline
    assert not core.failures


@pytest.mark.parametrize("claim", [False, True])
def test_repeated_transition_and_renewals_cannot_extend_initial_half_second(claim):
    core, clock, sent = claiming() if claim else running()
    started = clock.now
    for _ in range(4):
        clock.advance(.1)
        feedback(core, action_status=200)
        assert core.request_forward(5)
        result = core.tick()
        assert sent[-1] == FORWARD
        assert result["startup_transition_deadline"] == started + .5
    clock.now = started + .5
    assert not core.request_forward(5)
    core.tick()
    assert sent[-1] == ZERO
    assert "startup_transition_timeout" in core.failures
    clock.advance(.01)
    feedback(core)
    core.tick()
    assert sent[-1] == ZERO


def test_late_running_sample_cannot_hide_unresolved_startup_transition():
    core, clock, sent = claiming()
    started = clock.now
    for _ in range(4):
        clock.advance(.1)
        feedback(core, action_status=200)
        core.tick()
    clock.now = started + .5
    feedback(core, mode="LOCOMOTION_DEFAULT")
    core.tick()
    assert sent[-1] == ZERO
    assert "startup_transition_timeout" in core.failures


def test_transition_after_initial_window_is_rejected_even_after_renewal():
    core, clock, sent = running()
    for _ in range(5):
        clock.advance(.1)
        feedback(core)
        assert core.request_forward(5)
        core.tick()
    clock.advance(.1)
    feedback(core, action_status=200)
    core.tick()
    assert sent[-1] == ZERO
    assert "mc_not_ready" in core.failures


@pytest.mark.parametrize("overrides", [
    {"source": ""}, {"source": "competitor"}, {"mode": "PASSIVE_DEFAULT"},
    {"body_status": 2}, {"quaternion": (math.sin(.4), 0, 0, math.cos(.4))},
    {"action_status": 0}, {"action_status": 999},
])
def test_startup_transition_does_not_relax_other_feedback_gates(overrides):
    core, clock, sent = claiming()
    clock.advance(.1)
    feedback(core, **{"action_status": 200, **overrides})
    core.tick()
    assert sent[-1] == ZERO
    assert core.failures
    assert core.ownership_confirmed_at is None
    clock.advance(.01)
    feedback(core)
    core.tick()
    assert sent[-1] == ZERO


def test_startup_transition_never_counts_as_stopped():
    core, clock, sent = running()
    clock.advance(.1)
    feedback(core, action_status=200)
    core.request_stop()
    stable_for(core, clock, action_status=200)
    assert core.state != "STOPPED"
    assert sent[-1] == ZERO
    stable_for(core, clock)
    assert core.state == "STOPPED"


def test_startup_transition_cannot_outlive_original_intent_ttl():
    core, clock, sent = claiming(.2)
    clock.advance(.1)
    feedback(core, action_status=200)
    core.tick()
    assert sent[-1] == FORWARD
    clock.advance(.11)
    feedback(core, action_status=200)
    core.tick()
    assert sent[-1] == ZERO
    assert "intent_expired" in core.failures
