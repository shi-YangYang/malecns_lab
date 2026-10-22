"""Single-run file exchange and bounded neural-loop scheduling (stdlib only)."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .x2 import SOURCE

MAX_ROUNDS = 2
BRAIN_TIMEOUT = 30.0
OBSERVATION_TIMEOUT = 45.0
ROS_TIMEOUT = 120.0
PULSE_SECONDS = 1.0
STOP_TIMEOUT = 12.0
POSE_TIMEOUT = 40.0
POSE_WINDOW = 1.0
POSE_ANGULAR_LIMIT = .005
POSE_DRIFT_LIMIT = .005


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(value):
    require("message_hash" not in value, "Already sealed")
    return dict(value, message_hash=digest(value))


def verify(value, kind, run_id, sequence):
    require(isinstance(value, dict) and value.get("kind") == kind and value.get("run_id") == run_id
            and type(value.get("sequence")) is int and value["sequence"] == sequence, "Message identity or order differs")
    require(value.get("message_hash") == digest({k: v for k, v in value.items() if k != "message_hash"}), "Message hash differs")
    return value


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)
    require(isinstance(value, dict), "JSON object required")
    return value


def atomic(path, value):
    """Publish once: a complete immutable message appears in one rename."""
    path = Path(path)
    require(not path.exists(), "Refusing to replace published evidence: " + str(path))
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # One writer owns each channel; do not silently overwrite an existing message.
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def intent(action):
    require(isinstance(action, list) and len(action) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in action), "Invalid raw three-axis action")
    return "FORWARD" if action[0] > .03 else "STOP"


def unit(q):
    length = math.sqrt(sum(v*v for v in q))
    return [v/length for v in q]


def angle(a, b):
    return 2*math.acos(min(1., abs(sum(x*y for x, y in zip(unit(a), unit(b))))))


def yaw(q):
    x, y, z, w = unit(q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class StandObserver:
    """Independent initial stand evidence; an empty source is never ownership.

    Call on every received message AND each tick. A reset requires fresh samples
    from both streams, including after any transient invalid/fast feedback.
    """
    def __init__(self):
        self.barrier = -math.inf
        self.reset_pair = self.last_pair = (0, 0)
        self.start = self.last_time = None
        self.position = None
        self.path = 0.
        self.seen = (None, None)
        self.qualified = False

    def reset(self, now, pair):
        self.barrier, self.reset_pair = now, pair
        self.start = self.last_time = self.position = None
        self.last_pair = (0, 0)
        self.path = 0.
        self.qualified = False

    def update(self, core, now):
        odom, mc = core.odom, core.mc
        pair = (odom.sequence if odom else 0, mc.sequence if mc else 0)
        received = (odom.received if odom else None, mc.received if mc else None)
        gaps = any(new is not None and old is not None and new-old > limit
                   for new, old, limit in zip(received, self.seen, (core.odom_max_age, core.mc_max_age)))
        self.seen = received
        valid = (not core.failures and odom is not None and mc is not None
                 and 0 <= now-odom.received <= core.odom_max_age
                 and 0 <= now-mc.received <= core.mc_max_age
                 and mc.source in ("", SOURCE) and mc.mode == "STAND_DEFAULT"
                 and mc.action_status == 100 and mc.body_status == 1 and odom.tilt <= .5
                 and math.hypot(*odom.linear[:2]) < .02)
        if gaps or not valid:
            self.reset(now, pair)
            return False
        if min(received) <= self.barrier or any(n <= o for n, o in zip(pair, self.reset_pair)):
            return False
        if any(n <= o for n, o in zip(pair, self.last_pair)):
            return self.qualified
        stamp = min(received)
        if self.position is not None:
            distance, dt = math.dist(odom.position[:2], self.position[:2]), stamp-self.last_time
            self.path += distance
            if dt <= 0 or distance > .02*dt+.005 or self.path > .02*(stamp-self.start)+.005:
                self.reset(now, pair)
                return False
        if self.start is None:
            self.start = stamp
        self.last_pair, self.last_time, self.position = pair, stamp, odom.position
        self.qualified = stamp-self.start >= 1.
        return self.qualified


class PoseObserver:
    """Stricter input readiness, checked on every callback and publishing tick.

    Both streams must cross the reset barrier. Pair advances establish elapsed
    evidence; intermediate odometry still resets it immediately on any excess.
    """
    def __init__(self, core, now):
        self.seen = (core.odom.received, core.mc.received)
        self.resets = 0
        self.reset(now, (core.odom.sequence, core.mc.sequence), "input_wait_started")

    def reset(self, now, pair, reason):
        self.barrier, self.reset_pair = now, pair
        self.start = self.last_time = self.orientation = self.start_pair = None
        self.last_pair = (0, 0)
        self.max_angular = self.max_drift = 0.
        self.paired_samples = 0
        self.qualified = False
        self.resets += 1
        self.reset_reason = reason

    def update(self, core, now, *, owned):
        odom, mc = core.odom, core.mc
        pair = (odom.sequence if odom else 0, mc.sequence if mc else 0)
        received = (odom.received if odom else None, mc.received if mc else None)
        gaps = any(new is not None and old is not None and new-old > limit
                   for new, old, limit in zip(received, self.seen, (core.odom_max_age, core.mc_max_age)))
        self.seen = received
        valid = (not core.failures and odom is not None and mc is not None
                 and 0 <= now-odom.received <= core.odom_max_age
                 and 0 <= now-mc.received <= core.mc_max_age
                 and mc.source in ("", SOURCE) and mc.mode == "STAND_DEFAULT"
                 and mc.action_status == 100 and mc.body_status == 1 and odom.tilt <= .5
                 and math.hypot(*odom.linear[:2]) < .02
                 and (not owned or core.state == "STOPPED"))
        if gaps or not valid:
            self.reset(now, pair, "feedback_gap" if gaps else "stand_not_ready")
            return False
        angular = math.sqrt(sum(v*v for v in odom.angular))
        drift = angle(odom.quaternion, self.orientation) if self.orientation is not None else 0.
        # Check before pair gating: an odom transient must not be hidden by a
        # later good odom while the slower MC stream is still unchanged.
        if angular > POSE_ANGULAR_LIMIT or drift > POSE_DRIFT_LIMIT:
            self.reset(now, pair, "angular_speed" if angular > POSE_ANGULAR_LIMIT else "pose_drift")
            return False
        if min(received) <= self.barrier or any(n <= o for n, o in zip(pair, self.reset_pair)):
            return False
        self.max_angular = max(self.max_angular, angular)
        self.max_drift = max(self.max_drift, drift)
        if any(n <= o for n, o in zip(pair, self.last_pair)):
            return self.qualified
        if self.start is None:
            # The orientation reference exists only at the later receipt; do
            # not backdate its stability to the slower stream's old sample.
            self.start, self.start_pair = max(received), pair
            self.orientation = odom.quaternion
        self.last_pair, self.last_time = pair, min(received)
        self.paired_samples += 1
        self.qualified = self.last_time-self.start >= POSE_WINDOW
        return self.qualified

    def evidence(self):
        require(self.qualified, "Pose window is not qualified")
        return {"clock_domain": "ros_bridge_monotonic", "window_start_monotonic": self.start,
                "window_end_monotonic": self.last_time, "duration_seconds": self.last_time-self.start,
                "start_sequences": list(self.start_pair), "end_sequences": list(self.last_pair),
                "reference_orientation_xyzw": unit(self.orientation),
                "max_angular_speed_rad_s": self.max_angular, "max_orientation_drift_rad": self.max_drift,
                "angular_speed_limit_rad_s": POSE_ANGULAR_LIMIT, "orientation_drift_limit_rad": POSE_DRIFT_LIMIT,
                "required_seconds": POSE_WINDOW, "paired_samples": self.paired_samples,
                "reset_count": self.resets, "last_reset_reason": self.reset_reason}


class LoopController:
    """Pure scheduling around the unchanged motion adapter and atomic mailbox."""
    def __init__(self, core, directory, run_id, *, execution_id=None, clock=time.monotonic, record=lambda *a, **k: None):
        self.core, self.directory, self.run_id = core, Path(directory), run_id
        self.execution_id = execution_id
        self.clock, self.record = clock, record
        self.started = clock()
        self.deadline = self.started + ROS_TIMEOUT
        self.phase = "initial_stand"
        self.phase_deadline = min(self.deadline-STOP_TIMEOUT, self.started+15.)
        self.observer = StandObserver()
        self.pose_observer = None
        self.pose_wait_started = None
        self.sequence = 0
        self.target = None
        self.observation = self.response = None
        self.rounds = []
        self.failure = None
        self.ever_forward = False
        self.pulse_started = None
        self.stop_started = None
        self.done = False
        self.final_ack = False
        self.initial_position = None
        self.pending_stop_reason = None

    def feedback(self):
        now = self.clock()
        qualified = self.observer.update(self.core, now)
        if self.phase == "pose_settling":
            self.pose_observer.update(self.core, now, owned=self.ever_forward)
        return qualified

    def fail(self, reason):
        if self.failure is None:
            self.failure = reason
            self.core.request_stop(reason, failure=True)
            self.phase, self.stop_started = "failed_stop", self.clock()
            self.phase_deadline = min(self.deadline, self.clock()+STOP_TIMEOUT)
            self.record("failure", reason=reason)

    def stationary(self):
        qualified = self.feedback()
        if self.ever_forward:
            return self.core.state == "STOPPED" and self.core.ready_reason() is None
        return qualified

    def collect(self):
        require(self.stationary(), "Observation requires a qualified stand")
        require(self.phase == "pose_settling" and self.pose_observer.qualified,
                "Observation requires a qualified pose window")
        odom, mc = self.core.odom, self.core.mc
        if self.target is None:
            require(abs(yaw(odom.quaternion)) < .05, "Initial yaw exceeds registered bound")
            self.initial_position = list(odom.position)
            self.target = [odom.position[0]+2., odom.position[1], odom.position[2]]
        self.sequence += 1
        self.observation = seal({"kind": "observation", "run_id": self.run_id, "sequence": self.sequence,
            "source": "live_odometry_virtual_marker", "clock_domain": "ros_bridge_monotonic",
            "collected_monotonic": self.clock(), "odom": asdict(odom), "mc": asdict(mc),
            "camera_position_m": list(odom.position), "camera_orientation_xyzw": unit(odom.quaternion),
            "target_world_m": self.target, "stationary_evidence": "owned_core_STOPPED" if self.ever_forward else "initial_stand_observer",
            "initial_stand_window_start": self.observer.start,
            "pose_stability": dict(self.pose_observer.evidence(), wait_started_monotonic=self.pose_wait_started,
                                   wait_seconds=self.clock()-self.pose_wait_started, timeout_seconds=POSE_TIMEOUT)})
        atomic(self.directory / f"observation-{self.sequence:03d}.json", self.observation)
        self.record("observation", observation=self.observation)
        self.phase = "thinking"
        self.phase_deadline = min(self.deadline-STOP_TIMEOUT, self.clock()+BRAIN_TIMEOUT)

    def wait_for_pose(self):
        self.pose_wait_started = self.clock()
        self.pose_observer = PoseObserver(self.core, self.pose_wait_started)
        self.phase = "pose_settling"
        self.phase_deadline = min(self.deadline-STOP_TIMEOUT, self.pose_wait_started+POSE_TIMEOUT)
        self.record("pose_wait", next_sequence=self.sequence+1, deadline_monotonic=self.phase_deadline,
                    timeout_seconds=POSE_TIMEOUT, required_seconds=POSE_WINDOW,
                    angular_speed_limit_rad_s=POSE_ANGULAR_LIMIT, orientation_drift_limit_rad=POSE_DRIFT_LIMIT)

    def accept(self, response):
        verify(response, "decision", self.run_id, self.sequence)
        require(response.get("observation_hash") == self.observation["message_hash"], "Decision observation differs")
        require(response.get("execution_id") == self.execution_id, "Decision execution differs")
        require(response.get("intent") == intent(response.get("raw_action")), "Decision violates frozen gate")
        require(response.get("frame_index") == self.sequence and isinstance(response.get("input_sha256"), str)
                and len(response["input_sha256"]) == 64, "Decision input/state identity differs")
        require(self.stationary(), "Stand lost during inference")
        current, old = self.core.odom, self.observation["odom"]
        require(current.sequence > old["sequence"] and self.core.mc.sequence > self.observation["mc"]["sequence"],
                "No new feedback after inference")
        require(math.dist(current.position[:2], old["position"][:2]) <= .02
                and angle(current.quaternion, old["quaternion"]) <= .03, "Observation drift exceeds bound")
        self.response = response
        self.record("decision", decision=response, accepted_pose=asdict(current), accepted_mc=asdict(self.core.mc))
        self.pulse_started = None
        if response["intent"] == "FORWARD":
            require(self.core.request_forward(PULSE_SECONDS), "Forward rejected: " + self.core.reason)
            self.ever_forward = True
            self.pulse_started = self.clock()
            self.phase, self.phase_deadline = "forward", self.core.intent_deadline
        else:
            self.core.request_stop("neural_stop")
            self.phase, self.stop_started = "settling", self.clock()
            self.phase_deadline = min(self.deadline-STOP_TIMEOUT, self.clock()+STOP_TIMEOUT)

    def finish_round(self):
        record = {"sequence": self.sequence, "observation_hash": self.observation["message_hash"],
                  "decision_hash": self.response["message_hash"], "intent": self.response["intent"],
                  "raw_action": self.response["raw_action"], "input_sha256": self.response["input_sha256"],
                  "pulse_started_monotonic": self.pulse_started,
                  "stop_confirmed_monotonic": self.clock(), "stop_kind": "owned_core_STOPPED" if self.ever_forward else "initial_empty_source_stand",
                  "final_odom": asdict(self.core.odom), "final_mc": asdict(self.core.mc),
                  "forward_displacement_m": self.core.odom.position[0]-self.observation["odom"]["position"][0]}
        self.rounds.append(record)
        final = self.sequence == MAX_ROUNDS
        atomic(self.directory / f"ack-{self.sequence:03d}.json", seal({"kind": "ack", "run_id": self.run_id,
               "sequence": self.sequence, "decision_hash": self.response["message_hash"], "final": final,
               "stopped": True, "round": record}))
        self.record("round_stopped", **record)
        self.response = None
        if final:
            self.final_ack = True
            self.phase = "brain_seal"
            self.phase_deadline = min(self.deadline-1., self.clock()+BRAIN_TIMEOUT)
        else:
            self.wait_for_pose()

    def update(self):
        if self.done:
            return
        now = self.clock()
        self.feedback()
        for name in ("host_abort.json", "result.json", "worker_result.json"):
            path = self.directory / name
            if path.exists():
                try:
                    result = read(path)
                    if name == "host_abort.json" or result.get("status") == "failed":
                        self.fail("host_failure:" + str(result.get("reason")))
                except Exception as error:
                    self.fail("host_status_invalid:" + str(error))
        if self.core.mc is not None and self.core.mc.source not in ("", SOURCE):
            self.fail("other_input_source")
        if self.core.failures and self.failure is None:
            self.fail("motion_core:" + self.core.failures[0])
        if now >= self.deadline:
            if self.failure is None:
                self.fail("ros_total_timeout")
            self.done = True
            return
        if self.phase == "failed_stop":
            self.done = self.core.state == "STOPPED" or now >= self.phase_deadline
            return
        try:
            # Never accept a future output or a changed/old-run output as a new round.
            for path in self.directory.glob("decision-*.json"):
                number = int(path.stem.split("-")[1])
                require(1 <= number <= self.sequence, "Unsolicited/out-of-order decision")
                if number <= len(self.rounds):
                    old = read(path)
                    verify(old, "decision", self.run_id, number)
                    require(old["message_hash"] == self.rounds[number-1]["decision_hash"], "Consumed decision changed")
            if self.phase == "forward" and now >= self.phase_deadline:
                self.core.request_stop("planned_stop")
                self.phase, self.stop_started = "settling", now
                self.phase_deadline = min(self.deadline-1., now+STOP_TIMEOUT)
            elif now >= self.phase_deadline:
                raise ValueError(self.phase + "_timeout")
            if self.phase == "initial_stand" and self.stationary():
                self.wait_for_pose()
            elif self.phase == "pose_settling" and self.stationary() and self.pose_observer.qualified:
                self.collect()
            elif self.phase == "thinking":
                path = self.directory / f"decision-{self.sequence:03d}.json"
                if path.exists():
                    self.accept(read(path))
            elif self.phase == "settling" and self.stationary():
                self.finish_round()
            elif self.phase == "brain_seal":
                path = self.directory / "host_ack.json"
                if path.exists():
                    result = verify(read(path), "host_ack", self.run_id, MAX_ROUNDS)
                    require(result.get("status") == "completed" and result.get("parameters_unchanged") is True
                            and result.get("supervisor_status") == "completed",
                            "Host did not seal supervised frozen inference")
                    self.done = True
        except Exception as error:
            self.fail(type(error).__name__ + ":" + str(error))

    def summary(self):
        moved = [r for r in self.rounds if r["intent"] == "FORWARD" and r["forward_displacement_m"] >= .2]
        changed_after_motion = any(self.rounds[index]["intent"] == "FORWARD"
            and self.rounds[index]["forward_displacement_m"] >= .2
            and self.rounds[index+1]["input_sha256"] != self.rounds[index]["input_sha256"]
            for index in range(len(self.rounds)-1))
        stopped = self.stationary() and self.core.state not in ("CLAIMING", "FORWARD")
        success = self.done and self.final_ack and self.failure is None and not self.core.failures and stopped \
            and len(self.rounds) >= 2 and bool(moved) and changed_after_motion
        return {"run_id": self.run_id, "success": success, "reason": self.failure or (None if success else "engineering_loop_acceptance_not_met"),
                "rounds": self.rounds, "new_observation_after_motion": changed_after_motion,
                "final_stopped": stopped, "active_forward": self.core.state in ("CLAIMING", "FORWARD"),
                "core_failures": list(self.core.failures), "elapsed_seconds": self.clock()-self.started,
                "scope": "longitudinal engineering loop with live odometry virtual marker; not real camera or trained navigation"}
