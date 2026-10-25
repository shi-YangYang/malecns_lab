"""Bounded, single-skill execution. No ROS, task planner, or brain on import.

The caller must first apply x2_behavior.DecisionGate. This layer also rejects
replays and stale commands. It never chooses another behavior on rejection.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
import time
from types import MappingProxyType
from typing import Callable

from .x2 import X2MotionAdapter, SOURCE, ZERO, FORWARD

SKILL_PACK_ID = "malecns.x2.skills.v1"
BEHAVIORS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")
CONTINUOUS_BEHAVIORS = ("ADVANCE", "TURN_LEFT", "TURN_RIGHT")
RIGHT_ARM = tuple("right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll"))
WAVE_NAMES = frozenset(("1002", "SM_wave_right_hand", "SM_wave_right_hand.csv"))
# Spatial containment for the pinned simulation contract, never real hardware.
# Playing an upper-body animation permits balance sway; this is not a hardware
# safety rating. Completion still requires the original 2 cm / stopped window.
WAVE_PROBE_ACTIVE_TRANSLATION_M = .05


def validate_command_durations(values):
    """Copy a complete, bounded velocity budget; never retain caller state."""
    if not isinstance(values, Mapping) or set(values) != set(CONTINUOUS_BEHAVIORS):
        raise ValueError("Continuous skill durations require the complete behavior set")
    copied = dict(values)
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 2
           for v in copied.values()):
        raise ValueError("Continuous skill durations must be finite and within (0, 2]")
    return {name: float(copied[name]) for name in CONTINUOUS_BEHAVIORS}


def yaw(quaternion):
    x, y, z, w = quaternion
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def angle_delta(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


@dataclass(frozen=True)
class SkillCommand:
    decision_seq: int
    request_id: str
    behavior: str
    phase: str
    intensity: float
    observed_at: float


class VelocityCore(X2MotionAdapter):
    """Reuse original source/TTL guards; add angular stop evidence locally."""
    def __init__(self, publish, *, clock=time.monotonic, fixed_advance=False, fixed_turn=False):
        self.fixed_advance = fixed_advance
        self.fixed_turn = fixed_turn
        self.base_motion_requested = False
        self.target = ZERO
        self._angular_start = self._angular_last = self._angular_yaw = None
        self._angular_path = 0.
        self._angular_seq = 0
        self._sink = publish
        super().__init__(self._publish, clock=clock, allow_empty_source_claim=True)

    def _publish(self, command):
        self._sink(self.target if command == FORWARD else ZERO)

    def request_velocity(self, behavior, intensity, ttl):
        self.base_motion_requested = True
        if behavior not in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT") or not math.isfinite(intensity) or not 0 < intensity <= 1 or not 0 < ttl <= 2:
            self.request_stop("invalid_skill_target", failure=True)
            return False
        if self.fixed_advance and behavior == "ADVANCE" and ttl != 1.:
            self.request_stop("fixed_advance_duration", failure=True)
            return False
        self.target = self.resolve_velocity(behavior, intensity)
        return self.request_forward(ttl)

    def resolve_velocity(self, behavior, intensity):
        magnitude = .2 * intensity
        if behavior == "ADVANCE":
            return (.2 if self.fixed_advance else magnitude, 0., 0.)
        angular = .4 if self.fixed_turn else magnitude
        return (0., 0., angular if behavior == "TURN_LEFT" else -angular)

    def _reset_window(self):
        super()._reset_window()
        self._angular_start = self._angular_last = self._angular_yaw = None
        self._angular_path = 0.
        self._angular_seq = self._odom_sequence

    def _confirm_stop(self, now):
        odom = self.odom
        if odom is None or abs(odom.angular[2]) >= .03:
            self._reset_window()
            return
        if odom.sequence > self._angular_seq:
            current = yaw(odom.quaternion)
            if self._angular_yaw is not None:
                change = abs(angle_delta(current, self._angular_yaw))
                dt = odom.received - self._angular_last
                self._angular_path += change
                if dt <= 0 or change > .03*dt+.003 or self._angular_path > .03:
                    self._reset_window()
                    return
            if self._angular_start is None:
                self._angular_start = odom.received
            self._angular_last, self._angular_yaw, self._angular_seq = odom.received, current, odom.sequence
        super()._confirm_stop(now)
        if self.state == "STOPPED" and (self._angular_start is None or odom.received-self._angular_start < 1.):
            self.state = "STOPPING"

    def tick(self):
        result = super().tick()
        for attempt in result["attempts"]:
            if attempt["command"] == FORWARD:
                attempt["command"] = self.target
        result.update(angular_stop_start=self._angular_start, angular_stop_path_rad=self._angular_path)
        return result


class SkillExecutor:
    """Actual feedback completes skills; ACK alone cannot complete an action.

    Wave is disabled by default. ``wave_probe`` permits isolated simulation
    characterization only and never enables production cancellation. The fixed
    MC contract probe records global query state separately from task identity.
    The separately bound simulation qualification permits observed cancellation;
    all other wave modes retain their prior UNKNOWN/natural-finish behavior.
    Velocity zero is never arm cancellation.
    """
    def __init__(self, publish: Callable, preset: Callable, *, clock=time.monotonic,
                 enabled=("HOLD",), wave_probe=False, command_seconds=1., max_decision_age=30., record=None,
                 command_seconds_by_behavior=None, skill_profile_id=None,
                 wave_contract_probe=False, wave_cancel_probe=False,
                 wave_return_pose=None, wave_return_reference=None,
                 wave_qualification=None, wave_first_pose=None,
                 execution_profile=None, simulation_only=False, qualification_only=False):
        if not 0 < command_seconds <= 2 or not 0 < max_decision_age <= 30:
            raise ValueError("Invalid bounded execution budget")
        durations = validate_command_durations(command_seconds_by_behavior if command_seconds_by_behavior is not None
            else {name: command_seconds for name in CONTINUOUS_BEHAVIORS})
        if skill_profile_id is not None and (not isinstance(skill_profile_id, str) or not skill_profile_id.strip()):
            raise ValueError("Invalid skill profile identity")
        self.command_seconds_by_behavior = MappingProxyType(durations)
        self.skill_profile_id = skill_profile_id
        self.active_command_seconds = None
        from .x2_skill_profile import (SIM_FIXED_ADVANCE_PROFILE_ID, SIM_FROZEN_FULL_PROFILE_ID,
                                      validate_candidate_skill_profile)
        self.full_action = skill_profile_id == SIM_FROZEN_FULL_PROFILE_ID
        self.fixed_advance = execution_profile is not None
        if self.fixed_advance:
            profile = validate_candidate_skill_profile(execution_profile)
            expected_enabled = {'HOLD', 'ADVANCE', 'TURN_LEFT', 'TURN_RIGHT'}
            if self.full_action:
                expected_enabled.add('WAVE_RIGHT')
            if (profile['profile_id'] not in (SIM_FIXED_ADVANCE_PROFILE_ID, SIM_FROZEN_FULL_PROFILE_ID) or skill_profile_id != profile['profile_id']
                    or simulation_only is not True or qualification_only is not True
                    or durations != profile['command_seconds_by_behavior']
                    or set(enabled) != expected_enabled or wave_probe
                    or (wave_qualification != profile.get('wave_contract') if self.full_action else wave_qualification is not None)):
                raise ValueError("Explicit fixed ADVANCE simulation qualification required")
        elif skill_profile_id in (SIM_FIXED_ADVANCE_PROFILE_ID, SIM_FROZEN_FULL_PROFILE_ID):
            raise ValueError("Missing fixed ADVANCE execution contract")
        self.resolved_velocity = None
        self.clock, self.preset = clock, preset
        self.motion = VelocityCore(publish, clock=clock, fixed_advance=self.fixed_advance, fixed_turn=self.full_action)
        self.enabled, self.wave_probe = set(enabled), bool(wave_probe)
        if (wave_contract_probe and not wave_probe) or (wave_cancel_probe and not wave_contract_probe):
            raise ValueError("Fixed-contract probes require isolated wave_probe")
        self.wave_contract_probe, self.wave_cancel_probe = bool(wave_contract_probe), bool(wave_cancel_probe)
        self.wave_qualified = wave_qualification is not None
        if self.wave_qualified:
            from .x2_skill_profile import SIM_WAVE_PROFILE_ID, sim_wave_contract
            if (wave_qualification != sim_wave_contract() or skill_profile_id not in (SIM_WAVE_PROFILE_ID, SIM_FROZEN_FULL_PROFILE_ID)
                    or wave_probe or wave_contract_probe or wave_cancel_probe):
                raise ValueError("Explicit fixed simulation wave qualification required")
        self.fixed_wave = self.wave_contract_probe or self.wave_qualified
        if wave_return_pose is not None and (not self.fixed_wave or not isinstance(wave_return_pose, Mapping)
                or set(wave_return_pose) != set(RIGHT_ARM)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in wave_return_pose.values())
                or wave_return_reference not in ("official_animation_first_frame", "official_animation_terminal_frame")):
            raise ValueError("Invalid fixed-wave return pose/reference")
        if self.wave_qualified and (wave_return_pose is None or wave_return_reference != "official_animation_terminal_frame"
                or not isinstance(wave_first_pose, Mapping) or set(wave_first_pose) != set(RIGHT_ARM)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in wave_first_pose.values())):
            raise ValueError("Fixed official first and terminal wave poses required")
        self.wave_first_pose = dict(wave_first_pose) if wave_first_pose is not None else None
        self.wave_terminal_pose = dict(wave_return_pose) if wave_return_pose is not None else None
        self.wave_dispatch_at = None
        self.wave_return_confirmed = False
        self.wave_return_pose = dict(wave_return_pose) if wave_return_pose is not None else None
        self.wave_return_reference = wave_return_reference if wave_return_pose is not None else "pre_action_measured_pose"
        self.command_seconds, self.max_decision_age = command_seconds, max_decision_age
        self.record = record or (lambda event: None)
        self.last_seq = -1
        self.command = None
        self.finish_command = None
        self.events = []
        self.status, self.reason = "IDLE", "idle"
        self.execution_id = None
        self.started_at = self.deadline = self.stop_at = None
        self.start_position = self.start_yaw = None
        self.distance = self.signed_angle = 0.
        self.forward_distance = self.lateral_distance = 0.
        self.last_yaw = None
        self.moved = False
        self.cancel_requested = False
        self.player = None
        self.joints = None
        self.joints_at = -math.inf
        self.baseline = None
        self.joint_excursion = 0.
        self.joint_return = math.inf
        self.joint_baseline_return = math.inf
        self.wave_peak_distance = self.wave_peak_yaw = self.wave_peak_tilt = 0.
        self.query_state = None
        self.query_at = -math.inf
        self.task_id = None
        self.saw_query_running = self.saw_player_playing = False
        self.preset_sent_at = self.measured_wave_at = None
        self.probe_cancel_at = self.probe_cancel_ack = None
        self.probe_cancel_sent_at = None
        self.probe_cancel_task_id = None
        self.saw_cancel_interrupting = self.saw_cancel_idle = False
        self._idle_since = None
        self._idle_pose = None
        self._idle_pair = (0, 0)
        self._recovery_since = None
        self.terminal = False

    def _emit(self, status, reason, *, fact="status", **extra):
        self.status, self.reason = status, reason
        event = dict(event_seq=len(self.events), received_monotonic=self.clock(), status=status,
                     reason=reason, fact=fact, execution_id=self.execution_id,
                     decision_seq=self.command.decision_seq if self.command else None,
                     request_id=self.command.request_id if self.command else None,
                     behavior=self.command.behavior if self.command else "HOLD",
                     skill_profile_id=self.skill_profile_id, command_seconds=self.active_command_seconds,
                     intensity=self.command.intensity if self.command else None,
                     started_at=self.started_at, deadline=self.deadline,
                     finish_decision_seq=self.finish_command.decision_seq if self.finish_command else None,
                     finish_request_id=self.finish_command.request_id if self.finish_command else None,
                     **self._execution_details(), **extra)
        self.events.append(event)
        self.record(event)

    def update_odom(self, position, quaternion, linear, angular, *, received=None):
        ok = self.motion.update_odom(position, quaternion, linear, angular, received=received)
        if ok and self.command and self.start_position is not None:
            self.distance = math.dist(position[:2], self.start_position[:2])
            dx, dy = (position[i]-self.start_position[i] for i in (0, 1))
            self.forward_distance = dx*math.cos(self.start_yaw) + dy*math.sin(self.start_yaw)
            self.lateral_distance = -dx*math.sin(self.start_yaw) + dy*math.cos(self.start_yaw)
            current = yaw(quaternion)
            self.signed_angle += angle_delta(current, self.last_yaw)
            self.last_yaw = current
            if self.command.behavior == "WAVE_RIGHT":
                self.wave_peak_distance = max(self.wave_peak_distance, self.distance)
                self.wave_peak_yaw = max(self.wave_peak_yaw, abs(self.signed_angle))
                self.wave_peak_tilt = max(self.wave_peak_tilt, self.motion.odom.tilt)
            moving = self._advance_direction_ok() if self.command.behavior == "ADVANCE" else abs(self.signed_angle) > .02
            if self.command.behavior in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT") and moving and not self.moved and not self.terminal:
                self.moved = True
                if self.status != "UNKNOWN":
                    self._emit("STARTED", "measured_motion", **self._displacement_evidence())
        return ok

    def update_mc(self, source, mode, body_status, action_status, *, player_state=0, motion="", area=0, motion_type=0, received=None):
        ok = self.motion.update_mc(source, mode, body_status, action_status, received=received)
        if not ok:
            return False
        self.player = (self.motion.mc.received, player_state, motion, area, motion_type)
        if self.command and not self.terminal and self.command.behavior in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT") and player_state != 0:
            self._unknown("external_preset_during_velocity")
        if self.command and not self.terminal and self.command.behavior == "WAVE_RIGHT" and self.started_at is not None:
            if self.fixed_wave and (self.preset_sent_at is None or self.player[0] <= self.preset_sent_at):
                return True
            if player_state in (1, 2):
                name = str(motion).rsplit("/", 1)[-1]
                if name not in WAVE_NAMES or area != 2 or motion_type != 1:
                    self._unknown("preset_attribution_mismatch")
                elif self.saw_cancel_interrupting:
                    self._unknown("preset_restarted_after_cancel")
                elif player_state == 2:
                    self.saw_player_playing = True
            elif player_state == 3 and self.probe_cancel_at is not None:
                # DDS callbacks can precede the child's IPC receipt. Such a
                # sample is not cancellation evidence until actual-send time is
                # known and a subsequent fresh sample is observed.
                if self.probe_cancel_sent_at is not None and self.player[0] > self.probe_cancel_sent_at:
                    self.saw_cancel_interrupting = True
            elif player_state == 0 and self.saw_cancel_interrupting:
                self.saw_cancel_idle = True
            elif player_state != 0:
                self._unknown("preset_interrupted_or_error")
        return True

    def update_joints(self, joints, *, received=None):
        now = self.clock()
        stamp = now if received is None else received
        valid = math.isfinite(stamp) and self.joints_at < stamp <= now and all(
            name in joints and len(joints[name]) == 2 and all(math.isfinite(x) for x in joints[name]) for name in RIGHT_ARM)
        if not valid:
            self.joints = None
            return False
        self.joints, self.joints_at = dict(joints), stamp
        if self.baseline is not None and (not self.fixed_wave or
                (self.preset_sent_at is not None and stamp > self.preset_sent_at)):
            deviation = max(abs(joints[k][0]-self.baseline[k]) for k in RIGHT_ARM)
            self.joint_excursion = max(self.joint_excursion, deviation)
            self.joint_baseline_return = deviation
            reference = self.wave_return_pose if self.wave_return_pose is not None else self.baseline
            self.joint_return = max(abs(joints[k][0]-reference[k]) for k in RIGHT_ARM)
        return True

    def preset_ack(self, code, task_id, state):
        if (not self.command or self.command.behavior != "WAVE_RIGHT" or self.task_id is not None
                or self.status != "ACCEPTED" and not (self.wave_qualified and self.status == "UNKNOWN")):
            return False
        minimum = 0 if self.fixed_wave else 1
        if (code != 0 or type(task_id) is not int or not minimum <= task_id < 2**64
                or (self.fixed_wave and state != 400)):
            self._unknown("preset_not_accepted")
            return False
        self.task_id = task_id
        if self.status != "UNKNOWN":
            self._emit("ACCEPTED", "preset_ack", fact="preset_ack", task_id=task_id, query_state=state,
                       preset_send_confirmed=self.preset_sent_at is not None, preset_sent_at=self.preset_sent_at)
        return True

    def preset_query(self, code, task_id, state, *, received=None):
        now = self.clock()
        stamp = now if received is None else received
        if not math.isfinite(stamp) or not self.query_at < stamp <= now:
            return False
        valid = code == 0 and type(task_id) is int and 0 <= task_id < 2**64 and state in (1, 400)
        self.query_at, self.query_state = stamp, state if valid else None
        if self.command and not self.terminal and self.command.behavior == "WAVE_RIGHT" and (not self.wave_qualified or self.wave_dispatch_at is not None):
            if (code != 0 or self.task_id is None or type(task_id) is not int or not 0 <= task_id < 2**64
                    or (not self.fixed_wave and task_id != self.task_id) or state not in (1, 400)):
                self._unknown("preset_query_attribution_unknown")
                return False
            if state == 400:
                self.saw_query_running = True
        return valid

    def preset_sent(self, received):
        """Helper and parent share the Linux monotonic clock; record actual send."""
        if (not self.fixed_wave or self.command is None or self.preset_sent_at is not None
                or not math.isfinite(received) or not self.started_at <= received <= self.clock()):
            self._unknown("preset_send_time_unknown")
            return False
        self.preset_sent_at = received
        if self.wave_qualified:
            self.deadline = received + 12.
        self.joint_excursion = 0.
        return True

    def preset_dispatch(self):
        """Reserve the one IPC write before transport attempts it; never retry."""
        if (not self.wave_qualified or self.wave_dispatch_at is not None or self.terminal
                or self.cancel_requested or not self.standing_ready() or self.query_state != 1
                or self.clock()-self.query_at > .5 or self.joints is None or self.clock()-self.joints_at > .25):
            raise RuntimeError("Wave dispatch preconditions changed")
        self.wave_dispatch_at = self.clock()
        self.baseline = {k: self.joints[k][0] for k in RIGHT_ARM}

    def withdraw_unsent_wave(self):
        """Transport has reaped a helper without ever writing a wave request."""
        if not self.wave_qualified or self.wave_dispatch_at is not None or self.preset_sent_at is not None:
            raise RuntimeError("Cannot withdraw a possibly sent wave")
        if self.status != "UNKNOWN" and self.standing_ready():
            self._emit("CANCELLED", "wave_withdrawn_before_send", stop_confirmed=True,
                       preset_never_sent=True, cancellation_confirmed=False)
        elif self.status != "UNKNOWN":
            self._unknown("unsent_wave_standing_unconfirmed")
        self.terminal = True

    def _request_wave_cancel(self):
        """At most one zero for an acknowledged, measured, still-playing wave.

        A late FINISH waits for natural recovery. Before ACK/PLAYING it remains
        pending; a failed/ambiguous transport must never create a replacement.
        """
        now = self.clock()
        if (not self.wave_qualified or self.terminal or self.probe_cancel_at is not None
                or self.task_id is None or not self.moved or self.preset_sent_at is None
                or not self._wave_feedback_ready(now) or self.player[1] != 2
                or self.query_state != 400 or now-self.query_at > .5):
            return False
        self.probe_cancel_at = now
        self.wave_return_pose = dict(self.wave_first_pose)
        self.wave_return_reference = "official_animation_first_frame"
        self._recovery_since = None
        try:
            self.preset(dict(motion=0, area=2, interrupt=True))
        except Exception:
            self._unknown("preset_cancel_transport_unknown")
        return True

    def request_probe_cancel(self):
        """One simulation-only zero after 0.5 s of observed wave execution."""
        now = self.clock()
        if (self.wave_cancel_probe and not self.terminal and self.status == "STARTED"
                and self.probe_cancel_at is None and self.measured_wave_at is not None
                and now-self.measured_wave_at > .75):
            self._unknown("probe_cancel_window_missed")
            return False
        if (not self.wave_cancel_probe or self.terminal or self.status != "STARTED" or self.motion.failures
                or self.probe_cancel_at is not None or self.measured_wave_at is None
                or now-self.measured_wave_at < .5 or not self._wave_feedback_ready(now)
                or self.player[1] != 2 or self.query_state != 400 or now-self.query_at > .5):
            return False
        self.probe_cancel_at = now  # Unknown transport cannot retry this send.
        self.cancel_requested = True
        try:
            self.preset(dict(motion=0, area=2, interrupt=True))
            self._emit("STARTED", "fixed_simulation_cancel_requested", fact="probe_cancel_requested")
        except Exception:
            self._unknown("preset_cancel_transport_unknown")
            raise
        return True

    def preset_cancel_sent(self, received):
        if (self.probe_cancel_at is None or self.probe_cancel_sent_at is not None
                or not math.isfinite(received) or not self.probe_cancel_at <= received <= self.clock()):
            self._unknown("preset_cancel_send_time_unknown")
            return False
        self.probe_cancel_sent_at = received
        return True

    def preset_cancel_ack(self, code, task_id, state):
        if (not (self.wave_cancel_probe or self.wave_qualified) or self.probe_cancel_sent_at is None or self.probe_cancel_ack is not None
                or (self.status == "UNKNOWN" and not self.wave_qualified) or code != 0 or type(task_id) is not int
                or not 0 <= task_id < 2**64 or state not in (1, 400)):
            self._unknown("preset_cancel_ack_unknown")
            return False
        self.probe_cancel_ack, self.probe_cancel_task_id = self.clock(), task_id
        if not self.wave_qualified:
            self._emit("STARTED", "fixed_simulation_cancel_ack", fact="probe_cancel_ack", task_id=task_id)
        return True

    def _wave_feedback_ready(self, now):
        return self._wave_feedback_failure(now) is None

    def _wave_feedback_failure(self, now):
        m, o = self.motion.mc, self.motion.odom
        if self.joints is None or not 0 <= now-self.joints_at <= .25:
            return "wave_joint_feedback_unavailable"
        if self.player is None or not 0 <= now-self.player[0] <= .5 or m is None:
            return "wave_mc_feedback_unavailable"
        if o is None or not 0 <= now-o.received <= .25:
            return "wave_odom_unavailable"
        if o.tilt > .5:
            return "wave_not_upright"
        if m.body_status != 1 or m.mode != "STAND_DEFAULT":
            return "wave_stand_mode_lost"
        if self.fixed_wave:
            if self.motion.failures and not self.wave_qualified:
                return self.motion.failures[0]
            if m.source not in ("", SOURCE) or m.action_status != 100:
                return "wave_stand_ownership_lost"
            if self.distance > WAVE_PROBE_ACTIVE_TRANSLATION_M:
                return "wave_translation_limit"
            if abs(self.signed_angle) > .03:
                return "wave_yaw_limit"
        return None

    def standing_ready(self):
        now = self.clock()
        m, o = self.motion.mc, self.motion.odom
        if (not m or not o or not 0 <= now-m.received <= .5 or not 0 <= now-o.received <= .25
                or m.source not in ("", SOURCE) or m.mode != "STAND_DEFAULT" or m.body_status != 1 or m.action_status != 100
                or o.tilt > .5 or math.hypot(*o.linear[:2]) >= .02 or abs(o.angular[2]) >= .03
                or self.player is None or self.player[1] != 0):
            self._idle_since = self._idle_pose = None
            return False
        pair = (m.sequence, o.sequence)
        if any(new <= old for new, old in zip(pair, self._idle_pair)):
            return self._idle_since is not None and min(m.received, o.received)-self._idle_since >= 1.
        stamp = min(m.received, o.received)
        pose = (o.position, yaw(o.quaternion))
        if self._idle_pose and (math.dist(pose[0][:2], self._idle_pose[0][:2]) > .02 or abs(angle_delta(pose[1], self._idle_pose[1])) > .03):
            self._idle_since = None
        if self._idle_since is None:
            self._idle_since, self._idle_pose = stamp, pose
        self._idle_pair = pair
        return stamp-self._idle_since >= 1.

    def submit(self, command):
        now = self.clock()
        if (not isinstance(command, SkillCommand) or type(command.decision_seq) is not int
                or command.decision_seq <= self.last_seq or not command.request_id
                or command.behavior not in BEHAVIORS or command.phase not in ("START", "CONTINUE", "FINISH")
                or not math.isfinite(command.intensity) or not 0 <= command.intensity <= 1
                or not math.isfinite(command.observed_at) or not 0 <= now-command.observed_at <= self.max_decision_age):
            return False
        self.last_seq = command.decision_seq
        if command.behavior == "HOLD" and command.phase != "FINISH":
            return False
        if self.motion.failures or self.status in ("UNKNOWN", "FAILED"):
            return False
        if self.command is not None and not self.terminal:
            if command.phase == "FINISH" and command.behavior in (self.command.behavior, "HOLD"):
                self.finish_command = command
                self.cancel("brain_finish", protection=False)
                return True
            return command.phase == "CONTINUE" and command.behavior == self.command.behavior and command.request_id == self.command.request_id
        if command.behavior != "HOLD" and command.phase != "START":
            return False
        if command.behavior not in self.enabled or self.motion.failures or not self.standing_ready():
            return False
        if command.behavior == "WAVE_RIGHT":
            if (not (self.wave_probe or self.wave_qualified) or self.query_state != 1 or now-self.query_at > .5
                    or self.joints is None or now-self.joints_at > .25
                    or (self.wave_contract_probe and self.command is not None)):
                return False
        self.command, self.started_at = command, now
        self.finish_command = None
        self.stop_at = None
        self.execution_id = "skill-" + str(command.decision_seq)
        self.terminal, self.cancel_requested, self.moved = False, False, False
        self.start_position = self.motion.odom.position
        self.start_yaw = self.last_yaw = yaw(self.motion.odom.quaternion)
        self.distance = self.signed_angle = 0.
        self.forward_distance = self.lateral_distance = 0.
        # Latch once for the admitted START; CONTINUE never reaches this block.
        self.active_command_seconds = self.command_seconds_by_behavior.get(command.behavior)
        self.resolved_velocity = (self.motion.resolve_velocity(command.behavior, command.intensity)
                                  if command.behavior in CONTINUOUS_BEHAVIORS else ZERO)
        self.deadline = now + (self.active_command_seconds if self.active_command_seconds is not None
                               else (30. if self.full_action else 24. if self.wave_qualified else 12.) if command.behavior == "WAVE_RIGHT" else 0.)
        if command.behavior == "WAVE_RIGHT" and self.wave_qualified:
            self.wave_dispatch_at = self.preset_sent_at = self.measured_wave_at = None
            self.probe_cancel_at = self.probe_cancel_sent_at = self.probe_cancel_ack = self.probe_cancel_task_id = None
            self.saw_cancel_interrupting = self.saw_cancel_idle = self.wave_return_confirmed = False
            self.joint_return = self.joint_baseline_return = math.inf
            self.wave_peak_distance = self.wave_peak_yaw = self.wave_peak_tilt = 0.
            self.wave_return_pose = dict(self.wave_terminal_pose)
            self.wave_return_reference = "official_animation_terminal_frame"
        self._emit("ACCEPTED", "skill_admitted", fact="skill_admitted")
        if command.behavior == "HOLD":
            self._emit("STARTED", "standing_observed")
            self._emit("COMPLETED", "standing_confirmed", stop_confirmed=True)
            self.terminal = True
        elif command.behavior == "WAVE_RIGHT":
            self.baseline = {k: self.joints[k][0] for k in RIGHT_ARM}
            self.joint_excursion = 0.
            self.task_id = None
            self.saw_query_running = self.saw_player_playing = False
            self._recovery_since = None
            try:
                self.preset(dict(motion=1002, area=2, interrupt=False))
            except Exception:
                self._unknown("preset_transport_error")
                raise
        else:
            if not self.motion.request_velocity(command.behavior, command.intensity, self.active_command_seconds):
                self._emit("FAILED", self.motion.reason)
                return False
        return True

    def submit_decision(self, decision, observation_received):
        """Translate an already admitted Decision without choosing its behavior."""
        return self.submit(SkillCommand(decision.decision_seq, decision.request_id,
            str(decision.behavior), str(decision.phase), decision.intensity, observation_received))

    def _unknown(self, reason):
        if reason not in self.motion.failures:
            self.motion.request_stop(reason, failure=True)
        if self.stop_at is None:
            self.stop_at = self.clock()
        if self.status != "UNKNOWN":
            self._emit("UNKNOWN", reason, stop_confirmed=False, protection_reason=reason)

    def _advance_direction_ok(self):
        # Positive, predominantly forward displacement in the initial body frame.
        return self.forward_distance > .02 and self.forward_distance > abs(self.lateral_distance)

    def _displacement_evidence(self):
        return dict(distance_m=self.distance, forward_distance_m=self.forward_distance,
                    lateral_distance_m=self.lateral_distance, start_position=self.start_position,
                    start_yaw_rad=self.start_yaw, position=self.motion.odom.position,
                    advance_direction_ok=self._advance_direction_ok(), angle_rad=self.signed_angle)

    def cancel(self, reason="user_emergency_stop", *, protection=True):
        self.cancel_requested = True
        self.motion.request_stop(reason, failure=protection)
        self.stop_at = self.clock()
        if self.command and self.command.behavior == "WAVE_RIGHT":
            if self.wave_qualified:
                if not protection and self.status != "UNKNOWN":
                    self._emit(self.status, "brain_finish_requested", fact="finish_requested")
                self._request_wave_cancel()
                if protection:
                    self._unknown(reason)
            elif protection:
                self._unknown("wave_cancel_unverified:" + reason)
            elif self.status != "UNKNOWN":
                self._emit(self.status, "finish_pending_natural_completion", fact="finish_requested")
        elif self.command:
            if protection:
                self._unknown(reason)
            else:
                self._emit(self.status, reason, fact="finish_requested")

    def tick(self):
        now = self.clock()
        c = self.command
        if c and not self.terminal:
            if c.behavior in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT"):
                if (self.full_action and c.behavior in ('TURN_LEFT', 'TURN_RIGHT') and self.stop_at is None
                        and self.signed_angle * (1 if c.behavior == 'TURN_LEFT' else -1) >= .35):
                    self.motion.request_stop('fixed_turn_measured_angle_limit')
                    self.stop_at = now
                if self.player is None or not 0 <= now-self.player[0] <= .5 or self.player[1] != 0:
                    self._unknown("external_preset_during_velocity" if self.player and self.player[1] != 0 else "preset_state_unavailable")
                if now >= self.deadline and self.stop_at is None:
                    self.motion.request_stop("bounded_skill_complete")
                    self.stop_at = now
                if self.motion.failures and self.stop_at is None:
                    self.stop_at = now
            elif c.behavior == "WAVE_RIGHT":
                if self.wave_qualified and self.preset_sent_at is None:
                    if now >= self.started_at+(18. if self.full_action else 12.):
                        self._unknown("wave_preparation_timeout")
                        self.terminal = True
                    return self.motion.tick()
                feedback_failure = self._wave_feedback_failure(now)
                fresh = feedback_failure is None
                if not fresh:
                    self._unknown(feedback_failure)
                elif self.saw_player_playing and self.saw_query_running and self.joint_excursion >= .1 and not self.moved:
                    self.moved = True
                    self.measured_wave_at = now
                    if self.status != "UNKNOWN":
                        self._emit("STARTED", "measured_joint_motion")
                if self.wave_qualified and self.cancel_requested:
                    self._request_wave_cancel()
                recovering = (fresh and self.moved and self.query_state == 1 and now-self.query_at <= .5
                    and self.player[1] == 0 and self.joint_return <= .15
                    and max(abs(self.joints[k][1]) for k in RIGHT_ARM) < .08 and self.standing_ready())
                if self.fixed_wave:
                    recovering = recovering and self.distance <= .02 and abs(self.signed_angle) <= .03
                cancelled = self.wave_cancel_probe or (self.wave_qualified and self.probe_cancel_at is not None)
                if cancelled:
                    recovering = (recovering and self.probe_cancel_ack is not None
                        and self.saw_cancel_interrupting and self.saw_cancel_idle)
                if recovering:
                    if self._recovery_since is None:
                        self._recovery_since = now
                    if now-self._recovery_since >= 1.:
                        if self.status != "UNKNOWN":
                            self._emit("CANCELLED" if cancelled else "COMPLETED",
                                       "simulation_cancel_returned" if cancelled else "preset_finished_and_returned",
                                       stop_confirmed=True, joint_return_rad=self.joint_return,
                                       return_reference=self.wave_return_reference,
                                       cancellation_confirmed=cancelled)
                        self.wave_return_confirmed = True
                        self.terminal = True
                else:
                    self._recovery_since = None
                if now >= self.deadline and not self.terminal:
                    self._unknown("wave_deadline_unconfirmed")
                    self.terminal = True
        result = self.motion.tick()
        if c and not self.terminal and c.behavior in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT"):
            if self.stop_at is not None and self.motion.state == "STOPPED":
                sign_ok = self._advance_direction_ok() if c.behavior == "ADVANCE" else self.signed_angle*(1 if c.behavior == "TURN_LEFT" else -1) > .02
                status = "UNKNOWN" if self.status == "UNKNOWN" else "FAILED" if self.motion.failures or not self.moved or not sign_ok else "CANCELLED" if self.cancel_requested else "COMPLETED"
                self._emit(status, "base_stopped_protection_latched" if status == "UNKNOWN" else "stopped_with_feedback",
                           stop_confirmed=status != "UNKNOWN", base_stop_confirmed=True, **self._displacement_evidence())
                self.terminal = True
            elif self.stop_at is not None and now-self.stop_at >= 12.:
                self._emit("UNKNOWN", "stop_timeout", stop_confirmed=False)
                self.terminal = True
        return result

    def summary(self):
        return dict(status=self.status, reason=self.reason, terminal=self.terminal,
                    behavior=self.command.behavior if self.command else None,
                    skill_profile_id=self.skill_profile_id,
                    command_seconds_by_behavior=dict(self.command_seconds_by_behavior),
                    command_seconds=self.active_command_seconds,
                    intensity=self.command.intensity if self.command else None,
                    started_at=self.started_at, deadline=self.deadline,
                    execution_id=self.execution_id, measured_motion=self.moved, distance_m=self.distance,
                    forward_distance_m=self.forward_distance, lateral_distance_m=self.lateral_distance,
                    advance_direction_ok=self._advance_direction_ok(), start_position=self.start_position,
                    start_yaw_rad=self.start_yaw,
                    angle_rad=self.signed_angle, motion_state=self.motion.state,
                    failures=list(self.motion.failures), task_id=self.task_id,
                    query_running_seen=self.saw_query_running, player_playing_seen=self.saw_player_playing,
                    joint_excursion_rad=self.joint_excursion,
                    joint_return_rad=self.joint_return if math.isfinite(self.joint_return) else None,
                    joint_baseline_return_rad=self.joint_baseline_return if math.isfinite(self.joint_baseline_return) else None,
                    wave_return_reference=self.wave_return_reference, wave_return_pose=self.wave_return_pose,
                    wave_peak_distance_m=self.wave_peak_distance, wave_peak_yaw_rad=self.wave_peak_yaw,
                    wave_peak_tilt_rad=self.wave_peak_tilt,
                    wave_active_translation_limit_m=WAVE_PROBE_ACTIVE_TRANSLATION_M if self.fixed_wave else None,
                    wave_qualified_simulation=self.wave_qualified, wave_return_confirmed=self.wave_return_confirmed,
                    wave_contract_probe=self.wave_contract_probe, wave_cancel_probe=self.wave_cancel_probe,
                    attribution=("exclusive_simulation_single_wave; post_send_motion_name_area_type_and_joints; global_query_not_task_identity"
                                 if self.fixed_wave else "task_and_motion_feedback"),
                    preset_sent_at=self.preset_sent_at, measured_wave_at=self.measured_wave_at,
                    probe_cancel_at=self.probe_cancel_at, probe_cancel_sent_at=self.probe_cancel_sent_at,
                    probe_cancel_ack_at=self.probe_cancel_ack,
                    probe_cancel_task_id=self.probe_cancel_task_id,
                    cancel_interrupting_seen=self.saw_cancel_interrupting, cancel_idle_seen=self.saw_cancel_idle,
                    cancellation_trial_pass=self.wave_cancel_probe and self.status == "CANCELLED" and self.terminal,
                    wave_cancellation_verified=False, events=self.events, **self._execution_details())

    def _execution_details(self):
        if not self.fixed_advance:
            return {}
        return dict(model_intensity=self.command.intensity if self.command else None,
                    resolved_velocity=self.resolved_velocity,
                    velocity_resolution='fixed_advance_0.2_m_s' if self.command and self.command.behavior == 'ADVANCE'
                        else 'fixed_turn_0.4_rad_s_bounded_2s' if self.full_action and self.command and self.command.behavior in ('TURN_LEFT', 'TURN_RIGHT')
                        else 'legacy_behavior_mapping')
