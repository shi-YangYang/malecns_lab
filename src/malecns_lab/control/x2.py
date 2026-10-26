"""Bounded STOP/FORWARD intent adapter; no ROS or device IO on import.

Call ``tick`` regularly from the transport loop. Receipt timestamps must use the
same monotonic clock as this adapter; ROS header clocks are only log metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Optional

SOURCE = "malecns.x2_mvp"
ZERO = (0.0, 0.0, 0.0)
FORWARD = (0.2, 0.0, 0.0)
STARTUP_TRANSITION_TIMEOUT = 0.5


@dataclass(frozen=True)
class Odom:
    received: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear: tuple[float, float, float]
    angular: tuple[float, float, float]
    sequence: int

    @property
    def tilt(self) -> float:
        x, y, z, w = self.quaternion
        norm2 = x*x + y*y + z*z + w*w
        return math.acos(max(-1.0, min(1.0, 1.0 - 2.0*(x*x+y*y)/norm2)))


@dataclass(frozen=True)
class McState:
    received: float
    source: str
    mode: str
    body_status: int
    action_status: int
    sequence: int


class X2MotionAdapter:
    """Explicitly armed intent state, with a separately clocked publisher.

    Thresholds are local simulation monitoring defaults, not hardware safety
    guarantees. A rejected intent is cleared, never queued for later recovery.
    """

    def __init__(self, publish: Callable[[tuple[float, float, float]], None], *,
                 clock: Callable[[], float] = time.monotonic,
                 odom_max_age: float = 0.25, mc_max_age: float = 0.5,
                 allow_empty_source_claim: bool = False, max_segment_seconds: float = 5.):
        for value in (odom_max_age, mc_max_age):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("feedback age limits must be finite and positive")
        self.publish, self.clock = publish, clock
        if not math.isfinite(max_segment_seconds) or not 0 < max_segment_seconds <= 10.:
            raise ValueError('Motion segment budget must be within (0, 10] seconds')
        self.max_segment_seconds = max_segment_seconds
        self.odom_max_age, self.mc_max_age = odom_max_age, mc_max_age
        self.armed = False
        self.state, self.reason = "DISARMED", "not_armed"
        self.failures: list[str] = []
        self.odom: Optional[Odom] = None
        self.mc: Optional[McState] = None
        self._odom_sequence = self._mc_sequence = 0
        self._odom_last_time = self._mc_last_time = -math.inf
        self._last_clock = -math.inf
        self._segment_start: Optional[float] = None
        self._expires = -math.inf
        self._allow_empty_source_claim = allow_empty_source_claim
        self._source_registered = False
        self._claim_used = False
        self._claim_deadline: Optional[float] = None
        self._claim_request_time: Optional[float] = None
        self._claim_request_sequence = 0
        self._first_nonzero_published_at: Optional[float] = None
        self._startup_publish_deadline: Optional[float] = None
        self._startup_transition_deadline: Optional[float] = None
        self._startup_request_sequence = 0
        self._startup_transition_pending = False
        self.ownership_confirmed_at: Optional[float] = None
        self.stop_requested: Optional[float] = None
        self._stop_sequences = (0, 0)
        self._window_start: Optional[float] = None
        self._window_pair = (0, 0)
        self._window_position: Optional[tuple[float, float, float]] = None
        self._window_path = 0.0
        self._window_time: Optional[float] = None
        self._window_reset_time = -math.inf
        self._window_reset_sequences = (0, 0)

    def _reset_window(self) -> None:
        if self.state == "STOPPED":
            self.state = "STOPPING"
        self._window_start = self._window_time = None
        self._window_position = None
        self._window_pair = (0, 0)
        self._window_path = 0.0
        # A slower topic must not backdate the next window across this reset.
        # Both streams need a strictly later receipt and a new sample.
        self._window_reset_time = self._last_clock
        self._window_reset_sequences = (self._odom_sequence, self._mc_sequence)

    def _now(self) -> Optional[float]:
        try:
            now = float(self.clock())
        except (TypeError, ValueError, OverflowError):
            now = math.nan
        if not math.isfinite(now) or now < self._last_clock:
            self._fail("invalid_clock", self._last_clock)
            return None
        self._last_clock = now
        return now

    def _begin_stop(self, now: float, reason: str) -> None:
        self._expires = -math.inf
        self._startup_transition_pending = False
        self.reason = reason
        if self.armed:
            self.state = "STOPPING"
        self.stop_requested = now
        self._stop_sequences = (self._odom_sequence, self._mc_sequence)
        self._reset_window()

    def _fail(self, reason: str, now: float) -> None:
        if reason not in self.failures:
            self.failures.append(reason)
        if self.state in ("CLAIMING", "FORWARD") or self.stop_requested is None:
            self._begin_stop(now, reason)
        else:
            self.reason = reason
            if self.armed:
                self.state = "STOPPING"
            self._reset_window()

    def arm(self) -> None:
        now = self._now()
        if now is None:
            return
        self.armed = True
        self._begin_stop(now, "armed_zero")

    def confirm_source_registration(self) -> None:
        """Transport calls this only after successful registration of SOURCE.

        Registration alone does not permit motion: the optional one-time empty
        source claim still needs an explicit forward request and all feedback.
        """
        self._source_registered = True

    def request_stop(self, reason: str = "requested_stop", *, failure: bool = False) -> None:
        now = self._now()
        if now is None:
            return
        if failure and reason not in self.failures:
            self.failures.append(reason)
        self._begin_stop(now, reason)

    def _stamp(self, received: Optional[float], last: float) -> Optional[float]:
        now = self._now()
        if now is None:
            return None
        try:
            stamp = now if received is None else float(received)
        except (ValueError, TypeError, OverflowError):
            return None
        return stamp if math.isfinite(stamp) and last < stamp <= now else None

    def update_odom(self, position, quaternion, linear, angular, *, received=None) -> bool:
        stamp = self._stamp(received, self._odom_last_time)
        try:
            values = tuple(tuple(float(v) for v in group)
                           for group in (position, quaternion, linear, angular))
            valid = tuple(map(len, values)) == (3, 4, 3, 3)
            valid = valid and all(math.isfinite(v) for group in values for v in group)
            valid = valid and 0.9 <= math.sqrt(sum(v*v for v in values[1])) <= 1.1
        except (ValueError, TypeError, OverflowError):
            valid = False
        if stamp is None or not valid:
            self.odom = None
            self._fail("invalid_odom", self._last_clock)
            return False
        if stamp - self._odom_last_time > self.odom_max_age:
            if self.state in ("CLAIMING", "FORWARD"):
                self._fail("odom_gap", stamp)
            self._reset_window()
        self._odom_last_time = stamp
        self._odom_sequence += 1
        self.odom = Odom(stamp, *values, self._odom_sequence)
        if self.odom.tilt > 0.5 and self.state in ("CLAIMING", "FORWARD"):
            self._fail("not_upright", stamp)
        if self.odom.tilt > 0.5 or math.hypot(*self.odom.linear[:2]) >= 0.02:
            self._reset_window()
            if self.state == "STOPPED":
                self.state = "STOPPING"
        return True

    def update_mc(self, source: str, mode: str, body_status: int,
                  action_status: int, *, received=None) -> bool:
        stamp = self._stamp(received, self._mc_last_time)
        if (stamp is None or not isinstance(source, str) or not isinstance(mode, str)
                or type(body_status) is not int or type(action_status) is not int):
            self.mc = None
            self._fail("invalid_mc", self._last_clock)
            return False
        if stamp - self._mc_last_time > self.mc_max_age:
            if self.state in ("CLAIMING", "FORWARD"):
                self._fail("mc_gap", stamp)
            self._reset_window()
        # A late RUNNING sample must not hide an unresolved, expired transition.
        if (self.state in ("CLAIMING", "FORWARD") and self._startup_transition_pending
                and self._startup_transition_deadline is not None
                and self._last_clock >= self._startup_transition_deadline):
            self._fail("startup_transition_timeout", self._last_clock)
        self._mc_last_time = stamp
        self._mc_sequence += 1
        self.mc = McState(stamp, source, mode, body_status, action_status, self._mc_sequence)
        if self.state in ("CLAIMING", "FORWARD"):
            # Established motion checks TTL at the publishing tick, after the
            # runner can apply its planned stop at that same deadline.
            reason = (self._active_reason(self._last_clock) if self.state == "CLAIMING"
                      else self.ready_reason(self._last_clock))
            if reason:
                self._fail(reason, self._last_clock)
            else:
                self._startup_transition_pending = action_status == 200
                if (self.state == "CLAIMING" and source == SOURCE
                        and stamp > self._claim_request_time
                        and self._mc_sequence > self._claim_request_sequence):
                    self.state, self.reason = "FORWARD", "ownership_confirmed"
                    self.ownership_confirmed_at = stamp
        if source != SOURCE or mode != "STAND_DEFAULT" or body_status != 1 or action_status != 100:
            self._reset_window()
            if self.state == "STOPPED":
                self.state = "STOPPING"
        return True

    def _startup_transition_allowed(self, now: float) -> bool:
        """Only a new owned sample inside the first successful publish's window."""
        return (self.state in ("CLAIMING", "FORWARD")
                and self._startup_transition_deadline is not None
                and now < self._startup_transition_deadline
                and self.mc.source == SOURCE
                and self.mc.received > self._segment_start
                and self.mc.sequence > self._startup_request_sequence)

    def ready_reason(self, now: Optional[float] = None, *,
                     allow_empty_source: bool = False) -> Optional[str]:
        if now is None:
            now = self._now()
        if now is None:
            return "invalid_clock"
        if not self.armed:
            return "not_armed"
        if self.odom is None or not 0 <= now-self.odom.received <= self.odom_max_age:
            return "odom_unavailable"
        if self.mc is None or not 0 <= now-self.mc.received <= self.mc_max_age:
            return "mc_unavailable"
        if self.mc.source != SOURCE:
            if (self.mc.source != "" or not allow_empty_source
                    or not self._allow_empty_source_claim or not self._source_registered):
                return "ownership_lost"
            if self._claim_used and self.state != "CLAIMING":
                return "claim_already_used"
            if self.mc.mode != "STAND_DEFAULT":
                return "mc_not_ready"
        if self.mc.body_status != 1 or self.odom.tilt > 0.5:
            return "not_upright"
        if (self.mc.mode not in ("STAND_DEFAULT", "LOCOMOTION_DEFAULT")
                or (self.mc.action_status != 100
                    and not (self.mc.action_status == 200 and self._startup_transition_allowed(now)))):
            return "mc_not_ready"
        return None

    @property
    def intent_deadline(self) -> Optional[float]:
        return self._expires if self.state in ("CLAIMING", "FORWARD") else None

    def _active_reason(self, now: float) -> Optional[str]:
        if now >= self._expires:
            return "intent_expired"
        if self.state == "CLAIMING" and now >= self._claim_deadline:
            return "claim_timeout"
        if (self._first_nonzero_published_at is None and self._startup_publish_deadline is not None
                and now >= self._startup_publish_deadline):
            return "startup_publish_timeout"
        if (self._startup_transition_pending and self._startup_transition_deadline is not None
                and now >= self._startup_transition_deadline):
            return "startup_transition_timeout"
        return self.ready_reason(now, allow_empty_source=self.state == "CLAIMING")

    def request_forward(self, ttl: float) -> bool:
        now = self._now()
        if now is None:
            return False
        try:
            valid = math.isfinite(ttl) and 0 < ttl <= self.max_segment_seconds
        except TypeError:
            valid = False
        active = self.state in ("CLAIMING", "FORWARD")
        reason = self._active_reason(now) if active else self.ready_reason(now, allow_empty_source=True)
        if not valid:
            reason = "invalid_intent_ttl"
        elif self._segment_start is not None and (not active or now >= self._segment_start+self.max_segment_seconds):
            reason = "stop_required_before_restart"
        if reason:
            self._fail(reason, now)
            return False
        if not active:
            self._segment_start = now
            self._first_nonzero_published_at = None
            self._startup_publish_deadline = now + STARTUP_TRANSITION_TIMEOUT
            self._startup_transition_deadline = None
            self._startup_request_sequence = self._mc_sequence
            self._startup_transition_pending = False
            if self.mc.source == "":
                self._claim_used = True
                self._claim_deadline = now + 0.5
                self._claim_request_time = now
                self._claim_request_sequence = self._mc_sequence
                self.state, self.reason = "CLAIMING", "claiming_empty_source"
            else:
                self.state, self.reason = "FORWARD", "forward_intent"
        self._expires = min(now+ttl, self._segment_start+self.max_segment_seconds)
        self._reset_window()
        return True

    def request_body(self, target, ttl: float = 1.0) -> bool:
        """Accept exactly STOP or the fixed FLU target; reject every other axis."""
        try:
            values = tuple(float(getattr(target, axis)) for axis in ("vx", "vy", "vz", "wx", "wy", "wz"))
        except (AttributeError, TypeError, ValueError, OverflowError):
            values = ()
        if values == (0.0,)*6:
            self.request_stop()
            return True
        if values == (0.2, 0.0, 0.0, 0.0, 0.0, 0.0):
            return self.request_forward(ttl)
        self.request_stop("unsupported_body_target", failure=True)
        return False

    def _confirm_stop(self, now: float) -> None:
        if (self.ready_reason(now) is not None or self.mc.mode != "STAND_DEFAULT"
                or math.hypot(*self.odom.linear[:2]) >= 0.02):
            self._reset_window()
            if self.state == "STOPPED":
                self.state = "STOPPING"
            return
        odom, mc = self.odom, self.mc
        pair = (odom.sequence, mc.sequence)
        if self.stop_requested is None or min(odom.received, mc.received) <= self.stop_requested:
            return
        if any(new <= old for new, old in zip(pair, self._stop_sequences)):
            return
        if (min(odom.received, mc.received) <= self._window_reset_time
                or any(new <= old for new, old in zip(pair, self._window_reset_sequences))):
            return
        if any(new <= old for new, old in zip(pair, self._window_pair)):
            return
        evidence_time = min(odom.received, mc.received)
        if self._window_position is not None:
            distance = math.dist(odom.position[:2], self._window_position[:2])
            dt = evidence_time - self._window_time
            self._window_path += distance
            duration = evidence_time - self._window_start
            if dt <= 0 or distance > 0.02*dt + 0.005 or self._window_path > 0.02*duration + 0.005:
                self._reset_window()
                return
        if self._window_start is None:
            self._window_start = evidence_time
        self._window_pair = pair
        self._window_position, self._window_time = odom.position, evidence_time
        if evidence_time-self._window_start >= 1.0:
            self.state = "STOPPED"
            self._segment_start = None

    def _publish_effective(self, command):
        """Return what was sent, after the publisher succeeds.

        Adapters that transform commands must return the final effective value.
        The ordinary publisher callback keeps its existing no-return contract.
        """
        self.publish(command)
        return command

    def tick(self) -> dict:
        now = self._now()
        attempts = []
        if self.armed:
            if self.state in ("CLAIMING", "FORWARD"):
                reason = self._active_reason(now) if now is not None else "invalid_clock"
                if reason:
                    self._fail(reason, self._last_clock)
            command = FORWARD if self.state in ("CLAIMING", "FORWARD") else ZERO
            try:
                effective = self._publish_effective(command)
                attempts.append({"command": effective, "ok": True})
                if (effective != ZERO and self.state in ("CLAIMING", "FORWARD")
                        and self._first_nonzero_published_at is None):
                    published_at = self._now()
                    if published_at is not None:
                        self._first_nonzero_published_at = published_at
                        self._startup_transition_deadline = published_at + STARTUP_TRANSITION_TIMEOUT
                    else:
                        self._publish_effective(ZERO)
                        attempts.append({"command": ZERO, "ok": True})
            except Exception as exc:
                attempts.append({"command": command, "ok": False, "error": repr(exc)})
                self._fail("publish_error", self._last_clock)
                if command != ZERO:
                    try:
                        self._publish_effective(ZERO)
                        attempts.append({"command": ZERO, "ok": True})
                    except Exception as zero_exc:
                        attempts.append({"command": ZERO, "ok": False, "error": repr(zero_exc)})
            if now is not None and self.state not in ("CLAIMING", "FORWARD"):
                self._confirm_stop(now)
        return {"monotonic": now, "state": self.state, "reason": self.reason,
                "attempts": attempts, "failures": list(self.failures),
                "stop_requested": self.stop_requested,
                "stop_window_start": self._window_start,
                "odom_age": None if now is None or self.odom is None else now-self.odom.received,
                "mc_age": None if now is None or self.mc is None else now-self.mc.received,
                "claim_deadline": self._claim_deadline,
                "first_nonzero_published_at": self._first_nonzero_published_at,
                "startup_publish_deadline": self._startup_publish_deadline,
                "startup_transition_deadline": self._startup_transition_deadline,
                "startup_transition_pending": self._startup_transition_pending,
                "ownership_confirmed_at": self.ownership_confirmed_at,
                "intent_expires": self._expires if math.isfinite(self._expires) else None}
