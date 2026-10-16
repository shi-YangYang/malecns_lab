"""Pure RC limits and wall-clock admission policies for bounded trials."""

from __future__ import annotations

import math

PERIOD = 0.05
INPUT_TIMEOUT = 0.5
RESULT_TIMEOUT = 0.35
MAX_YAW_RC = 10
MAX_THROTTLE_RC = 10
MAX_FORWARD_RC = 15
MAX_LATERAL_RC = 10
MAX_YAW_DEGREES = 25
MIN_TAKEOFF_BATTERY = 40
CRUISE_VALUE = MAX_FORWARD_RC / 60.0
CRUISE_START = 2.0
CRUISE_ARM_DEADLINE = 2.5
CRUISE_END = 5.0
CRUISE_TRIAL_SECONDS = 7.0
FULL_START = 2.0
FULL_END = 10.0
FULL_TRIAL_SECONDS = 12.0
CRUISE_YAW_PEAK = 0.15
CRUISE_YAW_MEAN = 0.1
FORWARD_CHECK_RC = 15
FORWARD_CHECK_END = 3.0
MODES = ("yaw", "reflex", "cruise", "forward-check", "full")



def validate_duration(seconds, mode):
    if mode not in MODES:
        raise ValueError("Unknown trial mode")
    if mode == "full":
        if seconds != FULL_TRIAL_SECONDS:
            raise ValueError("Full trial requires exactly 12 seconds (2 s settle, up to 8 s control, 2 s zero RC)")
    elif mode == "cruise":
        if seconds != CRUISE_TRIAL_SECONDS:
            raise ValueError(f"{mode} trial requires exactly 7 seconds (2 s settle, up to 3 s control, 2 s zero RC)")
    elif mode == "forward-check":
        if seconds != 5:
            raise ValueError("Forward check requires exactly 5 seconds (2 s zero RC, 1 s forward, 2 s zero RC)")
    elif not math.isfinite(seconds) or not 1 <= seconds <= 5:
        raise ValueError("Trial duration must be between 1 and 5 seconds")


def yaw_rc(proposal: float) -> int:
    if not math.isfinite(proposal):
        raise RuntimeError("Non-finite yaw proposal")
    return max(-MAX_YAW_RC, min(MAX_YAW_RC, int(proposal * 60)))


def flight_rc(row, height_cm, mode):
    """Apply final limits using the controller's latest height, not worker data."""
    if mode not in MODES:
        raise ValueError("Unknown trial mode")
    if mode == "forward-check":
        # Explicit hardware probe, independent of neural proposals. Its fixed
        # time window is enforced by ForwardCheckGate in the parent process.
        return (0, FORWARD_CHECK_RC, 0, 0)
    yaw = yaw_rc(float(row["cmd_yaw"]))
    throttle = forward = lateral = 0
    if mode in ("reflex", "cruise", "full"):
        proposal = float(row["cmd_throttle"])
        if not math.isfinite(proposal):
            raise RuntimeError("Non-finite throttle proposal")
        if row.get("land_requested", False):
            raise RuntimeError("Brain safety requested landing")
        throttle = max(-MAX_THROTTLE_RC, min(MAX_THROTTLE_RC, int(proposal * 60)))
        # Leave room inside the trial's 30-150 cm abort range. These suppress
        # outward commands; they do not guarantee a physical altitude bound.
        if (height_cm <= 50 and throttle < 0) or (height_cm >= 120 and throttle > 0):
            throttle = 0
    if mode in ("cruise", "full"):
        proposal = float(row["cmd_forward"])
        if not math.isfinite(proposal):
            raise RuntimeError("Non-finite forward proposal")
        # Decoder smoothing approaches 0.25 from below. Round to the nearest
        # RC unit so the verified RC15 is reachable; never floor braking at 15.
        forward = max(0, min(MAX_FORWARD_RC, int(round(proposal * 60))))
        if row.get("escape", False):
            forward = 0
    if mode == "full":
        proposal = float(row["cmd_lateral"])
        if not math.isfinite(proposal):
            raise RuntimeError("Non-finite lateral proposal")
        lateral = max(-MAX_LATERAL_RC, min(MAX_LATERAL_RC, int(proposal * 60)))
    return (lateral, forward, throttle, yaw)


class CruiseGate:
    """Fixed wall-clock phases; no delay can extend the forward window."""

    stop_after = CRUISE_END

    def __init__(self, acknowledged_at):
        self.started = acknowledged_at
        self.recent = []
        self.armed = False
        self.phase = "settling"
        self.stop_reason = ""

    def observe(self, row, frame_at):
        # Called once per non-priming result, never for repeated RC samples.
        self.recent.append((frame_at, abs(float(row["cmd_yaw"])),
                            abs(float(row["cmd_throttle"])), bool(row.get("escape", False))))
        self.recent = [sample for sample in self.recent if sample[0] >= frame_at - 0.6]
        if self.armed and row.get("escape", False):
            self.stop_reason = "escape: forward disabled for remainder of trial"

    def apply(self, axes, frame_at, now):
        elapsed = now - self.started
        if elapsed >= CRUISE_END:
            self.phase = "stopping"
            return (0, 0, 0, 0)
        if elapsed < CRUISE_START:
            self.phase = "settling"
            return (0, 0, 0, 0)
        if not self.armed:
            self.phase = "waiting"
            span = self.recent[-1][0] - self.recent[0][0] if self.recent else 0.0
            yaw_peak = max((sample[1] for sample in self.recent), default=0.0)
            yaw_mean = sum(sample[1] for sample in self.recent) / len(self.recent) if self.recent else 0.0
            throttle_peak = max((sample[2] for sample in self.recent), default=0.0)
            escaped = any(sample[3] for sample in self.recent)
            # Real hover logs contain isolated +/-0.125 yaw proposals. Admit
            # those only when the average remains quiet; keep vertical and
            # escape criteria strict and reject sustained or larger yaw.
            quiet = (
                len(self.recent) >= 5
                and span >= 0.4
                and yaw_peak <= CRUISE_YAW_PEAK
                and yaw_mean <= CRUISE_YAW_MEAN
                and throttle_peak <= 0.1
                and not escaped
            )
            if (not self.stop_reason and elapsed < CRUISE_ARM_DEADLINE and quiet
                    and frame_at is not None and frame_at >= self.started + CRUISE_START):
                self.armed = True
            else:
                if elapsed >= CRUISE_ARM_DEADLINE and not self.stop_reason:
                    self.stop_reason = (
                        "airborne output did not settle: forward never enabled "
                        f"(samples={len(self.recent)}, span={span:.2f}s, "
                        f"yaw_peak={yaw_peak:.3f}, yaw_mean={yaw_mean:.3f}, "
                        f"throttle_peak={throttle_peak:.3f}, escape={escaped})"
                    )
                if self.stop_reason:
                    self.phase = "blocked"
                return (0, 0, 0, 0)
        if self.stop_reason:
            self.phase = "braked"
            return (0, 0, axes[2], axes[3])
        self.phase = "cruise"
        return axes


class ForwardCheckGate:
    """A fixed forward-only pulse, with no neural quiet/escape admission gate."""

    stop_after = FORWARD_CHECK_END

    def __init__(self, acknowledged_at):
        self.started = acknowledged_at
        self.armed = False
        self.phase = "settling"
        self.stop_reason = ""

    def observe(self, row, frame_at):
        # Proposals are logged but cannot steer this hardware diagnostic.
        pass

    def apply(self, axes, frame_at, now):
        if now >= self.started + self.stop_after:
            self.phase = "stopping"
            return (0, 0, 0, 0)
        if now < self.started + CRUISE_START:
            self.phase = "settling"
            return (0, 0, 0, 0)
        if frame_at is None or frame_at < self.started + CRUISE_START:
            self.phase = "waiting"
            return (0, 0, 0, 0)
        self.armed = True
        self.phase = "forward-check"
        return (0, FORWARD_CHECK_RC, 0, 0)


class FullGate:
    """Short whole-controller trial; timing limits do not depend on quiet yaw."""

    stop_after = FULL_END

    def __init__(self, acknowledged_at):
        self.started = acknowledged_at
        self.armed = False
        self.phase = "settling"
        self.stop_reason = ""

    def observe(self, row, frame_at):
        if frame_at >= self.started + FULL_START and row.get("escape", False):
            self.stop_reason = "escape: forward disabled for remainder of trial"

    def apply(self, axes, frame_at, now):
        if now >= self.started + self.stop_after:
            self.phase = "stopping"
            return (0, 0, 0, 0)
        if now < self.started + FULL_START:
            self.phase = "settling"
            return (0, 0, 0, 0)
        if frame_at is None or frame_at < self.started + FULL_START:
            self.phase = "waiting"
            return (0, 0, 0, 0)
        self.armed = True
        self.phase = "braked" if self.stop_reason else "full"
        return (axes[0], 0, axes[2], axes[3]) if self.stop_reason else axes
