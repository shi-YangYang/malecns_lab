"""Bounded Tello flight trials, at most 12 s after takeoff ACK.

Requires --takeoff. No connection is made without it. Default mode is yaw-only;
--mode reflex also permits limited vertical motion. --mode cruise permits up to
3 s of MiniFly cruise (forward RC 0-15) between 2 s settling and 2 s zero RC.
Cruise uses a 7 s trial; full uses 12 s; the other modes remain limited to 5 s.
--mode full runs the complete local MaleCNS graph and its calibrated readout:
2 s zero RC, up to 8 s visual flight control, then 2 s zero RC. Forward cruise
is capped at RC15; yaw, vertical and lateral at RC10. Neural braking and escape
remain enabled. The current readout has no learned lateral output. Its yaw
direction and axis separation are unverified; this is a bounded integration trial.
Cruise tolerates small isolated yaw pulses, but requires quiet average yaw,
quiet vertical output and no recent escape. Escape latches forward off for this run.
--mode forward-check is a hardware probe: 2 s zero RC, 1 s fixed forward RC15,
then 2 s zero RC. Other axes stay zero. The brain is recorded and supervised,
but does not select the probe's motion; this mode does not avoid obstacles.
Fit prop guards, use a clear, bright indoor area, and keep clear of the aircraft.
Ctrl+C requests landing. Confirm touchdown visually. Use open space, not a wall
or approaching object, for this first cruise test. RC limits are not distances.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing as mp
import queue
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from tello_hover import check_ground, cleanup
from tello_shadow import PROJECT, ShadowBrain

from cnsbridge.config import load_config

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


def trial_config(mode):
    if mode not in MODES:
        raise ValueError("Unknown trial mode")
    if mode == "full":
        cfg = load_config(PROJECT / "configs" / "malecns.yaml", overrides={
            "decoder": {"cruise": CRUISE_VALUE},
        })
        # Resolve both assets relative to the project, even when run in server/.
        # Missing assets must fail before SDK construction; never use MiniFly
        # or uncalibrated default weights as a fallback.
        for section, key in (("brain", "source"), ("decoder", "readout_file")):
            value = cfg[section].get(key)
            if not value:
                raise ValueError(f"Full trial requires {section}.{key}")
            path = (PROJECT / Path(value).expanduser()).resolve()
            if not path.is_file():
                raise ValueError(f"Full trial asset not found: {path}")
            cfg[section][key] = str(path)
        return cfg
    return load_config(PROJECT / "configs" / "tello.yaml", overrides={
        "brain": {"source": "minifly"},
        "decoder": {"cruise": CRUISE_VALUE if mode == "cruise" else 0.0},
    })


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


def motion_telemetry(state):
    """Preserve optional SDK motion readings; missing/invalid is not zero."""
    fields = {"pitch": "state_pitch_deg", "roll": "state_roll_deg",
              "vgx": "state_vgx", "vgy": "state_vgy", "vgz": "state_vgz"}
    readings = {}
    for source, target in fields.items():
        try:
            value = float(state[source])
            readings[target] = value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            readings[target] = None
    return readings


def report_motion(rows):
    for phase in ("settling", "forward-check", "stopping"):
        samples = [row for row in rows if row["trial_phase"] == phase]
        fields = []
        for field in ("state_pitch_deg", "state_roll_deg", "state_vgx", "state_vgy", "state_vgz"):
            values = [row[field] for row in samples if row[field] is not None]
            span = f"[{min(values):g},{max(values):g}]" if values else "unavailable"
            fields.append(f"{field.removeprefix('state_')}={span}")
        print(f"MOTION TELEMETRY | phase={phase} | " + " | ".join(fields))


def _brain_main(cfg, requests, results) -> None:
    # The control process owns Ctrl+C and the aircraft. This process only computes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    job_id = -1
    try:
        engine = ShadowBrain(cfg)
        while True:
            job_id, frame, state, t, reset = requests.get()
            if reset:
                engine.retina.reset()
                engine.retina.prev_exp = {"L": 0.0, "R": 0.0}
                engine.last_yaw = engine.last_t = None
                engine.yaw_rate = 0.0
            row = engine.step(frame, state, t)
            row["brain_source"] = str(cfg["brain"]["source"])
            row["brain_neurons"] = engine.brain.n_neurons
            results.put((job_id, row, None))
    except Exception as exc:
        results.put((job_id, None, f"{type(exc).__name__}: {exc}"))


class BrainProcess:
    """One outstanding image, tiny scalar results, no SDK access in the child."""

    def __init__(self, cfg):
        ctx = mp.get_context("spawn")
        self.requests = ctx.Queue(maxsize=1)
        self.results = ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_brain_main, args=(cfg, self.requests, self.results), daemon=True)
        self.process.start()
        self.sequence = 0

    def alive(self) -> bool:
        return self.process.is_alive()

    def submit(self, frame, state, t, reset=False) -> int:
        self.sequence += 1
        # put_nowait keeps a stalled worker from blocking the aircraft controller.
        self.requests.put_nowait((self.sequence, frame, dict(state), t, reset))
        return self.sequence

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()


class Inputs:
    def __init__(self, drone, reader):
        self.drone, self.reader = drone, reader
        self.previous = None
        self.state_at = self.frame_at = time.monotonic()

    def state(self, now):
        state = self.drone.get_current_state()
        if state is not self.previous:
            if not all(key in state and math.isfinite(float(state[key])) for key in ("h", "bat", "yaw")):
                raise RuntimeError("Invalid aircraft telemetry")
            self.previous, self.state_at = state, now
        if now - self.state_at > INPUT_TIMEOUT:
            raise RuntimeError("Telemetry stopped updating")
        return state

    def frame(self, now):
        rgb = self.reader.frame
        if rgb is not None:
            self.frame_at = now
            return np.ascontiguousarray(rgb[..., ::-1])
        return None


def _result(worker, pending, mode="yaw"):
    received = worker.poll()
    if received is None:
        if not worker.alive():
            raise RuntimeError("Brain process exited")
        return None
    job_id, row, error = received
    if error:
        raise RuntimeError(f"Brain failed: {error}")
    if pending is None or job_id != pending[0]:
        raise RuntimeError("Unexpected or old brain result")
    flight_rc(row, 80, mode)  # Validate newly enabled axes before use.
    return row


def warmup(inputs, worker, mode="yaw"):
    print("GROUND WARMUP | Keep the scene still; no motion commands are sent.", flush=True)
    started = next_frame = time.monotonic()
    pending = None
    recent = []
    received = 0
    while time.monotonic() - started < 15:
        now = time.monotonic()
        state = inputs.state(now)
        row = _result(worker, pending, mode)
        if row is not None:
            age = now - pending[1]
            pending = None
            received += 1
            motion = 0.0 if mode == "forward-check" else abs(float(row["cmd_yaw"]))
            if mode in ("reflex", "cruise", "full"):
                motion = max(motion, abs(float(row["cmd_throttle"])))
                if row.get("escape", False):
                    motion = max(motion, 1.0)
            if mode == "full":
                motion = max(motion, abs(float(row["cmd_lateral"])))
            recent.append((motion, age))
            recent = recent[-10:]
            if received >= 40 and row["phase"] == "ready":
                if max(age for _, age in recent) > 0.2:
                    raise RuntimeError("Brain too slow during ground warmup")
                if max(motion for motion, _ in recent) > 0.1:
                    raise RuntimeError("Ground motion is not settled; keep the camera scene still and retry")
                if mode == "full":
                    print(f"BRAIN READY | source={row['brain_source']} | neurons={row['brain_neurons']} | "
                          f"recent_max_image_age={max(age for _, age in recent) * 1000:.1f} ms", flush=True)
                return
        if now >= next_frame:
            frame = inputs.frame(now)
            next_frame = now + PERIOD
            if frame is not None and pending is None:
                job = worker.submit(frame, state, now - started)
                pending = (job, now)
        if now - inputs.frame_at > 3:
            raise RuntimeError("Video stopped during warmup")
        time.sleep(0.005)
    raise RuntimeError("Ground warmup did not complete within 15 seconds")


def fly_trial(drone, worker, inputs, seconds, rows, acknowledged_at, mode="yaw"):
    """Supervise the worker without waiting for it; return on deadline or raise."""
    initial = drone.get_current_state()
    launch_yaw = None
    # Discard the image left in the queue by takeoff; the next decoded frame
    # primes a reset retina, and its proposal is deliberately not sent.
    inputs.reader.frame
    inputs.previous = initial
    inputs.state_at = inputs.frame_at = acknowledged_at
    next_frame = next_rc = acknowledged_at
    pending = None
    latest = None
    latest_job = None
    result_at = None
    first_image = True
    allow_motion = False
    priming_job = None
    gate = (CruiseGate(acknowledged_at) if mode == "cruise" else
            FullGate(acknowledged_at) if mode == "full" else
            ForwardCheckGate(acknowledged_at) if mode == "forward-check" else None)
    while time.monotonic() - acknowledged_at < seconds:
        now = time.monotonic()
        state = inputs.state(now)
        # Do not treat the packet cached during takeoff as fresh airborne data.
        # Waiting stays inside this loop so the deadline and zero RC still run.
        if inputs.state_at <= acknowledged_at:
            if now >= next_rc:
                drone.send_rc_control(0, 0, 0, 0)
                next_rc = now + PERIOD
            time.sleep(0.005)
            continue
        if launch_yaw is None:
            launch_yaw = float(state["yaw"])
        if state["bat"] < 25:
            raise RuntimeError("Battery below 25%")
        if not 30 <= state["h"] <= 150:
            raise RuntimeError(f"Height outside trial range (30-150 cm): {state['h']}")
        delta = (float(state["yaw"]) - launch_yaw + 180) % 360 - 180
        if abs(delta) >= MAX_YAW_DEGREES:
            raise RuntimeError("Heading changed by 25 degrees; ending trial")

        row = _result(worker, pending, mode)
        if row is not None:
            result_at = pending[1]  # Age from image submission, not completion.
            allow_motion = pending[0] != priming_job
            latest_job = pending[0]
            pending = None
            latest = row
            if gate is not None and allow_motion:
                gate.observe(row, result_at)
        if (result_at is not None and now - result_at > RESULT_TIMEOUT) or (
            pending is not None and now - pending[1] > RESULT_TIMEOUT
        ):
            raise RuntimeError("No fresh brain output for 0.35 seconds")
        if now >= next_frame:
            frame = inputs.frame(now)
            next_frame = now + PERIOD
            if frame is not None and pending is None:
                job = worker.submit(frame, state, now - acknowledged_at, reset=first_image)
                if first_image:
                    priming_job = job
                    first_image = False
                pending = (job, now)
        if now - inputs.frame_at > INPUT_TIMEOUT:
            raise RuntimeError("Video stopped updating for 0.5 seconds")

        stopping_now = gate is not None and now >= acknowledged_at + gate.stop_after and gate.phase != "stopping"
        if now >= next_rc or stopping_now:
            # End forward output on the first control iteration past the fixed
            # boundary instead of waiting for the next periodic RC send.
            axes = flight_rc(latest, state["h"], mode) if latest is not None and allow_motion else (0, 0, 0, 0)
            if gate is not None:
                axes = gate.apply(axes, result_at if allow_motion else None, now)
            active = allow_motion and (gate is None or gate.phase in ("cruise", "braked", "forward-check", "full"))
            # Only the parent can actuate. Full mode admits all model axes;
            # other modes keep lateral locked. Forward uses a fixed time window.
            # ShadowBrain supplies proposals; its ground telemetry flag is not
            # relied on for airborne altitude protection.
            drone.send_rc_control(*axes)
            rows.append({
                "t_s": now - acknowledged_at,
                "mode": mode,
                "control_ready": active,
                "control_source": "fixed_probe" if mode == "forward-check" else ("malecns" if mode == "full" else "minifly"),
                "brain_source": latest.get("brain_source", "") if latest else "",
                "brain_neurons": latest.get("brain_neurons") if latest else None,
                "brain_job_id": latest_job,
                "model_t_s": latest.get("model_t_s") if latest else None,
                "trial_phase": gate.phase if gate else ("active" if active else "priming"),
                "cruise_armed": gate.armed if mode == "cruise" else False,
                "forward_check_armed": gate.armed if mode == "forward-check" else False,
                "forward_stop_reason": gate.stop_reason if gate else "",
                "raw_yaw": latest["raw_yaw"] if latest else 0.0,
                "proposed_yaw": latest["cmd_yaw"] if latest else 0.0,
                "raw_throttle": latest.get("raw_throttle", 0.0) if latest else 0.0,
                "proposed_throttle": latest.get("cmd_throttle", 0.0) if latest else 0.0,
                "raw_forward": latest.get("raw_forward", 0.0) if latest else 0.0,
                "proposed_forward": latest.get("cmd_forward", 0.0) if latest else 0.0,
                "proposed_lateral": latest.get("cmd_lateral", 0.0) if latest else 0.0,
                "rc_lateral": axes[0], "rc_forward": axes[1], "rc_throttle": axes[2], "rc_yaw": axes[3],
                "expansion": latest.get("expansion", 0.0) if latest else 0.0,
                "rotation": latest.get("rotation", 0.0) if latest else 0.0,
                "vertical": latest.get("vertical", 0.0) if latest else 0.0,
                "escape": bool(latest.get("escape", False)) if latest else False,
                **{f"rate_{name}": latest.get(f"rate_{name}", 0.0) if latest else 0.0
                   for name in ("DNp01_L", "DNp01_R", "DNp03_L", "DNp03_R", "DNg02_L", "DNg02_R")},
                "yaw_deg": state["yaw"], "yaw_delta_deg": delta,
                "height_cm": state["h"], "battery_pct": state["bat"],
                **motion_telemetry(state),
                "image_age_ms": (now - result_at) * 1000 if result_at is not None else None,
                "processing_ms": latest["processing_ms"] if latest else 0.0,
            })
            next_rc = now + PERIOD
        # No GUI, terminal output or disk writes in the airborne control loop.
        time.sleep(0.005)


def run_trial(drone, worker, seconds: float, log_path: Path, mode="yaw") -> int:
    takeoff_attempted = False
    rows = []
    result = 0
    failure = None
    logfile = None
    try:
        validate_duration(seconds, mode)
        logfile = Path(log_path).open("x", newline="", encoding="utf-8")
        drone.connect()
        check_ground(drone, min_battery=MIN_TAKEOFF_BATTERY)
        drone.streamon()
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        inputs = Inputs(drone, reader)
        warmup(inputs, worker, mode)
        for remaining in (3, 2, 1):
            print(f"TAKEOFF IN {remaining} | Ctrl+C cancels", flush=True)
            time.sleep(1)
        check_ground(drone, min_battery=MIN_TAKEOFF_BATTERY)
        if not worker.alive():
            raise RuntimeError("Brain process exited before takeoff")
        # Prove the video is still producing new decoded images after countdown.
        reader.frame
        deadline = time.monotonic() + 1
        while reader.frame is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("No new video before takeoff")
            time.sleep(0.01)
        motion = {
            "yaw": "yaw-only",
            "reflex": "MiniFly yaw + vertical (each RC capped at 10; no forward/lateral)",
            "cruise": "2 s zero RC, up to 3 s MiniFly cruise (forward RC 0-15; yaw/vertical capped at 10), 2 s zero RC",
            "forward-check": "2 s zero RC, up to 1 s fixed forward RC15, 2 s zero RC (other axes zero; no obstacle avoidance)",
            "full": "2 s zero RC, up to 8 s MaleCNS visual flight (forward RC 0-15; other axes capped at 10; neural braking/escape), 2 s zero RC",
        }[mode]
        print(f"TAKING OFF | then {motion} control for {seconds:g} s; Ctrl+C requests landing.", flush=True)
        takeoff_attempted = True
        drone.takeoff()
        acknowledged_at = time.monotonic()
        fly_trial(drone, worker, inputs, seconds, rows, acknowledged_at, mode)
    except KeyboardInterrupt:
        failure = "Interrupted; attempted landing."
        result = 130
    except Exception as exc:
        failure = f"TRIAL STOPPED: {exc}"
        result = 1
    finally:
        # Land before stopping worker processes or writing buffered logs.
        try:
            if not cleanup(drone, takeoff_attempted):
                result = 1
        finally:
            try:
                worker.close()
            finally:
                if logfile is not None:
                    try:
                        if rows:
                            writer = csv.DictWriter(logfile, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                    except Exception as exc:
                        print(f"Could not save trial log: {exc}", file=sys.stderr)
                        result = 1
                    finally:
                        logfile.close()
    if failure:
        print(failure, file=sys.stderr)
    if rows:
        nonzero = sum(row["rc_yaw"] != 0 for row in rows)
        excursion = max(abs(row["yaw_delta_deg"]) for row in rows)
        print(f"SUMMARY | samples={len(rows)} | nonzero_yaw={nonzero} | max_heading_change={excursion:.1f} deg")
        if mode in ("reflex", "cruise", "full"):
            vertical = sum(row["rc_throttle"] != 0 for row in rows)
            escape = sum(row["control_ready"] and row["escape"] for row in rows)
            low, high = min(row["height_cm"] for row in rows), max(row["height_cm"] for row in rows)
            print(f"REFLEX | nonzero_throttle={vertical} | escape_control_ticks={escape} | height={low:g}-{high:g} cm")
            if not escape:
                print("No escape flag observed during active control; looming escape remains unverified.")
        if mode == "full":
            # RC is sent at 20 Hz; repeated use of one model result is not a
            # second inference. Report unique results separately from RC ticks.
            outputs = {row["brain_job_id"]: row for row in rows if row["brain_job_id"] is not None}
            active_outputs = {row["brain_job_id"] for row in rows if row["control_ready"]}
            forwarded = sum(row["rc_forward"] > 0 for row in rows)
            p95 = float(np.percentile([row["processing_ms"] for row in outputs.values()], 95)) if outputs else 0.0
            print(f"MALECNS | unique_outputs={len(outputs)} | active_outputs={len(active_outputs)} | "
                  f"nonzero_forward={forwarded} | processing_p95={p95:.1f} ms")
            if not active_outputs:
                print("FULL UNVERIFIED | No fresh model output reached the active control window.")
                result = 1
            else:
                print("Full model outputs reached bounded control. Correct direction, distance and obstacle avoidance remain unverified.")
            for reason in sorted({row["forward_stop_reason"] for row in rows if row["forward_stop_reason"]}):
                print(f"FULL NOTE | {reason}")
        if mode == "cruise":
            forwarded = [row for row in rows if row["rc_forward"] > 0]
            print(f"CRUISE | nonzero_forward={len(forwarded)} | max_forward_rc={max(row['rc_forward'] for row in rows)}")
            reasons = {row["forward_stop_reason"] for row in rows if row["forward_stop_reason"]}
            for reason in sorted(reasons):
                print(f"CRUISE NOTE | {reason}")
            if not forwarded:
                print("CRUISE UNVERIFIED | No forward RC was sent; this was a hover-only run.")
                result = 1
            else:
                print("Forward RC was sent; visually verify forward motion and slowing. Distance and obstacle avoidance are unverified.")
        if mode == "forward-check":
            forwarded = [row for row in rows if row["rc_forward"] > 0]
            print(f"FORWARD CHECK | nonzero_forward={len(forwarded)} | max_forward_rc={max(row['rc_forward'] for row in rows)}")
            report_motion(rows)
            print("Velocity fields are raw SDK readings, not measured travel distance.")
            if not forwarded:
                print("FORWARD CHECK UNVERIFIED | No forward RC send was recorded.")
                if result == 0:
                    result = 1
            else:
                print("Fixed forward RC was sent; visually confirm forward motion and slowing. This does not verify MiniFly or obstacle avoidance.")
        print(f"CSV: {log_path}")
        if not nonzero and mode != "forward-check":
            print("No nonzero yaw was sent; directional response remains unverified.")
    elif mode in ("cruise", "forward-check", "full") and result == 0:
        print(f"{mode.upper()} UNVERIFIED | No airborne control samples were recorded.")
        result = 1
    if result == 0:
        print("TRIAL COMPLETE | Confirm actual motion, touchdown and stopped propellers visually.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takeoff", action="store_true", help="fly the real aircraft in the selected trial mode")
    parser.add_argument("--mode", choices=MODES, default="yaw",
                        help="yaw: turn only; reflex: add climb/descent; cruise: low MiniFly forward RC; forward-check: fixed RC15 probe; full: MaleCNS visual flight with braking/escape")
    parser.add_argument("--seconds", type=float,
                        help="total duration after takeoff ACK: full exactly 12 s; cruise exactly 7 s; forward-check exactly 5 s; yaw/reflex 1-5 s (default 5)")
    args = parser.parse_args(argv)
    if args.seconds is None:
        args.seconds = {"full": FULL_TRIAL_SECONDS, "cruise": CRUISE_TRIAL_SECONDS}.get(args.mode, 5.0)
    try:
        validate_duration(args.seconds, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.takeoff:
        parser.print_help()
        print("\nNo connection made. --takeoff is required to fly.")
        return 0
    try:
        cfg = trial_config(args.mode)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    log_prefix = {"yaw": "tello_yaw_trial", "reflex": "tello_minifly_trial",
                  "cruise": "tello_cruise_trial", "forward-check": "tello_forward_check",
                  "full": "tello_malecns_full_trial"}[args.mode]
    log_path = PROJECT / "logs" / f"{log_prefix}_{stamp}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    from djitellopy import Tello

    # SDK failures are reported after landing; avoid its synchronous log writes
    # inside airborne command/error paths.
    Tello.LOGGER.setLevel(logging.CRITICAL)
    worker = BrainProcess(cfg)
    try:
        drone = Tello(retry_count=1)
    except BaseException:
        worker.close()
        raise
    return run_trial(drone, worker, args.seconds, log_path, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
