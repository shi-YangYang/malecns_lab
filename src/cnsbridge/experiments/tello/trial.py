"""Bounded Tello flight trials, at most 12 s after takeoff ACK.

Requires --takeoff. No connection is made without it. Default mode is yaw-only;
--mode reflex also permits limited vertical motion. --mode cruise permits up to
3 s of MaleCNS cruise (forward RC 0-15) between 2 s settling and 2 s zero RC.
Cruise uses a 7 s trial; full uses 12 s; the other modes remain limited to 5 s.
All modes observe the complete local MaleCNS graph and its calibrated readout.
--mode full permits:
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
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from cnsbridge.config import load_config
from cnsbridge.paths import source_project_root
from .policy import (
    CRUISE_TRIAL_SECONDS, CRUISE_VALUE, FULL_TRIAL_SECONDS, INPUT_TIMEOUT,
    MAX_YAW_DEGREES, MIN_TAKEOFF_BATTERY, MODES, PERIOD, RESULT_TIMEOUT,
    CruiseGate, ForwardCheckGate, FullGate, flight_rc, validate_duration,
)
from .session import Inputs, check_ground, cleanup, motion_telemetry
from .worker import BrainProcess

PROJECT = source_project_root()


def trial_config(mode):
    if mode not in MODES:
        raise ValueError("Unknown trial mode")
    cfg = load_config(PROJECT / "configs" / "malecns.yaml", overrides={
        "decoder": {"cruise": CRUISE_VALUE if mode in ("cruise", "full") else 0.0},
    })
    if mode != "full":
        # Keep the existing bounded modes' camera/control/safety settings.
        tello_cfg = load_config(PROJECT / "configs" / "tello.yaml")
        for section in ("vision", "control", "safety"):
            cfg[section] = tello_cfg[section]
    # Missing model/calibration assets fail before SDK construction.
    for section, key in (("brain", "source"), ("decoder", "readout_file")):
        value = cfg[section].get(key)
        if not value:
            raise ValueError(f"Trial requires {section}.{key}")
        path = (PROJECT / Path(value).expanduser()).resolve()
        if not path.is_file():
            raise ValueError(f"Trial asset not found: {path}")
        cfg[section][key] = str(path)
    return cfg


def report_motion(rows):
    for phase in ("settling", "forward-check", "stopping"):
        samples = [row for row in rows if row["trial_phase"] == phase]
        fields = []
        for field in ("state_pitch_deg", "state_roll_deg", "state_vgx", "state_vgy", "state_vgz"):
            values = [row[field] for row in samples if row[field] is not None]
            span = f"[{min(values):g},{max(values):g}]" if values else "unavailable"
            fields.append(f"{field.removeprefix('state_')}={span}")
        print(f"MOTION TELEMETRY | phase={phase} | " + " | ".join(fields))


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
                "control_source": "fixed_probe" if mode == "forward-check" else "malecns",
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
            "reflex": "MaleCNS yaw + vertical (each RC capped at 10; no forward/lateral)",
            "cruise": "2 s zero RC, up to 3 s MaleCNS cruise (forward RC 0-15; yaw/vertical capped at 10), 2 s zero RC",
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
                print("Fixed forward RC was sent; visually confirm forward motion and slowing. This does not verify neural control or obstacle avoidance.")
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
                        help="yaw: turn only; reflex: add climb/descent; cruise: low MaleCNS forward RC; forward-check: fixed RC15 probe; full: MaleCNS visual flight with braking/escape")
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
    log_prefix = {"yaw": "tello_yaw_trial", "reflex": "tello_reflex_trial",
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
