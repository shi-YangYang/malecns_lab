"""Shared telemetry, input freshness, preflight and landing cleanup."""

from __future__ import annotations

import math
import signal
import sys
import time

import numpy as np

from .policy import INPUT_TIMEOUT


def fresh_state(drone, timeout: float = 1.0) -> dict:
    # DJITelloPy 2.5 replaces its state dict for every received UDP packet.
    # Keep the old object alive: equal field values can still be a new packet.
    previous = drone.get_current_state()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.05)
        state = drone.get_current_state()
        if state is not previous and "bat" in state and "h" in state:
            return state
    raise RuntimeError("No fresh telemetry for 1 second")


def check_ground(drone, min_battery: int = 50) -> None:
    state = fresh_state(drone)
    print(f"PREFLIGHT | Battery: {state['bat']}% | Height: {state['h']} cm", flush=True)
    if not min_battery <= state["bat"] <= 100:
        raise RuntimeError(f"This test requires at least {min_battery}% battery")
    if not 0 <= state["h"] <= 10:
        raise RuntimeError("Aircraft must be on the ground before this test")


def cleanup(drone, takeoff_attempted: bool) -> bool:
    ok = True
    messages = []
    # A second Ctrl+C must not skip the landing attempt already in progress.
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if takeoff_attempted:
            try:
                drone.send_rc_control(0, 0, 0, 0)
            except Exception as exc:
                messages.append((f"Could not send zero sticks: {exc}; still attempted landing.", True))

            landed = False
            for attempt in range(1, 3):
                try:
                    # Never gate this on is_flying: takeoff can succeed while
                    # its reply is lost, leaving the SDK's flag false.
                    drone.land()
                    messages.append((f"LAND ACKNOWLEDGED | attempt {attempt}/2 | Confirm touchdown and stopped propellers visually.", False))
                    landed = True
                    break
                except Exception as exc:
                    messages.append((f"Landing attempt failed: {exc}", True))
            if not landed:
                ok = False
                messages.append(("LANDING NOT CONFIRMED. Keep clear and watch the aircraft.", True))
        try:
            # SDK end() may make another landing attempt if is_flying is true.
            drone.end()
        except Exception as exc:
            ok = False
            messages.append((f"SDK cleanup failed: {exc}", True))
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    # Console output can block; never place it ahead of a landing attempt.
    for message, error in messages:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)
    return ok


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
