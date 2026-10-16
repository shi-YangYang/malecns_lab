"""First Tello flight: take off, hold zero sticks for about 5 seconds, then land.

No brain model or video. Without --takeoff, this script does not connect.
Close the phone app and video check before use. Fit prop guards and place the
aircraft on a clear, well-lit, textured floor, away from people and pets.
Ctrl+C requests landing; keep watching until the aircraft is on the ground.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time


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


def run_hover(drone, seconds: float = 5.0) -> int:
    takeoff_attempted = False
    result = 0
    try:
        if not math.isfinite(seconds) or not 1 <= seconds <= 5:
            raise ValueError("Hover duration must be between 1 and 5 seconds")
        drone.connect()
        check_ground(drone)
        for remaining in (3, 2, 1):
            print(f"TAKEOFF IN {remaining} | Ctrl+C cancels", flush=True)
            time.sleep(1)
        check_ground(drone)
        takeoff_attempted = True
        print("TAKING OFF", flush=True)
        drone.takeoff()
        print(f"HOVER | zero sticks for {seconds:g} seconds after takeoff acknowledgment", flush=True)
        started = last_report = time.monotonic()
        airborne_seen = False
        while time.monotonic() - started < seconds:
            state = fresh_state(drone)
            airborne_seen |= state["h"] > 20
            if state["bat"] < 25:
                raise RuntimeError("Battery fell below 25%; ending flight")
            drone.send_rc_control(0, 0, 0, 0)
            now = time.monotonic()
            if now - last_report >= 1:
                print(f"HOVER | {now - started:.1f}s | Height: {state['h']} cm | Battery: {state['bat']}%", flush=True)
                last_report = now
        if not airborne_seen:
            raise RuntimeError("Telemetry never showed height above 20 cm; flight was not verified")
    except KeyboardInterrupt:
        print("\nInterrupted; landing if takeoff was attempted.", flush=True)
        result = 130
    except Exception as exc:
        print(f"TEST FAILED: {exc}", file=sys.stderr)
        result = 1
    finally:
        if not cleanup(drone, takeoff_attempted):
            result = 1
    if result == 0:
        print("SDK TEST COMPLETE | Visual confirmation of landing is still required.", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takeoff", action="store_true", help="allow this script to fly the physical aircraft")
    parser.add_argument("--seconds", type=float, default=5, help="hover after takeoff acknowledgment, 1-5 seconds")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 5:
        parser.error("--seconds must be between 1 and 5")
    if not args.takeoff:
        parser.print_help()
        print("\nNo connection made. --takeoff is required to fly.")
        return 0

    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.WARNING)
    # Never automatically repeat takeoff after an ambiguous response.
    return run_hover(Tello(retry_count=1), args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
