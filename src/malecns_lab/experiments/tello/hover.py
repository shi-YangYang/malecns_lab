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
import sys
import time

from .session import check_ground, cleanup, fresh_state


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
