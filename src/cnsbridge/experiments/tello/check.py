"""Ground-only Tello SDK and video check. Never sends flight or motor commands.

Close the phone app, join the Tello Wi-Fi from this computer, then run:
    .venv/bin/python tools/tello_check.py

Keep the aircraft on the ground. Press Q or Escape to end the video preview.
"""

from __future__ import annotations

import argparse
import sys
import time

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20, help="preview duration, 1-60 seconds (default: 20)")
    parser.add_argument("--no-video", action="store_true", help="check SDK connection and telemetry only")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")

    import cv2
    from djitellopy import Tello

    print("GROUND CHECK: SDK connection, telemetry and video only. No takeoff or RC commands.", flush=True)
    drone = None
    window = "Tello ground check - Q / Escape to exit"
    preview_open = False
    try:
        drone = Tello(retry_count=1)
        drone.connect()
        print(f"SDK OK | Battery: {drone.get_battery()}% | Height: {drone.get_height()} cm", flush=True)
        if args.no_video:
            return 0

        drone.streamon()
        # Queue mode returns only decoded frames, not the SDK's initial blank image.
        reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        preview_open = True
        started = last_frame_at = time.monotonic()
        frames = 0
        while time.monotonic() - started < args.seconds:
            now = time.monotonic()
            frame = reader.frame
            if frame is not None:
                frames += 1
                last_frame_at = now
                if frames == 1:
                    print(f"VIDEO: first decoded frame, {frame.shape[1]} x {frame.shape[0]}", flush=True)
                cv2.imshow(window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            elif now - last_frame_at > (3 if frames else 10):
                raise RuntimeError("Video stopped updating" if frames else "No decoded video frames received")
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

        if frames < 2:
            raise RuntimeError("Too few decoded frames to confirm a live video stream")
        print(f"VIDEO CHECK COMPLETE: received {frames} decoded frames. Confirm that the picture changed normally.")
        return 0
    except KeyboardInterrupt:
        print("\nGround check cancelled.")
        return 130
    except Exception as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        print("Check the TELLO Wi-Fi connection, close the phone app, and allow local-network access.", file=sys.stderr)
        return 1
    finally:
        try:
            if drone is not None:
                # This fresh instance never takes off or sets is_flying; end() only
                # stops streaming and releases SDK resources in this script.
                drone.end()
        finally:
            if preview_open:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
