"""Record ground-only Tello video for later manual action labels.

Keep Tello on the ground and close the phone app and other Tello scripts.
Only SDK connection, telemetry and video operations are used. No brain model
or aircraft motion controller is loaded. Q / Escape ends the preview.

Label intervals use AVI playback seconds (frame_index / nominal_fps), which
are also shown in the preview and frames.csv. Separate host dequeue timestamps
audit recording gaps; neither clock represents sensor capture timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from malecns_lab.paths import source_project_root

PROJECT = source_project_root()
NOMINAL_FPS = 20
FRAME_FIELDS = ("frame_index", "t_s", "video_t_s")
LABEL_FIELDS = ("start_s", "end_s", "vx_mps", "wz_rad_s", "reason")


def _ground_state(state):
    if not isinstance(state, dict) or not all(key in state for key in ("bat", "h", "yaw")):
        raise RuntimeError("Incomplete Tello telemetry: battery, height and yaw are required")
    try:
        battery, height, yaw = (float(state[key]) for key in ("bat", "h", "yaw"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise RuntimeError("Invalid Tello telemetry") from exc
    if not all(math.isfinite(value) for value in (battery, height, yaw)) or not 0 <= battery <= 100:
        raise RuntimeError("Invalid Tello telemetry")
    if not 0 <= height <= 10:
        raise RuntimeError("Ground recording requires height between 0 and 10 cm; place Tello on the ground")
    return battery, height


def _video_writer(path, size, fps):
    import cv2

    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)


def run_recording(drone, seconds, out_dir, scene_id, *, preview=None,
                  writer_factory=None, clock=None, sleep=None) -> int:
    """Record fresh frames; injected dependencies let tests run without hardware.

    ``writer_factory(path, (width, height), fps)`` returns a VideoWriter-like
    object. ``preview(bgr_frame, row)`` may return False to end recording.
    The injected drone exposes only the SDK's ground connection/video methods.
    """
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    writer_factory = writer_factory or _video_writer
    out_dir = Path(out_dir)
    video_writer = None
    count = 0
    result = 0
    last_clock = None

    def now():
        nonlocal last_clock
        current = float(clock())
        if not math.isfinite(current) or (last_clock is not None and current < last_clock):
            raise RuntimeError("Recording clock must remain finite and monotonic")
        last_clock = current
        return current

    try:
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 5 <= seconds <= 60:
            raise ValueError("--seconds must be between 5 and 60")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("--scene-id must be a nonempty scene identifier")
        # Reserve the complete session before the connection. Existing sessions,
        # including interrupted ones, are never silently replaced or appended.
        out_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "format": "flydrones.ground_recording", "version": 1,
            "session_id": out_dir.name, "scene_id": scene_id.strip(),
            "video": "video.avi", "frames": "frames.csv", "labels": "labels.csv",
            "nominal_fps": NOMINAL_FPS, "timing": "host_dequeue", "ground_only": True,
            "label_timebase": "video_playback",
        }
        with (out_dir / "session.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        with (out_dir / "labels.csv").open("x", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(LABEL_FIELDS)
        with (out_dir / "frames.csv").open("x", newline="", encoding="utf-8") as frame_file:
            frame_csv = csv.DictWriter(frame_file, fieldnames=FRAME_FIELDS)
            frame_csv.writeheader()
            drone.connect()
            state = drone.get_current_state()
            battery, height = _ground_state(state)
            print(f"SDK OK | Battery: {battery:g}% | Height: {height:g} cm", flush=True)
            drone.streamon()
            reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
            started = next_tick = last_frame = last_state = last_report = now()
            previous_state = state
            previous_frame_time = None
            frame_size = None
            while True:
                tick = now()
                if tick - started >= seconds:
                    break
                state = drone.get_current_state()
                _ground_state(state)  # Check every poll, including while waiting for video.
                if state is not previous_state:
                    # DJITelloPy replaces the dict for each received state packet.
                    previous_state, last_state = state, tick
                if tick - last_state > 1:
                    raise RuntimeError("Telemetry stopped updating for more than 1 second")
                if tick < next_tick:
                    sleep(min(0.005, next_tick - tick))
                    continue
                rgb = reader.frame  # Queue mode consumes a decoded frame once.
                dequeued_at = now()
                if rgb is None:
                    if dequeued_at - last_frame > (3 if count else 10):
                        raise RuntimeError("Video stopped updating" if count else "No decoded video frames received")
                    sleep(0.005)
                    continue
                if dequeued_at - started >= seconds:
                    break
                if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
                    raise RuntimeError("Decoded video frame must be a nonempty uint8 RGB image")
                timestamp = dequeued_at - started
                if previous_frame_time is not None and timestamp <= previous_frame_time:
                    raise RuntimeError("Frame timestamps must increase strictly")
                size = (rgb.shape[1], rgb.shape[0])
                if video_writer is None:
                    frame_size = size
                    video_writer = writer_factory(out_dir / "video.avi", size, NOMINAL_FPS)
                    if not video_writer.isOpened():
                        raise RuntimeError("Cannot open MJPG video writer; check OpenCV codec support and output path")
                elif size != frame_size:
                    raise RuntimeError("Video dimensions changed during the recording")
                frame = np.ascontiguousarray(rgb[..., ::-1])
                video_writer.write(frame)
                row = {"frame_index": count, "t_s": timestamp, "video_t_s": count / NOMINAL_FPS}
                frame_csv.writerow(row)
                count += 1
                previous_frame_time = timestamp
                last_frame = dequeued_at
                next_tick = dequeued_at + 1.0 / NOMINAL_FPS
                if dequeued_at - last_report >= 1:
                    frame_file.flush()
                    print(f"RECORDING | video t_s={row['video_t_s']:.2f} | host t_s={timestamp:.2f} | frames={count} | GROUND ONLY", flush=True)
                    last_report = dequeued_at
                if preview is not None and preview(frame, row) is False:
                    break
            if count < 2:
                raise RuntimeError("Too few decoded frames to form a recording")
    except KeyboardInterrupt:
        print("\nRecording interrupted; partial files are retained.", flush=True)
        result = 130
    except Exception as exc:
        print(f"RECORDING FAILED: {exc}", file=sys.stderr)
        result = 1
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as exc:
                print(f"Video cleanup failed: {exc}", file=sys.stderr)
                result = 1
        try:
            # The CLI creates a fresh instance that never enters a flying state;
            # end() only releases that instance's streaming/session resources.
            drone.end()
        except Exception as exc:
            print(f"SDK cleanup failed: {exc}", file=sys.stderr)
            result = 1
    if count:
        print(f"RECORDED | frames={count} | session: {out_dir}")
    if result == 0:
        print("GROUND RECORDING COMPLETE | labels.csv contains a header only; add desired-action labels manually.")
        print("Label start_s/end_s use AVI playback seconds (video_t_s); frames.csv also preserves host dequeue t_s for timing audits.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True, help="scene identifier; reuse for recordings of the same scene")
    parser.add_argument("--seconds", type=float, default=30, help="recording duration, 5-60 seconds (default: 30)")
    parser.add_argument("--out-dir", type=Path, help="new session directory; must not already exist")
    parser.add_argument("--no-window", action="store_true", help="save video and timestamps without a preview")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 60:
        parser.error("--seconds must be between 5 and 60")
    if not args.scene_id.strip():
        parser.error("--scene-id must be nonempty")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = (args.out_dir or PROJECT / "data" / "ground_recordings" / f"tello_{stamp}").expanduser().resolve()
    if out_dir.exists():
        parser.error(f"--out-dir already exists; choose a new session directory: {out_dir}")
    print("GROUND RECORDING | Keep Tello on the ground. Video and telemetry only.", flush=True)
    cv2 = None
    try:
        import cv2
        from djitellopy import Tello

        Tello.LOGGER.setLevel(logging.WARNING)
        preview = None
        if not args.no_window:
            window = "Tello dataset recording - GROUND ONLY - Q / Escape"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            def preview(frame, row):
                canvas = frame.copy()
                text = f"GROUND ONLY | video={row['video_t_s']:.2f}s | host={row['t_s']:.2f}s | frames={row['frame_index'] + 1}"
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
                cv2.imshow(window, canvas)
                return cv2.waitKey(1) & 0xFF not in (ord("q"), 27) and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1

        return run_recording(Tello(retry_count=1), args.seconds, out_dir, args.scene_id, preview=preview)
    except Exception as exc:
        print(f"RECORDING FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
