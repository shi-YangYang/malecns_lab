"""Ground-only CNSBridge test: real Tello video -> brain -> logged commands.

Never sends takeoff, landing, emergency or RC commands. Keep the aircraft on
the ground. Close the phone app and other Tello scripts before running.
Keep the scene still during SETTLING, then move an object in front of the
camera. Q / Escape closes the preview. Default brain: synthetic MiniFly.
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

from cnsbridge.brain import Brain, load_connectome
from cnsbridge.config import load_config
from cnsbridge.motor import MotorDecoder
from cnsbridge.motor.command import AXES
from cnsbridge.safety import SafetyGovernor, Telemetry
from cnsbridge.senses import InputEncoder, Retina

PROJECT = Path(__file__).resolve().parents[1]


class ShadowBrain:
    """Run the existing algorithm components, with no aircraft output interface."""

    def __init__(self, cfg: dict):
        self.hz = float(cfg["control"]["hz"])
        if not math.isfinite(self.hz) or not 1 <= self.hz <= 30:
            raise ValueError("Ground test requires a control frequency between 1 and 30 Hz")
        self.dt = 1.0 / self.hz
        self.brain = Brain(load_connectome(cfg["brain"]["source"]), cfg)
        self.retina = Retina.from_config(cfg)
        self.encoder = InputEncoder(self.brain.connectome, cfg)
        self.decoder = MotorDecoder(cfg)
        self.safety = SafetyGovernor(cfg)
        self.steps = 0
        self.last_yaw = self.last_t = None
        self.yaw_rate = 0.0

    def step(self, frame_bgr: np.ndarray, state: dict, t: float) -> dict:
        started = time.perf_counter()
        yaw = float(state["yaw"])
        if self.last_yaw is not None and t > self.last_t:
            delta = (yaw - self.last_yaw + 180) % 360 - 180
            self.yaw_rate = 0.7 * self.yaw_rate + 0.3 * delta / (t - self.last_t)
        self.last_yaw, self.last_t = yaw, t
        vision = self.retina.encode(frame_bgr)
        inputs = self.encoder.encode(vision, self.yaw_rate)
        # Fixed brain time per new frame; actual wall cadence is reported
        # separately. A slow run must not appear to meet the target frequency.
        rates = self.brain.tick(inputs, ms=self.dt * 1000)
        raw = self.decoder.update(rates, self.dt)
        if not all(math.isfinite(float(v)) for v in rates.values()) or not all(
            math.isfinite(getattr(raw, axis)) for axis in AXES
        ):
            raise RuntimeError("Non-finite brain output; stopping ground test")
        tel = Telemetry(t=t, alt_m=state["h"] / 100, yaw_deg=yaw,
                        yaw_rate_dps=self.yaw_rate, battery_pct=state["bat"], flying=False)
        cmd = self.safety.filter(raw, tel, self.dt)
        self.steps += 1
        return {
            "t_s": t,
            "model_t_s": self.steps * self.dt,
            "phase": "ready" if self.decoder.settled else "settling",
            "battery_pct": state["bat"],
            "height_cm": state["h"],
            "yaw_rate_dps": self.yaw_rate,
            **vision.scalars(),
            **{f"raw_{axis}": getattr(raw, axis) for axis in AXES},
            **{f"cmd_{axis}": getattr(cmd, axis) for axis in AXES},
            "escape": cmd.escape,
            "land_requested": self.safety.land_requested,
            **{f"rate_{name}": rates[name] for name in self.brain.output_specs},
            "processing_ms": (time.perf_counter() - started) * 1000,
        }


def run_shadow(drone, engine: ShadowBrain, seconds: float, log_path: Path, preview=None) -> int:
    """Read a fresh SDK session; record proposals without invoking a flight loop."""
    rows = []
    completed_at = []
    result = 0
    try:
        # Fail before connecting if the log cannot be created; never overwrite.
        with Path(log_path).open("x", newline="", encoding="utf-8") as logfile:
            drone.connect()
            state = drone.get_current_state()
            if not all(key in state for key in ("bat", "h", "yaw")):
                raise RuntimeError("Incomplete Tello telemetry")
            if not 0 <= state["h"] <= 10:
                raise RuntimeError("This is a ground-only test; place Tello on the ground")
            print(f"SDK OK | Battery: {state['bat']}% | Height: {state['h']} cm", flush=True)
            drone.streamon()
            reader = drone.get_frame_read(with_queue=True, max_queue_len=1)
            started = next_tick = last_frame = last_state = last_report = time.monotonic()
            previous_state = state
            previous_tick = None
            writer = None
            while time.monotonic() - started < seconds:
                now = time.monotonic()
                state = drone.get_current_state()
                if state is not previous_state and all(key in state for key in ("bat", "h", "yaw")):
                    # DJITelloPy 2.5 replaces the state dict per received packet.
                    previous_state, last_state = state, now
                if now - last_state > 1:
                    raise RuntimeError("Telemetry stopped updating for more than 1 second")
                if now < next_tick:
                    time.sleep(min(0.005, next_tick - now))
                    continue
                rgb = reader.frame
                if rgb is None:
                    if now - last_frame > (3 if rows else 10):
                        raise RuntimeError("Video stopped updating" if rows else "No decoded video frames received")
                    time.sleep(0.005)
                    continue
                gap_ms = (now - last_frame) * 1000
                last_frame = now
                next_tick = now + engine.dt
                # Match TelloDrone.frame() and the existing Pilot input contract.
                frame = np.ascontiguousarray(rgb[..., ::-1])
                row = engine.step(frame, previous_state, now - started)
                # Start-to-start interval; summary Hz also counts preview/log work.
                row["tick_interval_ms"] = (now - previous_tick) * 1000 if previous_tick is not None else 0.0
                row["frame_gap_ms"] = gap_ms  # dequeue gap, not sensor capture latency
                previous_tick = now
                if writer is None:
                    writer = csv.DictWriter(logfile, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                if now - last_report >= 1:
                    values = " ".join(f"{axis}={row[f'cmd_{axis}']:+.2f}" for axis in AXES)
                    phase = row["phase"].upper()
                    print(f"{phase} | {values} | processing={row['processing_ms']:.1f} ms | NOT SENT", flush=True)
                    logfile.flush()
                    last_report = now
                keep_running = preview is None or preview(frame, row) is not False
                completed_at.append(time.monotonic() - started)
                if not keep_running:
                    break
            if sum(row["phase"] == "ready" for row in rows) < 2:
                raise RuntimeError("Too few frames after baseline settling; ground test is incomplete")
    except KeyboardInterrupt:
        print("\nGround test interrupted; recorded rows are kept.", flush=True)
        result = 130
    except Exception as exc:
        print(f"SHADOW FAILED: {exc}", file=sys.stderr)
        result = 1
    finally:
        try:
            # A fresh instance in this script never takes off or sets is_flying;
            # SDK end() only releases the stream/session here.
            drone.end()
        except Exception as exc:
            print(f"SDK cleanup failed: {exc}", file=sys.stderr)
            result = 1
    if rows:
        elapsed = completed_at[-1] if completed_at else 0.0
        actual_hz = len(completed_at) / elapsed if elapsed > 0 else 0.0
        p95 = float(np.percentile([row["processing_ms"] for row in rows], 95))
        print(f"SUMMARY | frames={len(rows)} | actual={actual_hz:.1f} Hz / target={engine.hz:g} Hz | processing_p95={p95:.1f} ms")
        print(f"CSV: {log_path}")
    if result == 0:
        print("SHADOW COMPLETE | Commands were recorded only. This does not validate in-flight behavior.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=30, help="ground test duration, 5-60 seconds")
    parser.add_argument("--brain", default="minifly", help="minifly or path to a connectome .npz")
    parser.add_argument("--config", default=str(PROJECT / "configs" / "tello.yaml"))
    parser.add_argument("--log", type=Path, help="new CSV file; default: project logs/tello_shadow_TIMESTAMP.csv")
    parser.add_argument("--no-window", action="store_true", help="print outputs and save CSV without a preview")
    args = parser.parse_args(argv)
    if not 5 <= args.seconds <= 60:
        parser.error("--seconds must be between 5 and 60")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    log_path = (args.log or PROJECT / "logs" / f"tello_shadow_{stamp}.csv").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, overrides={"brain": {"source": args.brain}})
    engine = ShadowBrain(cfg)
    print(f"GROUND ONLY | Brain: {args.brain} ({engine.brain.n_neurons} neurons) | target={engine.hz:g} Hz", flush=True)
    print("No flight or RC commands. Preview uses ground telemetry, not a simulated airborne position.", flush=True)
    print("Keep the scene still during SETTLING; then slowly move an object across/toward the camera.", flush=True)

    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.WARNING)
    cv2 = None
    window = "CNSBridge ground shadow - NO MOTION - Q / Escape to exit"
    preview = None
    try:
        if not args.no_window:
            import cv2

            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            def preview(frame, row):
                canvas = frame.copy()
                lines = [
                    f"{row['phase'].upper()} | GROUND ONLY - COMMANDS NOT SENT",
                    " ".join(f"{axis}={row[f'cmd_{axis}']:+.2f}" for axis in AXES),
                    f"processing={row['processing_ms']:.1f} ms | interval={row['tick_interval_ms']:.1f} ms",
                ]
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 105), (0, 0, 0), -1)
                for i, line in enumerate(lines):
                    cv2.putText(canvas, line, (12, 26 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)
                cv2.imshow(window, canvas)
                return cv2.waitKey(1) & 0xFF not in (ord("q"), 27) and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1

        return run_shadow(Tello(retry_count=1), engine, args.seconds, log_path, preview)
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
