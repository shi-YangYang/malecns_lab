"""Offline synthetic training example: one readout, aircraft and ground adapters.

This tests the interface, not MaleCNS control or transfer between real devices.
Only planar targets are used so both adapters can execute the same request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cnsbridge.control import (
    DifferentialDriveAdapter,
    FlightCommandAdapter,
)
from cnsbridge.learning import NeuralObservation, NeuralSchema, RidgeMotionDecoder


def targets(rates: np.ndarray) -> np.ndarray:
    """Independent synthetic labels in m/s and rad/s; no old decoder involved."""
    result = np.zeros((len(rates), 6))
    result[:, 0] = 0.006 * (rates[:, 0] - rates[:, 1])
    result[:, 5] = 0.01 * (rates[:, 2] - rates[:, 3])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Save a NEW synthetic model JSON; never overwrites")
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    schema = NeuralSchema(
        ("demo_a", "demo_b", "demo_c", "demo_d"), "synthetic-interface-demo-v1", 50.0
    )
    train = rng.uniform(0.0, 60.0, size=(256, 4))
    heldout = rng.uniform(0.0, 60.0, size=(64, 4))
    decoder = RidgeMotionDecoder.fit(schema, train, targets(train), ridge=0.01)
    error = np.abs(decoder.predict_batch(heldout, schema=schema) - targets(heldout)).mean(axis=0)
    print("SYNTHETIC OFFLINE DEMO | shared decoder; no aircraft or robot connected")
    print(f"Held-out MAE | vx={error[0]:.8f} m/s | wz={error[5]:.8f} rad/s")
    observation = NeuralObservation(schema, dict(zip(schema.feature_names, (50.0, 10.0, 45.0, 15.0))))
    motion = decoder.decode(observation)
    # Example scales/geometry only. These are NOT measured Tello calibration.
    aircraft = FlightCommandAdapter(
        forward_scale_mps=1.0, lateral_scale_mps=1.0,
        vertical_scale_mps=0.5, yaw_scale_rad_s=1.0,
    )
    ground = DifferentialDriveAdapter(
        wheel_radius_m=0.05, track_width_m=0.25, max_wheel_rad_s=12.0,
    )
    print(f"Shared body target | {motion}")
    print(f"Aircraft command | {aircraft.adapt(motion).command}")
    print(f"Ground wheel targets | {ground.adapt(motion).command}")
    print("Positive wz means left: aircraft yaw is negative; right wheel turns faster.")
    if args.out is not None:
        decoder.save(args.out)
        print(f"Synthetic model saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
