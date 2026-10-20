"""Marked-target pilot teachers, isolated policy API, rollout and scoring.

This module exercises an environment. A teacher passing these checks is not
evidence that a MaleCNS policy has learned anything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from malecns_lab.control.body import BodyVelocity
from malecns_lab.drones.follow_env import FollowEnvironment, FollowTruth, limit_follow_action
from malecns_lab.drones.follow_scenes import FollowScene, scene_registry
from malecns_lab.senses.target import TargetCamera, TargetObservation, detect_target


PROTOCOL_ID = "cnsbridge.marked_follow.pilot.v1"
POSITION_TOLERANCE = 0.15
DISTANCE_TOLERANCE_M = 0.25
SUCCESS_FRACTION = 0.8


@dataclass(frozen=True)
class FollowPolicyInput:
    """Deployment-visible inputs only. Frame is a read-only owned BGR image."""

    frame: NDArray[np.uint8]
    observation: TargetObservation
    camera: TargetCamera
    time_s: float


FollowPolicy = Callable[[FollowPolicyInput], BodyVelocity]


def zero_policy(_input: FollowPolicyInput) -> BodyVelocity:
    return BodyVelocity()


def visual_teacher(input_: FollowPolicyInput, *, desired_distance_m: float = 1.5) -> BodyVelocity:
    """Conventional image-based baseline, never a neural model component."""
    obs = input_.observation
    if not obs.valid:
        return BodyVelocity()
    assert obs.x_error is not None and obs.y_error is not None and obs.distance_m is not None
    horizontal = obs.x_error * input_.camera.width / (2 * input_.camera.focal_px)
    vertical = obs.y_error * input_.camera.height / (2 * input_.camera.focal_px)
    depth = obs.distance_m / math.sqrt(1 + horizontal ** 2 + vertical ** 2)
    return BodyVelocity(vx=0.7 * (obs.distance_m - desired_distance_m),
                        vz=0.8 * depth * vertical, wz=-1.8 * math.atan(horizontal))


def truth_teacher(truth: FollowTruth, *, desired_distance_m: float = 1.5) -> BodyVelocity:
    """Privileged training label, explicitly separate from the policy signature."""
    forward, left, up = truth.target_body_m
    return BodyVelocity(vx=0.7 * (truth.distance_m - desired_distance_m),
                        vz=0.8 * up, wz=1.8 * math.atan2(left, forward))


@dataclass(frozen=True)
class FollowRollout:
    metadata: dict
    metrics: dict
    records: tuple[dict, ...]

    def save(self, directory: str | Path) -> None:
        """New directories only: never overwrite previous experiment evidence."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps({"metadata": self.metadata, "metrics": self.metrics}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (output / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")


def _errors(truth: FollowTruth, camera: TargetCamera, desired_distance_m: float) -> dict:
    forward, left, up = truth.target_body_m
    return {
        "x_error": -camera.focal_px * left / forward / (camera.width / 2) if forward > 0 else None,
        "y_error": camera.focal_px * up / forward / (camera.height / 2) if forward > 0 else None,
        "distance_error_m": truth.distance_m - desired_distance_m,
    }


def _metrics(records: list[dict], planned_steps: int) -> dict:
    def mae(name: str) -> float | None:
        values = [abs(record["errors"][name]) for record in records if record["errors"][name] is not None]
        return float(np.mean(values)) if values else None

    valid = sum(record["observation"]["valid"] for record in records)
    qualified = sum(record["qualified"] for record in records)
    actions = np.array([[record["executed_action"][axis] for axis in ("vx", "vz", "wz")] for record in records])
    failure = next((record["next_truth"]["failure"] for record in records if record["next_truth"]["failure"]), None)
    losses, loss_start = [], None
    for record in records:
        if not record["observation"]["valid"] and loss_start is None:
            loss_start = record["time_s"]
        if record["observation"]["valid"] and loss_start is not None:
            losses.append({"lost_at_s": loss_start, "recovered_at_s": record["time_s"],
                           "duration_s": record["time_s"] - loss_start})
            loss_start = None
    if loss_start is not None:
        losses.append({"lost_at_s": loss_start, "recovered_at_s": None, "duration_s": None})
    return {
        "samples": len(records), "planned_samples": planned_steps,
        "observed_fraction": len(records) / planned_steps,
        "horizontal_mae": mae("x_error"), "vertical_mae": mae("y_error"),
        "distance_mae_m": mae("distance_error_m"),
        "projectable_samples": sum(record["errors"]["x_error"] is not None for record in records),
        "valid_observation_fraction": valid / planned_steps,
        "qualified_fraction": qualified / planned_steps,
        "success": qualified / planned_steps >= SUCCESS_FRACTION and failure is None and len(records) == planned_steps,
        "failure": failure,
        "axis_nonzero_samples": dict(zip(("vx", "vz", "wz"), np.count_nonzero(np.abs(actions) > 1e-8, axis=0).tolist())),
        "mean_absolute_action_change": dict(zip(("vx", "vz", "wz"),
                                                np.mean(np.abs(np.diff(actions, axis=0)), axis=0).tolist()
                                                if len(actions) > 1 else [0.0] * 3)),
        "intervention_samples": sum(bool(record["intervention"]) for record in records),
        "occluded_samples": sum(record["truth"]["occluded"] for record in records),
        "lost_target_episodes": losses,
    }


def rollout(
    scene: FollowScene, policy: FollowPolicy, *, policy_name: str,
    camera: TargetCamera = TargetCamera(), dt_s: float = 0.1, response_tau_s: float = 0.25,
    record_frames: Callable[[FollowPolicyInput], None] | None = None,
) -> FollowRollout:
    """Execute true closed loop; policy receives no environment or ground truth.

    record_frames can save or encode the same pixel-only sample for training.
    Truth labels are recorded separately. Invalid detections cause an external
    zero-command intervention; that response is not attributed to policy skill.
    """
    if not policy_name:
        raise ValueError("policy_name must identify the evaluated policy")
    environment = FollowEnvironment(scene, camera, dt_s=dt_s, response_tau_s=response_tau_s)
    records = []
    while not environment.done:
        frame = environment.frame()
        frame.setflags(write=False)
        observation = detect_target(frame, camera)
        input_ = FollowPolicyInput(frame, observation, camera, environment.time_s)
        if record_frames is not None:
            record_frames(input_)
        truth = environment.truth()
        raw = policy(input_)
        limited = limit_follow_action(raw)
        intervention = None
        if not observation.valid:
            limited = BodyVelocity()
            intervention = f"lost_target:{observation.reason}"
        elif limited != raw:
            intervention = "action_clipped"
        step = environment.step(limited)
        errors = _errors(truth, camera, scene.desired_distance_m)
        qualified = (observation.valid and errors["x_error"] is not None
                     and abs(errors["x_error"]) <= POSITION_TOLERANCE
                     and abs(errors["y_error"]) <= POSITION_TOLERANCE
                     and abs(errors["distance_error_m"]) <= DISTANCE_TOLERANCE_M)
        records.append({
            "time_s": truth.time_s, "observation": asdict(observation),
            "raw_policy_action": asdict(raw), "executed_action": asdict(step.executed),
            "teacher_action": asdict(truth_teacher(truth, desired_distance_m=scene.desired_distance_m)),
            "teacher_source": "simulator_truth", "intervention": intervention,
            "truth": asdict(truth), "next_truth": asdict(environment.truth()),
            "errors": errors, "qualified": bool(qualified),
        })
    metadata = {
        "schema": "cnsbridge.follow_rollout.v1", "protocol": PROTOCOL_ID,
        "scene": asdict(scene), "camera": asdict(camera), "policy": policy_name,
        "policy_inputs": ["bgr_frame", "pixel_target_observation", "camera", "time_s"],
        "action_axes": ["vx", "vz", "wz"], "action_units": ["m/s", "m/s", "rad/s"],
        "action_frame": "body_flu", "dt_s": dt_s, "response_tau_s": response_tau_s,
        "thresholds": {"position": POSITION_TOLERANCE, "distance_m": DISTANCE_TOLERANCE_M,
                       "success_fraction": SUCCESS_FRACTION},
        "evidence_scope": "simplified velocity-level simulation; not physical-flight or SNN-learning evidence",
    }
    return FollowRollout(metadata, _metrics(records, math.ceil(scene.duration_s / dt_s - 1e-10)), tuple(records))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--scene", default="train-sine-combined")
    parser.add_argument("--policy", choices=("visual", "zero"), default="visual")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    registry = {scene.scene_id: scene for scene in scene_registry()}
    if args.list_scenes:
        print(json.dumps([asdict(scene) for scene in registry.values()], indent=2))
        return
    if args.scene not in registry:
        parser.error("unknown scene; use --list-scenes")
    if args.out is None:
        parser.error("--out is required for a rollout")
    result = rollout(registry[args.scene], visual_teacher if args.policy == "visual" else zero_policy,
                     policy_name=f"{args.policy}_baseline")
    result.save(args.out)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
