"""Image/physics/sign/leakage checks; no neural model or hardware is exercised."""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from cnsbridge.control.body import BodyVelocity
from cnsbridge.drones.follow_env import FollowEnvironment, limit_follow_action
from cnsbridge.drones.follow_scenes import FollowScene, scene_registry
from cnsbridge.learning.follow_task import FollowPolicyInput, rollout, truth_teacher, visual_teacher, zero_policy
from cnsbridge.senses.target import TargetCamera, detect_target, target_mask


def static_scene(**kwargs):
    return FollowScene("pilot-static", "pilot", "static", 7, **kwargs)


def test_camera_marker_pixel_contract_and_invalid_observations():
    camera = TargetCamera()
    frame = FollowEnvironment(static_scene()).frame()
    observation = detect_target(frame, camera)
    assert observation.valid and observation.detected
    assert observation.x_error < 0 < observation.y_error
    assert observation.distance_m == pytest.approx(np.linalg.norm([1.95, 0.2, 0.15]), abs=0.15)
    with pytest.raises(FrozenInstanceError):
        observation.valid = False
    assert detect_target(np.zeros_like(frame)).reason == "not_detected"
    assert detect_target(frame.astype(float)).reason == "invalid_bgr_uint8"
    assert detect_target(frame[:, :-1]).reason == "invalid_image_shape"
    assert detect_target(None).reason == "invalid_bgr_uint8"
    for invalid in (None, np.zeros_like(frame), frame.astype(float)):
        obs = detect_target(invalid)
        assert not obs.valid and obs.x_error is None and obs.distance_m is None
    clipped = np.zeros_like(frame)
    clipped[40:60, :20, 2] = 220
    assert detect_target(clipped).reason == "edge_clipped"
    incomplete = np.zeros_like(frame)
    incomplete[40:60, 60:80, 2] = 220
    incomplete[44:56, 64:76] = 0
    assert detect_target(incomplete).reason == "incomplete_marker"
    ambiguous = np.zeros_like(frame)
    ambiguous[40:60, 30:50, 2] = ambiguous[40:60, 90:110, 2] = 220
    assert detect_target(ambiguous).reason == "ambiguous_markers"


def test_mask_preserves_positions_and_never_uses_truth():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[20:40, 20:40, 2] = 235
    left = detect_target(frame)
    shifted = np.roll(frame, 80, axis=1)
    right = detect_target(shifted)
    assert left.valid and right.valid and left.x_error < 0 < right.x_error
    mask = target_mask(frame)
    assert mask.shape == (12, 16) and mask[:, :8].sum() > 0 and mask[:, 8:].sum() == 0
    assert not mask.flags.writeable
    with pytest.raises(ValueError, match="invalid_bgr_uint8"):
        target_mask(None)


@pytest.mark.parametrize("action,axis", [
    (BodyVelocity(vx=0.3), "forward"), (BodyVelocity(vz=0.2), "vertical"), (BodyVelocity(wz=0.3), "yaw"),
])
def test_actions_change_next_camera_observation_with_flu_sign(action, axis):
    environment = FollowEnvironment(static_scene(initial_offset_m=(0.1, 0.45, 0.3)))
    before = detect_target(environment.frame())
    original_frame = environment.frame()
    # One second exceeds pixel quantization at the 160x120 camera resolution.
    for _ in range(10):
        environment.step(action)
    after = detect_target(environment.frame())
    assert before.valid and after.valid
    assert not np.array_equal(original_frame, environment.frame())
    if axis == "forward":
        assert after.apparent_height > before.apparent_height
    elif axis == "vertical":
        assert abs(after.y_error) < abs(before.y_error)
    else:
        assert abs(after.x_error) < abs(before.x_error)
        assert environment.truth().drone_yaw_rad > 0


def test_seed_replay_and_target_motion_independent_of_drone_actions():
    scene = next(scene for scene in scene_registry() if scene.scene_id == "train-sine-combined")
    first, second, different = (FollowEnvironment(scene) for _ in range(3))
    for _ in range(8):
        action = BodyVelocity(vx=0.2, vz=0.1, wz=0.15)
        first.step(action)
        second.step(action)
        different.step(BodyVelocity())
        np.testing.assert_array_equal(first.frame(), second.frame())
        assert first.truth() == second.truth()
        assert first.truth().target_position_m == different.truth().target_position_m
    first.reset()
    np.testing.assert_array_equal(first.frame(), FollowEnvironment(scene).frame())


def test_action_bounds_lag_failures_and_terminal_episode():
    action = limit_follow_action(BodyVelocity(vx=5, vz=-4, wz=6))
    assert action == BodyVelocity(vx=0.3, vz=-0.2, wz=0.5)
    for forbidden in (BodyVelocity(vy=0.1), BodyVelocity(wx=0.1), BodyVelocity(wy=0.1)):
        with pytest.raises(ValueError, match="vx/vz/wz"):
            limit_follow_action(forbidden)
    environment = FollowEnvironment(static_scene(duration_s=0.15))
    step = environment.step(BodyVelocity(vx=10))
    assert step.clipped and 0 < environment.truth().velocity_flu[0] < 0.3
    assert 0 < environment.truth().drone_position_m[0] < 0.03
    environment.step(BodyVelocity())
    assert environment.time_s == 0.15 and environment.done
    with pytest.raises(RuntimeError, match="complete"):
        environment.step(BodyVelocity())
    collision = FollowEnvironment(static_scene(duration_s=10))
    while not collision.done:
        collision.step(BodyVelocity(vz=-0.2))
    assert collision.truth().failure == "ground_collision"


def test_occlusion_and_out_of_view_are_invalid_and_do_not_score_success():
    scene = static_scene(duration_s=2, occlusion_s=(0.5, 1.0))
    result = rollout(scene, lambda _: BodyVelocity(vx=0.2), policy_name="constant_forward")
    occluded = [record for record in result.records if record["truth"]["occluded"]]
    assert occluded and all(not record["observation"]["valid"] for record in occluded)
    assert all(not record["qualified"] and record["intervention"].startswith("lost_target") for record in occluded)
    assert all(record["raw_policy_action"]["vx"] == 0.2 and record["executed_action"]["vx"] == 0 for record in occluded)
    assert result.metrics["lost_target_episodes"] == [{"lost_at_s": 0.5, "recovered_at_s": 1.0, "duration_s": 0.5}]
    outside = rollout(static_scene(duration_s=0.3, initial_offset_m=(0, 4, 0)), zero_policy, policy_name="zero")
    assert outside.metrics["valid_observation_fraction"] == 0
    assert outside.metrics["qualified_fraction"] == 0
    assert outside.metrics["success"] is False


def test_policy_boundary_logs_raw_executed_teacher_and_scoring_separately():
    seen = []

    def policy(input_):
        seen.append(input_)
        assert {field.name for field in fields(input_)} == {"frame", "observation", "camera", "time_s"}
        assert not input_.frame.flags.writeable
        return BodyVelocity(vx=1)

    result = rollout(static_scene(duration_s=0.2), policy, policy_name="deliberately_saturated")
    assert len(seen) == 2 and isinstance(seen[0], FollowPolicyInput)
    record = result.records[0]
    assert record["raw_policy_action"]["vx"] == 1
    assert record["executed_action"]["vx"] == 0.3
    assert record["intervention"] == "action_clipped"
    assert record["teacher_source"] == "simulator_truth"
    assert record["next_truth"]["drone_position_m"] != record["truth"]["drone_position_m"]
    assert asdict(truth_teacher(FollowEnvironment(static_scene()).truth())) == record["teacher_action"]
    with TemporaryDirectory() as directory:
        output = Path(directory) / "rollout"
        result.save(output)
        summary = json.loads((output / "summary.json").read_text())
        assert summary["metadata"]["action_frame"] == "body_flu"
        assert len((output / "trajectory.jsonl").read_text().splitlines()) == 2
        with pytest.raises(FileExistsError):
            result.save(output)


def test_registered_scene_splits_are_whole_distinct_families_and_cover_three_axes():
    registry = scene_registry()
    assert len({scene.scene_id for scene in registry}) == len(registry)
    assert len({scene.seed for scene in registry}) == len(registry)
    families = {split: {scene.family for scene in registry if scene.split == split} for split in ("train", "val", "test")}
    assert not families["train"] & families["val"]
    assert not families["train"] & families["test"]
    assert not families["val"] & families["test"]
    for split in families:
        selected = [scene for scene in registry if scene.split == split]
        assert all(any(scene.amplitude_m[axis] != 0 for scene in selected) for axis in range(3))
    with pytest.raises(ValueError):
        replace(registry[0], split="random_frames")


def test_visual_baseline_improves_closed_loop_over_zero_on_training_scene():
    scene = next(scene for scene in scene_registry() if scene.scene_id == "train-sine-combined")
    zero = rollout(scene, zero_policy, policy_name="zero")
    visual = rollout(scene, visual_teacher, policy_name="visual_baseline")
    for metric in ("horizontal_mae", "vertical_mae", "distance_mae_m"):
        assert visual.metrics[metric] < zero.metrics[metric]
    assert all(count > 0 for count in visual.metrics["axis_nonzero_samples"].values())
    assert visual.metrics["failure"] is None
