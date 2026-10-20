"""Deterministic environment replay and independent action/metric arithmetic."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from ..control.body import BodyVelocity
from ..drones.follow_env import FollowEnvironment, limit_follow_action
from ..drones.follow_scenes import FollowScene
from ..senses.target import TargetCamera, detect_target
from .evaluation_contract import normalized
from .follow_task import FollowPolicyInput, _errors, _metrics, visual_teacher, zero_policy
from .training_contract import array_digest, require

AXES = ("vx", "vy", "vz", "wx", "wy", "wz")


def environment_for(run, episode):
    return FollowEnvironment(FollowScene(**episode["scene"]), TargetCamera(**run["protocol"]["camera"]),
                             dt_s=run["protocol"]["dt_s"], response_tau_s=run["protocol"]["response_tau_s"])


def environment_summary(environment, frame):
    return normalized({"next_frame": frame, "truth": asdict(environment.truth()), "done": environment.done,
                       "next_image": array_digest(environment.frame())})


def vector(action):
    return np.asarray([getattr(action, name) for name in AXES], dtype=np.float64)


def action_from(values):
    require(values.shape == (6,) and values.dtype == np.float64 and np.isfinite(values).all(), "Invalid six-axis action")
    require(np.all(values[[1, 3, 4]] == 0), "Unsupported action axes")
    return BodyVelocity(**dict(zip(AXES, values.tolist())))


def baseline_action(name, frame, environment):
    observation = detect_target(frame, environment.camera)
    input_ = FollowPolicyInput(frame, observation, environment.camera, environment.time_s)
    return visual_teacher(input_, desired_distance_m=environment.scene.desired_distance_m) if name == "direct_visual" else zero_policy(input_)


def step_record(environment, raw, index):
    require(not environment.done, "Cannot advance a terminal environment")
    action_from(vector(raw))  # Reject nonfinite/unsupported axes before any clipping.
    observation, truth = detect_target(environment.frame(), environment.camera), environment.truth()
    limited = limit_follow_action(raw)
    intervention = None
    if not observation.valid:
        limited, intervention = BodyVelocity(), f"lost_target:{observation.reason}"
    elif limited != raw:
        intervention = "action_clipped"
    errors = _errors(truth, environment.camera, environment.scene.desired_distance_m)
    qualified = (observation.valid and errors["x_error"] is not None and abs(errors["x_error"]) <= .15 and abs(errors["y_error"]) <= .15
                 and abs(errors["distance_error_m"]) <= .25)
    outcome = environment.step(limited)
    return normalized({"index": index, "time_s": truth.time_s, "observation": asdict(observation),
                       "raw_action": asdict(raw), "executed_action": asdict(outcome.executed),
                       "intervention": intervention, "truth": asdict(truth), "next_truth": asdict(environment.truth()),
                       "errors": errors, "qualified": bool(qualified), "done": outcome.done})


def readout_actions(features, parameters, identity):
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == identity["features"]["width"]
            and np.isfinite(features).all(), "Invalid complete DN features")
    cfg = identity["features"]
    x = (features - np.float32(cfg["offset_mv"])) / np.float32(cfg["scale_mv"])
    return np.tanh(x @ parameters["readout_weight"].T + parameters["readout_bias"]) * np.asarray(identity["readout"]["scales"], np.float32)


def replay_chunk(environment, values, records, run, episode, start, parameters):
    count = len(records)
    require(type(start) is int and 0 < count <= 10, "Invalid chunk frame count")
    neural = parameters is not None
    expected = {"frames", "indices", "time_s", "raw_actions", "executed_actions"} | ({"features_mv"} if neural else set())
    require(set(values) == expected, "Evidence array inventory differs")
    camera = environment.camera
    declarations = {"frames": ((count, camera.height, camera.width, 3), np.uint8), "indices": ((count,), np.int64),
                    "time_s": ((count,), np.float64), "raw_actions": ((count, 6), np.float64),
                    "executed_actions": ((count, 6), np.float64)}
    for key, (shape, dtype) in declarations.items():
        require(values[key].shape == shape and values[key].dtype == dtype and np.isfinite(values[key]).all(), f"Invalid {key}")
    require(np.array_equal(values["indices"], np.arange(start, start + count, dtype=np.int64)), "Repeated/skipped frames")
    if neural:
        require(values["features_mv"].shape == (count, run["policy"]["features"]["width"]), "Incomplete DN rows")
        recomputed = readout_actions(values["features_mv"], parameters, run["policy"])
        require(np.allclose(values["raw_actions"][:, [0, 2, 5]], recomputed, rtol=1e-6, atol=1e-7), "DN to action conversion differs")
    for offset, record in enumerate(records):
        require(type(record.get("index")) is int and type(record.get("time_s")) is float, "Trajectory clocks require exact numeric types")
        require(not environment.done and np.array_equal(values["frames"][offset], environment.frame()), "Frame differs from closed-loop replay")
        require(values["time_s"][offset] == environment.time_s, "Environment time mismatch")
        raw = action_from(values["raw_actions"][offset])
        if not neural:
            require(raw == baseline_action(episode["condition"], values["frames"][offset], environment), "Baseline action mismatch")
        expected_record = step_record(environment, raw, start + offset)
        require(record == expected_record, "Trajectory/detection/supervision/truth differs from independent replay")
        require(np.array_equal(values["executed_actions"][offset], vector(BodyVelocity(**record["executed_action"]))),
                "Executed action array differs")
    return environment


def metrics(records, planned=80):
    return _metrics(records, planned) if records else None
