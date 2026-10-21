"""Registered yaw-learning scenes, causal teacher, and held-out scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np
from .yaw_env import DT_S, YawEnvironment, episode_metrics

WARMUP_FRAMES = 40
ACTION_LIMIT = .35


@dataclass(frozen=True)
class YawScene:
    name: str
    split: str
    texture_seed: int
    amplitude_rad_s: float
    onset_s: float
    pulse_s: float
    initial_yaw_rad: float
    observation_s: float

    def __post_init__(self):
        if self.split not in ("train", "validation", "test") or not self.name:
            raise ValueError("Invalid scene identity")
        if type(self.texture_seed) is not int or self.texture_seed < 0:
            raise ValueError("Invalid texture seed")
        for key in ("amplitude_rad_s", "onset_s", "pulse_s", "initial_yaw_rad", "observation_s"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError("Scene numbers must be finite")
        for key in ("onset_s", "pulse_s", "observation_s"):
            value = getattr(self, key)
            if value < 0 or not math.isclose(value / DT_S, round(value / DT_S), abs_tol=1e-9):
                raise ValueError("Scene times must be nonnegative whole control intervals")
        if self.observation_s <= 0 or self.onset_s + self.pulse_s > self.observation_s:
            raise ValueError("Disturbance must fit observation interval")

    @property
    def observation_frames(self):
        return round(self.observation_s / DT_S)

    @property
    def total_frames(self):
        return WARMUP_FRAMES + self.observation_frames

    def disturbance(self, frame_index):
        if type(frame_index) is not int or not 0 <= frame_index < self.total_frames:
            raise ValueError("Frame outside scene")
        relative = frame_index - WARMUP_FRAMES
        begin, end = round(self.onset_s / DT_S), round((self.onset_s + self.pulse_s) / DT_S)
        return self.amplitude_rad_s if begin <= relative < end else 0.


def scenes(split):
    if split == "train":
        rows = [(101, .20, 0., .50, 0., 2.), (101, -.20, .25, .50, 0., 2.),
                (102, .35, .50, .50, 0., 2.), (102, -.35, .75, .50, 0., 2.)]
    elif split == "validation":
        rows = [(201, sign * .275, .40, .50, .15, 2.) for sign in (1, -1)]
    elif split == "test":
        rows = [(301, sign * .225, .35, .40, -.15, 3.) for sign in (1, -1)]
        rows += [(302, sign * .325, .65, .60, -.15, 3.) for sign in (1, -1)]
        rows += [(303, 0., 0., 0., -.15, 3.)]
    else:
        raise ValueError("Unknown scene split")
    return tuple(YawScene(f"{split}-{index:02d}", split, *row) for index, row in enumerate(rows))


class YawTaskEnvironment(YawEnvironment):
    """Independent task instance; preserve Spec012's registered environment."""
    def __init__(self, scene: YawScene):
        if not isinstance(scene, YawScene):
            raise TypeError("YawScene required")
        super().__init__()
        self.scene = scene
        self._yaw = scene.initial_yaw_rad
        self._phase = np.random.default_rng(scene.texture_seed).uniform(-np.pi, np.pi, size=4)
        self.frame_index = 0
        self.observed_rate = 0.

    @property
    def teacher_action(self):
        """Only motion already completed and visible in the current frame."""
        return float(np.clip(-2 * self.observed_rate, -ACTION_LIMIT, ACTION_LIMIT))

    def advance(self, action):
        disturbance = self.scene.disturbance(self.frame_index)
        previous = self._yaw
        row = super().step(action, disturbance)
        self.observed_rate = (self._yaw - previous) / DT_S
        self.frame_index += 1
        return {**row, "disturbance_wz_rad_s": disturbance}

    def snapshot(self):
        return {"scene": asdict(self.scene), "frame_index": self.frame_index, "yaw_rad": self._yaw,
                "actuator_wz_rad_s": self._actuator_wz, "disturbance_state_rad_s": self._disturbance_wz,
                "observed_rate_rad_s": self.observed_rate}

    def restore(self, saved):
        if set(saved) != {"scene", "frame_index", "yaw_rad", "actuator_wz_rad_s", "disturbance_state_rad_s", "observed_rate_rad_s"}:
            raise ValueError("Environment snapshot fields differ")
        if saved["scene"] != asdict(self.scene) or type(saved["frame_index"]) is not int or not 0 <= saved["frame_index"] <= self.scene.total_frames:
            raise ValueError("Environment scene/clock differs")
        if not all(isinstance(saved[k], (int, float)) and math.isfinite(saved[k]) for k in saved if k not in ("scene", "frame_index")):
            raise ValueError("Environment state must be finite")
        self.frame_index = saved["frame_index"]
        self._yaw, self._actuator_wz = saved["yaw_rad"], saved["actuator_wz_rad_s"]
        self._disturbance_wz, self.observed_rate = saved["disturbance_state_rad_s"], saved["observed_rate_rad_s"]


def score_episode(scene, actual_rates, commands):
    rates, actions = np.asarray(actual_rates, np.float64), np.asarray(commands, np.float64)
    if rates.shape != (scene.total_frames,) or actions.shape != rates.shape or not np.isfinite(actions).all():
        raise ValueError("Complete finite scene required for metrics")
    if np.any(np.abs(actions) > ACTION_LIMIT + 1e-7) or np.any(actions[:WARMUP_FRAMES] != 0):
        raise ValueError("Action limit or warmup violated")
    result = episode_metrics(rates, WARMUP_FRAMES, DT_S, scene.onset_s + scene.pulse_s)
    result["tail_rms_rad_s"] = float(np.sqrt(np.mean(rates[-10:] ** 2)))
    result["max_command_rad_s"] = float(np.abs(actions).max())
    return result


def control_gate(rows, *, require_static):
    """Registered performance test; completion is distinct from passing it."""
    pairs = {}
    for row in rows:
        pairs.setdefault(row["scene"]["name"], {})[row["mode"]] = row
    checks, improvements, by_sign = {}, [], {1: [], -1: []}
    static_count = 0
    for name, pair in sorted(pairs.items()):
        if "joint" not in pair or "zero" not in pair:
            raise ValueError("Every scene requires paired joint and zero records")
        a, z = pair["joint"], pair["zero"]
        if a["scene"] != z["scene"]:
            raise ValueError("Pair scene identity differs")
        m, baseline = a["metrics"], z["metrics"]
        if a["scene"]["amplitude_rad_s"] == 0:
            static_count += 1
            checks[name] = m["rms_rotation_rad_s"] <= .02 and m["peak_rotation_rad_s"] <= .05
        else:
            if baseline["rms_rotation_rad_s"] <= 0:
                raise ValueError("Nonzero disturbance requires nonzero baseline")
            improvement = 1 - m["rms_rotation_rad_s"] / baseline["rms_rotation_rad_s"]
            improvements.append(improvement)
            by_sign[1 if a["scene"]["amplitude_rad_s"] > 0 else -1].append(improvement)
            checks[name] = (improvement >= -.05 and m["peak_rotation_rad_s"] <= 1.10 * baseline["peak_rotation_rad_s"]
                            and m["tail_rms_rad_s"] <= .03)
    aggregate = bool(improvements and all(by_sign.values()) and np.mean(improvements) >= .20
                     and all(np.mean(v) > 0 for v in by_sign.values()))
    static_ok = not require_static or static_count == 1
    return {"passed": bool(aggregate and static_ok and all(checks.values())), "scene_checks": checks,
            "mean_rms_improvement": float(np.mean(improvements)) if improvements else None,
            "direction_improvement": {str(k): float(np.mean(v)) if v else None for k, v in by_sign.items()},
            "static_required": require_static, "static_present": static_count, "aggregate_passed": aggregate}
