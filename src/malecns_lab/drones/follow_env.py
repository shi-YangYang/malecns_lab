"""Three-axis target-velocity simulation, not a calibrated Tello rigid-body model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from malecns_lab.control.body import BodyVelocity
from malecns_lab.drones.follow_scenes import FollowScene
from malecns_lab.senses.target import TargetCamera


@dataclass(frozen=True)
class FollowTruth:
    """Privileged simulator state: only teacher/scoring code may consume this."""

    time_s: float
    drone_position_m: tuple[float, float, float]
    drone_yaw_rad: float
    velocity_flu: tuple[float, float, float]  # vx, vz, wz
    target_position_m: tuple[float, float, float]
    target_body_m: tuple[float, float, float]  # forward, left, up
    distance_m: float
    occluded: bool
    failure: str | None


@dataclass(frozen=True)
class FollowStep:
    time_s: float
    executed: BodyVelocity
    clipped: bool
    done: bool
    failure: str | None


def limit_follow_action(action: BodyVelocity) -> BodyVelocity:
    """Only vx/vz/wz are supported; all inputs are SI FLU target velocities."""
    if not isinstance(action, BodyVelocity):
        raise TypeError("policy must return BodyVelocity")
    if any(getattr(action, name) != 0 for name in ("vy", "wx", "wy")):
        raise ValueError("following environment supports vx/vz/wz only")
    return BodyVelocity(vx=float(np.clip(action.vx, -0.3, 0.3)),
                        vz=float(np.clip(action.vz, -0.2, 0.2)),
                        wz=float(np.clip(action.wz, -0.5, 0.5)))


class FollowEnvironment:
    """Camera observations and privileged truth deliberately have separate APIs.

    World x is initial forward, y is left, z is up. The marker remains on a
    world-y/z plane at its independently moving x coordinate; it never billboards
    toward the drone. Attitude stabilization is assumed. Motion uses first-order
    velocity response with analytical average velocity and midpoint yaw.
    """

    def __init__(
        self, scene: FollowScene, camera: TargetCamera = TargetCamera(), *,
        dt_s: float = 0.1, response_tau_s: float = 0.25,
    ) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (dt_s, response_tau_s)):
            raise ValueError("dt and response time constant must be finite and positive")
        self.scene, self.camera = scene, camera
        self.dt_s, self.response_tau_s = dt_s, response_tau_s
        self.reset()

    def reset(self) -> None:
        self.time_s = 0.0
        self._position = np.array((0.0, 0.0, 1.0))
        self._yaw = 0.0
        self._velocity = np.zeros(3)
        self._failure: str | None = None
        self._ticks = 0
        yy, xx = np.indices((self.camera.height, self.camera.width))
        rng = np.random.default_rng(self.scene.seed)
        gray = np.full(xx.shape, 32.0) + rng.integers(-4, 5, xx.shape)
        if self.scene.background == "stripes":
            gray += (xx // 11 % 2) * 24
        elif self.scene.background == "checker":
            gray += ((xx // 13 + yy // 13) % 2) * 30
        self._background = np.repeat((gray * self.scene.illumination).astype(np.uint8)[..., None], 3, axis=2)

    @property
    def done(self) -> bool:
        return self._failure is not None or self.time_s >= self.scene.duration_s - 1e-10

    def _to_camera(self, world_points: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = world_points - self._position
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        return np.stack((c * delta[..., 0] + s * delta[..., 1],
                         -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), axis=-1)

    def truth(self) -> FollowTruth:
        target = self.scene.target_position(self.time_s)
        body = self._to_camera(target)
        occluded = self.scene.occlusion_s is not None and self.scene.occlusion_s[0] <= self.time_s < self.scene.occlusion_s[1]
        return FollowTruth(self.time_s, tuple(self._position), self._yaw, tuple(self._velocity),
                           tuple(target), tuple(body), float(np.linalg.norm(body)), occluded, self._failure)

    def frame(self) -> NDArray[np.uint8]:
        """Render a new BGR frame. Projection uses the four actual square corners."""
        image = self._background.copy()
        half = self.camera.marker_size_m / 2
        corners = self.scene.target_position(self.time_s) + np.array((
            (0, -half, -half), (0, half, -half), (0, half, half), (0, -half, half),
        ))
        body = self._to_camera(corners)
        if np.any(body[:, 0] <= 0.05):
            return image
        cx, cy = self.camera.center
        points = np.stack((cx - self.camera.focal_px * body[:, 1] / body[:, 0],
                           cy - self.camera.focal_px * body[:, 2] / body[:, 0]), axis=-1)
        yy, xx = np.indices(image.shape[:2])
        crosses = []
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            crosses.append((b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]))
        cross = np.stack(crosses)
        mask = np.all(cross >= 0, axis=0) | np.all(cross <= 0, axis=0)
        image[mask] = np.array((18, 18, 235)) * self.scene.illumination
        interval = self.scene.occlusion_s
        if interval is not None and interval[0] <= self.time_s < interval[1]:
            center = points.mean(axis=0)
            size = np.ptp(points, axis=0) * self.scene.occlusion_fraction
            cover = (np.abs(xx - center[0]) <= size[0] / 2) & (np.abs(yy - center[1]) <= size[1] / 2)
            image[cover] = int(75 * self.scene.illumination)
        return image

    def step(self, action: BodyVelocity) -> FollowStep:
        if self.done:
            raise RuntimeError("episode is complete; reset before stepping")
        executed = limit_follow_action(action)
        dt = min(self.dt_s, self.scene.duration_s - self.time_s)
        requested = np.array((executed.vx, executed.vz, executed.wz))
        decay = math.exp(-dt / self.response_tau_s)
        average = requested + (self._velocity - requested) * self.response_tau_s * (1 - decay) / dt
        self._velocity = requested + (self._velocity - requested) * decay
        yaw_middle = self._yaw + average[2] * dt / 2
        self._position += np.array((math.cos(yaw_middle) * average[0],
                                    math.sin(yaw_middle) * average[0], average[1])) * dt
        self._yaw = (self._yaw + average[2] * dt + math.pi) % (2 * math.pi) - math.pi
        self._ticks += 1
        self.time_s = min(self._ticks * self.dt_s, self.scene.duration_s)
        if self._position[2] <= 0.15:
            self._failure = "ground_collision"
        elif abs(self._position[0]) >= 5 or abs(self._position[1]) >= 5 or self._position[2] >= 3:
            self._failure = "out_of_bounds"
        elif self.truth().distance_m <= 0.2:
            self._failure = "target_collision"
        return FollowStep(self.time_s, executed, executed != action, self.done, self._failure)
