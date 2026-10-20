"""Explicit, immutable scene families for the marked-target pilot protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FollowScene:
    scene_id: str
    split: str
    family: str
    seed: int
    duration_s: float = 8.0
    # Target plane is normal to world x; tuple axes are world x/y/z.
    initial_offset_m: tuple[float, float, float] = (0.45, 0.2, 0.15)
    amplitude_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    period_s: float = 8.0
    background: str = "flat"
    illumination: float = 1.0
    occlusion_s: tuple[float, float] | None = None
    occlusion_fraction: float = 0.65
    desired_distance_m: float = 1.5

    def __post_init__(self) -> None:
        if not self.scene_id or self.split not in {"train", "val", "test", "pilot"}:
            raise ValueError("scene needs an ID and a declared split")
        if self.family not in {"static", "sine", "triangle", "chirp", "excursion"}:
            raise ValueError("unknown trajectory family")
        if self.background not in {"flat", "stripes", "checker"}:
            raise ValueError("unknown background")
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.duration_s, self.period_s, self.illumination, self.desired_distance_m,
        )) or self.illumination > 1:
            raise ValueError("duration, period, distance and illumination must be positive and finite; illumination <= 1")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for vector in (self.initial_offset_m, self.amplitude_m):
            if len(vector) != 3 or any(not math.isfinite(v) for v in vector):
                raise ValueError("scene vectors must contain three finite numbers")
        if not 0 <= self.occlusion_fraction <= 1:
            raise ValueError("occlusion fraction must be in [0, 1]")
        if self.occlusion_s is not None:
            start, end = self.occlusion_s
            if not 0 <= start < end <= self.duration_s:
                raise ValueError("occlusion interval must lie within the episode")

    def target_position(self, time_s: float) -> np.ndarray:
        """Independent of drone or policy. The seed fixes phases, not actions."""
        base = np.array((self.desired_distance_m, 0.0, 1.0)) + self.initial_offset_m
        phase = np.random.default_rng(self.seed).uniform(-0.35, 0.35, size=3)
        angle = 2 * np.pi * time_s / self.period_s * np.array((0.73, 1.0, 0.83)) + phase
        if self.family == "static":
            wave = np.zeros(3)
        elif self.family == "triangle":
            wave = 2 / np.pi * np.arcsin(np.sin(angle))
        elif self.family == "chirp":
            wave = np.sin(angle + 0.045 * time_s ** 2)
        elif self.family == "excursion":
            wave = np.sin(np.pi * np.clip(time_s / self.period_s, 0, 1))
        else:
            wave = np.sin(angle)
        return base + np.asarray(self.amplitude_m) * wave


def scene_registry() -> tuple[FollowScene, ...]:
    """Predeclared pilot split; whole trajectory families/backgrounds are disjoint."""
    scenes = []
    for split, family, background, light, seed in (
        ("train", "sine", "flat", 1.0, 101),
        ("val", "triangle", "stripes", 0.8, 201),
        ("test", "chirp", "checker", 0.65, 301),
    ):
        for index, (axis, amplitude) in enumerate((
            ("forward", (0.25, 0.0, 0.0)), ("yaw", (0.0, 0.3, 0.0)),
            ("vertical", (0.0, 0.0, 0.2)), ("combined", (0.2, 0.25, 0.15)),
        )):
            scenes.append(FollowScene(
                f"{split}-{family}-{axis}", split, family, seed + index,
                amplitude_m=amplitude, background=background, illumination=light,
            ))
    scenes.extend((
        FollowScene("test-chirp-occluded", "test", "chirp", 305,
                    amplitude_m=(0.1, 0.15, 0.1), background="checker", illumination=0.65,
                    occlusion_s=(2.0, 3.0)),
        FollowScene("test-excursion-out-of-view", "test", "excursion", 306,
                    amplitude_m=(0.0, 2.5, 0.0), background="checker", illumination=0.65),
    ))
    return tuple(scenes)
