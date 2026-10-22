"""World-state projection for a virtual marker, never a real camera observation.

There is no dynamics model here. The caller supplies a measured pose or labels
the input as an offline probe; only projected pixels may enter the neural policy.
"""
from __future__ import annotations

import math

import numpy as np

CAMERA = {"width": 160, "height": 120, "horizontal_fov_deg": 72.0, "marker_size_m": 0.2}
PROBE_DISTANCES = (2.0, 1.5, 1.0)


def marker_frame(position_m, orientation_xyzw, target_world_m):
    """Project a world-y/z square into FLU camera pixels using all four corners."""
    position = np.asarray(position_m, dtype=np.float64)
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    target = np.asarray(target_world_m, dtype=np.float64)
    if (position.shape != (3,) or target.shape != (3,) or quaternion.shape != (4,)
            or not all(np.isfinite(value).all() for value in (position, quaternion, target))):
        raise ValueError("Finite position, target and xyzw quaternion are required")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Camera orientation must be a unit quaternion")
    x, y, z, w = quaternion / norm
    # Rotation maps camera/body FLU vectors into world coordinates.
    rotation = np.array(((1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                         (2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)),
                         (2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y))))
    image = np.full((CAMERA["height"], CAMERA["width"], 3), 32, dtype=np.uint8)
    half = CAMERA["marker_size_m"] / 2
    corners = target + np.array(((0, -half, -half), (0, half, -half),
                                  (0, half, half), (0, -half, half)))
    body = (corners - position) @ rotation
    if np.any(body[:, 0] <= .05):
        return image
    focal = CAMERA["width"] / (2 * math.tan(math.radians(CAMERA["horizontal_fov_deg"]) / 2))
    center = ((CAMERA["width"] - 1) / 2, (CAMERA["height"] - 1) / 2)
    points = np.stack((center[0] - focal*body[:, 1]/body[:, 0],
                       center[1] - focal*body[:, 2]/body[:, 0]), axis=-1)
    yy, xx = np.indices(image.shape[:2])
    crosses = np.stack([(b[0]-a[0])*(yy-a[1]) - (b[1]-a[1])*(xx-a[0])
                        for a, b in zip(points, np.roll(points, -1, axis=0))])
    mask = np.all(crosses >= 0, axis=0) | np.all(crosses <= 0, axis=0)
    # A square viewed exactly edge-on has no projected area.
    area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], -1))))
    if area > 1e-10:
        image[mask] = (18, 18, 235)
    return image


def probe_inputs():
    """The preregistered offline sequence, with no scene dynamics or resets."""
    return [{"sample_id": f"center-{distance:.1f}m", "source": "offline_static_input_probe",
             "camera_position_m": [0.0, 0.0, .65], "camera_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
             "target_world_m": [distance, 0.0, .65], "distance_m": distance}
            for distance in PROBE_DISTANCES]
