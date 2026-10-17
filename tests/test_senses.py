import numpy as np

from cnsbridge.senses import Retina
from cnsbridge.senses.retina import box_blur


def shifted(tex, dx, dy, scale=1.0):
    H, W = 72, 96
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = 100 + (xx - W / 2) / scale + W / 2 + dx
    sy = 60 + (yy - H / 2) / scale + H / 2 + dy
    return (tex[np.clip(sy.astype(int), 0, 199), np.clip(sx.astype(int), 0, 299)] * 255).astype(np.uint8)


def texture():
    return box_blur(np.random.default_rng(0).random((200, 300)).astype(np.float32), 3)


def test_rightward_motion_channels():
    tex, r = texture(), Retina()
    r.encode(shifted(tex, 0, 0))
    v = r.encode(shifted(tex, -1, 0))  # content moves right
    assert v.eyes["L"].grids["btf"].mean() > 0.4 and v.eyes["L"].grids["ftb"].mean() < 0.1
    assert v.eyes["R"].grids["ftb"].mean() > 0.4 and v.eyes["R"].grids["btf"].mean() < 0.1


def test_upward_motion_channels():
    tex, r = texture(), Retina()
    r.encode(shifted(tex, 0, 0))
    v = r.encode(shifted(tex, 0, 1))
    assert v.eyes["L"].grids["up"].mean() > 0.4 and v.eyes["L"].grids["down"].mean() < 0.1


def test_rotation_is_not_looming_but_approach_is():
    # Numerical optical-flow stimuli, not a flight or room simulation. The
    # following environment uses a screen-fixed background and a solid marker;
    # it cannot provide the textured full-field flow needed by this regression.
    # A fixed textured plane viewed with pure yaw should not look like approach.
    tex = texture()
    rotating, approaching = Retina(), Retina()
    rot = [rotating.encode(planar_view(tex, yaw_rad=0.01 * step)).eyes["L"].grids["loom"].max()
           for step in range(30)]
    app = [approaching.encode(planar_view(tex, distance=1.0 - 0.015 * step)).eyes["L"].grids["loom"].max()
           for step in range(45)]
    assert max(rot) < 0.2
    assert max(app[-15:]) > 0.5


def planar_view(tex, *, yaw_rad=0.0, distance=1.0):
    """Pinhole projection onto a fixed texture plane, with no motion dynamics.

    Plane distance starts at one arbitrary length unit; decreasing it is forward
    translation along the optical axis. Yaw rotates camera rays without moving
    the camera. Both transformations therefore use the same stationary texture.
    """
    height, width = 72, 96
    focal = width / (2 * np.tan(np.deg2rad(82) / 2))
    yy, xx = np.mgrid[0:height, 0:width]
    ray_x = (xx + 0.5 - width / 2) / focal
    ray_y = (yy + 0.5 - height / 2) / focal
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    depth = cosine - sine * ray_x
    sx = tex.shape[1] / 2 + focal * distance * (cosine * ray_x + sine) / depth
    sy = tex.shape[0] / 2 + focal * distance * ray_y / depth
    # All rays stay in front of the plane and inside the source texture. Avoid
    # border clamping that could create false flow or expansion in the fixture.
    assert np.all(depth > 0)
    assert sx.min() >= 0 and sx.max() < tex.shape[1]
    assert sy.min() >= 0 and sy.max() < tex.shape[0]
    return (tex[sy.astype(int), sx.astype(int)] * 255).astype(np.uint8)
