"""Visual algorithm contracts only; these images are not brain/task evidence."""

from dataclasses import replace

import numpy as np
import pytest

from malecns_lab.senses import vision_v2
from malecns_lab.senses.vision_v2 import CHANNELS, VisionV2Config, VisionV2Encoder


def image(dx=0.0, dy=0.0, scale=1.0, width=96, height=72):
    y, x = np.mgrid[:height, :width].astype(float)
    x = (x - (width - 1) / 2 - dx) / scale
    y = (y - (height - 1) / 2 - dy) / scale
    luminance = 0.5 + 0.24 * np.sin(0.35 * x + 0.21 * y) + 0.2 * np.cos(0.2 * x - 0.3 * y)
    return np.repeat(np.rint(luminance * 255).astype(np.uint8)[..., None], 3, axis=2)


def test_bgr_initialization_order_and_area_resize():
    encoder = VisionV2Encoder()
    bgr = np.zeros((144, 192, 3), np.uint8)
    bgr[:, :96, 0] = 255  # Blue in the left half, red in the right.
    bgr[:, 96:, 2] = 255
    observation = encoder.encode(bgr, 0.0)
    assert observation.valid and observation.initialization and not observation.segment_stopped
    assert observation.input_cutoff_s == 0.0
    assert CHANNELS == tuple(f"{side}.{name}" for side in ("L", "R")
                             for name in ("brightness", "ftb", "btf", "up", "down", "loom", "loom_speed"))
    np.testing.assert_allclose(observation.features, [0.114, 0, 0, 0, 0, 0, 0, 0.299, 0, 0, 0, 0, 0, 0], atol=1e-12)
    for eye in observation.eyes.values():
        assert eye.valid_count == 0  # Missing flow was not disguised as a measured zero.
        assert np.isnan(eye.flow_px_s).all()
    assert not observation.features.flags.writeable


@pytest.mark.parametrize("dx,dy,left_channel,right_channel", [
    (0.2, 0.0, "btf", "ftb"), (-0.2, 0.0, "ftb", "btf"),
    (0.0, -0.2, "up", "up"), (0.0, 0.2, "down", "down"),
])
def test_direction_and_resized_pixels_per_second(dx, dy, left_channel, right_channel):
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0.0)
    observation = encoder.encode(image(dx, dy), 0.05)
    assert observation.valid and not observation.initialization
    for side, channel in (("L", left_channel), ("R", right_channel)):
        eye = observation.eyes[side]
        assert eye.valid_count >= 40
        flow = eye.flow_px_s[eye.flow_valid].mean(axis=0)
        np.testing.assert_allclose(flow, np.array([dx, dy]) / 0.05, atol=0.35)
        assert observation.features[CHANNELS.index(f"{side}.{channel}")] == pytest.approx(4 / 24, abs=0.025)
        assert abs(eye.expansion_s_inv) < 0.03


def test_static_texture_is_valid_zero_motion():
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0.0)
    result = encoder.encode(image(), 0.05)
    assert result.valid
    np.testing.assert_array_equal(result.features[[i for i in range(14) if i not in (0, 7)]], 0)
    assert all(eye.valid_fraction == 1.0 for eye in result.eyes.values())


def test_real_expansion_updates_each_eye_ema_and_loom():
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0.0)
    result = encoder.encode(image(scale=1.04), 0.05)
    assert result.valid
    for side, eye in result.eyes.items():
        # 4% scale change in 50 ms gives an expansion near 0.8/s, not per grid cell.
        assert 0.55 < eye.expansion_s_inv < 1.05
        assert eye.expansion_ema_s_inv == pytest.approx(eye.expansion_s_inv * (1 - np.exp(-0.5)))
        assert eye.loom > 0.1
        assert eye.loom_speed_s_inv == pytest.approx(eye.loom / 0.05)
        assert result.features[CHANNELS.index(f"{side}.loom_speed")] == pytest.approx(min(1, eye.loom_speed_s_inv / 5))


def test_expansion_uses_pixel_coordinates_and_requires_rank_without_ridge():
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0.0)
    eye = encoder.encode(image(), 0.05).eyes["L"]
    flow = np.stack((3 + 0.4 * eye.centers_px[..., 0], -2 + 0.7 * eye.centers_px[..., 1]), axis=-1)
    known = replace(eye, flow_px_s=flow)
    np.testing.assert_allclose(vision_v2._fit_expansion(known), [0.4, 0.7], atol=1e-12)
    one_row = np.zeros((6, 8), bool)
    one_row[0] = True
    with pytest.raises(vision_v2._MotionFailure, match="unidentifiable_expansion"):
        vision_v2._fit_expansion(replace(known, flow_valid=one_row))
    bad_flow = flow.copy()
    bad_flow[0, 0, 0] = np.nan
    with pytest.raises(vision_v2._MotionFailure, match="nonfinite_expansion"):
        vision_v2._fit_expansion(replace(known, flow_px_s=bad_flow))


@pytest.mark.parametrize("texture", ["flat", "one_dimensional"])
def test_absent_or_degenerate_texture_is_not_reliable_zero(texture):
    if texture == "flat":
        frame = np.full((72, 96, 3), 120, np.uint8)
    else:
        line = np.rint(120 + 80 * np.sin(np.arange(96) * 0.3)).astype(np.uint8)
        frame = np.broadcast_to(line[None, :, None], (72, 96, 3)).copy()
    encoder = VisionV2Encoder()
    assert encoder.encode(frame, 0.0).initialization
    result = encoder.encode(frame, 0.05)
    assert not result.valid and result.features is None
    assert result.segment_stopped and result.invalid_reason == "no_valid_flow"
    assert result.invalid_eye == "L"
    assert result.eyes["L"].valid_count == 0


def test_second_eye_failure_is_atomic_latches_stop_and_reset_restores_all_state():
    encoder = VisionV2Encoder()
    initial = image()
    initial[:, 48:] = 128
    encoder.encode(initial, 0.0)
    old_ema, old_loom = dict(encoder._expansion_ema), dict(encoder._previous_loom)
    # Keep the left half textured but remove all texture in the right pair.
    previous = encoder._previous_image.copy()
    current = image(scale=1.04)
    current[:, 48:] = 128
    result = encoder.encode(current, 0.05)
    assert result.invalid_reason == "no_valid_flow" and result.invalid_eye == "R"
    assert result.features is None and result.eyes["L"].valid_count > 0
    assert encoder.last_committed_time_s == 0.0
    np.testing.assert_array_equal(encoder._previous_image, previous)
    assert encoder._expansion_ema == old_ema and encoder._previous_loom == old_loom
    later = encoder.encode(image(), 0.15)
    assert later.invalid_reason == "segment_stopped" and later.segment_stop_reason == "no_valid_flow"
    encoder.reset()
    fresh = VisionV2Encoder()
    for time_s, frame in ((3.0, image()), (3.05, image(scale=1.04))):
        actual, expected = encoder.encode(frame, time_s), fresh.encode(frame, time_s)
        assert actual.valid
        np.testing.assert_array_equal(actual.features, expected.features)
        assert actual.initialization == expected.initialization
    assert any(value > 0 for value in encoder._expansion_ema.values())
    encoder.reset()
    fresh.reset()
    np.testing.assert_array_equal(encoder.encode(image(), 5.0).features, fresh.encode(image(), 5.0).features)
    assert encoder._previous_loom == {"L": 0.0, "R": 0.0}


def test_partial_valid_grids_are_counted_and_aggregated_without_texture_scaling(monkeypatch):
    original = vision_v2._eye_flow

    def some_cells(*args):
        eye = original(*args)
        if eye.valid_count:
            valid = eye.flow_valid.copy()
            valid[:, ::2] = False
            return replace(eye, flow_valid=valid)
        return eye

    monkeypatch.setattr(vision_v2, "_eye_flow", some_cells)
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0)
    result = encoder.encode(image(dx=0.2), 0.05)
    assert result.valid
    for side, feature in (("L", "btf"), ("R", "ftb")):
        eye = result.eyes[side]
        assert eye.valid_count == 24 and eye.valid_fraction == 0.5
        expected = np.clip(eye.flow_px_s[..., 0][eye.flow_valid] / 24, 0, 1).mean()
        assert result.features[CHANNELS.index(f"{side}.{feature}")] == expected


@pytest.mark.parametrize("frame", [None, np.zeros((72, 96), np.uint8), np.zeros((72, 96, 3), float),
                                   np.zeros((71, 96, 3), np.uint8), np.zeros((72, 95, 3), np.uint8)])
def test_invalid_images_stop_before_initialization(frame):
    encoder = VisionV2Encoder()
    result = encoder.encode(frame, 0.0)
    assert result.invalid_reason == "invalid_image" and result.segment_stopped and result.features is None
    assert encoder.last_committed_time_s is None


@pytest.mark.parametrize("time_s,reason", [
    (float("nan"), "invalid_time"), (-0.1, "invalid_time"), (True, "invalid_time"),
    (0.0, "nonmonotonic_time"), (0.1, "time_step_mismatch"),
])
def test_time_failures_do_not_commit(time_s, reason):
    encoder = VisionV2Encoder()
    encoder.encode(image(), 0.0)
    result = encoder.encode(image(), time_s)
    assert result.invalid_reason == reason and result.features is None and result.segment_stopped
    assert encoder.last_committed_time_s == 0.0


def test_engineering_parameters_are_validated_and_in_identity():
    default = VisionV2Config().identity()
    adjusted = VisionV2Config(min_texture_eigenvalue=2e-6).identity()
    assert default != adjusted
    assert adjusted["quality_parameter_evidence"] == "engineering_candidate_not_optimized"
    for values in ({"min_texture_eigenvalue": 0}, {"max_texture_condition": 0.5}, {"time_tolerance_s": 0.04}):
        with pytest.raises(ValueError):
            VisionV2Config(**values)
