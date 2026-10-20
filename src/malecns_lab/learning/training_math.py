"""NumPy-only loss, clipping, and general-step Adam evidence reductions."""
from __future__ import annotations

import math
import numpy as np

from .training_contract import PARAMETERS, ADAM_FIELDS, PROTOCOL, active_keys, require, read_arrays


def adam_moments(gradient, old_m, old_v):
    delta = gradient - old_m
    moment = (old_m.astype(np.float64) + float(np.float32(1 - .9)) * delta.astype(np.float64)).astype(np.float32)
    second = old_v * np.float32(.999) + (np.float32(1 - .999) * gradient) * gradient
    return moment, second


def adam_parameters(before, moment, second, step, lr):
    denominator = np.sqrt(second) / np.float32((1 - .999 ** step) ** .5)
    denominator += np.float32(1e-8)
    return before + (np.float32(-lr / (1 - .9 ** step)) * moment) / denominator


def float_array(value, shape):
    require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(), "Invalid float32 array")


def moment_check(actual, expected):
    require(np.allclose(actual.astype(np.float64), expected.astype(np.float64),
                        rtol=1e-6, atol=PROTOCOL["adam_check"]["moment_atol"]), "Adam moment mismatch")


def clipping(raw, mode):
    groups = {"theta": ("theta",), "readout": ("readout_weight", "readout_bias")}
    if mode == "frozen_brain":
        groups.pop("theta")
    result, scaled = {}, {}
    # Compute every norm before returning anything that the worker can mutate.
    require(set(raw) == set(active_keys(mode)) and all(np.isfinite(v).all() for v in raw.values()), "Nonfinite raw gradients")
    for group, keys in groups.items():
        norm = math.sqrt(sum(float(np.square(raw[k].astype(np.float64)).sum()) for k in keys))
        require(math.isfinite(norm), "Nonfinite stable gradient norm")
        coefficient = min(1., 1. / (norm + 1e-6))
        for key in keys:
            scaled[key] = raw[key] * np.float32(coefficient)
        after = math.sqrt(sum(float(np.square(scaled[k].astype(np.float64)).sum()) for k in keys))
        require(after <= 1 + 1e-6, "Post-clip group exceeds bound")
        result[group] = {"norm_before": norm, "norm_after": after, "coefficient_float64": coefficient,
                         "coefficient_float32": float(np.float32(coefficient)),
                         "nonzero_before": sum(int(np.count_nonzero(raw[k])) for k in keys),
                         "nonzero_after": sum(int(np.count_nonzero(scaled[k])) for k in keys)}
    return scaled, result


def loss_metrics(actions, targets, valid, gains):
    if not valid.any():
        return {"action": None, "regularization": None, "total": None, "per_axis": None}
    target = targets[valid].astype(np.float32).astype(np.float64)
    scale = np.array([.3, .2, .5], dtype=np.float32).astype(np.float64)
    axes = (((actions[valid].astype(np.float64) - target) / scale) ** 2).mean(axis=0)
    reg = float(1e-4 * np.square(gains.astype(np.float64) - 1).mean())
    return {"action": float(axes.mean()), "regularization": reg, "total": float(axes.mean() + reg), "per_axis": axes.tolist()}


def validate_window_numbers(values, identity, mode, before_steps):
    """No model execution; check all actual arrays and arithmetic in one window."""
    width = identity["features"]["width"]
    shapes = {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "readout_weight": (3, width), "readout_bias": (3,),
              "gains": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "rho": (len(identity["model"]["tonic"]["types"]),)}
    updated = bool(values["valid"].any())
    require(updated or before_steps > 0, "Zero-step invalid window cannot be sealed")
    require(type(before_steps) is int and before_steps >= 0, "Invalid prior step")
    active = active_keys(mode)
    expected = {"frames", "time_s", "valid", "teacher_target", "indices", "actions", "features_mv", "adam_before_present"}
    expected |= {f"{when}_{key}" for when in ("before", "after") for key in shapes}
    if updated:
        expected |= {f"{when}_{key}" for when in ("pre_clip", "post_clip") for key in active}
    if before_steps:
        expected |= {f"adam_before.{PARAMETERS[key]}.{field}" for key in active for field in ADAM_FIELDS}
    if before_steps or updated:
        expected |= {f"adam_after.{PARAMETERS[key]}.{field}" for key in active for field in ADAM_FIELDS}
    require(set(values) == expected, "Window evidence fields differ")
    require(values["adam_before_present"].dtype == np.bool_ and values["adam_before_present"].shape == ()
            and bool(values["adam_before_present"]) == bool(before_steps), "Adam initial absence marker differs")
    for when in ("before", "after"):
        for key, shape in shapes.items():
            float_array(values[f"{when}_{key}"], shape)
        theta, gains = values[f"{when}_theta"], values[f"{when}_gains"]
        expected_gains = np.clip(1 + np.float32(.5) * np.tanh(theta), np.nextafter(np.float32(.5), np.float32(1)),
                                 np.nextafter(np.float32(1.5), np.float32(1)))
        require(np.allclose(gains, expected_gains, rtol=0, atol=1.2e-7)
                and np.all((gains > .5) & (gains < 1.5)), "Gain bounds/formula mismatch")
        require(not np.any(values[f"{when}_rho"]), "Fixed background changed")
    float_array(values["actions"], (2, 3))
    float_array(values["features_mv"], (2, width))
    camera = identity["camera"]
    for key, shape, dtype in (("frames", (2, camera["height"], camera["width"], 3), np.uint8),
                              ("time_s", (2,), np.float64), ("valid", (2,), np.bool_),
                              ("teacher_target", (2, 3), np.float64), ("indices", (2,), np.int64)):
        require(values[key].shape == shape and values[key].dtype == dtype and np.isfinite(values[key]).all(), "Bad input arrays")
    require(np.all(np.abs(values["teacher_target"]) <= [.3, .2, .5]), "Target exceeds bounds")
    features = identity["features"]
    normalized = (values["features_mv"] - features["offset_mv"]) / features["scale_mv"]
    predicted = np.tanh(normalized @ values["before_readout_weight"].T + values["before_readout_bias"]) * np.array([.3, .2, .5], np.float32)
    require(np.allclose(values["actions"], predicted, rtol=1e-6, atol=1e-7), "Actions differ from neural readout")
    clip_report, moment_report = {}, {}
    if updated:
        raw = {key: values[f"pre_clip_{key}"] for key in active}
        for key, value in raw.items():
            float_array(value, shapes[key])
        expected_scaled, clip_report = clipping(raw, mode)
        for key in active:
            float_array(values[f"post_clip_{key}"], shapes[key])
            require(np.array_equal(values[f"post_clip_{key}"], expected_scaled[key]), "Clipped gradient differs")
    for key in active:
        name = PARAMETERS[key]
        prior = {}
        for field in ADAM_FIELDS:
            prior[field] = values[f"adam_before.{name}.{field}"] if before_steps else np.zeros(() if field == "step" else shapes[key], np.float32)
            if before_steps:
                float_array(prior[field], () if field == "step" else shapes[key])
        require(float(prior["step"]) == before_steps and np.all(prior["exp_avg_sq"] >= 0), "Prior Adam states invalid")
        after = {field: values[f"adam_after.{name}.{field}"] for field in ADAM_FIELDS}
        for field in ADAM_FIELDS:
            float_array(after[field], () if field == "step" else shapes[key])
        require(float(after["step"]) == before_steps + updated and np.all(after["exp_avg_sq"] >= 0), "Adam step/second moment invalid")
        if updated:
            g = values[f"post_clip_{key}"]
            m, v = adam_moments(g, prior["exp_avg"], prior["exp_avg_sq"])
            moment_check(after["exp_avg"], m)
            moment_check(after["exp_avg_sq"], v)
            parameter = adam_parameters(values[f"before_{key}"], after["exp_avg"], after["exp_avg_sq"], before_steps + 1,
                                        .001 if key == "theta" else .0001)
            require(np.allclose(values[f"after_{key}"], parameter, rtol=5e-6, atol=1e-9), "Adam parameter update mismatch")
            moment_report[name] = {"step": before_steps + 1,
                "zero_second_moment_with_nonzero_gradient": int(np.count_nonzero((after["exp_avg_sq"] == 0) & (g != 0)))}
        else:
            require(all(np.array_equal(prior[f], after[f]) for f in ADAM_FIELDS), "Skipped window changed Adam")
    if mode == "frozen_brain" or not updated:
        unchanged = ("theta",) if updated else tuple(shapes)
        require(all(np.array_equal(values[f"before_{key}"], values[f"after_{key}"]) for key in unchanged), "Frozen/skipped parameters changed")
    changes = {key: {"changed": int(np.count_nonzero(values[f"after_{key}"] != values[f"before_{key}"])),
                     "max_abs": float(np.max(np.abs(values[f"after_{key}"].astype(np.float64) - values[f"before_{key}"].astype(np.float64))))}
               for key in PARAMETERS}
    return {"updated": updated, "metrics": loss_metrics(values["actions"], values["teacher_target"], values["valid"], values["before_gains"]),
            "clipping": clip_report, "adam": moment_report, "parameter_updates": changes,
            "feature_range_mv": [float(values["features_mv"].min()), float(values["features_mv"].max())],
            "valid_frames": int(np.count_nonzero(values["valid"])),
            "label_axis_range": [values["teacher_target"].min(axis=0).tolist(), values["teacher_target"].max(axis=0).tolist()],
            "action_axis_range": [values["actions"].min(axis=0).tolist(), values["actions"].max(axis=0).tolist()]}


def compare_r2(values, directory):
    """Exact initialization/inputs, fixed-tolerance baseline/gradient/update check."""
    from pathlib import Path
    directory = Path(directory)
    reports = {}
    maps = {"initial_parameters.npz": {key: f"before_{key}" for key in ("theta", "readout_weight", "readout_bias", "gains", "rho")},
            "baseline.npz": {"actions": "actions", "features_mv": "features_mv"},
            "gradients_before_clip.npz": {key: f"pre_clip_{key}" for key in PARAMETERS},
            "update.npz": {f"after_{key}": f"after_{key}" for key in ("theta", "readout_weight", "readout_bias", "gains", "rho")},
            "selection.npz": {key: key for key in ("frames", "time_s", "valid", "teacher_target", "indices")}}
    for filename, keys in maps.items():
        old = read_arrays(directory / filename)
        for old_key, key in keys.items():
            a, b = old[old_key], values[key]
            require(a.shape == b.shape and a.dtype == b.dtype, "r2 shape/dtype differs")
            exact = bool(np.array_equal(a, b))
            close = bool(np.allclose(a, b, rtol=1e-6, atol=1e-7))
            required = exact if filename in ("initial_parameters.npz", "selection.npz") else close
            reports[f"{filename}:{old_key}"] = {"exact": exact, "allclose": close, "passed": required,
                                                   "max_abs_difference": float(np.max(np.abs(a.astype(float) - b.astype(float))))}
    return {"passed": all(item["passed"] for item in reports.values()), "comparisons": reports, "rtol": 1e-6, "atol": 1e-7}
