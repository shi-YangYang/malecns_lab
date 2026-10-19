"""Independent saved-array loss, derivative, clipping and fused-Adam arithmetic."""
import math
import numpy as np
from .training_contract import PARAMETERS, ADAM_FIELDS, require
from .gpu_contract import RTOL, ATOL
from .gpu_archive import parameter_shapes


def close(actual, expected, message, *, rtol=RTOL, atol=ATOL):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), message)


def norm(values):
    return math.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in values))


def validate_numbers(values, identity, before_steps, theta_norm_limit):
    shapes = parameter_shapes(identity)
    width = identity["features"]["width"]
    extra = {"gains": shapes["theta"], "rho": (len(identity["model"]["tonic"]["types"]),)}
    expected = {"frames", "time_s", "valid", "teacher_target", "indices", "actions", "logits", "features_mv", "post_fixed_logits"}
    expected |= {f"{when}_{k}" for when in ("before", "after") for k in shapes | extra}
    expected |= {f"{when}_{k}" for when in ("pre_clip", "post_clip") for k in PARAMETERS}
    expected |= {f"adam_{when}.{n}.{f}" for when in ("before", "after") for n in PARAMETERS.values() for f in ADAM_FIELDS}
    require(set(values) == expected, "Window numeric inventory differs")
    def float_value(key, shape):
        v = values[key]
        require(v.shape == shape and v.dtype == np.float32 and np.isfinite(v).all(), f"Invalid numeric {key}")
    for when in ("before", "after"):
        for key, shape in (shapes | extra).items():
            float_value(f"{when}_{key}", shape)
        theta, gains = values[f"{when}_theta"], values[f"{when}_gains"]
        formula = np.clip(1 + np.float32(.5) * np.tanh(theta), np.nextafter(np.float32(.5), np.float32(1)),
                          np.nextafter(np.float32(1.5), np.float32(1)))
        close(gains, formula, "Gain formula differs", rtol=0, atol=1.2e-7)
        require(np.all((gains > .5) & (gains < 1.5)) and not np.any(values[f"{when}_rho"]), "Background/gain bounds changed")
    camera = identity["camera"]
    for key, shape, dtype in (("frames", (2, camera["height"], camera["width"], 3), np.uint8),
                              ("time_s", (2,), np.float64), ("valid", (2,), np.bool_),
                              ("teacher_target", (2, 3), np.float64), ("indices", (2,), np.int64)):
        require(values[key].shape == shape and values[key].dtype == dtype and np.isfinite(values[key]).all(), "Input arrays differ")
    require(values["valid"].all() and np.all(np.abs(values["teacher_target"]) <= [.3, .2, .5]), "Invalid target or validity")
    for key in ("actions", "logits", "post_fixed_logits"):
        float_value(key, (2, 3))
    float_value("features_mv", (2, width))
    scales = np.asarray(identity["readout"]["scales"], np.float32)
    normalized = (values["features_mv"] - identity["features"]["offset_mv"]) / identity["features"]["scale_mv"]
    z, y = values["logits"].astype(np.float64), (values["teacher_target"].astype(np.float32) / scales).astype(np.float64)
    for when, key in (("before", "logits"), ("after", "post_fixed_logits")):
        predicted = normalized.astype(np.float64) @ values[f"{when}_readout_weight"].astype(np.float64).T + values[f"{when}_readout_bias"]
        close(values[key], predicted, "Logits differ from actual neural readout", rtol=2e-5, atol=2e-6)
    close(values["actions"], np.tanh(z) * scales, "Actions differ from logits", rtol=1e-6, atol=1e-7)
    derivative = (np.tanh(z) - y) / 6
    close(values["pre_clip_readout_weight"], derivative.T @ normalized.astype(np.float64), "Readout weight gradient differs", rtol=3e-5, atol=2e-6)
    close(values["pre_clip_readout_bias"], derivative.sum(axis=0), "Readout bias gradient differs", rtol=3e-5, atol=2e-7)
    reports = {}
    for group, keys in {"theta": ("theta",), "readout": ("readout_weight", "readout_bias")}.items():
        for key in keys:
            float_value(f"pre_clip_{key}", shapes[key])
            float_value(f"post_clip_{key}", shapes[key])
        magnitude = norm([values[f"pre_clip_{k}"] for k in keys])
        coefficient = min(1., 1. / (magnitude + 1e-6))
        for key in keys:
            raw, post = values[f"pre_clip_{key}"], values[f"post_clip_{key}"]
            close(post, raw.astype(np.float64) * coefficient, "GPU clipping differs")
            require(np.array_equal(raw != 0, post != 0) and np.array_equal(np.sign(raw), np.sign(post)), "Clipping lost gradient/sign")
        reports[group] = {"norm_before": magnitude, "norm_after": norm([values[f"post_clip_{k}"] for k in keys]),
                          "nonzero_before": sum(int(np.count_nonzero(values[f"pre_clip_{k}"])) for k in keys)}
        require(reports[group]["norm_after"] <= 1 + 1e-6, "GPU clip bound exceeded")
    require(0 < reports["theta"]["norm_before"] <= theta_norm_limit, "Theta gradient stability gate failed")
    errors = {}
    for key, name in PARAMETERS.items():
        old, new = {}, {}
        for when, collection in (("before", old), ("after", new)):
            for field in ADAM_FIELDS:
                label = f"adam_{when}.{name}.{field}"
                float_value(label, () if field == "step" else shapes[key])
                collection[field] = values[label].astype(np.float64)
            require(float(collection["step"]) == before_steps + (when == "after") and np.all(collection["exp_avg_sq"] >= 0), "Adam clock/moment differs")
        g = values[f"post_clip_{key}"].astype(np.float64)
        m, v = .9 * old["exp_avg"] + .1 * g, .999 * old["exp_avg_sq"] + .001 * g * g
        close(new["exp_avg"], m, "Fused Adam first moment differs")
        close(new["exp_avg_sq"], v, "Fused Adam second moment differs")
        require(not np.any((g != 0) & (new["exp_avg_sq"] == 0)), "Nonzero gradient lost second moment")
        step, lr = before_steps + 1, .001 if key == "theta" else .0001
        p = values[f"before_{key}"].astype(np.float64) - lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        close(values[f"after_{key}"], p, "Fused Adam parameter differs")
        errors[key] = {"parameter_max_abs": float(np.max(np.abs(values[f"after_{key}"] - p))),
                       "first_moment_max_abs": float(np.max(np.abs(new["exp_avg"] - m))),
                       "second_moment_max_abs": float(np.max(np.abs(new["exp_avg_sq"] - v)))}
    changes = {k: {"changed": int(np.count_nonzero(values[f"after_{k}"] != values[f"before_{k}"])),
                   "max_abs": float(np.max(np.abs(values[f"after_{k}"].astype(np.float64) - values[f"before_{k}"])))} for k in PARAMETERS}
    require(changes["theta"]["changed"] > 0, "No internal parameter change")
    terms = np.logaddexp(0, 2 * z) - (1 + y) * z
    after_z = values["post_fixed_logits"].astype(np.float64)
    post_terms = np.logaddexp(0, 2 * after_z) - (1 + y) * after_z
    reg = float(1e-4 * np.square(values["before_gains"].astype(np.float64) - 1).mean())
    losses = {"logit": float(terms.mean()), "action_mse": float(np.square((values["actions"].astype(np.float64) - values["teacher_target"].astype(np.float32)) / scales).mean()),
              "regularization": reg, "total": float(terms.mean() + reg)}
    return {"losses": losses, "clipping": reports, "adam_oracle_errors": errors, "parameter_updates": changes,
            "post_fixed_logit_loss": float(post_terms.mean()), "fixed_vx_loss_before": float(terms[:, 0].mean()),
            "fixed_vx_loss_after": float(post_terms[:, 0].mean()),
            "action_axis_range": [values["actions"].min(0).tolist(), values["actions"].max(0).tolist()],
            "label_axis_range": [values["teacher_target"].min(0).tolist(), values["teacher_target"].max(0).tolist()],
            "logit_axis_range": [values["logits"].min(0).tolist(), values["logits"].max(0).tolist()]}


def validate_correction(values, numeric):
    require(np.all(values["teacher_target"][:, 0] < 0) and np.all(values["logits"][:, 0] > 0), "Gate did not test wrong positive vx")
    require(values["pre_clip_readout_bias"][0] > 0 and np.count_nonzero(values["pre_clip_readout_weight"][0]) > 0,
            "Missing vx correction gradient")
    require(np.all(values["post_fixed_logits"][:, 0] < values["logits"][:, 0])
            and numeric["post_fixed_logit_loss"] < numeric["losses"]["logit"]
            and numeric["fixed_vx_loss_after"] < numeric["fixed_vx_loss_before"], "Actual fixed-feature update did not correct saturation")
    return {"fixed_feature_update_corrects_vx": True, "actual_action_sign_reversed": bool(np.all(np.tanh(values["post_fixed_logits"][:, 0]) < 0)),
            "interpretation": "One disposable update; fixed-feature loss improves, not closed-loop or learned-control evidence"}
