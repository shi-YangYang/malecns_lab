"""Independent NumPy check of r2 group clipping and actual float32 Adam evidence."""
from __future__ import annotations

import math

import numpy as np

from .joint_probe_contract import ADAM_FIELDS, CLIP_GROUPS, PARAMETERS, PROTOCOL, require


def adam_first_step(before, gradient, lr):
    """Mirror the recorded CPU foreach=False operations, including float32 underflow.

    This is not the idealized lr*g/(abs(g)+eps) expression: zero-initialized
    second moments can round to zero before sqrt. All intermediate arrays stay
    float32. Scalar bias corrections are computed as in non-capturable Adam.
    """
    config = PROTOCOL["optimizer"]
    beta1, beta2 = config["betas"]
    moment = np.float32(1 - beta1) * gradient
    second = (np.float32(1 - beta2) * gradient) * gradient
    denominator = np.sqrt(second) / np.float32((1 - beta2) ** 0.5)
    denominator += np.float32(config["eps"])
    result = before + (np.float32(-lr / (1 - beta1)) * moment) / denominator
    return moment, second, result


def validate_optimizer_update(update, states, before, after, shapes, worker):
    from .joint_probe_evidence import check_array
    clipping = worker["gradient_clipping"]
    require(set(clipping) == {"schema", "total_norm_before", "total_norm_after", "groups"}
            and clipping["schema"] == "cnsbridge.group_gradient_clip.v1"
            and set(clipping["groups"]) == set(CLIP_GROUPS), "Group clipping evidence inventory differs")
    require(worker["optimizer"] == PROTOCOL["optimizer"] and worker["optimizer_steps"] == dict.fromkeys(PARAMETERS, 1.0),
            "Optimizer declaration or one-step count differs")
    total_pre, total_post = 0.0, 0.0
    group_reports, group_updates = {}, {}
    for group, keys in CLIP_GROUPS.items():
        pre_squared = sum(float(np.square(update[f"pre_clip_{key}"].astype(np.float64)).sum()) for key in keys)
        post_squared = sum(float(np.square(update[f"post_clip_{key}"].astype(np.float64)).sum()) for key in keys)
        norm, post_norm = math.sqrt(pre_squared), math.sqrt(post_squared)
        require(math.isfinite(norm) and math.isfinite(post_norm), "Nonfinite stable gradient norm")
        coefficient = min(1.0, 1.0 / (norm + 1e-6))
        actual_coefficient = float(np.float32(coefficient))
        expected = {"parameters": [PARAMETERS[key] for key in keys], "norm_before": norm, "norm_after": post_norm,
                    "coefficient_float64": coefficient, "coefficient_float32": actual_coefficient,
                    "nonzero_before": sum(int(np.count_nonzero(update[f"pre_clip_{key}"])) for key in keys),
                    "nonzero_after": sum(int(np.count_nonzero(update[f"post_clip_{key}"])) for key in keys)}
        recorded = clipping["groups"][group]
        require(set(recorded) == set(expected), "Clipping group fields differ")
        for field, value in expected.items():
            if field in ("norm_before", "norm_after", "coefficient_float64"):
                require(type(recorded[field]) is float and math.isclose(recorded[field], value, rel_tol=1e-12, abs_tol=0),
                        f"Stable group {field} differs")
            else:
                require(recorded[field] == value, f"Clipping group {field} differs")
        require(post_norm <= 1 + PROTOCOL["clipping"]["post_norm_tolerance"], "Post-clip group norm exceeds cap")
        for key in keys:
            require(np.array_equal(update[f"post_clip_{key}"], update[f"pre_clip_{key}"] * np.float32(coefficient)),
                    "Post-clip gradient differs from actual float32 coefficient")
        total_pre += pre_squared
        total_post += post_squared
        group_reports[group] = expected
        deltas = [after[key].astype(np.float64) - before[key].astype(np.float64) for key in keys]
        group_updates[group] = {"parameters": [PARAMETERS[key] for key in keys],
                                "changed_elements": sum(int(np.count_nonzero(value)) for value in deltas),
                                "l2": math.sqrt(sum(float(np.square(value).sum()) for value in deltas)),
                                "max_abs": max(float(np.abs(value).max()) for value in deltas)}
    for label, squared in (("before", total_pre), ("after", total_post)):
        value = clipping[f"total_norm_{label}"]
        require(type(value) is float and math.isfinite(value) and math.isclose(value, math.sqrt(squared), rel_tol=1e-12),
                "Stable total gradient norm differs")
    require(math.sqrt(total_post) <= math.sqrt(2) * (1 + PROTOCOL["clipping"]["post_norm_tolerance"]),
            "Combined post-clip norm exceeds two-group bound")
    require(worker["parameter_updates"] == group_updates, "Group parameter update statistics differ")
    state_reports = {}
    evidence = PROTOCOL["adam_evidence"]
    for key, name in PARAMETERS.items():
        for field in ADAM_FIELDS:
            check_array(states[name + "." + field], () if field == "step" else shapes[key], "float32", name + "." + field)
        require(states[name + ".step"].item() == 1, "Adam scalar step must equal one")
        first, second, expected = adam_first_step(before[key], update[f"post_clip_{key}"], 0.001 if key == "theta" else 0.0001)
        for field, calculated in (("exp_avg", first), ("exp_avg_sq", second)):
            actual = states[name + "." + field]
            require(np.allclose(actual.astype(np.float64), calculated.astype(np.float64),
                                rtol=evidence["moment_rtol"], atol=evidence["moment_atol"]),
                    f"Actual float32 Adam {name}.{field} differs")
        require(np.allclose(after[key], expected, rtol=evidence["update_rtol"], atol=evidence["update_atol"]),
                "Parameters do not match the actual float32 Adam step")
        state_reports[name] = {"step": 1, "nonzero_exp_avg": int(np.count_nonzero(states[name + ".exp_avg"])),
                               "nonzero_exp_avg_sq": int(np.count_nonzero(states[name + ".exp_avg_sq"])),
                               "zero_second_moment_with_nonzero_gradient": int(np.count_nonzero(
                                   (states[name + ".exp_avg_sq"] == 0) & (update[f"post_clip_{key}"] != 0)))}
    return {"clipping_groups": group_reports, "parameter_updates": group_updates, "adam_states": state_reports}
