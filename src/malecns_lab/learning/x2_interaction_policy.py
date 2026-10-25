"""Explicit continuous-interaction policy identity and folded ridge readouts.

The seven float32 checkpoint parameters and original neural forward are kept.
Only the sensory identity and fitted linear heads differ from the old policy.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from ..control.x2_behavior import digest, require
from ..control.x2_interaction import interaction_profile, validate_interaction_profile
from ..senses.x2_relative_heading import RelativeHeadingEncoder

SCHEMA = "malecns_lab.x2_interaction_policy.v1"
FOLD_ATOL, FOLD_RTOL = 5e-4, 1e-4


def bind_interaction_policy(policy, *, source_checkpoint_id, profile=None):
    """Bind a freshly reconstructed v1 policy before strict new-schema restore."""
    previous = policy.identity
    require(previous["schema"] == "malecns_lab.x2_behavior_policy.v1", "Expected unmigrated behavior policy")
    require(isinstance(source_checkpoint_id, str) and len(source_checkpoint_id) == 64
            and all(c in "0123456789abcdef" for c in source_checkpoint_id), "Invalid source checkpoint identity")
    profile = validate_interaction_profile(interaction_profile() if profile is None else profile)
    policy.encoder = RelativeHeadingEncoder(policy.encoder)
    payload = deepcopy(previous)
    payload.pop("policy_id")
    payload.update(schema=SCHEMA, parent_policy_id=previous["policy_id"],
        source_checkpoint_id=source_checkpoint_id, input_map=policy.encoder.identity,
        interaction_profile=profile,
        readout_fit={"method": "ridge", "alpha": 1., "normalization": "train mean/std, std floor 0.1 mV",
                     "storage": "normalization folded into existing float32 linear weights/biases",
                     "theta": "frozen", "qualification_only": True},
        interaction_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        relative_encoder_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath("senses/x2_relative_heading.py").read_bytes()).hexdigest())
    policy.policy_id = digest(payload)
    policy._identity = {**payload, "policy_id": policy.policy_id}
    return policy


def fold_readout(readout, *, v_rest, span):
    """Algebraic coordinate conversion, computed in float64 then stored float32."""
    import numpy as np
    center, scale = np.asarray(readout["center"]), np.asarray(readout["scale"])
    weight, bias = np.asarray(readout["weight"]), np.asarray(readout["bias"])
    require(center.shape == scale.shape == (2628,) and weight.shape == (2628, 8)
            and bias.shape == (8,) and np.isfinite(span) and span > 0
            and all(np.isfinite(v).all() for v in readout.values()) and (scale >= .1).all(),
            "Invalid train ridge readout")
    combined_w = np.column_stack((weight, readout["parameter_weight"]))
    combined_b = np.r_[bias, readout["parameter_bias"]]
    folded_w = (span/scale[:, None]*combined_w).T.astype(np.float32)
    folded_b = (combined_b+((v_rest-center)/scale)@combined_w).astype(np.float32)
    require(np.isfinite(folded_w).all() and np.isfinite(folded_b).all(), "Nonfinite folded readout")
    return {"behavior_head.weight": folded_w[:5], "behavior_head.bias": folded_b[:5],
            "phase_head.weight": folded_w[5:8], "phase_head.bias": folded_b[5:8],
            "parameter_head.weight": folded_w[8:], "parameter_head.bias": folded_b[8:]}


def install_readout(policy, readout):
    import torch
    require(policy.identity["schema"] == SCHEMA, "Explicit interaction policy required")
    cfg = policy.brain.config
    values = fold_readout(readout, v_rest=cfg.v_rest_mv, span=cfg.v_threshold_mv-cfg.v_rest_mv)
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].copy_(torch.from_numpy(value.copy()).to(parameters[name].device))
    policy.validate_parameters()
    return values


def verify_fold(readout, features, actual_logits):
    """Compare every real DN row to the unfused float64 fit, including argmax."""
    import numpy as np
    features, actual = np.asarray(features, np.float64), np.asarray(actual_logits, np.float64)
    require(features.ndim == 2 and features.shape[1] == 2628 and actual.shape == (len(features), 9),
            "Fold audit must cover all supplied neural features")
    z = (features-readout["center"])/readout["scale"]
    expected = np.column_stack((z@readout["weight"]+readout["bias"],
                               z@readout["parameter_weight"]+readout["parameter_bias"]))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    close = finite and bool(np.allclose(actual, expected, atol=FOLD_ATOL, rtol=FOLD_RTOL))
    argmax = finite and bool(np.array_equal(actual[:, :5].argmax(1), expected[:, :5].argmax(1))
                            and np.array_equal(actual[:, 5:8].argmax(1), expected[:, 5:8].argmax(1)))
    evidence = {"frames": len(features), "atol": FOLD_ATOL, "rtol": FOLD_RTOL,
        "finite": finite, "logits_close": close, "behavior_and_phase_argmax_exact": argmax,
        "max_abs_logit_error": float(np.abs(actual-expected).max()) if finite else None,
        "passed": bool(close and argmax)}
    return evidence, expected
