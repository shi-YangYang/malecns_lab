"""Explicit numerical trial variant; unchanged complete-brain forward equations.

The existing production policy and its checkpoint identity remain untouched.
This variant changes the surrogate estimator and the coordinates in which the
three neural readouts are optimized. It is not automatically deployable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..control.x2_behavior import digest, require
from .x2_behavior_policy import X2BehaviorPolicy, array_identity


def normalization(features):
    """Fit fixed statistics from train neural features only, in float64."""
    values = np.asarray(features, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 2628 and len(values) > 1
            and np.isfinite(values).all(), "Invalid train neural features")
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), .1).astype(np.float32)


def affine_head(weight, bias, center, scale, *, v_rest=-52., span=7.):
    """Same affine function, new coordinates; tolerate float32 rounding only."""
    weight, bias = np.asarray(weight, np.float64), np.asarray(bias, np.float64)
    center, scale = np.asarray(center, np.float64), np.asarray(scale, np.float64)
    require(weight.ndim == 2 and weight.shape[1] == len(center) == len(scale)
            and bias.shape == (len(weight),) and np.isfinite(weight).all()
            and np.isfinite(bias).all() and np.isfinite(center).all()
            and np.isfinite(scale).all() and (scale > 0).all() and span > 0,
            "Invalid affine readout coordinates")
    return (weight * scale / span).astype(np.float32), (bias + weight @ ((center-v_rest)/span)).astype(np.float32)


class X2BehaviorTrialPolicy(X2BehaviorPolicy):
    """Construct normally, restore the old policy, then explicitly migrate once."""

    def read_features(self, features_mv):
        if not hasattr(self, "trial_center"):
            return super().read_features(features_mv)
        require(tuple(features_mv.shape) == (2628,), "All DN features are required")
        values = (features_mv-self.trial_center) / self.trial_scale
        behavior, phase, raw = self.behavior_head(values), self.phase_head(values), self.parameter_head(values)
        return behavior, phase, raw, torch.sigmoid(raw)

    def migrate_trial(self, center, scale, *, source_checkpoint_id, statistics_evidence):
        require(not hasattr(self, "trial_center"), "Trial migration may happen only once")
        require(center.shape == scale.shape == (2628,) and center.dtype == scale.dtype == np.float32
                and np.isfinite(center).all() and np.isfinite(scale).all() and (scale >= .1).all(),
                "Invalid fixed train normalization")
        previous = self.identity
        with torch.no_grad():
            for head in (self.behavior_head, self.phase_head, self.parameter_head):
                weight, bias = affine_head(head.weight.cpu().numpy(), head.bias.cpu().numpy(), center, scale,
                    v_rest=self.brain.config.v_rest_mv,
                    span=self.brain.config.v_threshold_mv-self.brain.config.v_rest_mv)
                head.weight.copy_(torch.from_numpy(weight).to("mps"))
                head.bias.copy_(torch.from_numpy(bias).to("mps"))
        self.register_buffer("trial_center", torch.from_numpy(center.copy()).to("mps"), persistent=False)
        self.register_buffer("trial_scale", torch.from_numpy(scale.copy()).to("mps"), persistent=False)
        # MPSLIF already supports this multiplier; no neuron/edge or forward
        # equation changes, and no state detach inside either decision.
        self.spike_backward_gain = .01
        payload = deepcopy(previous)
        payload.pop("policy_id")
        payload["schema"] = "malecns_lab.x2_behavior_trial_policy.v1"
        payload["parent_policy_id"] = previous["policy_id"]
        payload["source_checkpoint_id"] = source_checkpoint_id
        payload["backward"] = {"spike_backward_gain": .01, "credit_steps": 1000,
            "algorithm": "spikes.detach()+0.01*(spikes-spikes.detach())",
            "meaning": "new surrogate estimator; unchanged binary neural forward; no claim of exact gradient"}
        payload["trial_normalization"] = {"center_mv": center.tolist(), "scale_mv": scale.tolist(),
            "center_identity": array_identity(center), "scale_identity": array_identity(scale),
            "source": statistics_evidence, "std_floor_mv": .1,
            "initialization": "explicit affine head migration; same initial readout up to float32 rounding"}
        payload["trial_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.policy_id = digest(payload)
        self._identity = {**payload, "policy_id": self.policy_id}
        self.validate_parameters()
        return previous
