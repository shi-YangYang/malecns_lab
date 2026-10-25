"""Stable scientific identity for one theta-only internal-learning trial."""
from copy import deepcopy
from pathlib import Path

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_theta_trial_policy.v1"


def bind_theta_trial_policy(policy, source_checkpoint_id):
    from .x2_interaction_paired_policy import SCHEMA as PAIRED_SCHEMA
    previous = policy.identity
    frozen.require(previous["schema"] == PAIRED_SCHEMA, "Strict paired source policy required")
    names = dict(policy.named_parameters())
    frozen.require(set(names) == {"brain.theta", "behavior_head.weight", "behavior_head.bias",
        "phase_head.weight", "phase_head.bias", "parameter_head.weight", "parameter_head.bias"},
        "Unexpected trainable parameter inventory")
    policy.requires_grad_(False)
    policy.brain.theta.requires_grad_(True)
    policy.spike_backward_gain = .01
    payload = deepcopy(previous)
    payload.pop("policy_id")
    payload.update(schema=SCHEMA, parent_policy_id=previous["policy_id"],
        source_checkpoint_id=source_checkpoint_id,
        theta_trial_implementation_sha256=frozen.sha256(Path(__file__)),
        backward={"spike_backward_gain": .01, "maximum_credit_steps": 1000,
            "meaning": "new surrogate estimator; unchanged binary forward; no exact-gradient claim"},
        internal_update={"parameter": "brain.theta", "optimizer": "Adam", "lr": 1e-4,
            "gradient_l2_cap": 1., "updates": 1, "heads": "all six frozen", "loss": "behavior CE plus phase CE"},
        deployment="scientific trial only; unsupported by desktop loader")
    payload["readout_fit"]["theta"] = "source theta followed by one internal Adam update; fitted readout parameters frozen"
    policy.policy_id = frozen.canonical_hash(payload)
    policy._identity = {**payload, "policy_id": policy.policy_id}
    return policy
