"""Stable identity for same-prestate task-paired ridge readouts."""
from copy import deepcopy
from pathlib import Path

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_interaction_paired_policy.v1"
ALPHA = 1e-4


def bind_paired_policy(policy):
    """Explicit new identity; the previous interaction policy file is untouched."""
    from .x2_interaction_policy import SCHEMA as INTERACTION_SCHEMA
    previous = policy.identity
    frozen.require(previous["schema"] == INTERACTION_SCHEMA, "Original interaction policy identity required")
    payload = deepcopy(previous)
    payload.pop("policy_id")
    payload.update(schema=SCHEMA, parent_interaction_policy_id=previous["policy_id"],
        paired_implementation_sha256=frozen.sha256(Path(__file__)))
    payload["readout_fit"].update(alpha=ALPHA, training="24 original plus 18 same-prestate task branches; train-only statistics")
    policy.policy_id = frozen.canonical_hash(payload)
    policy._identity = {**payload, "policy_id": policy.policy_id}
    return policy
