"""Only the serialized joint decision contract; no model, brain or device."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from malecns_lab.control.x2_behavior import (
    Behavior, Decision, JOINT_READOUT_CLASSES, JOINT_READOUT_SCHEMA, Phase, SCHEMA,
)


def test_joint_decision_preserves_legacy_and_binds_members_and_consumed_source():
    legacy = Decision("run", "session", 0, "request", 4, "before", "after",
        (0., 1., 0., 0., 0.), (1., 0., 0.), (0.,), Behavior.ADVANCE, Phase.START,
        .5, "policy", "checkpoint", "input", 600, 1100)
    old = {"schema": SCHEMA, "kind": "Decision", "run_id": "run", "session_id": "session",
        "decision_seq": 0, "request_id": "request", "observation_seq": 4,
        "state_before_id": "before", "state_after_id": "after",
        "behavior_logits": (0., 1., 0., 0., 0.), "phase_logits": (1., 0., 0.),
        "parameter_raw": (0.,), "behavior": Behavior.ADVANCE, "phase": Phase.START,
        "intensity": .5, "policy_id": "policy", "checkpoint_id": "checkpoint",
        "input_map_id": "input", "neural_start_step": 600, "neural_end_step": 1100}
    assert legacy.to_dict() == old
    assert Decision.from_dict(json.loads(json.dumps(old))) == legacy
    # Unequal members require mean softmax; softmax(mean logits) is not equivalent.
    members = [[0., 4., 0., 0., 0., 0., 0., 0., 0.],
               [0., 2., 1., 0., 0., 0., 0., 0., 0.],
               [0., 3., 0., 0., 1., 0., 0., 0., 0.]]
    probs = [sum(math.exp(row[i]) / sum(map(math.exp, row)) for row in members) / 3 for i in range(9)]
    evidence = dict(schema=JOINT_READOUT_SCHEMA, classes=list(JOINT_READOUT_CLASSES),
        member_logits=members, probabilities=probs, selected_index=1,
        model_advanced_this_frame=True, source_observation_seq=4, source_request_id="request",
        source_neural_start=600, source_neural_end=1100)
    joint = replace(legacy, behavior_logits=(), phase_logits=(), joint_readout=evidence)
    assert Decision.from_dict(json.loads(json.dumps(joint.to_dict()))) == joint
    held = replace(joint, decision_seq=1, observation_seq=5,
        joint_readout={**evidence, "model_advanced_this_frame": False})
    assert (held.neural_start_step, held.neural_end_step) == (600, 1100)
    assert held.joint_readout["source_observation_seq"] == 4
    for change, error in [({"selected_index": 0}, "selected index"),
                          ({"probabilities": [1.] + [0.] * 8}, "probabilities differ"),
                          ({"classes": list(reversed(JOINT_READOUT_CLASSES))}, "classes differ"),
                          ({"source_request_id": "other"}, "crosses request"),
                          ({"source_neural_end": 1600}, "neural interval"),
                          ({"model_advanced_this_frame": False}, "consumption clock")]:
        with pytest.raises(ValueError, match=error):
            replace(joint, joint_readout={**evidence, **change})
    bad_members = deepcopy(members)
    bad_members[0][1] = float("nan")
    with pytest.raises(ValueError, match="member logits"):
        replace(joint, joint_readout={**evidence, "member_logits": bad_members})
    with pytest.raises(ValueError, match="fabricate"):
        replace(joint, behavior_logits=legacy.behavior_logits)
    with pytest.raises(ValueError, match="joint class"):
        replace(joint, behavior=Behavior.HOLD, phase=Phase.FINISH)
