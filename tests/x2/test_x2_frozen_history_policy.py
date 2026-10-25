"""Local adapter contract with a fake brain; no full graph or device startup."""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from malecns_lab.control.x2_behavior import Feedback, Observation
from malecns_lab.learning.x2_frozen_history_policy import (
    CANDIDATE_RELATIVE, FrozenHistoryPolicy, capture_actual_steps,
    inspect_frozen_history_candidate, load_input_contract,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BrainState:
    v: object
    g: object
    refractory_until: object
    pending: tuple
    step: int


@dataclass(frozen=True)
class Native:
    brain: BrainState
    phase: object
    event_counts: object


@dataclass(frozen=True)
class Prefix:
    native: Native
    tokens: tuple = ()
    node_id: str = "ROOT"


class FakeEngine:
    def __init__(self, channels):
        self.channel_names = tuple(channels)
        self.reference = SimpleNamespace(base=SimpleNamespace(brain=object()))
        self.kernel = SimpleNamespace()
        self.advances, self.initializations = [], 0

    def initial_prefix(self):
        self.initializations += 1
        b = BrainState(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.int64),
                       (torch.zeros(2),), 600)
        return Prefix(Native(b, torch.zeros(1), torch.zeros(1, dtype=torch.int64)))

    def advance_tokens(self, prefix, raw):
        self.advances.append((prefix, raw))
        following = replace(prefix, native=replace(prefix.native,
            brain=replace(prefix.native.brain, step=prefix.native.brain.step + 500)),
            tokens=prefix.tokens + (raw,), node_id=str(len(prefix.tokens) + 1))
        return torch.full((7110,), float(len(following.tokens))), following, {}

    def state_arrays(self, native):
        return {"v": native.brain.v.numpy().copy()}


class FakeMember:
    def __init__(self):
        self.inputs = []

    def eval(self):
        return self

    def project(self, value):
        self.inputs.append(value.clone())
        return value[:, :32]

    def gru(self, value, previous):
        return value, value.transpose(0, 1) + (0 if previous is None else previous)

    def head(self, value):
        return torch.tensor([[0., 4., 0., 0., 0., 0., 0., 0., 0.]])


def test_frozen_adapter_identity_clock_request_reset_and_actual_spike_hook():
    metadata = inspect_frozen_history_candidate(ROOT, ROOT / CANDIDATE_RELATIVE)
    assert metadata["policy"]["can_deploy"] is False
    contract = load_input_contract(ROOT)
    feedback = Feedback(valid=True, age_s=.05, source="actual", standing=True, mc_mode="STAND_DEFAULT")
    first = Observation("run", "session", 1, "request-1", "SHOW_FORWARD", 1.,
                        feedback=feedback, capabilities=("HOLD", "ADVANCE"), request_changed=True)
    # The channel inventory comes from the original pure encoder, not a mock.
    provisional = dict(task="SHOW_FORWARD", request_changed=True, capabilities=["HOLD", "ADVANCE"],
        protection_state="READY", feedback=dict(valid=True, age_s=.05, status="IDLE", behavior="HOLD",
        same_request=False, event_new=False, standing=True, heading_error_rad=0., vx_m_s=0.,
        wz_rad_s=0., mc_mode="STAND_DEFAULT"))
    engine = FakeEngine(contract[1](contract[0], provisional))
    members = [FakeMember() for _ in range(3)]
    policy = FrozenHistoryPolicy(engine, members, torch.zeros(7110), torch.ones(7110), metadata["policy"], contract)
    state = policy.initial_state("session")
    with pytest.raises(ValueError, match="runtime context"):
        policy(first, state)
    policy.set_runtime_context(1, "READY", -1)
    output = policy(first, state)
    assert output.state.brain.step == 1100 and len(engine.advances) == 1
    assert all(member.inputs[0].shape == (1, 7110) for member in members)
    decision = policy.decision(first, output, checkpoint_id=metadata["checkpoint_id"],
                              state_before_id="before", decision_seq=1)
    assert str(decision.behavior) == "ADVANCE" and str(decision.phase) == "START"
    assert not decision.behavior_logits and not decision.phase_logits
    second = replace(first, observation_seq=2, request_changed=False,
                     feedback=replace(feedback, age_s=.1, vx_m_s=.2, mc_mode="WALK"))
    policy.set_runtime_context(2, "READY", -1)
    held = policy(second, output.state.detach())
    assert len(engine.advances) == 1 and held.state.brain.step == 1100
    assert held.joint_readout["model_advanced_this_frame"] is False
    assert held.joint_readout["source_observation_seq"] == 1
    policy.decision(second, held, checkpoint_id=metadata["checkpoint_id"], state_before_id="before", decision_seq=2)
    # A new actual event fact advances even with unchanged status/behavior/body.
    third = replace(second, observation_seq=3)
    policy.set_runtime_context(3, "READY", 0)
    advanced = policy(third, held.state)
    assert len(engine.advances) == 2 and advanced.state.event_cursor == 0
    fourth = replace(third, observation_seq=4, request_id="request-2", request_changed=True)
    policy.set_runtime_context(4, "READY", 0)
    reset = policy(fourth, advanced.state)
    assert engine.initializations == 1 and reset.state.brain.step == 1100
    assert reset.state.event_cursor == 0 and len(reset.state.prefix.tokens) == 1
    assert all(torch.equal(h, torch.ones_like(h)) for h in reset.state.hidden)
    assert policy.state_identity(reset.state)["state_id"] != policy.state_identity(advanced.state)["state_id"]

    spikes, calls, seen = object(), [], []
    class Owner:
        def step(self, occupancy, brain):
            calls.append(brain.step)
            return SimpleNamespace(step=brain.step + 1), None, None, spikes
    class Telemetry:
        error = None
        def capture(self, value, step):
            seen.append((value, step))
        def flush(self, step):
            seen.append(step)
    owner = Owner()
    with pytest.raises(RuntimeError, match="stop"):
        with capture_actual_steps(owner, "step", Telemetry()):
            assert owner.step(None, SimpleNamespace(step=49))[3] is spikes
            raise RuntimeError("stop")
    assert calls == [49] and seen == [(spikes, 49), 50] and "step" not in vars(owner)
