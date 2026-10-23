"""Small error-monitor injections only; not full-graph equivalence evidence."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from malecns_lab.learning.x2_frozen import FrozenInference, _StickyError


@pytest.fixture(params=("cpu", "mps"))
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable; CPU unit checks do not establish MPS behavior")
    return request.param


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_transient_nonfinite_survives_finite_recovery(device, bad):
    errors = _StickyError(device)
    value = torch.tensor([bad], device=device)
    errors.finite(value)
    value = torch.zeros_like(value)
    errors.finite(value)
    with pytest.raises(ValueError, match="step 50: sticky_bits=1"):
        errors.check(50)


@pytest.mark.parametrize("bit", (2, 4, 8))
def test_transient_range_error_survives_later_valid_state(device, bit):
    errors = _StickyError(device)
    errors.add(torch.tensor([False, True], device=device), bit)
    errors.add(torch.tensor([False, False], device=device), bit)
    with pytest.raises(ValueError, match=f"sticky_bits={bit}"):
        errors.check(50)


def test_sticky_bits_accumulate_without_host_reads_until_check(device, monkeypatch):
    errors = _StickyError(device)
    original = torch.Tensor.item
    reads = []

    def record(value):
        reads.append(value)
        return original(value)

    monkeypatch.setattr(torch.Tensor, "item", record)
    errors.finite(torch.tensor([float("inf")], device=device))
    errors.add(torch.tensor([True], device=device), 4)
    assert reads == []
    with pytest.raises(ValueError, match="sticky_bits=5"):
        errors.check(50)
    assert len(reads) == 1


def test_finite_state_passes_without_erasing_prior_checks(device):
    errors = _StickyError(device)
    errors.finite(torch.tensor([-1., 0., 1.], device=device))
    errors.add(torch.tensor([False], device=device), 2)
    errors.check(50)
    errors.finite(torch.tensor([float("nan")], device=device))
    with pytest.raises(ValueError, match="sticky_bits=1"):
        errors.check(100)


@dataclass
class BrainValues:
    v: torch.Tensor
    g: torch.Tensor
    refractory_until: torch.Tensor
    pending: tuple
    step: int
    model_id: str


@dataclass
class DriveValues:
    phase: torch.Tensor
    event_counts: torch.Tensor
    step: int
    drive_id: str
    spatial_map_id: str


def test_step_detects_overflow_even_when_reset_returns_finite_values(device):
    # Exercise only the error-monitor integration with one injected scalar.
    # There is no graph, policy construction, or synthetic fallback here.
    wrapper = object.__new__(FrozenInference)
    wrapper._brain_state, wrapper._drive_state = BrainValues, DriveValues
    cfg = SimpleNamespace(v_rest_mv=0., v_reset_mv=0., v_threshold_mv=1., surrogate_scale_mv=1.)
    brain = SimpleNamespace(config=cfg, input_mask=torch.tensor([False], device=device),
                            refractory_steps=1, base_weights=None, base_weights_adjoint=None, model_id="fault-injection")
    drive = SimpleNamespace(mapping=SimpleNamespace(n_nodes=1, map_id="fault-injection"),
                            input_indices=torch.empty(0, dtype=torch.int64, device=device),
                            config=SimpleNamespace(event_voltage_mv=1.), drive_id="fault-injection")
    wrapper.policy = SimpleNamespace(brain=brain, drive=drive)
    wrapper._spike = lambda distance, scale: (distance > 0).to(distance.dtype)
    wrapper._matvec = lambda matrix, adjoint, vector: torch.zeros_like(vector)
    zero = torch.zeros(1, device=device)
    maximum = torch.full_like(zero, torch.finfo(torch.float32).max)
    state = BrainValues(zero, maximum, torch.zeros(1, dtype=torch.int64, device=device), (maximum,), 0, brain.model_id)
    sensory = DriveValues(torch.empty(0, device=device), torch.empty(0, dtype=torch.int64, device=device), 0,
                          drive.drive_id, drive.mapping.map_id)
    constants = (1., 1., 1., zero, torch.ones_like(zero), torch.empty(0, device=device))
    errors = _StickyError(device)
    advanced, _ = wrapper._step(state, sensory, constants, errors)
    assert torch.isfinite(advanced.v).all() and torch.isfinite(advanced.g).all()
    assert advanced.v.item() == advanced.g.item() == 0.
    with pytest.raises(ValueError, match="sticky_bits=1"):
        errors.check(50)


@pytest.mark.parametrize("fault_step", (1, 49, 50, 51, 123))
def test_run_stops_at_first_block_after_fault_including_final_remainder(fault_step):
    wrapper = object.__new__(FrozenInference)
    wrapper.policy = SimpleNamespace(brain=SimpleNamespace(refractory_steps=1))
    wrapper._constants = lambda mask: (1., 1., 1., torch.zeros(1), torch.ones(1), torch.zeros(1))
    calls = []

    def step(brain, sensory, constants, errors):
        number = brain.step + 1
        calls.append(number)
        errors.add(torch.tensor(number == fault_step), 2)
        return SimpleNamespace(step=number, v=brain.v), SimpleNamespace(step=number)

    wrapper._step = step
    expected = min(((fault_step - 1) // 50 + 1) * 50, 123)
    with pytest.raises(ValueError, match=f"step {expected}: sticky_bits=2"):
        wrapper._run(SimpleNamespace(step=0, v=torch.zeros(1)), SimpleNamespace(step=0), None, 123)
    assert len(calls) == expected


def test_run_checks_only_block_boundaries_and_final_remainder(monkeypatch):
    wrapper = object.__new__(FrozenInference)
    wrapper.policy = SimpleNamespace(brain=SimpleNamespace(refractory_steps=1))
    wrapper._constants = lambda mask: (1., 1., 1., torch.zeros(1), torch.ones(1), torch.zeros(1))
    wrapper._step = lambda b, d, c, e: (SimpleNamespace(step=b.step + 1, v=b.v), SimpleNamespace(step=d.step + 1))
    observed, original = [], _StickyError.check

    def check(errors, step):
        observed.append(step)
        original(errors, step)

    monkeypatch.setattr(_StickyError, "check", check)
    wrapper._run(SimpleNamespace(step=0, v=torch.zeros(1)), SimpleNamespace(step=0), None, 123)
    assert observed == [50, 100, 123]


def test_constructor_rejects_foreign_policy_before_graph_or_model_access():
    with pytest.raises(ValueError, match="historical cnsbridge GPUPolicy"):
        FrozenInference(object())


def test_forward_rejects_fixture_bypass_argument():
    wrapper = object.__new__(FrozenInference)
    with pytest.raises(TypeError, match="direct_numerical_fixture"):
        wrapper.forward_tensor(None, None, direct_numerical_fixture=True)


def test_execution_identity_is_a_copy_and_has_no_old_policy_id_alias():
    wrapper = object.__new__(FrozenInference)
    wrapper._identity = {"execution_id": "new", "source_policy_id": "old", "graph": {"nodes": 166700}}
    identity = wrapper.execution_identity
    identity["graph"]["nodes"] = 1
    assert wrapper.execution_identity["graph"]["nodes"] == 166700
    assert "policy_id" not in wrapper.execution_identity
