"""Independent tiny numerical references; no fixture is a replacement fly brain."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from cnsbridge.brain.tonic import LAMINA_TYPES, TONIC_SCHEMA, LaminaTonicLIF  # noqa: E402
from cnsbridge.brain.trainable import TrainableLIF  # noqa: E402


def tonic(types=("L1", "other"), *, tonic_types=("L1",), dtype=torch.float64, **kwargs):
    return LaminaTonicLIF.for_numerical_test(
        sparse.csc_matrix((len(types), len(types)), dtype=np.int64), types=types,
        tonic_types=tonic_types, dtype=dtype, **kwargs,
    )


def test_default_zero_model_stays_separate_and_exactly_at_rest():
    model = TrainableLIF.for_numerical_test(sparse.csc_matrix((2, 2), dtype=np.int64), types=["L1", "other"])
    assert set(dict(model.named_parameters())) == {"theta"} and not hasattr(model, "rho")
    assert model._equilibrium_offset_mv() is None
    spikes, state = model.run(torch.zeros(20, 2))
    assert not spikes.any()
    assert torch.equal(state.v, torch.full((2,), -52.0))


def test_analytic_constant_flow_strict_first_spike_reset_and_refractory():
    model = tonic()
    cfg = model.config
    # Solve rest + offset*(1-exp(-t/tau)) > threshold independently.
    crossing_ms = -cfg.tau_m_ms * math.log(1 - (cfg.v_threshold_mv - cfg.v_rest_mv) / 8)
    first_step = math.floor(crossing_ms / cfg.dt_ms) + 1
    assert first_step == 416
    state = model.reset()
    zero = torch.zeros(2, dtype=torch.float64)
    with torch.no_grad():
        for completed in range(1, first_step):
            spikes, state = model.step(zero, state)
            expected = cfg.v_rest_mv + 8 * (1 - math.exp(-completed * cfg.dt_ms / cfg.tau_m_ms))
            assert not spikes.any()
            assert state.v[0].item() == pytest.approx(expected, abs=1e-11)
            assert state.v[1].item() == cfg.v_rest_mv
            assert not state.g.any()
        spikes, state = model.step(zero, state)
        assert spikes.tolist() == [1, 0]
        assert state.v.tolist() == [cfg.v_reset_mv, cfg.v_rest_mv] and not state.g.any()
        until = first_step - 1 + round(cfg.refractory_ms / cfg.dt_ms)
        assert state.refractory_until.tolist() == [until, 0]
        # An inhibitory arrival during refractoriness accumulates, without tonic integration.
        state = replace(state, pending=(torch.tensor([-2.0, 0.0], dtype=torch.float64), *state.pending[1:]))
        while state.step < until:
            spikes, state = model.step(zero, state)
            assert not spikes.any() and state.v[0] == cfg.v_reset_mv and state.g[0] == -2
        spikes, state = model.step(zero, state)
    a, b = math.exp(-cfg.dt_ms / cfg.tau_m_ms), math.exp(-cfg.dt_ms / cfg.tau_syn_ms)
    coupling = cfg.tau_syn_ms / (cfg.tau_m_ms - cfg.tau_syn_ms) * (a - b)
    assert state.v[0].item() == pytest.approx(cfg.v_reset_mv + 8 * (1 - a) - 2 * coupling)
    assert state.g[0].item() == pytest.approx(-2 * b)
    assert not model.input_mask.any() and not spikes.any()


def test_fixed_and_learnable_share_forward_but_parameter_and_state_identities_differ():
    learned, fixed = tonic(), tonic(mode="fixed")
    assert set(dict(learned.named_parameters())) == {"theta", "rho"}
    assert set(dict(fixed.named_parameters())) == {"theta"}
    assert "rho" in dict(fixed.named_buffers()) and not fixed.rho.requires_grad
    increments = torch.zeros(8, 2, dtype=torch.float64)
    a, sa = learned.run(increments)
    b, sb = fixed.run(increments)
    assert torch.equal(a, b) and torch.equal(sa.v, sb.v) and torch.equal(sa.g, sb.g)
    assert learned.model_id != fixed.model_id
    with pytest.raises(ValueError, match="identity"):
        learned.step(increments[0], sb)
    before = fixed.rho.clone()
    optimizer = torch.optim.SGD(fixed.parameters(), lr=0.1)
    fixed.theta.sum().backward()
    optimizer.step()
    assert torch.equal(fixed.rho, before)


def test_rho_has_analytic_voltage_gradient_updates_and_survives_reset():
    model = tonic()
    _, state = model.step(torch.zeros(2, dtype=torch.float64), model.reset())
    state.v[0].backward()
    assert model.rho.grad.item() == pytest.approx(4 * (1 - math.exp(-0.1 / 20)))
    before = model.rho.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    optimizer.step()
    assert not torch.equal(model.rho, before)
    learned = model.rho.detach().clone()
    assert model.reset().step == 0 and torch.equal(model.rho, learned)
    _, changed = model.step(torch.zeros(2, dtype=torch.float64), model.reset())
    assert changed.v[0] < state.v[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_finite_rho_stays_strictly_bounded_and_nonmembers_exactly_zero(dtype):
    model = tonic(types=("L1", "L2", "other"), tonic_types=("L1", "L2"), dtype=dtype)
    with torch.no_grad():
        model.rho.copy_(torch.tensor([-1e6, 1e6], dtype=dtype))
    offsets = model.tonic_offsets_mv
    assert (offsets > 0).all() and (offsets < 16).all()
    assert model._equilibrium_offset_mv()[2].item() == 0


@pytest.mark.parametrize("value", [torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.float32),
                                   torch.tensor([float("nan")], dtype=torch.float64),
                                   torch.tensor([float("inf")], dtype=torch.float64),
                                   torch.zeros(1, dtype=torch.float64, device="meta")])
def test_step_rejects_invalid_rho_before_any_state_change(value):
    model = tonic()
    model.rho = torch.nn.Parameter(value)
    state = model.reset()
    before = state.v.clone()
    with pytest.raises(ValueError, match="rho"):
        model.step(torch.zeros(2, dtype=torch.float64), state)
    assert state.step == 0 and torch.equal(state.v, before)


@pytest.mark.parametrize("selected", [(), ("L1", "L1"), ("L2", "L1"), ("DN",), (" L1",)])
def test_test_factory_rejects_noncanonical_or_nonlamina_groups(selected):
    with pytest.raises(ValueError, match="ordered subset"):
        tonic(tonic_types=selected)


def test_types_must_exist_with_annotation_and_must_not_overlap_input():
    for types, missing in [([" L1", "other"], None), (["L1", "other"], [True, False]),
                           (["", "other"], None)]:
        with pytest.raises(ValueError, match="no annotated members"):
            tonic(types=types, missing=missing)
    with pytest.raises(ValueError, match="overlap"):
        tonic(input_indices=[0])
    with pytest.raises(ValueError, match="mode"):
        tonic(mode="off")


def test_all_groups_constructor_retains_graph_counts_members_and_independent_identity():
    types = ("L1", "L2", "L3", "L4", "L5", "L1", "L1 ", "")
    counts = sparse.csc_matrix((np.array([-1, 7, 2], dtype=np.int64), ([0, 2, 6], [1, 2, 7])), shape=(8, 8))
    base = TrainableLIF.for_numerical_test(counts, [7], types=types)
    original = base.graph.identity
    model = LaminaTonicLIF(base.graph, [7], dtype=torch.float64)
    assert model.graph is base.graph and model.graph.identity == original
    assert model.identity["graph"] == base.identity["graph"]
    np.testing.assert_array_equal(model.graph.counts.toarray(), counts.toarray())
    assert model.graph.n == 8 and model.graph.counts.nnz == 3
    assert model.tonic_types == LAMINA_TYPES and model.rho.shape == (5,)
    assert model.tonic_group_index.tolist() == [1, 2, 3, 4, 5, 1, 0, 0]
    assert model.input_mask.tolist() == base.input_mask.tolist()
    identity = model.identity
    assert identity["schema"] == TONIC_SCHEMA and identity["graph"]["role"] == "numerical_test_only"
    assert identity["tonic"]["types"] == list(LAMINA_TYPES)
    assert identity["tonic"]["member_count"] == 6
    assert [entry["count"] for entry in identity["tonic"]["members"]] == [2, 1, 1, 1, 1]
    assert all(len(entry["body_ids_sha256"]) == 64 for entry in identity["tonic"]["members"])
    assert "tonic.py" in identity["implementation_sha256"]
    identity["tonic"]["members"][0]["count"] = 500
    assert model.identity["tonic"]["members"][0]["count"] == 2
    with pytest.raises(ValueError, match="identity"):
        model.step(torch.zeros(8, dtype=torch.float64), base.reset())
    # The ordinary constructor cannot silently reduce the five-group contract.
    with pytest.raises(ValueError, match="no annotated members"):
        LaminaTonicLIF(tonic().graph, [])
    with pytest.raises(TypeError):
        LaminaTonicLIF(base.graph, [7], tonic_types=("L1",))


def test_variant_production_entry_rejects_missing_asset_without_test_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        LaminaTonicLIF.from_reference(tmp_path / "missing.npz", [])
