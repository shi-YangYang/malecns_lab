"""Small numerical fixtures only; no fixture serves as a production brain."""

from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm

torch = pytest.importorskip("torch")

from malecns_lab.brain.trainable import (  # noqa: E402
    TrainableLIF,
    TrainableLIFConfig,
    fixed_sparse_matvec,
    surrogate_spike,
)
from malecns_lab.brain.trainable_graph import group_by_type  # noqa: E402


def circuit(counts, inputs=(), **kwargs):
    return TrainableLIF.for_numerical_test(sparse.csc_matrix(np.asarray(counts, dtype=np.int64)),
                                         inputs, dtype=torch.float64, **kwargs)


def config(**kwargs):
    return TrainableLIFConfig(v_rest_mv=0, v_reset_mv=0, v_threshold_mv=1,
                              delay_ms=0.2, refractory_ms=0.3, **kwargs)


def independent_reference(counts, inputs, increments, cfg):
    """Matrix exponential of the two differential equations; separate event loop."""
    n = len(counts)
    flow = expm(np.array([[-1 / cfg.tau_m_ms, 1 / cfg.tau_m_ms], [0, -1 / cfg.tau_syn_ms]]) * cfg.dt_ms)
    v, g, until = np.full(n, cfg.v_rest_mv, dtype=float), np.zeros(n), np.zeros(n, dtype=int)
    arrival = {}
    history = []
    for tick, increment in enumerate(increments):
        g += arrival.pop(tick, np.zeros(n))
        fired = np.zeros(n)
        for cell in range(n):
            enabled = cell in inputs or tick >= until[cell]
            if enabled:
                pair = flow @ np.array([v[cell] - cfg.v_rest_mv, g[cell]])
                v[cell], g[cell] = pair[0] + cfg.v_rest_mv, pair[1]
            v[cell] += increment[cell]
            if enabled and v[cell] > cfg.v_threshold_mv:
                fired[cell] = 1
                v[cell], g[cell] = cfg.v_reset_mv, 0
                if cell not in inputs:
                    until[cell] = tick + round(cfg.refractory_ms / cfg.dt_ms)
        arrival[tick + round(cfg.delay_ms / cfg.dt_ms)] = cfg.weight_mv * counts @ fired
        history.append((fired.copy(), v.copy(), g.copy(), until.copy()))
    return history


@pytest.mark.parametrize("tau_syn", [5.0, 20.0])
def test_exact_integration_events_and_inhibition_match_independent_reference(tau_syn):
    counts = np.array([[0, 0, -7], [100, 0, 0], [0, 130, 0]], dtype=np.int64)
    cfg = config(tau_syn_ms=tau_syn)
    model = circuit(counts, [0], config=cfg)
    increments = np.zeros((60, 3))
    increments[[0, 1, 5, 11, 25], 0] = 1.2
    expected = independent_reference(counts, {0}, increments, cfg)
    state = model.reset()
    assert any(row[0][1] for row in expected)  # Actual recurrent firing, not only silent integration.
    assert any(row[2][0] < 0 for row in expected)  # Inhibitory arrivals really occurred.
    for tick, (increment, (fired, v, g, until)) in enumerate(zip(increments, expected)):
        spikes, state = model.step(torch.tensor(increment), state)
        np.testing.assert_array_equal(spikes.detach().numpy(), fired)
        np.testing.assert_allclose(state.v.detach().numpy(), v, atol=1e-12)
        np.testing.assert_allclose(state.g.detach().numpy(), g, atol=1e-12)
        np.testing.assert_array_equal(state.refractory_until.numpy(), until)
        assert state.step == tick + 1


def test_strict_threshold_input_exemption_and_hand_delay_timeline():
    model = circuit([[0, 0], [4, 0]], [0], config=config(weight_mv=1.0))
    state = model.reset()
    spikes, state = model.step(torch.tensor([1.0, 0.0], dtype=torch.float64), state)
    assert spikes.tolist() == [0, 0]  # Exactly threshold is not firing.
    spikes, state = model.step(torch.tensor([1.1, 0.0], dtype=torch.float64), state)
    assert spikes.tolist() == [1, 0]
    assert state.v[0] == 0 and state.g[0] == 0 and state.refractory_until[0] == 0
    assert state.g[1] == 0 and state.pending[-1][1] == 4
    spikes, state = model.step(torch.tensor([1.1, 0.0], dtype=torch.float64), state)
    assert spikes.tolist() == [1, 0]  # Registered input can fire again immediately.
    assert state.g[1] == 0  # k+1 has no arrival.
    _, state = model.step(torch.zeros(2, dtype=torch.float64), state)
    assert state.g[1].item() == pytest.approx(4 * np.exp(-0.1 / 5))  # k+2 arrival, then decay.
    assert state.v[1] > 0


def test_refractory_freezes_both_states_but_preserves_arrivals_until_exact_boundary():
    model = circuit([[0]], config=config())
    state = replace(model.reset(), v=torch.tensor([1.1], dtype=torch.float64))
    spikes, state = model.step(torch.zeros(1, dtype=torch.float64), state)
    assert spikes.item() == 1 and state.refractory_until.item() == 3
    # The arrival at step 1 is preserved while the neuron is blocked at steps 1 and 2.
    state = replace(state, pending=(torch.tensor([500.0], dtype=torch.float64), state.pending[1]))
    for tick in (1, 2):
        spikes, state = model.step(torch.zeros(1, dtype=torch.float64), state)
        assert spikes.item() == 0 and state.v.item() == 0 and state.g.item() == 500
        assert state.step == tick + 1
    spikes, state = model.step(torch.zeros(1, dtype=torch.float64), state)
    assert spikes.item() == 1  # Exactly k+ref_steps allows integration and firing.
    assert state.v.item() == state.g.item() == 0


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_binary_spike_backward_is_declared_surrogate_not_hard_threshold_derivative(scale):
    distance = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=torch.float64, requires_grad=True)
    spikes = surrogate_spike(distance, scale)
    assert spikes.tolist() == [0, 0, 0, 1, 1]
    spikes.sum().backward()
    expected = 1 / (scale * (1 + np.abs(distance.detach().numpy() / scale)) ** 2)
    np.testing.assert_allclose(distance.grad.numpy(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight_mv", [0.275, 1.0, 1.7])
@pytest.mark.parametrize("active_value", [0.0, 1.0])
def test_sparse_forward_and_cached_all_cell_adjoint_equal_dense_algebra(dtype, weight_mv, active_value):
    counts = np.array([[0, -2, 0, 0], [3, 0, 5, 0], [0, 4, -7, 0], [0, 0, 0, 0]], dtype=np.int64)
    model = TrainableLIF.for_numerical_test(sparse.csc_matrix(counts), dtype=dtype,
                                           config=config(weight_mv=weight_mv))
    dense = torch.tensor(counts, dtype=dtype) * weight_mv
    for matrix, expected in ((model.base_weights, dense), (model.base_weights_adjoint, dense.T)):
        assert matrix.layout == torch.sparse_csr and matrix.dtype == dtype and not matrix.requires_grad
        assert matrix._nnz() == np.count_nonzero(counts)
        torch.testing.assert_close(matrix.to_dense(), expected, rtol=0, atol=0)
    assert set(model.state_dict()) == {"theta"}  # Neither fixed direction is checkpoint content.
    np.testing.assert_array_equal(model.graph.counts.toarray(), counts)
    assert not model.graph.counts.data.flags.writeable
    vector = torch.tensor([0.0, active_value, 0.0, 0.0], dtype=dtype, requires_grad=True)
    upstream = torch.tensor([2.0, 3.0, -1.0, 2.0], dtype=dtype)
    actual = fixed_sparse_matvec(model._forward_csr, model.base_weights_adjoint, vector)
    (actual * upstream).sum().backward()
    torch.testing.assert_close(actual, dense @ vector)
    torch.testing.assert_close(vector.grad, dense.T @ upstream)
    # Zero-valued cells retain nonzero gradients, including an entirely silent vector.
    assert vector.grad[0] != 0 and vector.grad[2] != 0 and vector.grad[3] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_cache_readonly_shared_storage_lifetime_and_noncontiguous_input(dtype):
    counts = np.array([[0, -2, 0], [3, 0, 5], [0, 4, -7]], dtype=np.int64)
    model = TrainableLIF.for_numerical_test(sparse.csc_matrix(counts), dtype=dtype)
    matrix, adjoint = model._forward_csr, model.base_weights_adjoint
    assert isinstance(matrix, sparse.csr_array)
    assert model.identity["backend"] == "scipy_cpu_csr_forward_torch_csr_adjoint"
    assert set(model.state_dict()) == {"theta"}
    for view, tensor in zip((matrix.data, matrix.indices, matrix.indptr),
                            (model.base_weights.values(), model.base_weights.col_indices(), model.base_weights.crow_indices())):
        assert np.shares_memory(view, tensor.numpy())
        assert view.dtype == tensor.numpy().dtype and not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0] = 0
    # Once constructed, repeated steps must use this exact cached sparse view.
    model.run(torch.zeros((3, 3), dtype=dtype))
    assert model._forward_csr is matrix
    np.testing.assert_array_equal(model.graph.counts.toarray(), counts)
    # Retaining the view is sufficient to retain its underlying Torch buffers.
    reference = weakref.ref(model)
    del model, tensor
    gc.collect()
    assert reference() is None
    source = torch.tensor([1.0, 99.0, 0.0, 99.0, -0.5, 99.0], dtype=dtype, requires_grad=True)
    vector = source[::2]
    assert not vector.is_contiguous()
    original = source.detach().clone()
    output = fixed_sparse_matvec(matrix, adjoint, vector)
    expected_matrix = torch.tensor(counts, dtype=dtype) * 0.275
    torch.testing.assert_close(output, expected_matrix @ vector)
    assert output.dtype == dtype and output.device.type == "cpu"
    upstream = torch.tensor([2.0, 3.0, -1.0], dtype=dtype)
    (output * upstream).sum().backward()
    expected_gradient = torch.zeros_like(source)
    expected_gradient[::2] = expected_matrix.T @ upstream
    torch.testing.assert_close(source.grad, expected_gradient)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    assert not np.shares_memory(output.detach().numpy(), source.detach().numpy())
    assert not np.shares_memory(output.detach().numpy(), matrix.data)
    output.detach().fill_(123)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    np.testing.assert_array_equal(matrix.toarray(), expected_matrix.numpy())


def test_silent_cell_surrogate_survives_recurrent_delay_and_sparse_propagation():
    model = circuit([[0, 0], [3, 0]], [0], config=config(weight_mv=1.0))
    first = torch.tensor([0.5, 0.0], dtype=torch.float64, requires_grad=True)
    spikes, state = model.run([first, torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)])
    assert not spikes.any() and state.g[1] == 0
    state.g[1].backward()
    expected = 3 * np.exp(-0.1 / 5) / (1 + 0.5) ** 2
    assert first.grad[0].item() == pytest.approx(expected)
    assert first.grad[1].item() == 0  # Unregistered external channel cannot become a bypass.


def test_internal_gain_update_changes_recurrent_response_and_restores_with_parameters():
    model = circuit([[0, 0], [3, 0]], [0], types=["pre", "post"], config=config(weight_mv=1.0))
    increments = torch.tensor([[1.1, 0], [0, 0], [0, 0]], dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _, state = model.run(increments)
    before = state.g[1].detach().clone()
    (state.g[1] - 0.2).square().backward()
    pre_group = model.graph.group_names.index("pre")
    assert model.theta.grad[pre_group] != 0
    optimizer.step()
    assert model.theta[pre_group] != 0
    _, changed = model.run(increments)
    assert changed.g[1] < before
    with torch.no_grad():
        model.theta.zero_()
    _, restored = model.run(increments)
    torch.testing.assert_close(restored.g[1], before)
    assert model.graph.counts[1, 0] == 3


def test_grouping_bounds_and_immutable_raw_counts_are_explicit():
    labels, indices = group_by_type(np.array(["z", "a", "", " ", "z", "hidden"]),
                                   np.array([False, False, False, False, False, True]))
    assert labels == (None, "a", "z") and indices.tolist() == [2, 1, 0, 0, 2, 0]
    model = circuit([[0, 16777217], [-2, 0]], types=["z", "a"])
    assert model.gains.tolist() == [1, 1, 1]
    assert model.graph.counts.dtype == np.int64 and model.graph.counts[0, 1] == 16777217
    with pytest.raises(ValueError):
        model.graph.counts.data[0] = 0
    with torch.no_grad():
        model.theta.copy_(torch.tensor([-1e6, 1e6, 0.0]))
    assert torch.all(model.gains > 0.5) and torch.all(model.gains < 1.5)
    assert torch.sign(model.base_weights.values()).tolist() == [1, -1]
    assert model.identity["graph"]["role"] == "numerical_test_only"
    assert model.identity["graph"]["synapse_count"] == 16777219


def test_contiguous_windows_reset_and_detach_include_delayed_gradients():
    model = circuit([[0, 0], [3, 0]], [0], config=config())
    increments = torch.zeros((12, 2), dtype=torch.float64)
    increments[::3, 0] = 1.1
    entire, end = model.run(increments)
    first, middle = model.run(increments[:4])
    latter, split_end = model.run(increments[4:], middle)
    torch.testing.assert_close(torch.cat([first, latter]), entire)
    for a, b in zip((end.v, end.g, *end.pending), (split_end.v, split_end.g, *split_end.pending)):
        torch.testing.assert_close(a, b)
    assert end.step == split_end.step and torch.equal(end.refractory_until, split_end.refractory_until)
    assert any(item.requires_grad for item in middle.pending)
    detached = middle.detach()
    for a, b in zip((middle.v, middle.g, *middle.pending), (detached.v, detached.g, *detached.pending)):
        torch.testing.assert_close(a, b)
        assert b.grad_fn is None and not b.requires_grad
    assert detached.step == middle.step
    fresh = model.reset()
    assert fresh.step == 0 and not fresh.g.any() and not any(item.any() for item in fresh.pending)


@pytest.mark.parametrize("kwargs", [{"dt_ms": 0.07}, {"delay_ms": 0}, {"tau_m_ms": -1},
                                    {"v_threshold_mv": float("nan")}, {"v_reset_mv": -40}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        TrainableLIFConfig(**kwargs)


def test_invalid_input_state_and_parameter_values_fail_explicitly():
    model = circuit([[0, 0], [1, 0]], [0])
    for invalid in (torch.tensor([1.0], dtype=torch.float64), torch.tensor([float("nan"), 0.0]),
                    torch.tensor([0.0, 1.0], dtype=torch.float64), torch.zeros(2, dtype=torch.float32)):
        with pytest.raises(ValueError):
            model.step(invalid, model.reset())
    zero = torch.zeros(2, dtype=torch.float64)
    for state in (replace(model.reset(), step=-1), replace(model.reset(), model_id="foreign"),
                  replace(model.reset(), pending=()), replace(model.reset(), g=zero + float("inf"))):
        with pytest.raises(ValueError):
            model.step(zero, state)
    with torch.no_grad():
        model.theta.fill_(float("inf"))
    with pytest.raises(ValueError, match="parameters"):
        model.step(zero, model.reset())


def test_production_entry_rejects_filtered_or_research_asset_without_fallback(tmp_path):
    from malecns_lab.brain.connectome import Connectome
    for role in ("filtered", "research_control"):
        graph = Connectome("not-full", sparse.csc_matrix(np.array([[1]], dtype=np.int64)),
                            np.array(["type"]), np.array(["L"]), body_ids=np.array([1]),
                            meta={"asset_type": role, "format_version": 2})
        path = graph.save(tmp_path / f"{role}.npz")
        with pytest.raises(ValueError, match="full_reference"):
            TrainableLIF.from_reference(path, [1])
    with pytest.raises(FileNotFoundError):
        TrainableLIF.from_reference(tmp_path / "missing.npz", [])
