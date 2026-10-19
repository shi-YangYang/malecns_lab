"""Small numerical circuits and metadata-only fakes; no complete biological run."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from cnsbridge.brain.metadata import annotation_hash, canonical_hash
from cnsbridge.brain.policy_groups import _build_policy_map, load_policy_neuron_map, numerical_test_policy_map
from cnsbridge.brain.tonic import LaminaTonicLIF
from cnsbridge.brain.trainable import TrainableLIFConfig, fixed_sparse_matvec
from cnsbridge.brain.trainable_graph import _graph, numerical_test_graph
from cnsbridge.learning.neural_policy import NeuralPolicy, NeuralPolicyConfig
from cnsbridge.learning.research_admission import factory_sources
from cnsbridge.learning.research_cache import validate_cache_correspondence, validate_membership
from cnsbridge.learning.research_contract import common_numeric_contract, require_common_numeric_contract
from cnsbridge.learning.research_policy import _new_brain, _finish_policy, from_terminal_research_run
from cnsbridge.senses.spatial_drive import SpatialDriveConfig
from cnsbridge.senses.target import TargetCamera

CONFIG = NeuralPolicyConfig(frame_ms=10, precondition_ms=0, control_dt_s=.1, readout_seed=1703)
LIF = TrainableLIFConfig(dt_ms=1, delay_ms=1, refractory_ms=2)
CAMERA = TargetCamera()
DRIVE = SpatialDriveConfig(dt_ms=1, max_rate_hz=1000, phase_seed=1702)


def data(rewired=False):
    counts = np.zeros((9, 9), np.int64)
    counts[2, 0], counts[3, 1] = 20, -10
    counts[8 if rewired else 7, 2], counts[7 if rewired else 8, 3] = 15, 10
    counts[4, 4] = 3  # Keep a self edge; one DN's type is explicitly missing.
    types = np.array(["R1-R6", "R1-R6", "L1", "L2", "L3", "L4", "L5", "DNtest", ""])
    annotations = {"type": types, "superclass": np.array(["fixture"] * 7 + ["descending_neuron"] * 2),
                   "rootSide": np.array(["L", "R"] + ["M"] * 7), "somaSide": np.array(["M"] * 9)}
    missing = {name: np.zeros(9, bool) for name in annotations}
    missing["type"][8] = True
    graph = numerical_test_graph(sparse.csc_matrix(counts), types, missing["type"])
    mapping = numerical_test_policy_map(graph, annotations, missing)
    return graph, mapping, annotations, missing


def installed(rewired=True, *, replacement=None):
    return _new_brain(lambda: data()[:2], replacement or (lambda: data(rewired)), lif_config=LIF)


def policy(rewired=True):
    brain, mapping, evidence = installed(rewired)
    # Explicit numerical-only identity. This fake admission cannot enter the public factory.
    admission = {"admission_id": "numerical_test_only", "factory_sources": factory_sources()}
    return _finish_policy(brain, mapping, evidence, admission, config=CONFIG, camera=CAMERA, drive_config=DRIVE)


def baseline():
    graph, mapping, _, _ = data()
    brain = LaminaTonicLIF(graph, mapping.inputs.body_ids, mode="fixed", config=LIF)
    return NeuralPolicy.for_numerical_test(brain, mapping, background="fixed_lamina_8mv", config=CONFIG,
                                         camera=CAMERA, drive_config=DRIVE)


def test_same_graph_install_exact_initial_parameters_state_and_temporal_forward():
    old, new = baseline(), policy(False)
    require_common_numeric_contract(common_numeric_contract(old), common_numeric_contract(new))
    assert old.policy_id != new.policy_id  # Adapter identity is deliberately not hidden.
    assert type(new) is NeuralPolicy and type(new.brain) is LaminaTonicLIF
    assert type(new).forward_tensor is NeuralPolicy.forward_tensor
    assert new.brain.step.__func__ is old.brain.step.__func__
    a, b = old.initial_state(), new.initial_state()
    image = np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8)
    image[:, :, 2] = 255
    first, second = old.forward_tensor(image, a), new.forward_tensor(image, b)
    assert torch.equal(first.action, second.action) and torch.equal(first.features_mv, second.features_mv)
    for x, y in zip((first.state.brain.v, first.state.brain.g, *first.state.brain.pending),
                    (second.state.brain.v, second.state.brain.g, *second.state.brain.pending), strict=True):
        assert torch.equal(x, y)
    assert new.brain.graph.identity["role"] == "numerical_test_only"


def test_different_graph_actual_forward_and_all_cell_adjoint_independent_reference():
    old, new = policy(False), policy(True)
    require_common_numeric_contract(common_numeric_contract(old), common_numeric_contract(new))
    w = new.brain.graph.counts.toarray().astype(np.float32) * np.float32(LIF.weight_mv)
    x = torch.tensor([1., 0., 2., 0., 0., 0., 0., 0., 0.], requires_grad=True)
    coefficients = torch.arange(1, 10, dtype=torch.float32)
    y = fixed_sparse_matvec(new.brain._forward_csr, new.brain.base_weights_adjoint, x)
    np.testing.assert_array_equal(y.detach().numpy(), w @ x.detach().numpy())
    (y * coefficients).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), w.T @ coefficients.numpy())
    assert x[1] == 0 and x.grad[1] != 0  # A silent cell still receives the real adjoint.
    assert not np.array_equal(old.brain._forward_csr @ x.detach().numpy(), y.detach().numpy())
    stimulus = torch.zeros(9)
    stimulus[:2] = 68.75
    spikes, next_state = new.brain.step(stimulus, new.brain.reset())
    np.testing.assert_array_equal(next_state.pending[-1].detach().numpy(), w @ spikes.detach().numpy())
    assert new.brain.identity["graph"] == new.brain.graph.identity
    assert new.brain.graph.identity["raw_ordered_counts_sha256"] != old.brain.graph.identity["raw_ordered_counts_sha256"]
    assert new.policy_id != old.policy_id and new.spatial_map.map_id != old.spatial_map.map_id


@pytest.mark.parametrize("corruption", ["old_forward", "old_adjoint", "old_scipy", "sign", "transpose", "dtype", "pointer", "writable"])
def test_complete_cache_checker_rejects_wrong_storage_and_graph(corruption):
    old, new = policy(False), policy(True)
    brain = new.brain
    if corruption == "old_forward":
        brain.base_weights = old.brain.base_weights
    elif corruption == "old_adjoint":
        brain.base_weights_adjoint = old.brain.base_weights_adjoint
    elif corruption == "old_scipy":
        brain._forward_csr = old.brain._forward_csr
    elif corruption == "sign":
        brain.base_weights.values()[0].neg_()
    elif corruption == "transpose":
        brain.base_weights = brain.base_weights_adjoint
    elif corruption == "dtype":
        brain.base_weights = brain.base_weights.double()
    elif corruption == "pointer":
        brain.base_weights.crow_indices()[2] += 1
    else:
        brain._forward_csr.data = brain.base_weights.values().numpy()  # Same storage, newly writable Torch view.
    with pytest.raises(ValueError):
        validate_cache_correspondence(brain)


def test_every_raw_count_bound_and_read_only_shared_arrays():
    brain, _, report = installed()
    assert report["edges_checked"] == brain.graph.counts.nnz == 5
    arrays = (brain._forward_csr.data, brain._forward_csr.indices, brain._forward_csr.indptr)
    tensors = (brain.base_weights.values(), brain.base_weights.col_indices(), brain.base_weights.crow_indices())
    for array, tensor in zip(arrays, tensors, strict=True):
        assert array.ctypes.data == tensor.data_ptr() and not array.flags.writeable
    brain.graph.counts.data.flags.writeable = True
    brain.graph.counts.data[0] += 1
    brain.graph.counts.data.flags.writeable = False
    with pytest.raises(ValueError, match="raw counts"):
        validate_cache_correspondence(brain)


@pytest.mark.parametrize("field", ["body_order", "groups", "input", "tonic", "rho"])
def test_actual_membership_mismatch_rejected(field):
    graph, mapping, annotations, missing = data(True)
    if field == "body_order":
        bad = replace(graph, body_ids=graph.body_ids[::-1])
        with pytest.raises(ValueError, match="body order"):
            installed(replacement=lambda: (bad, mapping, annotations, missing))
        return
    brain, mapping, _ = installed()
    if field == "groups":
        brain.group_index[0] = 0
    elif field == "input":
        brain.input_mask[2] = True
    elif field == "tonic":
        brain.tonic_group_index[3] = 0
    else:
        brain.rho[0] = .1
    with pytest.raises(ValueError):
        validate_membership(brain, mapping, annotations, missing)


def test_state_isolation_factory_takes_no_existing_state_and_old_entry_rejects():
    old, new = policy(False), policy(True)
    with pytest.raises(ValueError, match="identity"):
        new.brain.step(torch.zeros(9), old.brain.reset())
    with pytest.raises(ValueError, match="identity"):
        new.forward_tensor(np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8), old.initial_state())
    with pytest.raises(TypeError):
        from_terminal_research_run({}, config=CONFIG, camera=CAMERA, drive_config=DRIVE, lif_config=LIF, state=old.initial_state())
    with pytest.raises(ValueError, match="directory"):
        from_terminal_research_run(new.brain.graph, config=CONFIG, camera=CAMERA, drive_config=DRIVE, lif_config=LIF)
    from cnsbridge.learning.policy_checkpoint import _validate_identity
    forged = deepcopy(new.identity)
    forged["role"] = "research_control"  # Negative role fixture, no neural run under a false biological identity.
    forged["policy_id"] = canonical_hash({k: v for k, v in forged.items() if k != "policy_id"})
    with pytest.raises(ValueError):
        _validate_identity(forged)
    research = _graph(new.brain.graph.counts, new.brain.graph.body_ids, data()[2]["type"], data()[3]["type"],
                      {"role": "research_control", "reference_source_id": None, "fixture": "numerical_test_only"})
    with pytest.raises(ValueError, match="full_reference"):
        load_policy_neuron_map(research, "not-a-reference")


def test_common_contract_rejects_changed_initial_parameters_or_hidden_dynamics():
    model = policy()
    original = common_numeric_contract(model)
    changed = deepcopy(original)
    changed["camera"]["horizontal_fov_deg"] += 1
    changed["common_contract_id"] = canonical_hash({k: v for k, v in changed.items() if k != "common_contract_id"})
    with pytest.raises(ValueError, match="differ"):
        require_common_numeric_contract(original, changed)
    model.brain._identity["unregistered_dynamics"] = "must not be stripped"
    model._identity["model"] = model.brain.identity
    with pytest.raises(ValueError, match="Unknown dynamics"):
        common_numeric_contract(model)
    with torch.no_grad():
        model.readout.weight[0, 0] += .1
    with pytest.raises(ValueError, match="initial readout"):
        common_numeric_contract(model)


def test_fresh_constructor_releases_old_storage_before_loading_and_never_steps(monkeypatch):
    import weakref
    from cnsbridge.brain.trainable import TrainableLIF
    references = []
    original_init = LaminaTonicLIF.__init__
    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        references.extend(weakref.ref(v) for v in (self._forward_csr, self.base_weights, self.base_weights_adjoint, self.graph))
    monkeypatch.setattr(LaminaTonicLIF, "__init__", observed_init)
    def no_dynamics(*args, **kwargs):
        raise AssertionError("Factory must not create state or perform neural computation")
    monkeypatch.setattr(TrainableLIF, "reset", no_dynamics)
    monkeypatch.setattr(TrainableLIF, "step", no_dynamics)
    def replacement():
        assert all(ref() is None for ref in references)
        return data(True)
    brain, _, _ = installed(replacement=replacement)
    assert type(brain) is LaminaTonicLIF


@pytest.mark.parametrize("mutation", ["lif_config", "policy_clock", "drive", "camera"])
def test_common_contract_rejects_live_configuration_changed_after_construction(mutation):
    model = policy()
    if mutation == "lif_config":
        model.brain.config = replace(model.brain.config, tau_m_ms=30)
    elif mutation == "policy_clock":
        model.config = replace(model.config, frame_ms=12)
    elif mutation == "drive":
        model.drive_config = replace(model.drive_config, max_rate_hz=500)
    else:
        model.camera = replace(model.camera, horizontal_fov_deg=80)
    with pytest.raises(ValueError, match="Live"):
        common_numeric_contract(model)
