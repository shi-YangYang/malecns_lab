"""Real MaleCNS integration and connectome storage contracts; no flight claims."""

import numpy as np
import pytest
from scipy import sparse

import cnsbridge.brain as brain_package
from cnsbridge.brain import Brain, Connectome, load_connectome


def test_real_malecns_load_groups_and_short_step(malecns_graph, malecns_config):
    brain = Brain(malecns_graph, malecns_config)
    assert brain.n_neurons == 166700
    assert brain.connectome.name == "malecns-v1.0"
    assert not brain.connectome.meta.get("synthetic", False)
    assert all(brain.connectome.group(name).size for name in malecns_config["outputs"])
    assert set(brain.empty) == {name for name in (*brain.input_specs, *brain.output_specs)
                                if brain.connectome.group(name).size == 0}
    rates = brain.tick({"T4c_L": 80.0}, 50.0)
    assert brain.net.t_ms == 50.0
    assert brain.last_counts.shape == (166700,)
    assert np.isfinite(list(rates.values())).all()
    assert (brain.last_counts >= 0).all()
    print(f"REAL MALECNS: {brain.n_neurons} neurons; {brain.connectome.n_connections} connections; 50 ms neural step; "
          f"{int(brain.last_counts.sum())} spikes; missing groups={brain.empty}")


def test_real_graph_copy_has_independent_state(malecns_graph, malecns_config):
    brain = Brain(malecns_graph, malecns_config)
    copied = brain.copy(seed=99)
    assert copied.connectome is brain.connectome
    brain.net.v[0] += 1
    assert brain.net.v[0] != copied.net.v[0]


def storage_fixture():
    # Disconnected storage rows, not a runnable sensorimotor model.
    return Connectome("storage_unit_fixture", sparse.csc_matrix((3, 3)),
                      np.array(["A", "B", "C"]), np.array(["L", "R", ""]))


def test_save_load_preserves_storage_rows(tmp_path):
    source = storage_fixture()
    source.groups["x"] = np.array([1, 2])
    loaded = Connectome.load(source.save(tmp_path / "storage.npz"))
    assert loaded.n == source.n and loaded.n_connections == source.n_connections
    np.testing.assert_array_equal(loaded.group("x"), [1, 2])


def test_removed_aliases_and_generator_are_unavailable():
    assert not hasattr(brain_package, "build_minifly")
    for source in ("minifly", "mini", "synthetic"):
        with pytest.raises((ValueError, FileNotFoundError)):
            load_connectome(source)


def test_explicit_synthetic_npz_rejected_and_missing_real_asset_fails(tmp_path):
    graph = storage_fixture()
    graph.meta["synthetic"] = True
    path = graph.save(tmp_path / "marked_synthetic.npz")
    with pytest.raises(ValueError, match="(?i)synthetic"):
        load_connectome(path)
    with pytest.raises(FileNotFoundError):
        load_connectome(tmp_path / "missing_real_graph.npz")
