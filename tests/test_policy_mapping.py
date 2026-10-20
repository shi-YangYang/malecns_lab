"""Numerical interface fixtures only: no full graph or brain dynamics are run."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from malecns_lab.brain.metadata import annotation_hash, canonical_hash
from malecns_lab.brain.policy_groups import (
    _read_bound_metadata,
    load_policy_neuron_map,
    numerical_test_policy_map,
)
from malecns_lab.brain.trainable_graph import numerical_test_graph
from malecns_lab.senses.spatial_drive import (
    SpatialDrive,
    SpatialDriveConfig,
    build_spatial_input_map,
)


@pytest.fixture
def fixture_metadata():
    # Includes an untyped descending neuron and a DN-prefixed non-descending
    # neuron, so prefix selection would silently produce the wrong members.
    types = np.asarray(["R1-R6"] * 600 + ["", "DNa01", "other", "DNfake", "R1-R6x"])
    n = len(types)
    annotations = {
        "type": types,
        "superclass": np.asarray(["optic"] * 600 + ["descending_neuron"] * 3 + ["central"] * 2),
        "rootSide": np.asarray(["L"] * 401 + ["R"] * 193 + [""] * 6 + ["L", "M", "R", "L", "R"]),
        "somaSide": np.asarray(["M"] * n),
    }
    missing = {key: np.zeros(n, dtype=bool) for key in annotations}
    missing["type"][600] = True
    missing["rootSide"][597:600] = True
    graph = numerical_test_graph(sparse.csc_matrix((n, n), dtype=np.int64), types, missing["type"])
    return graph, annotations, missing


@pytest.fixture
def policy(fixture_metadata):
    return numerical_test_policy_map(*fixture_metadata)


def test_exact_members_preserve_unknown_type_missing_side_and_source_identity(policy):
    assert policy.identity["role"] == "numerical_test_only"
    np.testing.assert_array_equal(policy.inputs.body_ids, np.arange(600))
    np.testing.assert_array_equal(policy.outputs.body_ids, [600, 601, 602])
    assert policy.inputs.root_side_missing.sum() == 3
    assert policy.outputs.root_side.tolist() == ["L", "M", "R"]
    assert policy.outputs.soma_side.tolist() == ["M"] * 3
    assert policy.identity["selection"]["side_filter"] is None
    identity = policy.identity
    map_id = identity.pop("policy_map_id")
    assert map_id == canonical_hash(identity)
    identity["inputs"]["body_ids"][0] = -100
    assert policy.inputs.body_ids[0] == policy.identity["inputs"]["body_ids"][0] == 0
    for array in (policy.inputs.body_ids, policy.inputs.indices, policy.inputs.root_side, policy.outputs.soma_side):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_production_entry_rejects_numerical_fixture_before_any_file_read(fixture_metadata, tmp_path):
    with pytest.raises(ValueError, match="full_reference"):
        load_policy_neuron_map(fixture_metadata[0], tmp_path / "does-not-exist.npz")


@pytest.mark.parametrize("absent", ["input", "output"])
def test_empty_selection_fails_explicitly(fixture_metadata, absent):
    graph, annotations, missing = fixture_metadata
    if absent == "input":
        annotations["type"][:] = "other"
        graph = numerical_test_graph(graph.counts, annotations["type"], missing["type"])
    else:
        annotations["superclass"][:] = "optic"
    with pytest.raises(ValueError, match="nonempty"):
        numerical_test_policy_map(graph, annotations, missing)


def _metadata_asset(tmp_path, fixture_metadata, *, mutation=None):
    graph, annotations, missing = fixture_metadata
    provenance = {
        "fixture_role": "numerical_test_only", "graph_sha256": "interface-fixture-no-production-graph",
        "node_metadata_sha256": annotation_hash(SimpleNamespace(annotations=annotations, annotation_missing=missing)),
    }
    source_id = canonical_hash(provenance)
    meta = {"format_version": 2, "asset_type": "numerical_test_only", "source_id": source_id, "provenance": provenance}
    arrays = {"meta": np.asarray(json.dumps(meta)), "body_ids": graph.body_ids.copy(),
              "annotation_fields": np.asarray(sorted(annotations))}
    arrays.update({f"annotation_{key}": value.copy() for key, value in annotations.items()})
    arrays.update({f"missing_{key}": value.copy() for key, value in missing.items()})
    if mutation == "annotation":
        arrays["annotation_rootSide"][0] = "R"
    elif mutation == "body_ids":
        arrays["body_ids"][[0, 1]] = arrays["body_ids"][[1, 0]]
    elif mutation == "source":
        arrays["meta"] = np.asarray(json.dumps({**meta, "source_id": "another-source"}))
    path = tmp_path / "numerical-metadata-only.npz"
    # There are intentionally no weight arrays in this archive: any access to
    # data/indices/indptr would fail instead of consuming a full graph.
    np.savez(path, **arrays)
    return path, provenance, source_id


def test_metadata_binding_reads_only_small_arrays_with_existing_hash_contract(fixture_metadata, tmp_path):
    path, provenance, source_id = _metadata_asset(tmp_path, fixture_metadata)
    observed, missing = _read_bound_metadata(path, fixture_metadata[0].body_ids, provenance, source_id,
                                             asset_type="numerical_test_only")
    for key, values in fixture_metadata[1].items():
        np.testing.assert_array_equal(observed[key], values)
        np.testing.assert_array_equal(missing[key], fixture_metadata[2][key])


@pytest.mark.parametrize("mutation,error", [("annotation", "annotation hash"), ("body_ids", "body ID order"),
                                           ("source", "source identity")])
def test_changed_metadata_source_annotations_or_id_order_rejected(fixture_metadata, tmp_path, mutation, error):
    path, provenance, source_id = _metadata_asset(tmp_path, fixture_metadata, mutation=mutation)
    with pytest.raises(ValueError, match=error):
        _read_bound_metadata(path, fixture_metadata[0].body_ids, provenance, source_id, asset_type="numerical_test_only")


def test_map_is_balanced_deterministic_and_missing_side_independent(policy):
    mapping = build_spatial_input_map(policy)
    repeated = build_spatial_input_map(policy)
    assert mapping.map_id == repeated.map_id
    np.testing.assert_array_equal(mapping.pixel_cells, repeated.pixel_cells)
    coverage = mapping.identity["coverage"]
    assert [(group["root_side"], group["neurons"]) for group in coverage] == [(None, 3), ("", 3), ("L", 401), ("R", 193)]
    assert [group["uncovered_count"] for group in coverage] == [189, 189, 0, 0]
    for group in coverage:
        counts = np.asarray(group["cell_counts"])
        assert counts.max() - counts.min() <= 1
    # An independent execution of the registered member permutation contract.
    rng = np.random.Generator(np.random.PCG64(1701))
    expected = np.empty(600, dtype=np.int64)
    for members in (np.arange(597, 600), np.arange(594, 597), np.arange(401), np.arange(401, 594)):
        expected[members[rng.permutation(len(members))]] = np.arange(len(members)) % 192
    np.testing.assert_array_equal(mapping.pixel_cells, expected)
    assert mapping.identity["numpy_version"] == np.__version__
    for array in (mapping.pixel_cells, mapping.input_indices, mapping.input_body_ids):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("cell", [0, 15, 176, 191])
def test_four_image_corners_drive_distinct_registered_inputs_only(policy, cell):
    mapping = build_spatial_input_map(policy)
    drive = SpatialDrive(mapping)
    mask = np.zeros((12, 16))
    mask.flat[cell] = 1
    touched = set()
    for _ in range(100):
        event = drive.step(mask)
        touched.update(event.input_indices.tolist())
        dense = event.to_dense()
        assert np.all(dense[600:] == 0)
        assert set(np.unique(dense)).issubset({0.0, 68.75})
    expected = set(mapping.input_indices[mapping.pixel_cells == cell].tolist())
    assert touched == expected and touched


def test_event_count_phase_continuity_reset_and_restore(policy):
    mapping = build_spatial_input_map(policy)
    drive = SpatialDrive(mapping)
    original = drive.export_state()
    independent_phase = np.random.Generator(np.random.PCG64(1702)).random(600)
    np.testing.assert_array_equal(original.phase, independent_phase)
    mask_a, mask_b = np.full((12, 16), 0.4), np.full((12, 16), 0.8)
    for _ in range(71):
        drive.step(mask_a)
    boundary = drive.export_state()
    resumed = SpatialDrive(mapping)
    resumed.restore_state(boundary)
    for _ in range(129):
        actual, replay = drive.step(mask_b), resumed.step(mask_b)
        np.testing.assert_array_equal(actual.input_indices, replay.input_indices)
    final = drive.export_state()
    integrated = independent_phase + (71 * 0.4 + 129 * 0.8) * 0.015
    np.testing.assert_array_equal(final.event_counts, np.floor(integrated).astype(np.int64))
    np.testing.assert_allclose(final.phase, integrated % 1, atol=1e-13, rtol=0)
    assert final.step == 200
    before_blank = final.phase.copy()
    assert len(drive.step(np.zeros((12, 16))).input_indices) == 0
    np.testing.assert_array_equal(drive.export_state().phase, before_blank)
    reset = drive.reset()
    assert reset.step == 0 and not reset.event_counts.any()
    np.testing.assert_array_equal(reset.phase, original.phase)
    for array in (original.phase, original.event_counts, actual.increments_mv):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("mask", [np.zeros((16, 12)), np.full((12, 16), np.nan), np.full((12, 16), np.inf),
                                  np.full((12, 16), -0.01), np.full((12, 16), 1.01),
                                  np.ones((12, 16), dtype=complex), [[0] * 16] * 12])
def test_invalid_masks_do_not_advance_state(policy, mask):
    drive = SpatialDrive(build_spatial_input_map(policy))
    drive.step(np.ones((12, 16)))
    before = drive.export_state()
    with pytest.raises(ValueError, match="occupancy"):
        drive.step(mask)
    after = drive.export_state()
    assert after.step == before.step
    np.testing.assert_array_equal(after.phase, before.phase)
    np.testing.assert_array_equal(after.event_counts, before.event_counts)


@pytest.mark.parametrize("field,value", [("drive_id", "another-config"), ("spatial_map_id", "another-map"),
                                         ("dt_ms", 0.2), ("step", -1), ("phase", np.full(600, np.nan)),
                                         ("phase", np.zeros(599)), ("event_counts", np.full(600, 2, dtype=np.int64))])
def test_bad_restore_is_atomic(policy, field, value):
    drive = SpatialDrive(build_spatial_input_map(policy))
    before = drive.export_state()
    with pytest.raises(ValueError):
        drive.restore_state(replace(before, **{field: value}))
    after = drive.export_state()
    assert after.step == before.step
    np.testing.assert_array_equal(after.phase, before.phase)
    np.testing.assert_array_equal(after.event_counts, before.event_counts)


def test_invalid_rate_cannot_drop_events_and_dense_tensor_is_independent(policy):
    with pytest.raises(ValueError, match="one event"):
        SpatialDriveConfig(max_rate_hz=10001)
    with pytest.raises(ValueError, match="dt_ms"):
        SpatialDriveConfig(dt_ms=float("nan"))
    mapping = build_spatial_input_map(policy)
    drive = SpatialDrive(mapping)
    for _ in range(67):
        event = drive.step(np.ones((12, 16)))
    dense = event.to_dense(np.float64)
    dense[:] = 900
    assert np.all(event.to_dense()[600:] == 0)
    torch = pytest.importorskip("torch")
    tensor = event.to_torch(dtype=torch.float64)
    assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
    np.testing.assert_array_equal(tensor.numpy(), event.to_dense(np.float64))
