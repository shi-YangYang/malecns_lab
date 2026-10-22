"""Protocol, pixels and file/parent guards only; no graph or model is constructed."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from malecns_lab.learning import x2_inference as inference
from malecns_lab.senses.x2_marker import marker_frame, probe_inputs


def test_default_entry_and_import_are_inert(tmp_path):
    output = tmp_path / "must-not-exist"
    completed = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--output", str(output)],
                               capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["brain_loaded"] is False
    assert not output.exists()
    probe = subprocess.run([sys.executable, "-c", "import sys; import malecns_lab.experiments.x2_inference; "
                            "import malecns_lab.learning.x2_inference; "
                            "assert not {'torch','numpy','scipy','rclpy','cnsbridge'}.intersection(sys.modules)"], check=False)
    assert probe.returncode == 0
    help_result = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--help"],
                                 capture_output=True, text=True)
    assert help_result.returncode == 0 and "--execute" in help_result.stdout


def test_execute_requires_new_explicit_directory(tmp_path):
    missing = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--execute"],
                             capture_output=True, text=True)
    assert missing.returncode != 0
    (tmp_path / "data/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        inference.execute(tmp_path / "data/existing", tmp_path)
    with pytest.raises(ValueError, match="beneath"):
        inference.execute(tmp_path / "outside", tmp_path)


def test_preregistered_projection_and_pose_are_causal():
    probes = probe_inputs()
    assert [sample["distance_m"] for sample in probes] == [2., 1.5, 1.]
    frames = [marker_frame(sample["camera_position_m"], sample["camera_orientation_xyzw"], sample["target_world_m"])
              for sample in probes]
    counts = []
    for frame in frames:
        assert frame.dtype == np.uint8 and frame.shape == (120, 160, 3)
        mask = frame[:, :, 2] == 235
        counts.append(int(mask.sum()))
        yy, xx = np.where(mask)
        assert yy.mean() == 59.5 and xx.mean() == 79.5
        assert np.all(frame[~mask] == 32) and np.all(frame[mask] == (18, 18, 235))
    assert 0 < counts[0] < counts[1] < counts[2]
    moved = marker_frame((1, 0, .65), (0, 0, 0, 1), (2, 0, .65))
    assert np.array_equal(moved, frames[2])
    turned = marker_frame((0, 0, .65), (0, 0, 1, 0), (2, 0, .65))
    assert np.all(turned == 32)


@pytest.mark.parametrize("position,quaternion,target", [
    ((0, 0), (0, 0, 0, 1), (2, 0, 0)),
    ((0, 0, 0), (0, 0, 0, 2), (2, 0, 0)),
    ((0, 0, 0), (0, 0, 0, 0), (2, 0, 0)),
    ((0, 0, 0), (0, 0, 0, 1), (np.nan, 0, 0)),
])
def test_projection_rejects_invalid_measurements(position, quaternion, target):
    with pytest.raises(ValueError):
        marker_frame(position, quaternion, target)


def parameter_archive(tmp_path, *, change=None):
    # Small serialization values, not a graph, neural network or research result.
    values = {"parameter.theta": np.ones(2, np.float32), "parameter.readout_weight": np.ones((3, 4), np.float32),
              "parameter.readout_bias": np.zeros(3, np.float32), "brain.old_state": np.zeros(1, np.float32)}
    if change:
        change(values)
    path = tmp_path / "parameters.npz"
    np.savez(path, **values)
    metadata = {"arrays_sha256": inference.sha256(path),
                "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in values.items()}}
    identity = {"model": {"graph": {"grouping": {"names": [None, "metadata-only"]}}}, "features": {"width": 4}}
    return path, metadata, identity


def test_parameter_loader_materializes_only_three_parameters(tmp_path):
    path, metadata, identity = parameter_archive(tmp_path)
    result = inference.load_parameters(path, metadata, identity)
    assert set(result) == {"theta", "readout_weight", "readout_bias"}
    assert result["theta"].dtype == np.float32
    metadata["arrays_sha256"] = "0"*64
    with pytest.raises(ValueError, match="checksum"):
        inference.load_parameters(path, metadata, identity)


@pytest.mark.parametrize("invalid", [np.ones(3, np.float32), np.ones(2, np.float64), np.array([0., np.nan], np.float32)])
def test_parameter_loader_rejects_invalid_inherited_arrays(tmp_path, invalid):
    path, metadata, identity = parameter_archive(tmp_path, change=lambda values: values.update({"parameter.theta": invalid}))
    with pytest.raises(ValueError, match="Invalid inherited"):
        inference.load_parameters(path, metadata, identity)


def test_parameter_inventory_is_bound(tmp_path):
    path, metadata, identity = parameter_archive(tmp_path)
    metadata["arrays"].pop("brain.old_state")
    with pytest.raises(ValueError, match="inventory"):
        inference.load_parameters(path, metadata, identity)


def test_shared_lock_is_exclusive_and_released(tmp_path):
    (tmp_path / "data").mkdir()
    with inference.numerical_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with inference.numerical_lock(tmp_path):
                pytest.fail("Second full-brain lock owner admitted")
    with inference.numerical_lock(tmp_path):
        pass


def test_direct_worker_rejected_before_heavy_imports(tmp_path):
    registration = {"parent_pid": -1, "output": str(tmp_path)}
    registration["registration_id"] = inference.canonical_hash(registration)
    (tmp_path / "registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="supervising parent"):
        inference.run_worker(tmp_path)
    assert not (tmp_path / "worker.claim").exists()


def test_metadata_parser_rejects_duplicates_and_nonfinite(tmp_path):
    path = tmp_path / "input.json"
    for text in ('{"field":1,"field":2}', '{"field":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            inference.read_json(path)


def test_simulation_preflight_is_read_only_and_rejects_running_gpu(tmp_path, monkeypatch):
    commands = []
    def command(argv, **kwargs):
        commands.append(argv)
        if argv[-1] == "status":
            return subprocess.CompletedProcess(argv, 0, "lighter is not running.", "")
        return subprocess.CompletedProcess(argv, 0, "exited\n", "")
    monkeypatch.setattr(inference.subprocess, "run", command)
    result = inference.simulator_preflight(tmp_path)
    assert result["cpu_status"] == "exited"
    assert commands[0][-1] == "status" and commands[1][1:4] == ["--context", "desktop-linux", "inspect"]
    socket = tmp_path / "data/x2_sim/gpu_probe/runtime-rebuild/docker.sock"
    socket.parent.mkdir(parents=True)
    socket.touch()
    with pytest.raises(ValueError, match="GPU simulation"):
        inference.simulator_preflight(tmp_path)


def test_preflight_failure_preserves_new_directory_without_launch(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    def fail(_):
        raise ValueError("simulation_running")
    monkeypatch.setattr(inference, "simulator_preflight", fail)
    result = inference.execute(tmp_path / "data/failed", tmp_path)
    assert result["status"] == "failed" and "simulation_running" in result["reason"]
    assert json.loads((tmp_path / "data/failed/phase.json").read_text())["phase"] == "preflight"
    assert (tmp_path / "data/failed/summary.json").exists()
    assert not (tmp_path / "data/failed/worker.claim").exists()


def test_execute_holds_lock_and_passes_bounded_budget_to_supervisor(tmp_path, monkeypatch):
    from malecns_lab.learning import resource_guard
    import psutil
    from types import SimpleNamespace
    (tmp_path / "data").mkdir()
    for name in inference.WRAPPER_FILES:
        path = tmp_path / "src/malecns_lab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# wrapper serialization fixture\n")
    monkeypatch.setattr(inference, "simulator_preflight", lambda root: {"cpu_status": "exited"})
    monkeypatch.setattr(inference, "read_anchors", lambda root: {
        "run": {"implementation_sha256": {}, "runtime": {}, "graph": {"path": "historical/path"}, "policy": {"policy_id": "fixture"}},
        "cpu": {}, "checkpoint": {}, "window": {}})
    monkeypatch.setattr(inference, "export_sources", lambda *args: {"registered_source_count": 54})
    monkeypatch.setattr(inference, "runtime", lambda backend="baseline": {"backend": "test-no-numerical-work"})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=3*inference.GIB))
    calls = []
    def supervised(command, directory, **budget):
        with pytest.raises(BlockingIOError):
            with inference.numerical_lock(tmp_path):
                pytest.fail("Parent lock was released before supervision")
        calls.append((command, budget))
        resource_guard.write_json(directory / "worker_result.json", {"status": "completed", "completed_frames": 3,
                                                                    "saved_state_boundaries": 4})
        return {"status": "completed", "reason": "worker_exited"}
    monkeypatch.setattr(resource_guard, "supervise", supervised)
    result = inference.execute(tmp_path / "data/run", tmp_path)
    assert result["status"] == "completed" and len(calls) == 1
    command, budget = calls[0]
    assert command[-2:] == ["--_worker", str(tmp_path / "data/run")]
    assert budget["rss_limit_bytes"] == 2*inference.GIB and budget["reserve_bytes"] == inference.GIB
    assert budget["poll_seconds"] == .1 and 0 < budget["wall_seconds"] <= 240
    registration = inference.read_json(tmp_path / "data/run/registration.json")
    inference.bound(registration, "registration_id")
    assert registration["resources"]["requested_rss_limit_bytes"] == 6*inference.GIB


def fake_archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
    return stream.getvalue()


def make_writable(directory):
    for path in [directory, *directory.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def test_snapshot_includes_dependency_files_and_rejects_mutation(tmp_path, monkeypatch):
    content = {"src/cnsbridge/__init__.py": b'"""test package only"""\n',
               "src/cnsbridge/learning/core.py": b"value = 1\n", "src/cnsbridge/defaults.yaml": b"fixture: true\n"}
    archive = fake_archive(content)
    def git(argv, **kwargs):
        return "a"*40 if argv[1] == "rev-parse" else archive
    monkeypatch.setattr(inference.subprocess, "check_output", git)
    expected = {"learning/core.py": inference.hashlib.sha256(content["src/cnsbridge/learning/core.py"]).hexdigest()}
    directory = tmp_path / "sources"
    try:
        manifest = inference.export_sources(directory, expected, tmp_path)
        assert len(manifest["files_sha256"]) == 3
        assert manifest["registered_source_count"] == 1
        assert (directory / "src/cnsbridge/defaults.yaml").stat().st_mode & 0o222 == 0
        inference.verify_sources(directory, manifest)
        path = directory / "src/cnsbridge/learning/core.py"
        path.chmod(0o644)
        path.write_text("changed")
        with pytest.raises(ValueError, match="Snapshot changed"):
            inference.verify_sources(directory, manifest)
    finally:
        make_writable(directory)


def test_snapshot_refuses_path_traversal(tmp_path, monkeypatch):
    archive = fake_archive({"src/cnsbridge/../../escape.py": b"pass"})
    monkeypatch.setattr(inference.subprocess, "check_output", lambda argv, **kw: "a"*40 if argv[1] == "rev-parse" else archive)
    with pytest.raises(ValueError, match="Invalid snapshot path"):
        inference.export_sources(tmp_path / "sources", {}, tmp_path)
    assert not (tmp_path / "escape.py").exists()


@pytest.mark.skipif(not (inference.ROOT / inference.CHECKPOINT / "checkpoint.json").exists(), reason="Local research metadata absent")
def test_actual_fixed_metadata_and_all_registered_sources_without_brain(tmp_path):
    anchors = inference.read_anchors()
    assert anchors["checkpoint"]["checkpoint_id"] == inference.CHECKPOINT_ID
    directory = tmp_path / "historical"
    try:
        manifest = inference.export_sources(directory, anchors["run"]["implementation_sha256"])
        assert manifest["registered_source_count"] == 54
        assert len(manifest["files_sha256"]) > 54
        inference.verify_sources(directory, manifest)
    finally:
        if directory.exists():
            make_writable(directory)


def test_optimized_cli_is_explicit_and_preview_stays_inert(tmp_path):
    output = tmp_path / "no-output"
    preview = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--backend", "optimized",
                              "--baseline", str(tmp_path / "absent"), "--output", str(output)],
                             capture_output=True, text=True)
    assert preview.returncode == 0 and json.loads(preview.stdout)["backend"] == "optimized"
    assert not output.exists()
    missing = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--execute", "--backend",
                              "optimized", "--output", str(output)], capture_output=True, text=True)
    assert missing.returncode != 0 and not output.exists()
    with pytest.raises(ValueError, match="requires a baseline"):
        inference.execute(output, tmp_path, backend="optimized")
    with pytest.raises(ValueError, match="must not modify"):
        inference.execute(tmp_path / "data/base/nested", tmp_path, backend="optimized", baseline=tmp_path / "data/base")


@pytest.mark.parametrize("actual,expected,equal,different", [
    (np.array([1., 2.], np.float32), np.array([1., 2.], np.float32), True, 0),
    (np.array([1., 3.], np.float32), np.array([1., 2.], np.float32), False, 1),
    (np.array([1., 2.], np.float64), np.array([1., 2.], np.float32), False, 0),
    (np.array([1, 2], np.int64), np.array([1, 3], np.int64), False, 1),
    (np.array([1, 2], np.int64), np.array([[1, 2]], np.int64), False, None),
    (np.array([np.nan], np.float32), np.array([np.nan], np.float32), False, 1),
])
def test_exact_array_comparison_never_uses_error_tolerance(actual, expected, equal, different):
    record = inference.compare_array(actual, expected)
    assert record["exact_equal"] is equal and record["different_elements"] == different
    assert record["actual_dtype"] == actual.dtype.str
    if different == 1 and record["finite"]:
        assert record["max_absolute_error"] == 1
        assert record["rms_error"] == pytest.approx(np.sqrt(.5))


def comparison_files(directory):
    # Only archive/comparator fixtures; no network or neural update is created.
    (directory / "states").mkdir(parents=True)
    (directory / "frames").mkdir()
    for index in range(4):
        np.savez(directory / f"states/{index:03d}.npz", **{
            "brain.v": np.array([index, 2], np.float32), "brain.g": np.array([0, 1], np.float32),
            "brain.refractory_until": np.array([index, 0], np.int64), "brain.pending": np.zeros((2, 2), np.float32),
            "drive.phase": np.array([.5], np.float32), "drive.event_counts": np.array([index], np.int64)})
        (directory / f"states/{index:03d}.json").write_text(json.dumps({"metadata": {"brain": {"step": 600+500*index},
                                                                                           "frame_index": index}}))
        if index:
            np.save(directory / f"frames/{index-1:03d}.npy", np.full((2, 2, 3), index, np.uint8), allow_pickle=False)
            np.savez(directory / f"frames/{index-1:03d}-output.npz", action=np.ones(3, np.float32),
                     features_mv=np.ones(2, np.float32), logits=np.zeros(3, np.float32))


def test_compare_all_four_boundaries_and_three_outputs_and_performance(tmp_path):
    actual, prior = tmp_path / "new", tmp_path / "baseline"
    comparison_files(actual)
    comparison_files(prior)
    binding = {"path": str(prior), "registration_id": "fixture", "files_sha256": {
        str(path.relative_to(prior)): inference.sha256(path) for path in prior.rglob("*") if path.is_file()},
        "compute_seconds": 10., "phase_compute_seconds": [4., 2., 2., 2.]}
    timing = {"precondition_seconds": 2., "frame_forward_seconds": [1., 1., 1.]}
    result = inference.compare_evidence(actual, binding, timing)
    assert result["status"] == "PASS" and len(result["comparisons"]) == 7
    assert result["first_difference"] is None and result["compute_reduction_fraction"] == .5
    boundary = inference.compare_evidence(actual, binding, {"precondition_seconds": 2., "frame_forward_seconds": [2., 2., 2.]})
    assert boundary["status"] == "PASS" and boundary["performance_pass"] is True
    slower = inference.compare_evidence(actual, binding, {"precondition_seconds": 4., "frame_forward_seconds": [2., 2., 2.]})
    assert slower["status"] == "FAIL" and slower["exact_equal"] is True and slower["performance_pass"] is False
    state = actual / "states/002.npz"
    with np.load(state, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["brain.pending"][1, 1] = np.nextafter(np.float32(0), np.float32(1))
    np.savez(state, **arrays)
    failed = inference.compare_evidence(actual, binding, timing)
    assert failed["status"] == "FAIL" and failed["first_difference"]["frame_index"] == 2
    assert failed["first_difference"]["arrays"] == ["brain.pending"]
    assert failed["comparisons"][3]["arrays"]["brain.pending"]["different_elements"] == 1
    prior.joinpath("states/000.json").write_text("{}")
    with pytest.raises(ValueError, match="changed during execution"):
        inference.compare_evidence(actual, binding, timing)


@pytest.mark.skipif(not (inference.ROOT / "data/x2_neural_inference_20261023_r1/summary.json").exists(), reason="Saved baseline absent")
def test_actual_completed_baseline_binding_allows_new_wrapper_sources():
    anchors = inference.read_anchors()
    binding = inference.baseline_evidence(inference.ROOT / "data/x2_neural_inference_20261023_r1", anchors)
    assert binding["checkpoint_id"] == inference.CHECKPOINT_ID
    assert binding["compute_seconds"] == pytest.approx(25.774687707889825)
    assert all(f"states/{index:03d}.npz" in binding["files_sha256"] for index in range(4))
    assert all(f"frames/{index:03d}.npy" in binding["files_sha256"] for index in range(3))
