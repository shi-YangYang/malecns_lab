"""Short registered-scene fixtures, not production collection evidence."""
from dataclasses import asdict, replace
from functools import partial
import json
import sys

import numpy as np
import pytest

from malecns_lab.brain.metadata import canonical_hash, file_hashes
from malecns_lab.control.body import BodyVelocity
from malecns_lab.drones.follow_env import limit_follow_action
from malecns_lab.drones.follow_scenes import scene_registry
from malecns_lab.learning.follow_task import rollout, visual_teacher
from malecns_lab.learning import neural_data as data
from malecns_lab.senses.target import TargetCamera


@pytest.fixture
def short_registry(monkeypatch):
    scenes = tuple(replace(s, duration_s=0.3, desired_distance_m=1.8) for s in scene_registry() if s.split == "train")
    # Retain an entirely invalid sequence. Its labels remain nonzero, while the
    # behavior's external intervention yields zero executed actions.
    scenes = tuple(replace(s, initial_offset_m=(0.45, 5.0, 0.15)) if s.scene_id.endswith("yaw") else s for s in scenes)
    monkeypatch.setattr(data, "scene_registry", lambda: scenes)
    monkeypatch.setattr(data, "DATA_ROLE", "numerical_test_only")
    return sorted(scenes, key=lambda s: s.scene_id)


def seal(path, manifest):
    manifest["dataset_id"] = canonical_hash({k: v for k, v in manifest.items() if k != "dataset_id"})
    (path / "manifest.json").write_text(json.dumps(manifest, allow_nan=False))


def collect(path):
    return data.collect_neural_data(path, camera=TargetCamera(), dt_s=0.1, response_tau_s=0.3)


def test_complete_sequences_match_independent_rollout_and_preserve_invalid_frames(short_registry, tmp_path):
    path = tmp_path / "data"
    manifest = collect(path)
    loaded = data.read_neural_data(path)
    assert manifest["status"] == "ready" and loaded.dataset_id == manifest["dataset_id"]
    assert loaded.identity["role"] == "numerical_test_only"
    assert tuple(s.scene["scene_id"] for s in loaded.sequences) == data.TRAIN_IDS
    for scene, sequence in zip(short_registry, loaded.sequences):
        captured = []
        original = rollout(scene, partial(visual_teacher, desired_distance_m=scene.desired_distance_m),
                           policy_name="visual_teacher_behavior", dt_s=0.1, response_tau_s=0.3,
                           record_frames=lambda item: captured.append(item))
        np.testing.assert_array_equal(sequence.frames, np.stack([item.frame for item in captured]))
        np.testing.assert_array_equal(sequence.time_s, [item.time_s for item in captured])
        np.testing.assert_array_equal(sequence.valid, [item.observation.valid for item in captured])
        np.testing.assert_array_equal(sequence.teacher_raw,
                                      [[r["teacher_action"][axis] for axis in data.AXES] for r in original.records])
        expected = [limit_follow_action(BodyVelocity(**r["teacher_action"])) for r in original.records]
        np.testing.assert_array_equal(sequence.teacher_target, [[getattr(row, a) for a in data.AXES] for row in expected])
        np.testing.assert_array_equal(sequence.executed_action,
                                      [[r["executed_action"][axis] for axis in data.AXES] for r in original.records])
        assert len(sequence.frames) == 3 and sequence.summary["metrics"]["failure"] is None
        with pytest.raises(ValueError):
            sequence.frames.setflags(write=True)
        assert set(vars(captured[0])) == {"frame", "observation", "camera", "time_s"}
    invalid = loaded.sequences[-1]
    assert invalid.scene["scene_id"] == "train-sine-yaw" and not invalid.valid.any()
    assert not invalid.executed_action.any() and invalid.teacher_target.any()
    assert np.any(np.abs(invalid.teacher_raw) > data.LIMITS)
    assert np.all(np.abs(invalid.teacher_target) <= data.LIMITS)
    # The dataset is a frame source only: it never imports a neural policy.
    assert "malecns_lab.learning.neural_policy" not in data.__dict__


def test_reject_overwrite_and_unsealed_output(short_registry, tmp_path):
    path = tmp_path / "data"
    manifest = collect(path)
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        collect(path)
    assert (path / "manifest.json").read_bytes() == before
    (path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        data.read_neural_data(path)


@pytest.mark.parametrize("mutation", ["missing_scene", "duplicate_scene", "val", "definition", "source", "camera", "status", "canonical"])
def test_dataset_identity_and_split_reject_even_rehashed_metadata(short_registry, tmp_path, mutation):
    path = tmp_path / "data"
    manifest = collect(path)
    if mutation == "missing_scene":
        manifest["sequences"].pop()
    elif mutation == "duplicate_scene":
        manifest["sequences"][1] = manifest["sequences"][0]
    elif mutation == "val":
        manifest["identity"]["scenes"][0]["split"] = "val"
    elif mutation == "definition":
        manifest["identity"]["scenes"][0]["seed"] += 1
    elif mutation == "source":
        manifest["identity"]["implementation_sha256"]["drones/follow_scenes.py"] = "0" * 64
    elif mutation == "camera":
        manifest["identity"]["camera"]["width"] = 176
    elif mutation == "status":
        manifest["status"] = "collection_failed"
    else:
        manifest["identity"]["dt_s"] = 0.2
    if mutation == "canonical":
        (path / "manifest.json").write_text(json.dumps(manifest))
    else:
        seal(path, manifest)
    with pytest.raises(ValueError):
        data.read_neural_data(path)


@pytest.mark.parametrize("mutation", ["field", "dtype", "shape", "nonfinite", "time", "valid", "target", "executed", "raw"])
def test_strict_array_semantics_reject_self_rehashed_bytes(short_registry, tmp_path, mutation):
    path = tmp_path / "data"
    manifest = collect(path)
    entry = manifest["sequences"][0]
    file = path / entry["scene_id"] / "sequence.npz"
    with np.load(file, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    if mutation == "field":
        arrays["truth_input"] = arrays["teacher_raw"].copy()
    elif mutation == "dtype":
        arrays["teacher_target"] = arrays["teacher_target"].astype(np.float32)
    elif mutation == "shape":
        arrays["frames"] = arrays["frames"][:-1]
    elif mutation == "nonfinite":
        arrays["teacher_raw"][0, 0] = np.nan
    elif mutation == "time":
        arrays["time_s"][1] = 0.2
    elif mutation == "valid":
        arrays["valid"][0] = not arrays["valid"][0]
    elif mutation == "target":
        arrays["teacher_target"][0, 0] = 1
    elif mutation == "executed":
        arrays["executed_action"][0, 0] = 0.123
    else:
        arrays["teacher_raw"][0, 0] += 1
    np.savez(file, **arrays)
    entry["sha256"]["sequence.npz"] = file_hashes(file)["sha256"]
    entry["fields"] = {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()}
    seal(path, manifest)
    with pytest.raises(ValueError):
        data.read_neural_data(path)


def test_missing_file_and_hash_change_rejected(short_registry, tmp_path):
    path = tmp_path / "data"
    manifest = collect(path)
    file = path / manifest["sequences"][0]["scene_id"] / "trajectory.jsonl"
    file.write_text(file.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        data.read_neural_data(path)
    file.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        data.read_neural_data(path)


def test_real_val_pixels_truth_and_labels_cannot_be_resealed_as_train(short_registry, tmp_path):
    path = tmp_path / "data"
    manifest = collect(path)
    entry = manifest["sequences"][0]
    folder = path / entry["scene_id"]
    original_summary = json.loads((folder / "summary.json").read_text())
    val = replace(next(s for s in scene_registry() if s.scene_id == "val-triangle-combined"),
                  duration_s=0.3, desired_distance_m=1.8)
    captured = []
    result = rollout(val, partial(visual_teacher, desired_distance_m=val.desired_distance_m),
                     policy_name="visual_teacher_behavior", dt_s=0.1, response_tau_s=0.3,
                     record_frames=lambda item: captured.append(item))
    # Swap the actual held-out trajectory while retaining train declarations and
    # valid hashes. Its pixels, truth, teacher labels and behavior all agree.
    arrays = {"frames": np.stack([item.frame for item in captured]),
              "time_s": np.asarray([item.time_s for item in captured], dtype=np.float64),
              "valid": np.asarray([item.observation.valid for item in captured]),
              "teacher_raw": np.asarray([[r["teacher_action"][axis] for axis in data.AXES] for r in result.records]),
              "teacher_target": np.asarray([[getattr(limit_follow_action(BodyVelocity(**r["teacher_action"])), axis)
                                              for axis in data.AXES] for r in result.records]),
              "executed_action": np.asarray([[r["executed_action"][axis] for axis in data.AXES] for r in result.records])}
    np.savez(folder / "sequence.npz", **arrays)
    (folder / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in result.records))
    (folder / "summary.json").write_text(json.dumps({"metadata": original_summary["metadata"], "metrics": result.metrics}))
    entry["fields"] = {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()}
    entry["sha256"] = {name: file_hashes(folder / name)["sha256"] for name in entry["sha256"]}
    seal(path, manifest)
    with pytest.raises(ValueError, match="registered scene replay"):
        data.read_neural_data(path)


@pytest.mark.parametrize("field", ["truth", "next_truth"])
def test_replayed_truth_and_terminal_state_reject_resealed_records(short_registry, tmp_path, field):
    path = tmp_path / "data"
    manifest = collect(path)
    entry = manifest["sequences"][0]
    file = path / entry["scene_id"] / "trajectory.jsonl"
    records = [json.loads(line) for line in file.read_text().splitlines()]
    # Drone pose is not used by truth_teacher: all original action/label checks
    # still agree, but this claimed state cannot result from the scene/actions.
    records[-1][field]["drone_position_m"][0] += 0.125
    file.write_text("".join(json.dumps(r) + "\n" for r in records))
    entry["sha256"]["trajectory.jsonl"] = file_hashes(file)["sha256"]
    seal(path, manifest)
    with pytest.raises(ValueError, match="registered scene replay"):
        data.read_neural_data(path)


def test_environment_failure_seals_nonready_and_retains_all_scenes(short_registry, tmp_path, monkeypatch):
    original = data.rollout
    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, metrics={**result.metrics, "failure": "out_of_bounds"})
    monkeypatch.setattr(data, "rollout", failed)
    path = tmp_path / "failed"
    manifest = collect(path)
    assert manifest["status"] == "collection_failed" and len(manifest["sequences"]) == 4
    assert all((path / scene / "trajectory.jsonl").is_file() for scene in data.TRAIN_IDS)
    with pytest.raises(ValueError, match="ready"):
        data.read_neural_data(path)


def test_loader_rejects_fixture_after_actual_registry_restored(short_registry, tmp_path, monkeypatch):
    path = tmp_path / "data"
    collect(path)
    monkeypatch.setattr(data, "scene_registry", scene_registry)
    monkeypatch.setattr(data, "DATA_ROLE", "registered_train_scenes")
    with pytest.raises(ValueError, match="identity"):
        data.read_neural_data(path)


def test_cli_collection_does_not_import_neural_model(short_registry, tmp_path, monkeypatch, capsys):
    # Import interception covers graph/model modules even if another test loaded them.
    import builtins
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if "trainable" in name or "neural_policy" in name:
            raise AssertionError("Data collection attempted a model import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    data.main(["--out", str(tmp_path / "cli"), "--response-tau-s", "0.3"])
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
