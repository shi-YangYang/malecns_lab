"""Small contract fixtures only: no synthetic brain or empirical research data."""

import copy
import json

import numpy as np
import pytest

from malecns_lab.learning.contracts_v2 import OUTPUT_SPEC, FeatureSchemaV2, digest
from malecns_lab.learning.dataset_v2 import assert_paired_rows, load_dataset
from malecns_lab.learning.decoder_v2 import MaskedRidgeV2
from malecns_lab.learning.evaluation_v2 import paired_scene_comparison, score_predictions
from malecns_lab.learning.train_v2 import fit_readouts, main


def write(path, data):
    path.write_text(json.dumps(data))


def contract_fixture(tmp_path, *, extra_failed_scene=False):
    task = {"policy_id": "contract-test", "teacher_id": "test-author", "teacher_kind": "predefined_task",
            "description": "Array-only implementation fixture, not action data",
            "tolerances": {"vx": 0.2, "wz": 0.3}, "stationary_tolerances": {"vx": 0.01, "wz": 0.01}}
    protocol = {"format": "cnsbridge.learning_protocol.v2", "protocol_id": "fixture", "phase": "confirmatory", "task": task,
                "practical_delta": 0.1, "budget_seconds": 30, "bootstrap": {"seed": 1, "replicates": 20}, "segments": []}
    provenance = {"features": {"neural_names": [f"body-{i}" for i in range(132)], "visual_names": [f"feature-{i}" for i in range(98)]},
                  "time": {"simulation_window_ms": 50, "target_horizon_s": 0.05}}
    splits = ["train", "val", "test"] + (["test"] if extra_failed_scene else [])
    for index, split in enumerate(splits):
        root = tmp_path / f"replay-{index}"
        segment_path = root / "segment"
        segment_path.mkdir(parents=True)
        source = {"kind": "recorded_video", "video_sha256": f"fixture-content-{index}", "scene_id": f"scene-{index}",
                  "session_id": f"session-{index}", "family_id": f"family-{index}"}
        source["source_id"] = digest(source)
        segment = {"start_frame": 0, "end_frame": 46, "segment_id": "segment", "source_index": 0, "repetition": 0}
        failed, count = extra_failed_scene and index == 3, 46
        if failed:
            count = 1
        frames = np.arange(count)
        valid = np.full(count, not failed)
        sim = (frames + 1) * 50 if not failed else np.zeros(count)
        prediction = valid & (sim > 2000) & (frames + 1 < 46)
        neural = np.repeat((frames / 50)[:, None], 132, axis=1)
        visual = np.repeat((frames / 50)[:, None], 98, axis=1)
        if failed:
            neural[:], visual[:] = np.nan, np.nan
        arrays = {"frame_index": frames, "input_cutoff_s": frames / 20, "target_time_s": (frames + 1) / 20,
                  "valid": valid, "prediction_valid": prediction, "target_available": frames + 1 < 46,
                  "simulation_end_ms": sim, "neural_history_hz": neural, "visual_history": visual}
        np.savez(segment_path / "rows.npz", **arrays)
        coverage = {"planned_frames": 46, "attempted_frames": count, "unprocessed_tail_frames": 46 - count,
                    "prediction_valid_frames": int(prediction.sum()), "valid_input_frames": int(valid.sum()),
                    "failed_frames": int((~valid).sum()), "warmup_frames": int(np.sum(valid & (sim <= 2000))),
                    "failure_reasons": {"fixture_failure": 1} if failed else {}}
        meta = {"format": "cnsbridge.neural_activity.v2", "source_id": digest(provenance), "input_source": source,
                "segment": segment, "run_seed": 0, "coverage": coverage}
        run = {"format": "cnsbridge.neural_replay_run.v2", "model_source_id": digest(provenance), "model_provenance": provenance,
               "planned_segments": [segment], "input_sources": [source], "run_seed": 0}
        write(segment_path / "metadata.json", meta)
        write(root / "run_manifest.json", run)
        labels = {"format": "cnsbridge.motion_labels.v2", "source_id": source["source_id"], "output": OUTPUT_SPEC,
                  "timebase": "20Hz_video_playback", **{k: task[k] for k in ("policy_id", "teacher_id", "teacher_kind")},
                  "intervals": [{"start_s": 0, "end_s": 2.1, "target": [1, None, None, None, None, -0.3],
                                 "mask": [True, False, False, False, False, True], "reason": "fixture A"},
                                {"start_s": 2.1, "end_s": 10, "target": [-1, None, None, None, None, 0.3],
                                 "mask": [True, False, False, False, False, True], "reason": "fixture B"}]}
        write(root / "labels.json", labels)
        protocol["segments"].append({"path": f"replay-{index}/segment", "labels": f"replay-{index}/labels.json", "split": split})
    path = tmp_path / "protocol.json"
    write(path, protocol)
    return path


def test_masked_fit_ignores_unknown_axes_and_missing_labels(tmp_path):
    schema = FeatureSchemaV2("contract-fixture", tuple(f"x{i}" for i in range(132)), "neural")
    x = np.zeros((6, 132))
    x[:, 0] = np.arange(6)
    y = np.full((6, 6), np.nan)
    y[:, 0] = 2 * x[:, 0] + 1
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, 0] = True
    model = MaskedRidgeV2.fit(schema, x, y, mask, ridge=0.0001)
    assert model.scale[1] == 1
    assert model.supported.tolist() == [True, False, False, False, False, False]
    assert not model.weights[:, 1:].any()
    assert np.allclose(model.predict_batch(x, schema=schema)[:, 0], y[:, 0], atol=0.001)
    inactive = model.decode(np.full(132, np.nan), schema=schema, prediction_valid=False)
    assert not inactive.prediction_valid and inactive.supported_axes == ("vx",)
    path = tmp_path / "model.json"
    model.save(path)
    assert np.array_equal(MaskedRidgeV2.load(path).weights, model.weights)
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="schema"):
        model.predict_batch(x, schema=FeatureSchemaV2("other-source", schema.feature_names, "neural"))


def test_causal_alignment_warmup_and_content_split_guard(tmp_path):
    path = contract_fixture(tmp_path)
    data = load_dataset(path)
    assert data.targets[40, 0] == 1  # input 2.00 -> target 2.05
    assert data.targets[41, 0] == -1  # input 2.05 -> target 2.10 (half-open next interval)
    assert not data.eligible[:40].any() and data.eligible[40]
    assert not data.eligible[45]
    assert not data.mask[:, [1, 2, 3, 4]].any()
    # Copy/rename with changed IDs still cannot leak identical content across splits.
    root = tmp_path / "replay-1"
    meta = json.loads((root / "segment/metadata.json").read_text())
    run = json.loads((root / "run_manifest.json").read_text())
    source = meta["input_source"]
    source["video_sha256"] = "fixture-content-0"
    source["source_id"] = digest({k: v for k, v in source.items() if k != "source_id"})
    run["input_sources"] = [source]
    labels = json.loads((root / "labels.json").read_text())
    labels["source_id"] = source["source_id"]
    write(root / "segment/metadata.json", meta)
    write(root / "run_manifest.json", run)
    write(root / "labels.json", labels)
    with pytest.raises(ValueError, match="Split leakage.*video_sha256"):
        load_dataset(path)


def test_validation_selects_and_test_never_changes_fit(tmp_path):
    data = load_dataset(contract_fixture(tmp_path))
    result = fit_readouts(data)
    altered = copy.deepcopy(data)
    use = altered.split_rows("test")
    altered.features["neural"][use] *= 100
    altered.features["visual"][use] /= 2
    altered.targets[use, 0] += 500
    changed = fit_readouts(altered)
    for kind in ("neural", "visual"):
        assert len(result["selection"][kind]["candidates"]) == 9
        scores = [item["validation"]["normalized_mae"] for item in result["selection"][kind]["candidates"]]
        assert result["selection"][kind]["selected_ridge"] == result["selection"][kind]["candidates"][int(np.argmin(scores))]["ridge"]
        for name in ("weights", "bias", "mean", "scale"):
            assert np.array_equal(getattr(result["models"][kind], name), getattr(changed["models"][kind], name))
    assert result["training_id"] == changed["training_id"]
    assert result["metrics"]["test"]["mean"]["scenes"][0]["coverage"] == result["metrics"]["test"]["neural"]["scenes"][0]["coverage"]
    pilot = copy.deepcopy(data)
    pilot.protocol["phase"] = "pilot"
    pilot_result = fit_readouts(pilot)
    assert "test" not in pilot_result["metrics"]
    assert np.isnan(pilot_result["predictions"]["neural"][use]).all()


def test_zero_coverage_scene_is_retained_and_unprocessed_tail_is_separate(tmp_path):
    data = load_dataset(contract_fixture(tmp_path, extra_failed_scene=True))
    prediction = np.zeros_like(data.targets)
    report = score_predictions(data, prediction, np.array([True, False, False, False, False, True]), split="test")
    assert report["registered_scene_count"] == 2 and report["scored_scene_count"] == 1
    zero = next(scene for scene in report["scenes"] if scene["scene_id"] == "scene-3")
    assert zero["normalized_mae"] is None
    assert zero["coverage"]["failed_frames"] == 1 and zero["coverage"]["unprocessed_tail_frames"] == 45
    comparison = paired_scene_comparison(report, report, practical_delta=0.1, seed=0, replicates=20)
    assert comparison["interpretation"] == "insufficient_evidence" and comparison["ci95"] is None
    altered = copy.deepcopy(data)
    altered.mask[40, 0] = False
    with pytest.raises(ValueError, match="supervision coverage"):
        assert_paired_rows(data, altered, "train")


def test_cluster_bootstrap_uses_shared_family_unit():
    def report(offset):
        return {"split": "test", "scoring_manifest_sha256": "same-fixture-scoring-rows", "coverage_status": "all_scenes_axes_seeds_scored", "scenes": [
            {"scene_id": name, "cluster_id": cluster, "normalized_mae": score + offset, "coverage": {"scored_frames": 2}}
            for name, cluster, score in (("a", "family-1", 1), ("b", "family-1", 2), ("c", "family-2", 3))]}
    result = paired_scene_comparison(report(0), report(0.5), practical_delta=0.1, seed=1, replicates=20)
    assert result["independent_clusters"] == 2
    assert result["ci95"] == [0.5, 0.5]
    assert result["interpretation"] == "practical_advantage_on_scored_data"


def test_paired_comparison_distinguishes_ci_threshold_and_equal_boundaries():
    def report(scores):
        return {"split": "test", "scoring_manifest_sha256": "same-fixture-scoring-rows",
                "coverage_status": "all_scenes_axes_seeds_scored", "scenes": [
                    {"scene_id": str(i), "cluster_id": str(i), "normalized_mae": score,
                     "coverage": {"scored_frames": 2}} for i, score in enumerate(scores)]}

    real = report([1.0, 1.0])
    cases = (
        ((0.5, 0.5), "practical_advantage_on_scored_data"),
        ((0.0, 0.0), "no_confirmed_practical_advantage"),
        ((0.0, 0.5), "insufficient_evidence"),
        ((0.25, 0.5), "insufficient_evidence"),
        ((0.0, 0.25), "insufficient_evidence"),
        ((0.25, 0.25), "insufficient_evidence"),
    )
    for differences, expected in cases:
        control = report([1.0 + delta for delta in differences])
        result = paired_scene_comparison(real, control, practical_delta=0.25, seed=0, replicates=100)
        assert result["ci95"] == list(differences)
        assert result["interpretation"] == expected


def test_cli_persistence_and_refusal_of_generated_diagnostic_training(tmp_path):
    path = contract_fixture(tmp_path)
    out = tmp_path / "learning-out"
    main(["--protocol", str(path), "--out", str(out)])
    assert (out / "input_seed_0/model_neural.json").exists()
    assert json.loads((out / "run_summary.json").read_text())["test_evaluated"]
    root = tmp_path / "replay-0"
    meta = json.loads((root / "segment/metadata.json").read_text())
    run = json.loads((root / "run_manifest.json").read_text())
    meta["input_source"]["kind"] = "generated_engineering_stimulus"
    run["input_sources"] = [meta["input_source"]]
    write(root / "segment/metadata.json", meta)
    write(root / "run_manifest.json", run)
    with pytest.raises(ValueError, match="Generated diagnostic stimuli"):
        load_dataset(path)
