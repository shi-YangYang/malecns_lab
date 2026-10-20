"""Offline orchestration contracts only; no fixture is biological/action evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning import experiment_protocol_v2 as registry
from malecns_lab.learning import experiment_v2 as experiment
from malecns_lab.learning import research_replay_v2 as replay
from malecns_lab.learning.contracts_v2 import OUTPUT_SPEC, FeatureSchemaV2, RIDGE_CANDIDATES, digest, file_digest
from malecns_lab.learning.dataset_v2 import DatasetV2, load_dataset
from malecns_lab.learning.train_v2 import fit_readouts
from test_learning_v2 import contract_fixture
from test_replay_v2 import contract_engine


def test_research_adapter_installs_control_network_and_preserves_common_protocol(monkeypatch):
    def parent_init(self, path, *, run_seed=0):
        source = contract_engine(run_seed)
        self.__dict__.update(source.__dict__)
        self.graph.groups.update(DNg02_L=np.array([31]), DNg02_R=np.array([32]))
        self.graph.meta = {"source_id": "parent-contract", "graph_sha256": "parent-graph"}
        self.provenance = {"graph_source_id": "parent-contract", "graph_sha256": "parent-graph", "edge_count": 64,
                           "node_count": 64, "features": self.history.identity(), "lif": self.params.to_dict(),
                           "bias_mv": {"DNg02_L": 8.0, "DNg02_R": 8.0}, "source_sha256": {"fixture": "not-research-evidence"}}
        self.source_id = digest(self.provenance)

    called = []

    def verified_fixture(path, parent):
        called.append((path, parent.meta["source_id"]))
        control = copy.deepcopy(parent)
        control.weights.data[:] *= 2
        control.meta = {"source_id": "control-contract", "graph_sha256": "control-graph",
                        "provenance": {"transform": "W", "seed": 201, "parameters": {}}}
        return control

    monkeypatch.setattr(replay.NeuralReplayV2, "__init__", parent_init)
    monkeypatch.setattr(replay, "load_research_control", verified_fixture)
    engine = replay.ResearchNeuralReplayV2("parent", "control", run_seed=3)
    assert called == [("control", "parent-contract")]
    assert np.all(engine.net.W.data == 2)  # Real LIF object uses replaced wiring, not parent.
    assert np.allclose(engine.net._data, 2 * engine.params.w_syn)
    assert engine.net.bias[31] == engine.net.bias[32] == 8
    assert engine.net._t == 0 and not engine.neuron_total_counts.any()
    assert engine.provenance["research_graph"]["common_computation_id"] == replay.computation_identity(engine.provenance)
    assert engine.source_id == digest(engine.provenance)


def _registration_fixture(tmp_path, monkeypatch):
    parent = tmp_path / "parent-contract.npz"
    parent.write_bytes(b"NOT A BRAIN: registration path/hash fixture")
    control = tmp_path / "control-contract.npz"
    control.write_bytes(b"NOT A BRAIN: control registration fixture")
    monkeypatch.setattr(registry, "load_reference", lambda path: SimpleNamespace(meta={"source_id": "G-source", "graph_sha256": "G-graph"}))
    monkeypatch.setattr(registry, "load_research_control", lambda path, parent: SimpleNamespace(meta={
        "source_id": "W-source", "graph_sha256": "W-graph", "provenance": {"transform": "W", "seed": 201, "parameters": {}}}))

    def source(entry):
        meta = {"kind": "recorded_video", "video": entry["video"], "video_sha256": file_digest(Path(entry["video"])),
                "scene_id": entry["scene_id"], "session_id": entry["session_id"], "family_id": entry["family_id"],
                "total_frames": 46, "nominal_fps": 20.0}
        meta["source_id"] = digest(meta)
        return SimpleNamespace(metadata=meta, total_frames=46)

    monkeypatch.setattr(registry, "source_from_entry", source)
    protocol = registry.template()
    protocol.update(protocol_id="contract-only", parent=str(parent), input_seeds=[0, 1], budget_seconds=60,
                    replay_budget_seconds=10, practical_delta=0.1, bootstrap={"seed": 1, "replicates": 20},
                    graphs=[{"id": "G", "role": "G", "path": str(parent)}, {"id": "W201", "role": "W", "path": str(control)}],
                    comparisons=[{"real": "G", "control": "W201"}, {"real": "G", "control": "visual"}])
    protocol["task"].update(policy_id="fixture", teacher_id="fixture", description="Contract fixture, not measured action targets",
                            tolerances={"vx": .2, "wz": .3}, stationary_tolerances={"vx": .01, "wz": .01})
    for split in ("train", "val", "test"):
        video = tmp_path / f"{split}.avi"
        video.write_bytes(f"Video metadata fixture only: {split}".encode())
        entry = {"id": split, "video": str(video), "scene_id": split, "session_id": split, "family_id": split,
                 "split": split, "segments": [{"segment_id": "a", "start_frame": 0, "end_frame": 46}]}
        labels = tmp_path / f"{split}-labels.json"
        labels.write_text(json.dumps({"format": "cnsbridge.motion_labels.v2", "source_id": source(entry).metadata["source_id"],
            "output": OUTPUT_SPEC, "timebase": "20Hz_video_playback", **{key: protocol["task"][key] for key in
            ("policy_id", "teacher_id", "teacher_kind")}, "intervals": []}))
        entry["labels"] = str(labels)
        protocol["sources"].append(entry)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol


def test_registration_freezes_graph_seed_labels_split_budget_and_code(tmp_path, monkeypatch):
    path, protocol = _registration_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "registered.json"
    frozen = registry.register_experiment(path, destination)
    assert frozen == registry.validate_registration(destination)
    assert frozen["protocol"]["graphs"][1]["graph_seed"] == 201
    assert frozen["protocol"]["ridge_candidates"] == list(RIDGE_CANDIDATES)
    assert frozen["protocol"]["input_seeds"] == [0, 1]
    assert {item["split"] for item in frozen["protocol"]["sources"]} == {"train", "val", "test"}
    with pytest.raises(FileExistsError):
        registry.register_experiment(path, destination)
    Path(protocol["sources"][0]["labels"]).write_text("changed after registration")
    with pytest.raises(ValueError, match="Frozen files_sha256 changed"):
        registry.validate_registration(destination)


def test_registration_rejects_cross_split_family_and_changed_search_budget(tmp_path, monkeypatch):
    path, protocol = _registration_fixture(tmp_path, monkeypatch)
    protocol["graphs"][1]["id"] = "visual"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="reserved baseline names"):
        registry.register_experiment(path, tmp_path / "ambiguous-name.json")
    protocol["graphs"][1]["id"] = "W201"
    protocol["ridge_candidates"] = [1.0]
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="nine"):
        registry.register_experiment(path, tmp_path / "bad-budget.json")
    protocol["ridge_candidates"] = list(RIDGE_CANDIDATES)
    protocol["sources"][1]["family_id"] = protocol["sources"][0]["family_id"]
    # Family also contributes to source identity, so bind the fixture label to the edited source.
    labels = Path(protocol["sources"][1]["labels"])
    payload = json.loads(labels.read_text())
    payload["source_id"] = registry.source_from_entry(protocol["sources"][1]).metadata["source_id"]
    labels.write_text(json.dumps(payload))
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Split leakage across family_id"):
        registry.register_experiment(path, tmp_path / "bad-split.json")


def _dataset_with_seeds(tmp_path, graph_id):
    fixture = tmp_path / graph_id
    fixture.mkdir()
    data = load_dataset(contract_fixture(fixture))
    use = ~data.split_rows("test")
    rows, segments = [], []
    for seed in (0, 1):
        for original, keep in zip(data.rows, use):
            if keep:
                row = copy.deepcopy(original)
                row["run_seed"] = seed
                row["row_key"] = digest({key: value for key, value in row.items() if key != "row_key"})
                rows.append(row)
        for original in data.segments:
            if original["split"] != "test":
                segment = copy.deepcopy(original)
                segment["run_seed"] = seed
                segments.append(segment)
    protocol = {**data.protocol, "phase": "pilot"}
    schemas = {name: FeatureSchemaV2(graph_id, schema.feature_names, name) for name, schema in data.schemas.items()}
    features = {name: np.concatenate([values[use]] * 2) for name, values in data.features.items()}
    if graph_id != "G":
        features["neural"] = features["neural"] * 2
    return DatasetV2(protocol, schemas, features, np.concatenate([data.targets[use]] * 2), np.concatenate([data.mask[use]] * 2),
                     np.concatenate([data.eligible[use]] * 2), rows, segments, data.provenance)


def _execution_fixture(tmp_path, monkeypatch, *, failed_replay=False, with_acute=False):
    graphs = [{"id": "G", "role": "G", "source_id": "G", "graph_seed": None, "parameters": {}, "path": "fixture"},
              {"id": "W", "role": "W", "source_id": "W", "graph_seed": 201, "parameters": {}, "path": "W"}]
    if with_acute:
        graphs += [{"id": "P", "role": "P_target", "source_id": "P", "graph_seed": 202,
                    "parameters": {"intervention_mode": "acute"}, "path": "P"}]
    datasets = {graph["id"]: _dataset_with_seeds(tmp_path, graph["id"]) for graph in graphs}
    first = datasets["G"]
    protocol = {**first.protocol, "sources": [{"id": "held-out", "split": "test"}], "parent": "fixture",
                "graphs": graphs, "input_seeds": [0, 1], "budget_seconds": 100, "replay_budget_seconds": 10,
                "comparisons": [{"real": "G", "control": "W"}, {"real": "G", "control": "visual"}]}
    monkeypatch.setattr(experiment, "validate_registration", lambda path: {"protocol": protocol, "registration_id": "fixture"})
    monkeypatch.setattr(experiment, "_planned_sources", lambda protocol: ([], [], []))

    def engine(graph, seed):
        return SimpleNamespace(graph=SimpleNamespace(meta={"source_id": graph}), run_seed=seed, source_id=graph,
                               provenance={"graph_source_id": graph, "graph_sha256": graph, "edge_count": 1,
                                           "common_input_protocol": "same-input-causal-history"})

    monkeypatch.setattr(experiment, "NeuralReplayV2", lambda path, run_seed: engine("G", run_seed))
    monkeypatch.setattr(experiment, "ResearchNeuralReplayV2", lambda parent, path, run_seed: engine(path, run_seed))
    monkeypatch.setattr(experiment, "record_replay", lambda engine, *args, **kwargs: {
        "status": "incomplete" if failed_replay and engine.source_id == "W" and engine.run_seed == 0 else "complete",
        "source_coverage": [{"zero_valid_scene": True}] if failed_replay else []})
    monkeypatch.setattr(experiment, "load_dataset", lambda path: datasets[path.stem.removeprefix("learning_")])
    calls = []

    def fit(current, deadline):
        calls.append((current.schemas["neural"].source_id, current.segments[0]["run_seed"]))
        return fit_readouts(current, deadline=deadline)

    monkeypatch.setattr(experiment, "fit_readouts", fit)
    return calls, datasets


def test_execution_fits_every_graph_seed_independently_and_acute_only_reuses_G(tmp_path, monkeypatch):
    calls, _ = _execution_fixture(tmp_path, monkeypatch, with_acute=True)
    output = tmp_path / "execution"
    result = experiment.execute_experiment("fixture", output)
    assert result["status"] == "complete"
    assert calls == [("G", 0), ("G", 1), ("W", 0), ("W", 1)]
    assert len(result["replay_jobs"]) == len(result["fit_jobs"]) == 6
    assert not result["test_evaluated"]
    assert all(item["split"] != "test" for item in result["comparisons"])
    assert all(item["result"]["interpretation"] == "exploratory_pilot_only" for item in result["comparisons"])
    g = json.loads((output / "fits/G_seed_0/model_neural.json").read_text())
    w = json.loads((output / "fits/W_seed_0/model_neural.json").read_text())
    assert g["schema"]["source_id"] == "G" and w["schema"]["source_id"] == "W"
    assert (output / "fits/P_seed_0/acute_report.json").exists()
    assert not (output / "fits/P_seed_0/model_neural.json").exists()


def test_any_failed_replay_preserves_entire_plan_and_prevents_subset_fit(tmp_path, monkeypatch):
    calls, _ = _execution_fixture(tmp_path, monkeypatch, failed_replay=True)
    result = experiment.execute_experiment("fixture", tmp_path / "execution")
    assert result["status"] == "incomplete" and result["comparisons"] == []
    assert len(result["replay_jobs"]) == len(result["fit_jobs"]) == 4
    assert calls == []
    assert all(item["status"] == "not_executed" for item in result["fit_jobs"])
    assert any(item["status"] == "incomplete" for item in result["replay_jobs"])


def test_cross_graph_coverage_difference_fails_before_fitting(tmp_path, monkeypatch):
    calls, datasets = _execution_fixture(tmp_path, monkeypatch)
    datasets["W"].eligible[40] = False
    output = tmp_path / "execution"
    with pytest.raises(ValueError, match="validity or supervision coverage"):
        experiment.execute_experiment("fixture", output)
    failed = json.loads((output / "run_failed.json").read_text())
    assert failed["status"] == "incomplete" and failed["comparisons"] == [] and calls == []


def test_fixed_readout_path_rejects_R_W_and_different_feature_order(tmp_path):
    dataset = load_dataset(contract_fixture(tmp_path))
    fitted = fit_readouts(dataset)
    with pytest.raises(ValueError, match="restricted"):
        experiment._acute_result(dataset, dataset, fitted, {"role": "W", "parameters": {}})
    changed = copy.deepcopy(dataset)
    schema = dataset.schemas["neural"]
    changed.schemas["neural"] = FeatureSchemaV2("P", tuple(reversed(schema.feature_names)), "neural")
    with pytest.raises(ValueError, match="output identities"):
        experiment._acute_result(dataset, changed, fitted, {"role": "P_target", "parameters": {"intervention_mode": "acute"}})


@pytest.mark.parametrize("operation,expire_on,expected_stage,not_executed,artifact", [
    ("save_fit", 1, "fit_persist:G:0", 5, "fits/G_seed_0/fit_report.json"),
    ("score_predictions", 1, "aggregate_score:G:train:neural", 4, None),
    ("_acute_result", 1, "acute_compute:P:0", 1, None),
    ("_save_acute", 1, "acute_persist:P:0", 1, "fits/P_seed_0/acute_report.json"),
    ("paired_scene_comparison", 4, "comparison:G:visual:val", 0, None),
    ("write_json", 1, "final_persist", 0, "run_summary.json"),
])
def test_total_budget_closes_after_computation_and_persistence(
        tmp_path, monkeypatch, operation, expire_on, expected_stage, not_executed, artifact):
    calls, _ = _execution_fixture(tmp_path, monkeypatch, with_acute=True)
    clock = [0.0]
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: clock[0])
    original = getattr(experiment, operation)
    operation_calls = []

    def exhaust_after_return(*args, **kwargs):
        # Only the final summary write is charged in the persistence case;
        # failure reporting must remain possible after the deadline.
        charged = operation != "write_json" or Path(args[0]).name == "run_summary.json"
        if charged:
            assert clock[0] < 100, "No new optional computation after budget exhaustion"
            operation_calls.append(True)
        result = original(*args, **kwargs)
        if charged and len(operation_calls) == expire_on:
            clock[0] = 1000.0
        return result

    monkeypatch.setattr(experiment, operation, exhaust_after_return)
    output = tmp_path / "execution"
    with pytest.raises(TimeoutError, match=expected_stage):
        experiment.execute_experiment("fixture", output)
    failed = json.loads((output / "run_failed.json").read_text())
    assert failed["status"] == "incomplete" and failed["comparisons"] == []
    assert failed["reason"] == "total_budget_exhausted" and failed["stage"] == expected_stage
    assert failed["wall_seconds"] == 1000
    assert len(failed["replay_jobs"]) == len(failed["fit_jobs"]) == 6
    assert all(job["status"] == "complete" for job in failed["replay_jobs"])
    assert all(job["status"] not in ("running", "pending") for job in failed["fit_jobs"])
    assert sum(job["status"] == "not_executed" for job in failed["fit_jobs"]) == not_executed
    assert len(operation_calls) == expire_on
    if not_executed == 5:
        assert calls == [("G", 0)]
    elif not_executed == 4:
        assert calls == [("G", 0), ("G", 1)]
    else:
        assert calls == [("G", 0), ("G", 1), ("W", 0), ("W", 1)]
    if artifact:
        assert (output / artifact).exists()
    if operation == "write_json":
        summary = json.loads((output / "run_summary.json").read_text())
        assert summary["status"] == "incomplete" and summary["comparisons"] == []
        assert summary["stage"] == "final_persist"
    else:
        assert not (output / "run_summary.json").exists()
