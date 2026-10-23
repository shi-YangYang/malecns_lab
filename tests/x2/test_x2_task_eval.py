"""Task protocol, scoring and supervision fixtures only; no neural model or VM."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning import x2_task_eval as task
from malecns_lab.learning import x2_inference as frozen


def frames_for(actual):
    records = []
    for name, distances, labels in task.SCENARIOS:
        for index, (distance, expected) in enumerate(zip(distances, labels)):
            command = expected if actual is None else actual
            raw = [.1 if command == "FORWARD" else 0., -.2, .3]
            records.append({"scenario_id": name, "frame_index": index + 1, "distance_m": distance,
                            "expected_intent": expected, "raw_action": raw, "actual_intent": command})
    return records


def test_preview_help_and_module_import_are_inert(tmp_path):
    output = tmp_path / "no-output"
    result = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_task_eval", "--output", str(output)],
                            capture_output=True, text=True, check=True)
    preview = json.loads(result.stdout)
    assert preview["brain_loaded"] is False and preview["protocol"] == task.protocol()
    assert not output.exists()
    probe = subprocess.run([sys.executable, "-c", "import sys; import malecns_lab.experiments.x2_task_eval; "
                            "import malecns_lab.learning.x2_task_eval; "
                            "assert not {'torch','numpy','scipy','rclpy','cnsbridge'}.intersection(sys.modules)"], check=False)
    assert probe.returncode == 0
    help_result = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_task_eval", "--help"],
                                 capture_output=True, text=True)
    assert help_result.returncode == 0 and "--execute" in help_result.stdout
    missing = subprocess.run([sys.executable, "-m", "malecns_lab.experiments.x2_task_eval", "--execute"],
                             capture_output=True, text=True)
    assert missing.returncode != 0


def test_frozen_task_counts_and_absent_is_actual_gray_pixels():
    protocol = task.protocol()
    assert protocol["expected_counts"] == {"FORWARD": 6, "STOP": 9}
    assert [item["scenario_id"] for item in protocol["scenarios"]] == [
        "far_hold", "near_hold", "absent_hold", "approach", "disappearance"]
    assert protocol["scenarios"][3]["distances_m"] == [2., 1.5, .5]
    protocol["scenarios"][0]["distances_m"][0] = 99
    assert task.protocol()["scenarios"][0]["distances_m"] == [2., 2., 2.]
    absent = task.render_input(None)
    assert absent.shape == (120, 160, 3) and absent.dtype == np.uint8 and np.all(absent == 32)
    counts = [np.count_nonzero(task.render_input(distance)[:, :, 2] == 235) for distance in (2., 1.5, .5)]
    assert 0 < counts[0] < counts[1] < counts[2]


@pytest.mark.parametrize("actual,correct,passed", [(None, 15, True), ("FORWARD", 6, False), ("STOP", 9, False)])
def test_scoring_does_not_turn_behavior_failure_into_incomplete_execution(actual, correct, passed):
    result = task.score_frames(frames_for(actual))
    assert result["behavior_pass"] is passed and result["correct"] == correct and result["valid_frames"] == 15
    assert result["forward_hits"]["total"] == 6 and result["stop_hits"]["total"] == 9
    assert result["constant_policies"] == {"FORWARD": {"correct": 6, "total": 15}, "STOP": {"correct": 9, "total": 15}}


def test_scoring_uses_strict_original_vx_threshold_and_rejects_incomplete_or_nonfinite():
    frames = frames_for("STOP")
    frames[0]["raw_action"] = [.03, 10., -10.]
    assert task.score_frames(frames)["frames"][0]["actual"] == "STOP"
    frames[0]["raw_action"][0] = .03000001
    with pytest.raises(ValueError, match="intent"):
        task.score_frames(frames)
    frames[0]["actual_intent"] = "FORWARD"
    assert task.score_frames(frames)["frames"][0]["actual"] == "FORWARD"
    with pytest.raises(ValueError, match="15 valid"):
        task.score_frames(frames[:-1])
    frames[0]["raw_action"][1] = float("nan")
    with pytest.raises(ValueError, match="Invalid raw"):
        task.score_frames(frames)


def test_scoring_rejects_label_or_order_changes():
    frames = frames_for(None)
    frames[0]["expected_intent"] = "STOP"
    with pytest.raises(ValueError, match="preregistered"):
        task.score_frames(frames)
    frames = frames_for(None)
    frames[0], frames[1] = frames[1], frames[0]
    with pytest.raises(ValueError, match="order"):
        task.score_frames(frames)


def test_output_and_direct_worker_guards_precede_heavy_imports(tmp_path):
    existing = tmp_path / "data/existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        task.execute(existing, tmp_path)
    with pytest.raises(ValueError, match="beneath"):
        task.execute(tmp_path / "outside", tmp_path)
    registration = {"parent_pid": -1, "output": str(existing)}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (existing / "registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="supervising parent"):
        task.run_worker(existing)
    assert not (existing / "worker.claim").exists()
    registration = {"parent_pid": os.getppid(), "output": str(existing), "root": str(tmp_path)}
    registration["registration_id"] = frozen.canonical_hash(registration)
    (existing / "registration.json").write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="shared complete-brain lock"):
        task.run_worker(existing)
    assert not (existing / "worker.claim").exists()
    with frozen.numerical_lock(tmp_path):
        (existing / "worker.claim").touch()
        with pytest.raises(FileExistsError):
            task.run_worker(existing)


def test_preflight_failure_preserves_failed_run_without_launch(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    def reject(_):
        raise ValueError("simulation_running")
    monkeypatch.setattr(frozen, "simulator_preflight", reject)
    result = task.execute(tmp_path / "data/failed", tmp_path)
    assert result["status"] == "failed" and result["behavior_pass"] is None
    assert "simulation_running" in result["reason"]
    assert (tmp_path / "data/failed/summary.json").exists()
    assert not (tmp_path / "data/failed/worker.claim").exists()


def test_parent_supervision_binds_protocol_and_preserves_valid_behavior_failure(tmp_path, monkeypatch):
    from malecns_lab.learning import resource_guard
    import psutil
    (tmp_path / "data").mkdir()
    for name in task.WRAPPER_FILES:
        path = tmp_path / "src/malecns_lab" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# wrapper serialization fixture\n")
    monkeypatch.setattr(frozen, "simulator_preflight", lambda root: {"gpu_socket_exists": False})
    monkeypatch.setattr(frozen, "read_anchors", lambda root: {
        "run": {"implementation_sha256": {}, "runtime": {}, "policy": {"policy_id": "serialization-only"}},
        "cpu": {}, "checkpoint": {}, "window": {}})
    monkeypatch.setattr(frozen, "export_sources", lambda *args: {"registered_source_count": 54})
    monkeypatch.setattr(task, "runtime", lambda: {"backend": "test-no-numerical-work"})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=3*frozen.GIB))
    calls = []
    def supervised(command, directory, **budget):
        with pytest.raises(BlockingIOError):
            with frozen.numerical_lock(tmp_path):
                pytest.fail("Parent lock was released before supervision")
        calls.append((command, budget))
        reg = frozen.read_json(directory / "registration.json")
        frames = frames_for("FORWARD")
        scores = task.score_frames(frames)
        resource_guard.write_json(directory / "scores.json", scores)
        resource_guard.write_json(directory / "worker_result.json", {
            "status": "completed", "completed_frames": 15, "completed_scenarios": 5,
            "saved_state_boundaries": 10, "parameters_unchanged": True,
            "registration_id": reg["registration_id"], "frames": frames, "scores": scores})
        return {"status": "completed", "reason": "worker_exited"}
    monkeypatch.setattr(resource_guard, "supervise", supervised)
    result = task.execute(tmp_path / "data/run", tmp_path)
    assert result["status"] == "completed" and result["behavior_pass"] is False and len(calls) == 1
    command, budget = calls[0]
    assert command[-2:] == ["--_worker", str(tmp_path / "data/run")]
    assert budget["rss_limit_bytes"] == 2*frozen.GIB and budget["reserve_bytes"] == frozen.GIB
    assert budget["poll_seconds"] == .1 and 0 < budget["wall_seconds"] <= 240
    reg = frozen.read_json(tmp_path / "data/run/registration.json")
    frozen.bound(reg, "registration_id")
    frozen.bound(reg["execution_identity"], "execution_id")
    assert reg["protocol"] == reg["execution_identity"]["protocol"] == task.protocol()
    reg["protocol"]["scenarios"][0]["expected_intents"][0] = "STOP"
    with pytest.raises(ValueError, match="registration_id"):
        frozen.bound(reg, "registration_id")


def telemetry_archive(directory):
    # Archive-only dictionaries, not a model, graph or spike simulation.
    from malecns_lab.learning.resource_guard import write_json
    from malecns_lab.learning.x2_telemetry import BATCH_SCHEMA, MANIFEST_SCHEMA, SUMMARY_SCHEMA
    telemetry = directory / "telemetry"
    (telemetry / "batches").mkdir(parents=True)
    summary = {"schema": SUMMARY_SCHEMA, "run_id": "scene", "execution_id": "backend", "complete": True,
               "status": "completed", "error": None, "errors": [], "dropped_batches": 0, "produced_batches": 42,
               "written_batches": 42, "batch_count": 42, "partial_batch_steps": 0}
    write_json(telemetry / "summary.json", summary)
    write_json(telemetry / "manifest.json", {"schema": MANIFEST_SCHEMA, "run_id": "scene", "execution_id": "backend",
               "checkpoint_id": frozen.CHECKPOINT_ID, "policy_id": "policy", "batch_steps": 50, "dt_ms": .1,
               "groups": [{} for _ in range(22)], "selected_neurons": [{"body_id": str(index)} for index in range(32)]})
    for index in range(1, 43):
        observation = None if index <= 12 else (index - 13) // 10 + 1
        batch = {"schema": BATCH_SCHEMA, "run_id": "scene", "execution_id": "backend", "batch_sequence": index,
                 "step_start": (index - 1)*50, "step_end": index*50, "dt_ms": .1, "dropped_batches": 0,
                 "phase": "precondition" if observation is None else "forward", "observation_sequence": observation,
                 "decision_sequence": None, "group_rates_hz": [0.] * 22, "spike_events": []}
        write_json(telemetry / "batches" / f"{index:06d}.json", batch)
    write_json(telemetry / "latest.json", batch)
    return summary


def test_telemetry_audit_requires_every_scene_clock_and_complete_batch(tmp_path):
    summary = telemetry_archive(tmp_path)
    audit = task.audit_telemetry(tmp_path, summary, "scene", "backend", "policy")
    assert len(audit["files_sha256"]) == 45
    summary["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        task.audit_telemetry(tmp_path, summary, "scene", "backend", "policy")
    summary["complete"] = True
    batch_path = tmp_path / "telemetry/batches/000013.json"
    batch = frozen.read_json(batch_path)
    batch["observation_sequence"] = 2
    batch_path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="clock"):
        task.audit_telemetry(tmp_path, summary, "scene", "backend", "policy")
    batch_path.unlink()
    with pytest.raises(ValueError, match="inventory"):
        task.audit_telemetry(tmp_path, summary, "scene", "backend", "policy")
