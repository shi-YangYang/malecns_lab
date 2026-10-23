"""Preregistered X2 task screening around the unchanged frozen full-brain path.

Imports are stdlib-only. Pixels, action scoring and evidence are new; the old
policy, numerical wrapper, checkpoint and serialized states remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import sys
import time
import traceback

from . import x2_inference as frozen

SCENARIOS = (
    ("far_hold", (2., 2., 2.), ("FORWARD", "FORWARD", "FORWARD")),
    ("near_hold", (.5, .5, .5), ("STOP", "STOP", "STOP")),
    ("absent_hold", (None, None, None), ("STOP", "STOP", "STOP")),
    ("approach", (2., 1.5, .5), ("FORWARD", "FORWARD", "STOP")),
    ("disappearance", (2., None, None), ("FORWARD", "STOP", "STOP")),
)
WRAPPER_FILES = frozen.WRAPPER_FILES + (
    "experiments/x2_task_eval.py", "learning/x2_task_eval.py", "learning/x2_frozen.py",
    "learning/x2_telemetry.py", "control/x2_protocol.py",
)
SCHEMA = "malecns_lab.x2_task_evaluation.v1"


def protocol():
    """Fresh JSON-compatible copy; labels are assessment data, never model input."""
    return {
        "schema": "malecns_lab.x2_task_protocol.v1", "camera": {
            "width": 160, "height": 120, "horizontal_fov_deg": 72., "marker_size_m": .2},
        "camera_position_m": [0., 0., .65], "camera_orientation_xyzw": [0., 0., 0., 1.],
        "source": "offline_virtual_marker_task_screen", "absent_pixels": "uint8 BGR gray32",
        "scenarios": [{"scenario_id": name, "distances_m": list(distances), "expected_intents": list(labels)}
                      for name, distances, labels in SCENARIOS],
        "precondition_steps_per_scene": 600, "steps_per_frame": 500, "neural_dt_ms": .1,
        "reset": "independent initial_state per scene; recursive state retained within scene",
        "intent": "control.x2_protocol.intent: raw vx > 0.03 => FORWARD; else STOP",
        "scene_pass": "3/3 valid correct frames", "behavior_pass": "15/15 valid correct frames",
        "expected_counts": {"FORWARD": 6, "STOP": 9}, "telemetry_batches_per_scene": 42,
        "scope": "offline screening; no camera sensor, dynamics, ROS, training or robot motion",
    }


def runtime():
    value = frozen.runtime("optimized")
    value["backend"] = "historical_GPUPolicy_optimized_sticky50_task_screen"
    return value


def render_input(distance):
    import numpy as np
    from ..senses.x2_marker import CAMERA, marker_frame
    frozen.require(CAMERA == protocol()["camera"], "Registered camera changed")
    if distance is None:
        return np.full((CAMERA["height"], CAMERA["width"], 3), 32, dtype=np.uint8)
    return marker_frame((0., 0., .65), (0., 0., 0., 1.), (distance, 0., .65))


def score_frames(frames):
    """Reject incomplete/invalid observations; a valid behavioral FAIL is a result."""
    from ..control.x2_protocol import intent
    frozen.require(len(frames) == 15, "Scoring requires all 15 valid frames")
    confusion = {expected: {actual: 0 for actual in ("FORWARD", "STOP")} for expected in ("FORWARD", "STOP")}
    scenes, scored = [], []
    for scene_index, (name, distances, labels) in enumerate(SCENARIOS):
        matches = 0
        for index, (distance, expected) in enumerate(zip(distances, labels)):
            frame = frames[scene_index * 3 + index]
            frozen.require(frame["scenario_id"] == name and frame["frame_index"] == index + 1
                           and frame["distance_m"] == distance and frame["expected_intent"] == expected,
                           "Scoring frame order or preregistered label differs")
            actual = intent(frame["raw_action"])
            frozen.require(frame["actual_intent"] == actual, "Recorded intent differs from raw action")
            correct = actual == expected
            matches += correct
            confusion[expected][actual] += 1
            scored.append({"scenario_id": name, "frame_index": index + 1, "expected": expected,
                           "actual": actual, "correct": correct, "raw_action": frame["raw_action"]})
        scenes.append({"scenario_id": name, "correct": matches, "total": 3, "pass": matches == 3})
    correct = sum(scene["correct"] for scene in scenes)
    return {"schema": "malecns_lab.x2_task_scores.v1", "valid_frames": 15, "correct": correct,
            "accuracy": correct / 15, "behavior_pass": correct == 15, "scenes": scenes, "frames": scored,
            "confusion_expected_actual": confusion,
            "forward_hits": {"correct": confusion["FORWARD"]["FORWARD"], "total": 6},
            "stop_hits": {"correct": confusion["STOP"]["STOP"], "total": 9},
            "constant_policies": {"FORWARD": {"correct": 6, "total": 15}, "STOP": {"correct": 9, "total": 15}},
            "limitations": "One fixed correlated sequence per scene; new X2 task labels, not original training labels. "
                           "No statistical generalization, physical stopping or real-time claim."}


def audit_telemetry(directory, summary, run_id, execution_id, policy_id):
    """Require every batch and clock, rather than treating observer failures as success."""
    from .x2_telemetry import BATCH_SCHEMA, MANIFEST_SCHEMA, SUMMARY_SCHEMA
    directory = Path(directory) / "telemetry"
    frozen.require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("complete") is True
                   and summary.get("status") == "completed" and summary.get("error") is None
                   and summary.get("errors") == [] and summary.get("dropped_batches") == 0
                   and summary.get("produced_batches") == summary.get("written_batches") == summary.get("batch_count") == 42
                   and summary.get("partial_batch_steps") == 0 and summary.get("run_id") == run_id
                   and summary.get("execution_id") == execution_id, "Telemetry is incomplete or identity differs")
    manifest = frozen.read_json(directory / "manifest.json")
    frozen.require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("run_id") == run_id
                   and manifest.get("execution_id") == execution_id and manifest.get("checkpoint_id") == frozen.CHECKPOINT_ID
                   and manifest.get("policy_id") == policy_id and manifest.get("batch_steps") == 50
                   and manifest.get("dt_ms") == .1 and len(manifest.get("groups", [])) == 22
                   and len(manifest.get("selected_neurons", [])) == 32, "Telemetry manifest differs")
    stored = frozen.read_json(directory / "summary.json")
    frozen.require({key: value for key, value in stored.items() if key != "host_monotonic_ns"}
                   == {key: value for key, value in summary.items() if key != "host_monotonic_ns"},
                   "Telemetry stored summary differs")
    selected = {item["body_id"] for item in manifest["selected_neurons"]}
    expected_files = {f"{index:06d}.json" for index in range(1, 43)}
    frozen.require({path.name for path in (directory / "batches").iterdir()} == expected_files, "Telemetry batch inventory differs")
    files = {name: frozen.sha256(directory / name) for name in ("manifest.json", "summary.json", "latest.json")}
    for index in range(1, 43):
        name = f"batches/{index:06d}.json"
        batch = frozen.read_json(directory / name)
        observation = None if index <= 12 else (index - 13) // 10 + 1
        frozen.require(batch.get("schema") == BATCH_SCHEMA and batch.get("run_id") == run_id
                       and batch.get("execution_id") == execution_id and batch.get("batch_sequence") == index
                       and batch.get("step_start") == (index - 1) * 50 and batch.get("step_end") == index * 50
                       and batch.get("phase") == ("precondition" if observation is None else "forward")
                       and batch.get("observation_sequence") == observation and batch.get("dt_ms") == .1
                       and batch.get("dropped_batches") == 0, "Telemetry batch clock or identity differs")
        rates, events = batch.get("group_rates_hz"), batch.get("spike_events")
        frozen.require(isinstance(rates, list) and len(rates) == 22
                       and all(type(rate) in (int, float) and math.isfinite(rate) and rate >= 0 for rate in rates)
                       and isinstance(events, list) and batch.get("decision_sequence") is None
                       and all(isinstance(event, dict) and event.get("body_id") in selected
                               and type(event.get("step")) is int
                               and batch["step_start"] <= event["step"] < batch["step_end"] for event in events),
                       "Invalid telemetry rates or selected spike events")
        files[name] = frozen.sha256(directory / name)
    frozen.require(frozen.read_json(directory / "latest.json") == batch, "Telemetry latest batch differs")
    return {"summary": summary, "files_sha256": files}


def execute(output, root=frozen.ROOT):
    from .resource_guard import resource_budget, supervise, write_json
    import psutil
    root, output = Path(root).resolve(), Path(output).resolve()
    frozen.require(any(output.is_relative_to(root / name) and output != root / name for name in ("data", "logs")),
                   "Output must be a new directory beneath this project's data/ or logs/")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": SCHEMA, "status": "failed", "reason": None, "output": str(output), "behavior_pass": None}
    try:
        with frozen.numerical_lock(root):
            write_json(output / "phase.json", {"phase": "preflight"})
            write_json(output / "simulator_preflight.json", frozen.simulator_preflight(root))
            anchors = frozen.read_anchors(root)
            sources = frozen.export_sources(output / "sources", anchors["run"]["implementation_sha256"], root)
            write_json(output / "sources.json", sources)
            (output / "provenance").mkdir()
            for key, value in anchors.items():
                write_json(output / "provenance" / f"{key}.json", value)
            wrappers = {name: frozen.sha256(root / "src/malecns_lab" / name) for name in WRAPPER_FILES}
            for name in WRAPPER_FILES:
                path = output / "wrapper_sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write((root / "src/malecns_lab" / name).read_bytes())
                path.chmod(0o444)
            resources = resource_budget(psutil.virtual_memory().available, rss_bytes=6*frozen.GIB, reserve_bytes=frozen.GIB)
            current = runtime()
            execution = {"schema": "malecns_lab.x2_task_execution.v1", "source_policy_id": anchors["run"]["policy"]["policy_id"],
                         "checkpoint_id": frozen.CHECKPOINT_ID, "wrapper_sha256": wrappers, "runtime": current,
                         "protocol": protocol(), "telemetry": True}
            execution["execution_id"] = frozen.canonical_hash(execution)
            registration = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                            "execution_identity": execution, "protocol": protocol(), "parent_pid": os.getpid(),
                            "root": str(root), "output": str(output), "resources": resources,
                            "wall_limit_seconds": frozen.WALL_SECONDS, "parent_started_monotonic": started,
                            "checkpoint_id": frozen.CHECKPOINT_ID, "checkpoint_path": str(root / frozen.CHECKPOINT),
                            "graph_path": str(root / frozen.REFERENCE), "source_manifest_sha256": frozen.sha256(output / "sources.json"),
                            "wrapper_sha256": wrappers, "runtime": current, "historical_runtime": anchors["run"]["runtime"],
                            "mode": "five_independent_scenes_no_grad; no_optimizer; no_historical_neural_state"}
            registration["registration_id"] = frozen.canonical_hash(registration)
            write_json(output / "registration.json", registration)
            frozen.require(resources["can_launch"], "Insufficient system reserve before launch")
            remaining = frozen.WALL_SECONDS - (time.monotonic() - started)
            frozen.require(remaining > 0, "Preparation exceeded wall limit")
            supervision = supervise([sys.executable, "-m", "malecns_lab.experiments.x2_task_eval", "--_worker", str(output)],
                                    output, rss_limit_bytes=resources["effective_rss_limit_bytes"], reserve_bytes=frozen.GIB,
                                    wall_seconds=remaining, poll_seconds=.1)
            result["supervisor"] = supervision
            worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
            frozen.require(supervision["status"] == "completed" and worker.get("status") == "completed"
                           and worker.get("completed_frames") == 15 and worker.get("completed_scenarios") == 5
                           and worker.get("saved_state_boundaries") == 10 and worker.get("parameters_unchanged") is True
                           and worker.get("registration_id") == registration["registration_id"],
                           f"Incomplete evaluation: {supervision['reason']}; {worker.get('reason', 'no worker result')}")
            scores = score_frames(worker["frames"])
            frozen.require(scores == frozen.read_json(output / "scores.json") and scores == worker["scores"], "Sealed scores differ")
            result.update(status="completed", reason="five_frozen_task_scenarios_sealed", behavior_pass=scores["behavior_pass"],
                          scores=scores, registration_id=registration["registration_id"], execution_id=execution["execution_id"])
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", result)
    return result


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).resolve()
    reg = frozen.read_json(directory / "registration.json")
    frozen.bound(reg, "registration_id")
    frozen.require(os.getppid() == reg["parent_pid"] and str(directory) == reg["output"], "Worker requires its supervising parent")
    root = Path(reg["root"])
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must hold the shared complete-brain lock")
    with (directory / "worker.claim").open("x") as claim:
        claim.write(f"{os.getpid()}\n")
    started = time.monotonic()
    result = {"status": "failed", "reason": None, "behavior_pass": None, "completed_frames": 0,
              "completed_scenarios": 0, "saved_state_boundaries": 0, "registration_id": reg["registration_id"],
              "checkpoint_id": frozen.CHECKPOINT_ID, "frames": [], "scenarios": []}
    timing = {}
    sampler = telemetry = None
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started, **details})
        if sampler is not None:
            sampler.sample(name)
    try:
        phase("validate_frozen_sources")
        frozen.require(reg["schema"] == SCHEMA and reg["protocol"] == protocol(), "Registered task protocol differs")
        frozen.require(frozen.sha256(directory / "sources.json") == reg["source_manifest_sha256"], "Source manifest changed")
        sources = frozen.read_json(directory / "sources.json")
        frozen.verify_sources(directory / "sources", sources)
        frozen.require(set(reg["wrapper_sha256"]) == set(WRAPPER_FILES), "Wrapper inventory differs")
        for name, digest in reg["wrapper_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == digest
                           and frozen.sha256(directory / "wrapper_sources" / name) == digest, f"Task wrapper changed: {name}")
        anchors = frozen.read_anchors(root)
        frozen.require(sources["registered_sha256"] == anchors["run"]["implementation_sha256"], "Snapshot registration differs")
        execution = reg["execution_identity"]
        frozen.bound(execution, "execution_id")
        frozen.require(execution["source_policy_id"] == anchors["run"]["policy"]["policy_id"]
                       and execution["checkpoint_id"] == reg["checkpoint_id"] == frozen.CHECKPOINT_ID
                       and execution["wrapper_sha256"] == reg["wrapper_sha256"] and execution["runtime"] == reg["runtime"]
                       and execution["protocol"] == reg["protocol"] and execution["telemetry"] is True,
                       "Execution identity differs")
        frozen.require(not any(name == "cnsbridge" or name.startswith("cnsbridge.") for name in sys.modules),
                       "Historical namespace was already imported")
        snapshot = directory / "sources/src/cnsbridge"
        sys.path.insert(0, str(snapshot.parent))
        sys.dont_write_bytecode = True
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        phase("imports")
        began = time.monotonic()
        import numpy as np
        import torch
        from cnsbridge.brain.mps_trainable import require_mps
        from cnsbridge.learning.gpu_policy import GPUPolicy
        from cnsbridge.learning.gpu_state import state_arrays, state_metadata
        from cnsbridge.learning.gpu_archive import validate_state
        from cnsbridge.learning.joint_probe_worker import raw_counts_digest
        from .mps_memory import MPSMemorySampler
        from .x2_frozen import FrozenInference
        from .x2_telemetry import SpikeTelemetry
        from ..control.x2_protocol import intent
        frozen._assert_snapshot_imports(snapshot)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require_mps()
        frozen.require(runtime() == reg["runtime"], "Worker runtime differs")
        timing["imports_seconds"] = time.monotonic() - began
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(directory / "mps_budget.json", sampler.budget)
        def clock():
            torch.mps.synchronize()
            return time.monotonic()
        phase("load_frozen_parameters")
        began = clock()
        values = frozen.load_parameters(root / frozen.CHECKPOINT / "arrays.npz", anchors["checkpoint"], anchors["run"]["policy"])
        before = {key: frozen._array_declaration(value) for key, value in values.items()}
        frozen.require(frozen.sha256(root / frozen.REFERENCE) == anchors["run"]["graph"]["file_sha256"], "Reference file changed")
        timing["parameter_and_reference_verification_seconds"] = clock() - began
        phase("construct_full_policy")
        began = clock()
        policy = GPUPolicy.from_reference(root / frozen.REFERENCE, anchors["cpu"]["policy"],
                                          credit_horizon=anchors["run"]["protocol"]["credit_horizon_steps"])
        frozen.require(policy.identity == anchors["run"]["policy"] and policy.camera.__dict__ == protocol()["camera"],
                       "Actual full GPU policy identity or camera differs")
        policy.restore_parameters(values)
        policy.eval()
        executor = FrozenInference(policy, checkpoint_id=frozen.CHECKPOINT_ID)
        backend = executor.execution_identity
        frozen.bound(backend, "execution_id")
        frozen.require(backend["checkpoint_id"] == frozen.CHECKPOINT_ID and backend["source_policy_id"] == policy.policy_id
                       and backend["implementation_sha256"] == reg["wrapper_sha256"]["learning/x2_frozen.py"],
                       "Actual optimized execution identity differs")
        write_json(directory / "execution_identity.json", {"registration_execution": execution, "backend_execution": backend})
        result.update(execution_id=execution["execution_id"], backend_execution_id=backend["execution_id"], policy_id=policy.policy_id)
        timing["construction_and_restore_seconds"] = clock() - began
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        frozen.require(result["raw_counts_sha256_before"] == policy.identity["model"]["graph"]["raw_ordered_counts_sha256"],
                       "Actual complete graph differs")
        write_json(directory / "actual_policy.json", policy.identity)
        (directory / "scenarios").mkdir()
        initial_digest = None
        with torch.no_grad():
            for scene_index, (name, distances, labels) in enumerate(SCENARIOS):
                scene_dir = directory / "scenarios" / name
                scene_dir.mkdir()
                (scene_dir / "states").mkdir()
                (scene_dir / "frames").mkdir()
                identity = {"schema": "malecns_lab.x2_task_scene.v1", "registration_id": reg["registration_id"],
                            "task_execution_id": execution["execution_id"], "backend_execution_id": backend["execution_id"],
                            "scenario_id": name, "scenario_index": scene_index, "distances_m": list(distances),
                            "expected_intents": list(labels)}
                identity["run_id"] = frozen.canonical_hash(identity)
                write_json(scene_dir / "identity.json", identity)
                telemetry = SpikeTelemetry(scene_dir, policy, backend, identity["run_id"])
                executor.telemetry = telemetry
                scene_result = {**identity, "boundaries": [], "frames": []}
                result["scenarios"].append(scene_result)
                def state_record(state, *, save):
                    policy.validate_state(state)
                    arrays = state_arrays(state, detached=True)
                    metadata = state_metadata(state, policy.drive_config.dt_ms)
                    digest = validate_state(metadata, arrays, policy.identity, state.frame_index)
                    record = {"frame_index": state.frame_index, "metadata": metadata, "state_digest": digest,
                              "arrays": {key: frozen._array_declaration(value) for key, value in arrays.items()}}
                    if save:
                        path = scene_dir / "states" / f"{state.frame_index:03d}.npz"
                        np.savez(path, **arrays)
                        record.update(file=str(path.relative_to(directory)), file_sha256=frozen.sha256(path))
                        result["saved_state_boundaries"] += 1
                    write_json(scene_dir / "states" / f"{state.frame_index:03d}.json", record)
                    scene_result["boundaries"].append(record)
                    return record
                phase("new_scene_blank_precondition", scenario_id=name, neural_steps=600)
                telemetry.begin("precondition")
                began = clock()
                state = executor.initial_state()
                scene_result["precondition_seconds"] = clock() - began
                previous = state_record(state, save=True)
                if initial_digest is None:
                    initial_digest = previous["state_digest"]
                frozen.require(previous["state_digest"] == initial_digest, "Independent scene initial states differ")
                for index, (distance, expected) in enumerate(zip(distances, labels)):
                    phase("render_input", scenario_id=name, frame_index=index + 1)
                    frame = render_input(distance)
                    input_path = scene_dir / "frames" / f"{index:03d}.npy"
                    np.save(input_path, frame, allow_pickle=False)
                    phase("forward", scenario_id=name, frame_index=index + 1, neural_steps=500)
                    telemetry.begin("forward", index + 1)
                    began = clock()
                    output = executor.forward_tensor(frame, state)
                    elapsed = clock() - began
                    state = output.state
                    arrays = {"action": output.action.detach().cpu().numpy().copy(),
                              "features_mv": output.features_mv.detach().cpu().numpy().copy(),
                              "logits": output.logits.detach().cpu().numpy().copy()}
                    output_path = scene_dir / "frames" / f"{index:03d}-output.npz"
                    np.savez(output_path, **arrays)  # Preserve invalid arrays if the finite check fails.
                    frozen.require(arrays["action"].shape == arrays["logits"].shape == (3,)
                                   and arrays["features_mv"].shape == (policy.identity["features"]["width"],)
                                   and all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values()),
                                   "Invalid neural output")
                    boundary = state_record(state, save=index == 2)
                    raw = arrays["action"].tolist()
                    record = {"scenario_id": name, "scenario_index": scene_index, "frame_index": index + 1,
                              "run_id": identity["run_id"], "task_execution_id": execution["execution_id"],
                              "backend_execution_id": backend["execution_id"], "source": protocol()["source"],
                              "distance_m": distance, "expected_intent": expected, "actual_intent": intent(raw),
                              "camera_position_m": [0., 0., .65], "camera_orientation_xyzw": [0., 0., 0., 1.],
                              "target_world_m": None if distance is None else [distance, 0., .65],
                              "input": frozen._array_declaration(frame), "input_file": str(input_path.relative_to(directory)),
                              "input_file_sha256": frozen.sha256(input_path), "output_file": str(output_path.relative_to(directory)),
                              "output_file_sha256": frozen.sha256(output_path),
                              "outputs": {key: frozen._array_declaration(value) for key, value in arrays.items()},
                              "raw_action": raw, "features_mv": arrays["features_mv"].tolist(), "logits": arrays["logits"].tolist(),
                              "action_axes": ["vx", "vz", "wz"], "action_units": ["m/s", "m/s", "rad/s"],
                              "forward_seconds": elapsed, "neural_frame_ms": 50., "brain_step": state.brain.step,
                              "drive_step": state.drive.step, "state_before": previous["state_digest"],
                              "state_after": boundary["state_digest"]}
                    write_json(scene_dir / "frames" / f"{index:03d}.json", record)
                    scene_result["frames"].append(record)
                    result["frames"].append(record)
                    result["completed_frames"] += 1
                    previous = boundary
                closed = telemetry.close(completed=True)
                executor.telemetry = None
                scene_result["telemetry_summary"] = closed
                scene_result["telemetry"] = audit_telemetry(scene_dir, closed, identity["run_id"], backend["execution_id"], policy.policy_id)
                telemetry = None
                write_json(scene_dir / "result.json", scene_result)
                result["completed_scenarios"] += 1
        phase("verify_frozen_parameters_and_graph")
        after = {key: frozen._array_declaration(dict(policy.named_parameters())[name].detach().cpu().numpy())
                 for key, name in frozen.PARAMETERS.items()}
        frozen.require(after == before and all(value.grad is None for value in policy.parameters()), "Frozen parameters changed or gradients exist")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        frozen.require(result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"], "Graph changed during inference")
        frozen._assert_snapshot_imports(snapshot)
        write_json(directory / "parameters.json", {"before": before, "after": after, "unchanged": True,
                                                   "autograd_enabled_for_inference": False, "optimizer_constructed": False})
        scores = score_frames(result["frames"])
        write_json(directory / "scores.json", scores)
        result.update(status="completed", reason="five_frozen_task_scenarios", scores=scores,
                      behavior_pass=scores["behavior_pass"], parameters_unchanged=True, initial_states_equal=True)
        phase("sealed")
    except BaseException as error:
        result.update(status="failed", behavior_pass=None, reason=f"{type(error).__name__}:{error}")
        traceback.print_exc()
    finally:
        if telemetry is not None:
            result["partial_telemetry"] = telemetry.close(completed=False)
        if sampler is not None:
            try:
                result["mps_memory"] = sampler.close()
                if result["mps_memory"]["error"] is not None:
                    result.update(status="failed", behavior_pass=None, reason="mps_memory_sampling_failed")
            except BaseException as error:
                result.update(status="failed", behavior_pass=None, reason=f"mps_memory_cleanup_failed:{type(error).__name__}:{error}")
        timing["worker_seconds"] = time.monotonic() - started
        write_json(directory / "timing.json", timing)
        result["timing"] = timing
        write_json(directory / "worker_result.json", result)
    return result
