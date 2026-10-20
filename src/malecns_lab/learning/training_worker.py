"""Supervised child for a bounded segment of full-reference neural training."""
from __future__ import annotations

from pathlib import Path
import time
import traceback

import numpy as np
import torch

from ..brain.metadata import canonical_hash
from ..brain.trainable import TrainableLIFConfig
from ..senses.spatial_drive import SpatialDriveConfig
from ..senses.target import TargetCamera
from .joint_probe_worker import raw_counts_digest, snapshot, state_digest
from .neural_policy import NeuralPolicy, NeuralPolicyConfig
from .resource_guard import write_json
from .training_checkpoint import load_training_checkpoint, save_training_checkpoint
from .training_contract import (PILOT, PARAMETERS, bytes_sha, cursor, expected_ids, next_selection, prefix_line,
                                read_arrays, read_json, require, sha256, validate_run)
from .training_evidence import scan_run, validate_resume
from .training_math import compare_r2
from .training_optimizer import make_training_optimizer
from .training_window import compute_window, optimizer_snapshot


def record_resume(segment, registration, policy, optimizer, state, context, previous, mode):
    """Write actual restored values before any subsequent forward or scene reset."""
    params = {key: value for key, value in snapshot(policy).items() if key in PARAMETERS}
    moments = optimizer_snapshot(policy, optimizer, mode, context["optimizer_steps"])
    np.savez(segment / "resume.npz", **{f"parameter.{key}": value for key, value in params.items()},
             **{f"adam.{key}": value for key, value in moments.items()})
    write_json(segment / "resume.json", {"schema": "cnsbridge.training_restored_state.v1",
        "segment_id": registration["segment_id"], "checkpoint_id": previous["checkpoint_id"], "context": context,
        "state": state_digest(state), "arrays_sha256": sha256(segment / "resume.npz")})
    validate_resume(segment, registration, previous)


def execute_windows(segment, run, data, registration, policy, optimizer, state, previous, prefix,
                    *, production=True, phase=lambda *a, **k: None):
    """Run exactly the registered count. Small-circuit callers are explicitly numerical tests.

    No graph constructor/fallback occurs here; run_worker is the sole production
    entry and always supplies a verified full_reference policy.
    """
    completed = registration["start_cursor"]["completed_windows"]
    steps = registration["start_cursor"]["optimizer_steps"]
    parent = previous["checkpoint_id"] if previous else None
    if previous:
        record_resume(segment, registration, policy, optimizer, state, previous["context"], previous, run["mode"])
    (segment / "windows").mkdir(exist_ok=False)
    for _ in range(registration["windows"]):
        number = completed + 1
        window = segment / "windows" / f"{number:06d}"
        window.mkdir(exist_ok=False)
        scene_index, selected = next_selection(data, completed)
        if selected["indices"][0] == 0:
            phase("scene_precondition", scene_index=scene_index, window=number)
            state = policy.initial_state()  # Exactly once per scene, including the first scene.
        require(state is not None and state.frame_index == selected["indices"][0], "State frame cursor mismatches input")
        before_cursor = cursor(run["scenes"], completed, steps)
        state, values, record = compute_window(policy, optimizer, state, selected, mode=run["mode"], before_steps=steps,
                                               phase=phase, failure_directory=window)
        np.savez(window / "evidence.npz", **values)
        comparison = None
        if production and number == 1:
            initial = read_arrays(Path(run["prior_run"]["path"]) / "initial_parameters.npz")
            require(all(np.array_equal(values[f"before_{key}"], initial[key]) for key in (*PARAMETERS, "gains", "rho")),
                    "Initial parameters differ from accepted r2")
            if run["mode"] == "joint":
                comparison = compare_r2(values, run["prior_run"]["path"])
                write_json(window / "r2_comparison.json", comparison)
                require(comparison["passed"], "First-window r2 gate mismatch; window remains unsealed")
        completed += 1
        steps += record["numeric"]["updated"]
        after_cursor = cursor(run["scenes"], completed, steps)
        event = {"schema": "cnsbridge.training_event.v1", "training_run_id": run["training_run_id"],
                 "segment_id": registration["segment_id"], "window": number, "scene_index": scene_index,
                 "scene_id": data.sequences[scene_index].scene["scene_id"], "before_cursor": before_cursor,
                 "after_cursor": after_cursor, "parent_checkpoint_id": parent,
                 "evidence_sha256": sha256(window / "evidence.npz"), "r2_comparison": comparison, **record}
        write_json(window / "event.json", event)
        prefix += prefix_line(registration["segment"], number, sha256(window / "event.json"))
        (window / "prefix.jsonl").write_bytes(prefix)
        context = {**expected_ids(run), "scenes": run["scenes"], **after_cursor,
                   "parent_checkpoint_id": parent, "log_prefix_sha256": bytes_sha(prefix)}
        phase("checkpoint_save", window=number)
        metadata = save_training_checkpoint(policy, optimizer, state, window / "checkpoint", mode=run["mode"], context=context)
        seal = {"schema": "cnsbridge.training_window.v1", "training_run_id": run["training_run_id"],
                "segment_id": registration["segment_id"], "window": number,
                "files_sha256": {name: sha256(window / name) for name in
                    ("evidence.npz", "event.json", "prefix.jsonl", "checkpoint/checkpoint.json")},
                "checkpoint_id": metadata["checkpoint_id"], "parent_checkpoint_id": parent,
                "cursor": after_cursor, "sealed_monotonic": time.monotonic()}
        seal["window_id"] = canonical_hash(seal)
        write_json(window / "window.json", seal)  # Only this final seal makes the window resumable.
        parent = metadata["checkpoint_id"]
        phase("window_sealed", completed_windows=completed)
    return state, completed


def run_worker(segment):
    segment = Path(segment).resolve()
    directory = segment.parents[1]
    started = time.monotonic()
    result = {"schema": "cnsbridge.training_worker.v1", "status": "failed", "completed_windows": 0}
    def phase(name, **details):
        write_json(segment / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - started, **details})
    try:
        phase("validate_registration")
        run = read_json(directory / "run.json")
        data = validate_run(run)
        registration = read_json(segment / "registration.json")
        result.update(training_run_id=run["training_run_id"], segment_id=registration["segment_id"])
        chain = scan_run(directory, run, data, current_segment=registration["segment"])
        require(chain["completed_windows"] == registration["start_cursor"]["completed_windows"], "Worker start prefix changed")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase("load_policy")
        previous = chain["last"]
        if previous:
            policy, optimizer, state, context = load_training_checkpoint(previous["path"], run["graph"]["path"], expected=expected_ids(run))
            require(context == previous["context"], "Actual restored context differs")
        else:
            policy = NeuralPolicy.from_reference(run["graph"]["path"], background=PILOT["background"], dtype=torch.float32,
                config=NeuralPolicyConfig(**PILOT["config"]), camera=TargetCamera(**PILOT["camera"]),
                drive_config=SpatialDriveConfig(**PILOT["drive_config"]), lif_config=TrainableLIFConfig(**PILOT["lif_config"]))
            optimizer = make_training_optimizer(policy, run["mode"])
            state = None
        require(policy.identity == run["policy"] and policy.identity["role"] == "full_reference", "Actual full policy differs from registration")
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == PILOT["raw_ordered_counts_sha256"], "Full graph raw counts changed")
        state, end = execute_windows(segment, run, data, registration, policy, optimizer, state, previous, chain["prefix"], phase=phase)
        result["completed_windows"] = end - registration["start_cursor"]["completed_windows"]
        phase("verify_raw_graph")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == result["raw_counts_sha256_after"], "Training changed graph counts")
        result.update(status="completed", reason="registered_windows_sealed")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        result["failure_kind"] = "io_or_interrupt" if isinstance(error, (OSError, KeyboardInterrupt)) else "numerical_or_protocol"
        traceback.print_exc()
    finally:
        phase("finished")
        result["wall_seconds"] = time.monotonic() - started
        files = ["phase.json"] + (["resume.json", "resume.npz"] if (segment / "resume.json").exists() else [])
        result["artifacts_sha256"] = {name: sha256(segment / name) for name in files if (segment / name).exists()}
        write_json(segment / "worker_result.json", result)
    return result
