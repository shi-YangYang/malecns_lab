"""Frozen full-reference session; heavy imports occur only in supervised worker."""
from __future__ import annotations

import fcntl
import os
import signal
from pathlib import Path
import sys
import time
import traceback
import uuid

from ..control.x2_protocol import (
    atomic, seal, verify, read, intent, MAX_ROUNDS, BRAIN_TIMEOUT, OBSERVATION_TIMEOUT, POSE_TIMEOUT,
)
from . import x2_inference as base

WRAPPERS = tuple(dict.fromkeys(base.WRAPPER_FILES + (
    "learning/x2_frozen.py", "learning/x2_session.py", "experiments/x2_neural.py",
    "experiments/x2_bridge.py", "experiments/x2.py", "control/x2_protocol.py", "control/x2.py")))


def execute(output, root=base.ROOT):
    from .resource_guard import resource_budget, supervise, write_json
    import psutil
    root, directory = Path(root).resolve(), Path(output).resolve()
    base.require(directory.is_relative_to(root / "data") and directory != root / "data", "New output must be beneath project data/")
    directory.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    run_id = uuid.uuid4().hex
    result = {"status": "failed", "reason": None, "run_id": run_id, "output": str(directory)}
    def interrupted(signum, _frame):
        raise KeyboardInterrupt("host_signal:" + str(signum))
    old_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        with base.numerical_lock(root):
            write_json(directory / "simulator_preflight.json", base.simulator_preflight(root))
            anchors = base.read_anchors(root)
            sources = base.export_sources(directory / "sources", anchors["run"]["implementation_sha256"], root)
            write_json(directory / "sources.json", sources)
            for key, value in anchors.items():
                path = directory / "provenance" / f"{key}.json"
                path.parent.mkdir(exist_ok=True)
                write_json(path, value)
            wrappers = {name: base.sha256(root / "src/malecns_lab" / name) for name in WRAPPERS}
            for name in WRAPPERS:
                path = directory / "wrapper_sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((root / "src/malecns_lab" / name).read_bytes())
                path.chmod(0o444)
            budget = resource_budget(psutil.virtual_memory().available, rss_bytes=6*base.GIB, reserve_bytes=base.GIB)
            registration = {"schema": "malecns_lab.x2_neural_loop.v1", "run_id": run_id,
                "parent_pid": os.getpid(), "root": str(root), "output": str(directory),
                "checkpoint_id": base.CHECKPOINT_ID, "source_manifest_sha256": base.sha256(directory / "sources.json"),
                "wrapper_sha256": wrappers, "resources": budget, "wall_seconds": 240,
                "runtime": base.runtime("optimized"), "rounds": MAX_ROUNDS,
                "brain_timeout_seconds": BRAIN_TIMEOUT, "observation_timeout_seconds": OBSERVATION_TIMEOUT,
                "pose_timeout_seconds": POSE_TIMEOUT,
                "sensor": "live_odometry_virtual_marker", "forward_threshold": .03, "pulse_seconds": 1.}
            registration["registration_id"] = base.canonical_hash(registration)
            write_json(directory / "registration.json", registration)
            base.require(budget["can_launch"], "Insufficient system reserve")
            remaining = 240-(time.monotonic()-began)
            base.require(remaining > 0, "Preparation exceeded host wall limit")
            supervision = supervise([sys.executable, "-m", "malecns_lab.experiments.x2_neural", "--_worker", str(directory)],
                directory, rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=base.GIB,
                wall_seconds=remaining, poll_seconds=.1)
            worker = read(directory / "worker_result.json") if (directory / "worker_result.json").exists() else {}
            base.require(supervision["status"] == "completed" and supervision["returncode"] == 0
                         and worker.get("status") == "completed" and worker.get("run_id") == run_id
                         and worker.get("parameters_unchanged") is True and worker.get("completed_frames") == MAX_ROUNDS,
                         f"Supervised worker incomplete: {supervision.get('reason')}; {worker.get('reason')}")
            atomic(directory / "host_ack.json", seal({"kind": "host_ack", "run_id": run_id, "sequence": MAX_ROUNDS,
                "status": "completed", "supervisor_status": supervision["status"], "parameters_unchanged": True,
                "supervisor_sha256": base.sha256(directory / "result.json"), "worker_sha256": base.sha256(directory / "worker_result.json")}))
            while not (directory / "bridge_summary.json").exists() and time.monotonic()-began < 240:
                time.sleep(.05)
            bridge = read(directory / "bridge_summary.json")
            base.require(bridge.get("run_id") == run_id and bridge.get("success") is True
                         and bridge.get("final_stopped") is True and not bridge.get("active_forward"),
                         "ROS bridge did not complete the engineering loop")
            result.update(status="completed", reason="supervised_frozen_brain_and_actual_motion_loop", bridge=bridge,
                          supervisor=supervision, registration_id=registration["registration_id"])
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
        if not (directory / "host_abort.json").exists():
            atomic(directory / "host_abort.json", seal({"kind": "abort", "run_id": run_id, "sequence": 0,
                                                      "reason": result["reason"]}))
    finally:
        signal.signal(signal.SIGTERM, old_term)
        result["wall_seconds"] = time.monotonic()-began
        write_json(directory / "summary.json", result)
    return result


def await_message(directory, name, deadline, *, abort=True):
    path = directory / name
    while time.monotonic() < deadline:
        if abort and (directory / "bridge_summary.json").exists():
            summary = read(directory / "bridge_summary.json")
            base.require(summary.get("success") is True, "Bridge failed: " + str(summary.get("reason")))
        if path.exists():
            return read(path)
        time.sleep(.02)
    raise TimeoutError("Waiting for " + name)


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).resolve()
    reg = read(directory / "registration.json")
    base.bound(reg, "registration_id")
    base.require(reg.get("rounds") == MAX_ROUNDS and reg.get("brain_timeout_seconds") == BRAIN_TIMEOUT
                 and reg.get("observation_timeout_seconds") == OBSERVATION_TIMEOUT
                 and reg.get("pose_timeout_seconds") == POSE_TIMEOUT, "Registered loop budgets differ")
    base.require(os.getppid() == reg["parent_pid"] and str(directory) == reg["output"], "Worker requires its supervisor")
    root, run_id = Path(reg["root"]), reg["run_id"]
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must hold the shared full-brain lock")
    with (directory / "worker.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    result = {"status": "failed", "reason": None, "run_id": run_id, "completed_frames": 0,
              "registration_id": reg["registration_id"], "checkpoint_id": base.CHECKPOINT_ID}
    sampler = None
    timing, boundaries, frames = {}, [], []
    def phase(name, **fields):
        write_json(directory / "phase.json", {"phase": name, "worker_seconds": time.monotonic()-started, **fields})
        if sampler is not None:
            sampler.sample(name)
    try:
        phase("validate_sources")
        base.require(base.sha256(directory / "sources.json") == reg["source_manifest_sha256"], "Sources manifest changed")
        sources = read(directory / "sources.json")
        base.verify_sources(directory / "sources", sources)
        for name, value in reg["wrapper_sha256"].items():
            base.require(base.sha256(root / "src/malecns_lab" / name) == value, "Wrapper changed: " + name)
        anchors = base.read_anchors(root)
        base.require(sources["registered_sha256"] == anchors["run"]["implementation_sha256"], "Historical sources differ")
        base.require(not any(n == "cnsbridge" or n.startswith("cnsbridge.") for n in sys.modules), "Historical namespace already imported")
        snapshot = directory / "sources/src/cnsbridge"
        sys.path.insert(0, str(snapshot.parent))
        sys.dont_write_bytecode = True
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        phase("imports")
        import numpy as np
        import torch
        from cnsbridge.brain.mps_trainable import require_mps
        from cnsbridge.learning.gpu_policy import GPUPolicy
        from cnsbridge.learning.gpu_state import state_arrays, state_metadata
        from cnsbridge.learning.gpu_archive import validate_state
        from cnsbridge.learning.joint_probe_worker import raw_counts_digest
        from .mps_memory import MPSMemorySampler
        from .x2_frozen import FrozenInference
        from ..senses.x2_marker import CAMERA, marker_frame
        base._assert_snapshot_imports(snapshot)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require_mps()
        base.require(base.runtime("optimized") == reg["runtime"], "Runtime changed")
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(directory / "mps_budget.json", sampler.budget)
        def clock():
            torch.mps.synchronize()
            return time.monotonic()
        phase("load_parameters")
        values = base.load_parameters(root / base.CHECKPOINT / "arrays.npz", anchors["checkpoint"], anchors["run"]["policy"])
        before = {key: base._array_declaration(value) for key, value in values.items()}
        base.require(base.sha256(root / base.REFERENCE) == anchors["run"]["graph"]["file_sha256"], "Reference changed")
        phase("construct_full_policy")
        policy = GPUPolicy.from_reference(root / base.REFERENCE, anchors["cpu"]["policy"],
                                         credit_horizon=anchors["run"]["protocol"]["credit_horizon_steps"])
        base.require(policy.identity == anchors["run"]["policy"] and policy.camera.__dict__ == CAMERA, "Full policy differs")
        policy.restore_parameters(values)
        policy.eval()
        executor = FrozenInference(policy, checkpoint_id=base.CHECKPOINT_ID)
        identity = executor.execution_identity
        base.bound(identity, "execution_id")
        base.require(identity["implementation_sha256"] == reg["wrapper_sha256"]["learning/x2_frozen.py"]
                     and identity["checkpoint_id"] == base.CHECKPOINT_ID and identity["source_policy_id"] == policy.policy_id,
                     "Frozen backend identity differs")
        write_json(directory / "execution_identity.json", identity)
        write_json(directory / "actual_policy.json", policy.identity)
        initial_counts = raw_counts_digest(policy.brain.graph)
        base.require(initial_counts == policy.identity["model"]["graph"]["raw_ordered_counts_sha256"], "Full graph counts differ")
        (directory / "frames").mkdir()
        (directory / "states").mkdir()
        def save_state(state):
            phase("save_state", frame_index=state.frame_index)
            policy.validate_state(state)
            arrays, metadata = state_arrays(state, detached=True), state_metadata(state, policy.drive_config.dt_ms)
            state_digest = validate_state(metadata, arrays, policy.identity, state.frame_index)
            path = directory / "states" / f"{state.frame_index:03d}.npz"
            np.savez(path, **arrays)
            record = {"frame_index": state.frame_index, "metadata": metadata, "state_digest": state_digest,
                      "arrays": {k: base._array_declaration(v) for k, v in arrays.items()},
                      "file": str(path.relative_to(directory)), "file_sha256": base.sha256(path)}
            write_json(path.with_suffix(".json"), record)
            boundaries.append(record)
            return record
        with torch.no_grad():
            phase("new_scene_precondition", steps=600)
            began = clock()
            state = executor.initial_state()
            timing["precondition_seconds"] = clock()-began
            save_state(state)
            atomic(directory / "ready.json", seal({"kind": "ready", "run_id": run_id, "sequence": 0,
                   "registration_id": reg["registration_id"], "execution_id": identity["execution_id"],
                   "rounds": MAX_ROUNDS,
                   "precondition_steps": 600, "brain_step": state.brain.step, "drive_step": state.drive.step}))
            previous_target = None
            previous_odom_sequence = previous_mc_sequence = 0
            for sequence in range(1, MAX_ROUNDS+1):
                phase("wait_observation", sequence=sequence)
                # First wait includes the coordinator's bounded VM preparation.
                timeout = 120. if sequence == 1 else OBSERVATION_TIMEOUT
                observation = verify(await_message(directory, f"observation-{sequence:03d}.json", time.monotonic()+timeout),
                                     "observation", run_id, sequence)
                received_local = time.monotonic()
                base.require(observation.get("source") == "live_odometry_virtual_marker"
                             and observation["odom"]["sequence"] > previous_odom_sequence
                             and observation["mc"]["sequence"] > previous_mc_sequence, "Observation provenance/order differs")
                if previous_target is not None:
                    base.require(observation["target_world_m"] == previous_target, "World target moved")
                previous_target = observation["target_world_m"]
                previous_odom_sequence, previous_mc_sequence = observation["odom"]["sequence"], observation["mc"]["sequence"]
                frame = marker_frame(observation["camera_position_m"], observation["camera_orientation_xyzw"], observation["target_world_m"])
                path = directory / "frames" / f"{sequence:03d}.npy"
                np.save(path, frame, allow_pickle=False)
                phase("forward", sequence=sequence)
                began = clock()
                output = executor.forward_tensor(frame, state)
                elapsed = clock()-began
                state = output.state
                action, features, logits = [v.detach().cpu().numpy().copy() for v in (output.action, output.features_mv, output.logits)]
                base.require(action.shape == (3,) and features.shape == (policy.identity["features"]["width"],)
                             and all(v.dtype == np.float32 and np.isfinite(v).all() for v in (action, features, logits)), "Invalid neural output")
                output_path = directory / "frames" / f"{sequence:03d}-output.npz"
                np.savez(output_path, action=action, features_mv=features, logits=logits)
                state_record = save_state(state)
                base.require(time.monotonic()-received_local < BRAIN_TIMEOUT, "Brain inference exceeded 30 seconds")
                response = seal({"kind": "decision", "run_id": run_id, "sequence": sequence,
                    "observation_hash": observation["message_hash"], "input_sha256": base._array_declaration(frame)["sha256"],
                    "input_file_sha256": base.sha256(path), "output_file_sha256": base.sha256(output_path),
                    "raw_action": action.tolist(), "action_axes": ["vx", "vz", "wz"], "unexecuted_axes": ["vz", "wz"],
                    "intent": intent(action.tolist()), "frame_index": state.frame_index,
                    "brain_step": state.brain.step, "drive_step": state.drive.step,
                    "state_sha256": state_record["file_sha256"], "execution_id": identity["execution_id"],
                    "forward_seconds": elapsed, "clock_domain": "host_worker_monotonic",
                    "host_received_monotonic": received_local, "host_completed_monotonic": time.monotonic()})
                write_json(directory / "frames" / f"{sequence:03d}.json", response)
                atomic(directory / f"decision-{sequence:03d}.json", response)
                frames.append(response)
                result["completed_frames"] += 1
                timing.setdefault("frame_forward_seconds", []).append(elapsed)
                phase("wait_stopped_ack", sequence=sequence)
                ack = verify(await_message(directory, f"ack-{sequence:03d}.json", time.monotonic()+BRAIN_TIMEOUT), "ack", run_id, sequence)
                base.require(ack.get("decision_hash") == response["message_hash"] and ack.get("stopped") is True
                             and ack.get("final") == (sequence == MAX_ROUNDS), "Stop acknowledgment differs")
        phase("verify_frozen_parameters")
        after = {key: base._array_declaration(dict(policy.named_parameters())[name].detach().cpu().numpy())
                 for key, name in base.PARAMETERS.items()}
        base.require(after == before and all(p.grad is None for p in policy.parameters()), "Frozen parameters changed")
        final_counts = raw_counts_digest(policy.brain.graph)
        base.require(final_counts == initial_counts, "Graph changed")
        base._assert_snapshot_imports(snapshot)
        write_json(directory / "parameters.json", {"before": before, "after": after, "unchanged": True,
                                                   "optimizer_constructed": False, "autograd_enabled_for_inference": False})
        result.update(status="completed", reason="final_bridge_stopped_ack_then_frozen_state_sealed", parameters_unchanged=True,
                      frames=frames, boundaries=boundaries, raw_counts_sha256_before=initial_counts,
                      raw_counts_sha256_after=final_counts, execution_id=identity["execution_id"])
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        if sampler is not None:
            try:
                result["mps_memory"] = sampler.close()
                if result["mps_memory"]["error"] is not None:
                    result.update(status="failed", reason="mps_memory_sampling_failed")
            except BaseException as error:
                result.update(status="failed", reason="mps_memory_cleanup_failed:"+str(error))
        timing["worker_seconds"] = time.monotonic()-started
        write_json(directory / "timing.json", timing)
        result["timing"] = timing
        write_json(directory / "worker_result.json", result)
    return result
