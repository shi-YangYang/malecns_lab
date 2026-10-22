"""Frozen full-MaleCNS baseline with historical source identity and supervision.

Importing this module does not import Torch, NumPy, ROS or any numerical model.
Historical training assets are read-only; this is not a training continuation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
SOURCE_COMMIT = "732ddc0"
RUN = Path("data/neural_training_gpu_recovery_20261019_r1")
CPU_RUN = Path("data/neural_training_joint_20261019_r1")
CHECKPOINT = RUN / "segments/000042/windows/000160/checkpoint"
REFERENCE = Path("data/malecns_v2/full_reference.npz")
RUN_ID = "601aad8e5ef0fd52034952d4790dc2c812d2063508c746ec45fd7e1dbec41b81"
CHECKPOINT_ID = "30b09f6a7d4a2a879472648fa49b504d918e8a1225fe797532327bb937198f63"
CPU_POLICY_ID = "2e2945c5c9925e7d05698912a034bb34a815ae2e52e295317ad7732d1d910f09"
PARAMETERS = {"theta": "brain.theta", "readout_weight": "readout.weight", "readout_bias": "readout.bias"}
GIB = 1024**3
WALL_SECONDS = 240
WRAPPER_FILES = ("experiments/x2_inference.py", "learning/x2_inference.py", "senses/x2_marker.py",
                 "learning/resource_guard.py", "learning/mps_memory.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound(value, key):
    require(value.get(key) == canonical_hash({k: v for k, v in value.items() if k != key}), f"Invalid {key}")


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=reject)
    require(isinstance(value, dict), "JSON object required")
    return value


def runtime():
    return {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
            "torch": version("torch"), "machine": platform.machine(), "system": platform.system(),
            "num_threads": 4, "num_interop_threads": 1, "device": "mps", "dtype": "float32",
            "fallback": "0", "backend": "historical_GPUPolicy_no_grad_frozen_inference"}


def read_anchors(root=ROOT):
    """Authenticate fixed metadata without loading arrays or following old paths."""
    run = read_json(root / RUN / "run.json")
    cpu = read_json(root / CPU_RUN / "run.json")
    checkpoint = read_json(root / CHECKPOINT / "checkpoint.json")
    window = read_json(root / CHECKPOINT.parent / "window.json")
    for value, field in ((run, "training_run_id"), (cpu, "training_run_id"), (checkpoint, "checkpoint_id"),
                         (window, "window_id"), (run["policy"], "policy_id"), (cpu["policy"], "policy_id")):
        bound(value, field)
    require(run["training_run_id"] == RUN_ID and checkpoint["checkpoint_id"] == CHECKPOINT_ID,
            "Only the preregistered final GPU checkpoint is admitted")
    require(cpu["policy"]["policy_id"] == CPU_POLICY_ID == run["policy"]["cpu_source_policy_id"], "CPU policy differs")
    require(checkpoint["training_run_id"] == RUN_ID and checkpoint["policy_id"] == run["policy"]["policy_id"]
            and checkpoint["schema"] == "cnsbridge.gpu_recovery_checkpoint.v1"
            and checkpoint["context"]["completed_windows"] == checkpoint["context"]["optimizer_steps"] == 160,
            "Checkpoint does not identify the completed registered run")
    for key in ("runtime", "implementation_sha256", "optimizer"):
        require(checkpoint[key] == run[key], f"Checkpoint {key} differs from the run")
    require(window["window"] == 160 and window["checkpoint_id"] == CHECKPOINT_ID
            and window["files_sha256"]["checkpoint/checkpoint.json"] == sha256(root / CHECKPOINT / "checkpoint.json"),
            "Final window checkpoint seal differs")
    graph = run["policy"]["model"]["graph"]
    require(graph["role"] == "full_reference" and graph["nodes"] == 166700 and graph["edges"] == 25582938
            and graph["synapse_count"] == 124177617 and run["policy"]["precondition_steps"] == 600
            and run["policy"]["frame_steps"] == 500 and len(run["implementation_sha256"]) == 54,
            "Complete graph, clocks or source inventory differs")
    require(cpu["graph"]["file_sha256"] == run["graph"]["file_sha256"], "CPU/GPU reference differs")
    return {"run": run, "cpu": cpu, "checkpoint": checkpoint, "window": window}


def export_sources(destination, expected, root=ROOT):
    """Export the entire historical package, including init/dependency files."""
    commit = subprocess.check_output(["git", "rev-parse", f"{SOURCE_COMMIT}^{{commit}}"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", commit, "src/cnsbridge"], cwd=root)
    destination.mkdir(exist_ok=False)
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            relative = Path(member.name)
            if member.isdir() and relative == Path("src"):
                continue
            require(relative.parts[:2] == ("src", "cnsbridge") and ".." not in relative.parts
                    and not relative.is_absolute(), "Invalid snapshot path")
            if member.isdir():
                continue
            require(member.isfile(), "Only regular historical files are allowed")
            content = stream.extractfile(member).read()
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(content)
            path.chmod(0o444)
            manifest[str(relative)] = hashlib.sha256(content).hexdigest()
    require("src/cnsbridge/__init__.py" in manifest, "Snapshot requires a regular package root")
    for name, digest in expected.items():
        require(manifest.get(f"src/cnsbridge/{name}") == digest, f"Historical source mismatch: {name}")
    for directory in sorted((p for p in destination.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.chmod(0o555)
    destination.chmod(0o555)
    return {"git_commit": commit, "registered_source_count": len(expected), "files_sha256": manifest,
            "registered_sha256": expected, "package": "cnsbridge", "read_only": True}


def verify_sources(directory, manifest):
    for name, digest in manifest["files_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe snapshot manifest path")
        require(sha256(directory / relative) == digest, f"Snapshot changed: {name}")


def simulator_preflight(root=ROOT):
    """Read only: never start/stop a service or change the Docker context."""
    runtime_path = root / "data/x2_sim/gpu_probe/runtime-rebuild"
    environment = dict(os.environ, LIGHTER_HOME=str(runtime_path))
    status = subprocess.run([str(runtime_path / "bin/lighter"), "status"], env=environment,
                            capture_output=True, text=True, timeout=8)
    output = status.stdout + status.stderr
    socket_exists = (runtime_path / "docker.sock").exists()
    require("lighter is not running." in output and not socket_exists, "GPU simulation VM must be stopped")
    cpu = subprocess.run(["docker", "--context", "desktop-linux", "inspect", "--format", "{{.State.Status}}",
                          "malecns-x2-x2-1"], capture_output=True, text=True, timeout=8)
    require(cpu.returncode == 0 and cpu.stdout.strip() == "exited", "CPU simulator must be stopped")
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "gpu_status_output": output.strip(),
            "gpu_status_returncode": status.returncode, "gpu_socket_exists": socket_exists,
            "cpu_container": "malecns-x2-x2-1", "cpu_context": "desktop-linux", "cpu_status": cpu.stdout.strip()}


@contextmanager
def numerical_lock(root=ROOT):
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def execute(output, root=ROOT):
    """Prepare a new evidence directory and run exactly one supervised worker."""
    from .resource_guard import resource_budget, supervise, write_json
    import psutil
    root, output = Path(root).resolve(), Path(output).resolve()
    require(any(output.is_relative_to(root / name) and output != root / name for name in ("data", "logs")),
            "Output must be a new directory beneath this project's data/ or logs/")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "malecns_lab.x2_frozen_inference.v1", "status": "failed", "reason": None, "output": str(output)}
    try:
        with numerical_lock(root):
            write_json(output / "phase.json", {"phase": "preflight"})
            preflight = simulator_preflight(root)
            write_json(output / "simulator_preflight.json", preflight)
            write_json(output / "phase.json", {"phase": "validate_metadata_and_export_sources"})
            anchors = read_anchors(root)
            sources = export_sources(output / "sources", anchors["run"]["implementation_sha256"], root)
            write_json(output / "sources.json", sources)
            provenance = output / "provenance"
            provenance.mkdir()
            for key, value in anchors.items():
                write_json(provenance / f"{key}.json", value)
            wrapper = {name: sha256(root / "src/malecns_lab" / name) for name in WRAPPER_FILES}
            for name in WRAPPER_FILES:
                path = output / "wrapper_sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write((root / "src/malecns_lab" / name).read_bytes())
                path.chmod(0o444)
            resources = resource_budget(psutil.virtual_memory().available, rss_bytes=6*GIB, reserve_bytes=GIB)
            current = runtime()
            historical = anchors["run"]["runtime"]
            registration = {"schema": result["schema"], "created_utc": datetime.now(timezone.utc).isoformat(),
                            "parent_pid": os.getpid(), "root": str(root), "output": str(output), "resources": resources,
                            "wall_limit_seconds": WALL_SECONDS, "parent_started_monotonic": started,
                            "checkpoint_id": CHECKPOINT_ID, "checkpoint_path": str(root / CHECKPOINT),
                            "graph_path": str(root / REFERENCE), "historical_graph_path": anchors["run"]["graph"]["path"],
                            "source_manifest_sha256": sha256(output / "sources.json"), "wrapper_sha256": wrapper,
                            "runtime": current, "historical_runtime": historical,
                            "runtime_differences": {key: {"historical": historical.get(key), "current": current.get(key)}
                                                    for key in historical.keys() | current.keys() if historical.get(key) != current.get(key)},
                            "mode": "new_scene_no_grad; no_optimizer; no_historical_neural_state",
                            "inputs_m": [2.0, 1.5, 1.0], "state_boundaries": [0, 1, 2, 3]}
            registration["registration_id"] = canonical_hash(registration)
            write_json(output / "registration.json", registration)
            require(resources["can_launch"], "Insufficient system reserve before launch")
            remaining = WALL_SECONDS - (time.monotonic() - started)
            require(remaining > 0, "Preparation exceeded wall limit")
            supervision = supervise([sys.executable, "-m", "malecns_lab.experiments.x2_inference", "--_worker", str(output)],
                                    output, rss_limit_bytes=resources["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                    wall_seconds=remaining, poll_seconds=.1)
            result["supervisor"] = supervision
            worker = read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
            require(supervision["status"] == "completed" and worker.get("status") == "completed"
                    and worker.get("completed_frames") == 3 and worker.get("saved_state_boundaries") == 4,
                    f"Incomplete baseline: {supervision['reason']}; {worker.get('reason', 'no worker result')}")
            result.update(status="completed", reason="three_frozen_input_probes_sealed", registration_id=registration["registration_id"])
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", result)
    return result


def load_parameters(path, metadata, identity):
    """Verify the entire NPZ hash, materialize only the three parameter arrays."""
    import numpy as np
    require(sha256(path) == metadata["arrays_sha256"], "Checkpoint arrays checksum differs")
    shapes = {"theta": (len(identity["model"]["graph"]["grouping"]["names"]),),
              "readout_weight": (3, identity["features"]["width"]), "readout_bias": (3,)}
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)) and set(data.files) == set(metadata["arrays"]),
                "Checkpoint array inventory differs")
        result = {}
        for key, shape in shapes.items():
            name = f"parameter.{key}"
            value = data[name].copy()
            require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all()
                    and metadata["arrays"][name] == {"shape": list(shape), "dtype": value.dtype.str},
                    f"Invalid inherited parameter: {key}")
            result[key] = value
    return result


def _array_declaration(value):
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _assert_snapshot_imports(snapshot):
    for name, module in tuple(sys.modules.items()):
        if name == "cnsbridge" or name.startswith("cnsbridge."):
            path = getattr(module, "__file__", None)
            require(path is not None and Path(path).resolve().is_relative_to(snapshot),
                    f"Historical namespace escaped snapshot: {name}")


def run_worker(directory):
    """Child-only work; parent, lock and exclusive claim precede heavy imports."""
    from .resource_guard import write_json
    directory = Path(directory).resolve()
    reg = read_json(directory / "registration.json")
    bound(reg, "registration_id")
    require(os.getppid() == reg["parent_pid"] and str(directory) == reg["output"], "Worker requires its supervising parent")
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
    result = {"status": "failed", "reason": None, "completed_frames": 0, "saved_state_boundaries": 0,
              "registration_id": reg["registration_id"], "checkpoint_id": CHECKPOINT_ID}
    timing, frames, boundaries = {}, [], []
    sampler = None
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started, **details})
        if sampler is not None:
            sampler.sample(name)
    try:
        phase("validate_frozen_sources")
        require(sha256(directory / "sources.json") == reg["source_manifest_sha256"], "Source manifest changed")
        sources = read_json(directory / "sources.json")
        verify_sources(directory / "sources", sources)
        for name, digest in reg["wrapper_sha256"].items():
            require(sha256(root / "src/malecns_lab" / name) == digest, f"Inference wrapper changed: {name}")
        anchors = read_anchors(root)
        require(sources["registered_sha256"] == anchors["run"]["implementation_sha256"], "Snapshot registration differs")
        require(not any(name == "cnsbridge" or name.startswith("cnsbridge.") for name in sys.modules),
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
        from ..senses.x2_marker import CAMERA, marker_frame, probe_inputs
        _assert_snapshot_imports(snapshot)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require_mps()
        require(runtime() == reg["runtime"], "Worker runtime differs from registration")
        timing["imports_seconds"] = time.monotonic() - began
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(directory / "mps_budget.json", sampler.budget)
        def clock():
            torch.mps.synchronize()
            return time.monotonic()
        phase("load_frozen_parameters")
        began = clock()
        values = load_parameters(root / CHECKPOINT / "arrays.npz", anchors["checkpoint"], anchors["run"]["policy"])
        before = {key: _array_declaration(value) for key, value in values.items()}
        require(sha256(root / REFERENCE) == anchors["run"]["graph"]["file_sha256"], "Reference file changed")
        timing["parameter_and_reference_verification_seconds"] = clock() - began
        phase("construct_full_policy")
        began = clock()
        policy = GPUPolicy.from_reference(root / REFERENCE, anchors["cpu"]["policy"],
                                          credit_horizon=anchors["run"]["protocol"]["credit_horizon_steps"])
        require(policy.identity == anchors["run"]["policy"] and policy.camera.__dict__ == CAMERA,
                "Actual full GPU policy identity or camera differs")
        policy.restore_parameters(values)
        policy.eval()
        # Historical validation requires requires_grad=True. no_grad and absence
        # of an optimizer freeze computation without changing historical source.
        timing["construction_and_restore_seconds"] = clock() - began
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == policy.identity["model"]["graph"]["raw_ordered_counts_sha256"],
                "Actual complete graph differs")
        write_json(directory / "actual_policy.json", policy.identity)
        (directory / "states").mkdir()
        (directory / "frames").mkdir()
        def save_state(state):
            phase("save_state", frame_index=state.frame_index)
            began = clock()
            policy.validate_state(state)
            arrays = state_arrays(state, detached=True)
            metadata = state_metadata(state, policy.drive_config.dt_ms)
            digest = validate_state(metadata, arrays, policy.identity, state.frame_index)
            path = directory / "states" / f"{state.frame_index:03d}.npz"
            np.savez(path, **arrays)
            record = {"frame_index": state.frame_index, "metadata": metadata, "state_digest": digest,
                      "arrays": {key: _array_declaration(value) for key, value in arrays.items()},
                      "file": str(path.relative_to(directory)), "file_sha256": sha256(path)}
            write_json(path.with_suffix(".json"), record)
            boundaries.append(record)
            result["saved_state_boundaries"] += 1
            timing.setdefault("state_save_seconds", []).append(clock()-began)
        with torch.no_grad():
            phase("new_scene_blank_precondition", neural_steps=600)
            began = clock()
            state = policy.initial_state()
            timing["precondition_seconds"] = clock() - began
            save_state(state)
            for index, sample in enumerate(probe_inputs()):
                phase("render_input", index=index)
                frame = marker_frame(sample["camera_position_m"], sample["camera_orientation_xyzw"], sample["target_world_m"])
                path = directory / "frames" / f"{index:03d}.npy"
                np.save(path, frame, allow_pickle=False)
                phase("forward", index=index, sample_id=sample["sample_id"], neural_steps=500)
                began = clock()
                output = policy.forward_tensor(frame, state)
                elapsed = clock() - began
                state = output.state
                action = output.action.detach().cpu().numpy().copy()
                features = output.features_mv.detach().cpu().numpy().copy()
                logits = output.logits.detach().cpu().numpy().copy()
                require(action.shape == (3,) and features.shape == (policy.identity["features"]["width"],)
                        and all(value.dtype == np.float32 and np.isfinite(value).all() for value in (action, features, logits)),
                        "Invalid neural output")
                feature_path = directory / "frames" / f"{index:03d}-output.npz"
                np.savez(feature_path, action=action, features_mv=features, logits=logits)
                record = {**sample, "input": _array_declaration(frame), "input_file_sha256": sha256(path),
                          "input_file": str(path.relative_to(directory)), "output_file": str(feature_path.relative_to(directory)),
                          "output_file_sha256": sha256(feature_path), "action_axes": ["vx", "vz", "wz"],
                          "action_units": ["m/s", "m/s", "rad/s"], "raw_action": action.tolist(),
                          "forward_seconds": elapsed, "neural_frame_ms": policy.config.frame_ms,
                          "brain_step": state.brain.step, "drive_step": state.drive.step, "frame_index": state.frame_index}
                write_json(directory / "frames" / f"{index:03d}.json", record)
                frames.append(record)
                timing.setdefault("frame_forward_seconds", []).append(elapsed)
                result["completed_frames"] += 1
                save_state(state)
        phase("verify_frozen_parameters_and_graph")
        after = {key: _array_declaration(dict(policy.named_parameters())[name].detach().cpu().numpy())
                 for key, name in PARAMETERS.items()}
        require(after == before and all(value.grad is None for value in policy.parameters()), "Frozen parameters changed or gradients exist")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"], "Graph changed during inference")
        _assert_snapshot_imports(snapshot)
        write_json(directory / "parameters.json", {"before": before, "after": after, "unchanged": True,
                                                   "autograd_enabled_for_inference": False, "optimizer_constructed": False})
        result.update(status="completed", reason="frozen_full_reference_three_probes", frames=frames,
                      boundaries=boundaries, policy_id=policy.policy_id, parameters_unchanged=True)
        phase("sealed")
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
                result.update(status="failed", reason=f"mps_memory_cleanup_failed:{type(error).__name__}:{error}")
        timing["worker_seconds"] = time.monotonic() - started
        write_json(directory / "timing.json", timing)
        result["timing"] = timing
        write_json(directory / "worker_result.json", result)
    return result
