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
import math
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


def runtime(backend="baseline"):
    return {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
            "torch": version("torch"), "machine": platform.machine(), "system": platform.system(),
            "num_threads": 4, "num_interop_threads": 1, "device": "mps", "dtype": "float32",
            "fallback": "0", "backend": ("historical_GPUPolicy_no_grad_frozen_inference" if backend == "baseline"
                                           else "historical_GPUPolicy_optimized_sticky50_comparison")}


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


def baseline_evidence(directory, anchors):
    """Authenticate completed evidence without loading a model or accepting new sources as old."""
    directory = Path(directory).resolve()
    reg, summary, supervisor, worker, timings, policy = [read_json(directory / name) for name in
        ("registration.json", "summary.json", "result.json", "worker_result.json", "timing.json", "actual_policy.json")]
    bound(reg, "registration_id")
    bound(policy, "policy_id")
    require(reg.get("backend", "baseline") == "baseline" and reg["checkpoint_id"] == CHECKPOINT_ID
            and summary["status"] == worker["status"] == supervisor["status"] == "completed"
            and supervisor["returncode"] == 0 and supervisor["worker_status"] == "completed"
            and summary["registration_id"] == worker["registration_id"] == reg["registration_id"]
            and summary["supervisor"] == supervisor, "Baseline is not a completed supervised baseline")
    require(worker["checkpoint_id"] == CHECKPOINT_ID and worker["policy_id"] == policy["policy_id"]
            and policy == anchors["run"]["policy"] and worker["completed_frames"] == 3
            and worker["saved_state_boundaries"] == 4 and reg["inputs_m"] == [2., 1.5, 1.]
            and reg["state_boundaries"] == [0, 1, 2, 3], "Baseline policy, checkpoint or probe inventory differs")
    require(worker["parameters_unchanged"] is True and worker["raw_counts_sha256_before"]
            == worker["raw_counts_sha256_after"] == policy["model"]["graph"]["raw_ordered_counts_sha256"],
            "Baseline complete graph or frozen parameters differ")
    parameters = read_json(directory / "parameters.json")
    require(parameters["unchanged"] is True and parameters["before"] == parameters["after"], "Baseline parameters changed")
    sources = read_json(directory / "sources.json")
    require(sha256(directory / "sources.json") == reg["source_manifest_sha256"]
            and sources["registered_sha256"] == anchors["run"]["implementation_sha256"], "Baseline old sources differ")
    verify_sources(directory / "sources", sources)
    for name, digest in reg["wrapper_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and sha256(directory / "wrapper_sources" / name) == digest, "Baseline wrapper archive changed")
    require(timings == worker["timing"] and len(timings["frame_forward_seconds"]) == 3, "Baseline timings differ")
    durations = [timings["precondition_seconds"], *timings["frame_forward_seconds"]]
    require(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in durations),
            "Invalid baseline compute durations")
    files = {name: sha256(directory / name) for name in ("registration.json", "summary.json", "result.json", "worker_result.json",
             "timing.json", "actual_policy.json", "parameters.json", "sources.json")}
    for index in range(4):
        name = f"states/{index:03d}.json"
        record = read_json(directory / name)
        require(record == worker["boundaries"][index] and record["frame_index"] == index
                and record["file"] == f"states/{index:03d}.npz", "Baseline state boundary differs")
        require(sha256(directory / record["file"]) == record["file_sha256"], "Baseline state file hash differs")
        files[name], files[record["file"]] = sha256(directory / name), record["file_sha256"]
    for index, distance in enumerate((2., 1.5, 1.)):
        name = f"frames/{index:03d}.json"
        record = read_json(directory / name)
        require(record == worker["frames"][index] and record["distance_m"] == distance
                and record["frame_index"] == index+1 and record["source"] == "offline_static_input_probe"
                and record["input_file"] == f"frames/{index:03d}.npy"
                and record["output_file"] == f"frames/{index:03d}-output.npz", "Baseline frame sequence differs")
        require(sha256(directory / record["input_file"]) == record["input_file_sha256"]
                and sha256(directory / record["output_file"]) == record["output_file_sha256"], "Baseline frame file hash differs")
        files[name] = sha256(directory / name)
        files[record["input_file"]], files[record["output_file"]] = record["input_file_sha256"], record["output_file_sha256"]
    return {"path": str(directory), "registration_id": reg["registration_id"], "policy_id": policy["policy_id"],
            "checkpoint_id": CHECKPOINT_ID, "files_sha256": files, "compute_seconds": sum(durations),
            "phase_compute_seconds": durations}


def compare_array(actual, expected):
    """Exact admission plus descriptive error metrics; no tolerance fallback."""
    import numpy as np
    same_shape, same_dtype = actual.shape == expected.shape, actual.dtype == expected.dtype
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    result = {"actual_shape": list(actual.shape), "baseline_shape": list(expected.shape),
              "actual_dtype": actual.dtype.str, "baseline_dtype": expected.dtype.str,
              "exact_equal": bool(same_shape and same_dtype and finite and np.array_equal(actual, expected)),
              "finite": finite, "different_elements": None, "max_absolute_error": None, "rms_error": None}
    if same_shape:
        result["different_elements"] = int(np.count_nonzero(actual != expected))
        if finite:
            # Float64 errors are reporting only; exact comparison uses original dtypes.
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            result["max_absolute_error"] = float(np.max(np.abs(delta), initial=0.))
            result["rms_error"] = float(np.sqrt(np.mean(delta*delta))) if delta.size else 0.
    return result


def compare_evidence(directory, baseline, timing):
    """Compare every saved recurrent boundary and output, retaining all differences."""
    import numpy as np
    directory, prior = Path(directory), Path(baseline["path"])
    for name, digest in baseline["files_sha256"].items():
        require(sha256(prior / name) == digest, f"Baseline evidence changed during execution: {name}")
    records, inputs = [], []
    for index in range(3):
        actual_path, prior_path = directory / f"frames/{index:03d}.npy", prior / f"frames/{index:03d}.npy"
        inputs.append({"frame_index": index+1, **compare_array(np.load(actual_path, allow_pickle=False),
                                                              np.load(prior_path, allow_pickle=False)),
                       "actual_file_sha256": sha256(actual_path), "baseline_file_sha256": sha256(prior_path)})
    for index in range(4):
        actual_meta = read_json(directory / f"states/{index:03d}.json")
        prior_meta = read_json(prior / f"states/{index:03d}.json")
        with np.load(directory / f"states/{index:03d}.npz", allow_pickle=False) as actual, \
                np.load(prior / f"states/{index:03d}.npz", allow_pickle=False) as expected:
            names = sorted(set(actual.files) | set(expected.files))
            arrays = {name: compare_array(actual[name], expected[name]) if name in actual and name in expected
                      else {"exact_equal": False, "reason": "array_inventory_differs"} for name in names}
        records.append({"kind": "state", "frame_index": index, "metadata_equal": actual_meta["metadata"] == prior_meta["metadata"],
                        "arrays": arrays, "exact_equal": actual_meta["metadata"] == prior_meta["metadata"]
                        and all(value["exact_equal"] for value in arrays.values())})
        if index > 0:
            frame = index - 1
            with np.load(directory / f"frames/{frame:03d}-output.npz", allow_pickle=False) as actual, \
                    np.load(prior / f"frames/{frame:03d}-output.npz", allow_pickle=False) as expected:
                names = sorted(set(actual.files) | set(expected.files))
                arrays = {name: compare_array(actual[name], expected[name]) if name in actual and name in expected
                          else {"exact_equal": False, "reason": "array_inventory_differs"} for name in names}
            records.append({"kind": "output", "frame_index": index, "arrays": arrays,
                            "exact_equal": all(value["exact_equal"] for value in arrays.values())})
    compute = timing["precondition_seconds"] + sum(timing["frame_forward_seconds"])
    require(math.isfinite(compute) and compute > 0, "Invalid optimized compute duration")
    reduction = 1 - compute / baseline["compute_seconds"]
    exact = all(record["exact_equal"] for record in [*inputs, *records])
    performance = compute <= .8 * baseline["compute_seconds"]
    return {"schema": "malecns_lab.x2_inference_comparison.v1", "baseline_registration_id": baseline["registration_id"],
            "baseline_path": baseline["path"], "exact_equal": exact, "inputs": inputs, "comparisons": records,
            "first_difference": next(({"kind": record["kind"], "frame_index": record["frame_index"],
                                       "arrays": [key for key, value in record["arrays"].items() if not value["exact_equal"]],
                                       "metadata_equal": record.get("metadata_equal")} for record in records if not record["exact_equal"]), None),
            "baseline_compute_seconds": baseline["compute_seconds"], "optimized_compute_seconds": compute,
            "baseline_phase_compute_seconds": baseline["phase_compute_seconds"],
            "optimized_phase_compute_seconds": [timing["precondition_seconds"], *timing["frame_forward_seconds"]],
            "compute_reduction_fraction": reduction, "required_reduction_fraction": .2, "performance_pass": performance,
            "status": "PASS" if exact and performance else "FAIL",
            "interpretation": "One synchronized run against the saved baseline; not a statistical speed or real-time guarantee"}


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


def execute(output, root=ROOT, *, backend="baseline", baseline=None):
    """Prepare a new evidence directory and run exactly one supervised worker."""
    from .resource_guard import resource_budget, supervise, write_json
    import psutil
    root, output = Path(root).resolve(), Path(output).resolve()
    require(backend in ("baseline", "optimized") and (baseline is not None) == (backend == "optimized"),
            "Optimized execution requires a baseline; baseline execution cannot take one")
    baseline_path = Path(baseline).resolve() if baseline is not None else None
    require(baseline_path is None or not output.is_relative_to(baseline_path), "Output must not modify the baseline directory")
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
            baseline_binding = baseline_evidence(baseline_path, anchors) if baseline_path is not None else None
            if baseline_binding is not None:
                write_json(output / "baseline_binding.json", baseline_binding)
            sources = export_sources(output / "sources", anchors["run"]["implementation_sha256"], root)
            write_json(output / "sources.json", sources)
            provenance = output / "provenance"
            provenance.mkdir()
            for key, value in anchors.items():
                write_json(provenance / f"{key}.json", value)
            wrapper_files = WRAPPER_FILES + (("learning/x2_frozen.py",) if backend == "optimized" else ())
            wrapper = {name: sha256(root / "src/malecns_lab" / name) for name in wrapper_files}
            for name in wrapper_files:
                path = output / "wrapper_sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write((root / "src/malecns_lab" / name).read_bytes())
                path.chmod(0o444)
            resources = resource_budget(psutil.virtual_memory().available, rss_bytes=6*GIB, reserve_bytes=GIB)
            current = runtime(backend)
            historical = anchors["run"]["runtime"]
            execution = {"schema": "malecns_lab.x2_inference_execution.v1", "backend": backend,
                         "source_policy_id": anchors["run"]["policy"]["policy_id"], "checkpoint_id": CHECKPOINT_ID,
                         "wrapper_sha256": wrapper, "runtime": current,
                         "baseline_registration_id": baseline_binding["registration_id"] if baseline_binding else None}
            execution["execution_id"] = canonical_hash(execution)
            registration = {"schema": result["schema"], "created_utc": datetime.now(timezone.utc).isoformat(),
                            "backend": backend, "execution_identity": execution, "baseline_binding": baseline_binding,
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
            if backend == "optimized":
                require(worker.get("comparison_status") == "PASS", "Optimized comparison did not pass")
            result.update(status="completed", reason=("optimized_exact_comparison_passed" if backend == "optimized"
                                                       else "three_frozen_input_probes_sealed"),
                          registration_id=registration["registration_id"], execution_id=execution["execution_id"])
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
    backend = reg.get("backend", "baseline")
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
        execution = reg["execution_identity"]
        bound(execution, "execution_id")
        require(execution["backend"] == backend and execution["source_policy_id"] == anchors["run"]["policy"]["policy_id"]
                and execution["checkpoint_id"] == CHECKPOINT_ID and execution["wrapper_sha256"] == reg["wrapper_sha256"]
                and execution["runtime"] == reg["runtime"], "Execution identity differs")
        baseline = reg["baseline_binding"]
        require(backend in ("baseline", "optimized") and (baseline is not None) == (backend == "optimized"),
                "Execution backend/baseline differs")
        if baseline is not None:
            require(baseline_evidence(baseline["path"], anchors) == baseline, "Registered baseline evidence changed")
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
        require(runtime(backend) == reg["runtime"], "Worker runtime differs from registration")
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
        phase("configure_execution_backend", backend=backend)
        began = clock()
        executor = policy
        if backend == "optimized":
            from .x2_frozen import FrozenInference
            executor = FrozenInference(policy, checkpoint_id=CHECKPOINT_ID)
            backend_identity = executor.execution_identity
            bound(backend_identity, "execution_id")
            require(backend_identity["checkpoint_id"] == CHECKPOINT_ID
                    and backend_identity["source_policy_id"] == policy.policy_id
                    and backend_identity["implementation_sha256"] == reg["wrapper_sha256"]["learning/x2_frozen.py"],
                    "Actual optimized execution identity differs")
        else:
            backend_identity = {"schema": "malecns_lab.x2_original_execution.v1", "backend": backend,
                                "source_policy_id": policy.policy_id, "checkpoint_id": CHECKPOINT_ID,
                                "source_files_sha256": sources["registered_sha256"], "mode": "torch.no_grad"}
            backend_identity["execution_id"] = canonical_hash(backend_identity)
        write_json(directory / "execution_identity.json", {"registration_execution": execution, "backend_execution": backend_identity})
        result.update(execution_id=execution["execution_id"], backend_execution_id=backend_identity["execution_id"],
                      execution_identity_sha256=sha256(directory / "execution_identity.json"))
        timing["execution_setup_seconds"] = clock() - began
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
            state = executor.initial_state()
            timing["precondition_seconds"] = clock() - began
            save_state(state)
            for index, sample in enumerate(probe_inputs()):
                phase("render_input", index=index)
                frame = marker_frame(sample["camera_position_m"], sample["camera_orientation_xyzw"], sample["target_world_m"])
                if baseline is not None:
                    prior = Path(baseline["path"])
                    original = np.load(prior / f"frames/{index:03d}.npy", allow_pickle=False)
                    original_info = read_json(prior / f"frames/{index:03d}.json")
                    require(np.array_equal(frame, original) and original.dtype == np.uint8
                            and _array_declaration(original) == original_info["input"]
                            and all(original_info[key] == value for key, value in sample.items()),
                            "Baseline input differs from the preregistered pixel probe")
                    frame = original
                path = directory / "frames" / f"{index:03d}.npy"
                np.save(path, frame, allow_pickle=False)
                phase("forward", index=index, sample_id=sample["sample_id"], neural_steps=500)
                began = clock()
                output = executor.forward_tensor(frame, state)
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
        result.update(frames=frames, boundaries=boundaries, policy_id=policy.policy_id, parameters_unchanged=True)
        if baseline is not None:
            phase("compare_saved_baseline")
            began = clock()
            comparison = compare_evidence(directory, baseline, timing)
            write_json(directory / "comparison.json", comparison)
            result.update(comparison_status=comparison["status"], comparison_sha256=sha256(directory / "comparison.json"))
            timing["comparison_seconds"] = clock() - began
            require(comparison["status"] == "PASS", "Optimized exactness or 20 percent performance gate failed; see comparison.json")
        result.update(status="completed", reason="frozen_full_reference_three_probes")
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
