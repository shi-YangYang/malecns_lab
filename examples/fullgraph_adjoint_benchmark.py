"""Register and supervise one complete MaleCNS adjoint CPU kernel/thread benchmark."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import psutil

from fullgraph_matvec_benchmark import canonical_hash, file_sha256
from cnsbridge.learning.resource_guard import GIB, resource_budget, supervise, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cnsbridge.adjoint_benchmark_registration.v1"
PROTOCOL = {
    "schema": "cnsbridge.fullgraph_adjoint_benchmark.v1", "role": "kernel_cost_and_numerics_only",
    "device": "cpu", "dtype": "float32", "weight_mv": 0.275, "model_input_raw_type": "R1-R6",
    "graph_file_sha256": "98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3",
    "graph_sha256": "af12b8dc8775f828740221f92c870d76450266759bca6f1540ff5e27002ac156",
    "raw_ordered_counts_sha256": "4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565",
    "graph_counts": {"nodes": 166700, "edges": 25582938, "synapse_count": 124177617},
    "matrix": "TrainableLIF.base_weights_adjoint: CSR(pre, post), W transpose",
    "torch_threads": [1, 2, 4, 8, 10], "torch_interop_threads": 1, "restore_threads": 4,
    "input_order": ["zero", "sparse", "dense"], "seed": 1901, "rng": "numpy.default_rng",
    "sparse_count": "floor(N / 100)", "sparse_value": 1.0,
    "dense": "same generator after choice: uniform(-1, 1, N).astype(float32)",
    "input_meaning": "synthetic numerical probes, not recorded upstream gradients or biological stimuli",
    "warmups_per_method_per_input": 1, "warmup_order": ["torch", "scipy"], "repetitions": 5,
    "order": "even round: torch then scipy; odd round: scipy then torch", "total_matvec_calls": 180,
    "torch_path": "torch.sparse.mm(adjoint, x[:, None])[:, 0]",
    "scipy_path": "torch.from_numpy(shared_adjoint @ x.detach().numpy())",
    "clocks": ["time.perf_counter", "time.process_time"],
    "timing_excludes": ["thread setting", "shared CSR construction", "input construction", "output copies", "artifact writes"],
    "rtol": 1e-5, "atol": 1e-5,
    "scipy_candidate": "all numeric checks pass and all three scipy medians at 4 threads < torch4 medians",
    "thread_candidate": "all numeric checks pass and all three torch medians < torch4 medians",
    "resource_policy": {"requested_rss_bytes": 6 * GIB, "reserve_bytes": GIB, "wall_seconds": 600, "poll_seconds": .1},
}


@contextmanager
def workspace_lock():
    # Equivalent small OS-lock boundary avoids importing generation's NumPy tree.
    path = ROOT / "data/.continuous-training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another training/generation parent owns the workspace lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_paths():
    """Conservative static local-import closure, including lazy imports and package initializers."""
    pending, found = [Path(__file__).resolve()], set()

    def resolve(module):
        base = ROOT / "src" / module.replace(".", "/") if module.startswith("cnsbridge") else ROOT / "examples" / module
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                pending.append(path)
                if module.startswith("cnsbridge"):
                    for directory in path.parents:
                        if directory == ROOT / "src":
                            break
                        if (directory / "__init__.py").is_file():
                            pending.append(directory / "__init__.py")
                break

    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        relative = path.relative_to(ROOT)
        package = list(relative.parts[1:-1]) if relative.parts[0] == "src" else []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *([node.module] if node.module else [])])
                resolve(module)
                for alias in node.names:
                    resolve(module + "." + alias.name)
    return sorted(found)


def implementation_hashes():
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths()}


def runtime():
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
            "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "psutil")},
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}


def _register(graph, directory):
    """Write registration in the empty directory already exclusively claimed by the parent."""
    graph, directory = graph.resolve(strict=True), directory.resolve()
    if any(directory.iterdir()):
        raise ValueError("Registration requires an empty parent-claimed directory")
    hashes = implementation_hashes()
    for relative, expected in hashes.items():
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise ValueError(f"Source changed during registration: {relative}")
    value = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(), "parent_pid": os.getpid(),
             "output_directory": str(directory), "graph_path": str(graph), "graph_file_sha256": file_sha256(graph),
             "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "implementation_sha256": hashes,
             "environment": runtime(),
             "resources": {**resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB),
                           "wall_limit_seconds": 600, "sampling_seconds": .1}}
    value["registration_id"] = canonical_hash(value)
    write_json(directory / "registration.json", value)
    if value["graph_file_sha256"] != PROTOCOL["graph_file_sha256"]:
        raise ValueError("This benchmark requires the registered complete MaleCNS asset")
    return value


def validate_registration(directory):
    value = json.loads((directory / "registration.json").read_text())
    expected = canonical_hash({key: item for key, item in value.items() if key != "registration_id"})
    if value["schema"] != SCHEMA or value["registration_id"] != expected:
        raise ValueError("Invalid registration identity")
    if value["protocol"] != PROTOCOL or value["protocol_id"] != canonical_hash(PROTOCOL):
        raise ValueError("Registered protocol differs")
    if value["output_directory"] != str(directory.resolve()) or value["environment"] != runtime():
        raise ValueError("Registered directory/runtime differs")
    hashes = implementation_hashes()
    if value["implementation_sha256"] != hashes:
        raise ValueError("Registered source differs")
    for relative, expected in hashes.items():
        if file_sha256(directory / "source" / relative) != expected:
            raise ValueError(f"Source snapshot differs: {relative}")
    budget = value["resources"]
    expected = {**resource_budget(budget["available_bytes_before_launch"], rss_bytes=6 * GIB, reserve_bytes=GIB),
                "wall_limit_seconds": 600, "sampling_seconds": .1}
    if budget != expected or not budget["can_launch"]:
        raise ValueError("Invalid/nonlaunchable registered budget")
    graph = Path(value["graph_path"])
    if (not graph.is_absolute() or value["graph_file_sha256"] != PROTOCOL["graph_file_sha256"]
            or file_sha256(graph) != value["graph_file_sha256"]):
        raise ValueError("Registered complete graph changed")
    return value


def observed_source_files():
    observed = {}
    for module in list(sys.modules.values()):
        location = getattr(module, "__file__", None)
        if location:
            path = Path(location).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and path.parts[len(ROOT.parts)] in ("src", "examples"):
                observed[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(observed.items()))


def run_worker(directory):
    directory = directory.resolve()
    preliminary = json.loads((directory / "registration.json").read_text())
    if os.getppid() != preliminary.get("parent_pid"):
        raise ValueError("Worker must be a child of its registered supervising parent")
    # The supervising parent owns the same nonblocking lock throughout this job.
    with (ROOT / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Supervising parent does not hold the workspace lock")
    with (directory / "worker.claim").open("x") as claim:
        claim.write(datetime.now(timezone.utc).isoformat() + "\n")
    started, phases, current = time.perf_counter(), {}, None
    phase_started = started
    result = {"schema": "cnsbridge.adjoint_benchmark_worker.v1", "status": "failed", "reason": None}
    torch = None

    def phase(name, **details):
        nonlocal current, phase_started
        now = time.perf_counter()
        if current is not None:
            phases[current] = phases.get(current, 0.0) + now - phase_started
        current, phase_started = name, now
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": now - started, **details})

    try:
        phase("validate_registration")
        registration = validate_registration(directory)
        identity = {key: registration[key] for key in ("registration_id", "protocol_id")}
        result.update(identity)
        phase("imports")
        import numpy as np
        import scipy
        import torch
        from cnsbridge.brain.trainable import TrainableLIF, TrainableLIFConfig
        from cnsbridge.brain.trainable_graph import load_training_graph
        from adjoint_benchmark_kernel import array_sha256, assess_cases, make_inputs, measure_cases, shared_adjoint, verify_storage

        torch.set_num_threads(PROTOCOL["restore_threads"])
        torch.set_num_interop_threads(PROTOCOL["torch_interop_threads"])
        result["environment"] = {**runtime(), "numpy_imported": np.__version__, "scipy_imported": scipy.__version__,
                                 "torch_imported": torch.__version__, "torch_build": torch.__config__.show(),
                                 "torch_mkl_available": torch.backends.mkl.is_available()}
        observed = observed_source_files()
        if not all(registration["implementation_sha256"].get(name) == digest for name, digest in observed.items()):
            raise ValueError("Actual local imports are not covered by the registered source closure")
        result["observed_imports_sha256"] = observed
        phase("load_graph")
        graph = load_training_graph(registration["graph_path"])
        result["graph"] = graph.identity
        if (graph.identity["role"] != "full_reference"
                or any(graph.identity[key] != count for key, count in PROTOCOL["graph_counts"].items())
                or graph.identity["reference_graph_sha256"] != PROTOCOL["graph_sha256"]
                or graph.identity["raw_ordered_counts_sha256"] != PROTOCOL["raw_ordered_counts_sha256"]):
            raise ValueError("Loaded graph differs from the registered complete logical graph")

        def raw_counts_hash():
            digest = hashlib.sha256()
            for array in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
                for offset in range(0, len(array), 1_000_000):
                    digest.update(np.asarray(array[offset:offset + 1_000_000], dtype="<i8").tobytes())
            return digest.hexdigest()

        result["raw_counts_sha256_before"] = raw_counts_hash()
        if result["raw_counts_sha256_before"] != PROTOCOL["raw_ordered_counts_sha256"]:
            raise ValueError("Initial raw graph digest differs")
        phase("construct_model")
        input_ids = graph.body_ids[graph.group_index == graph.group_names.index(PROTOCOL["model_input_raw_type"])]
        tick = time.perf_counter()
        model = TrainableLIF(graph, input_ids, config=TrainableLIFConfig(weight_mv=PROTOCOL["weight_mv"]), dtype=torch.float32)
        result["model_construction_seconds"] = time.perf_counter() - tick
        write_json(directory / "model_identity.json", model.identity)
        result["model"] = model.identity
        matrix = model.base_weights_adjoint
        phase("construct_shared_adjoint")
        candidate, views, storage = shared_adjoint(matrix)
        result["storage"] = storage
        write_json(directory / "storage_before.json", storage)
        phase("construct_inputs")
        inputs, selected = make_inputs(graph.n, PROTOCOL["seed"])
        result["input_sha256_before"] = {name: array_sha256(array) for name, array in inputs.items()}
        result["selected_indices_sha256_before"] = array_sha256(selected)
        np.savez(directory / "inputs.npz", **inputs, sparse_selected_indices=selected, **identity)
        records, settings = measure_cases(matrix, candidate, inputs, directory, PROTOCOL, identity, phase)
        if len(records) != PROTOCOL["total_matvec_calls"]:
            raise ValueError("Incomplete fixed call schedule")
        result["matvec_calls"], result["thread_settings"] = len(records), settings
        phase("compare_and_verify")
        result.update(assess_cases(directory, records, graph.n, PROTOCOL))
        result["raw_counts_sha256_after"] = raw_counts_hash()
        result["raw_graph_unchanged"] = result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"]
        result["fixed_adjoint_unchanged"] = verify_storage(candidate, views, storage)
        result["input_sha256_after"] = {name: array_sha256(array) for name, array in inputs.items()}
        result["selected_indices_sha256_after"] = array_sha256(selected)
        result["inputs_unchanged"] = (result["input_sha256_before"] == result["input_sha256_after"]
                                      and result["selected_indices_sha256_before"] == result["selected_indices_sha256_after"])
        result["source_unchanged"] = implementation_hashes() == registration["implementation_sha256"]
        result["graph_file_sha256_after"] = file_sha256(Path(registration["graph_path"]))
        result["graph_file_unchanged"] = result["graph_file_sha256_after"] == registration["graph_file_sha256"]
        result["observed_imports_sha256"] = observed_source_files()
        result["source_closure_complete"] = all(registration["implementation_sha256"].get(name) == digest
                                               for name, digest in result["observed_imports_sha256"].items())
        result["integrity_pass"] = all(result[key] for key in
                                       ("raw_graph_unchanged", "fixed_adjoint_unchanged", "inputs_unchanged",
                                        "source_unchanged", "graph_file_unchanged", "source_closure_complete"))
        if not result["integrity_pass"]:
            result["scipy_kernel_candidate"] = False
            for item in result["torch_thread_candidates"].values():
                item["kernel_candidate"] = False
        result["status"] = "protocol_complete" if result["integrity_pass"] else "failed"
        result["reason"] = "kernel_evidence_recorded" if result["integrity_pass"] else "integrity_failure"
        result["interpretation"] = "Kernel evidence only; production autograd and full-window equivalence are not tested."
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        if torch is not None:
            torch.set_num_threads(PROTOCOL["restore_threads"])
            result["restored_threads"] = torch.get_num_threads()
            result["final_interop_threads"] = torch.get_num_interop_threads()
        phase("finished")
        result["phases_seconds"], result["wall_seconds"] = phases, time.perf_counter() - started
        artifacts = [path for path in directory.iterdir() if path.is_file()
                     and (path.suffix == ".npz" or path.name in ("model_identity.json", "storage_before.json", "timings.json"))]
        result["artifacts_sha256"] = {path.name: file_sha256(path) for path in sorted(artifacts)}
        write_json(directory / "worker_result.json", result)
    return result


def run_registered(graph, directory):
    directory = directory.resolve()
    with workspace_lock():
        # A preexisting directory must remain wholly untouched, including failure files.
        directory.mkdir(parents=True, exist_ok=False)
        try:
            registration = _register(graph, directory)
            budget = registration["resources"]
            if not budget["can_launch"]:
                result = {"status": "failed", "reason": "resource_preflight_failed", "worker_status": "not_launched"}
                write_json(directory / "result.json", result)
            else:
                result = supervise([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)], directory,
                                   rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB,
                                   wall_seconds=600, poll_seconds=.1)
        except BaseException as error:
            if not directory.is_dir():
                raise
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
            write_json(directory / "result.json", result)
        # Preserve the resource result verbatim. This inventory is evidence, not scientific PASS.
        manifest = {"schema": "cnsbridge.adjoint_benchmark_evidence.v1", "resource_result": result,
                    "artifacts_sha256": {str(path.relative_to(directory)): file_sha256(path)
                                         for path in sorted(directory.rglob("*")) if path.is_file()}}
        manifest["manifest_id"] = canonical_hash(manifest)
        write_json(directory / "manifest.json", manifest)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="The complete full_reference.npz asset")
    parser.add_argument("--out", type=Path, help="New evidence directory; existing directories are rejected")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None and (args.graph is not None or args.out is not None):
        parser.error("--worker is exclusive")
    if args.worker is None and (args.graph is None or args.out is None):
        parser.error("--graph and --out are required")
    try:
        result = run_worker(args.worker) if args.worker is not None else run_registered(args.graph, args.out)
    except BaseException as error:
        result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("completed", "protocol_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
