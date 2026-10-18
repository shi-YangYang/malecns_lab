"""Registered complete-graph Torch/SciPy CPU matvec comparison, without flight or training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import psutil

from cnsbridge.learning.resource_guard import GIB, resource_budget, supervise, write_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SCHEMA = "cnsbridge.matvec_registration.v1"
PROTOCOL = {
    "schema": "cnsbridge.fullgraph_matvec_benchmark.v1",
    "role": "kernel_cost_and_numerics_only",
    "device": "cpu", "dtype": "float32", "weight_mv": 0.275,
    "torch_threads": 4, "torch_interop_threads": 1,
    "model_input_raw_type": "R1-R6",
    "input_order": ["zero", "sparse", "dense"],
    "rng": "numpy.default_rng", "seed": 1901,
    "sparse_count": "floor(N / 100)", "sparse_value": 1.0,
    "dense": "same generator after choice: uniform(-1, 1, N).astype(float32)",
    "warmups_per_method_per_input": 1, "repetitions": 5,
    "order": "even round: torch then scipy; odd round: scipy then torch",
    "torch_path": "torch.sparse.mm(W, x[:, None])[:, 0]",
    "scipy_path": "torch.from_numpy(csr @ x.detach().numpy())",
    "scipy_constructor": "csr_array((values.numpy(), col_indices.numpy(), crow_indices.numpy()), copy=False)",
    "clock": "time.perf_counter", "timing_excludes": ["input construction", "output copies", "artifact writes"],
    "rtol": 1e-5, "atol": 1e-5,
    "candidate_rule": "all three inputs numerically pass and each scipy median is strictly lower",
}


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    # Include the package import chain as well as the two numerical core modules.
    return [Path(__file__).resolve(), ROOT / "src/cnsbridge/__init__.py",
            ROOT / "src/cnsbridge/learning/__init__.py", ROOT / "src/cnsbridge/learning/resource_guard.py",
            *sorted((ROOT / "src/cnsbridge/brain").glob("*.py"))]


def implementation_hashes() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths()}


def register(graph: Path, directory: Path) -> dict:
    graph = graph.resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=False)
    hashes = implementation_hashes()
    for relative, expected in hashes.items():
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise ValueError(f"Source changed during registration: {relative}")
    # Stream the on-disk asset; do not import NumPy/Torch or decode the graph here.
    graph_hash = file_sha256(graph)
    registration = {
        "schema": REGISTRATION_SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "graph_path": str(graph), "graph_file_sha256": graph_hash,
        "output_directory": str(directory.resolve()),
        "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "implementation_sha256": hashes,
        "resources": {**resource_budget(psutil.virtual_memory().available, rss_bytes=6 * GIB, reserve_bytes=GIB),
                      "wall_limit_seconds": 600, "sampling_seconds": 0.1},
        "environment": {"python": sys.version, "platform": platform.platform(), "psutil": psutil.__version__},
    }
    registration["registration_id"] = canonical_hash(registration)
    write_json(directory / "registration.json", registration)
    return registration


def validate_registration(directory: Path) -> dict:
    registration = json.loads((directory / "registration.json").read_text())
    expected_id = canonical_hash({key: value for key, value in registration.items() if key != "registration_id"})
    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("registration_id") != expected_id:
        raise ValueError("Invalid registration identity")
    if registration.get("protocol") != PROTOCOL or registration.get("protocol_id") != canonical_hash(PROTOCOL):
        raise ValueError("Registered protocol differs from this fixed benchmark")
    if registration.get("output_directory") != str(directory.resolve()):
        raise ValueError("Registered output directory differs")
    hashes = implementation_hashes()
    if registration.get("implementation_sha256") != hashes:
        raise ValueError("Registered implementation differs from current source")
    for relative, expected in hashes.items():
        if file_sha256(directory / "source" / relative) != expected:
            raise ValueError(f"Source snapshot identity differs: {relative}")
    budget = registration["resources"]
    expected_budget = {**resource_budget(budget["available_bytes_before_launch"], rss_bytes=6 * GIB, reserve_bytes=GIB),
                       "wall_limit_seconds": 600, "sampling_seconds": 0.1}
    if budget != expected_budget or not budget["can_launch"]:
        raise ValueError("Invalid or nonlaunchable registered resource budget")
    graph = Path(registration["graph_path"])
    if not graph.is_absolute() or file_sha256(graph) != registration["graph_file_sha256"]:
        raise ValueError("Registered graph file changed")
    return registration


def run_registered(graph: Path, directory: Path) -> dict:
    registration = register(graph, directory)
    budget = registration["resources"]
    if not budget["can_launch"]:
        result = {"schema": "cnsbridge.resource_run.v1", "status": "failed", "reason": "resource_preflight_failed",
                  "worker_status": "not_launched", "returncode": None, "peak_observed_rss_bytes": 0,
                  "wall_seconds": 0.0, "last_phase": None}
        write_json(directory / "result.json", result)
        return result
    # This terminal status deliberately remains resource completion, not scientific PASS.
    return supervise([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory.resolve())],
                     directory, rss_limit_bytes=budget["effective_rss_limit_bytes"],
                     wall_seconds=budget["wall_limit_seconds"], reserve_bytes=budget["reserve_bytes"],
                     poll_seconds=budget["sampling_seconds"])


def run_worker(directory: Path) -> dict:
    # An existing claim prevents accidental reruns from overwriting any evidence.
    with (directory / "worker.claim").open("x") as claim:
        claim.write(datetime.now(timezone.utc).isoformat() + "\n")
    started = time.perf_counter()
    result = {"schema": "cnsbridge.matvec_worker_result.v1", "status": "failed", "reason": None}
    phases = {}
    current, phase_start = None, started

    def phase(name: str, **details):
        nonlocal current, phase_start
        now = time.perf_counter()
        if current is not None:
            phases[current] = phases.get(current, 0.0) + now - phase_start
        current, phase_start = name, now
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": now - started, **details})

    try:
        phase("validate_registration")
        registration = validate_registration(directory)
        run_identity = {key: registration[key] for key in ("registration_id", "protocol_id")}
        result.update(**run_identity, implementation_sha256=registration["implementation_sha256"])
        phase("imports")
        import numpy as np
        import scipy
        from scipy import sparse
        import torch
        from cnsbridge.brain.trainable import TrainableLIF, TrainableLIFConfig
        from cnsbridge.brain.trainable_graph import load_training_graph

        torch.set_num_threads(PROTOCOL["torch_threads"])
        torch.set_num_interop_threads(PROTOCOL["torch_interop_threads"])
        result["environment"] = {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "psutil": psutil.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_mkl_available": torch.backends.mkl.is_available(), "torch_build": torch.__config__.show(),
        }
        phase("load_graph")
        graph = load_training_graph(registration["graph_path"])
        result["graph"] = graph.identity

        def original_hash():
            digest = hashlib.sha256()
            # Exactly TrainingGraph's body_ids, CSC indptr/indices/data order and <i8 encoding.
            for array in (graph.body_ids, graph.counts.indptr, graph.counts.indices, graph.counts.data):
                for offset in range(0, len(array), 1_000_000):
                    digest.update(np.asarray(array[offset:offset + 1_000_000], dtype="<i8").tobytes())
            return digest.hexdigest()

        result["raw_counts_sha256_before"] = original_hash()
        if result["raw_counts_sha256_before"] != graph.identity["raw_ordered_counts_sha256"]:
            raise ValueError("Loaded graph digest differs from TrainingGraph identity")
        phase("construct_model")
        input_ids = graph.body_ids[graph.group_index == graph.group_names.index(PROTOCOL["model_input_raw_type"])]
        construction_start = time.perf_counter()
        model = TrainableLIF(graph, input_ids, config=TrainableLIFConfig(weight_mv=PROTOCOL["weight_mv"]), dtype=torch.float32)
        result["model_construction_seconds"] = time.perf_counter() - construction_start
        result["model"] = model.identity
        write_json(directory / "model_identity.json", model.identity)
        matrix = model.base_weights
        phase("construct_scipy")
        construction_start = time.perf_counter()
        views = {"data": matrix.values().numpy(), "indices": matrix.col_indices().numpy(), "indptr": matrix.crow_indices().numpy()}
        candidate = sparse.csr_array((views["data"], views["indices"], views["indptr"]), shape=tuple(matrix.shape), copy=False)
        result["scipy_construction_seconds"] = time.perf_counter() - construction_start

        def array_hash(array):
            return hashlib.sha256(memoryview(array).cast("B")).hexdigest()

        shared = {}
        for name, view in views.items():
            target = getattr(candidate, name)
            shared[name] = {"shares_memory": bool(np.shares_memory(target, view)), "torch_view_dtype": str(view.dtype),
                            "scipy_dtype": str(target.dtype), "torch_view_shape": list(view.shape),
                            "scipy_shape": list(target.shape), "torch_view_strides": list(view.strides),
                            "scipy_strides": list(target.strides), "same_order_and_values": bool(np.array_equal(target, view)),
                            "torch_view_address": view.ctypes.data, "scipy_address": target.ctypes.data,
                            "sha256_before": array_hash(view)}
        result["storage"] = shared
        result["storage_contract_pass"] = all(item["shares_memory"] and item["same_order_and_values"]
                                               and item["torch_view_dtype"] == item["scipy_dtype"]
                                               and item["torch_view_strides"] == item["scipy_strides"] for item in shared.values())
        if not result["storage_contract_pass"]:
            raise ValueError("SciPy candidate did not preserve shared arrays, dtypes and order")

        phase("construct_inputs")
        rng = np.random.default_rng(PROTOCOL["seed"])
        inputs = {"zero": np.zeros(graph.n, dtype=np.float32), "sparse": np.zeros(graph.n, dtype=np.float32)}
        selected = rng.choice(graph.n, size=graph.n // 100, replace=False)
        inputs["sparse"][selected] = 1.0
        inputs["dense"] = rng.uniform(-1, 1, size=graph.n).astype(np.float32)
        np.savez(directory / "inputs.npz", **inputs, sparse_selected_indices=selected, **run_identity)
        outputs, warmup_outputs, timings, records = {}, {}, {}, []
        methods = {"torch": lambda x: torch.sparse.mm(matrix, x[:, None])[:, 0],
                   "scipy": lambda x: torch.from_numpy(candidate @ x.detach().numpy())}
        with torch.no_grad():
            for input_name in PROTOCOL["input_order"]:
                x = torch.from_numpy(inputs[input_name])
                phase("benchmark", input=input_name)
                for method, operation in methods.items():
                    warmup_outputs[f"{input_name}_{method}"] = operation(x).numpy().copy()
                    outputs[f"{input_name}_{method}"] = []
                    timings[f"{input_name}_{method}"] = []
                for repetition in range(PROTOCOL["repetitions"]):
                    order = ("torch", "scipy") if repetition % 2 == 0 else ("scipy", "torch")
                    for method in order:
                        operation = methods[method]
                        tick = time.perf_counter()
                        value = operation(x)
                        seconds = time.perf_counter() - tick
                        key = f"{input_name}_{method}"
                        timings[key].append(seconds)
                        records.append({"sequence": len(records), "input": input_name, "round": repetition,
                                        "method": method, "seconds": seconds})
                        outputs[key].append(value.numpy().copy())
                # Persist each completed input before continuing, including raw repetitions.
                np.savez(directory / f"{input_name}_outputs.npz",
                         torch=np.stack(outputs[f"{input_name}_torch"]), scipy=np.stack(outputs[f"{input_name}_scipy"]),
                         warmup_torch=warmup_outputs[f"{input_name}_torch"],
                         warmup_scipy=warmup_outputs[f"{input_name}_scipy"], **run_identity)
                write_json(directory / "timings.json", {**run_identity, "records": records, "seconds_by_input_method": timings})

        phase("compare_and_verify")
        comparisons = {}
        for input_name in PROTOCOL["input_order"]:
            baseline, alternative = (np.stack(outputs[f"{input_name}_{name}"]) for name in ("torch", "scipy"))
            difference = alternative.astype(np.float64) - baseline.astype(np.float64)
            comparison = {
                "torch_shape": list(baseline.shape), "scipy_shape": list(alternative.shape),
                "torch_dtype": str(baseline.dtype), "scipy_dtype": str(alternative.dtype),
                "torch_finite": bool(np.isfinite(baseline).all()), "scipy_finite": bool(np.isfinite(alternative).all()),
                "max_abs_difference": float(np.max(np.abs(difference))) if np.isfinite(difference).all() else None,
                "rms_difference": float(np.sqrt(np.mean(difference ** 2))) if np.isfinite(difference).all() else None,
                "allclose": bool(np.allclose(alternative, baseline, rtol=PROTOCOL["rtol"], atol=PROTOCOL["atol"])),
                "torch_median_seconds": float(np.median(timings[f"{input_name}_torch"])),
                "scipy_median_seconds": float(np.median(timings[f"{input_name}_scipy"])),
            }
            comparison["numeric_pass"] = (comparison["allclose"] and comparison["torch_finite"] and comparison["scipy_finite"]
                                          and baseline.dtype == alternative.dtype == np.dtype("float32")
                                          and baseline.shape == alternative.shape == (PROTOCOL["repetitions"], graph.n))
            comparison["scipy_faster"] = comparison["scipy_median_seconds"] < comparison["torch_median_seconds"]
            comparisons[input_name] = comparison
        result["comparisons"] = comparisons
        result["raw_counts_sha256_after"] = original_hash()
        result["raw_graph_unchanged"] = result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"]
        for name, view in views.items():
            shared[name]["sha256_after"] = array_hash(view)
        result["fixed_weights_unchanged"] = all(item["sha256_before"] == item["sha256_after"] for item in shared.values())
        result["source_unchanged"] = implementation_hashes() == registration["implementation_sha256"]
        result["graph_file_sha256_after"] = file_sha256(Path(registration["graph_path"]))
        result["graph_file_unchanged"] = result["graph_file_sha256_after"] == registration["graph_file_sha256"]
        result["numeric_comparison_pass"] = all(item["numeric_pass"] for item in comparisons.values())
        integrity = all(result[key] for key in ("raw_graph_unchanged", "fixed_weights_unchanged", "source_unchanged", "graph_file_unchanged"))
        result["production_candidate"] = (integrity and result["numeric_comparison_pass"]
                                          and all(item["scipy_faster"] for item in comparisons.values()))
        result["interpretation"] = "Kernel evidence only; independent review and full dynamics/gradient comparisons remain required."
        artifacts = ["inputs.npz", "model_identity.json", "timings.json", *[f"{name}_outputs.npz" for name in PROTOCOL["input_order"]]]
        result["artifacts_sha256"] = {name: file_sha256(directory / name) for name in artifacts}
        result["status"] = "protocol_complete" if integrity else "failed"
        result["reason"] = "kernel_evidence_recorded" if integrity else "integrity_check_failed"
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        phase("finished")
        result["phases_seconds"] = phases
        result["wall_seconds"] = time.perf_counter() - started
        write_json(directory / "worker_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, help="Verified complete full_reference NPZ")
    parser.add_argument("--out", type=Path, help="New artifact directory; existing directories are rejected")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        if args.graph is not None or args.out is not None:
            parser.error("--worker cannot be combined with --graph or --out")
        result = run_worker(args.worker)
    else:
        if args.graph is None or args.out is None:
            parser.error("--graph and --out are required")
        result = run_registered(args.graph, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("completed", "protocol_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
