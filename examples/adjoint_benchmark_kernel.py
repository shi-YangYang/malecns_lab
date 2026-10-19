"""Numerical worker helpers for the isolated adjoint cost experiment.

Import this module only inside a supervised worker (or an explicit tiny kernel
check). It does not implement a neural model or a production backward path.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy import sparse
import torch

from cnsbridge.learning.resource_guard import write_json


def array_sha256(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def shared_adjoint(matrix):
    """Build one read-only SciPy CSR over all existing Torch adjoint buffers."""
    if matrix.layout != torch.sparse_csr or matrix.device.type != "cpu" or matrix.dtype != torch.float32:
        raise ValueError("Expected the complete CPU float32 CSR adjoint")
    started = time.perf_counter()
    views = {"data": matrix.values().numpy(), "indices": matrix.col_indices().numpy(),
             "indptr": matrix.crow_indices().numpy()}
    for view in views.values():
        view.flags.writeable = False
    candidate = sparse.csr_array((views["data"], views["indices"], views["indptr"]),
                                 shape=tuple(matrix.shape), copy=False)
    for name in views:
        getattr(candidate, name).flags.writeable = False
    seconds = time.perf_counter() - started
    evidence = {"matrix_shape": list(matrix.shape), "candidate_shape": list(candidate.shape),
                "matrix_nnz": matrix._nnz(), "candidate_nnz": candidate.nnz, "construction_seconds": seconds,
                "arrays": {}}
    for name, view in views.items():
        target = getattr(candidate, name)
        item = {"shares_memory": bool(np.shares_memory(target, view)),
                "torch_view_address": view.ctypes.data, "scipy_address": target.ctypes.data,
                "torch_view_dtype": str(view.dtype), "scipy_dtype": str(target.dtype),
                "torch_view_shape": list(view.shape), "scipy_shape": list(target.shape),
                "torch_view_strides": list(view.strides), "scipy_strides": list(target.strides),
                "torch_view_readonly": not view.flags.writeable, "scipy_readonly": not target.flags.writeable,
                "same_order_and_values": bool(np.array_equal(view, target)),
                "torch_sha256_before": array_sha256(view), "scipy_sha256_before": array_sha256(target)}
        evidence["arrays"][name] = item
    evidence["contract_pass"] = (
        evidence["matrix_shape"] == evidence["candidate_shape"] and evidence["matrix_nnz"] == evidence["candidate_nnz"]
        and views["data"].dtype == np.dtype("float32")
        and all(views[name].dtype == np.dtype("int64") for name in ("indices", "indptr"))
        and all(item["shares_memory"] and item["torch_view_address"] == item["scipy_address"]
                and item["torch_view_dtype"] == item["scipy_dtype"]
                and item["torch_view_shape"] == item["scipy_shape"]
                and item["torch_view_strides"] == item["scipy_strides"]
                and item["torch_view_readonly"] and item["scipy_readonly"] and item["same_order_and_values"]
                for item in evidence["arrays"].values()))
    if not evidence["contract_pass"]:
        raise ValueError("The SciPy adjoint must preserve every read-only Torch buffer without conversion")
    return candidate, views, evidence


def verify_storage(candidate, views, evidence):
    for name, view in views.items():
        item, target = evidence["arrays"][name], getattr(candidate, name)
        item.update(torch_sha256_after=array_sha256(view), scipy_sha256_after=array_sha256(target),
                    torch_readonly_after=not view.flags.writeable, scipy_readonly_after=not target.flags.writeable,
                    pointers_unchanged=(view.ctypes.data == item["torch_view_address"]
                                        and target.ctypes.data == item["scipy_address"]))
    return all(item["torch_sha256_before"] == item["torch_sha256_after"]
               == item["scipy_sha256_before"] == item["scipy_sha256_after"]
               and item["torch_readonly_after"] and item["scipy_readonly_after"] and item["pointers_unchanged"]
               for item in evidence["arrays"].values())


def make_inputs(n, seed):
    rng = np.random.default_rng(seed)
    inputs = {"zero": np.zeros(n, dtype=np.float32), "sparse": np.zeros(n, dtype=np.float32)}
    selected = rng.choice(n, size=n // 100, replace=False)
    inputs["sparse"][selected] = 1.0
    inputs["dense"] = rng.uniform(-1, 1, n).astype(np.float32)
    return inputs, selected


def measure_cases(matrix, candidate, inputs, directory, protocol, identity, phase):
    """Persist each case before releasing it; every warmup and timed call is saved."""
    operations = {"torch": lambda x: torch.sparse.mm(matrix, x[:, None])[:, 0],
                  "scipy": lambda x: torch.from_numpy(candidate @ x.detach().numpy())}
    records, thread_settings = [], []
    with torch.no_grad():
        for threads in protocol["torch_threads"]:
            torch.set_num_threads(threads)
            setting = {"requested": threads, "actual": torch.get_num_threads(),
                       "interop": torch.get_num_interop_threads()}
            thread_settings.append(setting)
            if setting["actual"] != threads or setting["interop"] != protocol["torch_interop_threads"]:
                raise ValueError("Actual thread settings differ from protocol")
            for name in protocol["input_order"]:
                phase("benchmark", threads=threads, input=name)
                x = torch.from_numpy(inputs[name])
                outputs = {"torch": [], "scipy": []}
                for repetition in range(-1, protocol["repetitions"]):
                    # -1 is the single warmup of each method, in the declared order.
                    order = ("torch", "scipy") if repetition == -1 or repetition % 2 == 0 else ("scipy", "torch")
                    for method in order:
                        operation = operations[method]
                        cpu_start, wall_start = time.process_time(), time.perf_counter()
                        value = operation(x)
                        wall_seconds, cpu_seconds = time.perf_counter() - wall_start, time.process_time() - cpu_start
                        array = value.numpy().copy()  # Intentionally outside the measured operation.
                        records.append({"sequence": len(records), "threads": threads, "input": name,
                                        "round": repetition, "warmup": repetition == -1, "method": method,
                                        "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
                                        "cpu_seconds_per_wall_second": cpu_seconds / wall_seconds,
                                        "dtype": str(array.dtype), "shape": list(array.shape),
                                        "finite": bool(np.isfinite(array).all()), "output_sha256": array_sha256(array)})
                        outputs[method].append(array)
                np.savez(directory / f"threads_{threads}_{name}.npz",
                         **{key: np.stack(value) for key, value in outputs.items()}, **identity)
                write_json(directory / "timings.json", {**identity, "records": records, "thread_settings": thread_settings})
                del outputs, array, value, x
    return records, thread_settings


def comparison(actual, reference, n, protocol):
    shape_pass = actual.shape == reference.shape == (n,)
    dtype_pass = actual.dtype == reference.dtype == np.dtype("float32")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    difference = actual.astype(np.float64) - reference.astype(np.float64) if shape_pass else None
    allclose = shape_pass and bool(np.allclose(actual, reference, rtol=protocol["rtol"], atol=protocol["atol"]))
    return {"actual_shape": list(actual.shape), "reference_shape": list(reference.shape),
            "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
            "finite": finite, "exact": bool(np.array_equal(actual, reference)), "allclose": allclose,
            "max_abs_difference": float(np.max(np.abs(difference))) if finite and shape_pass else None,
            "rms_difference": float(np.sqrt(np.mean(difference ** 2))) if finite and shape_pass else None,
            "numeric_pass": bool(shape_pass and dtype_pass and finite and allclose)}


def assess_cases(directory, records, n, protocol):
    """Use all outputs, including warmups; compare every thread to the fixed 4-thread reference."""
    comparisons, summaries, failures = {}, {}, []
    for name in protocol["input_order"]:
        with np.load(directory / f"threads_4_{name}.npz", allow_pickle=False) as archive:
            baseline = archive["torch"]
        for threads in protocol["torch_threads"]:
            key = f"{threads}/{name}"
            with np.load(directory / f"threads_{threads}_{name}.npz", allow_pickle=False) as archive:
                original, alternative = archive["torch"], archive["scipy"]
            checks = {}
            for repetition in range(protocol["repetitions"] + 1):
                checks[str(repetition - 1)] = {
                    "scipy_vs_torch": comparison(alternative[repetition], original[repetition], n, protocol),
                    "torch_vs_torch4": comparison(original[repetition], baseline[repetition], n, protocol),
                    "scipy_vs_torch4": comparison(alternative[repetition], baseline[repetition], n, protocol)}
                failures.extend(f"{key}/round{repetition - 1}/{kind}" for kind, item in checks[str(repetition - 1)].items()
                                if not item["numeric_pass"])
            comparisons[key] = checks
            summaries[key] = {}
            for method in ("torch", "scipy"):
                selected = [item for item in records if item["threads"] == threads and item["input"] == name
                            and item["method"] == method and not item["warmup"]]
                wall = float(np.median([item["wall_seconds"] for item in selected]))
                cpu = float(np.median([item["cpu_seconds"] for item in selected]))
                summaries[key][method] = {"repetitions": len(selected), "median_wall_seconds": wall,
                                          "median_cpu_seconds": cpu, "median_cpu_div_median_wall": cpu / wall}
            del original, alternative
        del baseline
    numeric_pass = not failures
    scipy_ratios = {name: summaries[f"4/{name}"]["scipy"]["median_wall_seconds"]
                   / summaries[f"4/{name}"]["torch"]["median_wall_seconds"] for name in protocol["input_order"]}
    thread_candidates = {}
    for threads in protocol["torch_threads"]:
        ratios = {name: summaries[f"{threads}/{name}"]["torch"]["median_wall_seconds"]
                  / summaries[f"4/{name}"]["torch"]["median_wall_seconds"] for name in protocol["input_order"]}
        thread_candidates[str(threads)] = {"ratio_to_torch4": ratios,
                                           "kernel_candidate": numeric_pass and all(value < 1 for value in ratios.values())}
    return {"comparisons": comparisons, "timing_summary": summaries, "numeric_failures": failures,
            "numeric_comparison_pass": numeric_pass, "scipy_ratio_to_torch4": scipy_ratios,
            "scipy_kernel_candidate": numeric_pass and all(value < 1 for value in scipy_ratios.values()),
            "torch_thread_candidates": thread_candidates}
