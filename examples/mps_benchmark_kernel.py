"""Fixed sparse algebra helpers; no neural model, optimizer or edge gradients."""
from __future__ import annotations

import time

import numpy as np
import torch

from adjoint_benchmark_kernel import array_sha256
from malecns_lab.learning.resource_guard import write_json


def verify_cache(cpu_coo, gpu_coo, chunk):
    if (cpu_coo.layout != torch.sparse_coo or gpu_coo.layout != torch.sparse_coo
            or cpu_coo.device.type != "cpu" or gpu_coo.device.type != "mps"
            or cpu_coo.dtype != torch.float32 or gpu_coo.dtype != torch.float32
            or cpu_coo.requires_grad or gpu_coo.requires_grad
            or not cpu_coo.is_coalesced() or not gpu_coo.is_coalesced()
            or cpu_coo.shape != gpu_coo.shape or cpu_coo._nnz() != gpu_coo._nnz()):
        raise ValueError("Fixed canonical COO device/dtype/shape/edge contract failed")
    started = time.perf_counter()
    for start in range(0, cpu_coo._nnz(), chunk):
        stop = start + chunk
        if (not torch.equal(cpu_coo.indices()[:, start:stop], gpu_coo.indices()[:, start:stop].cpu())
                or not torch.equal(cpu_coo.values()[start:stop], gpu_coo.values()[start:stop].cpu())):
            raise ValueError("MPS COO coordinates/values differ from the complete CPU matrix")
    torch.mps.synchronize()
    return {"exact_coordinates_and_values": True, "chunk_edges": chunk,
            "verification_readback_seconds": time.perf_counter() - started,
            "nnz": cpu_coo._nnz(), "shape": list(cpu_coo.shape),
            "cpu_device": str(cpu_coo.device), "mps_device": str(gpu_coo.device),
            "dtype": str(gpu_coo.dtype), "requires_grad": gpu_coo.requires_grad,
            "indices_sha256": array_sha256(cpu_coo.indices().numpy()),
            "values_sha256": array_sha256(cpu_coo.values().numpy())}


def build_cache(cpu_csr, chunk):
    if cpu_csr.layout != torch.sparse_csr or cpu_csr.device.type != "cpu":
        raise ValueError("Expected original full CPU CSR")
    started = time.perf_counter()
    cpu_coo = cpu_csr.to_sparse_coo().coalesce()
    conversion = time.perf_counter() - started
    if cpu_coo._nnz() != cpu_csr._nnz():
        raise ValueError("COO conversion changed the edge count")
    torch.mps.synchronize()
    started = time.perf_counter()
    gpu = cpu_coo.to("mps")
    torch.mps.synchronize()
    transfer = time.perf_counter() - started
    evidence = verify_cache(cpu_coo, gpu, chunk)
    evidence.update(conversion_seconds=conversion, transfer_seconds=transfer)
    # CPU COO is temporary; only original CSR and the MPS cache survive.
    return gpu, evidence


class _FixedMPSMatvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, adjoint, vector):
        if any(item.device.type != "mps" for item in (matrix, adjoint, vector)):
            raise ValueError("All fixed-matvec operands must reside on MPS")
        if matrix.requires_grad or adjoint.requires_grad:
            raise ValueError("Per-edge gradients are forbidden")
        ctx.save_for_backward(adjoint)
        return torch.sparse.mm(matrix, vector[:, None])[:, 0]

    @staticmethod
    def backward(ctx, gradient):
        (adjoint,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(adjoint, gradient[:, None])[:, 0]


def fixed_mps_matvec(matrix, adjoint, vector):
    return _FixedMPSMatvec.apply(matrix, adjoint, vector)


def comparison(actual, reference, n, protocol):
    shape = actual.shape == reference.shape == (n,)
    dtype = actual.dtype == reference.dtype == np.dtype("float32")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    result = {"shape_pass": shape, "dtype_pass": dtype, "finite": finite, "exact": bool(np.array_equal(actual, reference)),
              "max_abs_difference": None, "rms_difference": None, "relative_l2": None, "relative_linf": None,
              "relative_error_defined": False, "outside_tolerance_count": None,
              "zero_support_difference_count": None, "nonzero_sign_difference_count": None, "numeric_pass": False}
    if not shape or not finite:
        return result
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    diff = a - b
    maximum = float(np.max(np.abs(diff)))
    l2, ref_l2, ref_linf = float(np.linalg.norm(diff)), float(np.linalg.norm(b)), float(np.max(np.abs(b)))
    outside = int(np.count_nonzero(np.abs(diff) > protocol["atol"] + protocol["rtol"] * np.abs(b)))
    # Undefined relative error for a zero reference is explicit, never hidden with an arbitrary epsilon.
    result.update(max_abs_difference=maximum, rms_difference=float(np.sqrt(np.mean(diff * diff))),
                  relative_l2=l2 / ref_l2 if ref_l2 else (0.0 if not l2 else None),
                  relative_linf=maximum / ref_linf if ref_linf else (0.0 if not maximum else None),
                  relative_error_defined=bool(ref_l2 and ref_linf or not maximum), outside_tolerance_count=outside,
                  zero_support_difference_count=int(np.count_nonzero((a == 0) != (b == 0))),
                  nonzero_sign_difference_count=int(np.count_nonzero((a != 0) & (b != 0) & (np.signbit(a) != np.signbit(b)))),
                  numeric_pass=bool(dtype and outside == 0))
    return result


def measure_cases(forward_csr, adjoint_csr, caches, inputs, directory, protocol, identity, phase):
    records, transfers = [], {}
    with torch.no_grad():
        for name in protocol["input_order"]:
            cpu_x = torch.from_numpy(inputs[name])
            torch.mps.synchronize()
            started = time.perf_counter()
            gpu_x = cpu_x.to("mps")
            torch.mps.synchronize()
            transfers[name] = time.perf_counter() - started
            if gpu_x.device.type != "mps" or not torch.equal(gpu_x.cpu(), cpu_x):
                raise ValueError("MPS input transfer changed the complete vector")
            for direction in protocol["directions"]:
                phase("benchmark", input=name, direction=direction)
                for repetition in range(-1, protocol["repetitions"]):
                    order = ("cpu", "mps") if repetition == -1 or repetition % 2 == 0 else ("mps", "cpu")
                    for method in order:
                        if method == "mps":
                            torch.mps.synchronize()
                        wall_start, cpu_start = time.perf_counter(), time.process_time()
                        if method == "mps":
                            value = torch.sparse.mm(caches[direction], gpu_x[:, None])[:, 0]
                            torch.mps.synchronize()
                        elif direction == "forward":
                            value = torch.from_numpy(forward_csr @ inputs[name])
                        else:
                            value = torch.sparse.mm(adjoint_csr, cpu_x[:, None])[:, 0]
                        wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
                        if value.device.type != method or value.dtype != torch.float32:
                            raise ValueError("Actual result device/dtype differs")
                        readback_start = time.perf_counter()
                        array = value.cpu().numpy().copy()
                        if method == "mps":
                            torch.mps.synchronize()
                        readback = time.perf_counter() - readback_start
                        filename = f"output_{len(records):03d}.npz"
                        write_start = time.perf_counter()
                        np.savez(directory / filename, output=array, **identity)
                        records.append({"sequence": len(records), "input": name, "direction": direction, "round": repetition,
                                        "warmup": repetition == -1, "method": method, "input_device": str(gpu_x.device if method == "mps" else cpu_x.device),
                                        "output_device": str(value.device), "wall_seconds": wall, "cpu_seconds": cpu,
                                        "readback_seconds": readback, "artifact_write_seconds": time.perf_counter() - write_start,
                                        "file": filename, "sha256": array_sha256(array), "finite": bool(np.isfinite(array).all()),
                                        "shape": list(array.shape), "dtype": str(array.dtype)})
                        write_json(directory / "timings.json", {**identity, "input_transfer_seconds": transfers, "records": records})
            if not torch.equal(gpu_x.cpu(), cpu_x):
                raise ValueError("MPS input changed during fixed call schedule")
            del cpu_x, gpu_x
    return records, transfers


def assess_cases(directory, records, n, protocol):
    checks, summary = {}, {}
    by_case = {(item["input"], item["direction"], item["round"], item["method"]): item for item in records}
    for item in records:
        reference_record = by_case[item["input"], item["direction"], item["round"] if item["method"] == "mps" else -1, "cpu"]
        with np.load(directory / item["file"], allow_pickle=False) as saved:
            actual = saved["output"]
        with np.load(directory / reference_record["file"], allow_pickle=False) as saved:
            reference = saved["output"]
        checks[str(item["sequence"])] = {**comparison(actual, reference, n, protocol), "reference_file": reference_record["file"],
                                       "saved_output_hash_matches": array_sha256(actual) == item["sha256"]}
    for name in protocol["input_order"]:
        summary[name] = {}
        for direction in protocol["directions"]:
            medians = {method: float(np.median([row["wall_seconds"] for row in records
                                              if row["input"] == name and row["direction"] == direction
                                              and row["method"] == method and not row["warmup"]])) for method in ("cpu", "mps")}
            summary[name][direction] = {**medians, "mps_to_cpu_ratio": medians["mps"] / medians["cpu"]}
        totals = {method: sum(summary[name][direction][method] for direction in protocol["directions"])
                  for method in ("cpu", "mps")}
        summary[name]["forward_plus_adjoint"] = {**totals, "mps_to_cpu_ratio": totals["mps"] / totals["cpu"]}
    numeric = all(item["numeric_pass"] and item["saved_output_hash_matches"] for item in checks.values())
    return {"comparisons": checks, "timing_summary_seconds": summary, "numeric_comparison_pass": numeric,
            "faster_all_inputs": all(value["forward_plus_adjoint"]["mps_to_cpu_ratio"] < 1 for value in summary.values())}


def autograd_check(forward_csr, adjoint_csr, caches, dense, directory, protocol, identity):
    # Use the same fixed dense vector for x and upstream gradient, both saved explicitly.
    cpu_x = torch.from_numpy(dense.copy())
    x = cpu_x.to("mps").requires_grad_(True)
    upstream = cpu_x.to("mps")
    torch.mps.synchronize()
    started = time.perf_counter()
    output = fixed_mps_matvec(caches["forward"], caches["adjoint"], x)
    output.backward(upstream)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    if x.grad is None or x.grad.device.type != "mps" or output.device.type != "mps":
        raise ValueError("Full input autograd did not stay on MPS")
    actual_output, actual_gradient = output.detach().cpu().numpy(), x.grad.cpu().numpy()
    reference_output = forward_csr @ dense
    reference_gradient = torch.sparse.mm(adjoint_csr, cpu_x[:, None])[:, 0].numpy()
    np.savez(directory / "autograd.npz", input=dense, upstream=dense, output=actual_output, gradient=actual_gradient,
             cpu_output=reference_output, cpu_gradient=reference_gradient, **identity)
    checks = {"forward": comparison(actual_output, reference_output, len(dense), protocol),
              "gradient": comparison(actual_gradient, reference_gradient, len(dense), protocol)}
    return {"comparisons": checks, "numeric_pass": all(item["numeric_pass"] for item in checks.values()),
            "mps_forward_backward_seconds": elapsed, "input_device": str(x.device), "output_device": str(output.device),
            "gradient_device": str(x.grad.device), "matrix_gradients": False,
            "scope": "Fixed-matrix all-cell input gradient only; not SNN BPTT, Adam or task learning"}
