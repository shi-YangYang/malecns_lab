"""Register and supervise an isolated complete MaleCNS CPU/MPS kernel comparison."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import psutil

from fullgraph_adjoint_benchmark import workspace_lock, runtime, observed_source_files
from fullgraph_matvec_benchmark import canonical_hash, file_sha256
from cnsbridge.learning.resource_guard import GIB, resource_budget, supervise, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cnsbridge.mps_benchmark_registration.v1"
PROTOCOL = {
    "schema": "cnsbridge.fullgraph_mps_benchmark.v1", "role": "kernel_cost_and_numerics_only",
    "dtype": "float32", "weight_mv": .275, "model_input_raw_type": "R1-R6",
    "graph_file_sha256": "98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3",
    "graph_sha256": "af12b8dc8775f828740221f92c870d76450266759bca6f1540ff5e27002ac156",
    "raw_ordered_counts_sha256": "4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565",
    "graph_counts": {"nodes": 166700, "edges": 25582938, "synapse_count": 124177617},
    "torch_threads": 4, "torch_interop_threads": 1, "fallback": "0",
    "directions": ["forward", "adjoint"], "input_order": ["zero", "sparse", "dense"],
    "seed": 1901, "input_generator": "adjoint_benchmark_kernel.make_inputs(N, 1901)",
    "input_meaning": "numerical probes, not neural activity or biological stimuli",
    "warmups": 1, "repetitions": 5, "total_matvec_calls": 72,
    "order": "warmup and even round: cpu then mps; odd round: mps then cpu",
    "cpu_forward": "existing shared scipy CSR @ vector", "cpu_adjoint": "torch.sparse.mm(CSR, vector[:, None])[:, 0]",
    "mps": "resident canonical COO; torch.sparse.mm(COO, vector[:, None])[:, 0]",
    "timing": "perf_counter/process_time; synchronize before and after MPS operation",
    "timing_excludes": ["conversion", "transfer", "output readback", "artifact writes"],
    "autograd": "one dense W x and WT grad; fixed matrices, no edge gradients",
    "rtol": 1e-5, "atol": 1e-5, "verification_chunk_edges": 262144,
    "candidate": "all numeric/integrity checks and each input MPS forward+adjoint median sum strictly below CPU",
    "resources": {"requested_rss_bytes": 6 * GIB, "reserve_bytes": GIB, "wall_seconds": 600,
                  "poll_seconds": .1, "mps_cap": "min(effective_rss_budget, recommended_max_memory)"},
}


def source_paths():
    """Conservative local AST import closure rooted at this new entry, including lazy imports."""
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


def environment():
    return {**runtime(), "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}


def register(graph, directory):
    graph = graph.resolve(strict=True)
    if any(directory.iterdir()):
        raise ValueError("Registration requires an empty parent-claimed directory")
    hashes = implementation_hashes()
    for relative, digest in hashes.items():
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if file_sha256(target) != digest:
            raise ValueError(f"Source changed during registration: {relative}")
    value = {"schema": SCHEMA, "registered_utc": datetime.now(timezone.utc).isoformat(), "parent_pid": os.getpid(),
             "output_directory": str(directory.resolve()), "graph_path": str(graph), "graph_file_sha256": file_sha256(graph),
             "protocol": PROTOCOL, "protocol_id": canonical_hash(PROTOCOL), "implementation_sha256": hashes,
             "environment": environment(), "resources": resource_budget(psutil.virtual_memory().available,
                                                                         rss_bytes=6 * GIB, reserve_bytes=GIB)}
    value["registration_id"] = canonical_hash(value)
    write_json(directory / "registration.json", value)
    if value["graph_file_sha256"] != PROTOCOL["graph_file_sha256"]:
        raise ValueError("This benchmark requires the registered complete MaleCNS asset")
    return value


def validate_registration(directory):
    value = json.loads((directory / "registration.json").read_text())
    if value["schema"] != SCHEMA or value["registration_id"] != canonical_hash({k: v for k, v in value.items()
                                                                            if k != "registration_id"}):
        raise ValueError("Invalid registration identity")
    if value["protocol"] != PROTOCOL or value["protocol_id"] != canonical_hash(PROTOCOL):
        raise ValueError("Registered protocol differs")
    if (value["output_directory"] != str(directory.resolve()) or value["environment"] != environment()
            or environment()["PYTORCH_ENABLE_MPS_FALLBACK"] != "0"):
        raise ValueError("Registered directory/runtime/fallback differs")
    hashes = implementation_hashes()
    if value["implementation_sha256"] != hashes:
        raise ValueError("Registered source differs")
    for relative, digest in hashes.items():
        if file_sha256(directory / "source" / relative) != digest:
            raise ValueError(f"Source snapshot differs: {relative}")
    budget = value["resources"]
    if budget != resource_budget(budget["available_bytes_before_launch"], rss_bytes=6 * GIB, reserve_bytes=GIB) or not budget["can_launch"]:
        raise ValueError("Invalid/nonlaunchable registered budget")
    graph = Path(value["graph_path"])
    if (not graph.is_absolute() or value["graph_file_sha256"] != PROTOCOL["graph_file_sha256"]
            or file_sha256(graph) != value["graph_file_sha256"]):
        raise ValueError("Registered complete graph changed")
    return value


def run_registered(graph, directory):
    directory = directory.resolve()
    with workspace_lock():
        directory.mkdir(parents=True, exist_ok=False)
        try:
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
                raise ValueError("Launch with PYTORCH_ENABLE_MPS_FALLBACK=0; fallback is forbidden")
            registration = register(graph, directory)
            budget = registration["resources"]
            if not budget["can_launch"]:
                raise ValueError("resource_preflight_failed")
            result = supervise([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)], directory,
                               rss_limit_bytes=budget["effective_rss_limit_bytes"], reserve_bytes=GIB,
                               wall_seconds=600, poll_seconds=.1)
        except BaseException as error:
            result = {"status": "failed", "reason": f"{type(error).__name__}:{error}"}
            write_json(directory / "result.json", result)
        manifest = {"schema": "cnsbridge.mps_benchmark_evidence.v1", "resource_result": result,
                    "artifacts_sha256": {str(path.relative_to(directory)): file_sha256(path)
                                         for path in sorted(directory.rglob("*")) if path.is_file()}}
        manifest["manifest_id"] = canonical_hash(manifest)
        write_json(directory / "manifest.json", manifest)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--out", type=Path, help="New evidence directory; existing directories are rejected")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.worker is not None and (args.graph is not None or args.out is not None)
            or args.worker is None and (args.graph is None or args.out is None)):
        parser.error("Use --graph and --out, or the internal exclusive --worker")
    if args.worker is not None:
        from mps_benchmark_worker import run_worker
        result = run_worker(args.worker)
    else:
        result = run_registered(args.graph, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("completed", "protocol_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
