"""Isolated probe checks; only the last check performs explicit tiny matrix algebra."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import fullgraph_mps_benchmark as benchmark  # noqa: E402


def test_lightweight_entry_and_complete_new_source_closure():
    subprocess.run([sys.executable, "-c", (
        "import sys; sys.path.insert(0, 'examples'); import fullgraph_mps_benchmark as b; "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
        "paths = {str(p.relative_to(b.ROOT)) for p in b.source_paths()}; "
        "assert {'examples/fullgraph_mps_benchmark.py', 'examples/mps_benchmark_worker.py', "
        "'examples/mps_benchmark_kernel.py', 'examples/mps_benchmark_memory.py', "
        "'src/malecns_lab/brain/trainable.py', 'src/malecns_lab/learning/resource_guard.py'} <= paths"
    )], cwd=ROOT, check=True)
    result = subprocess.run([sys.executable, str(ROOT / "examples/fullgraph_mps_benchmark.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--graph" in result.stdout and "--out" in result.stdout


def test_failed_registration_preserves_evidence_and_exclusivity(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    graph = tmp_path / "invalid.npz"
    graph.write_bytes(b"invalid asset, not a connectome")
    out = tmp_path / "failed"
    result = benchmark.run_registered(graph, out)
    assert result["status"] == "failed" and "registered complete MaleCNS" in result["reason"]
    assert (out / "registration.json").exists() and (out / "manifest.json").exists()
    before = {str(p.relative_to(out)): benchmark.file_sha256(p) for p in out.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        benchmark.run_registered(graph, out)
    assert before == {str(p.relative_to(out)): benchmark.file_sha256(p) for p in out.rglob("*") if p.is_file()}
    with benchmark.workspace_lock():
        with pytest.raises(ValueError, match="owns the workspace lock"):
            benchmark.run_registered(graph, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert "fallback is forbidden" in benchmark.run_registered(graph, tmp_path / "fallback")["reason"]


def test_registration_identity_snapshot_and_protocol_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    # Only streams file bytes; does not open arrays or construct any graph/model.
    with benchmark.workspace_lock():
        registration = benchmark.register(ROOT / "data/malecns_v2/full_reference.npz", tmp_path)
        assert benchmark.validate_registration(tmp_path) == registration
        snapshot = tmp_path / "source/examples/mps_benchmark_kernel.py"
        snapshot.write_text(snapshot.read_text() + "\n# tamper\n")
        with pytest.raises(ValueError, match="Source snapshot differs"):
            benchmark.validate_registration(tmp_path)
        changed = dict(registration)
        changed["protocol"] = {**registration["protocol"], "repetitions": 1}
        changed["protocol_id"] = benchmark.canonical_hash(changed["protocol"])
        changed["registration_id"] = benchmark.canonical_hash({k: v for k, v in changed.items() if k != "registration_id"})
        (tmp_path / "registration.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="Registered protocol differs"):
            benchmark.validate_registration(tmp_path)


def test_tiny_signed_nonsymmetric_mps_schedule_autograd_and_memory(tmp_path):
    with benchmark.workspace_lock():
        import numpy as np
        import torch
        from adjoint_benchmark_kernel import make_inputs, shared_adjoint
        from mps_benchmark_kernel import build_cache, measure_cases, assess_cases, autograd_check, comparison
        from mps_benchmark_memory import MPSMemorySampler

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available; GPU checks not executed")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        matrix = torch.sparse_coo_tensor(torch.tensor([[0, 0, 2, 99, 199], [0, 3, 1, 4, 170]]),
                                         torch.tensor([.5, -2., 3., 4., -1.]), (200, 200)).to_sparse_csr()
        adjoint = matrix.transpose(0, 1).to_sparse_csr()
        forward, _, _ = shared_adjoint(matrix)
        budget = benchmark.resource_budget(benchmark.psutil.virtual_memory().available)["effective_rss_limit_bytes"]
        sampler = MPSMemorySampler(tmp_path / "mps_memory.csv", budget)
        try:
            caches = {"forward": build_cache(matrix, 2)[0], "adjoint": build_cache(adjoint, 2)[0]}
            inputs, selected = make_inputs(200, 1901)
            assert selected.shape == (2,)
            identity = {"registration_id": "explicit-tiny-algebra", "protocol_id": "not-a-brain"}
            records, transfers = measure_cases(forward, adjoint, caches, inputs, tmp_path, benchmark.PROTOCOL,
                                               identity, lambda name, **kw: sampler.sample(name))
            expected = [(name, direction, repetition, method) for name in ("zero", "sparse", "dense")
                        for direction in ("forward", "adjoint") for repetition in range(-1, 5)
                        for method in (("cpu", "mps") if repetition == -1 or repetition % 2 == 0 else ("mps", "cpu"))]
            assert len(records) == 72
            assert [(r["input"], r["direction"], r["round"], r["method"]) for r in records] == expected
            assert all(r["wall_seconds"] > 0 and r["finite"] for r in records) and len(transfers) == 3
            assessed = assess_cases(tmp_path, records, 200, benchmark.PROTOCOL)
            assert assessed["numeric_comparison_pass"]
            gradient = autograd_check(forward, adjoint, caches, inputs["dense"], tmp_path, benchmark.PROTOCOL, identity)
            assert gradient["numeric_pass"]
            with np.load(tmp_path / "autograd.npz") as saved:
                direct = matrix.to_dense().numpy()
                assert np.allclose(saved["output"], direct @ inputs["dense"], rtol=1e-5, atol=1e-5)
                assert np.allclose(saved["gradient"], direct.T @ inputs["dense"], rtol=1e-5, atol=1e-5)
            bad = tmp_path / records[1]["file"]  # A warmup failure must not be excluded.
            np.savez(bad, output=np.full(200, np.nan, dtype=np.float32))
            assert not assess_cases(tmp_path, records, 200, benchmark.PROTOCOL)["numeric_comparison_pass"]
            difference = comparison(np.array([1., 0., -2.], np.float32), np.array([0., 1., 2.], np.float32), 3, benchmark.PROTOCOL)
            assert difference["zero_support_difference_count"] == 2 and difference["nonzero_sign_difference_count"] == 1
        finally:
            memory = sampler.close()
        assert memory["samples"] >= 2 and memory["error"] is None
        assert memory["budget"]["allocator_cap_bytes"] <= budget
        assert memory["sampled_peak_bytes"]["driver_allocated_bytes"] > 0
