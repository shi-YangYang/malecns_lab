"""Isolated experiment checks; no complete graph or neural training is loaded."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import fullgraph_adjoint_benchmark as benchmark  # noqa: E402


def test_parent_import_cli_and_source_closure():
    code = (
        "import sys; sys.path.insert(0, 'examples'); import fullgraph_adjoint_benchmark as b; "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules; "
        "paths = {str(p.relative_to(b.ROOT)) for p in b.source_paths()}; "
        "assert {'examples/fullgraph_adjoint_benchmark.py', 'examples/adjoint_benchmark_kernel.py', "
        "'examples/fullgraph_matvec_benchmark.py', 'src/cnsbridge/__init__.py', "
        "'src/cnsbridge/brain/__init__.py', 'src/cnsbridge/paths.py', "
        "'src/cnsbridge/learning/__init__.py', 'src/cnsbridge/learning/resource_guard.py'} <= paths"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    help_result = subprocess.run([sys.executable, str(ROOT / "examples/fullgraph_adjoint_benchmark.py"), "--help"],
                                 capture_output=True, text=True, check=True)
    assert "--graph" in help_result.stdout and "--out" in help_result.stdout


def test_registration_failure_evidence_no_overwrite_and_lock(tmp_path):
    graph = tmp_path / "not-a-connectome.npz"
    graph.write_bytes(b"explicit invalid asset, never a model")
    out = tmp_path / "failed_registration"
    result = benchmark.run_registered(graph, out)
    assert result["status"] == "failed" and "registered complete MaleCNS" in result["reason"]
    before = {str(path.relative_to(out)): benchmark.file_sha256(path) for path in out.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        benchmark.run_registered(graph, out)
    assert before == {str(path.relative_to(out)): benchmark.file_sha256(path) for path in out.rglob("*") if path.is_file()}
    with benchmark.workspace_lock():
        with pytest.raises(ValueError, match="owns the workspace lock"):
            benchmark.run_registered(graph, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    # Exception exits release the lock.
    with pytest.raises(RuntimeError), benchmark.workspace_lock():
        raise RuntimeError("test")
    with benchmark.workspace_lock():
        pass


def test_lightweight_real_asset_registration_and_tamper_rejection(tmp_path):
    # Stream the real file SHA only; no archive arrays, NumPy, Torch or model are loaded here.
    graph = ROOT / "data/malecns_v2/full_reference.npz"
    tmp_path.mkdir(exist_ok=True)
    with benchmark.workspace_lock():
        registration = benchmark._register(graph, tmp_path)
        assert benchmark.validate_registration(tmp_path) == registration
        source = tmp_path / "source/examples/adjoint_benchmark_kernel.py"
        source.write_text(source.read_text() + "\n# altered snapshot\n")
        with pytest.raises(ValueError, match="Source snapshot differs"):
            benchmark.validate_registration(tmp_path)
        altered = dict(registration)
        altered["protocol"] = {**registration["protocol"], "repetitions": 1}
        altered["protocol_id"] = benchmark.canonical_hash(altered["protocol"])
        altered["registration_id"] = benchmark.canonical_hash({k: v for k, v in altered.items() if k != "registration_id"})
        (tmp_path / "registration.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="Registered protocol differs"):
            benchmark.validate_registration(tmp_path)


def test_explicit_small_adjoint_kernel_schedule_and_evidence(tmp_path):
    import numpy as np
    import torch
    from adjoint_benchmark_kernel import assess_cases, make_inputs, measure_cases, shared_adjoint, verify_storage

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # Explicit 200-cell algebra fixture, including signed asymmetric edges, self edges and silent rows.
    matrix = torch.sparse_coo_tensor(torch.tensor([[0, 0, 2, 99, 199], [0, 3, 1, 4, 170]]),
                                     torch.tensor([.5, -2., 3., 4., -1.]), (200, 200)).to_sparse_csr()
    candidate, views, evidence = shared_adjoint(matrix)
    assert evidence["contract_pass"]
    assert all(not getattr(candidate, name).flags.writeable for name in views)
    inputs, selected = make_inputs(200, 1901)
    assert selected.shape == (2,) and np.count_nonzero(inputs["sparse"]) == 2
    try:
        records, settings = measure_cases(matrix, candidate, inputs, tmp_path, benchmark.PROTOCOL,
                                          {"registration_id": "explicit-kernel-fixture", "protocol_id": "test"}, lambda *a, **k: None)
        assert len(records) == 180
        expected = [(threads, name, repetition, method) for threads in (1, 2, 4, 8, 10)
                    for name in ("zero", "sparse", "dense") for repetition in range(-1, 5)
                    for method in (("torch", "scipy") if repetition == -1 or repetition % 2 == 0 else ("scipy", "torch"))]
        assert [(item["threads"], item["input"], item["round"], item["method"]) for item in records] == expected
        assert [item["actual"] for item in settings] == [1, 2, 4, 8, 10]
        assert all(item["wall_seconds"] > 0 and item["cpu_seconds"] >= 0 and item["finite"] for item in records)
        result = assess_cases(tmp_path, records, 200, benchmark.PROTOCOL)
        assert result["numeric_comparison_pass"] and not result["numeric_failures"]
        reference = (matrix.to_dense() @ torch.from_numpy(inputs["dense"])).numpy()
        with np.load(tmp_path / "threads_4_dense.npz", allow_pickle=False) as archive:
            assert np.allclose(archive["torch"], reference, rtol=1e-5, atol=1e-5)
            changed = {name: archive[name] for name in archive.files}
        changed["scipy"][0, 0] = np.nan  # A bad warmup is not hidden by timed repeats.
        np.savez(tmp_path / "threads_4_dense.npz", **changed)
        invalid = assess_cases(tmp_path, records, 200, benchmark.PROTOCOL)
        assert not invalid["numeric_comparison_pass"] and not invalid["scipy_kernel_candidate"]
        assert all(not item["kernel_candidate"] for item in invalid["torch_thread_candidates"].values())
        assert verify_storage(candidate, views, evidence)
    finally:
        torch.set_num_threads(4)
