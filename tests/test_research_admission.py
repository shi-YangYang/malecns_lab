"""Metadata-only admission fixtures. Declared full dimensions are not real graphs."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from malecns_lab.learning import research_admission as module
from malecns_lab.learning.rewiring_contract import PROTOCOL, canonical, sha


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def candidate_file(path, meta):
    # Header-only payloads deliberately cannot be loaded as sparse matrices.
    # The real accepted_candidate gate is replaced only in these metadata tests.
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in (("meta", np.array(json.dumps(meta))), ("shape", np.array([PROTOCOL["nodes"]] * 2))):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, value, allow_pickle=False)
            archive.writestr(name + ".npy", stream.getvalue())
        for name, length in (("data", PROTOCOL["edges"]), ("indices", PROTOCOL["edges"]),
                             ("indptr", PROTOCOL["nodes"] + 1), ("body_ids", PROTOCOL["nodes"])):
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream, {"shape": (length,), "fortran_order": False, "descr": "<i8"})
            archive.writestr(name + ".npy", stream.getvalue())


def fixture(tmp_path, monkeypatch):
    segment = tmp_path / "segments" / "000001"
    segment.mkdir(parents=True)
    (segment / "mechanical_vectors.npz").write_bytes(b"metadata fixture, not actual mechanical vectors")
    counts = {"attempts": PROTOCOL["max_attempts"], "successful_swaps": 1, "rejections": {"fixture": 0}}
    diagnostics = {"complete_vectors_sha256": sha(segment / "mechanical_vectors.npz"),
                   "complete_policy": {"paths": {label: {"inputs": 3377, "outputs": 1314} for label in ("parent", "control")}}}
    run = {"run_id": "metadata-only-fixture", "protocol": dict(PROTOCOL), "implementation_sha256": {"fixture": True},
           "runtime": {"fixture": True}, "mother": {"source_id": "fixture-mother", "graph_sha256": "fixture-g",
           "path": "never-read-as-graph", "provenance": {"node_metadata_sha256": "fixture-metadata"}}}
    approved = {"path": str(segment / "candidate.npz"), "run_id": run["run_id"], "checkpoint_id": "fixture-checkpoint",
                "prefix": "fixture-prefix", "graph_sha256": "fixture-r"}
    parameters = {"max_attempts": PROTOCOL["max_attempts"], "target_swaps": PROTOCOL["target_swaps"], "self_edges": "fixed",
                  "generation_run_id": run["run_id"], "generation_checkpoint_id": approved["checkpoint_id"],
                  "generation_terminal": True, "with_paths": "complete_policy_dual_multisource"}
    p = {"format": "cnsbridge.research_graph.v1", "transform": "R", "seed": 1901, "protocol": run["run_id"],
         "parameters": parameters, "parent_source_id": "fixture-mother", "parent_graph_sha256": "fixture-g",
         "graph_sha256": "fixture-r", "node_metadata_sha256": "fixture-metadata", "implementation_sha256": {"fixture": True},
         "diagnostics": diagnostics}
    meta = {"asset_type": "research_control", "format_version": 2, "source_id": canonical(p), "provenance": p,
            "graph_sha256": "fixture-r"}
    candidate_file(Path(approved["path"]), meta)
    proof = {"source_id": meta["source_id"], "graph_sha256": "fixture-r", "counts": counts,
             "run_id": run["run_id"], "checkpoint_id": approved["checkpoint_id"], "candidate_sha256": sha(approved["path"])}
    checkpoint = {"state": {"graph_sha256": "fixture-r", "counts": counts}}
    assessment = {"status": "completed", "accepted": {"terminal": True, "checkpoint_id": approved["checkpoint_id"],
                                                       "prefix": approved["prefix"]}}
    assessment["assessment_id"] = canonical(assessment)
    for name, value in (("run.json", run), ("segments/000001/assessment.json", assessment),
                        ("segments/000001/candidate_verification.json", proof), ("segments/000001/diagnostics.json", diagnostics),
                        ("segments/000001/checkpoint/checkpoint.json", checkpoint)):
        write_json(tmp_path / name, value)
    calls = []
    def checked_gate(directory):
        calls.append(directory)
        return approved
    monkeypatch.setattr(module, "accepted_candidate", checked_gate)
    return run, approved, meta, proof, checkpoint, diagnostics, calls


def test_snapshot_rechecks_gate_and_binds_all_artifacts_without_loading_arrays(tmp_path, monkeypatch):
    *_, calls = fixture(tmp_path, monkeypatch)
    record = module.admit_research_graph(tmp_path)
    assert len(calls) == 1 and record["purpose"] == "offline_terminal_R_factory_only"
    assert record["candidate"]["headers"]["data"]["shape"] == [25582938]
    assert set(record["evidence_sha256"]) == {"candidate.npz", "diagnostics.json", "mechanical_vectors.npz",
                                            "candidate_verification.json", "assessment.json"}
    assert record["factory_sources"] == module.factory_sources()
    assert module.revalidate_admission(record) == record and len(calls) == 2
    forged = deepcopy(record)
    forged["counts"]["successful_swaps"] += 1
    forged["admission_id"] = canonical({k: v for k, v in forged.items() if k != "admission_id"})
    with pytest.raises(ValueError, match="not authority"):
        module.revalidate_admission(forged)


@pytest.mark.parametrize("wrong", ["W", "P_target", "probe", "nonterminal", "checkpoint", "source", "graph", "selection"])
def test_semantic_candidate_rejections_despite_rehashed_metadata(tmp_path, monkeypatch, wrong):
    run, approved, meta, proof, checkpoint, diagnostics, _ = fixture(tmp_path, monkeypatch)
    p = meta["provenance"]
    if wrong in ("W", "P_target"):
        p["transform"] = wrong
    elif wrong == "probe":
        p["parameters"]["generation_probe_only"] = True
    elif wrong == "nonterminal":
        p["parameters"]["generation_terminal"] = False
    elif wrong == "checkpoint":
        p["parameters"]["generation_checkpoint_id"] = "different"
    elif wrong == "source":
        p["implementation_sha256"] = {"tampered": True}
    elif wrong == "graph":
        p["graph_sha256"] = "another-graph"
    else:
        p["diagnostics"]["complete_policy"]["paths"]["control"]["inputs"] = 1
    meta["source_id"] = canonical(p)
    proof["source_id"] = meta["source_id"]
    with pytest.raises(ValueError):
        module._candidate_binding(meta, run, approved, proof, checkpoint, diagnostics,
                                  diagnostics["complete_vectors_sha256"])


def test_changed_file_parent_failure_and_changed_factory_source_rejected(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    record = module.admit_research_graph(tmp_path)
    sources = {**record["factory_sources"], "tampered": "source"}
    with monkeypatch.context() as patch:
        patch.setattr(module, "factory_sources", lambda: sources)
        with pytest.raises(ValueError, match="changed"):
            module.revalidate_admission(record)
    segment = tmp_path / "segments" / "000001"
    assessment = json.loads((segment / "assessment.json").read_text())
    assessment["status"] = "failed"
    assessment["assessment_id"] = canonical({k: v for k, v in assessment.items() if k != "assessment_id"})
    write_json(segment / "assessment.json", assessment)
    with pytest.raises(ValueError, match="Parent"):
        module.admit_research_graph(tmp_path)
    assessment["status"] = "completed"
    assessment["assessment_id"] = canonical({k: v for k, v in assessment.items() if k != "assessment_id"})
    write_json(segment / "assessment.json", assessment)
    with (segment / "candidate.npz").open("ab") as stream:
        stream.write(b"replaced file")
    with pytest.raises(ValueError, match="file changed"):
        module.admit_research_graph(tmp_path)


def test_existing_strict_gate_errors_are_not_bypassed(tmp_path, monkeypatch):
    write_json(tmp_path / "run.json", {})
    calls = []
    def gate(path):
        calls.append(path)
        raise ValueError("source/runtime changed or no approved terminal")
    monkeypatch.setattr(module, "accepted_candidate", gate)
    with pytest.raises(ValueError, match="source/runtime"):
        module.admit_research_graph(tmp_path)
    assert calls == [tmp_path.resolve()]
    for arbitrary in ({"admission_id": "forged"}, tmp_path / "candidate.npz"):
        with pytest.raises(ValueError, match="directory"):
            module.admit_research_graph(arbitrary)


def test_lightweight_import_never_imports_torch_or_model_in_new_process():
    code = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname in {
            'malecns_lab.learning.research_policy', 'malecns_lab.brain.trainable', 'malecns_lab.brain.tonic'}:
            raise AssertionError('Lightweight admission attempted model import: ' + fullname)
sys.meta_path.insert(0, Block())
from malecns_lab.learning.research_admission import factory_sources
assert len(factory_sources()) == 4
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["meta", "shape"])
def test_declared_oversized_small_metadata_rejected_before_payload_allocation(tmp_path, name):
    path = tmp_path / "bad.npz"
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"shape": (2 ** 40,), "fortran_order": False, "descr": "<i8"})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name + ".npy", stream.getvalue())
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(ValueError, match="header"):
            module._small_array(archive, name)
