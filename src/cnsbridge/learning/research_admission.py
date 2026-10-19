"""Lightweight terminal-R admission. No Torch, model, or sparse-array loading."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from .generate_rewiring import accepted_candidate
from .rewiring_contract import PROTOCOL, bind, canonical, read_json, require, sha

SCHEMA = "cnsbridge.terminal_R_admission.v1"
FACTORY_FILES = ("research_admission.py", "research_cache.py", "research_contract.py", "research_policy.py")


def factory_sources():
    return {"learning/" + name: sha(Path(__file__).with_name(name)) for name in FACTORY_FILES}


def _header(stream):
    version = np.lib.format.read_magic(stream)
    reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    require(reader is not None, "Unsupported candidate NPY version")
    return reader(stream, max_header_size=4096)


def _small_array(archive, name):
    # ZIP byte length alone cannot bound an array allocation claimed by a forged
    # NPY header. Check shape/dtype before reading any payload or allocating it.
    limit = 4 * 1024 * 1024 if name == "meta" else 1024
    require(archive.getinfo(name + ".npy").file_size <= limit, "Oversized " + name + " metadata")
    with archive.open(name + ".npy") as stream:
        shape, fortran, dtype = _header(stream)
        require(not fortran and (shape == () and dtype.kind in "US" if name == "meta"
                                else shape == (2,) and dtype.kind in "iu"), "Invalid small metadata header")
        size = dtype.itemsize * (1 if name == "meta" else 2)
        require(size <= limit, "Oversized metadata payload declaration")
        raw = stream.read(size + 1)
        require(len(raw) == size, "Small metadata payload length differs")
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _small_metadata(path):
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(str(_small_array(archive, "meta")))
        shape = _small_array(archive, "shape")
        headers = {}
        for name in ("data", "indices", "indptr", "body_ids"):
            with archive.open(name + ".npy") as stream:
                dimensions, fortran, dtype = _header(stream)
                headers[name] = {"shape": list(dimensions), "dtype": dtype.str, "fortran": fortran}
    require(shape.tolist() == [PROTOCOL["nodes"]] * 2 and shape.dtype.kind in "iu", "Candidate shape differs")
    for name, count in (("data", PROTOCOL["edges"]), ("indices", PROTOCOL["edges"]),
                        ("indptr", PROTOCOL["nodes"] + 1), ("body_ids", PROTOCOL["nodes"])):
        header = headers[name]
        require(header["shape"] == [count] and np.dtype(header["dtype"]).kind == "i" and not header["fortran"],
                "Candidate integer array header differs: " + name)
    return meta, headers


def _candidate_binding(meta, run, approved, proof, checkpoint, diagnostics, vector_sha):
    """Check cross-file semantics after the original strict chain gate succeeds."""
    p, mother = meta["provenance"], run["mother"]
    require(meta.get("asset_type") == "research_control" and meta.get("format_version") == 2,
            "Expected research_control candidate")
    require(meta["source_id"] == canonical(p) == proof["source_id"] and p["format"] == "cnsbridge.research_graph.v1",
            "Candidate provenance/source differs")
    require(p["transform"] == "R" and p["seed"] == PROTOCOL["seed"] and p["protocol"] == run["run_id"],
            "Only the formal registered R is admitted")
    expected = {"max_attempts": PROTOCOL["max_attempts"], "target_swaps": PROTOCOL["target_swaps"], "self_edges": "fixed",
                "generation_run_id": run["run_id"], "generation_checkpoint_id": approved["checkpoint_id"],
                "generation_terminal": True, "with_paths": "complete_policy_dual_multisource"}
    require(p["parameters"] == expected and p["parameters"]["generation_terminal"] is True,
            "Candidate is not the exact terminal generation asset")
    require(p["parent_source_id"] == mother["source_id"] and p["parent_graph_sha256"] == mother["graph_sha256"]
            and p["node_metadata_sha256"] == mother["provenance"]["node_metadata_sha256"], "Candidate mother differs")
    require(meta["graph_sha256"] == p["graph_sha256"] == approved["graph_sha256"] == proof["graph_sha256"]
            == checkpoint["state"]["graph_sha256"], "Candidate actual graph identity differs")
    require(p["implementation_sha256"] == run["implementation_sha256"] and p["diagnostics"] == diagnostics
            and diagnostics["complete_vectors_sha256"] == vector_sha, "Generation diagnostics/source differs")
    require(proof["counts"] == checkpoint["state"]["counts"] and proof["run_id"] == run["run_id"]
            and proof["checkpoint_id"] == approved["checkpoint_id"], "Candidate terminal counts differ")
    for label in ("parent", "control"):
        paths = diagnostics["complete_policy"]["paths"][label]
        require(paths["inputs"] == PROTOCOL["inputs"] and paths["outputs"] == PROTOCOL["outputs"],
                "Complete policy diagnostic selection differs")


def admit_research_graph(run_directory):
    """Re-read the complete approved chain; a caller's dictionary is never authority.

    Returned JSON is an audit snapshot, not a transferable permission token.
    A worker must call this again before construction and before returning.
    """
    require(isinstance(run_directory, (str, Path)), "A formal generation run directory is required")
    directory = Path(run_directory).resolve()
    require(directory.is_dir(), "A formal generation run directory is required")
    run_file = directory / "run.json"
    run_sha = sha(run_file)
    approved = accepted_candidate(directory)  # Strict source/runtime, full parent chain and actual file digests.
    run = read_json(run_file)
    require(run["protocol"] == PROTOCOL and approved["run_id"] == run["run_id"], "Generation registration differs")
    candidate = Path(approved["path"])
    segment = candidate.parent
    assessment = read_json(segment / "assessment.json")
    bind(assessment, "assessment_id")
    require(assessment["status"] == "completed" and assessment["accepted"]["terminal"] is True
            and assessment["accepted"]["checkpoint_id"] == approved["checkpoint_id"]
            and assessment["accepted"]["prefix"] == approved["prefix"], "Parent has not approved this terminal")
    proof = read_json(segment / "candidate_verification.json")
    checkpoint = read_json(segment / "checkpoint" / "checkpoint.json")
    diagnostics = read_json(segment / "diagnostics.json")
    hashes = {name: sha(segment / name) for name in
              ("candidate.npz", "diagnostics.json", "mechanical_vectors.npz", "candidate_verification.json", "assessment.json")}
    require(hashes["candidate.npz"] == proof["candidate_sha256"], "Candidate file changed")
    meta, headers = _small_metadata(candidate)
    _candidate_binding(meta, run, approved, proof, checkpoint, diagnostics, hashes["mechanical_vectors.npz"])
    require(sha(run_file) == run_sha, "Generation registration changed during admission")
    result = {"schema": SCHEMA, "purpose": "offline_terminal_R_factory_only", "generation_directory": str(directory),
              "run_id": run["run_id"], "run_file_sha256": run_sha, "checkpoint_id": approved["checkpoint_id"],
              "prefix": approved["prefix"], "terminal_segment": str(segment), "assessment_id": assessment["assessment_id"],
              "candidate": {"path": str(candidate), "source_id": meta["source_id"], "graph_sha256": approved["graph_sha256"],
                            "file_sha256": hashes["candidate.npz"], "headers": headers},
              "mother": run["mother"], "protocol": run["protocol"], "counts": proof["counts"],
              "evidence_sha256": hashes, "generation_sources": run["implementation_sha256"], "generation_runtime": run["runtime"],
              "factory_sources": factory_sources()}
    result["admission_id"] = canonical(result)
    return result


def revalidate_admission(snapshot):
    require(isinstance(snapshot, dict), "Admission snapshot must be JSON")
    bind(snapshot, "admission_id")
    actual = admit_research_graph(snapshot["generation_directory"])
    require(actual == snapshot, "Research admission changed; supplied JSON is not authority")
    return actual
