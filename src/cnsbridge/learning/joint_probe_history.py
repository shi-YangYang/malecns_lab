"""Frozen r1 failure anchors and pre-optimization comparison; never loads a model.

The failed r1 did not seal its numerical files in a worker manifest. Those file
hashes were frozen here for r2 after independent r1 evidence acceptance. They
are not claimed to have appeared in the original r1 terminal manifest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .joint_probe_contract import PARAMETERS, PROTOCOL, bound, read_json, require, sha256

R1_ID = {"registration_id": "7ce97c64a3b03dc6884f69a7be0de91ab3f9b5bf7e9c8eca4b5a85acefabeb7f",
         "protocol_id": "4062748f4b7932db11f6c8fb68fa94b0de413433b80afc874883634a8563e562"}
R1_SHA256 = {
    "registration.json": "f7baeede7ce44f16169ead68b8af30a8a26e95ac2621180b6f696b8e2f4f6f68",
    "result.json": "ff4af48fc6ce76ed1b1e8df1436ff60a80b6eaac465a6a42c535a1e3608c094b",
    "selection.npz": "fd50d05fcde4a7caff4c2e21269fbf7da6e281788515cef2f92d9b73fb449adb",
    "initial_parameters.npz": "b46a174978ffbd6f33edc2fa3543139c469bbe598cdc13f08c445e2ef49ef522",
    "baseline.npz": "d2adbbeea3e3aa1b45586f41052a54a4db3c6ac96d6e04935cee762132f7c36d",
    "baseline.json": "f075c8d8c148fabaa4425c1723ab2789a778afa0a861cb5c50e3540f84742d2e",
    "gradients_before_clip.npz": "c67429d3e14018ff4cda3b97824cc9e7dea624c65ec4d9f445cc88a6bb4a4628",
    "policy_identity.json": "573e3e5f96f76b11e290629924026d5dd54f594a460020d90df9deccc34c65d8",
    "dn_ids.npz": "0a7b17b1e15d19620fa1ebb09a54eb083b8c9acad78892ed5c6ce70383d53886",
}


def history_binding(directory, implementation, graph):
    directory = Path(directory)
    require(directory.is_absolute(), "r1 path must be absolute")
    for name, expected in R1_SHA256.items():
        require(sha256(directory / name) == expected, f"Frozen r1 file changed: {name}")
    old = read_json(directory / "registration.json")
    bound(old, "registration_id")
    require(all(old[k] == v for k, v in R1_ID.items()) and old["schema"] == "cnsbridge.joint_probe_registration.v1",
            "Wrong r1 registration/protocol identity")
    from ..brain.metadata import canonical_hash
    require(old["protocol_id"] == canonical_hash(old["protocol"]), "r1 protocol binding differs")
    terminal = read_json(directory / "result.json")
    require(terminal["schema"] == "cnsbridge.joint_probe_result.v1" and terminal["status"] == "failed"
            and terminal["protocol_complete"] is False and terminal["joint_numeric_update"] is None
            and terminal["internal_action_effect"] is None, "r1 failure outcome changed")
    expected_terminal = {"registration.json", "worker_result.json", "resources.csv", "worker.log", "phase.json",
                         "supervisor_result.json"}
    require(set(terminal["terminal_artifacts_sha256"]) == expected_terminal, "r1 terminal inventory changed")
    for name, expected in terminal["terminal_artifacts_sha256"].items():
        require(sha256(directory / name) == expected, f"r1 terminal artifact changed: {name}")
    worker = read_json(directory / "worker_result.json")
    require(worker["schema"] == "cnsbridge.joint_probe_worker.v1" and worker["status"] == "failed"
            and all(worker[k] == v for k, v in R1_ID.items())
            and worker["implementation_sha256"] == old["implementation_sha256"], "r1 worker binding differs")
    old_sources = old["implementation_sha256"]
    unchanged = {name: digest for name, digest in old_sources.items() if not name.startswith("learning/joint_probe")}
    require(unchanged and all(implementation.get(k) == v for k, v in unchanged.items()),
            "A source outside the r2 optimization/evidence scope changed since r1")
    require(all(graph[k] == old["graph"][k] for k in ("file_sha256", "source_id", "provenance")),
            "Graph differs from r1")
    return {"path": str(directory), **R1_ID, "software_commit": "0373c87", "files_sha256": dict(R1_SHA256),
            "numerical_hash_origin": "r2 preregistration anchors after independent r1 evidence acceptance",
            "unchanged_sources_sha256": unchanged}


def compare_r1(directory, prior_run, identity):
    """Preserve every comparison even on mismatch; caller decides whether to stop."""
    from .joint_probe_evidence import arrays, comparison
    directory, previous = Path(directory), Path(prior_run["path"])
    for name, expected in prior_run["files_sha256"].items():
        require(sha256(previous / name) == expected, f"r1 comparison source changed: {name}")
    param_keys = set(PARAMETERS) | {"gains", "rho"}
    inventory = {"initial_parameters.npz": param_keys,
                 "baseline.npz": param_keys | {"actions", "features_mv"},
                 "gradients_before_clip.npz": set(PARAMETERS) | {"reg_gradient"},
                 "selection.npz": {"frames", "time_s", "valid", "teacher_target", "indices"},
                 "dn_ids.npz": {"body_ids"}}
    reports = {}
    valid = True
    for filename, keys in inventory.items():
        left, right = arrays(previous / filename, keys, R1_ID), arrays(directory / filename, keys, identity)
        reports[filename] = {}
        for name in sorted(keys):
            a, b = left[name], right[name]
            require(a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),
                    f"r1 comparison shape/dtype/finite mismatch: {filename}:{name}")
            report = comparison(a, b)
            exact_required = filename in ("initial_parameters.npz", "selection.npz", "dn_ids.npz") or (
                filename == "baseline.npz" and name in param_keys)
            report["required"] = "exact" if exact_required else "allclose"
            report["passed"] = report[report["required"]]
            valid = valid and report["passed"]
            reports[filename][name] = report
    policy_equal = read_json(directory / "policy_identity.json") == read_json(previous / "policy_identity.json")
    return {"schema": "cnsbridge.joint_probe_r1_comparison.v1", "prior_registration_id": R1_ID["registration_id"],
            "rtol": PROTOCOL["rtol"], "atol": PROTOCOL["atol"], "comparisons": reports,
            "policy_identity_exact": policy_equal, "passed": bool(valid and policy_equal)}
