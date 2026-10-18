"""Lightweight immutable identities for the P3e1 full-reference R candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

SCHEMA = "cnsbridge.bounded_rewiring.v1"
PROTOCOL = {"algorithm": "uniform-edge-pair-sign-preserving-sorted-column-v1", "seed": 1901,
            "max_attempts": 102331348, "target_swaps": 25582837, "max_segment_proposals": 1000000,
            "nodes": 166700, "edges": 25582938, "synapses": 124177617, "self_edges": 101,
            "inputs": 3377, "outputs": 1314, "reserve_bytes": 1073741824,
            "requested_rss_bytes": 6442450944, "wall_seconds": 600, "poll_seconds": .1,
            "meaning": "single seed engineering budget; no uniform sampling or mixing claim"}
MOTHER = {"file_sha256": "98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3",
          "source_id": "873a58e99f5347f8bd5d3bee76e718bdccb340a67f070dc4f837f02262b41342",
          "graph_sha256": "af12b8dc8775f828740221f92c870d76450266759bca6f1540ff5e27002ac156"}
SOURCE_FILES = ("learning/rewiring_steps.py", "learning/rewiring_contract.py", "learning/rewiring_archive.py",
                "learning/rewiring_evidence.py", "learning/rewiring_worker.py", "learning/generate_rewiring.py",
                "learning/graph_controls.py", "learning/graph_diagnostics.py", "learning/resource_guard.py",
                "brain/reference.py", "brain/reference_audit.py", "brain/metadata.py", "brain/connectome.py")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    require(isinstance(value, dict), "Expected JSON object")
    return value


def bind(value, field):
    require(value.get(field) == canonical({k: v for k, v in value.items() if k != field}), "Invalid " + field)


def source_hashes():
    package = Path(__file__).resolve().parents[1]
    return {name: sha(package / name) for name in SOURCE_FILES}


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            **{name: version(name) for name in ("numpy", "scipy", "psutil")}}


def graph_metadata(path):
    # Only the small ZIP metadata is decoded. Sparse arrays are worker-only.
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
    require(meta.get("asset_type") == "full_reference" and meta.get("format_version") == 2,
            "Genuine complete full_reference required")
    require(canonical(meta["provenance"]) == meta["source_id"], "Mother provenance mismatch")
    observed = {"file_sha256": sha(path), "source_id": meta["source_id"],
                "graph_sha256": meta["provenance"]["graph_sha256"]}
    require(observed == MOTHER, "Mother differs from the registered complete MaleCNS graph")
    return {"path": str(Path(path).resolve()), **observed, "provenance": meta["provenance"]}


def register_run(graph):
    run = {"schema": SCHEMA, "purpose": "full_reference_bounded_R_generation_only", "protocol": dict(PROTOCOL),
           "mother": graph_metadata(graph), "implementation_sha256": source_hashes(), "runtime": runtime(),
           "registered_utc": datetime.now(timezone.utc).isoformat()}
    run["run_id"] = canonical(run)
    return run


def validate_run(run):
    bind(run, "run_id")
    require(run["schema"] == SCHEMA and run["purpose"] == "full_reference_bounded_R_generation_only"
            and run["protocol"] == PROTOCOL, "Generation protocol changed")
    require(run["implementation_sha256"] == source_hashes() and run["runtime"] == runtime(), "Generation source/runtime changed")
    require(graph_metadata(run["mother"]["path"]) == run["mother"], "Registered mother changed")


def boundary(counts, protocol):
    return counts["attempts"] == protocol["max_attempts"] or counts["successful_swaps"] == protocol["target_swaps"]


def prefix_hash(previous, checkpoint_id, segment_id):
    return canonical({"previous_prefix": previous, "checkpoint_id": checkpoint_id, "segment_id": segment_id})
