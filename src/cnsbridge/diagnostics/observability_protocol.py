"""Strict preregistration contract for engineering-input observability (no Torch)."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cnsbridge.learning.resource_guard import DEFAULT_SYSTEM_RESERVE_BYTES, GIB

SCHEMA = "cnsbridge.observability_protocol.v1"
REGISTRATION_SCHEMA = "cnsbridge.observability_registration.v1"
WORKER_SCHEMA = "cnsbridge.observability_worker.v1"
MODES = ("zero_background", "fixed_lamina_8mv")
CONDITIONS = ("blank", "center", "left", "right", "up", "down", "large")
LAMINA = ("L1", "L2", "L3", "L4", "L5")
RECTANGLES = (None, (5, 7, 7, 9), (5, 7, 3, 5), (5, 7, 11, 13), (2, 4, 7, 9), (8, 10, 7, 9), (4, 8, 6, 10))


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    value = json.loads(path.read_text(), parse_constant=reject)
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(value: dict, key: str):
    require(isinstance(value, dict) and value.get(key) == canonical_hash({k: v for k, v in value.items() if k != key}),
            f"Invalid {key}")


def implementation_hashes() -> dict:
    root = Path(__file__).parent.parent
    names = ("diagnostics/observability.py", "diagnostics/observability_protocol.py",
             "diagnostics/observability_collect.py", "diagnostics/observability_evidence.py",
             "diagnostics/observability_summary.py", "brain/trainable.py", "brain/trainable_graph.py", "brain/tonic.py",
             "brain/policy_groups.py", "brain/reference.py", "brain/connectome.py", "brain/metadata.py", "senses/spatial_drive.py",
             "learning/resource_guard.py")
    return {name: file_hash(root / name) for name in names}


def template() -> dict:
    return {"schema": SCHEMA, "role": "engineering_input_observability", "device": "cpu", "dtype": "torch.float32",
            "torch_threads": 4, "dt_ms": 0.1, "bin_ms": 1, "theta": 0, "modes": list(MODES),
            "input": {"raw_type": "R1-R6", "map_seed": 1701, "phase_seed": 1702,
                      "max_rate_hz": 150.0, "event_voltage_mv": 68.75},
            "observations": {"superclass": "descending_neuron", "lamina_types": list(LAMINA)},
            "conditions": [{"name": name, "rectangle": list(rect) if rect else None}
                           for name, rect in zip(CONDITIONS, RECTANGLES)],
            "windows_ms": {"precondition": 100, "stimulus": 50, "withdrawal": 50},
            "difference": {"atol_mv": 1e-5, "rtol": 1e-6, "spikes": "exact integer"},
            "resources": {"wall_seconds": 600, "rss_bytes": 6 * GIB,
                          "reserve_bytes": DEFAULT_SYSTEM_RESERVE_BYTES, "poll_seconds": 0.1}}


def validate_protocol(value: dict) -> dict:
    expected = template()
    require(isinstance(value, dict) and set(value) == set(expected), "Wrong protocol keys")
    for key in set(expected) - {"windows_ms", "resources"}:
        # Canonical JSON also distinguishes true from 1 and numbers from strings.
        require(canonical_hash(value[key]) == canonical_hash(expected[key]), f"Unsupported protocol field: {key}")
    windows = value["windows_ms"]
    require(isinstance(windows, dict) and set(windows) == set(expected["windows_ms"]), "Wrong window keys")
    for name, duration in windows.items():
        require(type(duration) in (int, float) and math.isfinite(duration) and 0 < duration <= 1000
                and duration == int(duration), f"{name} must be whole milliseconds in (0, 1000]")
    resources = value["resources"]
    require(isinstance(resources, dict) and set(resources) == set(expected["resources"]), "Wrong resource keys")
    for name, limit in (("wall_seconds", 600), ("poll_seconds", 0.1)):
        v = resources[name]
        require(type(v) in (int, float) and math.isfinite(v) and 0 < v <= limit, f"Resource limit may only tighten: {name}")
    require(type(resources["rss_bytes"]) is int and 0 < resources["rss_bytes"] <= 6 * GIB, "RSS may only tighten")
    require(type(resources["reserve_bytes"]) is int and resources["reserve_bytes"] >= DEFAULT_SYSTEM_RESERVE_BYTES,
            "Reserve may only increase from the 1 GiB minimum")
    normalized = deepcopy(value)
    normalized["windows_ms"] = {name: int(v) for name, v in windows.items()}
    return normalized


def ledger(protocol: dict) -> dict:
    windows = protocol["windows_ms"]
    steps = {name: int(value * 10) for name, value in windows.items()}
    per_mode = steps["precondition"] + len(CONDITIONS) * (steps["stimulus"] + steps["withdrawal"])
    return {"phase_steps": steps, "per_mode_steps": per_mode, "total_steps": len(MODES) * per_mode,
            "cumulative_simulated_ms": len(MODES) * per_mode / 10}


def masks() -> dict[str, np.ndarray]:
    result = {}
    for name, rect in zip(CONDITIONS, RECTANGLES):
        mask = np.zeros((12, 16), dtype=np.float64)
        if rect:
            top, bottom, left, right = rect
            mask[top:bottom, left:right] = 1
        result[name] = mask
    return result


def validate_registration(value: dict) -> dict:
    require(value.get("schema") == REGISTRATION_SCHEMA, "Wrong registration schema")
    identity(value, "registration_id")
    protocol = validate_protocol(value["protocol"])
    require(protocol == value["protocol"] and canonical_hash(protocol) == value["protocol_id"], "Protocol hash mismatch")
    require(value["ledger"] == ledger(protocol), "Registered step ledger mismatch")
    require(value["implementation_sha256"] == implementation_hashes(), "Registered source changed")
    require(isinstance(value["graph_path"], str) and Path(value["graph_path"]).is_absolute(), "Graph path must be absolute")
    require(datetime.fromisoformat(value["registered_utc"]).utcoffset() is not None, "Missing registration timezone")
    budget = value["budget"]
    resources = protocol["resources"]
    require(type(budget["available_bytes_before_launch"]) is int and budget["available_bytes_before_launch"] >= 0,
            "Invalid available memory")
    limit = min(resources["rss_bytes"], max(0, budget["available_bytes_before_launch"] - resources["reserve_bytes"]))
    require(budget == {"available_bytes_before_launch": budget["available_bytes_before_launch"],
                       "requested_rss_limit_bytes": resources["rss_bytes"], "reserve_bytes": resources["reserve_bytes"],
                       "effective_rss_limit_bytes": limit, "can_launch": limit > 0}, "Budget mismatch")
    return protocol
