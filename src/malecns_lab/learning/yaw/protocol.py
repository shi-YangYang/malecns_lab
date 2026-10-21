"""Immutable independent yaw experiment registration; no inherited old run."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import shutil
import time

from ...brain.metadata import canonical_hash
from ...drones.yaw_task import scenes
from ..resource_guard import write_json

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "malecns_lab.yaw_learning.v1"
PARAMETERS = ("brain.theta", "readout.weight", "readout.bias")
MODES = ("zero", "untrained", "frozen", "joint", "visual_p")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected JSON object")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_protocol():
    return {"schema": SCHEMA, "control_dt_s": .05, "snn_dt_ms": .1, "steps_per_frame": 500,
            "warmup_frames": 40, "baseline_frames": 20, "frames_per_window": 2, "windows": 80,
            "credit_horizon_steps": 100, "checkpoint_steps": 50, "action_limit_rad_s": .35,
            "teacher": "clip(-2*previous_observed_yaw_increment/.05,-.35,.35)",
            "loss": "mean(softplus(2*z)-(1+teacher/.35)*z)+1e-4*mean((gains-1)^2)",
            "theta_lr": .001, "readout_lr": .001, "clip_group_norm": 1., "theta_raw_norm_limit": 1000.,
            "resource": {"reserve_bytes": 1024**3, "requested_rss_bytes": 6*1024**3, "wall_seconds": 600, "sampling_seconds": .1},
            "scenes": {split: [asdict(scene) for scene in scenes(split)] for split in ("train", "validation", "test")},
            "gate": {"mean_improvement": .20, "direction_improvement_min": 0., "max_case_degradation": .05,
                     "peak_ratio_max": 1.10, "tail_rms_max": .03, "static_rms_max": .02, "static_peak_max": .05}}


def source_files():
    # Bind only numerical/persistence dependencies, so a later browser change
    # cannot invalidate the checkpoint while an actual mathematics edit does.
    base = ROOT / "src/malecns_lab"
    paths = []
    for folder in ("brain", "senses", "learning/yaw"):
        paths.extend((base / folder).rglob("*.py"))
    for relative in ("drones/yaw_env.py", "drones/yaw_task.py", "control/body.py", "learning/resource_guard.py",
                     "learning/train_snn.py", "learning/mps_memory.py"):
        paths.append(base / relative)
    return sorted(set(paths))


def register(directory, graph, mode):
    directory, graph = Path(directory).resolve(), Path(graph).resolve()
    require(mode in ("joint", "frozen"), "Choose joint or frozen")
    require(not directory.exists(), "Run directory already exists; old runs are never overwritten")
    require(graph.is_file(), "Missing full reference asset")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    family = {"protocol": frozen_protocol(), "graph": {"path": str(graph), "sha256": sha256(graph)}, "sources": sources}
    run = {"schema": SCHEMA, "family_id": canonical_hash(family), **family, "mode": mode}
    run["run_id"] = canonical_hash(run)
    directory.mkdir(parents=True)
    for name in sources:
        destination = directory / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(directory / "run.json", run)
    write_json(directory / "registration.json", {"created_unix_s": time.time(), "run_id": run["run_id"],
                                                "status": "registered", "model_executed": False})
    return run


def validate_run(directory):
    directory = Path(directory).resolve()
    run = read_json(directory / "run.json")
    identity = dict(run)
    require(identity.pop("run_id", None) == canonical_hash(identity), "Run identity changed")
    require(run["schema"] == SCHEMA and run["mode"] in ("joint", "frozen"), "Foreign run")
    require(run["protocol"] == frozen_protocol(), "Frozen protocol differs; register a new version")
    family = {key: run[key] for key in ("protocol", "graph", "sources")}
    require(canonical_hash(family) == run["family_id"], "Run family differs")
    require(all((ROOT / name).is_file() and sha256(ROOT / name) == digest for name, digest in run["sources"].items()), "Numerical source files changed")
    for name, digest in run["sources"].items():
        require(sha256(directory / "sources" / name) == digest, "Archived source differs")
    require(sha256(run["graph"]["path"]) == run["graph"]["sha256"], "Reference asset changed")
    return run
