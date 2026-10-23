"""Bounded X2 multi-behavior training and replay evaluation, new experiment IDs.

Preview/registration helpers are stdlib-only. The supervising parent owns the
single complete-brain lock. Only a claimed worker imports Torch or the model.
Training never opens locked-test data and no teacher is callable by the policy.
"""
from __future__ import annotations

from datetime import datetime, timezone
import ast
import fcntl
from importlib.metadata import version
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_behavior_training.v1"
CHECKPOINT_SCHEMA = "malecns_lab.x2_behavior_checkpoint.v1"
PROBE_MODES = ("probe", "probe-stabilized")
PAIR_DIRECTORY = "data/x2_behavior_probe_20261023_r1"
PAIR_POLICY_ID = "b8cb7e89613e73071c66b9c3a01007b92c0002187154b56ca49fe3414bd44a55"
PAIR_CHECKPOINT_ID = "66c4717399a9f5319252f1f0b56ea2665c8a0afda02ab4af412c134604957d57"
SOURCE_FILES = (
    "__init__.py", "brain/__init__.py", "control/__init__.py", "senses/__init__.py", "learning/__init__.py",
    "experiments/__init__.py", "experiments/x2_behavior_training.py", "learning/x2_behavior_policy.py",
    "learning/x2_behavior_training.py", "learning/x2_behavior_data.py", "senses/x2_behavior.py",
    "control/x2_behavior.py", "learning/x2_inference.py", "learning/resource_guard.py", "learning/mps_memory.py",
    "brain/mps_trainable.py", "brain/mps_identity.py", "brain/trainable.py", "brain/trainable_graph.py",
    "brain/policy_groups.py", "brain/metadata.py", "brain/reference.py", "brain/tonic.py",
)


def contract(mode="probe", *, seed=2401, max_updates=None, train_episodes=5,
             validation_episodes=5, wall_seconds=None, initialization="migrated", spike_backward_gain=None):
    frozen.require(mode in (*PROBE_MODES, "readout", "joint", "evaluate"), "Unknown experiment mode")
    frozen.require(initialization in ("migrated", "neutral"), "Unknown initialization")
    max_updates = {"probe": 1, "probe-stabilized": 1, "readout": 20, "joint": 4, "evaluate": 0}[mode] if max_updates is None else max_updates
    wall_seconds = {"probe": 600, "probe-stabilized": 600, "readout": 1800, "joint": 1800, "evaluate": 7200}[mode] if wall_seconds is None else wall_seconds
    gain = (0.1 if mode == "probe-stabilized" else 1.0) if spike_backward_gain is None else spike_backward_gain
    frozen.require(type(gain) in (int, float) and gain in (1.0, 0.1), "Unregistered spike backward gain")
    frozen.require(type(seed) is int and 0 <= seed < 2**32, "Invalid seed")
    frozen.require(type(max_updates) is int and 0 <= max_updates <= 200
                   and (mode == "evaluate") == (max_updates == 0), "Invalid update bound")
    frozen.require(type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and 0 < wall_seconds <= 7200,
                   "Wall budget must be finite and at most 7200 seconds")
    frozen.require(type(train_episodes) is int and 5 <= train_episodes <= 200
                   and type(validation_episodes) is int and 5 <= validation_episodes <= 50, "Invalid episode budget")
    if mode in PROBE_MODES:
        frozen.require(max_updates == 1 and seed == 2401 and wall_seconds == 600
                       and initialization == "migrated" and gain == (0.1 if mode == "probe-stabilized" else 1.0),
                       "The registered probe is fixed")
    return {"schema": SCHEMA, "mode": mode, "seed": seed, "initialization": initialization,
            "max_updates": max_updates, "train_episodes": train_episodes,
            "validation_episodes": validation_episodes, "wall_seconds": wall_seconds,
            "precondition_steps": 600, "decision_steps": 500, "dt_ms": .1,
            "checkpoint_block_steps": 50, "credit_decisions": 2, "credit_steps": 1000,
            "spike_backward_gain": float(gain), "loss_backward_scale": 1.0,
            "backward": "fixed per-spike surrogate derivative multiplier; binary forward and surrogate width unchanged",
            "paired_reference": PAIR_DIRECTORY if mode == "probe-stabilized" else None,
            "probe": "train[0] first two observations; loss only on second; retain first-boundary v gradient; exactly one update",
            "optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": 0., "amsgrad": False},
            "loss": {"behavior_ce": 1., "phase_ce": 1., "active_parameter_mse": .1,
                     "parameter_mask": "target behavior ADVANCE/TURN_LEFT/TURN_RIGHT, target phase START/CONTINUE"},
            "theta_gain": "unchanged MPSLIF 1+0.5*tanh(theta), float32 open-bound clipping",
            "heads": "CPU torch.nn.Linear default initialization seed2401 in behavior5/phase3/parameter1 order",
            "state": "reset per whole episode; persistent across decisions; detach only after 2-decision update window",
            "selection": "deterministic task-round-robin episode prefix; validation loss only; lowest loss, earliest tie",
            "validation": "readout: cached full validation episodes each update; joint: full validation only after final bounded update",
            "locked_test": "only evaluate mode; all 100 sealed episodes; never checkpoint selection",
            "controls": "same checkpoint and recorded DN features: per-episode first-feature constant and reversed time",
            "resource": {"rss_cap_gib": 6, "mps_cap": "same effective RSS byte budget", "system_reserve_gib": 1,
                         "cpu_threads": 4, "interop_threads": 1, "poll_seconds": .1, "fallback": "0"},
            "claim": "offline teacher agreement; simulated feedback cannot prove P3 real-feedback task success or P5 motion"}


def select_episodes(episodes, count):
    """Stratified deterministic prefix; no row labels or scores drive selection."""
    names = ("GREET", "FACE_LEFT", "FACE_RIGHT", "SHOW_FORWARD", "END_INTERACTION")
    groups = {name: [e for e in episodes if e["task"] == name] for name in names}
    frozen.require(all(groups.values()), "All five tasks are required")
    ordered = [groups[name][index] for index in range(max(map(len, groups.values())))
               for name in names if index < len(groups[name])]
    frozen.require(0 < count <= len(ordered), "Episode selection exceeds split")
    return ordered[:count]


def source_inventory(root):
    """Hash transitive local imports including package entrypoints, without imports."""
    base = Path(root) / "src/malecns_lab"
    pending, found = list(SOURCE_FILES), {}
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = base / relative
        frozen.require(path.is_file(), f"Missing registered source: {relative}")
        found[relative] = frozen.sha256(path)
        parts = Path(relative).with_suffix("").parts
        package = parts[:-1]
        for depth in range(len(package)+1):
            init = str(Path(*package[:depth], "__init__.py"))
            if (base / init).is_file() and init not in found:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    stem = package[:len(package)-node.level+1]
                    suffix = tuple((node.module or "").split(".")) if node.module else ()
                    modules = [(*stem, *suffix)] if suffix else [(*stem, alias.name) for alias in node.names]
                elif node.module and node.module.startswith("malecns_lab."):
                    modules = [tuple(node.module.split(".")[1:])]
            elif isinstance(node, ast.Import):
                modules = [tuple(a.name.split(".")[1:]) for a in node.names if a.name.startswith("malecns_lab.")]
            for module in modules:
                for candidate in (Path(*module).with_suffix(".py"), Path(*module, "__init__.py")):
                    if (base / candidate).is_file() and str(candidate) not in found:
                        pending.append(str(candidate))
    return dict(sorted(found.items()))


def _inside_new_output(output, root):
    output = Path(output).resolve()
    frozen.require(output.is_relative_to(root / "data") and output != root / "data", "Run output must be a new project data/ directory")
    output.mkdir(parents=True, exist_ok=False)
    return output


def execute(output, *, configuration=None, checkpoint_path=None, feature_cache=None, root=frozen.ROOT):
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_data import write_dataset
    import psutil
    cfg = configuration or contract()
    frozen.require(cfg == contract(cfg["mode"], seed=cfg["seed"], max_updates=cfg["max_updates"],
        train_episodes=cfg["train_episodes"], validation_episodes=cfg["validation_episodes"],
        wall_seconds=cfg["wall_seconds"], initialization=cfg["initialization"],
        spike_backward_gain=cfg["spike_backward_gain"]), "Unregistered configuration fields")
    frozen.require(cfg["mode"] != "evaluate" or checkpoint_path is not None, "Evaluation requires a selected new checkpoint")
    frozen.require(cfg["mode"] not in PROBE_MODES or checkpoint_path is None, "Probe uses authenticated historical theta initialization")
    root = Path(root).resolve()
    if feature_cache is not None:
        from .x2_behavior_cache import validate_request
        validate_request(cfg, checkpoint_path, feature_cache, root)
    output = _inside_new_output(output, root)
    started = time.monotonic()
    result = {"schema": SCHEMA, "mode": cfg["mode"], "status": "failed", "reason": None,
              "output": str(output), "behavior_pass": None}
    try:
        with frozen.numerical_lock(root):
            write_json(output / "phase.json", {"phase": "preflight"})
            write_json(output / "simulator_preflight.json", frozen.simulator_preflight(root))
            anchors = frozen.read_anchors(root)
            (output / "provenance").mkdir()
            for name, value in anchors.items():
                write_json(output / "provenance" / f"{name}.json", value)
            sources = source_inventory(root)
            for name in sources:
                target = output / "sources" / "malecns_lab" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "src/malecns_lab" / name, target)
                target.chmod(0o444)
            dataset = write_dataset(output / "dataset", seed=cfg["seed"]) if feature_cache is None else None
            paired_probe = prepare_paired_probe(root) if cfg["mode"] == "probe-stabilized" else None
            if paired_probe is not None:
                frozen.require(dataset == frozen.read_json(root / PAIR_DIRECTORY / "registration.json")["dataset"],
                               "Paired probe dataset identity changed")
            resources = resource_budget(psutil.virtual_memory().available, rss_bytes=6*frozen.GIB, reserve_bytes=frozen.GIB)
            checkpoint = None
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path).resolve()
                frozen.require(checkpoint_path.is_relative_to(root / "data"), "Checkpoint must be archived under project data/")
                meta = inspect_checkpoint(checkpoint_path)
                frozen.require(meta["policy"].get("backward", {}).get("spike_backward_gain", 1.0) == cfg["spike_backward_gain"],
                               "Selected checkpoint backward identity differs")
                checkpoint = {"path": str(checkpoint_path), "checkpoint_id": meta["checkpoint_id"],
                              "metadata_sha256": frozen.sha256(checkpoint_path / "checkpoint.json"),
                              "arrays_sha256": meta["arrays_sha256"]}
            reg = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "configuration": cfg, "source_sha256": sources, "dataset": dataset, "paired_probe": paired_probe,
                   "source_checkpoint_id": frozen.CHECKPOINT_ID, "checkpoint": checkpoint,
                   "graph_sha256": anchors["run"]["graph"]["file_sha256"],
                   "graph_path": str(root / frozen.REFERENCE), "source_policy_id": anchors["run"]["policy"]["policy_id"],
                   "runtime": {"python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
                       "torch": version("torch"), "machine": platform.machine(), "system": platform.system(),
                       "backend": "X2_MPSLIF_with_explicit_spike_backward_gain", "spike_backward_gain": cfg["spike_backward_gain"],
                       "device": "mps", "dtype": "float32",
                       "cpu_threads": 4, "interop_threads": 1, "fallback": "0"},
                   "root": str(root), "output": str(output), "parent_pid": os.getpid(),
                   "parent_started_monotonic": started, "resources": resources}
            if feature_cache is not None:
                from .x2_behavior_cache import audit_cache
                def cache_progress():
                    frozen.require(time.monotonic()-started < cfg["wall_seconds"], "Cache admission exceeded wall budget")
                cache_manifest, _, dataset_bytes = audit_cache(root, feature_cache, configuration=cfg,
                    checkpoint=checkpoint, sources=sources, runtime=reg["runtime"], progress=cache_progress)
                write_json(output / "cache_manifest.json", cache_manifest)
                (output / "dataset").mkdir()
                for name, content in dataset_bytes.items():
                    with (output / "dataset" / name).open("xb") as stream:
                        stream.write(content)
                reg["dataset"] = frozen.read_json(output / "dataset/manifest.json")
                reg["feature_cache"] = {"schema": cache_manifest["schema"], "cache_root": cache_manifest["cache_root"],
                    "cache_id": cache_manifest["cache_id"], "manifest_sha256": frozen.sha256(output / "cache_manifest.json"),
                    "source_registration_id": cache_manifest["source_registration_id"], "initial_checkpoint_id": checkpoint["checkpoint_id"]}
            reg["registration_id"] = frozen.canonical_hash(reg)
            write_json(output / "registration.json", reg)
            frozen.require(resources["can_launch"], "Insufficient system reserve")
            remaining = cfg["wall_seconds"] - (time.monotonic() - started)
            frozen.require(remaining > 0, "Preparation exceeded wall budget")
            supervisor = supervise([sys.executable, "-m", "malecns_lab.experiments.x2_behavior_training", "--_worker", str(output)],
                output, rss_limit_bytes=resources["effective_rss_limit_bytes"], reserve_bytes=frozen.GIB,
                wall_seconds=remaining, poll_seconds=.1)
            result["supervisor"] = supervisor
            worker = frozen.read_json(output / "worker_result.json") if (output / "worker_result.json").exists() else {}
            frozen.require(supervisor["status"] == "completed" and worker.get("status") == "completed"
                           and worker.get("registration_id") == reg["registration_id"],
                           f"Incomplete run: {supervisor['reason']}; {worker.get('reason', 'no worker result')}")
            frozen.require(0 <= worker["optimizer_steps"] <= cfg["max_updates"], "Optimizer exceeded registered limit")
            if cfg["mode"] != "evaluate":
                frozen.require(worker["optimizer_steps"] == cfg["max_updates"], "Registered updates are incomplete")
            result.update(status="completed", reason="bounded_run_sealed", worker=worker,
                          registration_id=reg["registration_id"], behavior_pass=worker.get("behavior_pass"))
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", result)
    return result


def inspect_checkpoint(directory):
    directory = Path(directory)
    metadata = frozen.read_json(directory / "checkpoint.json")
    frozen.bound(metadata, "checkpoint_id")
    frozen.bound(metadata["policy"], "policy_id")
    frozen.require(metadata["schema"] == CHECKPOINT_SCHEMA
                   and frozen.sha256(directory / "arrays.npz") == metadata["arrays_sha256"], "Checkpoint integrity differs")
    return metadata


def prepare_paired_probe(root):
    """Authenticate the fixed failed run without touching its sealed artifacts."""
    base = Path(root) / PAIR_DIRECTORY
    registration = frozen.read_json(base / "registration.json")
    frozen.bound(registration, "registration_id")
    metadata = inspect_checkpoint(base / "initial-checkpoint")
    frozen.require(metadata["checkpoint_id"] == PAIR_CHECKPOINT_ID
                   and metadata["policy"]["policy_id"] == PAIR_POLICY_ID
                   and registration["configuration"]["mode"] == "probe", "Paired probe anchor differs")
    names = ["registration.json", "policy.json", "dataset/train.json", "dataset/manifest.json",
             "initial-checkpoint/checkpoint.json", "initial-checkpoint/arrays.npz"]
    for index in range(3):
        names.extend(f"probe/state-{index:04d}.{suffix}" for suffix in ("json", "npz"))
    for index in range(2):
        names.extend(f"probe/frame-{index:04d}.{suffix}" for suffix in ("json", "npz"))
    for name, expected in registration["source_sha256"].items():
        source = f"sources/malecns_lab/{name}"
        frozen.require(frozen.sha256(base / source) == expected, "Paired historical source changed")
        names.append(source)
    return {"path": str(base), "policy_id": PAIR_POLICY_ID, "checkpoint_id": PAIR_CHECKPOINT_ID,
            "registration_id": registration["registration_id"],
            "files": {name: frozen.sha256(base / name) for name in names},
            "comparison": "exact dtype, shape and array bytes; new identities remain distinct"}


def _paired_initialization(policy, paired):
    """Explicit parameter migration across the newly declared backward identity."""
    import numpy as np
    import torch
    from copy import deepcopy
    from .x2_behavior_policy import array_identity, PARAMETERS
    base = Path(paired["path"])
    for name, expected in paired["files"].items():
        frozen.require(frozen.sha256(base / name) == expected, "Paired probe artifact changed")
    meta = inspect_checkpoint(base / "initial-checkpoint")
    old, new = deepcopy(meta["policy"]), deepcopy(policy.identity)
    frozen.require(old["policy_id"] == paired["policy_id"] != new["policy_id"], "Paired policy identities must remain distinct")
    old_model, new_model = old.pop("model"), new.pop("model")
    for model in (old_model, new_model):
        model.pop("model_id")
        model["implementation_sha256"].pop("mps_trainable.py")
    frozen.require(old_model == new_model, "Paired forward graph/configuration differs")
    for key in ("policy_id", "implementation_sha256"):
        old.pop(key)
        new.pop(key)
    frozen.require(new.pop("backward")["spike_backward_gain"] == 0.1, "Paired experiment needs the fixed new backward")
    new.pop("mps_step_sha256")
    frozen.require(old == new, "Paired neural mapping/head/initialization contract differs")
    with np.load(base / "initial-checkpoint/arrays.npz", allow_pickle=False) as archive:
        frozen.require(set(archive.files) == {f"parameter.{name}" for name in PARAMETERS}, "Paired parameter inventory differs")
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                key = f"parameter.{name}"
                values = archive[key]
                frozen.require(array_identity(values) == meta["arrays"][key] and np.isfinite(values).all()
                               and values.dtype == np.float32 and values.shape == tuple(parameter.shape), "Paired parameter declaration differs")
                parameter.copy_(torch.from_numpy(values.copy()).to(parameter.device))
                frozen.require(array_identity(parameter.detach().cpu().numpy()) == meta["arrays"][key], "Paired parameter transfer differs")
    policy.validate_parameters()
    return {"source_policy_id": paired["policy_id"], "source_checkpoint_id": paired["checkpoint_id"],
            "new_policy_id": policy.identity["policy_id"], "parameters": meta["arrays"],
            "method": "explicit exact parameter copy, new optimizer; no old identity restoration"}


def compare_paired_artifact(reference, actual, report, *, policy_id, report_path):
    """Compare saved forward arrays in the same run; persist before rejecting."""
    import numpy as np
    from .resource_guard import write_json
    from .x2_behavior_policy import array_identity
    reference, actual = Path(reference), Path(actual)
    old, new = frozen.read_json(reference.with_suffix(".json")), frozen.read_json(actual.with_suffix(".json"))
    state = "state_id" in old
    hash_key = "file_sha256" if state else "arrays_sha256"
    frozen.require(frozen.sha256(reference) == old[hash_key] and frozen.sha256(actual) == new[hash_key], "Paired artifact integrity differs")
    old_policy = old["policy_id"] if state else old["decision"]["policy_id"]
    new_policy = new["policy_id"] if state else new["decision"]["policy_id"]
    checks = {"distinct_policy_ids": old_policy != new_policy == policy_id}
    if state:
        for key in ("schema", "session_id", "frame_index", "observation_seq", "neural_step"):
            checks[key] = old[key] == new[key]
    else:
        checks["observation"] = old["observation"] == new["observation"]
    arrays = {}
    with np.load(reference, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        checks["array_inventory"] = len(left.files) == len(set(left.files)) == len(right.files) == len(set(right.files)) and set(left.files) == set(right.files)
        for name in sorted(set(left.files) | set(right.files)):
            lvalue, rvalue = left[name] if name in left else None, right[name] if name in right else None
            left_id = array_identity(lvalue) if lvalue is not None else None
            right_id = array_identity(rvalue) if rvalue is not None else None
            arrays[name] = {"reference": left_id, "actual": right_id,
                            "exact": lvalue is not None and rvalue is not None and left_id == right_id}
            if state:
                checks[f"declared:{name}"] = old["arrays"].get(name) == left_id and new["arrays"].get(name) == right_id
    item = {"reference": str(reference), "actual": str(actual), "reference_policy_id": old_policy,
            "actual_policy_id": new_policy, "checks": checks, "arrays": arrays,
            "exact": all(checks.values()) and all(value["exact"] for value in arrays.values())}
    report.setdefault("artifacts", []).append(item)
    report["all_exact"] = all(value["exact"] for value in report["artifacts"])
    write_json(report_path, report)
    frozen.require(item["exact"], "Paired forward artifact differs; update rejected")
    return item


class BoundaryGradientAudit:
    """Stream boundary adjoints; hooks return the exact incoming gradient."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records, self.handles = {}, []

    def observe(self, state):
        import numpy as np
        from .resource_guard import write_json
        values = {"v": state.v, "g": state.g, **{f"pending.{i:02d}": value for i, value in enumerate(state.pending)}}
        for name, value in values.items():
            key = f"{state.step:06d}.{name}"
            frozen.require(key not in self.records, "Duplicate gradient boundary")
            self.records[key] = {"neural_step": state.step, "tensor": name, "requires_grad": value.requires_grad,
                                 "gradient": _gradient(None)}
            if not value.requires_grad:
                continue
            def capture(gradient, key=key):
                record = self.records[key]
                record["gradient"] = _gradient(gradient)
                path = self.directory / f"{key}.npy"
                with path.open("xb") as stream:
                    np.save(stream, gradient.detach().cpu().numpy(), allow_pickle=False)
                record["file"] = path.name
                record["sha256"] = frozen.sha256(path)
                write_json(self.directory / f"{key}.json", record)
                return gradient
            self.handles.append(value.register_hook(capture))

    def finish(self):
        from .resource_guard import write_json
        result = {"schema": "malecns_lab.x2_boundary_gradients.v1", "records": self.records,
                  "note": "absent gradient means no traversed loss path; final pending/g need not affect this loss",
                  "observed_finite": all(value["gradient"]["finite"] for value in self.records.values() if value["gradient"]["present"])}
        write_json(self.directory / "summary.json", result)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return result


def save_checkpoint(directory, policy, context, *, optimizer=None):
    import numpy as np
    from .resource_guard import write_json
    from .x2_behavior_policy import array_identity
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    values = {f"parameter.{name}": p.detach().cpu().numpy().copy() for name, p in policy.named_parameters()}
    if optimizer is not None:
        for name, p in policy.named_parameters():
            for key, value in optimizer.state.get(p, {}).items():
                values[f"optimizer.{name}.{key}"] = value.detach().cpu().numpy().copy()
    with (directory / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **values)
    payload = {"schema": CHECKPOINT_SCHEMA, "policy": policy.identity, "context": context,
               "arrays": {name: array_identity(value) for name, value in values.items()},
               "arrays_sha256": frozen.sha256(directory / "arrays.npz")}
    payload["checkpoint_id"] = frozen.canonical_hash(payload)
    write_json(directory / "checkpoint.json", payload)
    return payload


def restore_checkpoint(directory, policy):
    """Strict non-pickle restoration. Training starts a fresh optimizer."""
    import numpy as np
    import torch
    from .x2_behavior_policy import array_identity, PARAMETERS
    meta = inspect_checkpoint(directory)
    frozen.require(meta["policy"] == policy.identity, "Checkpoint model/input/source identity differs")
    with np.load(Path(directory) / "arrays.npz", allow_pickle=False) as archive:
        frozen.require(len(archive.files) == len(set(archive.files)) and set(archive.files) == set(meta["arrays"]), "Checkpoint array inventory differs")
        expected = {f"parameter.{name}" for name in PARAMETERS}
        frozen.require({name for name in archive.files if name.startswith("parameter.")} == expected, "Checkpoint parameter inventory differs")
        for key in archive.files:
            value = archive[key]
            frozen.require(array_identity(value) == meta["arrays"][key] and np.isfinite(value).all(), "Checkpoint array declaration differs")
        with torch.no_grad():
            for name, p in policy.named_parameters():
                value = archive[f"parameter.{name}"]
                frozen.require(value.dtype == np.float32 and value.shape == tuple(p.shape), "Parameter shape/dtype differs")
                p.copy_(torch.from_numpy(value.copy()).to(p.device))
                frozen.require(np.array_equal(p.cpu().numpy(), value), "Checkpoint transfer changed parameter")
    policy.validate_parameters()
    return meta


def loss_for(output, target):
    import torch
    from torch.nn import functional as F
    from ..control.x2_behavior import BEHAVIORS, PHASES
    device = output.behavior_logits.device
    b = tuple(str(x) for x in BEHAVIORS).index(target["behavior"])
    p = tuple(str(x) for x in PHASES).index(target["phase"])
    ce_behavior = F.cross_entropy(output.behavior_logits[None], torch.tensor([b], device=device))
    ce_phase = F.cross_entropy(output.phase_logits[None], torch.tensor([p], device=device))
    active = target["behavior"] in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT") and target["phase"] in ("START", "CONTINUE")
    param = (output.intensity[0] - float(target["intensity"]))**2 if active else output.intensity[0]*0
    return ce_behavior + ce_phase + .1*param


def _prediction(heads):
    from ..control.x2_behavior import BEHAVIORS, PHASES
    behavior, phase, raw, intensity = heads
    return {"behavior": str(BEHAVIORS[int(behavior.detach().argmax().cpu())]),
            "phase": str(PHASES[int(phase.detach().argmax().cpu())]),
            "intensity": float(intensity.detach().cpu()[0]),
            "behavior_logits": behavior.detach().cpu().tolist(), "phase_logits": phase.detach().cpu().tolist(),
            "parameter_raw": raw.detach().cpu().tolist()}


def _gradient(value):
    if value is None:
        return {"present": False, "finite": False, "nonzero": 0, "finite_nonzero": 0,
                "elements": 0, "nonfinite_counts": {"nan": 0, "positive_inf": 0, "negative_inf": 0},
                "l2": None, "max_abs": None, "finite_l2": None, "finite_max_abs": None}
    import numpy as np
    values = value.detach().cpu().numpy()
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64)
    maximum = float(np.abs(finite_values).max(initial=0.))
    # Scale on the CPU so large finite float32 gradients do not overflow their norm.
    norm = maximum * float(np.linalg.norm(finite_values / maximum)) if maximum else 0.
    norm = norm if math.isfinite(norm) else None
    all_finite = bool(finite.all())
    return {"present": True, "finite": all_finite, "elements": int(values.size),
            "nonzero": int(np.count_nonzero(values)), "finite_nonzero": int(np.count_nonzero(finite_values)),
            "nonfinite_counts": {"nan": int(np.isnan(values).sum()), "positive_inf": int(np.isposinf(values).sum()),
                                 "negative_inf": int(np.isneginf(values).sum())},
            "l2": norm if all_finite else None, "max_abs": maximum if all_finite else None,
            "finite_l2": norm, "finite_max_abs": maximum}


def _audited_gradient_update(policy, optimizer, loss, *, directory, result, max_updates,
                             boundary=None, required_nonzero=(), expected_zero=(),
                             boundary_audit=None, check_adam_scale=False):
    """Persist evidence before admission or mutation, including rejected updates."""
    import numpy as np
    from .resource_guard import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    parameters = dict(policy.named_parameters())
    before = {name: p.detach().cpu().numpy().copy() for name, p in parameters.items()}
    loss_value = float(loss.detach().cpu())
    audit = {"status": "in_progress", "stage": "before_backward", "reason": None,
             "update": result["optimizer_steps"] + 1, "loss": loss_value if math.isfinite(loss_value) else None,
             "loss_finite": math.isfinite(loss_value), "backward_completed": False,
             "optimizer_steps_before": result["optimizer_steps"], "optimizer_steps_after": result["optimizer_steps"],
             "optimizer_step_attempted": False, "optimizer_step_completed": False,
             "parameter_changed_counts": {name: 0 for name in parameters}}
    result["last_update_audit"] = str(directory / "audit.json")

    def save_arrays(name, arrays):
        path = directory / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return {"file": name, "sha256": frozen.sha256(path),
                "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in arrays.items()}}

    def save_gradients():
        if "gradient_arrays" in audit:
            return
        gradients = {name: p.grad for name, p in parameters.items() if p.requires_grad}
        audit["parameter_gradients"] = {name: _gradient(value) for name, value in gradients.items()}
        audit["first_boundary_v_gradient"] = _gradient(boundary.grad) if boundary is not None else None
        arrays = {f"parameter_before.{name}": value for name, value in before.items()}
        arrays.update({f"gradient.{name}": value.detach().cpu().numpy().copy()
                       for name, value in gradients.items() if value is not None})
        if boundary is not None and boundary.grad is not None:
            arrays["first_boundary_v_gradient"] = boundary.grad.detach().cpu().numpy().copy()
        audit["gradient_arrays"] = save_arrays("gradients.npz", arrays)
        if boundary_audit is not None:
            audit["boundary_gradients"] = boundary_audit.finish()
        if check_adam_scale:
            # Fresh Adam in this fixed probe. Bound squaring before mutation;
            # a finite float32 gradient alone is insufficient for finite Adam.
            limit = math.sqrt(float(np.finfo(np.float32).max))
            audit["adam_scale"] = {"fresh_optimizer": not optimizer.state,
                "max_abs_limit": limit, "rule": "all raw gradients safely square in float32; no rescaling/clipping",
                "safe": all(value["finite"] and value["max_abs"] <= limit
                            for value in audit["parameter_gradients"].values())}

    def measure_changes():
        after = {name: p.detach().cpu().numpy().copy() for name, p in parameters.items()}
        audit["parameter_changed_counts"] = {name: int(np.count_nonzero(before[name] != value))
                                              for name, value in after.items()}
        return after

    write_json(directory / "audit.json", audit)
    try:
        frozen.require(result["optimizer_steps"] < max_updates and audit["loss_finite"], "Update bound or loss invalid")
        audit["stage"] = "backward"
        write_json(directory / "audit.json", audit)
        loss.backward()
        audit["backward_completed"] = True
        save_gradients()
        audit["stage"] = "gradient_admission"
        write_json(directory / "audit.json", audit)
        grads, cross = audit["parameter_gradients"], audit["first_boundary_v_gradient"]
        frozen.require(all(g["present"] and g["finite"] for g in grads.values()), "Missing/nonfinite trainable gradient")
        if cross is not None:
            frozen.require(cross["present"] and cross["finite"] and cross["nonzero"] > 0,
                           "Missing, zero or nonfinite cross-decision gradient")
        frozen.require(all(grads[name]["nonzero"] > 0 for name in required_nonzero), "Probe expected nonzero gradient is zero")
        frozen.require(all(grads[name]["nonzero"] == 0 for name in expected_zero), "Probe masked gradient is nonzero")
        if boundary_audit is not None:
            frozen.require(audit["boundary_gradients"]["observed_finite"], "Nonfinite intermediate boundary gradient")
        if check_adam_scale:
            frozen.require(audit["adam_scale"]["fresh_optimizer"] and audit["adam_scale"]["safe"], "Unsafe float32 Adam gradient scale")
        audit["stage"], audit["optimizer_step_attempted"] = "optimizer_step", True
        write_json(directory / "audit.json", audit)
        optimizer.step()
        result["optimizer_steps"] += 1
        audit["optimizer_step_completed"] = True
        audit["optimizer_steps_after"] = result["optimizer_steps"]
        audit["stage"] = "parameter_validation"
        measure_changes()
        write_json(directory / "audit.json", audit)
        policy.validate_parameters()
        if check_adam_scale:
            import torch
            audit["adam_state_finite"] = all(torch.isfinite(value).all().item()
                for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value))
            frozen.require(audit["adam_state_finite"], "Nonfinite Adam state after update")
        changes = audit["parameter_changed_counts"]
        frozen.require(all(changes[name] > 0 for name in required_nonzero), "Probe expected update did not occur")
        frozen.require(all(changes[name] == 0 for name in expected_zero), "Probe masked parameter changed")
        audit.update(status="completed", stage="completed")
    except BaseException as error:
        audit.update(status="failed", reason=f"{type(error).__name__}:{error}")
        raise
    finally:
        # Even backward/step exceptions retain partial gradients and actual changes.
        save_gradients()
        audit["parameters_after"] = save_arrays("parameters-after.npz", measure_changes())
        audit["optimizer_steps_after"] = result["optimizer_steps"]
        write_json(directory / "audit.json", audit)
    return audit


def _save_state(directory, name, policy, state):
    import numpy as np
    from .resource_guard import write_json
    arrays = policy.state_arrays(state)
    identity = policy.state_identity(state, arrays)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(directory / f"{name}.json", {**identity, "file_sha256": frozen.sha256(path)})
    return identity["state_id"]


def _save_output(directory, index, observation, output, policy, before_id, checkpoint_id):
    import numpy as np
    from .resource_guard import write_json
    arrays = {"encoded_input": policy.encoder.encode(observation),
              "features_mv": output.features_mv.detach().cpu().numpy().copy(),
              "behavior_logits": output.behavior_logits.detach().cpu().numpy().copy(),
              "phase_logits": output.phase_logits.detach().cpu().numpy().copy(),
              "parameter_raw": output.parameter_raw.detach().cpu().numpy().copy()}
    path = directory / f"frame-{index:04d}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    decision = policy.decision(observation, output, checkpoint_id=checkpoint_id,
                               state_before_id=before_id, decision_seq=index)
    write_json(directory / f"frame-{index:04d}.json", {"observation": observation.to_dict(),
               "decision": decision.to_dict(), "arrays_sha256": frozen.sha256(path),
               "host_monotonic": time.monotonic()})
    return _prediction((output.behavior_logits, output.phase_logits, output.parameter_raw, output.intensity))


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).resolve()
    reg = frozen.read_json(directory / "registration.json")
    frozen.bound(reg, "registration_id")
    frozen.require(os.getppid() == reg["parent_pid"] and str(directory) == reg["output"], "Worker needs supervising parent")
    root, cfg = Path(reg["root"]), reg["configuration"]
    with (root / "data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must hold the shared complete-brain lock")
    with (directory / "worker.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    started, sampler, restore_neural_guard = time.monotonic(), None, None
    result = {"status": "failed", "reason": None, "mode": cfg["mode"], "registration_id": reg["registration_id"],
              "optimizer_steps": 0, "behavior_pass": None, "opened_splits": []}
    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic()-started, **details})
        if sampler is not None:
            sampler.sample(name)
    try:
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected
                           and frozen.sha256(directory / "sources/malecns_lab" / name) == expected, "Registered source changed")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from .mps_memory import MPSMemorySampler
        from .x2_behavior_policy import X2BehaviorPolicy, BehaviorOutput, PARAMETERS, array_identity
        from .x2_behavior_data import load_split, score_predictions
        from ..control.x2_behavior import Observation
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        cached_features, cached_dataset_bytes = None, None
        cache_ref = reg.get("feature_cache")
        if cache_ref is not None:
            from .x2_behavior_cache import audit_cache, forbid_neural_calls
            from ..brain.mps_trainable import MPSLIF
            phase("cache_revalidation")
            frozen.require(frozen.sha256(directory / "cache_manifest.json") == cache_ref["manifest_sha256"], "Sealed cache manifest changed")
            cache_manifest = frozen.read_json(directory / "cache_manifest.json")
            frozen.bound(cache_manifest, "cache_id")
            frozen.require(cache_ref["cache_id"] == cache_manifest["cache_id"] and cache_ref["source_registration_id"] == cache_manifest["source_registration_id"]
                           and cache_ref["initial_checkpoint_id"] == reg["checkpoint"]["checkpoint_id"], "Cache registration binding differs")
            _, cached_features, cached_dataset_bytes = audit_cache(root, cache_ref["cache_root"], configuration=cfg, checkpoint=reg["checkpoint"],
                sources=reg["source_sha256"], runtime=reg["runtime"], expected=cache_manifest, keep_features=True)
            result["feature_source"] = cache_ref
            result["cache_revalidated"] = True
            restore_neural_guard = forbid_neural_calls(X2BehaviorPolicy, MPSLIF, result)
        phase("load_parameters")
        anchors = frozen.read_anchors(root)
        source = anchors["run"]["policy"]
        values = frozen.load_parameters(root / frozen.CHECKPOINT / "arrays.npz", anchors["checkpoint"], source)
        frozen.require(frozen.sha256(root / frozen.REFERENCE) == reg["graph_sha256"], "Reference asset changed")
        phase("load_complete_graph")
        policy = X2BehaviorPolicy.from_reference(root / frozen.REFERENCE, source, theta=values["theta"],
                         initialization=cfg["initialization"], seed=cfg["seed"],
                         spike_backward_gain=cfg["spike_backward_gain"])
        del values
        pairing = None
        if cfg["mode"] == "probe-stabilized":
            frozen.require(reg["paired_probe"] is not None, "Stabilized probe requires sealed first-probe pairing")
            pairing = _paired_initialization(policy, reg["paired_probe"])
            write_json(directory / "paired-forward.json", pairing)
        write_json(directory / "policy.json", policy.identity)
        write_json(directory / "input_map.json", policy.encoder.identity)
        checkpoint_ref = reg["checkpoint"]
        if checkpoint_ref is not None:
            frozen.require(frozen.sha256(Path(checkpoint_ref["path"]) / "checkpoint.json") == checkpoint_ref["metadata_sha256"],
                           "Selected checkpoint changed")
            initial = restore_checkpoint(checkpoint_ref["path"], policy)
        else:
            initial = save_checkpoint(directory / "initial-checkpoint", policy,
                                     {"registration_id": reg["registration_id"], "optimizer_steps": 0, "role": "initialization"})
        initial_id = initial["checkpoint_id"]
        result.update(policy_id=policy.policy_id, input_map_id=policy.encoder.input_map_id, initial_checkpoint_id=initial_id)
        before_theta = policy.brain.theta.detach().cpu().numpy().copy()
        parameters_before = {name: p.detach().cpu().numpy().copy() for name, p in policy.named_parameters()}
        if cache_ref is not None:
            frozen.require(policy.policy_id == cache_manifest["policy_id"] and policy.encoder.input_map_id == cache_manifest["input_map_id"]
                           and initial_id == cache_manifest["initial_checkpoint"]["checkpoint_id"], "Restored policy differs from cached features")
            frozen.require(set(initial["arrays"]) == {"parameter."+name for name in PARAMETERS}, "Recovery initial checkpoint contains optimizer or extra arrays")
            result["initial_parameter_identity"] = {name: array_identity(value) for name, value in parameters_before.items()}
        raw_hash_before = hashlib_counts(policy)
        mode = cfg["mode"]
        policy.brain.theta.requires_grad_(mode in (*PROBE_MODES, "joint"))
        def partition(name):
            frozen.require(name != "locked_test" or mode == "evaluate", "Training cannot read locked-test")
            result["opened_splits"].append(name)
            if cached_dataset_bytes is not None:
                # These exact bytes were checked with the features by this
                # worker. The new run's disk copy is archival, not an input.
                frozen.require(name in ("train", "validation"), "Cache cannot supply another split")
                return json.loads(cached_dataset_bytes[name + ".json"])
            return load_split(directory / "dataset", name)
        optcfg = cfg["optimizer"]
        optimizer = None if mode == "evaluate" else torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad], lr=optcfg["lr"], betas=tuple(optcfg["betas"]),
            eps=optcfg["eps"], weight_decay=optcfg["weight_decay"], amsgrad=optcfg["amsgrad"])
        if cache_ref is not None:
            frozen.require(optimizer is not None and not optimizer.state and result["optimizer_steps"] == 0, "Recovery must start fresh Adam at update zero")
            result["optimizer_initial_state_entries"] = len(optimizer.state)
        def gradient_update(loss, *, boundary=None, boundary_audit=None):
            return _audited_gradient_update(policy, optimizer, loss,
                directory=directory / "gradient-audits" / f"{result['optimizer_steps']+1:04d}",
                result=result, max_updates=cfg["max_updates"], boundary=boundary,
                required_nonzero=("brain.theta", "behavior_head.weight", "behavior_head.bias",
                                  "phase_head.weight", "phase_head.bias") if mode in PROBE_MODES else (),
                expected_zero=("parameter_head.weight", "parameter_head.bias") if mode in PROBE_MODES else (),
                boundary_audit=boundary_audit, check_adam_scale=mode == "probe-stabilized")

        if mode in PROBE_MODES:
            episode = partition("train")[0]
            frozen.require(len(episode["frames"]) >= 2, "Probe episode must contain two observations")
            folder = directory / "probe"
            folder.mkdir()
            frames = episode["frames"][:2]
            if pairing is not None:
                reference = Path(reg["paired_probe"]["path"])
                for index, frame in enumerate(frames):
                    frozen.require(frame["observation"] == frozen.read_json(reference / "probe" / f"frame-{index:04d}.json")["observation"],
                                   "Paired probe observation differs")
                frozen.require([(frame["target"]["behavior"], frame["target"]["phase"]) for frame in frames]
                               == [("WAVE_RIGHT", "START"), ("HOLD", "FINISH")], "Paired probe targets changed")
            def compare(name):
                if pairing is not None:
                    compare_paired_artifact(reference / "probe" / f"{name}.npz", folder / f"{name}.npz", pairing,
                                           policy_id=policy.policy_id, report_path=directory / "paired-forward.json")
            phase("probe_precondition", episode_id=episode["episode_id"])
            state = policy.initial_state(frames[0]["observation"]["session_id"])
            before_id = _save_state(folder, "state-0000", policy, state)
            compare("state-0000")
            boundary_audit = BoundaryGradientAudit(directory / "boundary-gradients") if pairing is not None else None
            outputs, records = [], []
            for index, frame in enumerate(frames):
                phase("probe_forward", decision=index+1)
                observation = Observation.from_dict(frame["observation"])
                output = policy(observation, state, boundary_observer=boundary_audit.observe if boundary_audit else None)
                if index == 0:
                    output.state.brain.v.retain_grad()
                    boundary = output.state.brain.v
                records.append(_save_output(folder, index, observation, output, policy, before_id, initial_id))
                before_id = _save_state(folder, f"state-{index+1:04d}", policy, output.state)
                compare(f"frame-{index:04d}")
                compare(f"state-{index+1:04d}")
                outputs.append(output)
                state = output.state
            phase("probe_backward_second_window_only")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(outputs[1], frames[1]["target"])
            update = gradient_update(loss, boundary=boundary, boundary_audit=boundary_audit)
            write_json(directory / "probe.json", {"episode_id": episode["episode_id"], "frame_count": 2,
                "targets": [f["target"] for f in frames], "predictions": records, "update": update,
                "interpretation": "finite/nonzero gradients measure engineering credit only, not learned task ability"})
            final = save_checkpoint(directory / "final-checkpoint", policy,
                {"registration_id": reg["registration_id"], "optimizer_steps": 1, "role": "probe_only_not_selected_for_deployment"}, optimizer=optimizer)
            result.update(probe=update, checkpoint_id=final["checkpoint_id"], checkpoint_path=str(directory / "final-checkpoint"))
        else:
            def cache(episodes, name, *, parameters_id=initial_id):
                features, predictions = {}, {}
                for eindex, episode in enumerate(episodes):
                    phase(f"{name}_episode", episode=eindex+1, total=len(episodes))
                    folder = directory / name / episode["episode_id"]
                    folder.mkdir(parents=True)
                    with torch.no_grad():
                        state = policy.initial_state(episode["frames"][0]["observation"]["session_id"])
                        before_id = _save_state(folder, "state-0000", policy, state)
                        features[episode["episode_id"]], predictions[episode["episode_id"]] = [], []
                        for index, frame in enumerate(episode["frames"]):
                            observation = Observation.from_dict(frame["observation"])
                            output = policy(observation, state, checkpoint_blocks=False)
                            predictions[episode["episode_id"]].append(_save_output(folder, index, observation, output, policy, before_id, parameters_id))
                            features[episode["episode_id"]].append(output.features_mv.detach().cpu())
                            state = output.state
                            before_id = _save_state(folder, f"state-{index+1:04d}", policy, state)
                return features, predictions
            def cached_outputs(episodes, features, *, intervention=None):
                predictions, losses = {}, []
                for episode in episodes:
                    rows = features[episode["episode_id"]]
                    used = [rows[0]]*len(rows) if intervention == "constant" else list(reversed(rows)) if intervention == "reversed_time" else rows
                    predictions[episode["episode_id"]] = []
                    for frame, feature in zip(episode["frames"], used):
                        heads = policy.read_features(feature.to("mps"))
                        predictions[episode["episode_id"]].append(_prediction(heads))
                        losses.append(loss_for(BehaviorOutput(*heads, feature, None), frame["target"]))
                return predictions, torch.stack(losses).mean()
            if mode == "evaluate":
                episodes = partition("locked_test")
                frozen.require(len(episodes) == 100, "Locked evaluation must retain all 100 episodes")
                features, predictions = cache(episodes, "locked-evaluation")
                scores = {"actual": score_predictions(episodes, predictions)}
                with torch.no_grad():
                    for intervention in ("constant", "reversed_time"):
                        predicted, _ = cached_outputs(episodes, features, intervention=intervention)
                        scores[intervention] = score_predictions(episodes, predicted)
                scores["functional_comparison"] = {name: {"sequence_match_difference": scores["actual"]["sequence_matches"]-scores[name]["sequence_matches"],
                    "behavior_f1_difference": scores["actual"]["behavior_macro_f1"]-scores[name]["behavior_macro_f1"]} for name in ("constant", "reversed_time")}
                scores["p3_real_feedback_gate"] = False
                scores["p3_reason"] = "Registered v1 data contains simulated execution feedback; teacher agreement does not prove closed-loop task completion"
                write_json(directory / "scores.json", scores)
                result.update(scores=scores, checkpoint_id=initial_id, behavior_pass=None,
                              locked_test_consumed=True, parameter_selection="none; selected checkpoint supplied before test opened")
            else:
                train = select_episodes(partition("train"), cfg["train_episodes"])
                validation = select_episodes(partition("validation"), cfg["validation_episodes"])
                write_json(directory / "selected_episodes.json", {"train": [e["episode_id"] for e in train],
                    "validation": [e["episode_id"] for e in validation], "locked_test_opened": False})
                history, best, selected = [], float("inf"), None
                if mode == "readout":
                    if cached_features is None:
                        features, _ = cache(train, "train-features")
                        validation_features, _ = cache(validation, "validation-features")
                    else:
                        frozen.require([e["episode_id"] for e in train] == cache_manifest["selected_episodes"]["train"]
                                       and [e["episode_id"] for e in validation] == cache_manifest["selected_episodes"]["validation"], "Worker cache episode order differs")
                        features, validation_features = [{key: [torch.from_numpy(value) for value in rows] for key, rows in cached_features[split].items()} for split in ("train", "validation")]
                        del cached_features
                    for index in range(cfg["max_updates"]):
                        phase("readout_update", update=index+1)
                        episode = train[index % len(train)]
                        optimizer.zero_grad(set_to_none=True)
                        _, loss = cached_outputs([episode], features)
                        update = gradient_update(loss)
                        with torch.no_grad():
                            predicted, val_loss = cached_outputs(validation, validation_features)
                            value = float(val_loss.cpu())
                        update["validation_loss"] = value
                        history.append(update)
                        if value < best:
                            best = value
                            selected = save_checkpoint(directory / "checkpoints" / f"{index+1:04d}", policy,
                                {"registration_id": reg["registration_id"], "optimizer_steps": index+1,
                                 "selection": "validation_loss", "validation_loss": best}, optimizer=optimizer)
                        write_json(directory / "updates.json", {"history": history})
                else:
                    states, positions = {}, {}
                    for index in range(cfg["max_updates"]):
                        episode = train[index % len(train)]
                        eid = episode["episode_id"]
                        folder = directory / "joint-windows" / f"{index+1:04d}"
                        folder.mkdir(parents=True)
                        position = positions.get(eid, 0)
                        phase("joint_window", update=index+1, episode_id=eid, first_frame=position)
                        state = states.get(eid)
                        if state is None:
                            state = policy.initial_state(episode["frames"][0]["observation"]["session_id"])
                        before_id = _save_state(folder, "state-0000", policy, state)
                        current = save_checkpoint(folder / "input-checkpoint", policy,
                            {"registration_id": reg["registration_id"], "optimizer_steps": index,
                             "role": "parameters_actually_used_for_window"})
                        losses = []
                        optimizer.zero_grad(set_to_none=True)
                        window = episode["frames"][position:position+2]
                        for within, frame in enumerate(window):
                            observation = Observation.from_dict(frame["observation"])
                            output = policy(observation, state)
                            _save_output(folder, within, observation, output, policy, before_id, current["checkpoint_id"])
                            losses.append(loss_for(output, frame["target"]))
                            state = output.state
                            before_id = _save_state(folder, f"state-{within+1:04d}", policy, state)
                        update = gradient_update(torch.stack(losses).mean())
                        history.append(update)
                        position += len(window)
                        positions[eid] = position if position < len(episode["frames"]) else 0
                        states[eid] = state.detach() if positions[eid] else None
                        write_json(directory / "updates.json", {"history": history})
                    # Exactly one final validation checkpoint; no test-directed continuation.
                    candidate = save_checkpoint(directory / "validation-input-checkpoint", policy,
                        {"registration_id": reg["registration_id"], "optimizer_steps": cfg["max_updates"],
                         "role": "final_candidate_parameters_for_validation"})
                    validation_features, _ = cache(validation, "validation-features", parameters_id=candidate["checkpoint_id"])
                    with torch.no_grad():
                        predicted, val_loss = cached_outputs(validation, validation_features)
                        best = float(val_loss.cpu())
                    selected = save_checkpoint(directory / "checkpoints" / f"{cfg['max_updates']:04d}", policy,
                        {"registration_id": reg["registration_id"], "optimizer_steps": cfg["max_updates"],
                         "selection": "final_registered_candidate_validated", "validation_loss": best}, optimizer=optimizer)
                frozen.require(selected is not None, "No checkpoint selected")
                selection = {"checkpoint_id": selected["checkpoint_id"], "validation_loss": best,
                    "checkpoint_path": str(directory / "checkpoints" / f"{selected['context']['optimizer_steps']:04d}"),
                    "locked_test_opened": False, "training_source": "simulated_execution_feedback"}
                write_json(directory / "selection.json", selection)
                restore_checkpoint(selection["checkpoint_path"], policy)
                with torch.no_grad():
                    predicted, selected_loss = cached_outputs(validation, validation_features)
                write_json(directory / "validation-scores.json", {"checkpoint_id": selected["checkpoint_id"],
                    "loss": float(selected_loss.cpu()), "scores": score_predictions(validation, predicted),
                    "predictions": predicted})
                result.update(selection=selection, checkpoint_id=selected["checkpoint_id"], checkpoint_path=selection["checkpoint_path"])
        frozen.require(hashlib_counts(policy) == raw_hash_before, "Raw graph counts changed")
        after_theta = policy.brain.theta.detach().cpu().numpy().copy()
        changed = int(np.count_nonzero(before_theta != after_theta))
        parameter_changes = {name: int(np.count_nonzero(parameters_before[name] != p.detach().cpu().numpy()))
                             for name, p in policy.named_parameters()}
        result.update(theta_changed_count=changed, parameter_changed_counts=parameter_changes,
                      theta_before=array_identity(before_theta), theta_after=array_identity(after_theta),
                      raw_counts_unchanged=True)
        if mode in PROBE_MODES:
            frozen.require(all(parameter_changes[name] > 0 for name in ("brain.theta", "behavior_head.weight",
                           "behavior_head.bias", "phase_head.weight", "phase_head.bias")), "Probe expected update did not occur")
        if mode in ("readout", "evaluate"):
            frozen.require(changed == 0, "Frozen theta changed")
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root / "src/malecns_lab" / name) == expected, "Source changed during experiment")
        if cache_ref is not None:
            frozen.require(result["optimizer_steps"] == 200 and len(history) == 200
                           and result["opened_splits"] == ["train", "validation"] and not any(result["neural_calls"].values()), "Cache recovery incomplete or performed neural computation")
            result.update(cache_complete=True, frozen_theta=True, completed_validation_updates=len(history))
        result.update(status="completed", reason="registered_bounded_computation_complete")
        phase("completed")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}:{error}"
        traceback.print_exc()
    finally:
        if restore_neural_guard is not None:
            restore_neural_guard()
        if sampler is not None:
            try:
                result["mps_memory"] = sampler.close()
                if result["mps_memory"]["error"] is not None:
                    result.update(status="failed", reason=result["mps_memory"]["error"])
            except BaseException as error:
                result.update(status="failed", reason=f"memory_close:{type(error).__name__}:{error}")
        result["wall_seconds"] = time.monotonic()-started
        write_json(directory / "worker_result.json", result)
    return result


def hashlib_counts(policy):
    """Hash immutable raw signed graph arrays without allocating another graph."""
    import hashlib
    digest = hashlib.sha256()
    for array in (policy.brain.graph.body_ids, policy.brain.graph.counts.indptr,
                  policy.brain.graph.counts.indices, policy.brain.graph.counts.data):
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()
