"""Sealed, whole-scene pixel sequences with separate privileged teacher labels.

Collection runs only the velocity-level environment, never a neural model or
hardware. Invalid observations remain in sequence and have an explicit loss mask.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash, file_hashes
from ..control.body import BodyVelocity
from ..drones.follow_env import FollowEnvironment, limit_follow_action
from ..drones.follow_scenes import scene_registry
from ..senses.target import TargetCamera, detect_target
from .follow_task import rollout, truth_teacher, visual_teacher

SCHEMA = "cnsbridge.neural_training_data.v1"
DATA_ROLE = "registered_train_scenes"
TRAIN_IDS = tuple(sorted(f"train-sine-{axis}" for axis in ("forward", "yaw", "vertical", "combined")))
AXES = ("vx", "vz", "wz")
LIMITS = (0.3, 0.2, 0.5)
FIELDS = {"frames", "time_s", "teacher_raw", "teacher_target", "executed_action", "valid"}
SOURCES = ("learning/neural_data.py", "learning/follow_task.py", "drones/follow_env.py",
           "drones/follow_scenes.py", "senses/target.py", "control/body.py", "brain/metadata.py")


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _sources():
    package = Path(__file__).resolve().parents[1]
    return {name: file_hashes(package / name)["sha256"] for name in SOURCES}


def _scenes():
    scenes = sorted((scene for scene in scene_registry() if scene.split == "train"), key=lambda s: s.scene_id)
    if tuple(scene.scene_id for scene in scenes) != TRAIN_IDS:
        raise ValueError("Training data requires exactly the registered four train-sine scenes")
    return scenes


def _settings(camera, dt_s, response_tau_s):
    if not isinstance(camera, TargetCamera):
        raise ValueError("Explicit TargetCamera required")
    for value in (dt_s, response_tau_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Environment times must be positive finite numbers")
    return {"camera": asdict(camera), "dt_s": dt_s, "response_tau_s": response_tau_s}


def _vector(action):
    return [getattr(action, name) for name in AXES]


def _identity(camera, dt_s, response_tau_s):
    return {"role": DATA_ROLE, **_settings(camera, dt_s, response_tau_s),
            "scenes": [_json(asdict(scene)) for scene in _scenes()],
            "behavior": "visual_teacher(scene.desired_distance_m); rollout external target-loss intervention",
            "teacher": "simulator_truth: truth_teacher(scene.desired_distance_m)",
            "target": "limit_follow_action(teacher_raw); invalid frames retained with valid=False loss mask",
            "axes": list(AXES), "units": ["m/s", "m/s", "rad/s"], "frame": "body_flu", "limits": list(LIMITS),
            "implementation_sha256": _sources(),
            "evidence_scope": "velocity-level simulation data only; not hardware or neural-learning evidence"}


def _shape_spec(arrays):
    return {name: {"shape": list(value.shape), "dtype": value.dtype.str} for name, value in arrays.items()}


@dataclass(frozen=True)
class NeuralSequence:
    scene: dict
    frames: np.ndarray
    time_s: np.ndarray
    teacher_raw: np.ndarray
    teacher_target: np.ndarray
    executed_action: np.ndarray
    valid: np.ndarray
    summary: dict
    records: tuple[dict, ...]


@dataclass(frozen=True)
class NeuralDataset:
    identity: dict
    dataset_id: str
    sequences: tuple[NeuralSequence, ...]


def collect_neural_data(directory: str | Path, *, camera: TargetCamera = TargetCamera(),
                        dt_s: float = 0.1, response_tau_s: float = 0.25) -> dict:
    """Collect every registered train scene into a new directory; seal last.

    Environmental failures still produce records and a non-ready manifest. An
    interrupted write has no manifest and is never a loadable training dataset.
    """
    identity = _identity(camera, dt_s, response_tau_s)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for scene in _scenes():
        captured = []
        result = rollout(scene, partial(visual_teacher, desired_distance_m=scene.desired_distance_m),
                         policy_name="visual_teacher_behavior", camera=camera, dt_s=dt_s,
                         response_tau_s=response_tau_s,
                         record_frames=lambda sample: captured.append((sample.time_s, sample.frame.copy(),
                                                                       sample.observation.valid)))
        result.save(directory / scene.scene_id)
        if len(captured) != len(result.records):
            raise ValueError("Frame callbacks and rollout records differ; directory remains unsealed")
        arrays = {
            "frames": np.stack([item[1] for item in captured]),
            "time_s": np.asarray([item[0] for item in captured], dtype=np.float64),
            "teacher_raw": np.asarray([[r["teacher_action"][axis] for axis in AXES] for r in result.records],
                                      dtype=np.float64),
            "teacher_target": np.asarray([_vector(limit_follow_action(BodyVelocity(**r["teacher_action"])))
                                          for r in result.records], dtype=np.float64),
            "executed_action": np.asarray([[r["executed_action"][axis] for axis in AXES] for r in result.records],
                                           dtype=np.float64),
            "valid": np.asarray([item[2] for item in captured], dtype=np.bool_),
        }
        folder = directory / scene.scene_id
        np.savez(folder / "sequence.npz", **arrays)
        entries.append({"scene_id": scene.scene_id, "samples": len(captured),
                        "planned_samples": result.metrics["planned_samples"],
                        "failure": result.metrics["failure"], "observed_fraction": result.metrics["observed_fraction"],
                        "fields": _shape_spec(arrays),
                        "sha256": {name: file_hashes(folder / name)["sha256"] for name in
                                   ("sequence.npz", "summary.json", "trajectory.jsonl")}})
    ready = all(e["samples"] == e["planned_samples"] and e["failure"] is None for e in entries)
    manifest = {"schema": SCHEMA, "status": "ready" if ready else "collection_failed", "identity": identity,
                "sequences": entries}
    manifest["dataset_id"] = canonical_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def _validate_sequence(folder, entry, scene, camera, settings):
    if set(entry) != {"scene_id", "samples", "planned_samples", "failure", "observed_fraction", "fields", "sha256"}:
        raise ValueError("Unexpected sequence manifest fields")
    files = {"sequence.npz", "summary.json", "trajectory.jsonl"}
    if set(entry["sha256"]) != files:
        raise ValueError("Sequence file inventory mismatch")
    for name in files:
        if file_hashes(folder / name)["sha256"] != entry["sha256"][name]:
            raise ValueError("Sequence file hash mismatch")
    with np.load(folder / "sequence.npz", allow_pickle=False) as payload:
        if set(payload.files) != FIELDS or len(payload.files) != len(FIELDS):
            raise ValueError("Unexpected sequence array fields")
        arrays = {name: payload[name].copy() for name in FIELDS}
    planned = math.ceil(scene.duration_s / settings["dt_s"] - 1e-10)
    shapes = {"frames": (planned, camera.height, camera.width, 3), "time_s": (planned,), "valid": (planned,),
              **{name: (planned, 3) for name in ("teacher_raw", "teacher_target", "executed_action")}}
    for name, value in arrays.items():
        dtype = np.dtype(np.uint8 if name == "frames" else np.bool_ if name == "valid" else np.float64)
        if value.shape != shapes[name] or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid sequence array shape/dtype/values: {name}")
    if entry["fields"] != _shape_spec(arrays):
        raise ValueError("Array declarations do not match stored arrays")
    if (type(entry["samples"]) is not int or type(entry["planned_samples"]) is not int
            or entry["samples"] != planned or entry["planned_samples"] != planned
            or entry["failure"] is not None or entry["observed_fraction"] != 1):
        raise ValueError("Incomplete or failed scene cannot be training-ready")
    expected_times = np.arange(planned, dtype=np.float64) * settings["dt_s"]
    if not np.array_equal(arrays["time_s"], expected_times):
        raise ValueError("Sequence time axis must retain every planned frame in order")
    for name in ("teacher_target", "executed_action"):
        if np.any(np.abs(arrays[name]) > LIMITS):
            raise ValueError("Action exceeds registered SI limits")
    summary = _read_json(folder / "summary.json")
    records = tuple(_read_json_line(line) for line in (folder / "trajectory.jsonl").read_text().splitlines())
    if len(records) != planned or set(summary) != {"metadata", "metrics"}:
        raise ValueError("Incomplete rollout records")
    meta, metrics = summary["metadata"], summary["metrics"]
    expected_meta = {"schema": "cnsbridge.follow_rollout.v1", "scene": _json(asdict(scene)), "camera": asdict(camera),
                     "policy": "visual_teacher_behavior", "dt_s": settings["dt_s"],
                     "response_tau_s": settings["response_tau_s"], "action_axes": list(AXES),
                     "action_units": ["m/s", "m/s", "rad/s"], "action_frame": "body_flu"}
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("Rollout identity does not match the registered scene/settings")
    if (metrics.get("samples") != planned or metrics.get("planned_samples") != planned
            or metrics.get("observed_fraction") != 1 or metrics.get("failure") is not None
            or metrics.get("valid_observation_fraction") != np.count_nonzero(arrays["valid"]) / planned):
        raise ValueError("Rollout coverage/failure differs from the sealed sequence")
    # File hashes and internally consistent labels do not prove scene origin.
    # Replay the registered environment without a neural model, accepting only
    # the independently verified behavior actions below as state transitions.
    environment = FollowEnvironment(scene, camera, dt_s=settings["dt_s"], response_tau_s=settings["response_tau_s"])
    for index, record in enumerate(records):
        if environment.done or environment.time_s != expected_times[index]:
            raise ValueError("Registered scene replay ended early or lost its continuous clock")
        truth = environment.truth()
        if (not np.array_equal(arrays["frames"][index], environment.frame())
                or record["truth"] != _json(asdict(truth))):
            raise ValueError("Sequence pixels or truth do not come from the registered scene replay")
        observation = detect_target(arrays["frames"][index], camera)
        if (_json(asdict(observation)) != record["observation"] or observation.valid != arrays["valid"][index]
                or record["time_s"] != expected_times[index] or record["truth"]["time_s"] != expected_times[index]
                or record["teacher_source"] != "simulator_truth" or record["truth"]["failure"] is not None
                or record["next_truth"]["failure"] is not None):
            raise ValueError("Pixel validity, time or truth source mismatch")
        raw = truth_teacher(truth, desired_distance_m=scene.desired_distance_m)
        if asdict(raw) != record["teacher_action"]:
            raise ValueError("Teacher action does not match its privileged truth source")
        from .follow_task import FollowPolicyInput
        behavior = visual_teacher(FollowPolicyInput(arrays["frames"][index], observation, camera, expected_times[index]),
                                  desired_distance_m=scene.desired_distance_m)
        executed = limit_follow_action(behavior) if observation.valid else BodyVelocity()
        if asdict(behavior) != record["raw_policy_action"] or asdict(executed) != record["executed_action"]:
            raise ValueError("Behavior or executed action does not match the recorded visual policy")
        for name, expected in (("teacher_raw", raw), ("teacher_target", limit_follow_action(raw)),
                               ("executed_action", executed)):
            if not np.array_equal(arrays[name][index], _vector(expected)):
                raise ValueError(f"Sequence {name} is not aligned with its rollout record")
        step = environment.step(executed)
        if (asdict(step.executed) != record["executed_action"]
                or record["next_truth"] != _json(asdict(environment.truth()))
                or step.failure is not None or step.done != (index == planned - 1)):
            raise ValueError("Sequence next truth or termination does not match the registered scene replay")
    if not environment.done or environment.time_s != scene.duration_s:
        raise ValueError("Registered scene replay must finish at the complete scene boundary")
    # Bytes-backed values cannot be made writable by a consumer.
    immutable = {name: np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
                 for name, value in arrays.items()}
    return NeuralSequence(_json(asdict(scene)), **immutable, summary=summary, records=records)


def _read_json_line(line):
    value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {token}")))
    _json(value)
    return value


def read_neural_data(directory: str | Path) -> NeuralDataset:
    """Validate registration, split, source and all bytes/semantics; no graph load."""
    directory = Path(directory)
    try:
        manifest = _read_json(directory / "manifest.json")
        if set(manifest) != {"schema", "status", "identity", "sequences", "dataset_id"}:
            raise ValueError("Unexpected dataset manifest fields")
        if manifest["schema"] != SCHEMA or manifest["status"] != "ready":
            raise ValueError("Only sealed ready neural training data may be loaded")
        content = {key: value for key, value in manifest.items() if key != "dataset_id"}
        if manifest["dataset_id"] != canonical_hash(content):
            raise ValueError("Dataset canonical identity mismatch")
        identity = manifest["identity"]
        camera = TargetCamera(**identity["camera"])
        expected = _identity(camera, identity["dt_s"], identity["response_tau_s"])
        if identity != expected:
            raise ValueError("Dataset scene/split, source or semantic identity mismatch")
        entries = manifest["sequences"]
        if [entry["scene_id"] for entry in entries] != list(TRAIN_IDS):
            raise ValueError("All registered train scenes must occur once in sorted order")
        sequences = tuple(_validate_sequence(directory / scene.scene_id, entry, scene, camera, identity)
                          for scene, entry in zip(_scenes(), entries))
        return NeuralDataset(identity, manifest["dataset_id"], sequences)
    except (KeyError, TypeError, OverflowError, OSError) as exc:
        raise ValueError(f"Invalid or incomplete neural training dataset: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--response-tau-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    manifest = collect_neural_data(args.out, dt_s=args.dt_s, response_tau_s=args.response_tau_s)
    print(json.dumps({key: manifest[key] for key in ("status", "dataset_id")}, indent=2))
    if manifest["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
