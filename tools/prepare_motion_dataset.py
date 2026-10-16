"""Replay a manually labelled ground recording through a frozen fly network.

No SDK or actuator interface is used. Labels use AVI playback seconds, while
frames.csv preserves host dequeue time for cadence diagnostics. Unlabelled
frames advance the network but are never silently assigned a stop target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from cnsbridge import __file__ as PACKAGE_FILE
from cnsbridge.brain import Brain, load_connectome
from cnsbridge.config import load_config
from cnsbridge.motor.dataset import MotionDataset
from cnsbridge.motor.neural import NeuralSchema
from cnsbridge.senses import InputEncoder, Retina
from cnsbridge.senses.retina import FEATURES

PROJECT = Path(__file__).resolve().parents[1]
POLICY = "ground-planar-human-desired-velocity-v1"


def read_recording(directory: Path):
    manifest = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if (manifest.get("format"), manifest.get("version")) != ("flydrones.ground_recording", 1):
        raise ValueError("Unsupported recording format/version")
    if manifest.get("label_timebase") != "video_playback" or manifest.get("ground_only") is not True:
        raise ValueError("Expected a ground recording with video_playback label times")
    for key in ("session_id", "scene_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Missing {key}")
    if float(manifest.get("nominal_fps", 0)) != 20.0:
        raise ValueError("This pipeline requires the recorder's nominal 20 fps")
    # Fixed names prevent accidentally pairing a manifest with another movie.
    for key, name in (("video", "video.avi"), ("frames", "frames.csv"), ("labels", "labels.csv")):
        if manifest.get(key) != name:
            raise ValueError(f"Expected {key}={name}")
    with (directory / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("Recording needs at least two frames")
    timestamps = np.array([float(row["t_s"]) for row in rows], dtype=np.float64)
    if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Frame indices must be contiguous and start at zero")
    if not np.isfinite(timestamps).all() or (timestamps < 0).any() or (np.diff(timestamps) <= 0).any():
        raise ValueError("Host timestamps must be finite, nonnegative and strictly increasing")
    return manifest, timestamps


def read_labels(path: Path, duration_s: float) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"start_s", "end_s", "vx_mps", "wz_rad_s", "reason"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Labels require columns: {', '.join(sorted(required))}")
        labels = []
        for line, row in enumerate(reader, 2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                values = [float(row[key]) for key in ("start_s", "end_s", "vx_mps", "wz_rad_s")]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid or incomplete label at line {line}") from exc
            start, end, vx, wz = values
            if not all(math.isfinite(value) for value in values) or not 0 <= start < end <= duration_s + 1e-9:
                raise ValueError(f"Invalid label times/values at line {line}; duration={duration_s:g} s")
            reason = (row.get("reason") or "").strip()
            if not reason:
                raise ValueError(f"Label at line {line} requires a reason based on the visible scene")
            labels.append({"start_s": start, "end_s": end, "vx_mps": vx, "wz_rad_s": wz, "reason": reason})
    labels.sort(key=lambda label: label["start_s"])
    if not labels:
        raise ValueError("No manual labels yet. Fill labels.csv before extracting training data")
    if any(a["end_s"] > b["start_s"] for a, b in zip(labels, labels[1:])):
        raise ValueError("Label intervals overlap")
    return labels


def target_at(labels: list[dict], playback_s: float):
    for label in labels:
        if label["start_s"] <= playback_s < label["end_s"]:
            target = np.zeros(6, dtype=np.float64)
            target[0], target[5] = label["vx_mps"], label["wz_rad_s"]
            return target
    return None


def _hash_array(digest, array):
    value = np.ascontiguousarray(array)
    digest.update(str((value.dtype.str, value.shape)).encode())
    digest.update(memoryview(value).cast("B"))


class RecordingBrain:
    """Fresh state per session; labels never enter the neural computation."""

    def __init__(self, cfg: dict, readout_level: str = "neuron"):
        self.brain = Brain(load_connectome(cfg["brain"]["source"]), cfg)
        self.retina = Retina.from_config(cfg)
        self.encoder = InputEncoder(self.brain.connectome, cfg)
        self.window_ms = 50.0
        groups = {name: self.brain.connectome.group(name) for name in self.brain.output_specs}
        self.groups = {name: indices for name, indices in groups.items() if len(indices)}
        if not self.groups:
            raise ValueError("No output neurons matched this configuration")
        self.indices = np.unique(np.concatenate(list(self.groups.values())))
        if readout_level not in ("neuron", "group"):
            raise ValueError("readout_level must be neuron or group")
        self.readout_level = readout_level
        if readout_level == "neuron":
            names = tuple(f"neuron_index_{index}" for index in self.indices)
        else:
            names = tuple(self.groups)
        effective = {key: cfg[key] for key in ("brain", "inputs", "outputs", "vision", "imu")}
        effective["brain"] = {key: value for key, value in cfg["brain"].items() if key not in ("source", "record_neurons")}
        digest = hashlib.sha256()
        digest.update(json.dumps(effective, sort_keys=True, allow_nan=False).encode())
        digest.update(f"ground-replay-v1|BGR-existing-retina|yaw-input-zero|{readout_level}|50ms".encode())
        matrix = self.brain.connectome.weights
        for value in (np.array(matrix.shape), matrix.indptr, matrix.indices, matrix.data):
            _hash_array(digest, value)
        for name in sorted(set(self.brain.input_specs) | set(self.brain.output_specs)):
            digest.update(name.encode())
            _hash_array(digest, self.brain.connectome.group(name))
        # Include the implementation, not just graph labels and filesystem paths.
        for relative in ("brain/brain.py", "brain/lif.py", "senses/retina.py", "senses/encoder.py"):
            digest.update((Path(PACKAGE_FILE).parent / relative).read_bytes())
        self.schema = NeuralSchema(names, f"ground-replay-sha256:{digest.hexdigest()}", self.window_ms)
        rows, cols = cfg["vision"]["grid"]
        self.visual_names = tuple(
            f"{eye}_{feature}_{row}_{column}"
            for eye in "LR" for feature in FEATURES for row in range(rows) for column in range(cols)
        ) + ("expansion", "rotation", "vertical")
        self.provenance = {
            "connectome_name": self.brain.connectome.name,
            "neurons": self.brain.n_neurons,
            "readout_level": readout_level,
            "output_group_sizes": {name: len(indices) for name, indices in self.groups.items()},
            "missing_groups": self.brain.empty,
            "effective_config": effective,
            "input_yaw_rate_dps": 0.0,
            "model_ms_per_recorded_frame": self.window_ms,
            "color_contract": "BGR through existing Retina (same convention as Tello shadow)",
        }

    def step(self, frame):
        vision = self.retina.encode(frame)
        rates = self.brain.tick(self.encoder.encode(vision, yaw_rate_dps=0.0), ms=self.window_ms)
        if self.readout_level == "neuron":
            neural = self.brain.last_counts[self.indices].astype(np.float64) * (1000.0 / self.window_ms)
        else:
            neural = np.array([rates[name] for name in self.groups], dtype=np.float64)
        visual = np.concatenate(
            [vision.eyes[eye].grids[feature].ravel() for eye in "LR" for feature in FEATURES]
            + [np.array([vision.expansion, vision.rotation, vision.vertical])]
        )
        return neural, visual


def extract_session(directory: Path, cfg: dict, *, readout_level="neuron", warmup_s=2.0, progress=None):
    """Compute all frames sequentially; only labelled post-warmup rows are saved."""
    import cv2

    directory = Path(directory)
    manifest, timestamps = read_recording(directory)
    fps = manifest["nominal_fps"]
    labels = read_labels(directory / "labels.csv", len(timestamps) / fps)
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("warmup_s must be finite and nonnegative")
    if not any(target_at(labels, index / fps) is not None and index / fps >= warmup_s for index in range(len(timestamps))):
        raise ValueError("No labelled frames remain after warmup")
    engine = RecordingBrain(cfg, readout_level)
    capture = cv2.VideoCapture(str(directory / "video.avi"))
    neural_rows, visual_rows, targets, sample_times, processing_ms = [], [], [], [], []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video.avi")
        for index, host_time in enumerate(timestamps):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended at frame {index}; frames.csv contains {len(timestamps)} rows")
            started = time.perf_counter()
            neural, visual = engine.step(frame)
            processing_ms.append((time.perf_counter() - started) * 1000)
            target = target_at(labels, index / fps)
            if index / fps >= warmup_s and target is not None:
                neural_rows.append(neural)
                visual_rows.append(visual)
                targets.append(target)
                sample_times.append(index / fps)
            if progress is not None and (index + 1) % 100 == 0:
                progress(index + 1, len(timestamps))
        if capture.read()[0]:
            raise ValueError("Video has more frames than frames.csv")
    finally:
        capture.release()
    gap_ms = np.diff(timestamps) * 1000
    metadata = {
        "policy_id": POLICY,
        "label_source": "human_desired_velocity",
        "label_timebase": "video_playback",
        "labels": labels,
        "warmup_s": warmup_s,
        "session": manifest,
        "neural_provenance": engine.provenance,
        "cadence": {"host_gap_p95_ms": float(np.percentile(gap_ms, 95)),
                    "host_gap_max_ms": float(gap_ms.max()),
                    "processing_p95_ms": float(np.percentile(processing_ms, 95))},
        "claim_scope": "grounded offline imitation only; no measured motion or closed-loop result",
    }
    return MotionDataset(
        engine.schema, np.asarray(neural_rows), np.asarray(visual_rows), engine.visual_names,
        np.asarray(targets), (manifest["session_id"],) * len(targets),
        (manifest["scene_id"],) * len(targets), np.asarray(sample_times), metadata,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="New recorder's session directory")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "malecns.yaml")
    parser.add_argument("--brain", help="Override brain source, e.g. minifly for a software smoke test")
    parser.add_argument("--readout-level", choices=("neuron", "group"), default="neuron")
    parser.add_argument("--out", type=Path, required=True, help="New dataset .npz; never overwrites")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out already exists")
    cfg = load_config(args.config)
    source = args.brain or cfg["brain"]["source"]
    if str(source).lower() not in ("minifly", "synthetic", "mini"):
        source = str((PROJECT / source).resolve())
    cfg["brain"]["source"] = source
    try:
        print("OFFLINE REPLAY | frozen brain; manual labels; no aircraft connection", flush=True)
        dataset = extract_session(args.session, cfg, readout_level=args.readout_level,
                                  progress=lambda done, total: print(f"Frames: {done}/{total}", flush=True))
        dataset.save(args.out)
        print(f"Saved {len(dataset.targets)} labelled samples: {args.out.resolve()}")
        print(json.dumps(dataset.metadata["cadence"], indent=2))
        print("Ground video does not validate control during flight. Keep new models offline.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"DATASET FAILED: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
