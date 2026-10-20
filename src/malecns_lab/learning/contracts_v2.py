"""V2 offline learning identities and independently supplied masked targets."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..control.body import BODY_AXES, BODY_FRAME, BODY_UNITS

RIDGE_CANDIDATES = tuple(10.0 ** power for power in range(-4, 5))
SUPPORTED_AXES = ("vx", "wz")
OUTPUT_SPEC = {"axes": list(BODY_AXES), "frame": BODY_FRAME, "units": list(BODY_UNITS)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nonempty {name} required")
    return value


@dataclass(frozen=True)
class FeatureSchemaV2:
    source_id: str
    feature_names: tuple[str, ...]
    representation: str
    window_ms: float = 50.0
    target_horizon_s: float = 0.05

    def __post_init__(self):
        nonempty(self.source_id, "model source_id")
        names = tuple(self.feature_names)
        expected = {"neural": 132, "visual": 98}.get(self.representation)
        if expected is None or len(names) != expected or len(set(names)) != expected:
            raise ValueError("V2 requires ordered unique 132 neural or 98 visual feature names")
        for name in names:
            nonempty(name, "feature name")
        if self.window_ms != 50.0 or self.target_horizon_s != 0.05:
            raise ValueError("V2 time contract is 50 ms window and 0.05 s target horizon")
        object.__setattr__(self, "feature_names", names)

    def as_dict(self):
        return {"source_id": self.source_id, "feature_names": list(self.feature_names),
                "representation": self.representation, "window_ms": self.window_ms,
                "target_horizon_s": self.target_horizon_s}


def validate_protocol(value: dict) -> dict:
    if value.get("format") != "cnsbridge.learning_protocol.v2":
        raise ValueError("Expected cnsbridge.learning_protocol.v2")
    nonempty(value.get("protocol_id"), "protocol_id")
    if value.get("phase") not in ("pilot", "confirmatory"):
        raise ValueError("phase must be pilot or confirmatory")
    task = value["task"]
    for name in ("policy_id", "teacher_id", "description"):
        nonempty(task.get(name), name)
    if task.get("teacher_kind") not in ("independent_human", "predefined_task"):
        raise ValueError("Independent human/task teacher required; RC, measured velocity and neural outputs are not labels")
    for field in ("tolerances", "stationary_tolerances"):
        if set(task.get(field, {})) != set(SUPPORTED_AXES):
            raise ValueError(f"Explicit vx/wz {field} required")
        if any(not np.isfinite(v) or v <= 0 for v in task[field].values()):
            raise ValueError(f"{field} must contain positive finite SI values")
    delta = value.get("practical_delta")
    if not isinstance(delta, (int, float)) or not np.isfinite(delta) or delta < 0:
        raise ValueError("Predeclared nonnegative practical_delta in normalized-error units required")
    budget = value.get("budget_seconds")
    if not isinstance(budget, (int, float)) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Explicit positive wall budget required")
    bootstrap = value["bootstrap"]
    for name in ("seed", "replicates"):
        v = bootstrap.get(name)
        if isinstance(v, bool) or not isinstance(v, int) or v < (1 if name == "replicates" else 0):
            raise ValueError(f"Invalid bootstrap {name}")
    if not isinstance(value.get("segments"), list) or not value["segments"]:
        raise ValueError("Predeclared segment/split/label entries required")
    for entry in value["segments"]:
        nonempty(entry.get("path"), "segment path")
        nonempty(entry.get("labels"), "labels path")
        if entry.get("split") not in ("train", "val", "test"):
            raise ValueError("Explicit train/val/test split required; no frame split")
        if entry.get("label_format", "v2") not in ("v2", "legacy_v1_intervals"):
            raise ValueError("Unrecognized explicit label format")
    return value


def load_labels(path: Path, *, source_id: str, task: dict, legacy: bool = False) -> tuple[list[dict], dict]:
    """Intervals are half-open in video-playback seconds; missing is unsupervised."""
    if legacy:
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        intervals = [{"start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                      "target": [float(r["vx_mps"]), 0, 0, 0, 0, float(r["wz_rad_s"])],
                      "mask": [True, False, False, False, False, True], "reason": r["reason"]} for r in records]
        identity = {"format": "explicit_legacy_v1_interval_import", "source_id": source_id,
                    "policy_id": task["policy_id"], "teacher_id": task["teacher_id"], "teacher_kind": task["teacher_kind"],
                    "timebase": "20Hz_video_playback", "output": OUTPUT_SPEC}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "cnsbridge.motion_labels.v2" or data.get("source_id") != source_id:
            raise ValueError("Label format/source identity mismatch")
        if data.get("output") != OUTPUT_SPEC or data.get("timebase") != "20Hz_video_playback":
            raise ValueError("Label coordinate/units/playback timebase mismatch")
        for name in ("policy_id", "teacher_id", "teacher_kind"):
            if data.get(name) != task[name]:
                raise ValueError(f"Label {name} differs from registered independent teacher")
        intervals = data["intervals"]
        identity = {key: val for key, val in data.items() if key != "intervals"}
    normalized, previous_end = [], -np.inf
    for item in intervals:
        start, end = float(item["start_s"]), float(item["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end:
            raise ValueError("Labels must be sorted nonoverlapping finite half-open intervals")
        mask = np.asarray(item["mask"])
        if mask.dtype != np.bool_ or mask.shape != (6,):
            raise ValueError("Label mask must be six booleans")
        if mask[[1, 2, 3, 4]].any():
            raise ValueError("Stage A protocol only authorizes vx/wz supervision")
        raw = item["target"]
        if len(raw) != 6:
            raise ValueError("Target needs six FLU entries")
        target = np.asarray([float(v) if supervised else 0.0 for v, supervised in zip(raw, mask)], dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("Supervised targets must be finite")
        nonempty(item.get("reason"), "label reason")
        normalized.append({"start_s": start, "end_s": end, "target": target, "mask": mask, "reason": item["reason"]})
        previous_end = end
    return normalized, {**identity, "file_sha256": file_digest(path)}


def target_at(intervals: list[dict], time_s: float) -> tuple[np.ndarray, np.ndarray, str]:
    for interval in intervals:
        if interval["start_s"] <= time_s < interval["end_s"]:
            return interval["target"].copy(), interval["mask"].copy(), interval["reason"]
    return np.zeros(6), np.zeros(6, dtype=bool), "unlabelled"
