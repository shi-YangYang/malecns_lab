"""Explicit recorded or generated visual sources; never reads action labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash, file_hashes

STIMULUS_KINDS = ("static", "left", "right", "up", "down", "expansion")


class VideoSourceV2:
    def __init__(self, video: Path, *, scene_id: str, session_id: str, family_id: str,
                 stimulus_kind: str = "unknown", recording_dir: Path | None = None):
        import cv2

        for name, value in (("scene_id", scene_id), ("session_id", session_id), ("family_id", family_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Explicit {name} is required")
        self.path = video.resolve()
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {self.path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if abs(fps - 20.0) > 1e-6 or self.total_frames < 2:
                raise ValueError("V2 requires a nonempty nominal-20-fps recording; resampling is not implicit")
        finally:
            capture.release()
        self.host_times = np.full(self.total_frames, np.nan)
        self.metadata = {"kind": "recorded_video", "video": str(self.path), "video_sha256": file_hashes(self.path)["sha256"],
                         "scene_id": scene_id, "session_id": session_id, "family_id": family_id,
                         "stimulus_kind": stimulus_kind, "nominal_fps": 20.0, "total_frames": self.total_frames,
                         "timebase": "frame_index / 20; not verified camera exposure timestamps"}
        if recording_dir is not None:
            manifest_path, frames_path = recording_dir / "session.json", recording_dir / "frames.csv"
            manifest = json.loads(manifest_path.read_text())
            if ((manifest.get("format"), manifest.get("version")) != ("flydrones.ground_recording", 1)
                    or manifest.get("ground_only") is not True or manifest.get("label_timebase") != "video_playback"
                    or manifest.get("nominal_fps") != 20 or manifest.get("video") != "video.avi"
                    or manifest.get("frames") != "frames.csv"):
                raise ValueError("Unsupported ground recording manifest")
            if manifest.get("scene_id") != scene_id or manifest.get("session_id") != session_id:
                raise ValueError("Explicit scene/session identity differs from recording")
            with frames_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["frame_index"]) for row in rows] != list(range(self.total_frames)):
                raise ValueError("frames.csv and decoded-video frame inventory disagree")
            self.host_times = np.array([float(row["t_s"]) for row in rows])
            if not np.isfinite(self.host_times).all() or np.any(self.host_times < 0) or np.any(np.diff(self.host_times) <= 0):
                raise ValueError("Invalid host dequeue timestamps")
            self.metadata["recording"] = {"manifest_sha256": file_hashes(manifest_path)["sha256"],
                                           "frames_sha256": file_hashes(frames_path)["sha256"],
                                           "labels_read": False, "host_time_role": "dequeue diagnostic only"}
        self.metadata["source_id"] = canonical_hash(self.metadata)

    def frames(self, start: int, end: int):
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        try:
            for index in range(end):
                ok, frame = capture.read()
                if index >= start:
                    yield index, frame if ok else None, float(self.host_times[index])
                    if not ok:
                        break
                elif not ok:
                    yield start, None, float(self.host_times[start])
                    break
        finally:
            capture.release()


class GeneratedSourceV2:
    def __init__(self, stimulus: str, *, observation_seconds: float = 0.5, texture_seed: int = 7):
        if stimulus not in STIMULUS_KINDS:
            raise ValueError("Unknown engineering stimulus")
        if not np.isfinite(observation_seconds) or observation_seconds < 0.1 or abs(observation_seconds * 20 - round(observation_seconds * 20)) > 1e-8:
            raise ValueError("Observation duration must be at least 0.1 s and an integer number of 20 Hz frames")
        self.stimulus = stimulus
        self.total_frames = 40 + int(round(observation_seconds * 20))
        self.phase = np.random.default_rng(texture_seed).uniform(-np.pi, np.pi, size=4)
        self.metadata = {"kind": "generated_engineering_stimulus", "stimulus_kind": stimulus, "texture_seed": texture_seed,
                         "scene_id": f"engineering_{stimulus}", "session_id": "generated_diagnostic",
                         "family_id": f"engineering_texture_{texture_seed}", "total_frames": self.total_frames,
                         "nominal_fps": 20.0, "warmup_s": 2.0, "observation_s": observation_seconds,
                         "translation_px_s": 10.0, "expansion_s_inv": 0.35,
                         "generator_sha256": file_hashes(Path(__file__))["sha256"],
                         "claim_scope": "engineered visual diagnostic; not recorded action data or synthetic brain"}
        self.metadata["source_id"] = canonical_hash(self.metadata)

    def frame(self, index: int):
        y, x = np.mgrid[:72, :96].astype(np.float64)
        t = max(0.0, index / 20 - 2.0)
        if self.stimulus in ("left", "right"):
            x -= (-1 if self.stimulus == "left" else 1) * 10 * t
        if self.stimulus in ("up", "down"):
            y -= (-1 if self.stimulus == "up" else 1) * 10 * t
        if self.stimulus == "expansion":
            scale = np.exp(0.35 * t)
            x, y = 47.5 + (x - 47.5) / scale, 35.5 + (y - 35.5) / scale
        gray = 127 + 34 * np.sin(x * 0.49 + self.phase[0]) + 31 * np.cos(y * 0.43 + self.phase[1])
        gray += 25 * np.sin(0.21 * x + 0.31 * y + self.phase[2]) + 19 * np.cos(0.37 * x - 0.27 * y + self.phase[3])
        return np.repeat(np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None], 3, axis=2)

    def frames(self, start: int, end: int):
        for index in range(start, end):
            yield index, self.frame(index), float("nan")


def segment_ranges(path: Path | None, total_frames: int) -> list[dict]:
    """Explicit predeclared ranges only; no label/score-guided automatic restart."""
    ranges = [{"segment_id": "segment_000", "start_frame": 0, "end_frame": total_frames}] if path is None else json.loads(path.read_text())
    previous_end, names = 0, set()
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Expected nonempty segment list")
    for item in ranges:
        name, start, end = item.get("segment_id"), item.get("start_frame"), item.get("end_frame")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError("Segment IDs must be unique safe path components")
        if (type(start) is not int or type(end) is not int or not previous_end <= start < end <= total_frames):
            raise ValueError("Segment frame ranges must be ordered, nonoverlapping and within the source")
        names.add(name)
        previous_end = end
    return ranges
