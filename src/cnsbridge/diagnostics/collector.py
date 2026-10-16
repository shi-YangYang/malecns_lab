"""Sequential video replay and bounded neural observations; no motor decoder."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from cnsbridge.brain import Brain, load_connectome
from cnsbridge.learning.prepare import read_recording
from cnsbridge.senses import InputEncoder, Retina

from .stimuli import read_stimuli

WINDOW_MS = 50.0
MAX_NEURONS = 4096


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict:
    package = Path(__file__).resolve().parents[1]
    paths = ["brain/brain.py", "brain/lif.py", "brain/connectome.py", "config.py", "paths.py",
             "senses/retina.py", "senses/encoder.py"]
    paths += [str(path.relative_to(package)) for path in sorted((package / "diagnostics").glob("*.py"))]
    return {name: file_hash(package / name) for name in paths}


def graph_hash(connectome) -> str:
    digest = hashlib.sha256()
    matrix = connectome.weights
    for value in (np.array(matrix.shape), matrix.indptr, matrix.indices, matrix.data,
                  np.asarray(connectome.types, dtype=str), np.asarray(connectome.sides, dtype=str),
                  np.asarray(connectome.body_ids, dtype=str) if connectome.body_ids is not None else np.array([], dtype=str)):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def select_neurons(groups: dict[str, np.ndarray], maximum: int) -> np.ndarray:
    """Round robin sorted groups and ascending members, with no duplicates."""
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= MAX_NEURONS:
        raise ValueError(f"max_neurons must be an integer between 1 and {MAX_NEURONS}")
    pending = [iter(np.unique(groups[name])) for name in sorted(groups)]
    chosen, seen = [], set()
    while pending and len(chosen) < maximum:
        remaining = []
        for members in pending:
            for index in members:
                if int(index) not in seen:
                    seen.add(int(index))
                    chosen.append(int(index))
                    remaining.append(members)
                    break
            if len(chosen) == maximum:
                break
        pending = remaining
    return np.asarray(chosen, dtype=np.int64)


class ActivityBrain:
    def __init__(self, cfg: dict, max_neurons: int = 128):
        dt = float(cfg.get("brain", {}).get("lif", {}).get("dt", .5))
        steps = round(WINDOW_MS / dt) if math.isfinite(dt) and dt > 0 and math.isfinite(WINDOW_MS / dt) else 0
        if steps < 1 or not math.isclose(steps * dt, WINDOW_MS, rel_tol=0, abs_tol=1e-9):
            raise ValueError("brain.lif.dt must be positive and divide the 50 ms window exactly")
        # Validate the bound before allocating the full connectome.
        select_neurons({}, max_neurons)
        self.brain = Brain(load_connectome(cfg["brain"]["source"]), cfg)
        self.retina = Retina.from_config(cfg)
        self.encoder = InputEncoder(self.brain.connectome, cfg)
        self.input_names = list(self.brain.input_specs)
        names = list(dict.fromkeys(self.input_names + list(self.brain.output_specs)))
        self.groups = {name: self.brain.connectome.group(name) for name in names}
        self.indices = select_neurons(self.groups, max_neurons)
        # last_counts already covers all neurons; avoid constructing any raster.
        self.brain.record = np.zeros(0, dtype=np.int64)
        connectome = self.brain.connectome
        group_metadata = [{
            "name": name, "roles": [role for role, specs in (("input", self.brain.input_specs),
                                                               ("output", self.brain.output_specs)) if name in specs],
            "size": int(indices.size), "missing": not bool(indices.size), "indices": indices.tolist(),
        } for name, indices in self.groups.items()]
        self.provenance = {
            "effective_config": cfg, "seed": cfg["brain"].get("seed", 0),
            "network": {"source": str(cfg["brain"]["source"]), "name": connectome.name,
                        "neurons": connectome.n, "connections": connectome.n_connections,
                        "graph_sha256": graph_hash(connectome)},
            "source_sha256": source_hashes(), "groups": group_metadata,
            "missing_groups": [group["name"] for group in group_metadata if group["missing"]],
            "sampling": {"method": "round_robin_sorted_groups_ascending_indices_unique", "max_neurons": max_neurons,
                         "representative_of_whole_brain": False},
            "neurons": [{"index": int(index), "type": str(connectome.types[index]), "side": str(connectome.sides[index]),
                         "body_id": str(connectome.body_ids[index]) if connectome.body_ids is not None else None,
                         "groups": [name for name, members in self.groups.items() if index in members]}
                        for index in self.indices],
            "window_ms": WINDOW_MS, "dt_ms": dt, "input_yaw_rate_dps": 0.0,
            "color_contract": "BGR through existing Retina, unchanged from ground replay",
            "raster_recorded": False,
        }

    def step(self, frame):
        rates = self.encoder.encode(self.retina.encode(frame), yaw_rate_dps=0.0)
        drive = np.array([float(np.mean(rates[name])) if name in rates and self.groups[name].size
                          else (0.0 if self.groups[name].size else np.nan) for name in self.input_names])
        start_ms = self.brain.net.t_ms
        self.brain.tick(rates, ms=WINDOW_MS)
        end_ms = self.brain.net.t_ms
        counts = self.brain.last_counts
        group_rates = np.array([float(np.mean(counts[indices])) * 1000 / WINDOW_MS if indices.size else np.nan
                                for indices in self.groups.values()])
        return drive, group_rates, counts[self.indices].copy(), start_ms, end_ms


def collect(session: Path, cfg: dict, *, stimuli_path: Path | None = None, max_neurons=128, warmup_s=2.0, progress=None):
    import cv2

    session = Path(session)
    manifest, host_times = read_recording(session)
    if not math.isfinite(warmup_s) or warmup_s < 0:
        raise ValueError("warmup_s must be finite and nonnegative")
    intervals = read_stimuli(stimuli_path, len(host_times) / 20)
    engine = ActivityBrain(cfg, max_neurons)
    capture = cv2.VideoCapture(str(session / "video.avi"))
    drives, rates, counts, simulation, processing = [], [], [], [], []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video.avi")
        for index in range(len(host_times)):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended at frame {index}; frames.csv contains {len(host_times)} rows")
            started = time.perf_counter()
            drive, rate, count, start_ms, end_ms = engine.step(frame)
            processing.append((time.perf_counter() - started) * 1000)
            drives.append(drive)
            rates.append(rate)
            counts.append(count)
            simulation.append((start_ms, end_ms))
            if progress is not None and ((index + 1) % 100 == 0 or index + 1 == len(host_times)):
                progress(index + 1, len(host_times))
        if capture.read()[0]:
            raise ValueError("Video has more frames than frames.csv")
    finally:
        capture.release()
    times = np.arange(len(host_times), dtype=np.float64) / 20
    count_array = np.asarray(counts, dtype=np.int32)
    arrays = {
        "frame_index": np.arange(len(times)), "video_start_s": times, "video_end_s": times + .05,
        "host_dequeue_s": host_times, "simulation_start_ms": np.array(simulation)[:, 0],
        "simulation_end_ms": np.array(simulation)[:, 1], "processing_ms": np.array(processing),
        "input_names": np.asarray(engine.input_names, dtype=str), "group_names": np.asarray(list(engine.groups), dtype=str),
        "input_drive_hz": np.asarray(drives), "group_rates_hz": np.asarray(rates),
        "neuron_indices": engine.indices, "neuron_counts": count_array,
        "neuron_rates_hz": count_array.astype(np.float64) * 1000 / WINDOW_MS,
    }
    sources = {name: {"path": str((session / name).resolve()), "sha256": file_hash(session / name)}
               for name in ("session.json", "frames.csv", "video.avi")}
    if stimuli_path is not None:
        sources["stimuli.csv"] = {"path": str(Path(stimuli_path).resolve()), "sha256": file_hash(Path(stimuli_path))}
    metadata = {
        "format": "cnsbridge.neural_activity", "version": 1, "status": "complete", "session": manifest,
        "sources": sources, "frame_count": len(times), "duration_s": len(times) / 20,
        "warmup_s": warmup_s, "stimuli": intervals, "neural_provenance": engine.provenance,
        "time_contract": {"video": "AVI playback: frame_index / 20, half-open 50 ms windows",
                          "host": "recorder host dequeue timestamp; not camera exposure time",
                          "simulation": "cumulative LIF clock, milliseconds", "processing": "wall-clock computation, not biological latency"},
        "units": {"drive": "Hz external Poisson input per neuron", "firing": "Hz spikes per neuron per 50 ms window",
                  "counts": "spikes per neuron per window"},
        "comparison": "Exclude warmup and boundary-crossing windows; equal-weight interval means per stimulus; no significance tests",
        "display_smoothing": "none", "claim_scope": "Simulated neural activity only; no live biological measurement, training, actuation or avoidance evidence",
        "cadence": {"host_gap_max_ms": float(np.diff(host_times).max() * 1000),
                    "processing_p95_ms": float(np.percentile(processing, 95))},
    }
    # Fail before writing a report if custom configuration is not JSON-safe.
    json.dumps(metadata, allow_nan=False)
    return arrays, metadata
