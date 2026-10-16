"""Explicit stimulus intervals and descriptive, interval-first comparisons."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def read_stimuli(path: Path | None, duration_s: float) -> list[dict]:
    if path is None:
        return []
    intervals = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"start_s", "end_s", "stimulus"}.issubset(reader.fieldnames or ()):
            raise ValueError("Stimuli require columns: start_s,end_s,stimulus")
        for line, row in enumerate(reader, 2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                start, end = float(row["start_s"]), float(row["end_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid stimulus times at line {line}") from exc
            label = (row.get("stimulus") or "").strip()
            if (not label or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= duration_s + 1e-9):
                raise ValueError(f"Invalid stimulus at line {line}; duration={duration_s:g} s")
            intervals.append({"start_s": start, "end_s": end, "stimulus": label})
    intervals.sort(key=lambda item: item["start_s"])
    if any(a["end_s"] > b["start_s"] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Stimulus intervals overlap")
    return intervals


def interval_mask(starts, ends, interval, warmup_s):
    """Use whole windows only; unlabelled gaps are never implicit baselines."""
    return ((starts >= warmup_s - 1e-9)
            & (starts >= interval["start_s"] - 1e-9)
            & (ends <= interval["end_s"] + 1e-9))


def summarize(arrays: dict, metadata: dict) -> list[dict]:
    """Summarize intervals first, then equal-weight valid interval means.

    Adjacent simulation windows are not independent experimental repetitions.
    No significance tests or standard errors are calculated.
    """
    rows = []
    series = [("input_drive", arrays["input_names"], arrays["input_drive_hz"]),
              ("group_firing", arrays["group_names"], arrays["group_rates_hz"])]
    intervals = metadata["stimuli"]
    for signal, names, values in series:
        for column, name in enumerate(names):
            group_rows = []
            for number, interval in enumerate(intervals):
                mask = interval_mask(arrays["video_start_s"], arrays["video_end_s"], interval, metadata["warmup_s"])
                sample = values[mask, column]
                finite = sample[np.isfinite(sample)]
                group_rows.append({
                    "scope": "interval", "interval_index": number, **interval,
                    "signal": signal, "group": str(name), "window_count": int(mask.sum()),
                    "valid_window_count": int(finite.size), "interval_count": int(bool(finite.size)),
                    "mean_hz": float(finite.mean()) if finite.size else None,
                    "delta_baseline_hz": None,
                })
            baseline = [r["mean_hz"] for r in group_rows if r["stimulus"] == "baseline" and r["mean_hz"] is not None]
            baseline_mean = float(np.mean(baseline)) if baseline else None
            for row in group_rows:
                if row["mean_hz"] is not None and baseline_mean is not None:
                    row["delta_baseline_hz"] = row["mean_hz"] - baseline_mean
            rows.extend(group_rows)
            for stimulus in dict.fromkeys(item["stimulus"] for item in intervals):
                selected = [row for row in group_rows if row["stimulus"] == stimulus]
                means = [row["mean_hz"] for row in selected if row["mean_hz"] is not None]
                mean = float(np.mean(means)) if means else None
                rows.append({
                    "scope": "stimulus", "interval_index": None, "start_s": None, "end_s": None,
                    "stimulus": stimulus, "signal": signal, "group": str(name),
                    "window_count": sum(row["window_count"] for row in selected),
                    "valid_window_count": sum(row["valid_window_count"] for row in selected),
                    "interval_count": len(means), "mean_hz": mean,
                    "delta_baseline_hz": mean - baseline_mean if mean is not None and baseline_mean is not None else None,
                })
    return rows
