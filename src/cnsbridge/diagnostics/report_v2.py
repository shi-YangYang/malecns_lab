"""Summarize and plot an existing V2 neural recording without running a brain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..brain.metadata import file_hashes
from .replay_storage_v2 import source_coverage, write_json


def _mean(values: np.ndarray, valid: np.ndarray) -> float | list | None:
    if not valid.any():
        return None
    if not np.isfinite(values[valid]).all():
        raise ValueError("Nonfinite values on prediction-valid rows")
    return values[valid].mean(axis=0).tolist()


def summarize_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest_path, summary_path = run_dir / "run_manifest.json", run_dir / "run_summary.json"
    manifest = json.loads(manifest_path.read_text())
    run = json.loads(summary_path.read_text())
    if manifest.get("format") != "cnsbridge.neural_replay_run.v2":
        raise ValueError("Expected a V2 replay run")
    provenance = manifest["model_provenance"]
    channels = provenance["vision"]["channels"]
    mapping = provenance["group_mapping"]
    output_groups = [name for name, rule in mapping["rules"].items() if rule["role"] == "output"]
    summaries = []
    for segment in run["segments"]:
        name = segment["segment"]["segment_id"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid segment path in recording")
        if segment["source_id"] != manifest["model_source_id"]:
            raise ValueError("Segment model identity differs from the run")
        rows_path = run_dir / name / "rows.npz"
        with np.load(rows_path, allow_pickle=False) as rows:
            valid = rows["prediction_valid"]
            if valid.dtype != bool or valid.ndim != 1 or np.any(valid & ~rows["valid"].astype(bool)):
                raise ValueError("Invalid prediction validity mask")
            groups = segment["groups"]
            positions = [groups.index(group) for group in output_groups]
            for field, width in (("visual_features", len(channels)), ("neural_rates_hz", len(mapping["output_body_ids"])),
                                 ("group_rates_hz", len(groups)), ("input_event_counts", len(channels))):
                if rows[field].shape != (len(valid), width):
                    raise ValueError(f"Unexpected {field} shape")
            source = segment["input_source"]
            label = source.get("stimulus_kind") if source.get("kind") == "generated_engineering_stimulus" else source["scene_id"]
            summary = {
                "label": f"{label} / {name}", "segment_id": name, "input_source": source,
                "status": segment["status"], "coverage": segment["coverage"],
                "eligible_prediction_windows": int(valid.sum()),
                "input_mean_features": _mean(rows["visual_features"], valid),
                "output_mean_hz": _mean(rows["neural_rates_hz"], valid),
                "output_group_mean_hz": _mean(rows["group_rates_hz"][:, positions], valid),
                "input_events_eligible_windows": rows["input_event_counts"][valid].sum(axis=0).tolist() if valid.any() else None,
                "network_silent_fraction_mean": _mean(rows["network_silent_fraction"], valid),
                "network_high_rate_fraction_mean": _mean(rows["network_high_rate_fraction"], valid),
                "computation": segment["computation"], "rows_sha256": file_hashes(rows_path)["sha256"],
            }
            summaries.append(summary)
    return {
        "format": "cnsbridge.neural_report.v2", "run_directory": str(run_dir),
        "model_source_id": manifest["model_source_id"], "run_seed": manifest["run_seed"],
        "channels": channels, "output_body_ids": mapping["output_body_ids"], "output_groups": output_groups,
        "window_selection": "prediction_valid; eligible times only, no action labels or action scores",
        "source_coverage": source_coverage(run["segments"]), "segments": summaries,
        "wall_seconds": run["wall_seconds"], "peak_process_rss_bytes": run["peak_process_rss_bytes"],
        "input_manifest_sha256": file_hashes(manifest_path)["sha256"],
        "input_summary_sha256": file_hashes(summary_path)["sha256"],
        "report_code_sha256": file_hashes(Path(__file__))["sha256"],
        "claim_scope": "descriptive neural diagnostic; repeats and related material are not independent scenes; no behavior claim",
        "missing": "null means no eligible prediction windows, never a measured zero response",
    }


def plot_report(report: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segments = report["segments"]
    if not segments:
        raise ValueError("No segment inventory to plot")
    labels = []
    for index, row in enumerate(segments):
        source = row["input_source"]
        label = source["stimulus_kind"] if source["kind"] == "generated_engineering_stimulus" else source["scene_id"]
        labels.append(f"{label}\n{index}: n={row['eligible_prediction_windows']}")
    figure, axes = plt.subplots(3, 1, figsize=(16, max(9, len(segments) * .65 + 6)), constrained_layout=True)
    panels = (
        ("input_mean_features", report["channels"], "Engineering visual input", "normalized feature"),
        ("output_group_mean_hz", report["output_groups"], "Output-group mean activity (no action decoder)", "Hz"),
        ("output_mean_hz", [str(value) for value in report["output_body_ids"]], "Output body IDs", "Hz"),
    )
    for axis, (key, columns, title, unit) in zip(axes, panels):
        values = np.array([row[key] if row[key] is not None else [np.nan] * len(columns) for row in segments])
        masked = np.ma.masked_invalid(values)
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad("#dddddd")
        maximum = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
        panel = axis.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(maximum, 1e-12))
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
        axis.set_title(title)
        figure.colorbar(panel, ax=axis, label=unit, fraction=.02, pad=.01)
    figure.suptitle("MaleCNS V2 — eligible prediction windows, no action scores\nGray rows: no eligible windows; inspect coverage and failure reasons", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing replay_v2 run directory")
    parser.add_argument("--out", type=Path, required=True, help="New report directory")
    parser.add_argument("--no-plot", action="store_true", help="Write JSON only; does not require Matplotlib")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("report output already exists")
    report = summarize_run(args.run)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "diagnostic_summary.json", report)
    if not args.no_plot:
        plot_report(report, args.out / "diagnostic_summary.png")
    print(f"Saved {args.out}; segments={len(report['segments'])}; no brain simulation performed")


if __name__ == "__main__":
    main()
