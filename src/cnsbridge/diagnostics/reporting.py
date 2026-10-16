"""Portable numeric artifacts and static scientific plots (Matplotlib Agg)."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .stimuli import summarize

SUMMARY_FIELDS = ("scope", "interval_index", "start_s", "end_s", "stimulus", "signal", "group", "window_count",
                  "valid_window_count", "interval_count", "mean_hz", "delta_baseline_hz")


def _cell(value):
    return "" if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value


def _annotate(ax, metadata):
    colors = {label: f"C{index % 10}" for index, label in enumerate(dict.fromkeys(i["stimulus"] for i in metadata["stimuli"]))}
    for interval in metadata["stimuli"]:
        ax.axvspan(interval["start_s"], interval["end_s"], color=colors[interval["stimulus"]], alpha=.08)
    if metadata["warmup_s"] > 0:
        ax.axvspan(0, min(metadata["warmup_s"], metadata["duration_s"]), color="#606060", alpha=.12)
    ax.set_xlim(0, metadata["duration_s"])


def _curves(plt, path, names, values, arrays, metadata, title):
    panels = max(1, (len(names) + 5) // 6)
    fig, axes = plt.subplots(panels, 1, figsize=(13, max(3, panels * 2.6)), sharex=True, squeeze=False)
    times = arrays["video_start_s"] + .025
    for panel, ax in enumerate(axes[:, 0]):
        columns = list(range(panel * 6, min((panel + 1) * 6, len(names))))
        for column in columns:
            missing = not np.isfinite(values[:, column]).any()
            ax.plot(times, values[:, column], linewidth=.9, label=str(names[column]) + (" [N/A: no neurons]" if missing else ""))
        _annotate(ax, metadata)
        ax.set_ylabel("Hz / neuron")
        ax.grid(alpha=.18)
        if columns:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
        else:
            ax.text(.5, .5, "No configured groups", ha="center", transform=ax.transAxes)
    axes[-1, 0].set_xlabel("AVI playback time (s); raw 50 ms windows; no smoothing")
    fig.suptitle(title + "\nSimulated activity | shaded stimulus intervals; grey = warmup", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_plots(directory: Path, arrays: dict, metadata: dict):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _curves(plt, directory / "input_drive.png", arrays["input_names"], arrays["input_drive_hz"], arrays, metadata,
            "Encoded external input drive (not neuron firing)")
    _curves(plt, directory / "group_rates.png", arrays["group_names"], arrays["group_rates_hz"], arrays, metadata,
            "Actual simulated firing — full mean within each configured group")
    fig, ax = plt.subplots(figsize=(13, 7))
    if arrays["neuron_indices"].size:
        rates = arrays["neuron_rates_hz"].T
        image = ax.imshow(rates, aspect="auto", interpolation="nearest", origin="lower",
                          extent=(0, metadata["duration_s"], -.5, rates.shape[0] - .5), cmap="magma", vmin=0,
                          vmax=max(20, float(rates.max())))
        fig.colorbar(image, ax=ax, label="Firing Hz / neuron (1 spike per window = 20 Hz)")
        positions = np.unique(np.linspace(0, rates.shape[0] - 1, min(12, rates.shape[0]), dtype=int))
        ax.set_yticks(positions, [str(arrays["neuron_indices"][index]) for index in positions])
        ax.set_ylabel("Sampled global neuron index (selection order)")
    else:
        ax.text(.5, .5, "No sampled neurons: all configured groups are empty", ha="center", transform=ax.transAxes)
        ax.set_xlim(0, metadata["duration_s"])
    for interval in metadata["stimuli"]:
        ax.axvline(interval["start_s"], color="#70cde3", linewidth=.6, alpha=.7)
        ax.axvline(interval["end_s"], color="#70cde3", linewidth=.6, alpha=.7)
    ax.set_xlabel("AVI playback time (s); raw 50 ms windows, not a spike raster")
    ax.set_title("Sampled simulated neuron firing\nBounded deterministic sample; not representative of the whole brain")
    fig.tight_layout()
    fig.savefig(directory / "neuron_heatmap.png", dpi=140)
    plt.close(fig)


def _html_report(directory, report):
    def escape(value):
        return html.escape(str(value))

    p = report["neural_provenance"]
    intervals = "".join(f"<tr><td>{item['start_s']:g}–{item['end_s']:g}</td><td>{escape(item['stimulus'])}</td></tr>"
                        for item in report["stimuli"])
    rows = []
    for row in report["stimulus_summary"]:
        if row["scope"] != "stimulus":
            continue
        mean = "N/A" if row["mean_hz"] is None else f"{row['mean_hz']:.3f}"
        delta = "N/A" if row["delta_baseline_hz"] is None else f"{row['delta_baseline_hz']:+.3f}"
        rows.append(f"<tr><td>{escape(row['stimulus'])}</td><td>{escape(row['signal'])}</td><td>{escape(row['group'])}</td>"
                    f"<td>{row['interval_count']}</td><td>{mean}</td><td>{delta}</td></tr>")
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>CNSBridge neural activity — {escape(report['session']['session_id'])}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 20px;color:#203040;background:#fbfcfe}}
img{{max-width:100%;background:white;border:1px solid #dae1e7}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid #dde3eb}}.note{{background:#edf3f8;padding:16px}}code{{overflow-wrap:anywhere}}</style>
<h1>Offline simulated neural activity</h1>
<p>Session: <b>{escape(report['session']['session_id'])}</b> · Scene: {escape(report['session']['scene_id'])}<br>
Network: {escape(p['network']['name'])} · {report['frame_count']} frames · {report['duration_s']:g} s video · {p['window_ms']:g} ms simulation/frame</p>
<p class="note">{escape(report['claim_scope'])}. Configuration includes engineered input mapping
and bias: <code>{escape(json.dumps(p['effective_config']['brain'].get('bias', {})))}</code>.
Raw recordings include warmup; comparisons exclude the first {report['warmup_s']:g} s and windows crossing interval boundaries.</p>
<p>Missing groups: {escape(', '.join(p['missing_groups']) or 'none')}. N/A denotes missing neurons or no eligible observations;
0 Hz means matched neurons with no spikes. Input drive is external stimulation, not firing.</p>
<p><a href="activity.npz">Raw arrays (NPZ)</a> · <a href="report.json">Metadata and summary (JSON)</a> ·
<a href="group_rates.csv">Window signals (CSV)</a> · <a href="stimulus_summary.csv">Interval/stimulus comparison (CSV)</a></p>
<h2>Stimulus intervals</h2><p>AVI playback seconds, [start, end). Unmarked gaps are not baseline. Grey plot shading = warmup.</p>
<table><tr><th>Video seconds</th><th>Stimulus</th></tr>{intervals}</table>
<h2>Encoded input drive</h2><img src="input_drive.png" alt="External encoded input drive in Hz, by group">
<h2>Group firing</h2><img src="group_rates.png" alt="Actual simulated firing rate in Hz, by group">
<h2>Sampled neurons</h2><img src="neuron_heatmap.png" alt="Heatmap of sampled neuron firing in 50 millisecond windows">
<p>Selection: {escape(p['sampling']['method'])}; {len(p['neurons'])} sampled neurons. This heatmap is not a full-brain raster.
Simulation milliseconds, video seconds and host dequeue seconds are saved separately. Processing duration is not biological response latency.</p>
<h2>Descriptive stimulus comparison</h2><p>Equal weight per eligible interval, then compare with explicitly labelled baseline intervals.
No baseline means no delta; no significance tests. Interval-level details are in the CSV.</p>
<table><tr><th>Stimulus</th><th>Signal</th><th>Group</th><th>Valid intervals</th><th>Mean Hz</th><th>Δ baseline Hz</th></tr>{''.join(rows)}</table>
<p>{'No stimulus labels supplied; only time-series observations are available.' if not report['stimuli'] else ''}</p>
</html>"""
    (directory / "index.html").write_text(text, encoding="utf-8")


def write_report(directory: Path, arrays: dict, metadata: dict):
    summary = summarize(arrays, metadata)
    report = {**metadata, "stimulus_summary": summary}
    render_plots(directory, arrays, metadata)
    with (directory / "activity.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    with (directory / "group_rates.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        timing = ("frame_index", "video_start_s", "video_end_s", "host_dequeue_s", "simulation_start_ms", "simulation_end_ms", "processing_ms")
        writer.writerow([*timing, *(f"input_drive_hz:{name}" for name in arrays["input_names"]),
                         *(f"firing_hz:{name}" for name in arrays["group_names"])])
        for index in range(metadata["frame_count"]):
            writer.writerow([*(arrays[key][index] for key in timing),
                             *(_cell(value) for value in arrays["input_drive_hz"][index]),
                             *(_cell(value) for value in arrays["group_rates_hz"][index])])
    with (directory / "stimulus_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _html_report(directory, report)
    return report
