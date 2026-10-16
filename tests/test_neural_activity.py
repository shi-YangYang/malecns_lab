"""Focused offline diagnostics contracts; no SDK or device interaction."""

import builtins
import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np
import pytest
from scipy import sparse

from cnsbridge.brain.connectome import Connectome
from cnsbridge.config import default_config
from cnsbridge.diagnostics import activity, collector, reporting, stimuli


def recording(directory, frames=6, movie_frames=None):
    directory.mkdir(parents=True)
    manifest = {
        "format": "flydrones.ground_recording", "version": 1,
        "session_id": directory.name, "scene_id": "synthetic_checkerboard_software_test",
        "ground_only": True, "video": "video.avi", "frames": "frames.csv", "labels": "labels.csv",
        "nominal_fps": 20, "timing": "host_dequeue", "label_timebase": "video_playback",
    }
    (directory / "session.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (directory / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_index", "t_s", "video_t_s"))
        writer.writerows((index, index * .12, index / 20) for index in range(frames))
    video = cv2.VideoWriter(str(directory / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20, (96, 72))
    assert video.isOpened()
    y, x = np.indices((72, 96))
    for index in range(frames if movie_frames is None else movie_frames):
        offset = (index - 20) * 3 if 20 <= index < 40 else 0
        pattern = (((x + offset) // 8 + y // 8) % 2 * 200).astype(np.uint8)
        video.write(np.stack((pattern, 255 - pattern, pattern), axis=-1))
    video.release()
    # Deliberately no labels.csv: diagnostics must never open action labels.
    return directory


def write_stimuli(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("start_s", "end_s", "stimulus"))
        writer.writerows(rows)
    return path


def tiny_graph():
    return Connectome("tiny_known_counts", sparse.csc_matrix((4, 4)), np.array(["A", "A", "B", "C"]),
                      np.array(["L", "L", "R", "R"]), body_ids=np.array([9007199254740993, 2, 3, 4], dtype=np.int64))


def tiny_config(empty=False):
    cfg = default_config()
    cfg["inputs"] = {"a": {"types": ["A"], "feature": "brightness", "max_hz": 10},
                     "missing": {"types": ["absent"], "feature": "brightness"}}
    cfg["outputs"] = {"b": {"types": ["B"]}, "overlap": {"types": ["A"]}, "quiet": {"types": ["C"]}}
    cfg["brain"]["bias"] = {}
    if empty:
        cfg["inputs"] = {"missing_input": {"types": ["absent"], "feature": "brightness"}}
        cfg["outputs"] = {"missing_output": {"types": ["absent"]}}
    return cfg


def test_known_counts_units_missing_groups_sampling_and_identity():
    cfg = tiny_config()
    with patch.object(collector, "load_connectome", side_effect=lambda _: tiny_graph()), pytest.warns(UserWarning):
        engine = collector.ActivityBrain(cfg, max_neurons=4)
    np.testing.assert_array_equal(engine.indices, [0, 2, 1, 3])
    assert len(np.unique(engine.indices)) == 4
    assert engine.brain.record.size == 0
    engine.encoder.encode = lambda *_args, **_kwargs: {"a": np.array([5, 15])}

    def tick(_rates, ms):
        engine.brain.last_counts = np.array([1, 3, 0, 0])
        engine.brain.net.t_ms += ms

    engine.brain.tick = tick
    drive, rates, counts, start, end = engine.step(np.zeros((72, 96, 3), dtype=np.uint8))
    np.testing.assert_allclose(drive, [10, np.nan], equal_nan=True)
    np.testing.assert_allclose(rates, [40, np.nan, 0, 40, 0], equal_nan=True)
    np.testing.assert_array_equal(counts, [1, 0, 3, 0])
    assert (start, end) == (0, 50)
    assert engine.provenance["neurons"][0]["body_id"] == "9007199254740993"
    assert engine.provenance["missing_groups"] == ["missing"]
    assert engine.provenance["groups"][1]["indices"] == []
    assert engine.provenance["effective_config"] == cfg
    assert len(engine.provenance["network"]["graph_sha256"]) == 64
    np.testing.assert_array_equal(collector.select_neurons(engine.groups, 2), [0, 2])
    assert collector.select_neurons({"empty": np.array([], dtype=int)}, 1).size == 0


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf"), 3, 1e15, .5000000001])
def test_invalid_dt_fails_before_graph_load(dt):
    cfg = tiny_config()
    cfg["brain"]["lif"]["dt"] = dt
    with patch.object(collector, "load_connectome") as load, pytest.raises(ValueError, match="divide"):
        collector.ActivityBrain(cfg)
    load.assert_not_called()


@pytest.mark.parametrize("maximum", [0, 4097, 2.5, True])
def test_sample_bound(maximum):
    with pytest.raises(ValueError, match="max_neurons"):
        collector.select_neurons({}, maximum)


def test_stimulus_validation_and_interval_first_comparison(tmp_path):
    path = tmp_path / "stimuli.csv"
    for rows in ([(0, .5, "")], [(0, .5, "a"), (.4, .6, "b")], [(0, float("nan"), "a")],
                 [(-.1, .2, "a")], [(0, 1.1, "a")], [(0, 0, "a")]):
        with pytest.raises(ValueError):
            stimuli.read_stimuli(write_stimuli(path, rows), 1)
    intervals = stimuli.read_stimuli(write_stimuli(path, [(0, .1, "baseline"), (.11, .25, "move"),
                                                        (.3, .35, "move"), (.4, .45, "baseline")]), 1)
    starts = np.arange(10) / 20
    arrays = {"video_start_s": starts, "video_end_s": starts + .05,
              "input_names": np.array(["a"]), "group_names": np.array(["a", "missing"]),
              "input_drive_hz": np.arange(10)[:, None] * 10,
              "group_rates_hz": np.column_stack((np.arange(10) * 20, np.full(10, np.nan)))}
    metadata = {"stimuli": intervals, "warmup_s": .05}
    rows = stimuli.summarize(arrays, metadata)
    selected = [row for row in rows if row["signal"] == "group_firing" and row["group"] == "a"]
    # First baseline excludes warmup; movement starts at .11, excluding the .10-.15 mixed window.
    assert [row["window_count"] for row in selected[:4]] == [1, 2, 1, 1]
    assert selected[1]["mean_hz"] == 70  # .15 and .20 windows
    moving = next(row for row in selected if row["scope"] == "stimulus" and row["stimulus"] == "move")
    assert moving["mean_hz"] == 95  # equally weight 70 and 120; not three correlated windows
    assert moving["delta_baseline_hz"] == 5  # baseline interval means 20 and 160 -> 90
    assert moving["interval_count"] == 2
    assert all(row["mean_hz"] is None for row in rows if row["group"] == "missing")
    no_baseline = stimuli.summarize(arrays, {"stimuli": intervals[1:3], "warmup_s": 0})
    assert all(row["delta_baseline_hz"] is None for row in no_baseline)
    assert stimuli.summarize(arrays, {"stimuli": [], "warmup_s": 0}) == []


@pytest.mark.parametrize("movie_frames", [4, 8])
def test_video_mismatch_removes_output(tmp_path, movie_frames):
    session = recording(tmp_path / "session", movie_frames=movie_frames)
    output = tmp_path / "report"
    with patch.object(collector, "load_connectome", side_effect=lambda _: tiny_graph()), pytest.warns(UserWarning):
        with patch.object(activity, "analyze", wraps=activity.analyze), pytest.raises(ValueError, match="Video"):
            activity.analyze(session, output, brain="minifly", max_neurons=2)
    assert not output.exists()


def test_all_empty_groups_and_no_annotations_still_report(tmp_path):
    session = recording(tmp_path / "empty", frames=2)
    with patch.object(collector, "load_connectome", side_effect=lambda _: tiny_graph()), pytest.warns(UserWarning):
        arrays, metadata = collector.collect(session, tiny_config(empty=True), max_neurons=2)
    assert np.isnan(arrays["group_rates_hz"]).all()
    assert np.isnan(arrays["input_drive_hz"]).all()
    assert arrays["neuron_counts"].shape == (2, 0)
    output = tmp_path / "empty_report"
    output.mkdir()
    report = reporting.write_report(output, arrays, metadata)
    assert report["stimulus_summary"] == []
    assert "No stimulus labels supplied" in (output / "index.html").read_text()


def test_help_without_optional_dependencies_and_missing_dependency_error(tmp_path):
    script = """
import builtins, runpy, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('cv2', 'matplotlib', 'djitellopy'):
        raise ImportError('blocked optional dependency ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = ['activity', '--help']
runpy.run_module('cnsbridge.diagnostics.activity', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--out-dir" in result.stdout
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked), pytest.raises(ValueError, match="analysis"):
        activity.analyze(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_minifly_cli_exports_repeatable_ground_video_without_action_labels():
    # Retain this single synthetic E2E artifact for coordinator image inspection.
    project = Path(__file__).resolve().parents[1]
    demo = project / "data" / "neural_activity_demo" / f"synthetic_{uuid4().hex[:10]}"
    session = recording(demo / "synthetic_session", frames=60)
    annotations = write_stimuli(session / "stimuli.csv", [(0, 1, "baseline"), (1, 2, "checkerboard_right"), (2, 3, "baseline")])
    output = demo / "report"
    assert activity.main([str(session), "--out-dir", str(output), "--brain", "minifly", "--stimuli", str(annotations),
                          "--max-neurons", "16", "--warmup-s", ".2"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["frame_count"] == 60
    assert report["session"]["scene_id"] == "synthetic_checkerboard_software_test"
    assert report["neural_provenance"]["sampling"]["max_neurons"] == 16
    assert "motor" not in report["units"]
    assert len(report["sources"]["video.avi"]["sha256"]) == 64
    with np.load(output / "activity.npz", allow_pickle=False) as saved:
        assert all(saved[name].dtype.kind != "O" for name in saved.files)
        assert saved["neuron_counts"].shape == (60, 16)
        np.testing.assert_allclose(saved["neuron_rates_hz"], saved["neuron_counts"] * 20)
        np.testing.assert_allclose(saved["video_start_s"], np.arange(60) / 20)
        np.testing.assert_allclose(saved["host_dequeue_s"], np.arange(60) * .12)
        np.testing.assert_allclose(saved["simulation_start_ms"], np.arange(60) * 50)
        np.testing.assert_allclose(saved["simulation_end_ms"], np.arange(1, 61) * 50)
        assert json.loads(saved["metadata_json"].item())["format"] == "cnsbridge.neural_activity"
    expected = ("activity.npz", "report.json", "group_rates.csv", "stimulus_summary.csv", "input_drive.png", "group_rates.png",
                "neuron_heatmap.png", "index.html")
    assert all((output / name).stat().st_size > 0 for name in expected)
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        activity.analyze(session, output, brain="minifly")
    assert (output / "report.json").read_bytes() == before
    print(f"SYNTHETIC VISUAL QA: {output}")
