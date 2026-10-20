import json

import numpy as np
import pytest

from malecns_lab.diagnostics.report_v2 import summarize_run


def test_single_recording_with_zero_eligible_windows_preserves_coverage(tmp_path):
    segment = tmp_path / "segment_000"
    segment.mkdir()
    np.savez(segment / "rows.npz", valid=np.array([True, False]), prediction_valid=np.array([False, False]),
             visual_features=np.full((2, 14), np.nan), neural_rates_hz=np.full((2, 33), np.nan),
             group_rates_hz=np.full((2, 1), np.nan), input_event_counts=np.full((2, 14), -1),
             network_silent_fraction=np.array([0.9, np.nan]), network_high_rate_fraction=np.array([0.0, np.nan]))
    manifest = {"format": "cnsbridge.neural_replay_run.v2", "model_source_id": "identity", "run_seed": 1,
                "model_provenance": {"vision": {"channels": [str(i) for i in range(14)]},
                    "group_mapping": {"output_body_ids": list(range(33)), "rules": {"out": {"role": "output"}}}}}
    coverage = {"planned_frames": 50, "valid_input_frames": 1, "failed_frames": 1,
                "unprocessed_tail_frames": 48, "prediction_valid_frames": 0}
    summary = {"wall_seconds": 1.0, "peak_process_rss_bytes": 100,
               "segments": [{"segment": {"segment_id": "segment_000"}, "source_id": "identity", "groups": ["out"],
                    "input_source": {"kind": "recorded_video", "scene_id": "real_scene", "session_id": "s", "family_id": "f", "total_frames": 60},
                    "status": "segment_stopped", "coverage": coverage, "computation": {}}]}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    report = summarize_run(tmp_path)
    row = report["segments"][0]
    assert row["label"].startswith("real_scene")
    assert row["eligible_prediction_windows"] == 0 and row["output_mean_hz"] is None
    assert row["input_mean_features"] is None and row["input_events_eligible_windows"] is None
    assert report["source_coverage"][0]["outside_predeclared_segments"] == 10
    json.dumps(report, allow_nan=False)
    summary["segments"][0]["source_id"] = "other"
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="model identity"):
        summarize_run(tmp_path)
