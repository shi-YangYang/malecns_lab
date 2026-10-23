"""Whole-episode registration and teacher boundaries; no neural execution."""
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from malecns_lab.control.x2_behavior import Behavior as B, ExecutionStatus as S, Feedback, Observation, Task as T
from malecns_lab.learning.x2_behavior_data import generate_dataset, load_split, score_predictions, teacher_step, write_dataset


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


def test_module_import_is_inert():
    result = subprocess.run([sys.executable, "-c", "import sys; import malecns_lab.learning.x2_behavior_data; "
                             "assert not {'torch','numpy','scipy','rclpy'}.intersection(sys.modules)"], check=False)
    assert result.returncode == 0


def test_complete_episode_splits_and_families_are_separate(dataset):
    assert dataset["identity"]["split_counts"] == {"train": 200, "validation": 50, "locked_test": 100}
    seen = set()
    for split, episodes in dataset["splits"].items():
        families = {e["family_id"] for e in episodes}
        assert not seen.intersection(families)
        seen.update(families)
        for task in T:
            selected = [e for e in episodes if e["task"] == task]
            assert len(selected) == len(episodes) // 5
            assert all(len(e["frames"]) >= 2 for e in selected)
        for e in episodes:
            assert all("target" not in row["observation"] for row in e["frames"])
            assert all(row["observation"]["perception"]["valid"] is False for row in e["frames"])
            assert e["frames"][-1]["target"]["behavior"] == "HOLD"


def test_teacher_uses_current_feedback_and_memory_not_static_task_table():
    base = Observation("r", "s", 0, "q", T.FACE_LEFT, 0.,
                       Feedback(valid=True, source="simulated_execution_feedback", standing=True, heading_error_rad=.5),
                       capabilities=tuple(B))
    memory = {}
    assert teacher_step(base, memory)["behavior"] == "TURN_LEFT"
    executing = replace(base, feedback=replace(base.feedback, status=S.RUNNING, behavior=B.TURN_LEFT, standing=False, execution_id="e", request_id="q"))
    assert teacher_step(executing, memory) == {"behavior": "TURN_LEFT", "phase": "CONTINUE", "intensity": .5}
    complete = replace(base, feedback=replace(base.feedback, status=S.COMPLETED, behavior=B.TURN_LEFT, execution_id="e", request_id="q"))
    assert teacher_step(complete, memory)["behavior"] == "WAVE_RIGHT"
    assert teacher_step(replace(base, task=T.END_INTERACTION, request_id="end"), memory)["behavior"] == "HOLD"


def test_rejected_and_missing_feedback_never_fake_success(dataset):
    episodes = dataset["splits"]["locked_test"]
    for e in episodes:
        if e["outcome"] in ("rejected", "missing_feedback"):
            assert e["frames"][-1]["target"]["behavior"] == "HOLD"
            assert not any(row["observation"]["feedback"]["status"] == "COMPLETED" for row in e["frames"])


def test_split_loading_never_opens_locked_test_and_detects_tampering(tmp_path):
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    assert len(manifest["identity"]["dataset_id"]) == 64
    # Removing locked-test does not affect reading training/validation.
    (directory / "locked-test.json").unlink()
    assert len(load_split(directory, "train")) == 200
    assert len(load_split(directory, "validation")) == 50
    (directory / "train.json").write_text("[]")
    with pytest.raises(ValueError, match="file hash"):
        load_split(directory, "train")
    with pytest.raises(FileExistsError):
        write_dataset(directory)


def test_scoring_reports_raw_sequences_and_no_protection_credit(dataset):
    episodes = dataset["splits"]["locked_test"]
    predictions = {e["episode_id"]: [dict(f["target"]) for f in e["frames"]] for e in episodes}
    result = score_predictions(episodes, predictions)
    assert result["sequence_matches"] == 100 and result["behavior_macro_f1"] == result["phase_macro_f1"] == 1
    predictions[episodes[0]["episode_id"]][0]["protection_reason"] = "emergency"
    assert score_predictions(episodes, predictions)["sequence_matches"] == 99
