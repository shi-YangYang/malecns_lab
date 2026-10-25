"""Offline contracts; no GPU model, simulator, or locked-test files are opened."""
from dataclasses import replace
from copy import deepcopy

import numpy as np
import pytest

from malecns_lab.control.x2_behavior import Observation, digest
from malecns_lab.learning.x2_behavior_data import generate_dataset
from malecns_lab.learning.x2_yaw_diagnostic import (
    BEHAVIORS, PHASES, RelativeHeadingEncoder, budget_projection, fit_readout,
    output_directory, predict, score_complete, select_complete,
)
from malecns_lab.senses.x2_behavior import CHANNELS, channel_values


class EncodingFixture:
    """Only tests channel masking, never supplies a graph or production policy."""
    def __init__(self):
        self.channel_indices = tuple(c for c in range(57) for _ in range(60 if c < 14 else 59))
        self.input_body_ids = tuple(range(3377))
        self.input_indices = tuple(range(3377))
        payload = {"channels": list(CHANNELS), "channel_indices": list(self.channel_indices)}
        self._identity = {**payload, "input_map_id": digest(payload)}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def encode(self, observation):
        values = list(channel_values(observation).values())
        return np.array([values[c] for c in self.channel_indices], dtype=np.float32)


def test_only_absolute_yaw_is_removed_without_changing_observation_or_heading_error():
    raw = generate_dataset()["splits"]["train"][0]["frames"][0]["observation"]
    observation = Observation.from_dict(raw)
    encoder = RelativeHeadingEncoder(EncodingFixture())
    changed_yaw = replace(observation, feedback=replace(observation.feedback, yaw_rad=-1.7))
    np.testing.assert_array_equal(encoder.encode(observation), encoder.encode(changed_yaw))
    assert observation.to_dict() == raw
    assert encoder.input_map_id != encoder.parent_input_map_id
    changed_error = replace(observation, feedback=replace(observation.feedback, heading_error_rad=-.45))
    assert np.any(encoder.encode(observation) != encoder.encode(changed_error))
    assert len(encoder.suppressed_indices) == 118


def test_selection_requires_all_complete_frames_and_budget_cannot_shorten_them():
    data = generate_dataset()["splits"]
    train, validation = select_complete(data["train"], "train"), select_complete(data["validation"], "validation")
    assert [len(e["frames"]) for e in train] == [6, 11, 11, 11, 3]
    assert [len(e["frames"]) for e in validation] == [8, 15, 15, 15, 4]
    truncated = deepcopy(data["train"])
    next(e for e in truncated if e["family_id"] == "family-007" and e["task"] == "GREET")["frames"].pop()
    with pytest.raises(ValueError, match="frame count"):
        select_complete(truncated, "train")
    assert budget_projection(10, 52., 530.)["can_finish"]
    assert not budget_projection(10, 65., 530.)["can_finish"]


def test_ridge_statistics_are_train_only_and_validation_does_not_refit():
    rng = np.random.default_rng(9)
    train = rng.normal(-52., 2., (8, 2628))
    targets = [{"behavior": BEHAVIORS[i % 5], "phase": PHASES[i % 3], "intensity": .5} for i in range(8)]
    readout = fit_readout(train, targets)
    before = {k: v.copy() for k, v in readout.items()}
    np.testing.assert_allclose(readout["center"], train.mean(0))
    np.testing.assert_allclose(readout["scale"], np.maximum(train.std(0), .1))
    predictions = predict(readout, np.full((4, 2628), 1000.))
    assert len(predictions) == 4
    for key in before:
        np.testing.assert_array_equal(readout[key], before[key])
    assert all(np.isfinite(row["intensity"]) for row in predictions)


def test_complete_scoring_rejects_missing_frames_and_records_each_task():
    episodes = select_complete(generate_dataset()["splits"]["validation"], "validation")
    rows = [dict(f["target"]) for e in episodes for f in e["frames"]]
    scored = score_complete(episodes, rows)
    assert scored["scores"]["sequence_matches"] == 5
    assert len(scored["scores"]["episodes"]) == 5
    with pytest.raises(ValueError):
        score_complete(episodes, rows[:-1])


def test_output_is_new_dedicated_root_child(tmp_path):
    (tmp_path / "data").mkdir()
    valid = tmp_path / "data/x2_behavior_short_trial_yaw_new"
    assert output_directory(valid, tmp_path) == valid
    with pytest.raises(ValueError):
        output_directory(tmp_path / "data/x2_behavior_short_trial_other", tmp_path)
    with pytest.raises(ValueError):
        output_directory(tmp_path / "data/archive/x2_behavior_short_trial_yaw_nested", tmp_path)
    valid.mkdir()
    with pytest.raises(ValueError):
        output_directory(valid, tmp_path)
