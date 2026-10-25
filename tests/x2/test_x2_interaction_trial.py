"""CPU contract checks only; no complete graph, MPS, VM, or robot is started."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from malecns_lab.learning.x2_interaction_policy import (
    SCHEMA, bind_interaction_policy, fold_readout, verify_fold,
)
from malecns_lab.learning.x2_interaction_trial import (
    EXPECTED_FRAMES, check_dataset, configuration, output_directory, score_requests,
)
from malecns_lab.learning.x2_interaction_data import generate_dataset
from malecns_lab.learning.x2_yaw_diagnostic import fit_readout
from malecns_lab.control.x2_behavior import digest
from malecns_lab.senses.x2_behavior import CHANNELS


def test_preview_does_not_import_numerical_models_or_open_locked_test():
    script = ("import sys; from malecns_lab.learning.x2_interaction_trial import main; "
              "main([]); assert not any(n in sys.modules for n in ('torch','numpy','rclpy'))")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    value = json.loads(done.stdout)
    assert value["mode"] == "preview" and value["brain_loaded"] is False
    assert value["configuration"]["wall_seconds"] == 600
    assert value["configuration"]["locked_test_opened"] is False


def test_fixed_complete_sessions_and_request_level_forward_scoring():
    dataset = check_dataset(generate_dataset())
    assert {k:sum(len(e["frames"]) for e in v) for k,v in dataset["splits"].items()} == EXPECTED_FRAMES
    episodes = dataset["splits"]["validation"]
    rows = [dict(f["target"]) for e in episodes for f in e["frames"]]
    perfect = score_requests(episodes, rows)
    assert perfect["forward_candidate_pass"] and perfect["forward_starts"] == 2
    assert perfect["request_total"] == 7 and perfect["end_sequences"] == 2
    assert perfect["greet_motion_qualified"] is perfect["continue_qualified"] is False
    assert any(r["capability_unavailable"] for r in perfect["requests"])
    first = next(i for i,f in enumerate(episodes[0]["frames"]) if f["target"]["behavior"] == "ADVANCE")
    rows[first] = {**rows[first], "behavior": "HOLD"}
    failed = score_requests(episodes, rows)
    assert not failed["forward_candidate_pass"] and failed["forward_starts"] == 1
    with pytest.raises(ValueError, match="Missing complete"):
        score_requests(episodes, rows[:-1])


def test_fold_preserves_trained_raw_logits_in_existing_float32_coordinates():
    import torch
    rng = np.random.default_rng(2401)
    # Wide membrane range and a subset at the variance floor exercise bias
    # cancellation without substituting this fixture for actual MPS validation.
    train = rng.normal(-180., 70., (24, 2628)).astype(np.float32)
    train[:, :300] = -52.+rng.normal(0, .01, (24, 300))
    targets = [{"behavior": ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT")[i % 4],
                "phase": ("START", "FINISH")[i % 2], "intensity": .3+.1*(i % 4)} for i in range(24)]
    readout = fit_readout(train, targets)
    arrays = fold_readout(readout, v_rest=-52., span=7.)
    validation = rng.normal(-180., 70., (10, 2628)).astype(np.float32)
    validation[:, :300] = -52.+rng.normal(0, .01, (10, 300))
    all_features = np.concatenate((train, validation))
    x = (torch.from_numpy(all_features)+52.)/7.
    actual = np.concatenate([torch.nn.functional.linear(x, torch.from_numpy(arrays[name+".weight"]),
        torch.from_numpy(arrays[name+".bias"])).numpy() for name in ("behavior_head", "phase_head", "parameter_head")], axis=1)
    audit, _ = verify_fold(readout, all_features, actual)
    assert audit["passed"] and audit["frames"] == 34
    altered = actual.copy()
    altered[0, 0] += .05
    assert not verify_fold(readout, all_features, altered)[0]["passed"]
    altered[0, 0] = np.nan
    assert not verify_fold(readout, all_features, altered)[0]["finite"]


class EncodingFixture:
    input_body_ids = tuple(range(3377))
    input_indices = tuple(range(3377))
    channel_indices = tuple(c for c in range(57) for _ in range(60 if c < 14 else 59))

    @property
    def identity(self):
        value = {"channels": list(CHANNELS), "channel_indices": list(self.channel_indices)}
        return {**value, "input_map_id": digest(value)}


class PolicyFixture:
    def __init__(self, identity):
        self._identity = deepcopy(identity)
        self.encoder = EncodingFixture()

    @property
    def identity(self):
        return deepcopy(self._identity)


def test_shared_loader_reconstructs_new_identity_before_unchanged_strict_restore(tmp_path, monkeypatch):
    from malecns_lab.learning import x2_inference as frozen, x2_behavior_training as training
    from malecns_lab.learning.x2_behavior_policy import X2BehaviorPolicy, backward_identity
    from malecns_lab.learning.x2_behavior_session import checkpoint_registration_metadata, load_candidate_policy
    base = {"schema": "malecns_lab.x2_behavior_policy.v1", "policy_id": "base",
        "backward": backward_identity(.1), "initialization": {"theta": "migrated", "head_seed": 2401}}
    source = PolicyFixture(base)
    bind_interaction_policy(source, source_checkpoint_id="a"*64)
    metadata = {"checkpoint_id": "saved", "policy": source.identity}
    monkeypatch.setattr(frozen, "read_anchors", lambda *_:{"run": {"policy": {}}, "checkpoint": {}})
    monkeypatch.setattr(frozen, "load_parameters", lambda *_:{"theta": None})
    monkeypatch.setattr(X2BehaviorPolicy, "from_reference", lambda *a, **k:PolicyFixture(base))
    calls = []
    def strict(path, policy):
        assert policy.identity == metadata["policy"]
        calls.append(path)
        return metadata
    monkeypatch.setattr(training, "restore_checkpoint", strict)
    registered = checkpoint_registration_metadata(metadata)
    assert "input_map" not in registered["policy"]
    loaded = load_candidate_policy(tmp_path, registered, tmp_path / "checkpoint")
    assert loaded.identity == metadata["policy"] and calls == [tmp_path / "checkpoint"]
    altered = deepcopy(registered)
    altered["policy"]["interaction_profile"]["wave_enabled"] = True
    with pytest.raises(ValueError, match="profile differs"):
        load_candidate_policy(tmp_path, altered, tmp_path / "checkpoint")
    assert len(calls) == 1


@pytest.mark.parametrize("interaction", [False, True])
def test_large_real_policy_registration_stays_within_transport_budget(tmp_path, monkeypatch, interaction):
    from malecns_lab.learning import x2_behavior_training as training
    from malecns_lab.learning.x2_behavior_session import (
        MAX_PACKET_BYTES, atomic, checkpoint_registration_metadata, read_json,
    )
    checkpoint = Path(__file__).resolve().parents[2] / "data/x2_behavior_readout_20261023_r2/checkpoints/0149"
    if not checkpoint.is_dir():
        pytest.skip("Archived 0149 checkpoint is a local research asset")
    metadata = training.inspect_checkpoint(checkpoint)  # Read-only identity and array hash check.
    if interaction:
        policy = PolicyFixture(metadata["policy"])
        mapping = metadata["policy"]["input_map"]
        policy.encoder = SimpleNamespace(identity=deepcopy(mapping),
            **{name: mapping[name] for name in ("input_body_ids", "input_indices", "channel_indices")})
        bind_interaction_policy(policy, source_checkpoint_id=metadata["checkpoint_id"])
        metadata = {**metadata, "policy": policy.identity}
    assert len(json.dumps(metadata["policy"], separators=(",", ":")).encode()) >= 491270
    with pytest.raises(ValueError, match="packet exceeds budget"):
        atomic(tmp_path / "oversized.json", {"checkpoint_metadata": metadata})
    registered = checkpoint_registration_metadata(metadata)
    path = tmp_path / "behavior_registration.json"
    atomic(path, {"checkpoint_metadata": registered})
    assert path.stat().st_size < MAX_PACKET_BYTES == 256*1024
    assert read_json(path)["checkpoint_metadata"] == registered
    expected = {"schema", "policy_id", "initialization", "backward"}
    if interaction:
        expected |= {"source_checkpoint_id", "interaction_profile"}
    assert set(registered["policy"]) == expected
    assert all(registered["policy"][key] == metadata["policy"][key] for key in expected)
    # Fields omitted from transport still participate in strict disk restoration.
    changed = deepcopy(metadata["policy"])
    changed["implementation_sha256"] = "0"*64
    monkeypatch.setattr(training, "inspect_checkpoint", lambda _: metadata)
    with pytest.raises(ValueError, match="model/input/source identity differs"):
        training.restore_checkpoint(checkpoint, SimpleNamespace(identity=changed))


def test_candidate_schema_is_explicit_and_output_cannot_touch_old_assets(tmp_path, monkeypatch):
    from malecns_lab.runtime.x2.behavior import candidate
    from malecns_lab.learning import x2_behavior_training as training
    data = tmp_path / "data"
    checkpoint = data / "existing"
    checkpoint.mkdir(parents=True)
    metadata = {"policy": {"schema": SCHEMA}}
    monkeypatch.setattr(training, "inspect_checkpoint", lambda *_:metadata)
    assert candidate(tmp_path, checkpoint) == (checkpoint, metadata)
    metadata["policy"]["schema"] = "unregistered"
    with pytest.raises(ValueError, match="checkpoint required"):
        candidate(tmp_path, checkpoint)
    output = data / "x2_behavior_short_trial_interaction_new"
    assert output_directory(output, tmp_path, inputs=(checkpoint,)) == output
    with pytest.raises(ValueError, match="direct data"):
        output_directory(checkpoint / output.name, tmp_path)
    with pytest.raises(ValueError, match="dedicated"):
        output_directory(data / "x2_behavior_short_trial_other", tmp_path)
    assert not output.exists() and configuration()["automatic_deployment"] is False
