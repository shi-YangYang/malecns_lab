"""Complete continuation on explicit small circuits, never full-brain evidence."""
from dataclasses import replace
import gc
import io
import json
from pathlib import Path
import shutil
import weakref
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cnsbridge.brain.metadata import canonical_hash, file_hashes
from cnsbridge.brain.tonic import LaminaTonicLIF
from cnsbridge.learning import training_checkpoint as checkpoint
from cnsbridge.learning import training_optimizer as adam
from cnsbridge.learning.neural_policy import NeuralPolicy, NeuralPolicyConfig
from cnsbridge.senses.spatial_drive import SpatialDriveConfig
from cnsbridge.senses.target import TargetCamera
from test_neural_policy import assert_same_state, circuit, frame

EXPECTED = {key: canonical_hash({"numerical_test": key}) for key in checkpoint.IDENTITIES}


def policy():
    source, mapping = circuit(dtype=torch.float32, tonic_mode="fixed")
    brain = LaminaTonicLIF.for_numerical_test(
        source.graph.counts, (0, 1), types=np.array(["R1-R6", "R1-R6", "DNtest", "L1", ""]),
        missing=np.array([False] * 4 + [True]), tonic_types=("L1",), mode="fixed", dtype=torch.float32)
    return NeuralPolicy.for_numerical_test(
        brain, mapping, background="fixed_lamina_8mv", camera=TargetCamera(),
        config=NeuralPolicyConfig(frame_ms=4, precondition_ms=2, control_dt_s=.1, readout_seed=123),
        drive_config=SpatialDriveConfig(max_rate_hz=1000))


def context(next_frame=2, parent=None):
    return {**EXPECTED, "scenes": [{"scene_id": "numerical-scene", "frame_count": 4}],
            "epoch": 0, "scene_index": 0, "scene_id": "numerical-scene", "next_frame": next_frame,
            "completed_windows": next_frame // 2, "optimizer_steps": next_frame // 2,
            "parent_checkpoint_id": parent, "log_prefix_sha256": canonical_hash({"logged_frames": next_frame})}


def update(model, optimizer, state):
    outputs = []
    for _ in range(2):
        result = model.forward_tensor(frame(), state)
        outputs.append(result)
        state = result.state
    action = torch.stack([out.action for out in outputs])
    loss = (action - action.new_tensor([.1, -.1, .2])).square().mean()
    loss.backward()
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], 1, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    state = state.detach()
    optimizer.zero_grad(set_to_none=True)
    return state, action.detach().clone(), torch.stack([out.features_mv.detach() for out in outputs]).clone()


def snapshot(model, optimizer):
    return ({name: value.detach().clone() for name, value in model.named_parameters()},
            {name: {key: value.detach().clone() for key, value in optimizer.state[param].items()}
             for name, param in model.named_parameters() if param in optimizer.state})


def equal_snapshots(left, right):
    assert left[0].keys() == right[0].keys() and left[1].keys() == right[1].keys()
    for key in left[0]:
        assert torch.equal(left[0][key], right[0][key])
    for name in left[1]:
        assert left[1][name].keys() == right[1][name].keys()
        for key in left[1][name]:
            assert torch.equal(left[1][name][key], right[1][name][key])


@pytest.mark.parametrize("mode", adam.MODES)
def test_two_updates_equal_destroy_rebuild_restore_with_nonzero_state(mode, tmp_path, monkeypatch):
    continuous = policy()
    optimizer_a = adam.make_training_optimizer(continuous, mode)
    initial_theta = continuous.brain.theta.detach().clone()
    initial_readout = continuous.readout.weight.detach().clone()
    state_a, _, _ = update(continuous, optimizer_a, continuous.initial_state())
    assert len(state_a.brain.pending) == 18
    assert any(torch.count_nonzero(value) for value in state_a.brain.pending)
    assert np.count_nonzero(state_a.drive.phase) and np.count_nonzero(state_a.drive.event_counts)
    assert state_a.brain.step == state_a.drive.step == 100 and state_a.frame_index == 2

    split = policy()
    optimizer_b = adam.make_training_optimizer(split, mode)
    state_b, _, _ = update(split, optimizer_b, split.initial_state())
    assert_same_state(state_a, state_b)
    path = tmp_path / mode
    metadata = checkpoint.save_training_checkpoint(split, optimizer_b, state_b, path, mode=mode, context=context())
    before_state = checkpoint.read_training_checkpoint(path, expected=EXPECTED).state
    reference = weakref.ref(split)
    del split, optimizer_b, state_b
    gc.collect()
    assert reference() is None

    restored = policy()
    optimizer_c = adam.make_training_optimizer(restored, mode)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resume must not construct or precondition a graph")
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.NeuralPolicy, "from_reference", forbidden)
        patch.setattr(restored, "initial_state", forbidden)
        state_c, cursor = checkpoint.restore_training_checkpoint(restored, optimizer_c, path, expected=EXPECTED)
    assert cursor == context()
    assert_same_state(state_a, state_c)
    equal_snapshots(snapshot(continuous, optimizer_a), snapshot(restored, optimizer_c))
    state_a, actions_a, features_a = update(continuous, optimizer_a, state_a)
    state_c, actions_c, features_c = update(restored, optimizer_c, state_c)
    assert torch.equal(actions_a, actions_c) and torch.equal(features_a, features_c)
    assert_same_state(state_a, state_c)
    equal_snapshots(snapshot(continuous, optimizer_a), snapshot(restored, optimizer_c))
    # Both full next cursors seal identically, including the parent link and logs.
    next_context = context(4, metadata["checkpoint_id"])
    left = checkpoint.save_training_checkpoint(continuous, optimizer_a, state_a, tmp_path / "a2", mode=mode, context=next_context)
    right = checkpoint.save_training_checkpoint(restored, optimizer_c, state_c, tmp_path / "c2", mode=mode, context=next_context)
    assert left == right
    assert_same_state(before_state, checkpoint.read_training_checkpoint(path, expected=EXPECTED).state)
    assert not torch.equal(initial_readout, restored.readout.weight)
    if mode == "frozen_brain":
        assert torch.equal(initial_theta, restored.brain.theta)
        assert "brain.theta" not in snapshot(restored, optimizer_c)[1]
    else:
        assert not torch.equal(initial_theta, restored.brain.theta)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    path = tmp_path_factory.mktemp("training-checkpoint") / "sealed"
    model = policy()
    optimizer = adam.make_training_optimizer(model, "joint")
    state, _, _ = update(model, optimizer, model.initial_state())
    checkpoint.save_training_checkpoint(model, optimizer, state, path, mode="joint", context=context())
    return path


def read_meta(path):
    return json.loads((path / "checkpoint.json").read_text())


def reseal(path, metadata):
    metadata["checkpoint_id"] = canonical_hash({key: value for key, value in metadata.items() if key != "checkpoint_id"})
    (path / "checkpoint.json").write_text(json.dumps(metadata, allow_nan=False))


def replace_arrays(path, metadata, arrays):
    np.savez(path / "training.npz", **arrays)
    metadata["arrays"] = checkpoint._declarations(arrays)
    metadata["files_sha256"]["training.npz"] = file_hashes(path / "training.npz")["sha256"]


@pytest.mark.parametrize("mutation", [
    "missing_file", "hash", "unsealed", "extra_field", "missing_field", "source", "runtime", "threads",
    "expected", "mode", "groups", "step_zero", "step_fraction", "step_disagree", "moment_dtype", "moment_shape",
    "moment_nan", "negative_second", "missing_moment", "extra_moment", "phase", "phase_dtype", "event_count",
    "queue_length", "queue_dtype", "refractory", "state_clock", "frame_clock", "drive_id", "scene_id", "scene_count",
    "odd_cursor", "windows", "bool_epoch", "parent", "log_digest", "nested_parameter", "duplicate_npz", "duplicate_json",
])
def test_bad_archives_rejected_atomically(mutation, sealed, tmp_path):
    path = tmp_path / mutation
    shutil.copytree(sealed, path)
    metadata = read_meta(path)
    with np.load(path / "training.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    expected = dict(EXPECTED)
    if mutation == "missing_file":
        (path / "training.npz").unlink()
    elif mutation == "hash":
        metadata["files_sha256"]["training.npz"] = "0" * 64
    elif mutation == "unsealed":
        (path / "checkpoint.json").unlink()
    elif mutation == "extra_field":
        metadata["unknown"] = 1
    elif mutation == "missing_field":
        metadata.pop("purpose")
    elif mutation == "source":
        metadata["implementation_sha256"]["learning/training_checkpoint.py"] = "0" * 64
    elif mutation in ("runtime", "threads"):
        metadata["runtime"]["python" if mutation == "runtime" else "num_threads"] = "wrong"
    elif mutation == "expected":
        expected["protocol_id"] = "0" * 64
    elif mutation == "mode":
        metadata["optimizer"]["mode"] = "unknown"
    elif mutation == "groups":
        metadata["optimizer"]["groups"][0]["lr"] = .02
    elif mutation.startswith("step_"):
        arrays["brain.theta.step"] = np.array({"step_zero": 0, "step_fraction": 1.5, "step_disagree": 2}[mutation], np.float32)
    elif mutation == "moment_dtype":
        arrays["brain.theta.exp_avg"] = arrays["brain.theta.exp_avg"].astype(np.float64)
    elif mutation == "moment_shape":
        arrays["brain.theta.exp_avg"] = arrays["brain.theta.exp_avg"][:-1]
    elif mutation == "moment_nan":
        arrays["brain.theta.exp_avg"][0] = np.nan
    elif mutation == "negative_second":
        arrays["readout.bias.exp_avg_sq"][0] = -1
    elif mutation == "missing_moment":
        arrays.pop("readout.bias.exp_avg")
    elif mutation == "extra_moment":
        arrays["rho.step"] = np.array(1, np.float32)
    elif mutation == "phase":
        arrays["drive.phase"][0] = 1
    elif mutation == "phase_dtype":
        arrays["drive.phase"] = arrays["drive.phase"].astype(np.float32)
    elif mutation == "event_count":
        arrays["drive.event_counts"][0] = 100000
    elif mutation == "queue_length":
        arrays["brain.pending"] = arrays["brain.pending"][:-1]
    elif mutation == "queue_dtype":
        arrays["brain.pending"] = arrays["brain.pending"].astype(np.float64)
    elif mutation == "refractory":
        arrays["brain.refractory_until"][0] = -1
    elif mutation == "state_clock":
        metadata["state"]["brain"]["step"] += 1
    elif mutation == "frame_clock":
        metadata["state"]["frame_index"] += 2
    elif mutation == "drive_id":
        metadata["state"]["drive"]["drive_id"] = "0" * 64
    elif mutation == "scene_id":
        metadata["context"]["scene_id"] = "wrong"
    elif mutation == "scene_count":
        metadata["context"]["scenes"][0]["frame_count"] = 3
    elif mutation == "odd_cursor":
        metadata["context"]["next_frame"] = 3
    elif mutation == "windows":
        metadata["context"]["completed_windows"] = 3
    elif mutation == "bool_epoch":
        metadata["context"]["epoch"] = False
    elif mutation == "parent":
        metadata["context"]["parent_checkpoint_id"] = "anything"
    elif mutation == "log_digest":
        metadata["context"]["log_prefix_sha256"] = "anything"
    elif mutation == "nested_parameter":
        nested = json.loads((path / "policy/checkpoint.json").read_text())
        nested["policy"]["readout"]["axes"] = ["vz", "vx", "wz"]
        (path / "policy/checkpoint.json").write_text(json.dumps(nested))
        metadata["files_sha256"]["policy/checkpoint.json"] = file_hashes(path / "policy/checkpoint.json")["sha256"]
    if mutation in {"phase", "phase_dtype", "event_count", "queue_length", "queue_dtype", "refractory",
                    "moment_dtype", "moment_shape", "moment_nan", "negative_second", "missing_moment", "extra_moment"} or mutation.startswith("step_"):
        replace_arrays(path, metadata, arrays)
    if mutation == "duplicate_npz":
        stream = io.BytesIO()
        np.save(stream, arrays["drive.phase"], allow_pickle=False)
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(path / "training.npz", "a") as archive:
                archive.writestr("drive.phase.npy", stream.getvalue())
        metadata["files_sha256"]["training.npz"] = file_hashes(path / "training.npz")["sha256"]
    if mutation != "unsealed":
        reseal(path, metadata)
    if mutation == "duplicate_json":
        original = (path / "checkpoint.json").read_text()
        (path / "checkpoint.json").write_text('{"schema": "duplicate",' + original[1:])
    target = policy()
    optimizer = adam.make_training_optimizer(target, "joint")
    update(target, optimizer, target.initial_state())
    with torch.no_grad():
        target.readout.bias.add_(.5)
    before = snapshot(target, optimizer)
    with pytest.raises(ValueError):
        checkpoint.restore_training_checkpoint(target, optimizer, path, expected=expected)
    equal_snapshots(before, snapshot(target, optimizer))


@pytest.mark.parametrize("mutation", ["live_state", "gradients", "unstepped", "duplicate_binding", "wrong_group", "extra_state", "mode"])
def test_save_validates_before_creating_directory(mutation, tmp_path):
    model = policy()
    optimizer = adam.make_training_optimizer(model, "joint")
    state, _, _ = update(model, optimizer, model.initial_state())
    if mutation == "live_state":
        state = replace(state, brain=replace(state.brain, v=state.brain.v.clone().requires_grad_()))
    elif mutation == "gradients":
        model.readout.bias.grad = torch.zeros_like(model.readout.bias)
    elif mutation == "unstepped":
        optimizer.state.clear()
    elif mutation == "duplicate_binding":
        optimizer.param_groups[1]["params"][1] = model.readout.weight
    elif mutation == "wrong_group":
        optimizer.param_groups[1]["lr"] = .2
    elif mutation == "extra_state":
        optimizer.state[torch.nn.Parameter(torch.zeros(1))] = {}
    elif mutation == "mode":
        model.brain.theta.requires_grad_(False)
    path = tmp_path / mutation
    with pytest.raises(ValueError):
        checkpoint.save_training_checkpoint(model, optimizer, state, path, mode="joint", context=context())
    assert not path.exists()


def test_no_graph_read_no_numerical_production_no_deployment_resume(sealed, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Graph allocation is forbidden")
    monkeypatch.setattr(NeuralPolicy, "from_reference", forbidden)
    archive = checkpoint.read_training_checkpoint(sealed, expected=EXPECTED)
    assert archive.metadata["purpose"] == checkpoint.PURPOSE
    with pytest.raises(ValueError, match="full_reference"):
        checkpoint.load_training_checkpoint(sealed, "unused.npz", expected=EXPECTED)
    with pytest.raises(ValueError, match="schema/purpose"):
        checkpoint.read_training_checkpoint(sealed / "policy", expected=EXPECTED)
    target = policy()
    optimizer = adam.make_training_optimizer(target, "frozen_brain")
    before = snapshot(target, optimizer)
    with pytest.raises(ValueError):
        checkpoint.restore_training_checkpoint(target, optimizer, sealed, expected=EXPECTED)
    equal_snapshots(before, snapshot(target, optimizer))


def test_existing_directory_is_not_overwritten_and_epoch_cursor_is_derived(sealed, tmp_path):
    model = policy()
    optimizer = adam.make_training_optimizer(model, "joint")
    state, _, _ = update(model, optimizer, model.initial_state())
    cursor = context()
    cursor.update(epoch=1, scene_index=1, scene_id="second", completed_windows=8)
    cursor["scenes"] = [{"scene_id": "first", "frame_count": 4}, {"scene_id": "second", "frame_count": 6}]
    path = tmp_path / "cursor"
    metadata = checkpoint.save_training_checkpoint(model, optimizer, state, path, mode="joint", context=cursor)
    assert checkpoint.read_training_checkpoint(path, expected=EXPECTED).context == cursor
    with pytest.raises(FileExistsError):
        checkpoint.save_training_checkpoint(model, optimizer, state, path, mode="joint", context=cursor)
    assert read_meta(path) == metadata


def test_mid_write_failure_is_unsealed_and_retained(tmp_path, monkeypatch):
    model = policy()
    optimizer = adam.make_training_optimizer(model, "joint")
    state, _, _ = update(model, optimizer, model.initial_state())
    path = tmp_path / "partial"
    real_savez = np.savez
    def fail(filename, **arrays):
        if Path(filename).name == "training.npz":
            raise OSError("simulated storage failure")
        return real_savez(filename, **arrays)
    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="storage failure"):
        checkpoint.save_training_checkpoint(model, optimizer, state, path, mode="joint", context=context())
    assert path.exists() and (path / "policy/checkpoint.json").exists() and not (path / "checkpoint.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint.read_training_checkpoint(path, expected=EXPECTED)
