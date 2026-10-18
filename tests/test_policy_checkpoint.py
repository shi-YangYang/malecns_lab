"""Complete parameter archives on explicit numerical circuits only."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cnsbridge.brain.metadata import canonical_hash, file_hashes
from cnsbridge.learning.checkpoint import read_checkpoint, save_checkpoint
from cnsbridge.learning import policy_checkpoint as checkpoint
from cnsbridge.learning.neural_policy import NeuralPolicyConfig
from cnsbridge.senses.target import TargetCamera
from test_neural_policy import frame, policy


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def unchanged(model, saved):
    for name, value in model.named_parameters():
        assert torch.equal(value, saved[name])


def reseal(path, meta):
    meta["checkpoint_id"] = canonical_hash({key: value for key, value in meta.items() if key != "checkpoint_id"})
    (path / "checkpoint.json").write_text(json.dumps(meta, allow_nan=False))


def rebind(identity, key):
    identity[key] = canonical_hash({k: v for k, v in identity.items() if k != key})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("background", ["zero_background", "fixed_lamina_8mv"])
def test_joint_update_bitwise_roundtrip_and_same_outputs(dtype, background, tmp_path, monkeypatch):
    source = policy(dtype=dtype, background=background)
    before = snapshot(source)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    output = source.forward_tensor(frame(), source.initial_state())
    (output.action - output.action.new_tensor([0.1, -0.1, 0.2])).square().sum().backward()
    optimizer.step()
    assert all(not torch.equal(before[name], value) for name, value in source.named_parameters())
    path = tmp_path / "checkpoint"
    identity = source.policy_id
    meta = checkpoint.save_policy_checkpoint(source, path, run_identity={"trial": "numerical_fixture", "update": 1})
    target = policy(dtype=dtype, background=background)
    def forbidden(*args, **kwargs):
        raise AssertionError("Archive IO performed neural preprocessing or graph loading")
    with monkeypatch.context() as patch:
        patch.setattr(source, "initial_state", forbidden)
        patch.setattr(target, "initial_state", forbidden)
        patch.setattr(checkpoint.NeuralPolicy, "from_reference", forbidden)
        read_meta, arrays = checkpoint.read_policy_checkpoint(path)
        checkpoint.restore_policy_checkpoint(target, path)
    assert meta == read_meta and meta["purpose"] == "deployment_parameters_only"
    assert set(arrays) == {"theta", "readout_weight", "readout_bias"}
    unchanged(target, snapshot(source))
    assert target.policy_id == source.policy_id == identity
    with torch.no_grad():
        a = source.forward_tensor(frame(), source.initial_state())
        b = target.forward_tensor(frame(), target.initial_state())
    assert torch.equal(a.features_mv, b.features_mv) and torch.equal(a.action, b.action)
    assert not any("optimizer" in key or "state" in key for key in meta)


@pytest.mark.parametrize("variant", ["camera", "window", "background", "dtype"])
def test_target_identity_mismatch_is_atomic(variant, tmp_path):
    source = policy()
    path = tmp_path / "checkpoint"
    checkpoint.save_policy_checkpoint(source, path)
    variants = {"camera": {"camera": TargetCamera(horizontal_fov_deg=80)},
                "window": {"config": NeuralPolicyConfig(10, 2, 0.1, 123)},
                "background": {"background": "fixed_lamina_8mv"}, "dtype": {"dtype": torch.float32}}
    target = policy(**variants[variant])
    with torch.no_grad():
        target.brain.theta.fill_(0.2)
    saved = snapshot(target)
    with pytest.raises(ValueError, match="identity"):
        checkpoint.restore_policy_checkpoint(target, path)
    unchanged(target, saved)


@pytest.mark.parametrize("mutation", ["source", "role", "order", "clock", "feature_scale", "gain_shape", "nonfinite",
                                     "dtype", "extra", "missing", "metadata_shape", "file_hash", "canonical", "run_nonfinite"])
def test_bad_archive_rejected_before_any_target_mutation(mutation, tmp_path):
    source, target = policy(), policy()
    with torch.no_grad():
        target.brain.theta.fill_(0.234)
        target.readout.bias.fill_(0.567)
    saved = snapshot(target)
    path = tmp_path / "checkpoint"
    meta = checkpoint.save_policy_checkpoint(source, path)
    identity = meta["policy"]
    if mutation == "source":
        identity["implementation_sha256"]["learning/neural_policy.py"] = "0" * 64
    elif mutation == "role":
        identity["role"] = "full_reference"
    elif mutation == "order":
        identity["features"]["body_ids"].reverse()
    elif mutation == "clock":
        identity["frame_steps"] += 2
    elif mutation == "feature_scale":
        identity["features"]["scale_mv"] = 10
    elif mutation in {"gain_shape", "nonfinite", "dtype", "extra", "missing"}:
        with np.load(path / "parameters.npz", allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
        if mutation == "gain_shape":
            arrays["theta"] = arrays["theta"][:-1]
        elif mutation == "nonfinite":
            arrays["readout_bias"][0] = np.inf
        elif mutation == "dtype":
            arrays["readout_weight"] = arrays["readout_weight"].astype(np.float32)
        elif mutation == "extra":
            arrays["rho"] = np.zeros(1)
        else:
            arrays.pop("readout_bias")
        np.savez(path / "parameters.npz", **arrays)
        meta["arrays_sha256"] = file_hashes(path / "parameters.npz")["sha256"]
        meta["parameters"] = {k: {"parameter": checkpoint.PARAMETERS.get(k, k), "shape": list(v.shape), "dtype": v.dtype.str}
                              for k, v in arrays.items()}
    elif mutation == "metadata_shape":
        meta["parameters"]["readout_weight"]["shape"] = [3, 999]
    elif mutation == "file_hash":
        with (path / "parameters.npz").open("ab") as file:
            file.write(b"changed")
    elif mutation == "canonical":
        meta["checkpoint_id"] = "0" * 64
    else:
        meta["run_identity"] = {"loss": float("nan")}
    rebind(identity, "policy_id")
    if mutation in {"canonical", "run_nonfinite"}:
        (path / "checkpoint.json").write_text(json.dumps(meta))
    else:
        reseal(path, meta)
    with pytest.raises(ValueError):
        checkpoint.restore_policy_checkpoint(target, path)
    unchanged(target, saved)


def test_production_loader_rejects_numerical_role_without_graph_load(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    checkpoint.save_policy_checkpoint(policy(), path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Production graph loader should not be called for a fixture")
    monkeypatch.setattr(checkpoint.NeuralPolicy, "from_reference", forbidden)
    with pytest.raises(ValueError, match="full_reference"):
        checkpoint.load_policy_checkpoint(path, tmp_path / "no-reference.npz")


@pytest.mark.parametrize("mutation", ["gain", "gradient", "event_order", "extra", "missing",
                                     "tonic_formula", "tonic_range", "tonic_integration", "tonic_hypothesis",
                                     "tonic_extra", "tonic_missing", "tonic_types", "member_type", "member_count",
                                     "member_total", "member_capacity", "member_digest", "mapping_digest", "member_extra"])
def test_light_reader_rejects_rehashed_incompatible_dynamics(mutation, tmp_path, monkeypatch):
    background = "zero_background" if mutation in {"gain", "gradient", "event_order", "extra", "missing"} else "fixed_lamina_8mv"
    source, target = policy(background=background), policy(background=background)
    with torch.no_grad():
        target.brain.theta.fill_(0.2)
        target.readout.bias.fill_(0.4)
    saved = snapshot(target)
    path = tmp_path / "checkpoint"
    meta = checkpoint.save_policy_checkpoint(source, path)
    model = meta["policy"]["model"]
    if mutation in {"gain", "gradient", "event_order"}:
        model[mutation] = "exp(theta); UNBOUNDED incompatible dynamics"
    elif mutation == "extra":
        model["undeclared_dynamics"] = "different"
    elif mutation == "missing":
        model.pop("gain")
    else:
        tonic = model["tonic"]
        if mutation in {"tonic_formula", "tonic_integration", "tonic_hypothesis"}:
            tonic[mutation.removeprefix("tonic_")] = "incompatible semantics"
        elif mutation == "tonic_range":
            tonic["range_mv"]["inclusive"] = True
        elif mutation == "tonic_extra":
            tonic["undeclared_offset"] = 5
        elif mutation == "tonic_missing":
            tonic.pop("integration")
        elif mutation == "tonic_types":
            tonic["types"] = ["L5"]  # Not in this fixture's graph grouping.
            tonic["members"][0]["type"] = "L5"
        elif mutation == "member_type":
            tonic["members"][0]["type"] = "L2"
        elif mutation == "member_count":
            tonic["members"][0]["count"] = True
        elif mutation == "member_total":
            tonic["member_count"] += 1
        elif mutation == "member_capacity":
            tonic["member_count"] = model["graph"]["nodes"]
            tonic["members"][0]["count"] = model["graph"]["nodes"]
        elif mutation == "member_digest":
            tonic["members"][0]["body_ids_sha256"] = "not-a-digest"
        elif mutation == "mapping_digest":
            tonic["mapping_sha256"] = "not-a-digest"
        else:
            tonic["members"][0]["undeclared"] = 1
    rebind(model, "model_id")
    rebind(meta["policy"], "policy_id")
    reseal(path, meta)
    def forbidden(*args, **kwargs):
        raise AssertionError("Lightweight semantics validation must not load a reference graph")
    monkeypatch.setattr(checkpoint.NeuralPolicy, "from_reference", forbidden)
    with pytest.raises(ValueError):
        checkpoint.read_policy_checkpoint(path)
    with pytest.raises(ValueError):
        checkpoint.restore_policy_checkpoint(target, path)
    unchanged(target, saved)


def test_new_and_old_formats_reject_each_other(tmp_path):
    model = policy()
    old, new = tmp_path / "old", tmp_path / "new"
    save_checkpoint(model.brain, old)
    checkpoint.save_policy_checkpoint(model, new)
    with pytest.raises(ValueError, match="schema"):
        checkpoint.read_policy_checkpoint(old)
    with pytest.raises(ValueError, match="schema"):
        read_checkpoint(new)


def test_save_validates_before_creating_output_and_never_overwrites(tmp_path):
    model = policy()
    path = tmp_path / "good"
    checkpoint.save_policy_checkpoint(model, path)
    before = (path / "checkpoint.json").read_bytes()
    with pytest.raises(FileExistsError):
        checkpoint.save_policy_checkpoint(model, path)
    assert (path / "checkpoint.json").read_bytes() == before
    with pytest.raises(ValueError):
        checkpoint.save_policy_checkpoint(model, tmp_path / "bad-run", run_identity={"loss": np.nan})
    assert not (tmp_path / "bad-run").exists()
    with torch.no_grad():
        model.readout.bias[0] = float("nan")
    with pytest.raises(ValueError):
        checkpoint.save_policy_checkpoint(model, tmp_path / "bad-policy")
    assert not (tmp_path / "bad-policy").exists()


def test_invalid_target_parameter_set_and_shape_are_atomic(tmp_path):
    source, target = policy(), policy()
    path = tmp_path / "checkpoint"
    checkpoint.save_policy_checkpoint(source, path)
    target.readout.weight = torch.nn.Parameter(torch.ones((3, 3), dtype=torch.float64))
    saved = snapshot(target)
    with pytest.raises(ValueError, match="shape"):
        checkpoint.restore_policy_checkpoint(target, path)
    unchanged(target, saved)
    target.extra = torch.nn.Parameter(torch.zeros(1))
    saved = snapshot(target)
    with pytest.raises(ValueError, match="parameters"):
        checkpoint.restore_policy_checkpoint(target, path)
    unchanged(target, saved)
