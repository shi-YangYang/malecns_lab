"""Task/protocol tests use explicit tiny software doubles, never brain evidence."""
from dataclasses import dataclass
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from malecns_lab.control.body import BodyVelocity
from malecns_lab.drones.yaw_task import YawTaskEnvironment, scenes, control_gate, score_episode
from malecns_lab.learning.yaw import archive, protocol, rollout, training


class TinyBrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(3))

    @property
    def gains(self):
        return 1 + .5 * torch.tanh(self.theta)


@dataclass
class TinyState:
    frame_index: int
    value: torch.Tensor

    def detach(self):
        return TinyState(self.frame_index, self.value.detach())


class SoftwarePolicy(torch.nn.Module):
    """No connectome; only verifies causal orchestration/autograd/persistence."""
    def __init__(self):
        super().__init__()
        self.brain = TinyBrain()
        self.readout = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.readout.weight.fill_(.04)
            self.readout.bias.zero_()
        self.identity = {"role": "software_test_only", "group_names": ["test"], "policy_id": "fixture"}

    def initial_state(self):
        return TinyState(0, torch.tensor(0.))

    def forward_tensor(self, image, state, frame_offset=0):
        assert frame_offset in (0, 500)
        value = state.value * .2 + float(image.mean()) / 255 + self.brain.theta.sum() * .1
        features = torch.stack((value, value * .5))
        logit = self.readout(features).squeeze()
        action = .35 * torch.tanh(logit) if state.frame_index >= 40 else logit * 0
        return SimpleNamespace(logit=logit, action=action, features_mv=features, visual_features=torch.zeros(14),
                               group_rates_hz=value.detach().reshape(1), network_spikes=torch.tensor(1),
                               state=TinyState(state.frame_index+1, value))

    def parameters_numpy(self):
        return {name: p.detach().numpy().copy() for name,p in self.named_parameters()}

    def load_parameters(self, arrays):
        with torch.no_grad():
            for name,p in self.named_parameters():
                p.copy_(torch.tensor(arrays[name]))

    def state_arrays(self, state):
        return {"frame_index": np.array(state.frame_index, dtype=np.int64), "value": state.value.detach().numpy().copy(),
                "policy_id": np.array("fixture")}

    def restore_state(self, arrays):
        return TinyState(int(arrays["frame_index"]), torch.tensor(arrays["value"]))


class SoftwareAdam:
    def __init__(self, policy, mode="joint", snapshot=None):
        self.policy, self.mode = policy, mode
        policy.brain.theta.requires_grad_(mode == "joint")
        self.parameters = {n:p for n,p in policy.named_parameters() if p.requires_grad}
        self.optimizer = torch.optim.Adam(list(self.parameters.values()), lr=.001)
        self.steps = snapshot["steps"] if snapshot else 0
        self.identity = {"kind": "software_fixture", "mode": mode}
        for n,p in self.parameters.items():
            self.optimizer.state[p] = {field: torch.tensor(snapshot["moments"][f"{n}.{field}"]) if snapshot else
                                      torch.zeros(()) if field == "step" else torch.zeros_like(p)
                                      for field in ("step", "exp_avg", "exp_avg_sq")}

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def snapshot(self):
        return {"schema": "software_fixture", "mode": self.mode, "steps": self.steps, "identity": self.identity,
                "moments": {f"{n}.{key}": v.detach().numpy().copy() for n,p in self.parameters.items()
                            for key,v in self.optimizer.state[p].items()}}

    def step(self):
        old = self.policy.parameters_numpy()
        raw = {n: int(torch.count_nonzero(p.grad)) for n,p in self.parameters.items()}
        self.optimizer.step()
        self.steps += 1
        new = self.policy.parameters_numpy()
        return {"changed": {n: int(np.count_nonzero(old[n] != new[n])) for n in old}, "raw_nonzero": raw,
                "steps": self.steps, "mode": self.mode, "residency": "CPU software fixture only"}


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "module.py"
    source.write_text("value = 1\n")
    graph = tmp_path / "graph_fixture.bin"
    graph.write_bytes(b"software_fixture_not_connectome")
    monkeypatch.setattr(protocol, "ROOT", root)
    monkeypatch.setattr(protocol, "source_files", lambda: [source])
    directory = tmp_path / "run"
    run = protocol.register(directory, graph, "joint")
    return directory, run, source


def test_scene_splits_and_fixed_counts():
    assert [len(scenes(s)) for s in ("train", "validation", "test")] == [4,2,5]
    seeds = [{s.texture_seed for s in scenes(split)} for split in ("train", "validation", "test")]
    assert not any(seeds[a] & seeds[b] for a,b in ((0,1),(0,2),(1,2)))
    assert sum(s.observation_frames//2 for s in scenes("train")) == 80
    assert all(s.total_frames == 100 for s in scenes("test"))
    assert scenes("test")[-1].amplitude_rad_s == 0


def test_teacher_is_past_interval_not_upcoming_disturbance():
    environment = YawTaskEnvironment(scenes("train")[0])
    for _ in range(40):
        assert environment.teacher_action == 0
        environment.advance(BodyVelocity())
    assert environment.scene.disturbance(40) == .2
    assert environment.teacher_action == 0
    previous = environment.yaw_rad
    environment.advance(BodyVelocity())
    assert environment.teacher_action == pytest.approx(-2*(environment.yaw_rad-previous)/.05)
    assert environment.teacher_action < 0


def test_environment_restore_preserves_filter_and_causal_teacher():
    scene = scenes("train")[1]
    continuous = YawTaskEnvironment(scene)
    for index in range(56):
        continuous.advance(BodyVelocity(wz=.03 if index >= 40 else 0))
    saved = continuous.snapshot()
    resumed = YawTaskEnvironment(scene)
    resumed.restore(saved)
    assert saved["disturbance_state_rad_s"] != 0
    for index in range(56,80):
        assert resumed.teacher_action == continuous.teacher_action
        np.testing.assert_array_equal(resumed.frame(), continuous.frame())
        assert resumed.advance(BodyVelocity(wz=-.02)) == continuous.advance(BodyVelocity(wz=-.02))
    assert resumed.snapshot() == continuous.snapshot()


def test_registration_source_binding_and_no_overwrite(registered):
    directory, run, source = registered
    assert protocol.validate_run(directory) == run
    with pytest.raises(ValueError, match="already exists"):
        protocol.register(directory, run["graph"]["path"], "joint")
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source"):
        protocol.validate_run(directory)


def test_joint_frozen_share_family_with_distinct_runs(registered, tmp_path):
    _, joint, _ = registered
    frozen = protocol.register(tmp_path / "frozen", joint["graph"]["path"], "frozen")
    assert frozen["family_id"] == joint["family_id"]
    assert frozen["run_id"] != joint["run_id"]


def test_window_archive_resume_matches_uninterrupted_including_environment(registered):
    directory, run, _ = registered
    policy, scene = SoftwarePolicy(), scenes("train")[0]
    optimizer, environment = SoftwareAdam(policy), YawTaskEnvironment(scene)
    state = policy.initial_state()
    archive.write_checkpoint(directory / "checkpoints/000000", run=run, policy=policy, optimizer=optimizer,
                             state=state, environment=environment, completed_windows=0, scene_index=0, parent_id=None)
    state, _ = rollout.warmup(policy, state, environment)
    state, arrays, report = rollout.compute_window(policy, optimizer, state, environment)
    assert report["optimizer"]["changed"]["brain.theta"] > 0
    assert arrays["frame_index"].tolist() == [40,41]
    first = archive.read_checkpoint(directory / "checkpoints/000000", run)
    metadata = archive.write_checkpoint(directory / "checkpoints/000001", run=run, policy=policy, optimizer=optimizer,
        state=state, environment=environment, completed_windows=1, scene_index=0,
        parent_id=first["metadata"]["checkpoint_id"])
    event_path = directory / "windows/000001"
    event_path.mkdir(parents=True)
    archive.write_arrays(event_path / "frames.npz", arrays)
    training._seal(event_path, {"run_id": run["run_id"], "window": 1, "checkpoint_id": metadata["checkpoint_id"],
        "parent_checkpoint_id": first["metadata"]["checkpoint_id"]}, ("frames.npz",))
    chain = archive.scan_checkpoints(directory, run)
    assert len(chain) == 2 and "state" not in chain[0] and "state" in chain[-1]
    recovered, restored_env = SoftwarePolicy(), YawTaskEnvironment(scene)
    restored_state, restored_optimizer = archive.restore_checkpoint(chain[-1], recovered, SoftwareAdam, restored_env)
    state, values1, _ = rollout.compute_window(policy, optimizer, state, environment)
    restored_state, values2, _ = rollout.compute_window(recovered, restored_optimizer, restored_state, restored_env)
    for key in values1:
        np.testing.assert_array_equal(values1[key], values2[key])
    for key,value in policy.state_arrays(state).items():
        np.testing.assert_array_equal(value, recovered.state_arrays(restored_state)[key])
    assert environment.snapshot() == restored_env.snapshot()
    for key,value in optimizer.snapshot()["moments"].items():
        np.testing.assert_array_equal(value, restored_optimizer.snapshot()["moments"][key])


def test_frozen_software_window_updates_readout_only():
    policy = SoftwarePolicy()
    optimizer, environment = SoftwareAdam(policy, "frozen"), YawTaskEnvironment(scenes("train")[0])
    state, _ = rollout.warmup(policy, policy.initial_state(), environment)
    _, _, report = rollout.compute_window(policy, optimizer, state, environment)
    assert report["optimizer"]["changed"]["brain.theta"] == 0
    assert "brain.theta" not in optimizer.snapshot()["moments"]
    assert report["optimizer"]["changed"]["readout.weight"] > 0


def test_checkpoint_corruption_rejected(registered):
    directory, run, _ = registered
    policy = SoftwarePolicy()
    path = directory / "checkpoint"
    archive.write_checkpoint(path, run=run, policy=policy, optimizer=SoftwareAdam(policy), state=policy.initial_state(),
        environment=YawTaskEnvironment(scenes("train")[0]), completed_windows=0, scene_index=0, parent_id=None)
    with (path / "arrays.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        archive.read_checkpoint(path, run)


def test_zero_is_explicitly_non_neural_and_limits_are_scored():
    scene = scenes("validation")[0]
    summary, arrays = rollout.evaluate_episode(scene, "zero")
    assert summary["neural_executed"] is False and summary["neural_steps"] == 0
    assert "group_rates_hz" not in arrays and summary["group_names"] == []
    assert summary["metrics"]["rms_rotation_rad_s"] > 0
    bad = arrays["command_wz_rad_s"].copy()
    bad[42] = .36
    with pytest.raises(ValueError, match="limit"):
        score_episode(scene, arrays["actual_wz_rad_s"], bad)


def test_control_gate_rejects_no_improvement_and_missing_static():
    rows = []
    for scene in scenes("validation"):
        base = {"rms_rotation_rad_s": .1, "peak_rotation_rad_s": .2, "tail_rms_rad_s": .01}
        rows.extend([{"scene": scene.__dict__, "mode": mode, "metrics": dict(base)} for mode in ("zero", "joint")])
    assert not control_gate(rows, require_static=False)["passed"]
    for row in rows:
        if row["mode"] == "joint":
            row["metrics"]["rms_rotation_rad_s"] = .07
    assert control_gate(rows, require_static=False)["passed"]
    assert not control_gate(rows, require_static=True)["passed"]


def test_window_count_guard_precedes_launch():
    with pytest.raises(ValueError, match="1 to 4"):
        training.train_segment(Path("not_a_run"), 5)


def test_successful_process_without_evidence_is_not_success(registered, monkeypatch):
    directory, run, _ = registered
    monkeypatch.setattr(training, "supervise", lambda *args, **kwargs: {"status": "completed", "reason": "worker_exited", "returncode": 0})
    result = training._job(directory, run, "gate")
    assert result["status"] == "failed"
    assert "worker_result.json" in result["detail"]


def test_evaluation_refuses_to_open_test_on_failed_validation(tmp_path, monkeypatch):
    directory, frozen_directory = tmp_path / "joint", tmp_path / "frozen"
    (directory / "evaluation/validation").mkdir(parents=True)
    frozen_directory.mkdir()
    selection = {"family_id": "family", "joint_checkpoint_id": "joint", "frozen_checkpoint_id": "frozen",
                 "frozen_directory": str(frozen_directory)}
    protocol.write_json(directory / "evaluation/validation/summary.json", {"selection": selection, "status": "completed", "control_gate": {"passed": False}})
    monkeypatch.setattr(training, "validate_run", lambda path: {"mode": "joint" if Path(path) == directory else "frozen", "family_id": "family"})
    monkeypatch.setattr(training, "_final_checkpoint", lambda path, run: {"metadata": {"checkpoint_id": run["mode"]}})
    with pytest.raises(ValueError, match="Validation must pass"):
        training.evaluate(directory, frozen_directory, "test")
    assert not (directory / "test_lock.json").exists()
