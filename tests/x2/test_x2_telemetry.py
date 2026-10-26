"""Small observer/transport fixtures only, never complete-brain evidence."""
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from malecns_lab.learning import x2_telemetry as telemetry
from malecns_lab.learning.x2_frozen import FrozenInference, _StickyError


def fixture_policy():
    # Identity structure fixture; no reference loader, connectivity or policy execution.
    mapping = {"version": "cnsbridge.malecns_v2_groups.v1", "body_ids": {"B": [12], "A": [10, 12]},
               "rules": {"A": {"fixture": True}, "B": {"fixture": True}}, "output_body_ids": [12, 10]}
    graph = SimpleNamespace(n=3, body_ids=np.array([10, 11, 12]),
                            identity={"role": "full_reference", "reference_provenance": {"group_mapping": mapping}})
    policy = SimpleNamespace(brain=SimpleNamespace(graph=graph, model_id="fixture-model", config=SimpleNamespace(dt_ms=.1)),
                             policy_id="fixture-policy")
    identity = {"execution_id": "fixture-execution", "checkpoint_id": "fixture-checkpoint", "source_files_sha256": {}}
    return policy, identity


def recorder(tmp_path, **kwargs):
    policy, identity = fixture_policy()
    return telemetry.SpikeTelemetry(tmp_path, policy, identity, "fixture-run", **kwargs)


def read(path):
    return json.loads(path.read_text())


def test_reference_member_denominators_and_selected_real_ids(tmp_path):
    policy, identity = fixture_policy()
    manifest, lookup = telemetry.selection_manifest(policy, identity, "fixture-run")
    assert [item["name"] for item in manifest["groups"]] == ["A", "B"]
    assert manifest["groups"][0]["body_ids"] == ["10", "12"]
    assert manifest["groups"][0]["neuron_count"] == 2
    assert [item["body_id"] for item in manifest["selected_neurons"]] == ["10", "12"]
    assert lookup[12] == 2
    with pytest.raises(ValueError, match="Selected neuron IDs"):
        telemetry.selection_manifest(policy, identity, "fixture", selected_body_ids=[999])
    with pytest.raises(ValueError, match="Selected neuron IDs"):
        telemetry.selection_manifest(policy, identity, "fixture", selected_body_ids=[10, 10])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_exact_spikes_rates_partial_batch_and_single_host_copy(tmp_path, device, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    sink = recorder(tmp_path)
    copies = []
    original = torch.Tensor.cpu

    def count_cpu(value, *args, **kwargs):
        copies.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", count_cpu)
    sink.begin("forward", 7)
    expected_events = []
    for step in range(60, 113):
        values = [float(step % 2 == 0), 1., float(step % 3 == 0)]
        spikes = torch.tensor(values, device=device)
        sink.capture(spikes, step)
        assert copies == ([] if step < 110 else [(102,)])
        for column, body_id in [(0, "10"), (2, "12")]:
            if values[column]:
                expected_events.append({"body_id": body_id, "step": step})
        if step == 109:
            sink.flush(110)
    sink.flush(113)
    summary = sink.close()
    assert copies == [(102,), (8,)]
    assert summary["complete"] and summary["batch_count"] == 2 and summary["dropped_batches"] == 0
    first = read(tmp_path / "telemetry/batches/000001.json")
    final = read(tmp_path / "telemetry/latest.json")
    assert first["schema"] == telemetry.BATCH_SCHEMA
    assert first["step_start"] == 60 and first["step_end"] == 110
    assert first["observation_sequence"] == 7 and first["decision_sequence"] is None
    assert first["group_rates_hz"] == [4200., 3400.]
    assert first["spike_events"] + final["spike_events"] == expected_events
    assert final["step_start"] == 110 and final["step_end"] == 113
    assert final == read(tmp_path / "telemetry/batches/000002.json")
    assert not list((tmp_path / "telemetry").rglob("*.tmp"))
    assert summary["total_bytes"] == sum(path.stat().st_size for path in (tmp_path / "telemetry").rglob("*.json"))


def test_queue_backpressure_drops_explicitly_and_never_marks_complete(tmp_path):
    sink = recorder(tmp_path)
    # Stop the real writer before providing data, then exercise the bounded queue.
    sink._stop.set()
    sink._thread.join(timeout=1.)
    sink._queue = Queue(maxsize=8)
    for batch in range(10):
        sink.capture(torch.tensor([1., 0., 0.]), batch)
        sink.flush(batch + 1)
    assert sink._queue.qsize() == 8 and sink.dropped_batches == 2
    summary = sink.close()
    assert not summary["complete"] and summary["produced_batches"] == 10 and summary["dropped_batches"] == 2


def test_disk_failure_and_archive_budget_are_fail_isolated(tmp_path, monkeypatch):
    sink = recorder(tmp_path)
    original = telemetry._atomic

    def fail_batch(path, content):
        if path.parent.name == "batches":
            raise OSError("injected storage failure")
        return original(path, content)

    monkeypatch.setattr(telemetry, "_atomic", fail_batch)
    sink.capture(torch.tensor([1., 0., 0.]), 0)
    sink.flush(1)
    summary = sink.close()
    assert not summary["complete"] and summary["dropped_batches"] == 1
    assert "storage failure" in summary["error"]
    assert read(tmp_path / "telemetry/summary.json")["status"] == "failed"
    # Failure does not propagate into the numerical caller.
    sink.capture(torch.tensor([1., 0., 0.]), 1)
    sink.flush(2)


def test_byte_budget_rejects_without_unbounded_output(tmp_path, monkeypatch):
    sink = recorder(tmp_path)
    monkeypatch.setattr(telemetry, "MAX_ARCHIVE_BYTES", sink.archive_bytes)
    sink.capture(torch.tensor([1., 0., 0.]), 0)
    sink.flush(1)
    summary = sink.close()
    assert not summary["complete"] and "byte budget" in summary["error"]
    assert not list((tmp_path / "telemetry/batches").iterdir())


def behavior_recorder(tmp_path):
    from malecns_lab.learning.x2_behavior_telemetry import BehaviorTelemetry
    policy, identity = fixture_policy()
    policy.encoder = SimpleNamespace(input_map_id='fixture-input')
    return BehaviorTelemetry(tmp_path, policy, identity, 'fixture-run', 'fixture-session', continuous=True)


def test_continuous_limit_preserves_archive_and_updates_real_latest(tmp_path, monkeypatch):
    sink = behavior_recorder(tmp_path)
    sink.capture(torch.tensor([1., 0., 0.]), 0); sink.flush(1)
    sink._queue.join()
    original = (tmp_path / 'telemetry/batches/000001.json').read_bytes()
    monkeypatch.setattr(telemetry, 'MAX_ARCHIVE_BYTES', sink.archive_bytes + telemetry.MAX_BATCH_BYTES + 8192)
    for step in (1, 2):
        sink.capture(torch.tensor([0., 0., 1.]), step); sink.flush(step + 1)
        sink._queue.join()
    summary = sink.close()
    assert summary['error'] is None and not summary['complete']
    assert summary['archive_truncated'] and summary['archive_limit_reason'] == 'archive_byte_limit'
    assert summary['written_batches'] == 1 and summary['live_only_batches'] == 2
    assert summary['dropped_batches'] == 0 and summary['archive_last_batch_sequence'] == 1
    assert (tmp_path / 'telemetry/batches/000001.json').read_bytes() == original
    assert len(list((tmp_path / 'telemetry/batches').iterdir())) == 1
    latest = read(tmp_path / 'telemetry/latest.json')
    assert latest['batch_sequence'] == 3 and latest['spike_events'] == [{'body_id': '12', 'step': 2}]


def test_continuous_live_storage_failure_is_not_normal_capacity(tmp_path, monkeypatch):
    sink = behavior_recorder(tmp_path)
    sink.archive_limit_reason = 'archive_byte_limit'
    original = telemetry._atomic
    def fail_live(path, content):
        if path.name == 'latest.json':
            raise OSError('injected live write failure')
        original(path, content)
    monkeypatch.setattr(telemetry, '_atomic', fail_live)
    sink.capture(torch.tensor([1., 0., 0.]), 0); sink.flush(1)
    summary = sink.close()
    assert not summary['complete'] and 'injected live write failure' in summary['error']


def test_abnormal_close_and_unflushed_tail_are_partial(tmp_path):
    sink = recorder(tmp_path)
    sink.capture(torch.tensor([1., 0., 0.]), 0)
    summary = sink.close(completed=False)
    assert not summary["complete"] and summary["partial_batch_steps"] == 1


@dataclass
class BrainValues:
    v: torch.Tensor
    g: torch.Tensor
    refractory_until: torch.Tensor
    pending: tuple
    step: int
    model_id: str


@dataclass
class DriveValues:
    phase: torch.Tensor
    event_counts: torch.Tensor
    step: int
    drive_id: str
    spatial_map_id: str


def test_observer_keeps_step_equations_exact_and_captures_pre_reset_spikes(tmp_path):
    sink = recorder(tmp_path)
    wrapper = object.__new__(FrozenInference)
    wrapper._brain_state, wrapper._drive_state = BrainValues, DriveValues
    cfg = SimpleNamespace(v_rest_mv=0., v_reset_mv=0., v_threshold_mv=1., surrogate_scale_mv=1.)
    brain = SimpleNamespace(config=cfg, input_mask=torch.tensor([False, False, False]), refractory_steps=1,
                            base_weights=None, base_weights_adjoint=None, model_id="fixture-model")
    drive = SimpleNamespace(mapping=SimpleNamespace(n_nodes=3, map_id="fixture"), input_indices=torch.empty(0, dtype=torch.int64),
                            config=SimpleNamespace(event_voltage_mv=1.), drive_id="fixture")
    wrapper.policy = SimpleNamespace(brain=brain, drive=drive)
    wrapper._spike = lambda distance, scale: (distance > 0).to(distance.dtype)
    wrapper._matvec = lambda matrix, adjoint, vector: vector * .5
    zero = torch.zeros(3)
    state = BrainValues(torch.tensor([2., 0., 3.]), zero.clone(), torch.zeros(3, dtype=torch.int64), (zero.clone(),), 0, brain.model_id)
    sensory = DriveValues(torch.empty(0), torch.empty(0, dtype=torch.int64), 0, drive.drive_id, drive.mapping.map_id)
    constants = (1., 1., 1., zero, torch.ones_like(zero), torch.empty(0))
    wrapper.telemetry = None
    off, off_drive = wrapper._step(state, sensory, constants, _StickyError("cpu"))
    wrapper.telemetry = sink
    on, on_drive = wrapper._step(state, sensory, constants, _StickyError("cpu"))
    sink.flush(1)
    assert sink.close()["complete"]
    for name in ("v", "g", "refractory_until"):
        assert torch.equal(getattr(off, name), getattr(on, name))
    assert torch.equal(off.pending[0], on.pending[0]) and torch.equal(off_drive.phase, on_drive.phase)
    assert on.v.tolist() == [0., 0., 0.]
    assert read(tmp_path / "telemetry/latest.json")["spike_events"] == [{"body_id": "10", "step": 0}, {"body_id": "12", "step": 0}]


@pytest.mark.parametrize("module,args", [("x2_neural", ["--telemetry"]), ("x2_inference", ["--backend", "optimized", "--telemetry"])])
def test_telemetry_cli_preview_stays_inert(module, args):
    script = f'''import sys
from malecns_lab.experiments import {module} as cli
assert cli.main({args!r}) == 0
assert not any(name in sys.modules for name in ("torch", "numpy", "rclpy"))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["telemetry"] is True
