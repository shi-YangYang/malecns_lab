"""Small observer/transport fixtures only, never complete-brain evidence."""
from dataclasses import dataclass
import hashlib
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


def whole_brain_recorder(tmp_path):
    from malecns_lab.learning.x2_behavior_telemetry import BehaviorTelemetry
    policy, identity = fixture_policy()
    policy.encoder = SimpleNamespace(input_map_id='fixture-input')
    policy.telemetry_neural_state = True
    policy.brain.config.v_rest_mv, policy.brain.config.v_threshold_mv = -60., -45.
    policy.brain.graph.identity['reference_provenance']['graph_sha256'] = 'a' * 64
    return BehaviorTelemetry(tmp_path, policy, identity, 'fixture-run', 'fixture-session', continuous=True)


def whole_brain_decision(sink, request, seq):
    sink.begin_decision(SimpleNamespace(request_id=request, observation_seq=seq,
        feedback=SimpleNamespace(execution_id='fixture-skill')), seq)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_whole_brain_voltage_silent_cells_request_baseline_and_single_copy(tmp_path, device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS not available')
    sink = whole_brain_recorder(tmp_path)
    assert sink.error is None
    copies, before_states, after_states = [], [], []
    original_cpu = torch.Tensor.cpu
    def cpu_copy(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu_copy)
    original = torch.tensor([-60., -60., -60.], device=device)
    # Cell 11 belongs to neither reference group and never spikes. Its actual
    # changed voltage must nevertheless be present in the full-cell payload.
    for window, (request, start, base_voltage, voltage) in enumerate([
        ('request-1', 600, original, [-58., -56., -60.]),
        ('request-1', 650, torch.tensor([-58., -56., -60.], device=device), [-57., -54., -60.]),
        ('request-2', 600, original, [-59., -59., -60.]),
    ]):
        whole_brain_decision(sink, request, window + 1)
        after_v = torch.tensor(voltage, device=device)
        for offset in range(50):
            step = start + offset
            pre = SimpleNamespace(step=step, v=base_voltage if offset == 0 else after_v)
            post = SimpleNamespace(step=step + 1, v=after_v)
            spikes = torch.tensor([float(offset % 2 == 0), 0., 0.], device=device)
            before_states.append((pre.v, pre.v.clone()))
            after_states.append((post.v, post.v.clone(), spikes, spikes.clone()))
            sink.capture_state(spikes, pre, post)
            assert len(copies) == window
        sink.flush(start + 50)
    summary = sink.close()
    assert summary['complete'] and summary['neural_state_complete']
    assert copies == [(111,), (111,), (111,)]
    for value, saved in before_states:
        assert torch.equal(value, saved)
    for value, saved, spikes, saved_spikes in after_states:
        assert torch.equal(value, saved) and torch.equal(spikes, saved_spikes)
    manifest = read(tmp_path / 'telemetry/manifest.json')['neural_state']
    assert manifest['body_ids'] == ['10', '11', '12'] and manifest['node_count'] == 3
    assert manifest['body_ids_sha256'] == hashlib.sha256(b'["10","11","12"]').hexdigest()
    assert manifest['graph_sha256'] == 'a' * 64 and manifest['sample'] == 'post_step_window_mean'
    for index, expected in enumerate([
        [[-58, -56, -60], [2, 4, 0], [5000, 0, 0]],
        [[-57, -54, -60], [3, 6, 0], [5000, 0, 0]],
        [[-59, -59, -60], [1, 1, 0], [5000, 0, 0]],
    ], 1):
        batch = read(tmp_path / f'telemetry/batches/{index:06d}.json')
        ref = batch['neural_state']
        content = (tmp_path / 'telemetry' / ref['file']).read_bytes()
        assert ref['bytes'] == 36 and ref['sha256'] == hashlib.sha256(content).hexdigest()
        assert ref['body_ids_sha256'] == manifest['body_ids_sha256'] and ref['archived']
        assert batch['request_id'] == ('request-2' if index == 3 else 'request-1')
        assert batch['checkpoint_id'] == 'fixture-checkpoint' and batch['input_map_id'] == 'fixture-input'
        np.testing.assert_array_equal(np.frombuffer(content, dtype='<f4').reshape(3, 3), expected)
    assert summary['total_bytes'] == sum(p.stat().st_size for p in (tmp_path / 'telemetry').rglob('*') if p.is_file())


def test_whole_brain_bounded_archive_keeps_original_and_marks_live_refs(tmp_path, monkeypatch):
    from malecns_lab.learning import x2_neural_state as neural
    monkeypatch.setattr(neural, 'MAX_ARCHIVE_BYTES', 36)
    sink = whole_brain_recorder(tmp_path)
    whole_brain_decision(sink, 'request-1', 1)
    initial = None
    for step in range(11):
        before = SimpleNamespace(step=step, v=torch.full((3,), -60.))
        after = SimpleNamespace(step=step + 1, v=torch.full((3,), -59. + step / 100))
        sink.capture_state(torch.zeros(3), before, after)
        sink.flush(step + 1)
        sink._queue.join()
        path = tmp_path / 'telemetry/neural/000001.f32'
        if initial is None:
            initial = path.read_bytes()
        assert path.read_bytes() == initial
    summary = sink.close()
    assert summary['error'] is None and summary['complete'] and not summary['neural_state_complete']
    assert summary['neural_state_archive_truncated'] and summary['neural_state_archived_batches'] == 1
    assert summary['neural_state_archive_bytes'] == 36 and summary['neural_state_live_bytes'] == 36 * 8
    assert summary['neural_state_live_batches'] == 10
    assert len(list((tmp_path / 'telemetry/neural').iterdir())) == 9
    assert read(tmp_path / 'telemetry/batches/000002.json')['neural_state']['archived'] is False
    latest = read(tmp_path / 'telemetry/latest.json')
    ref = latest['neural_state']
    assert ref['file'] == 'neural/live-2.f32' and ref['archived'] is False
    assert ref['sha256'] == hashlib.sha256((tmp_path / 'telemetry' / ref['file']).read_bytes()).hexdigest()
    assert summary['total_bytes'] == sum(p.stat().st_size for p in (tmp_path / 'telemetry').rglob('*') if p.is_file())


def test_whole_brain_same_step_hook_and_sidecar_failure_are_observable(tmp_path, monkeypatch):
    from malecns_lab.learning.x2_frozen_history_policy import capture_actual_steps
    sink = whole_brain_recorder(tmp_path)
    before = SimpleNamespace(step=49, v=torch.tensor([-60., -60., -60.]))
    after = SimpleNamespace(step=50, v=torch.tensor([-60., -50., -60.]))
    spikes, calls = torch.tensor([1., 0., 0.]), []
    class Kernel:
        def _advance(self, occupancy, state):
            calls.append(state)
            return after, None, None, spikes
    kernel = Kernel()
    with capture_actual_steps(kernel, '_advance', sink):
        result = kernel._advance(None, before)
    sink._queue.join()
    assert calls == [before] and result[0] is after and result[3] is spikes
    assert '_advance' not in vars(kernel)
    assert sink.error is None
    saved_latest = (tmp_path / 'telemetry/latest.json').read_bytes()
    original = telemetry._atomic
    def fail_state(path, content):
        if path.suffix == '.f32':
            raise OSError('injected neural sidecar failure')
        original(path, content)
    monkeypatch.setattr(telemetry, '_atomic', fail_state)
    sink.capture_state(spikes, after, SimpleNamespace(step=51, v=after.v))
    sink.flush(51)
    summary = sink.close()
    assert not summary['complete'] and 'neural sidecar failure' in summary['error']
    assert (tmp_path / 'telemetry/latest.json').read_bytes() == saved_latest
    assert not (tmp_path / 'telemetry/batches/000002.json').exists()


def test_behavior_selection_covers_22_groups_with_bounded_deterministic_real_ids():
    from malecns_lab.learning.x2_behavior_telemetry import selection_manifest
    policy, identity = fixture_policy()
    policy.encoder = SimpleNamespace(input_map_id='fixture-input')
    groups = {f'G{i:02}': list(range(100 + i * 10, 104 + i * 10)) for i in range(20)}
    groups.update(R16_L=list(range(1000, 1020)), R16_R=list(range(2000, 2020)))
    graph = policy.brain.graph
    graph.body_ids = np.array(sorted({body for members in groups.values() for body in members}))
    graph.n = len(graph.body_ids)
    mapping = graph.identity['reference_provenance']['group_mapping']
    mapping.update(body_ids=groups, rules={name: {'fixture': True} for name in groups}, output_body_ids=groups['G00'])
    manifest, lookup = selection_manifest(policy, identity, 'fixture-run', 'fixture-session')
    selected = {int(item['body_id']) for item in manifest['selected_neurons']}
    assert len(selected) == manifest['selected_neuron_count'] == 32
    assert all(selected.intersection(members) for members in groups.values())
    assert len(selected.intersection(groups['R16_L'])) >= 2
    assert len(selected.intersection(groups['R16_R'])) >= 2
    assert all(lookup[int(item['body_id'])] == item['graph_index'] for item in manifest['selected_neurons'])
    mapping['body_ids'] = {name: list(reversed(members)) for name, members in reversed(list(groups.items()))}
    reordered, _ = selection_manifest(policy, identity, 'fixture-run', 'fixture-session')
    assert reordered['selected_neurons'] == manifest['selected_neurons']
    # Existing frozen-policy observation retains its output-only default.
    legacy, _ = telemetry.selection_manifest(policy, identity, 'fixture-run')
    assert [int(item['body_id']) for item in legacy['selected_neurons']] == groups['G00']


def test_behavior_selection_deduplicates_overlapping_groups():
    from malecns_lab.learning.x2_behavior_telemetry import selection_manifest
    policy, identity = fixture_policy()
    policy.encoder = SimpleNamespace(input_map_id='fixture-input')
    manifest, _ = selection_manifest(policy, identity, 'fixture-run', 'fixture-session')
    assert [item['body_id'] for item in manifest['selected_neurons']] == ['10', '12']
    assert all(set(group['body_ids']).intersection(item['body_id'] for item in manifest['selected_neurons'])
               for group in manifest['groups'])


def test_behavior_observer_records_sensory_spikes_while_output_is_silent(tmp_path):
    from malecns_lab.learning.x2_behavior_telemetry import BehaviorTelemetry
    policy, identity = fixture_policy()
    policy.encoder = SimpleNamespace(input_map_id='fixture-input')
    mapping = policy.brain.graph.identity['reference_provenance']['group_mapping']
    groups = {'DNg02_L': [10], 'R16_L': [11], 'R16_R': [12]}
    mapping.update(body_ids=groups, rules={name: {'fixture': True} for name in groups}, output_body_ids=[10])
    sink = BehaviorTelemetry(tmp_path, policy, identity, 'fixture-run', 'fixture-session')
    spikes = torch.tensor([0., 1., 1.])
    sink.capture(spikes, 0)
    sink.flush(1)
    assert sink.close()['complete']
    batch = read(tmp_path / 'telemetry/latest.json')
    assert batch['group_rates_hz'] == [0., 10000., 10000.]
    assert batch['spike_events'] == [{'body_id': '11', 'step': 0}, {'body_id': '12', 'step': 0}]
    assert spikes.tolist() == [0., 1., 1.]


def frozen_input_fixture():
    from scipy.sparse import csc_matrix
    from malecns_lab.learning.x2_behavior_telemetry import INPUT_SAMPLE_CHANNELS
    policy, identity = fixture_policy()
    graph = policy.brain.graph
    graph.n = 156
    graph.body_ids = np.arange(100, 256, dtype=np.int64)
    inputs, indices = tuple(range(100, 232)), tuple(range(132))
    outputs = list(range(232, 256))
    groups = {'R16_L': list(inputs[::2]), 'R16_R': list(inputs[1::2]),
              'DN_A': outputs[:12], 'DN_B': outputs[12:], 'DN_SHARED': [outputs[0], outputs[-1]]}
    mapping = graph.identity['reference_provenance']['group_mapping']
    mapping.update(body_ids=groups, output_body_ids=outputs,
                   rules={name: {'fixture': True} for name in groups})
    # Three real fixture members per channel: the latter two tie at the highest
    # outgoing degree. Their smaller body ID must win regardless of order.
    rows, columns = [], []
    for index in indices:
        degree = (1, 3, 3)[index % 3]
        rows.extend(range(degree)); columns.extend([index] * degree)
    graph.counts = csc_matrix((np.ones(len(rows)), (rows, columns)), shape=(graph.n, graph.n))
    encoder = SimpleNamespace(input_body_ids=inputs, input_indices=indices)
    policy.encoder = SimpleNamespace(input_map_id='fixture-input44')
    policy.engine = SimpleNamespace(
        channel_names=INPUT_SAMPLE_CHANNELS + tuple(f'other.{i}' for i in range(28)),
        assignments=np.repeat(np.arange(44, dtype=np.int64), 3),
        reference=SimpleNamespace(base=SimpleNamespace(brain=policy.brain, encoder=encoder)))
    return policy, identity


def test_frozen_input_samples_cover_fixed_channels_and_dn_without_activity():
    from malecns_lab.learning.x2_behavior_telemetry import INPUT_SAMPLE_CHANNELS, selection_manifest
    policy, identity = frozen_input_fixture()
    manifest, lookup = selection_manifest(policy, identity, 'fixture-run', 'fixture-session')
    selected = manifest['selected_neurons']
    assert len({item['body_id'] for item in selected}) == manifest['selected_neuron_count'] == 32
    assert [item['channel'] for item in selected[:16]] == list(INPUT_SAMPLE_CHANNELS)
    assert [int(item['body_id']) for item in selected[:16]] == list(range(101, 149, 3))
    assert manifest['sampled_input_channels'] == list(INPUT_SAMPLE_CHANNELS)
    assert manifest['unsampled_input_channels'] == list(policy.engine.channel_names[16:])
    outputs = set(policy.brain.graph.identity['reference_provenance']['group_mapping']['output_body_ids'])
    assert len(selected[16:]) == 16 and all(int(item['body_id']) in outputs for item in selected[16:])
    assert {item['group_id'] for item in selected[16:]} == {'DN_A', 'DN_B', 'DN_SHARED'}
    assert all(lookup[int(item['body_id'])] == item['graph_index'] for item in selected)
    # Selection reads neither requests nor spike values and is invariant to
    # storage order when each assignment stays paired with its real input ID.
    policy.current_task = 'SHOW_FORWARD'
    policy.spikes = object()
    encoder = policy.engine.reference.base.encoder
    encoder.input_body_ids = encoder.input_body_ids[::-1]
    encoder.input_indices = encoder.input_indices[::-1]
    policy.engine.assignments = policy.engine.assignments[::-1]
    again, _ = selection_manifest(policy, identity, 'another-run', 'another-session')
    assert again['selected_neurons'] == selected


@pytest.mark.parametrize('damage', ['missing', 'float', 'bool', 'length', 'range', 'index', 'duplicate', 'foreign_graph'])
def test_frozen_input_samples_reject_corrupt_mapping_instead_of_fallback(damage):
    from malecns_lab.learning.x2_behavior_telemetry import selection_manifest
    policy, identity = frozen_input_fixture()
    engine, encoder = policy.engine, policy.engine.reference.base.encoder
    if damage == 'missing':
        del engine.assignments
    elif damage == 'float':
        engine.assignments = engine.assignments.astype(float)
    elif damage == 'bool':
        encoder.input_indices = (False,) + encoder.input_indices[1:]
    elif damage == 'length':
        engine.assignments = engine.assignments[:-1]
    elif damage == 'range':
        engine.assignments[0] = 44
    elif damage == 'index':
        encoder.input_indices = (1, 0) + encoder.input_indices[2:]
    elif damage == 'duplicate':
        encoder.input_body_ids = (101,) + encoder.input_body_ids[1:]
    else:
        engine.reference.base.brain = SimpleNamespace(graph=SimpleNamespace())
    with pytest.raises(ValueError, match='[Ff]rozen telemetry'):
        selection_manifest(policy, identity, 'fixture-run', 'fixture-session')


def test_frozen_input_observer_records_only_actual_selected_spikes(tmp_path):
    from malecns_lab.learning.x2_behavior_telemetry import BehaviorTelemetry
    policy, identity = frozen_input_fixture()
    sink = BehaviorTelemetry(tmp_path, policy, identity, 'fixture-run', 'fixture-session')
    assert sink.error is None
    spikes = torch.zeros(policy.brain.graph.n)
    # Channel 3's selected member (task.SHOW_FORWARD) fires. Another real but
    # unselected member fires as well: it contributes to group rates only.
    spikes[10], spikes[9] = 1., 1.
    before = spikes.clone()
    sink.capture(spikes, 0); sink.flush(1)
    assert sink.close()['complete']
    batch = read(tmp_path / 'telemetry/latest.json')
    assert batch['spike_events'] == [{'body_id': '110', 'step': 0}]
    assert sum(batch['group_rates_hz']) > 0
    assert torch.equal(spikes, before)


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
