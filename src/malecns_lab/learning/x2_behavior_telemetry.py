"""Same-instance X2 behavior spikes with explicit new provenance and linkage."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from numbers import Integral
from pathlib import Path
from queue import Queue
import threading
import shutil
import time

from . import x2_telemetry as base
from . import x2_neural_state as neural

MANIFEST_SCHEMA = 'malecns_lab.x2_behavior_telemetry_manifest.v1'
BATCH_SCHEMA = 'malecns_lab.x2_behavior_telemetry_batch.v1'
SUMMARY_SCHEMA = 'malecns_lab.x2_behavior_telemetry_summary.v1'
INPUT_SAMPLE_CHANNELS = (
    'task.GREET', 'task.FACE_LEFT', 'task.FACE_RIGHT', 'task.SHOW_FORWARD', 'task.END_INTERACTION',
    'request.changed', 'capability.HOLD', 'capability.ADVANCE', 'capability.TURN_LEFT',
    'capability.TURN_RIGHT', 'capability.WAVE_RIGHT', 'protection.READY', 'protection.LATCHED',
    'protection.RECOVERING', 'feedback_valid', 'feedback_fresh',
)


def _round_robin(groups, lookup, limit, seen):
    groups = sorted(groups, key=lambda group: (-group['neuron_count'], group['id'] or ''))
    remaining = [(group['id'], iter(sorted(map(int, group['body_ids'])))) for group in groups]
    selected = []
    while remaining and len(selected) < limit:
        next_round = []
        for group_id, members in remaining:
            body_id = next((body for body in members if body not in seen), None)
            if body_id is None:
                continue
            seen.add(body_id)
            selected.append({'body_id': str(body_id), 'graph_index': lookup[body_id], 'group_id': group_id})
            next_round.append((group_id, members))
            if len(selected) == limit:
                break
        remaining = next_round
    return selected


def _frozen_input_samples(policy, manifest, lookup):
    # The frozen adapter's public encoder only carries its identity. The actual
    # input44 mapping belongs to this already-loaded numerical engine.
    from .x2_frozen_history_policy import FrozenHistoryPolicy
    engine = getattr(policy, 'engine', None)
    if not isinstance(policy, FrozenHistoryPolicy) and not any(
            hasattr(engine, name) for name in ('assignments', 'channel_names', 'reference')):
        return None
    import numpy as np
    channels = getattr(engine, 'channel_names', None)
    base._require(isinstance(channels, (tuple, list)) and len(channels) == 44
                  and all(isinstance(name, str) for name in channels) and len(set(channels)) == 44
                  and tuple(channels[:16]) == INPUT_SAMPLE_CHANNELS, 'Frozen telemetry input44 channels differ')
    reference = getattr(engine, 'reference', None)
    original = getattr(reference, 'base', None)
    encoder = getattr(original, 'encoder', None)
    base._require(getattr(getattr(original, 'brain', None), 'graph', None) is policy.brain.graph,
                  'Frozen telemetry input mapping belongs to another graph')

    def integer_vector(value, label):
        valid_container = (isinstance(value, (tuple, list)) or
            isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in 'iu')
        base._require(valid_container and len(value) > 0 and all(
            isinstance(item, Integral) and not isinstance(item, (bool, np.bool_)) for item in value),
            f'Invalid frozen telemetry {label}')
        return tuple(int(item) for item in value)

    bodies = integer_vector(getattr(encoder, 'input_body_ids', None), 'input body IDs')
    indices = integer_vector(getattr(encoder, 'input_indices', None), 'input indices')
    assignments = integer_vector(getattr(engine, 'assignments', None), 'channel assignments')
    outputs = set(policy.brain.graph.identity['reference_provenance']['group_mapping']['output_body_ids'])
    base._require(len(bodies) == len(indices) == len(assignments) and len(set(bodies)) == len(bodies)
                  and len(set(indices)) == len(indices) and not outputs.intersection(bodies)
                  and all(lookup.get(body) == index for body, index in zip(bodies, indices))
                  and set(assignments) == set(range(44)), 'Frozen telemetry input mapping differs from graph/channels')
    counts = policy.brain.graph.counts
    base._require(getattr(counts, 'format', None) == 'csc' and counts.shape == (policy.brain.graph.n,) * 2,
                  'Frozen telemetry requires the existing CSC connection counts')
    # Validated full-reference counts store one nonzero connection per CSC row
    # entry. Read only two CPU offsets per input; no graph copy or GPU readback.
    selected = []
    for channel_index, channel in enumerate(INPUT_SAMPLE_CHANNELS):
        members = [(body, index) for body, index, assigned in zip(bodies, indices, assignments)
                   if assigned == channel_index]
        body, index = min(members, key=lambda item: (
            -int(counts.indptr[item[1] + 1] - counts.indptr[item[1]]), item[0]))
        group_id = next((group['id'] for group in manifest['groups'] if str(body) in group['body_ids']), None)
        selected.append(dict(body_id=str(body), graph_index=index, group_id=group_id, channel=channel))
    seen = {int(item['body_id']) for item in selected}
    output_groups, grouped_outputs = [], set()
    for group in manifest['groups']:
        members = outputs.intersection(map(int, group['body_ids']))
        if members:
            output_groups.append({**group, 'body_ids': members})
            grouped_outputs.update(members)
    if outputs - grouped_outputs:
        output_groups.append(dict(id=None, neuron_count=len(outputs - grouped_outputs),
                                  body_ids=outputs - grouped_outputs))
    selected.extend(_round_robin(output_groups, lookup, base.MAX_SELECTED - len(selected), seen))
    manifest.update(sampled_input_channels=list(INPUT_SAMPLE_CHANNELS), unsampled_input_channels=list(channels[16:]))
    return selected


def selection_manifest(policy, identity, run_id, session_id):
    manifest, lookup = base.selection_manifest(policy, identity, run_id)
    selected = _frozen_input_samples(policy, manifest, lookup)
    if selected is None:
        selected = _round_robin(manifest['groups'], lookup, base.MAX_SELECTED, set())
        selection = ('round-robin across validated reference groups ordered by descending member count then group ID; '
                     'ascending numeric body IDs within each group')
    else:
        selection = ('one actual input neuron per fixed first 16 input44 channels, ranked by descending nonzero '
                     'outgoing connection count then numeric body ID; remaining slots round-robin across reference '
                     'DN output groups by descending group size then group ID, ascending body IDs within each group; '
                     '28 input channels and most neurons have no individual spike sample')
    base._require(selected, 'Behavior telemetry requires nonempty reference groups')
    manifest.update(selected_neurons=selected, selected_neuron_count=len(selected),
        selection=selection + '; globally unique, at most 32; independent of task and activity')
    manifest.update(schema=MANIFEST_SCHEMA, session_id=session_id,
        input_map_id=policy.encoder.input_map_id,
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampler_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        spike_source=getattr(policy, 'telemetry_spike_source',
            'actual X2BehaviorPolicy._step MPSLIF spikes from the unique decision policy instance'),
        execution_id_meaning='neural session execution; individual MC skills are linked by skill_execution_id',
        phase_source='engineering sensory stimulation; no biological task-region attribution')
    if getattr(policy, 'telemetry_neural_state', False):
        manifest['neural_state'] = neural.selection_manifest(policy)
        manifest['neural_state_observer_sha256'] = hashlib.sha256(Path(neural.__file__).read_bytes()).hexdigest()
    return manifest, lookup


class BehaviorTelemetry(base.SpikeTelemetry):
    """Reuse read-only device reduction/writer, never legacy policy provenance."""
    def __init__(self, directory, policy, identity, run_id, session_id, *, continuous=False):
        self.directory = Path(directory) / 'telemetry'
        self.run_id, self.execution_id, self.session_id = run_id, identity['execution_id'], session_id
        self.manifest = self.error = None
        self.dropped_batches = self.written_batches = self.produced_batches = 0
        self.archive_bytes = self._latest_bytes = self._summary_bytes = self._count = 0
        self._closed = False
        self.continuous = continuous
        self.live_only_batches = 0
        self.archive_limit_reason = None
        self.archive_last_batch_sequence = None
        self._space_checked_at = -float('inf')
        self._context = {'phase': 'precondition', 'observation_sequence': None,
                         'decision_sequence': None, 'request_id': None, 'skill_execution_id': None}
        self._queue, self._stop = Queue(maxsize=base.QUEUE_BATCHES), threading.Event()
        self._thread = self._device_counts = None
        self._neural_state = None
        try:
            self.manifest, self._lookup = selection_manifest(policy, identity, run_id, session_id)
            if continuous:
                self.manifest.update(archive_scope='initial_window', archive_batch_limit=4096,
                    archive_limit_behavior='preserve initial batches; continue replacing latest.json',
                    session_profile='continuous')
            self._n = policy.brain.graph.n
            self._members = [self._lookup[int(body)] for group in self.manifest['groups'] for body in group['body_ids']]
            self._member_groups = [i for i, group in enumerate(self.manifest['groups']) for _ in group['body_ids']]
            self._selected = [item['graph_index'] for item in self.manifest['selected_neurons']]
            self.directory.mkdir(exist_ok=False)
            (self.directory / 'batches').mkdir()
            if 'neural_state' in self.manifest:
                (self.directory / 'neural').mkdir()
                self._neural_state = neural.NeuralStateObserver(self.directory, self.manifest['neural_state'])
            content = base._encoded(self.manifest)
            budget = neural.MAX_MANIFEST_BYTES if self._neural_state is not None else 1024 * 1024
            base._require(len(content) <= budget, 'Telemetry manifest exceeds budget')
            base._atomic(self.directory / 'manifest.json', content)
            self.archive_bytes = len(content)
            self._write_summary('running')
            self._thread = threading.Thread(target=self._writer, name='behavior-telemetry', daemon=True)
            self._thread.start()
        except Exception as error:
            self._fail(error)

    def begin_decision(self, observation, decision_seq):
        self.begin('forward', observation.observation_seq)
        self._context.update(request_id=observation.request_id, decision_sequence=decision_seq,
                             skill_execution_id=observation.feedback.execution_id or None)
        if self._neural_state is not None and self.error is None:
            try:
                self._neural_state.begin_request(observation.request_id)
            except Exception as error:
                self._fail(error)

    def capture_state(self, spikes, before, after):
        if self.error is not None or self._closed:
            return
        try:
            if self._neural_state is not None:
                self._neural_state.capture(before, after, before.step)
            self.capture(spikes, before.step)
        except Exception as error:
            self._fail(error)

    def _extra_device_values(self):
        if self._neural_state is None:
            return ()
        return self._neural_state.window_values(self._count, self._device_counts, self.manifest['dt_ms'])

    def _clear_extra_values(self):
        if self._neural_state is not None:
            self._neural_state.clear_window()

    def _write_extra_values(self, record, values):
        if self._neural_state is None:
            return super()._write_extra_values(record, values)
        self._neural_state.write(record, values)

    def _extra_storage_bytes(self):
        return self._neural_state.storage_bytes if self._neural_state is not None else 0

    def _batch(self, record, values):
        record = super()._batch(record, values)
        record.update(schema=BATCH_SCHEMA, session_id=self.session_id,
                      checkpoint_id=self.manifest['checkpoint_id'], input_map_id=self.manifest['input_map_id'])
        return record

    def _summary(self, status):
        result = {**super()._summary(status), 'schema': SUMMARY_SCHEMA, 'session_id': self.session_id}
        if self._neural_state is not None:
            result.update(self._neural_state.summary())
            result['neural_state_complete'] = result['complete'] and not self._neural_state.archive_truncated
        if self.continuous:
            result.update(archive_scope='initial_window', archive_truncated=self.archive_limit_reason is not None,
                archive_limit_reason=self.archive_limit_reason,
                archive_last_batch_sequence=self.archive_last_batch_sequence,
                live_only_batches=self.live_only_batches)
        return result

    def _archive_batch(self, record, content):
        if not self.continuous:
            return super()._archive_batch(record, content)
        now = time.monotonic()
        if now - self._space_checked_at >= 1.:
            base._require(shutil.disk_usage(self.directory).free >= 512 * 1024 * 1024,
                          'Telemetry disk reserve below 512 MiB')
            self._space_checked_at = now
        if self.archive_limit_reason is None:
            if self.archive_bytes + len(content) + base.MAX_BATCH_BYTES + 8192 > base.MAX_ARCHIVE_BYTES:
                self.archive_limit_reason = 'archive_byte_limit'
            elif self.written_batches >= 4096 or record['batch_sequence'] > 999999:
                self.archive_limit_reason = 'archive_batch_limit'
        if self.archive_limit_reason:
            # This is an explicit partial archive, never a complete-session record.
            # The same real spike batch still updates latest.json in the writer.
            self.live_only_batches += 1
            return False
        archived = super()._archive_batch(record, content)
        self.archive_last_batch_sequence = record['batch_sequence']
        return archived
