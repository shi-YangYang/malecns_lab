"""Same-instance X2 behavior spikes with explicit new provenance and linkage."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
from queue import Queue
import threading
import shutil
import time

from . import x2_telemetry as base

MANIFEST_SCHEMA = 'malecns_lab.x2_behavior_telemetry_manifest.v1'
BATCH_SCHEMA = 'malecns_lab.x2_behavior_telemetry_batch.v1'
SUMMARY_SCHEMA = 'malecns_lab.x2_behavior_telemetry_summary.v1'


def selection_manifest(policy, identity, run_id, session_id):
    manifest, lookup = base.selection_manifest(policy, identity, run_id)
    manifest.update(schema=MANIFEST_SCHEMA, session_id=session_id,
        input_map_id=policy.encoder.input_map_id,
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sampler_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        spike_source=getattr(policy, 'telemetry_spike_source',
            'actual X2BehaviorPolicy._step MPSLIF spikes from the unique decision policy instance'),
        execution_id_meaning='neural session execution; individual MC skills are linked by skill_execution_id',
        phase_source='engineering sensory stimulation; no biological task-region attribution')
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
            content = base._encoded(self.manifest)
            base._require(len(content) <= 1024*1024, 'Telemetry manifest exceeds budget')
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

    def _batch(self, record, values):
        record = super()._batch(record, values)
        record.update(schema=BATCH_SCHEMA, session_id=self.session_id,
                      checkpoint_id=self.manifest['checkpoint_id'], input_map_id=self.manifest['input_map_id'])
        return record

    def _summary(self, status):
        result = {**super()._summary(status), 'schema': SUMMARY_SCHEMA, 'session_id': self.session_id}
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
