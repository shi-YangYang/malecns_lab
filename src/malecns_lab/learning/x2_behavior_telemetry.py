"""Same-instance X2 behavior spikes with explicit new provenance and linkage."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
from queue import Queue
import threading

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
        spike_source='actual X2BehaviorPolicy._step MPSLIF spikes from the unique decision policy instance',
        execution_id_meaning='neural session execution; individual MC skills are linked by skill_execution_id',
        phase_source='engineering sensory stimulation; no biological task-region attribution')
    return manifest, lookup


class BehaviorTelemetry(base.SpikeTelemetry):
    """Reuse read-only device reduction/writer, never legacy policy provenance."""
    def __init__(self, directory, policy, identity, run_id, session_id):
        self.directory = Path(directory) / 'telemetry'
        self.run_id, self.execution_id, self.session_id = run_id, identity['execution_id'], session_id
        self.manifest = self.error = None
        self.dropped_batches = self.written_batches = self.produced_batches = 0
        self.archive_bytes = self._latest_bytes = self._summary_bytes = self._count = 0
        self._closed = False
        self._context = {'phase': 'precondition', 'observation_sequence': None,
                         'decision_sequence': None, 'request_id': None, 'skill_execution_id': None}
        self._queue, self._stop = Queue(maxsize=base.QUEUE_BATCHES), threading.Event()
        self._thread = self._device_counts = None
        try:
            self.manifest, self._lookup = selection_manifest(policy, identity, run_id, session_id)
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
        return {**super()._summary(status), 'schema': SUMMARY_SCHEMA, 'session_id': self.session_id}
