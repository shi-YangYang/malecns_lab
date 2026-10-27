"""Read-only, full-cell voltage/spike observation for the live frozen policy.

This is a display observer, not a second integration or a model input. Returned
LIF voltages are post-step (including spike reset); membrane peaks before reset
cannot be recovered from this channel. Every cell remains in graph order.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from . import x2_telemetry as base

MANIFEST_SCHEMA = 'malecns_lab.neural_state_manifest.v1'
REF_SCHEMA = 'malecns_lab.neural_state_ref.v1'
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
LIVE_SLOTS = 8


def selection_manifest(policy):
    graph, config = policy.brain.graph, policy.brain.config
    bodies = [str(int(value)) for value in graph.body_ids]
    graph_hash = graph.identity['reference_provenance'].get('graph_sha256')
    base._require(len(bodies) == graph.n and len(set(bodies)) == graph.n
                  and isinstance(graph_hash, str) and re.fullmatch(r'[a-f0-9]{64}', graph_hash),
                  'Whole-brain telemetry graph identity is invalid')
    base._require(graph.identity.get('reference_graph_sha256', graph_hash) == graph_hash,
                  'Whole-brain reference graph hashes differ')
    rest, threshold = float(config.v_rest_mv), float(config.v_threshold_mv)
    base._require(math.isfinite(rest) and math.isfinite(threshold) and rest < threshold,
                  'Whole-brain telemetry voltage bounds are invalid')
    return dict(schema=MANIFEST_SCHEMA, node_count=graph.n, body_ids=bodies,
        body_ids_sha256=hashlib.sha256(base._encoded(bodies)).hexdigest(), graph_sha256=graph_hash,
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        channels=['voltage_mean_mv', 'voltage_delta_mean_mv', 'spike_rate_hz'], dtype='float32-le',
        v_rest_mv=rest, v_threshold_mv=threshold, window_steps=base.BLOCK_STEPS,
        baseline='request_start_pre_step', sample='post_step_window_mean',
        voltage_semantics='post-integration and post-spike-reset LIF state; not pre-reset peaks',
        archive_bytes_limit=MAX_ARCHIVE_BYTES, live_slots=LIVE_SLOTS,
        live_archive_semantics='archived=false references are ephemeral and cannot be used for historical replay')


class NeuralStateObserver:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = directory, manifest
        self.n = manifest['node_count']
        self._request = object()
        self._baseline = self._voltage_sum = self._delta_sum = None
        self.count = 0
        self.archive_bytes = self.live_bytes = self.archived_batches = self.live_batches = 0
        self.archive_truncated = False
        self._slots = set()

    @property
    def storage_bytes(self):
        return self.archive_bytes + self.live_bytes

    def begin_request(self, request_id):
        base._require(self.count == 0, 'Whole-brain request changed mid-window')
        if request_id != self._request:
            self._request = request_id
            self._baseline = None

    def capture(self, before, after, step):
        import torch
        base._require(before.step == step and after.step == step + 1,
                      'Whole-brain state step differs from spike step')
        v, previous = after.v.detach(), before.v.detach()
        base._require(v.shape == previous.shape == (self.n,) and v.dtype == previous.dtype == torch.float32
                      and v.device == previous.device and self.count < base.BLOCK_STEPS,
                      'Whole-brain voltage shape, dtype or device differs')
        if self._baseline is None:
            self._baseline = previous.clone()
        if self._voltage_sum is None:
            self._voltage_sum, self._delta_sum = torch.zeros_like(v), torch.zeros_like(v)
        # Only observer-owned buffers are mutated. No CPU readback in a step.
        self._voltage_sum.add_(v)
        self._delta_sum.add_(v - self._baseline)
        self.count += 1

    def window_values(self, count, spikes, dt_ms):
        base._require(self.count == count and count > 0,
                      'Whole-brain voltage and spike windows differ')
        return (self._voltage_sum / count, self._delta_sum / count,
                spikes / (count * dt_ms / 1000))

    def clear_window(self):
        self._voltage_sum.zero_()
        self._delta_sum.zero_()
        self.count = 0

    def write(self, record, values):
        import numpy as np
        base._require(values.shape == (self.n * 3,) and np.isfinite(values).all()
                      and (values[self.n * 2:] >= 0).all(), 'Invalid whole-brain state payload')
        content = values.astype('<f4', copy=False).tobytes()
        archived = self.archive_bytes + len(content) <= MAX_ARCHIVE_BYTES and not self.archive_truncated
        if archived:
            relative = f"neural/{record['batch_sequence']:06d}.f32"
        else:
            relative = f"neural/live-{(record['batch_sequence'] - 1) % LIVE_SLOTS}.f32"
        # The state is atomically replaced before its batch/latest is published.
        base._atomic(self.directory / relative, content)
        if archived:
            self.archive_bytes += len(content)
            self.archived_batches += 1
        else:
            self.archive_truncated = True
            self.live_batches += 1
            if relative not in self._slots:
                self._slots.add(relative)
                self.live_bytes += len(content)
        record['neural_state'] = dict(schema=REF_SCHEMA, file=relative, bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(), body_ids_sha256=self.manifest['body_ids_sha256'],
            node_count=self.n, archived=archived)

    def summary(self):
        return dict(neural_state_archive_bytes=self.archive_bytes, neural_state_live_bytes=self.live_bytes,
            neural_state_archived_batches=self.archived_batches, neural_state_live_batches=self.live_batches,
            neural_state_archive_truncated=self.archive_truncated)
