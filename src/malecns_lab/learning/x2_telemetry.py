"""Bounded observation of actual frozen-policy spikes, never a second brain.

Import and CLI preview are stdlib-only. Device sampling is enabled explicitly
by a supervised worker after its historical graph and execution are validated.
The writer owns no policy/state references and receives only copied CPU data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

MANIFEST_SCHEMA = "malecns_lab.x2_telemetry_manifest.v1"
BATCH_SCHEMA = "malecns_lab.x2_telemetry_batch.v1"
SUMMARY_SCHEMA = "malecns_lab.x2_telemetry_summary.v1"
BLOCK_STEPS = 50
MAX_SELECTED = 32
QUEUE_BATCHES = 8
MAX_BATCH_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def selection_manifest(policy, execution_identity, run_id, *, selected_body_ids=None):
    """Use authenticated reference groups, including cells without a soma."""
    graph = policy.brain.graph
    identity = graph.identity
    _require(identity.get("role") == "full_reference", "Telemetry requires the validated complete reference")
    mapping = identity["reference_provenance"]["group_mapping"]
    _require(mapping["version"] == "cnsbridge.malecns_v2_groups.v1", "Unknown reference group mapping")
    lookup = {int(body_id): index for index, body_id in enumerate(graph.body_ids)}
    _require(len(lookup) == graph.n, "Graph body IDs are not unique")
    groups = []
    for name in sorted(mapping["body_ids"]):
        members = mapping["body_ids"][name]
        _require(members and len(set(members)) == len(members) and all(body_id in lookup for body_id in members),
                 "Reference telemetry group is empty or differs from graph")
        groups.append({"id": name, "name": name, "body_ids": [str(body_id) for body_id in members],
                       "neuron_count": len(members), "rule": deepcopy(mapping["rules"][name])})
    outputs = sorted(mapping["output_body_ids"])
    _require(outputs and len(set(outputs)) == len(outputs) and all(body_id in lookup for body_id in outputs),
             "Reference output IDs differ from graph")
    chosen = outputs[:MAX_SELECTED] if selected_body_ids is None else [int(value) for value in selected_body_ids]
    _require(0 < len(chosen) <= MAX_SELECTED and len(set(chosen)) == len(chosen)
             and all(body_id in lookup for body_id in chosen), "Selected neuron IDs must identify at most 32 graph cells")
    selected = [{"body_id": str(body_id), "graph_index": lookup[body_id],
                 "group_id": next((group["id"] for group in groups if str(body_id) in group["body_ids"]), None)}
                for body_id in chosen]
    manifest = {
        "schema": MANIFEST_SCHEMA, "run_id": run_id, "execution_id": execution_identity["execution_id"],
        "checkpoint_id": execution_identity["checkpoint_id"], "policy_id": policy.policy_id,
        "model_id": policy.brain.model_id, "graph": deepcopy(identity),
        "source_files_sha256": deepcopy(execution_identity["source_files_sha256"]),
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dt_ms": policy.brain.config.dt_ms, "batch_steps": BLOCK_STEPS,
        "groups": groups, "selected_neurons": selected, "selected_neuron_count": len(selected),
        "reference_output_neuron_count": len(outputs),
        "selection": "ascending reference output body IDs, first 32" if selected_body_ids is None else "explicit graph body IDs",
        "spike_source": "actual FrozenInference._step spikes tensor; not reconstructed from voltage",
        "group_rate_definition": "sum of member spikes / member count / sampled neural seconds",
        "step_interval": "step_start inclusive, step_end exclusive; spike step is the pre-update step",
        "clock_domain": "host_worker_monotonic", "host_monotonic_ns": time.monotonic_ns(),
        "limits": {"selected_neurons": MAX_SELECTED, "queue_batches": QUEUE_BATCHES,
                   "batch_bytes": MAX_BATCH_BYTES, "archive_bytes": MAX_ARCHIVE_BYTES, "display_hz": 10},
        "unrecorded_neurons": "No individual spike events are recorded for unselected cells",
    }
    return manifest, lookup


class SpikeTelemetry:
    """Fail-isolated sampler; all numerical tensors are read-only inputs.

    Capture adds spikes to a private device vector and stores selected cells in
    a 50-step device buffer. Only flush performs one device-to-host copy. GPU
    reductions, JSON encoding and bounded disk I/O never alter model tensors.
    """

    def __init__(self, directory, policy, execution_identity, run_id, *, selected_body_ids=None):
        self.directory = Path(directory) / "telemetry"
        self.run_id, self.execution_id = run_id, execution_identity["execution_id"]
        self.manifest = None
        self.error = None
        self.dropped_batches = self.written_batches = self.produced_batches = 0
        self.archive_bytes = 0
        self._latest_bytes = 0
        self._summary_bytes = 0
        self._count = 0
        self._closed = False
        self._context = {"phase": "precondition", "observation_sequence": None, "decision_sequence": None}
        self._queue = Queue(maxsize=QUEUE_BATCHES)
        self._stop = threading.Event()
        self._thread = None
        self._device_counts = None
        try:
            self.manifest, self._lookup = selection_manifest(policy, execution_identity, run_id,
                                                            selected_body_ids=selected_body_ids)
            self._n = policy.brain.graph.n
            self._members = [self._lookup[int(body_id)] for group in self.manifest["groups"] for body_id in group["body_ids"]]
            self._member_groups = [index for index, group in enumerate(self.manifest["groups"]) for _ in group["body_ids"]]
            self._selected = [item["graph_index"] for item in self.manifest["selected_neurons"]]
            self.directory.mkdir(exist_ok=False)
            (self.directory / "batches").mkdir()
            content = _encoded(self.manifest)
            _require(len(content) <= 1024 * 1024, "Telemetry manifest exceeds archive budget")
            _atomic(self.directory / "manifest.json", content)
            self.archive_bytes = len(content)
            self._write_summary("running")
            self._thread = threading.Thread(target=self._writer, name="x2-telemetry-writer", daemon=True)
            self._thread.start()
        except Exception as error:
            self._fail(error)

    def _fail(self, error):
        if self.error is None:
            self.error = f"{type(error).__name__}:{error}"

    def begin(self, phase, observation_sequence=None):
        if self.error is not None or self._closed:
            return
        try:
            _require(phase in ("precondition", "forward") and self._count == 0, "Telemetry phase changed mid-batch")
            _require((phase == "precondition" and observation_sequence is None)
                     or (phase == "forward" and type(observation_sequence) is int and observation_sequence > 0),
                     "Invalid telemetry observation sequence")
            self._context = {"phase": phase, "observation_sequence": observation_sequence, "decision_sequence": None}
        except Exception as error:
            self._fail(error)

    def capture(self, spikes, step):
        if self.error is not None or self._closed:
            return
        try:
            import torch
            if self._device_counts is None:
                self._device_counts = torch.zeros(self._n, dtype=spikes.dtype, device=spikes.device)
                self._device_spikes = torch.zeros((BLOCK_STEPS, len(self._selected)), dtype=spikes.dtype, device=spikes.device)
                self._device_selected = torch.tensor(self._selected, dtype=torch.int64, device=spikes.device)
                self._device_members = torch.tensor(self._members, dtype=torch.int64, device=spikes.device)
                self._device_groups = torch.tensor(self._member_groups, dtype=torch.int64, device=spikes.device)
            _require(tuple(spikes.shape) == (self._n,) and self._count < BLOCK_STEPS, "Telemetry spike shape or batch overflow")
            if self._count == 0:
                self._step_start = step
            _require(step == self._step_start + self._count, "Telemetry neural steps are not contiguous")
            self._device_counts.add_(spikes.detach())
            self._device_spikes[self._count].copy_(spikes.detach().index_select(0, self._device_selected))
            self._count += 1
        except Exception as error:
            self._fail(error)

    def flush(self, step_end):
        if self.error is not None or self._closed or self._count == 0:
            return
        try:
            import torch
            _require(step_end == self._step_start + self._count, "Telemetry flush step differs")
            sums = self._device_counts.new_zeros(len(self.manifest["groups"]))
            sums.index_add_(0, self._device_groups, self._device_counts.index_select(0, self._device_members))
            packed = torch.cat((sums, self._device_spikes[:self._count].flatten()))
            # One CPU transfer per batch; no .item(), .tolist() or CPU copy in capture.
            values = packed.detach().cpu().numpy().copy()
            self.produced_batches += 1
            record = {"schema": BATCH_SCHEMA, "run_id": self.run_id, "execution_id": self.execution_id,
                      "batch_sequence": self.produced_batches, **self._context,
                      "step_start": self._step_start, "step_end": step_end, "dt_ms": self.manifest["dt_ms"],
                      "host_monotonic_ns": time.monotonic_ns(), "dropped_batches": self.dropped_batches}
            try:
                self._queue.put_nowait((record, values))
            except Full:
                self.dropped_batches += 1
            self._device_counts.zero_()
            self._count = 0
        except Exception as error:
            self._fail(error)

    def _batch(self, record, values):
        import numpy as np
        count = len(self.manifest["groups"])
        steps = record["step_end"] - record["step_start"]
        _require(np.isfinite(values).all(), "Nonfinite telemetry observation")
        seconds = steps * self.manifest["dt_ms"] / 1000
        record["group_rates_hz"] = [float(values[index]) / group["neuron_count"] / seconds
                                    for index, group in enumerate(self.manifest["groups"])]
        selected = values[count:].reshape(steps, len(self._selected))
        _require(np.isin(selected, (0., 1.)).all(), "Telemetry spike tensor is not binary")
        record["spike_events"] = [{"body_id": self.manifest["selected_neurons"][column]["body_id"],
                                    "step": record["step_start"] + int(offset)}
                                   for offset, column in zip(*np.nonzero(selected))]
        record["dropped_batches"] = self.dropped_batches
        return record

    def _archive_batch(self, record, content):
        # Reserve room for the live snapshot and summary in the same budget.
        _require(self.archive_bytes + len(content) + MAX_BATCH_BYTES + 8192 <= MAX_ARCHIVE_BYTES,
                 "Telemetry archive exceeds byte budget")
        _atomic(self.directory / "batches" / f"{record['batch_sequence']:06d}.json", content)
        return True

    def _writer(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=.05)
            except Empty:
                continue
            try:
                if self.error is not None:
                    self.dropped_batches += 1
                    continue
                record, values = item
                content = _encoded(self._batch(record, values))
                _require(len(content) <= MAX_BATCH_BYTES, "Telemetry batch exceeds byte budget")
                archived = self._archive_batch(record, content)
                _atomic(self.directory / "latest.json", content)
                if archived:
                    self.archive_bytes += len(content)
                    self.written_batches += 1
                self._latest_bytes = len(content)
                self._write_summary("running")
            except Exception as error:
                self.dropped_batches += 1
                self._fail(error)
                try:
                    self._write_summary("failed")
                except Exception:
                    pass  # The worker result still carries a storage failure.
            finally:
                self._queue.task_done()
        try:
            self._write_summary("failed" if self.error else "completed")
        except Exception as error:
            self._fail(error)

    def _summary(self, status):
        return {"schema": SUMMARY_SCHEMA, "run_id": self.run_id, "execution_id": self.execution_id,
                "status": status, "error": self.error, "errors": [self.error] if self.error else [],
                "produced_batches": self.produced_batches, "batch_count": self.written_batches,
                "written_batches": self.written_batches, "dropped_batches": self.dropped_batches,
                "complete": self._closed and status == "completed" and self.error is None and self.dropped_batches == 0
                            and self._count == 0 and self.written_batches == self.produced_batches,
                "partial_batch_steps": self._count, "archive_bytes": self.archive_bytes,
                "total_bytes": self.archive_bytes + self._latest_bytes + self._summary_bytes,
                "host_monotonic_ns": time.monotonic_ns()}

    def _write_summary(self, status):
        summary = self._summary(status)
        # Include the size of this summary itself; the decimal length stabilizes.
        for _ in range(4):
            content = _encoded(summary)
            total = self.archive_bytes + self._latest_bytes + len(content)
            if summary["total_bytes"] == total:
                break
            summary["total_bytes"] = total
        content = _encoded(summary)
        _atomic(self.directory / "summary.json", content)
        self._summary_bytes = len(content)

    def close(self, *, completed=True):
        self._closed = True
        if not completed:
            self._fail(RuntimeError("Numerical worker did not complete; telemetry is partial"))
        if self._count:
            self._fail(RuntimeError("Unflushed partial telemetry batch"))
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                self._fail(TimeoutError("Telemetry writer did not close within 2 seconds"))
                return self._summary("failed")
        status = "failed" if self.error else "completed"
        try:
            self._write_summary(status)
        except Exception as error:
            self._fail(error)
            status = "failed"
        return self._summary(status)
