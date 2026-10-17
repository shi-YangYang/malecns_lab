"""Bounded-memory, source-row-addressable ledgers for the raw segment graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CATEGORIES = ("author_defined_neuron", "annotated_glia", "annotated_outside_neuron_definition", "unannotated_segment")


class IdentityClassifier:
    def __init__(self, annotations):
        annotations = annotations.sort_values("bodyId")
        self.ids = annotations.bodyId.to_numpy(np.int64)
        neuron = (annotations.superclass.notna() & annotations.superclass.ne("")).to_numpy(bool)
        glia = annotations.status.eq("Glia").fillna(False).to_numpy(bool)
        self.classes = np.where(neuron, 0, np.where(glia, 1, 2)).astype(np.uint8)

    def classify(self, ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.ids, ids).clip(0, len(self.ids) - 1)
        return np.where(self.ids[pos] == ids, self.classes[pos], 3).astype(np.uint8)


class RawGraphLedger:
    """Every edge retains its original zero-based global row; nodes are bucketed."""

    def __init__(self, directory: Path, classifier: IdentityClassifier, buckets: int = 64):
        self.directory = directory
        self.classifier = classifier
        self.buckets = buckets
        self.node_dir = directory / "raw_nodes"
        self.node_dir.mkdir()
        self.temp_dir = directory / ".node_partitions"
        self.temp_dir.mkdir()
        self.streams = [(self.temp_dir / f"{index:02d}.bin").open("xb") for index in range(buckets)]
        self.edge_file = (directory / "raw_edge_classes.u8").open("xb")
        self.edge_rows = np.zeros(32, dtype=np.int64)
        self.synapses = np.zeros(32, dtype=np.int64)
        self.offsets = [0]

    def append(self, pre: np.ndarray, post: np.ndarray, weights: np.ndarray) -> np.ndarray:
        codes = 4 * self.classifier.classify(pre) + self.classifier.classify(post)
        codes += (weights == 0).astype(np.uint8) * 16
        codes.tofile(self.edge_file)
        self.edge_rows += np.bincount(codes, minlength=32)
        for code in np.unique(codes):
            self.synapses[code] += weights[codes == code].sum(dtype=np.int64)
        self.offsets.append(self.offsets[-1] + len(weights))
        # Partition before deduplication: bounded RAM even for tens of millions of raw bodies.
        endpoints = np.concatenate((pre, post))
        bucket = endpoints % self.buckets
        order = np.argsort(bucket, kind="stable")
        sizes = np.bincount(bucket, minlength=self.buckets)
        ends = np.cumsum(sizes)
        ordered = endpoints[order]
        start = 0
        for stream, end in zip(self.streams, ends):
            ordered[start:end].tofile(stream)
            start = end
        return codes

    def close(self) -> None:
        self.edge_file.close()
        for stream in self.streams:
            stream.close()

    def finish(self) -> dict:
        self.close()
        node_counts = np.zeros(4, dtype=np.int64)
        observed_annotated = []
        for index in range(self.buckets):
            temp = self.temp_dir / f"{index:02d}.bin"
            ids = np.unique(np.fromfile(temp, dtype=np.int64))
            classes = self.classifier.classify(ids)
            node_counts += np.bincount(classes, minlength=4)
            observed_annotated.append(ids[classes != 3])
            np.savez_compressed(self.node_dir / f"bucket_{index:02d}.npz", body_ids=ids, identity_class=classes)
            temp.unlink()
        self.temp_dir.rmdir()
        observed = np.sort(np.concatenate(observed_annotated))
        absent = np.setdiff1d(self.classifier.ids, observed, assume_unique=True)
        np.savez_compressed(self.directory / "annotated_nodes_without_raw_edges.npz", body_ids=absent,
                            identity_class=self.classifier.classify(absent))
        rows = []
        for code in range(32):
            pair = code % 16
            rows.append({"code": code, "pre_class": CATEGORIES[pair // 4], "post_class": CATEGORIES[pair % 4],
                         "weight_rule": "positive" if code < 16 else "zero",
                         "rows": int(self.edge_rows[code]), "synapses": int(self.synapses[code])})
        return {
            "format": "cnsbridge.raw_segment_ledger.v1",
            "categories": dict(enumerate(CATEGORIES)),
            "raw_edge_rows": int(self.edge_rows.sum()), "raw_synapses": int(self.synapses.sum()),
            "edge_categories": rows,
            "row_index": {"file": "raw_edge_classes.u8", "dtype": "uint8", "index": "zero-based source table global row",
                          "record_batch_offsets": self.offsets, "reference_rule": "code == 0",
                          "decode": "code % 16 = pre_class * 4 + post_class; code >= 16 means zero weight"},
            "raw_endpoint_nodes": {name: int(count) for name, count in zip(CATEGORIES, node_counts)},
            "raw_endpoint_nodes_total": int(node_counts.sum()),
            "node_index": {"directory": "raw_nodes", "bucket_rule": f"body_id % {self.buckets}",
                           "arrays": "body_ids (sorted unique int64), identity_class (uint8)",
                           "absence": "annotated_nodes_without_raw_edges.npz includes isolated annotations"},
            "annotated_nodes_without_raw_edges": int(len(absent)),
        }
