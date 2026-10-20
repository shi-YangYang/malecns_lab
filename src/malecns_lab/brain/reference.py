"""Build an unpruned MaleCNS neuron reference from the complete official release.

Run ``python -m malecns_lab.brain.reference --help``. No device is imported or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .connectome import MALECNS_FILES, NT_SIGN, Connectome
from .metadata import annotation_hash, canonical_hash, file_hashes, load_identity, node_annotations, resolve_reference_groups, verified_release
from .reference_audit import IdentityClassifier, RawGraphLedger


def graph_hash(connectome: Connectome) -> str:
    """Canonical ordered identity and integer CSC contents, independent of ZIP metadata."""
    digest = hashlib.sha256()
    for values in (connectome.body_ids, np.asarray(connectome.weights.shape), connectome.weights.indptr,
                   connectome.weights.indices, connectome.weights.data):
        # Uniform byte order and width make matrix indexing width irrelevant.
        array = np.asarray(values, dtype="<i8")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_reference(connectome: Connectome) -> None:
    """Reject corrupted or stale full-reference identities before simulation."""
    if connectome.meta.get("asset_type") != "full_reference" or connectome.meta.get("format_version") != 2:
        raise ValueError("Expected a V2 full_reference asset")
    if connectome.body_ids is None or np.any(connectome.body_ids[1:] <= connectome.body_ids[:-1]):
        raise ValueError("Reference body IDs must be sorted and unique")
    if not np.issubdtype(connectome.weights.data.dtype, np.integer):
        raise ValueError("Reference requires exact integer synapse weights")
    provenance = connectome.meta["provenance"]
    if canonical_hash(provenance) != connectome.meta["source_id"]:
        raise ValueError("Reference provenance identity mismatch")
    if graph_hash(connectome) != provenance["graph_sha256"]:
        raise ValueError("Reference graph contents do not match the recorded identity")
    if annotation_hash(connectome) != provenance["node_metadata_sha256"]:
        raise ValueError("Reference node metadata does not match the recorded identity")
    for field, values in connectome.annotations.items():
        if values.shape != (connectome.n,) or connectome.annotation_missing[field].shape != (connectome.n,):
            raise ValueError(f"Invalid reference annotation shape: {field}")
    actual = resolve_reference_groups(connectome)
    if actual != provenance["group_mapping"]:
        raise ValueError("Reference group membership does not match recorded identity")


def load_reference(path: str | Path) -> Connectome:
    connectome = Connectome.load(path)
    validate_reference(connectome)
    return connectome


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_reference(data_dir: str | Path, identity_dir: str | Path, out_dir: str | Path, *, verbose: bool = True) -> dict:
    """Build once, refusing any existing destination; failure leaves an explicit report."""
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    data_dir, identity_dir, out_dir = Path(data_dir).resolve(), Path(identity_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log = print if verbose else lambda *args, **kwargs: None
    ledger = None
    try:
        manifest, sources = verified_release(data_dir)
        log("Verified complete release and current checksums of the three consumed tables", flush=True)
        ann = feather.read_table(data_dir / MALECNS_FILES["annotations"]).to_pandas()
        body_ids, identity = load_identity(identity_dir, ann, sources)
        selected = ann.set_index("bodyId", drop=False).loc[body_ids].reset_index(drop=True)
        nt = feather.read_table(data_dir / MALECNS_FILES["neurotransmitters"], columns=["body", "consensus_nt"]).to_pandas()
        annotations, missing, signs = node_annotations(selected, nt)
        del nt, selected
        log(f"Author-defined neuron identities: {len(body_ids):,}; no type, status, degree or side exclusion", flush=True)
        classifier = IdentityClassifier(ann)
        evidence_dir = out_dir / "identity_evidence"
        evidence_dir.mkdir()
        copied_evidence = {}
        for name in ("identity_findings.txt", "identity_reconciliation.json", identity["id_list_file"],
                     "all_annotated_identity_decisions.csv", "traced_without_superclass.csv", "neurons_without_traced_status.csv"):
            path = identity_dir / name
            if not path.is_file():
                raise ValueError(f"Identity audit evidence missing: {path}")
            shutil.copyfile(path, evidence_dir / name)
            copied_evidence[name] = file_hashes(evidence_dir / name)["sha256"]
        _write_json(out_dir / "download_manifest.json", manifest)
        ledger = RawGraphLedger(out_dir, classifier)
        rows, cols, values = [], [], []
        self_rows = self_synapses = 0
        reader = ipc.open_file(str(data_dir / MALECNS_FILES["weights"]))
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre, post, weight = (batch.column(name).to_numpy() for name in ("body_pre", "body_post", "weight"))
            if not np.issubdtype(weight.dtype, np.integer) or np.any(weight < 0):
                raise ValueError("Raw weights must be nonnegative integer synapse counts")
            codes = ledger.append(pre, post, weight)
            keep = codes == 0
            source = np.searchsorted(body_ids, pre[keep]).astype(np.int32)
            target = np.searchsorted(body_ids, post[keep]).astype(np.int32)
            counts = weight[keep].astype(np.int64)
            rows.append(target)
            cols.append(source)
            values.append(counts * signs[source])
            self_mask = source == target
            self_rows += int(self_mask.sum())
            self_synapses += int(counts[self_mask].sum(dtype=np.int64))
            if verbose and (batch_index + 1) % 400 == 0:
                log(f"Weights: {ledger.offsets[-1]:,} raw rows classified", flush=True)
        edge_ledger = ledger.finish()
        ledger = None
        log("Raw edge and node ledgers complete; assembling integer CSC", flush=True)
        matrix = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(len(body_ids), len(body_ids)), dtype=np.int64)
        del rows, cols, values
        matrix.sort_indices()
        expected_rows = edge_ledger["edge_categories"][0]["rows"]
        if matrix.nnz != expected_rows:
            raise ValueError("Repeated source body pairs in raw weights; review before aggregating them")
        connectome = Connectome("malecns-v1.0-full-reference-v2", matrix, annotations["type"], annotations["somaSide"],
                                annotations["superclass"], body_ids, annotations=annotations, annotation_missing=missing)
        if connectome.n_synapses != edge_ledger["edge_categories"][0]["synapses"]:
            raise ValueError("Integer graph synapse count differs from source ledger")
        group_mapping = resolve_reference_groups(connectome)
        incident = np.diff(matrix.indptr) > 0
        incident[matrix.indices] = True
        isolated_ids = body_ids[~incident]
        np.save(out_dir / "isolated_neuron_ids.npy", isolated_ids, allow_pickle=False)
        graph_summary = {
            "neurons": connectome.n, "connections": connectome.n_connections, "synapses": connectome.n_synapses,
            "self_connections": self_rows, "self_synapses": self_synapses, "isolated_neurons": len(isolated_ids),
            "min_positive_weight": int(np.abs(matrix.data).min()), "max_weight": int(np.abs(matrix.data).max()),
            "unknown_nt_sign_assumptions": int(annotations["nt_sign_assumed"].sum()),
            "missing_nt": int(missing["consensus_nt"].sum()),
            "missing_annotations": {name: int(mask.sum()) for name, mask in missing.items()},
        }
        edge_ledger["source"] = sources[MALECNS_FILES["weights"]]
        edge_ledger["reference"] = graph_summary
        _write_json(out_dir / "graph_ledger.json", edge_ledger)
        _write_json(out_dir / "group_mapping.json", group_mapping)
        graph_identity = graph_hash(connectome)
        rules = {
            "node_definition": identity["definition"], "node_order": "ascending numeric body ID",
            "positive_internal_weights": "all retained; no minimum above 1", "self_edges": "retained",
            "isolated_neurons": "retained", "weight_dtype": "signed int64 original synapse counts",
            "sign_map": NT_SIGN, "unknown_nt_sign": 1,
            "sign_assumption": "source consensus NT; all unrecognized or missing values explicitly assumed excitatory",
            "no_amine_amplitude_scaling": True,
            "published_boundary": "official minconf 0.5; pre-threshold observations unavailable",
        }
        code_files = (Path(__file__), Path(__file__).with_name("metadata.py"), Path(__file__).with_name("reference_audit.py"),
                      Path(__file__).with_name("connectome.py"))
        identity_payload = {
            "format": "cnsbridge.malecns_full_reference.v2", "graph_sha256": graph_identity,
            "node_metadata_sha256": annotation_hash(connectome),
            "identity_ids_sha256": identity["id_list_sha256"], "sources": sources, "rules": rules,
            "group_mapping": group_mapping, "identity_evidence_sha256": copied_evidence,
            "builder_source_sha256": {path.name: file_hashes(path)["sha256"] for path in code_files},
        }
        connectome.meta = {
            "format_version": 2, "asset_type": "full_reference", "source_id": canonical_hash(identity_payload),
            "graph_sha256": graph_identity, "source": "male-cns:v1.0", "license": "CC-BY-4.0",
            "provenance": identity_payload, "statistics": graph_summary,
            "roles": {name: rule["role"] for name, rule in group_mapping["rules"].items()},
        }
        asset = connectome.save(out_dir / "full_reference.npz", overwrite=False)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {
            "status": "complete", "asset": str(asset), "source_id": connectome.meta["source_id"],
            "asset_sha256": file_hashes(asset)["sha256"], "graph_sha256": graph_identity,
            "statistics": graph_summary, "raw_graph": {key: edge_ledger[key] for key in
                ("raw_edge_rows", "raw_synapses", "raw_endpoint_nodes", "raw_endpoint_nodes_total")},
            "resource_measurement": {"elapsed_seconds": time.perf_counter() - started,
                "peak_process_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
                "method": "perf_counter and getrusage(RUSAGE_SELF); process peak includes imports and checksums"},
            "limits": identity["unresolved_biological_limits"],
        }
        _write_json(out_dir / "build_report.json", result)
        log(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException as error:
        if ledger is not None:
            ledger.close()
        _write_json(out_dir / "build_failed.json", {"status": "failed", "error": repr(error),
                                                   "elapsed_seconds": time.perf_counter() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--identity-dir", type=Path, default=Path("data/audits/malecns_v2_identity"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/malecns_v2"))
    args = parser.parse_args()
    build_reference(args.data_dir, args.identity_dir, args.out_dir)


if __name__ == "__main__":
    main()
