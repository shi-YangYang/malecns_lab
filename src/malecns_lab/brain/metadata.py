"""Verified release provenance and nullable MaleCNS V2 node metadata."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from .connectome import MALECNS_BASE, MALECNS_FILES, NT_SIGN, Connectome, GroupSpec

ANNOTATION_FIELDS = ("type", "superclass", "class", "subclass", "status", "statusLabel", "somaSide", "rootSide")
SOURCE_USES = {
    MALECNS_FILES["annotations"]: "author-defined neuron identity, nullable annotations and group membership",
    MALECNS_FILES["neurotransmitters"]: "raw consensus NT and explicit source-sign hypothesis",
    MALECNS_FILES["weights"]: "sole graph edge source; every positive internal edge including self edges",
    "body-stats-male-cns-v1.0-minconf-0.5.feather": "retained segment statistics; not added as edges",
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather": "retained official derivative; not added again",
    "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather": "retained official derivative; not added again",
    "syn-partners-male-cns-v1.0-minconf-0.5.feather": "retained synapse-level representation; not added again",
    "syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather": "retained official derivative; not added again",
    "syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather": "retained official derivative; not added again",
    "syn-points-male-cns-v1.0-minconf-0.5.feather": "retained synaptic coordinates; no spatial mapping inferred",
    "tbar-neurotransmitters-male-cns-v1.0.feather": "retained per-presynapse predictions; body consensus used initially",
}


def file_hashes(path: Path) -> dict:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": base64.b64encode(md5.digest()).decode()}


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def annotation_hash(connectome: Connectome) -> str:
    digest = hashlib.sha256()
    for key in sorted(connectome.annotations):
        for array in (connectome.annotations[key], connectome.annotation_missing[key]):
            contiguous = np.ascontiguousarray(array)
            digest.update(json.dumps([key, contiguous.dtype.str, contiguous.shape]).encode())
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def verified_release(data_dir: Path) -> tuple[dict, dict]:
    """Validate all 11 preserved files; rehash only the three consumed tables."""
    manifest = json.loads((data_dir / "download_manifest.json").read_text())
    if manifest.get("dataset") != "male-cns:v1.0" or manifest.get("download_complete") is not True:
        raise ValueError("Complete official male-cns:v1.0 download manifest required")
    entries = {Path(item["name"]).name: item for item in manifest["files"]}
    if set(entries) != set(SOURCE_USES) or len(manifest["files"]) != len(SOURCE_USES):
        raise ValueError("Manifest must contain exactly the 11 official flat-connectome objects")
    verified = {}
    for name, item in entries.items():
        path = data_dir / name
        generation = item.get("generation")
        expected_url = f"{MALECNS_BASE}/{name}?generation={generation}"
        if not generation or item.get("url") != expected_url:
            raise ValueError(f"Missing or inconsistent pinned official generation: {name}")
        if item.get("verified_md5") != item.get("md5Hash") or not item.get("md5Hash"):
            raise ValueError(f"No successful download checksum record: {name}")
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing or changed release file: {name}")
        verified[name] = {**item, "use": SOURCE_USES[name], "current_check": "size and prior verified MD5"}
        if name in MALECNS_FILES.values():
            hashes = file_hashes(path)
            if hashes["md5"] != item["md5Hash"]:
                raise ValueError(f"Current file MD5 mismatch: {name}")
            verified[name].update(hashes, current_check="current MD5 and SHA256")
    if sum(item["size"] for item in entries.values()) != manifest["total_bytes"]:
        raise ValueError("Manifest total size mismatch")
    return manifest, verified


def load_identity(identity_dir: Path, annotations, sources: dict) -> tuple[np.ndarray, dict]:
    """Check a separately evidenced ID manifest against every published annotation."""
    evidence = json.loads((identity_dir / "identity_reconciliation.json").read_text())
    id_path = identity_dir / evidence["id_list_file"]
    if id_path.parent.resolve() != identity_dir.resolve():
        raise ValueError("Identity ID-list must be in the evidence directory")
    if file_hashes(id_path)["sha256"] != evidence["id_list_sha256"]:
        raise ValueError("Identity ID-list hash mismatch")
    annotation_source = sources[MALECNS_FILES["annotations"]]
    if evidence["annotation_sha256"] != annotation_source["sha256"]:
        raise ValueError("Identity evidence was derived from a different annotation table")
    for field in ("generation", "md5Hash", "name"):
        if evidence["source_annotation"][field] != annotation_source[field]:
            raise ValueError(f"Identity annotation provenance differs: {field}")
    if evidence.get("definition_quote") != "Bodies in the dataset are defined as neurons if they have a superclass.":
        raise ValueError("Unrecognized official identity definition; review a new identity protocol first")
    if evidence.get("primary_methods_url") != "https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/":
        raise ValueError("Missing reviewed author Methods identity evidence")
    ids = np.loadtxt(id_path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=1)
    if not ids.size or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Identity IDs must be nonempty, unique and ascending")
    if annotations.bodyId.isna().any() or annotations.bodyId.duplicated().any():
        raise ValueError("Annotation body IDs must be unique and non-null")
    selected = annotations.superclass.notna() & annotations.superclass.ne("")
    derived = np.sort(annotations.loc[selected, "bodyId"].to_numpy(np.int64))
    if not np.array_equal(ids, derived) or len(ids) != evidence["neuron_count"]:
        raise ValueError("Full annotation identity and independently evidenced ID-list differ")
    return ids, evidence


def nullable_strings(values) -> tuple[np.ndarray, np.ndarray]:
    missing = values.isna().to_numpy(bool)
    return values.fillna("").to_numpy(dtype=str), missing


def node_annotations(selected, nt) -> tuple[dict, dict, np.ndarray]:
    values, missing = {}, {}
    for field in ANNOTATION_FIELDS:
        values[field], missing[field] = nullable_strings(selected[field])
    if nt.body.duplicated().any():
        raise ValueError("Duplicate NT body identities")
    nt_values = nt.set_index("body").reindex(selected.bodyId.to_numpy())["consensus_nt"]
    values["consensus_nt"], missing["consensus_nt"] = nullable_strings(nt_values)
    signs = np.array([int(NT_SIGN.get(value.lower(), 1)) for value in values["consensus_nt"]], dtype=np.int8)
    values["nt_sign_assumed"] = np.array([value.lower() not in NT_SIGN for value in values["consensus_nt"]])
    missing["nt_sign_assumed"] = np.zeros(len(selected), bool)
    values["nt_sign"] = signs
    missing["nt_sign"] = np.zeros(len(selected), bool)
    return values, missing, signs


def reference_group_specs() -> dict[str, GroupSpec]:
    specs = {}
    patterns = {"R16": ["^R1-R6$"], **{f"T4{letter}": [f"^T4{letter}$", f"^T5{letter}$"] for letter in "abcd"},
                "LPLC2": ["^LPLC2$"], "LC4": ["^LC4$"], "HAL": ["^haltere$"],
                "DNg02": ["^DNg02"], "DNp03": ["^DNp03$"], "DNp01": ["^DNp01$", "^GF$"]}
    for group, types in patterns.items():
        for side in ("L", "R"):
            name = f"{group}_{side}"
            specs[name] = GroupSpec(name, types, side, "output" if group.startswith("DN") else "input",
                                    field="subclass" if group == "HAL" else "type",
                                    side_field="rootSide" if group in ("R16", "HAL") else "somaSide")
    return specs


def resolve_reference_groups(connectome: Connectome) -> dict:
    specs = reference_group_specs()
    connectome.resolve_groups(specs)
    outputs = np.unique(np.concatenate([connectome.group(name) for name, spec in specs.items() if spec.role == "output"]))
    if len(outputs) != 33:
        raise ValueError(f"V2 output schema requires exactly 33 cells; observed {len(outputs)}")
    return {
        "version": "cnsbridge.malecns_v2_groups.v1",
        "rules": {name: {"patterns": spec.types, "field": spec.field, "side": spec.side,
                         "side_field": spec.side_field, "role": spec.role} for name, spec in specs.items()},
        "body_ids": {name: connectome.body_ids[idx].tolist() for name, idx in connectome.groups.items()},
        "output_body_ids": np.sort(connectome.body_ids[outputs]).tolist(),
        "haltere_main_experiment": "disabled; missing telemetry is not measured zero",
    }
