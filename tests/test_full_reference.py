"""Small source-table fixtures exercise contracts, never substitute for a real brain."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

pa = pytest.importorskip("pyarrow")
import pyarrow.feather as feather  # noqa: E402

from malecns_lab.brain.brain import Brain, load_connectome  # noqa: E402
from malecns_lab.brain.connectome import MALECNS_BASE, MALECNS_FILES, Connectome  # noqa: E402
from malecns_lab.brain.metadata import ANNOTATION_FIELDS, SOURCE_USES, file_hashes  # noqa: E402
from malecns_lab.brain.reference import build_reference, load_reference, validate_reference  # noqa: E402
from malecns_lab.config import load_config  # noqa: E402


def source_fixture(tmp_path):
    data, evidence = tmp_path / "release", tmp_path / "identity"
    data.mkdir()
    evidence.mkdir()
    types = ["DNg02"] * 29 + ["DNp03"] * 2 + ["DNp01"] * 2 + ["R1-R6", None, None, "glia", None]
    sides = ["L"] * 15 + ["R"] * 14 + ["L", "R", "L", "R", None, None, None, None, None]
    ids = np.arange(100, 100 + len(types), dtype=np.int64)
    fields = {name: [None] * len(ids) for name in ANNOTATION_FIELDS}
    fields.update(bodyId=ids, type=types, somaSide=sides, superclass=["neuron"] * 36 + [None, None],
                  status=["Traced"] * 33 + [None, "Orphan", "Anchor", "Glia", "Traced"],
                  rootSide=[None] * 33 + ["L", "R", None, None, None],
                  subclass=[None] * 34 + ["haltere", None, None, None])
    feather.write_feather(pa.table(fields), data / MALECNS_FILES["annotations"])
    feather.write_feather(pa.table({"body": ids[:34], "consensus_nt": ["gaba"] + ["acetylcholine"] * 33}),
                          data / MALECNS_FILES["neurotransmitters"])
    feather.write_feather(pa.table({"body_pre": [100, 100, 101, 136, 137, 999, 135],
                                   "body_post": [100, 101, 133, 100, 100, 100, 100],
                                   "weight": [2, 1, 16777217, 3, 4, 5, 0]}), data / MALECNS_FILES["weights"])
    entries = []
    for name in SOURCE_USES:
        path = data / name
        if not path.exists():
            path.write_bytes(b"preserved unconsumed fixture")
        md5 = file_hashes(path)["md5"]
        entries.append({"name": "v1.0/connectome-data/flat-connectome/" + name, "size": path.stat().st_size,
                        "md5Hash": md5, "verified_md5": md5, "generation": "1",
                        "url": f"{MALECNS_BASE}/{name}?generation=1"})
    (data / "download_manifest.json").write_text(json.dumps({"dataset": "male-cns:v1.0", "download_complete": True,
        "files": entries, "total_bytes": sum(item["size"] for item in entries)}))
    id_path = evidence / "official_neuron_ids_v1.csv"
    id_path.write_text("body_id\n" + "".join(f"{value}\n" for value in ids[:36]))
    info = {
        "definition": "Nonempty superclass, per author Methods; no other exclusion",
        "definition_quote": "Bodies in the dataset are defined as neurons if they have a superclass.",
        "primary_methods_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/",
        "id_list_file": id_path.name, "id_list_sha256": file_hashes(id_path)["sha256"], "neuron_count": 36,
        "annotation_sha256": file_hashes(data / MALECNS_FILES["annotations"])["sha256"],
        "source_annotation": entries[0], "unresolved_biological_limits": ["Fixture only, not biological evidence"],
    }
    (evidence / "identity_reconciliation.json").write_text(json.dumps(info))
    for name in ("identity_findings.txt", "all_annotated_identity_decisions.csv", "traced_without_superclass.csv",
                 "neurons_without_traced_status.csv"):
        (evidence / name).write_text("fixture only\n")
    return data, evidence


def test_complete_reference_retains_edges_identity_and_nullable_metadata(tmp_path):
    data, evidence = source_fixture(tmp_path)
    destination = tmp_path / "reference"
    result = build_reference(data, evidence, destination, verbose=False)
    graph = load_reference(result["asset"])
    assert graph.n == 36 and graph.n_connections == 3
    assert graph.n_synapses == 16777220  # exact beyond float32 single-integer precision
    assert graph.weights[0, 0] == -2 and graph.weights[1, 0] == -1
    assert graph.meta["statistics"]["isolated_neurons"] == 33
    assert graph.annotation_missing["type"][34:36].all()
    assert graph.annotations["status"][34:36].tolist() == ["Orphan", "Anchor"]
    assert graph.annotation_missing["consensus_nt"][34:36].all()
    assert graph.annotations["nt_sign_assumed"][34:36].all()
    assert graph.body_ids[graph.group("R16_L")].tolist() == [133]
    assert graph.body_ids[graph.group("HAL_R")].tolist() == [134]
    assert len(graph.meta["provenance"]["group_mapping"]["output_body_ids"]) == 33
    assert np.fromfile(destination / "raw_edge_classes.u8", dtype=np.uint8).tolist() == [0, 0, 0, 4, 8, 12, 16]
    assert result["raw_graph"]["raw_synapses"] == 16777232
    assert result["raw_graph"]["raw_endpoint_nodes_total"] == 7
    with pytest.raises(FileExistsError):
        build_reference(data, evidence, destination, verbose=False)
    with pytest.raises(FileExistsError):
        graph.save(result["asset"], overwrite=False)
    cfg = load_config(Path(__file__).parents[1] / "configs/malecns_v2.yaml")
    with pytest.warns(UserWarning):
        Brain(graph, cfg)
    recorded_roles = dict(graph.meta["roles"])
    recorded_groups = {name: indices.tolist() for name, indices in graph.groups.items()}
    moved_output = deepcopy(cfg)
    moved_output["inputs"]["DNg02_L"] = moved_output["outputs"].pop("DNg02_L")
    with pytest.raises(ValueError, match="group roles"):
        Brain(graph, moved_output)
    overlapping_groups = deepcopy(cfg)
    overlapping_groups["inputs"]["DNg02_L"] = overlapping_groups["outputs"]["DNg02_L"]
    with pytest.raises(ValueError, match="groups overlap: DNg02_L"):
        Brain(graph, overlapping_groups)
    assert graph.meta["roles"] == recorded_roles
    assert {name: indices.tolist() for name, indices in graph.groups.items()} == recorded_groups
    with pytest.raises(ValueError, match="V2 configuration"):
        Brain(graph, load_config())
    with pytest.raises(ValueError, match="graph contents"):
        Brain(graph.subgraph(np.arange(35)), cfg)
    graph.annotations["status"][0] = "changed"
    with pytest.raises(ValueError, match="node metadata"):
        validate_reference(graph)


def test_unverified_release_and_identity_mismatch_stop_before_build(tmp_path):
    data, evidence = source_fixture(tmp_path)
    id_path = evidence / "official_neuron_ids_v1.csv"
    id_path.write_text(id_path.read_text().replace("100\n", "99\n"))
    with pytest.raises(ValueError, match="ID-list hash"):
        build_reference(data, evidence, tmp_path / "wrong_identity", verbose=False)
    assert not (tmp_path / "wrong_identity/full_reference.npz").exists()
    assert (tmp_path / "wrong_identity/build_failed.json").exists()
    (data / MALECNS_FILES["weights"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed release file"):
        build_reference(data, evidence, tmp_path / "wrong_release", verbose=False)


def test_research_controls_rejected_from_both_production_paths(tmp_path):
    graph = Connectome("research", sparse.eye(1, format="csc"), np.array(["x"]), np.array(["L"]),
                        meta={"asset_type": "research_control"})
    with pytest.raises(ValueError, match="offline research"):
        Brain(graph, {})
    path = graph.save(tmp_path / "research.npz")
    with pytest.raises(ValueError, match="offline research"):
        load_connectome(path)
