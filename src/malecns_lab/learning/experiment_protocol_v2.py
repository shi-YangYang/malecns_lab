"""Immutable registration of real material, graph identities and offline budgets."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..brain.reference import load_reference
from ..diagnostics.runtime_v2 import MODEL_FILES
from ..diagnostics.sources_v2 import VideoSourceV2
from .contracts_v2 import RIDGE_CANDIDATES, digest, file_digest, load_labels, nonempty, validate_protocol, write_json
from .graph_controls import load_research_control

FORMAT = "cnsbridge.connectome_experiment.v2"
REGISTRATION_FORMAT = "cnsbridge.registered_connectome_experiment.v2"


def code_inventory() -> dict[str, str]:
    package = Path(__file__).parents[1]
    names = set(MODEL_FILES) | {"diagnostics/replay_storage_v2.py", "diagnostics/sources_v2.py"}
    names |= {f"learning/{name}.py" for name in ("graph_controls", "graph_diagnostics", "research_replay_v2",
              "experiment_protocol_v2", "experiment_v2", "contracts_v2", "dataset_v2", "decoder_v2", "evaluation_v2", "train_v2")}
    return {str(package / name): file_digest(package / name) for name in sorted(names)}


def template() -> dict:
    """An intentionally incomplete template; null values are not empirical values."""
    return {"format": FORMAT, "protocol_id": "REPLACE_WITH_PREDECLARED_ID", "phase": "pilot",
            "parent": "REPLACE_WITH_FULL_REFERENCE_NPZ", "graphs": [], "input_seeds": [], "sources": [],
            "task": {"policy_id": "REPLACE", "teacher_id": "REPLACE", "teacher_kind": "independent_human",
                     "description": "REPLACE_WITH_ACTION_SEMANTICS_BEFORE_LABELLING",
                     "tolerances": {"vx": None, "wz": None}, "stationary_tolerances": {"vx": None, "wz": None}},
            "practical_delta": None, "budget_seconds": None, "replay_budget_seconds": None,
            "bootstrap": {"seed": None, "replicates": None}, "comparisons": [],
            "ridge_candidates": list(RIDGE_CANDIDATES),
            "template_note": "Not registered and not executable until real materials and explicit SI tolerances/budgets are supplied"}


def _safe_id(value, name):
    nonempty(value, name)
    if Path(value).name != value or value in (".", ".."):
        raise ValueError(f"Unsafe {name}")
    return value


def source_from_entry(entry: dict) -> VideoSourceV2:
    recording = Path(entry["recording"]) if entry.get("recording") else None
    return VideoSourceV2(Path(entry["video"]), scene_id=entry["scene_id"], session_id=entry["session_id"],
                         family_id=entry["family_id"], stimulus_kind=entry.get("stimulus_kind", "unknown"), recording_dir=recording)


def _freeze_sources(protocol: dict, base: Path, files: dict) -> list[dict]:
    frozen, assigned, names = [], {}, set()
    if not isinstance(protocol.get("sources"), list) or not protocol["sources"]:
        raise ValueError("Real recorded sources and independent action labels are required for registration")
    for raw in protocol["sources"]:
        entry = copy.deepcopy(raw)
        name = _safe_id(entry.get("id"), "source id")
        if name in names:
            raise ValueError("Duplicate source id")
        names.add(name)
        if entry.get("split") not in ("train", "val", "test"):
            raise ValueError("Each source needs a train/val/test split before extraction")
        if bool(entry.get("video")) == bool(entry.get("recording")):
            raise ValueError("Each source selects exactly one video or ground recording")
        if entry.get("recording"):
            recording = (base / entry["recording"]).resolve()
            entry["recording"] = str(recording)
            entry["video"] = str(recording / "video.avi")
            for name in ("session.json", "frames.csv"):
                files[str(recording / name)] = file_digest(recording / name)
        else:
            entry["video"] = str((base / entry["video"]).resolve())
        current = source_from_entry(entry)
        files[entry["video"]] = current.metadata["video_sha256"]
        label_path = (base / entry["labels"]).resolve()
        entry["labels"] = str(label_path)
        if entry.get("label_format", "v2") not in ("v2", "legacy_v1_intervals"):
            raise ValueError("Unknown label format")
        _, labels = load_labels(label_path, source_id=current.metadata["source_id"], task=protocol["task"],
                                 legacy=entry.get("label_format") == "legacy_v1_intervals")
        files[str(label_path)] = labels["file_sha256"]
        ranges = entry.get("segments")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("Predeclared nonempty segment ranges required; no outcome-selected restarts")
        previous, segment_names = 0, set()
        for segment in ranges:
            segment_name = _safe_id(segment.get("segment_id"), "segment id")
            start, end = segment.get("start_frame"), segment.get("end_frame")
            if (segment_name in segment_names or type(start) is not int or type(end) is not int
                    or not previous <= start < end <= current.total_frames):
                raise ValueError("Segments must be unique, ordered, nonoverlapping and within the video")
            segment_names.add(segment_name)
            previous = end
        for field in ("scene_id", "session_id", "family_id", "source_id", "video_sha256"):
            token = (field, current.metadata[field])
            if token in assigned and assigned[token] != entry["split"]:
                raise ValueError(f"Split leakage across {field}")
            assigned[token] = entry["split"]
        entry["input_source"] = current.metadata
        entry["label_identity"] = labels
        frozen.append(entry)
    splits = {entry["split"] for entry in frozen}
    required = {"train", "val", "test"} if protocol["phase"] == "confirmatory" else {"train", "val"}
    if not required <= splits:
        raise ValueError(f"Phase requires source splits {sorted(required)}")
    return frozen


def register_experiment(protocol_path: str | Path, out: str | Path) -> dict:
    """Hash a complete plan before any extraction/fit/test metrics; never overwrite."""
    protocol_path, out = Path(protocol_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT}")
    if protocol.get("ridge_candidates") != list(RIDGE_CANDIDATES):
        raise ValueError("All graph/seed fits must use the same nine registered ridge candidates")
    # Reuse P2's task/phase/scales/delta/budget contract without invented segments.
    validate_protocol({**protocol, "format": "cnsbridge.learning_protocol.v2",
                       "segments": [{"path": "not-read", "labels": "not-read", "split": "train"}]})
    if not isinstance(protocol.get("replay_budget_seconds"), (int, float)) or not 0 < protocol["replay_budget_seconds"] <= protocol["budget_seconds"]:
        raise ValueError("Explicit positive per-replay budget no larger than the total budget required")
    seeds = protocol.get("input_seeds")
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Unique explicit uint64 input seeds required")
    base = protocol_path.parent
    parent_path = (base / protocol["parent"]).resolve()
    parent = load_reference(parent_path)
    files = {str(parent_path): file_digest(parent_path)}
    graphs, ids, graph_sources = [], set(), set()
    if not isinstance(protocol.get("graphs"), list) or not protocol["graphs"]:
        raise ValueError("Explicit graph inventory required")
    for entry in protocol["graphs"]:
        graph_id = _safe_id(entry.get("id"), "graph id")
        if graph_id in {"visual", "mean", "zero"}:
            raise ValueError("Graph ids cannot use reserved baseline names visual, mean or zero")
        if graph_id in ids:
            raise ValueError("Duplicate graph id")
        ids.add(graph_id)
        graph_path = (base / entry["path"]).resolve()
        if entry["role"] == "G":
            if graph_path != parent_path:
                raise ValueError("G must be the registered complete parent asset")
            graph, provenance = parent, None
        else:
            graph = load_research_control(graph_path, parent)
            provenance = graph.meta["provenance"]
            if entry["role"] != provenance["transform"]:
                raise ValueError("Graph role differs from verified transformation")
        if graph.meta["source_id"] in graph_sources:
            raise ValueError("Repeated graph identity cannot count as an independent graph condition")
        graph_sources.add(graph.meta["source_id"])
        files[str(graph_path)] = file_digest(graph_path)
        graphs.append({"id": graph_id, "role": entry["role"], "path": str(graph_path),
                       "source_id": graph.meta["source_id"], "graph_sha256": graph.meta["graph_sha256"],
                       "graph_seed": provenance["seed"] if provenance else None,
                       "parameters": provenance["parameters"] if provenance else {}})
    real = [graph for graph in graphs if graph["role"] == "G"]
    if len(real) != 1:
        raise ValueError("Exactly one complete real reference G required")
    pairs = {}
    for graph in graphs:
        if graph["role"].startswith("P_"):
            pairs.setdefault(graph["parameters"]["pair_id"], set()).add(graph["role"])
    if any(roles != {"P_target", "P_matched"} for roles in pairs.values()):
        raise ValueError("P target and its exact matched control must both be registered")
    comparisons = protocol.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("Predeclare comparisons; no selecting comparisons after scores")
    seen_comparisons = set()
    for comparison in comparisons:
        key = (comparison.get("real"), comparison.get("control"))
        if key[0] != real[0]["id"] or key[1] not in (ids - {real[0]["id"]}) | {"visual", "mean", "zero"}:
            raise ValueError("Comparisons must name G and an explicit registered graph or V/M/Z baseline")
        if key in seen_comparisons:
            raise ValueError("Duplicate comparison")
        seen_comparisons.add(key)
    frozen = copy.deepcopy(protocol)
    frozen.update(parent=str(parent_path), graphs=graphs)
    frozen["sources"] = _freeze_sources(protocol, base, files)
    registration = {"format": REGISTRATION_FORMAT, "protocol": frozen, "files_sha256": files,
                    "code_sha256": code_inventory(), "input_protocol_sha256": file_digest(protocol_path),
                    "test_policy": "registration checks file hashes/metadata/label format; pilot has no test extraction/fit/metrics; confirmatory test extraction and metrics require frozen registration",
                    "claim_scope": "registration integrity does not establish previously unseen data or biological advantage",
                    "candidate_policy": "all mechanical-valid registered graphs retained, regardless of silence, failure or score"}
    registration["registration_id"] = digest(registration)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, registration)
    return registration


def validate_registration(path: str | Path) -> dict:
    registration = json.loads(Path(path).read_text())
    if registration.get("format") != REGISTRATION_FORMAT or digest({key: value for key, value in registration.items()
            if key != "registration_id"}) != registration.get("registration_id"):
        raise ValueError("Experiment registration identity mismatch")
    for field in ("files_sha256", "code_sha256"):
        for name, expected in registration[field].items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"Frozen {field} changed: {name}; register a new protocol before execution")
    return registration
