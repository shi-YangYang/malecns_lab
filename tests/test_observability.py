"""Small numerical/evidence fixtures only; never a MaleCNS observation result."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
import torch

from cnsbridge.brain.policy_groups import numerical_test_policy_map
from cnsbridge.brain.tonic import LaminaTonicLIF
from cnsbridge.brain.trainable import TrainableLIF
from cnsbridge.brain.trainable_graph import numerical_test_graph
from cnsbridge.senses.spatial_drive import build_spatial_input_map
from cnsbridge.diagnostics import observability as cli
from cnsbridge.diagnostics.observability_collect import collect_mode, selections, state_digest
from cnsbridge.diagnostics.observability_evidence import (BINDINGS, SAMPLE_KEYS, STATE_KEYS, arrays,
    validate_evidence, validate_mode_identity, validate_samples, validate_selections, validate_supervision)
from cnsbridge.diagnostics.observability_evidence import validate_mode_artifacts, validate_summary_files
from cnsbridge.diagnostics.observability_protocol import (CONDITIONS, LAMINA, MODES, GIB, canonical_hash,
    implementation_hashes, ledger, masks, read_json, template, validate_protocol, validate_registration)
from cnsbridge.diagnostics.observability_summary import (PAIRS, report_text, summaries_from_files, summarize_arrays, write_summaries)
from cnsbridge.learning.resource_guard import write_json


@pytest.fixture(scope="module")
def numerical_records(tmp_path_factory):
    """199-cell explicit software circuit, with disconnected DN fixtures."""
    directory = tmp_path_factory.mktemp("numerical_observation_only")
    types = np.array(["R1-R6"] * 192 + list(LAMINA) + ["fixture_dn_a", "fixture_dn_b"])
    # Inhibition into one lamina cell; this is not a biological reference graph.
    counts = sparse.csc_matrix((np.array([-1], np.int64), ([192], [0])), shape=(199, 199))
    graph = numerical_test_graph(counts, types=types)
    annotations = {"type": types, "superclass": np.array(["fixture"] * 197 + ["descending_neuron"] * 2),
                   "rootSide": np.array(["L"] * 199), "somaSide": np.array(["L"] * 198 + ["R"])}
    policy = numerical_test_policy_map(graph, annotations, {name: np.zeros(199, bool) for name in annotations})
    spatial = build_spatial_input_map(policy)
    protocol = template()
    protocol["windows_ms"] = {"precondition": 45, "stimulus": 8, "withdrawal": 4}
    selected = selections(graph, policy)
    records = {}
    progress = []
    torch.set_num_threads(1)
    for mode in MODES:
        model = (TrainableLIF(graph, policy.inputs.body_ids) if mode == MODES[0]
                 else LaminaTonicLIF(graph, policy.inputs.body_ids, mode="fixed"))
        binding = {"registration_id": "a" * 64, "protocol_id": canonical_hash(protocol),
                   "model_id": model.model_id, "role": "numerical_test_only"}
        records[mode] = collect_mode(model, policy, spatial, protocol, directory / mode, binding,
                                    lambda *values: progress.append(values))
    write_summaries(directory, protocol)
    return SimpleNamespace(directory=directory, graph=graph, policy=policy, spatial=spatial, protocol=protocol,
                           selected=selected, records=records, progress=progress)


def phases(fixture, condition):
    steps = ledger(fixture.protocol)["phase_steps"]
    stimulus = masks()
    return ([(stimulus["blank"], steps["precondition"])] if condition == "precondition"
            else [(stimulus[condition], steps["stimulus"]), (stimulus["blank"], steps["withdrawal"])])


def check_samples(fixture, mode, condition, values):
    record = fixture.records[mode]
    ni = len(fixture.selected["input_indices"])
    if condition == "precondition":
        initial = {"v": np.full(199, -52, np.float32), "g": np.zeros(199, np.float32),
                   "refractory_until": np.zeros(199, np.int64), "pending": np.zeros((18, 199), np.float32),
                   "step": np.array(0, np.int64), "model_id": np.array(record["model"]["model_id"])}
        phase = np.random.Generator(np.random.PCG64(1702)).random(ni)
        counts, step, state = np.zeros(ni, np.int64), 0, state_digest(initial)
    else:
        warm = arrays(fixture.directory / mode / "precondition.npz", SAMPLE_KEYS | BINDINGS)
        phase, counts, step, state = warm["drive_phase_end"], warm["drive_counts_end"], 450, record["base_state_sha256"]
    binding = {"registration_id": "a" * 64, "protocol_id": canonical_hash(fixture.protocol),
               "model_id": record["model"]["model_id"], "role": "numerical_test_only"}
    validate_samples(values, fixture.selected, fixture.spatial.identity, fixture.protocol, phases(fixture, condition),
                     binding, phase, counts, step, state)


@pytest.mark.parametrize("field,value", [("dt_ms", 1), ("theta", 1), ("dtype", "torch.float64"),
                                         ("modes", ["zero_background"]), ("torch_threads", True)])
def test_protocol_rejects_unregistered_variations(field, value):
    p = template()
    p[field] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


@pytest.mark.parametrize("section,key,value", [("windows_ms", "stimulus", 0), ("windows_ms", "withdrawal", 1.5),
    ("windows_ms", "precondition", float("nan")), ("windows_ms", "precondition", 1001),
    ("resources", "wall_seconds", 601), ("resources", "reserve_bytes", 3 * GIB),
    ("resources", "rss_bytes", 7 * GIB), ("resources", "poll_seconds", 0.2)])
def test_windows_and_budget_only_permit_registered_tightening(section, key, value):
    p = template()
    p[section][key] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_fixed_masks_and_step_ledger():
    p = validate_protocol(template())
    assert ledger(p)["total_steps"] == 16000
    values = masks()
    assert tuple(values) == CONDITIONS
    assert [int(v.sum()) for v in values.values()] == [0, 4, 4, 4, 4, 4, 16]
    assert np.array_equal(np.argwhere(values["left"]), [[5, 3], [5, 4], [6, 3], [6, 4]])
    p["resources"]["rss_bytes"] = GIB
    p["resources"]["wall_seconds"] = 30
    p["resources"]["reserve_bytes"] = 5 * GIB
    assert validate_protocol(p) == p


def test_registered_preflight_does_not_launch_or_overwrite(tmp_path, monkeypatch):
    protocol = tmp_path / "candidate.json"
    write_json(protocol, template())
    monkeypatch.setattr(cli.psutil, "virtual_memory", lambda: SimpleNamespace(available=3 * GIB))
    monkeypatch.setattr(cli, "supervise", lambda *a, **kw: pytest.fail("Must not start a worker"))
    destination = tmp_path / "run"
    result = cli.run_registered(tmp_path / "not_loaded.npz", protocol, destination)
    assert result["status"] == "incomplete" and result["reason"] == "resource_preflight_failed"
    registration = read_json(destination / "registration.json")
    assert validate_registration(registration) == template()
    assert read_json(destination / "supervisor.json")["worker_status"] == "not_launched"
    with pytest.raises(FileExistsError):
        cli.register(tmp_path / "not_loaded.npz", protocol, destination)


def test_registration_rejects_source_changes(tmp_path, monkeypatch):
    protocol = tmp_path / "candidate.json"
    write_json(protocol, template())
    registration = cli.register(tmp_path / "not_loaded.npz", protocol, tmp_path / "run")
    registration["implementation_sha256"]["brain/connectome.py"] = "f" * 64
    registration["registration_id"] = canonical_hash({k: v for k, v in registration.items() if k != "registration_id"})
    with pytest.raises(ValueError, match="source changed"):
        validate_registration(registration)


def test_parent_and_validator_import_without_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import cnsbridge.diagnostics.observability; "
                             "assert 'torch' not in sys.modules; assert 'cnsbridge.brain.trainable_graph' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_template_cli_is_nonexecuting_and_exclusive(tmp_path):
    path = tmp_path / "candidate.json"
    assert cli.main(["--write-template", str(path)]) == 0
    assert read_json(path) == template()
    assert cli.main(["--write-template", str(path)]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--write-template", str(path), "--out", str(tmp_path / "no")])


def test_numerical_role_never_becomes_production(numerical_records):
    f = numerical_records
    validate_selections(f.selected, f.graph.identity, f.policy.identity, f.spatial.identity, role="numerical_test_only")
    with pytest.raises(ValueError, match="role mismatch"):
        validate_selections(f.selected, f.graph.identity, f.policy.identity, f.spatial.identity)
    for mode in MODES:
        validate_mode_identity(f.records[mode], mode, f.graph.identity, f.policy.identity, f.spatial.identity,
                               f.selected, implementation_hashes())


def test_paired_collection_matches_independent_events_and_analytic_warmup(numerical_records):
    f = numerical_records
    for mode in MODES:
        for condition in ("precondition", *CONDITIONS):
            sample = arrays(f.directory / mode / f"{condition}.npz", SAMPLE_KEYS | BINDINGS)
            check_samples(f, mode, condition, sample)
            assert np.all(sample["dn_spikes"] == 0)
            assert np.all(sample["dn_v_mv"] == -52)
            assert np.all(sample["dn_g_mv"] == 0)
        base = arrays(f.directory / mode / "base_state.npz", STATE_KEYS)
        assert state_digest(base) == f.records[mode]["base_state_sha256"]
        warm = arrays(f.directory / mode / "precondition.npz", SAMPLE_KEYS | BINDINGS)
        expected_spikes = 0 if mode == MODES[0] else 1  # first strict threshold crossing around 41.6 ms
        assert np.array_equal(warm["node_phase_spikes"][0, 192:197], [expected_spikes] * 5)
        assert np.count_nonzero(warm["input_events"]) == 0
        if mode == MODES[1]:
            assert np.flatnonzero(warm["lamina_spikes"][:, 0]).tolist() == [41]
    assert sum(item[-1] for item in f.progress) == ledger(f.protocol)["total_steps"]
    # Both dynamics receive identical phase/event clocks, independent of background.
    for condition in CONDITIONS:
        a = arrays(f.directory / MODES[0] / f"{condition}.npz", SAMPLE_KEYS | BINDINGS)
        b = arrays(f.directory / MODES[1] / f"{condition}.npz", SAMPLE_KEYS | BINDINGS)
        assert np.array_equal(a["input_phase_events"], b["input_phase_events"])
        assert np.array_equal(a["drive_phase_end"], b["drive_phase_end"])
        assert np.all(a["input_events"][8:] == 0)  # withdrawal has no input, phase does not reset


@pytest.mark.parametrize("key,mutation", [
    ("start_state_sha256", lambda x: np.array("f" * 64)),
    ("drive_phase_start", lambda x: x + 0.001),
    ("drive_step_start", lambda x: x + 1),
    ("input_events", lambda x: x + 1),
    ("input_phase_end_phase", lambda x: x + 0.001),
    ("dn_spikes", lambda x: x + 1),
    ("node_phase_spikes", lambda x: x + 1),
    ("time_ms", lambda x: x + 1),
    ("dn_v_mv", lambda x: x.astype(np.float32)),
    ("dn_g_mv", lambda x: x + float("nan")),
    ("phase_end_steps", lambda x: x + 1),
])
def test_corrupt_sample_evidence_rejected(numerical_records, key, mutation):
    f = numerical_records
    values = arrays(f.directory / MODES[0] / "center.npz", SAMPLE_KEYS | BINDINGS)
    values[key] = mutation(values[key])
    with pytest.raises(ValueError):
        check_samples(f, MODES[0], "center", values)


def test_complete_zero_response_is_not_a_failed_observation(numerical_records):
    f = numerical_records
    expected, differences = summaries_from_files(f.directory, f.protocol)
    assert read_json(f.directory / "summary.json") == expected
    assert len(differences) == len(MODES) * len(PAIRS) * 3
    for mode in MODES:
        assert expected["modes"][mode]["response_observation"] == {"dn_spikes": False, "dn_v_mv": False, "dn_g_mv": False}
        assert expected["modes"][mode]["inputs"]["large"]["input_event_count"] > 0
    assert (f.directory / "report.md").read_text() == report_text(expected, f.protocol)
    assert all(np.all(value == 0) for value in differences.values())


def test_summary_retains_both_signed_directions_and_raw_subthreshold_changes(numerical_records):
    f = numerical_records
    branches = {c: arrays(f.directory / MODES[0] / f"{c}.npz", SAMPLE_KEYS | BINDINGS) for c in CONDITIONS}
    branches["left"]["dn_v_mv"][0] += np.array([0.1, -0.1])
    branches["right"]["dn_v_mv"][0, 0] += 1e-7
    summary, delta = summarize_arrays(branches, f.protocol)
    left = summary["pairs"]["left_minus_blank"]["dn_v_mv"]
    right = summary["pairs"]["right_minus_blank"]["dn_v_mv"]
    assert left["increased_cells"] == left["decreased_cells"] == 1
    assert right["raw_changed_cells"] == 1 and right["above_tolerance_cells"] == 0
    assert delta["left_minus_right__dn_v_mv"][0, 1] < 0


@pytest.mark.parametrize("status,reason", [("completed", "worker_exited"), ("failed", "wall_limit"),
                                          ("failed", "rss_limit"), ("failed", "system_reserve")])
def test_clean_exit_or_resource_failure_without_evidence_cannot_complete(tmp_path, status, reason):
    outcome = {"schema": "cnsbridge.resource_run.v1", "status": status, "reason": reason, "returncode": 0}
    result = cli.finalize_run(tmp_path, outcome)
    assert result["status"] == "incomplete"
    assert read_json(tmp_path / "supervisor.json") == outcome
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        validate_evidence(tmp_path)


def test_supervision_checks_original_samples_and_terminal_state(tmp_path):
    registration = {"budget": {"can_launch": True, "effective_rss_limit_bytes": GIB},
                    "protocol": template(), "ledger": {"total_steps": 16000}}
    worker = {"wall_seconds": 1.0}
    phase = {"phase": "finished", "completed_steps": 16000, "wall_seconds": 1.0}
    supervisor = {"schema": "cnsbridge.resource_run.v1", "status": "completed", "reason": "worker_exited", "returncode": 0,
                  "worker_status": "collected", "wall_seconds": 1.2, "peak_observed_rss_bytes": 100, "last_phase": phase}
    write_json(tmp_path / "supervisor.json", supervisor)
    write_json(tmp_path / "phase.json", phase)
    (tmp_path / "resources.csv").write_text(f"wall_seconds,tree_rss_bytes,system_available_bytes\n0.2,100,{5 * GIB}\n1.1,0,{5 * GIB}\n")
    assert validate_supervision(tmp_path, registration, worker) == supervisor
    supervisor["peak_observed_rss_bytes"] = 99
    write_json(tmp_path / "supervisor.json", supervisor)
    with pytest.raises(ValueError, match="Peak RSS"):
        validate_supervision(tmp_path, registration, worker)


def test_completed_branch_hash_survives_later_interruption(numerical_records, tmp_path):
    from cnsbridge.diagnostics.observability_protocol import file_hash
    from cnsbridge.diagnostics.observability_evidence import _manifest, MODE_FILES
    f = numerical_records
    for mode in MODES:
        folder = f.directory / mode
        _manifest(folder, f.records[mode]["artifacts_sha256"], MODE_FILES)
        durable = read_json(folder / "artifacts.json")
        assert durable == {key: value for key, value in f.records[mode]["artifacts_sha256"].items() if key != "artifacts.json"}
    model = TrainableLIF(f.graph, f.policy.inputs.body_ids)
    binding = {"registration_id": "a" * 64, "protocol_id": canonical_hash(f.protocol),
               "model_id": model.model_id, "role": "numerical_test_only"}

    def interrupt_after_first_branch(condition, *unused):
        if condition == "center":
            raise RuntimeError("explicit numerical fixture interruption")

    folder = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="fixture interruption"):
        collect_mode(model, f.policy, f.spatial, f.protocol, folder, binding, interrupt_after_first_branch)
    durable = read_json(folder / "artifacts.json")
    assert set(durable) == {"identity.json", "precondition.npz", "base_state.npz", "blank.npz"}
    assert all(file_hash(folder / name) == digest for name, digest in durable.items())
    persisted_identity = read_json(folder / "identity.json")
    assert persisted_identity["model"] == model.identity
    assert persisted_identity["run"] == binding
    assert persisted_identity["drive"]["spatial_map_id"] == f.spatial.map_id
    assert not (folder / "center.npz").exists()
    with pytest.raises(ValueError, match="Incomplete manifest"):
        _manifest(folder, durable, MODE_FILES)


def test_complete_numerical_artifact_contract_and_zero_summary(numerical_records):
    f = numerical_records
    for mode in MODES:
        validate_mode_artifacts(f.directory / mode, f.records[mode], mode, f.selected, f.graph.identity, f.policy.identity,
                                f.spatial.identity, f.protocol, {"registration_id": "a" * 64,
                                "protocol_id": canonical_hash(f.protocol)}, implementation_hashes(), 10.0)
    assert validate_summary_files(f.directory, f.protocol) == read_json(f.directory / "summary.json")


@pytest.mark.parametrize("damage", ["missing_branch", "base_digest", "parameters", "identity"])
def test_invalid_persisted_mode_cannot_validate(numerical_records, tmp_path, damage):
    from cnsbridge.diagnostics.observability_protocol import file_hash
    f, mode = numerical_records, MODES[0]
    folder = tmp_path / mode
    shutil.copytree(f.directory / mode, folder)
    record = deepcopy(f.records[mode])
    if damage == "missing_branch":
        (folder / "center.npz").unlink()
    elif damage == "base_digest":
        record["base_state_sha256"] = "f" * 64
    elif damage == "parameters":
        values = arrays(folder / "parameters.npz", {"theta_before", "theta_after", "rho_before", "rho_after"})
        values["theta_after"][0] = 0.1
        np.savez(folder / "parameters.npz", **values)
        record["artifacts_sha256"]["parameters.npz"] = file_hash(folder / "parameters.npz")
        ledger_value = read_json(folder / "artifacts.json")
        ledger_value["parameters.npz"] = record["artifacts_sha256"]["parameters.npz"]
        write_json(folder / "artifacts.json", ledger_value)
        record["artifacts_sha256"]["artifacts.json"] = file_hash(folder / "artifacts.json")
    else:
        record["model"]["model_id"] = "f" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_mode_artifacts(folder, record, mode, f.selected, f.graph.identity, f.policy.identity, f.spatial.identity,
                                f.protocol, {"registration_id": "a" * 64, "protocol_id": canonical_hash(f.protocol)},
                                implementation_hashes(), 10.0)


@pytest.mark.parametrize("damage", ["summary", "differences", "report"])
def test_saved_summary_is_recomputed_not_trusted(numerical_records, tmp_path, damage):
    f = numerical_records
    folder = tmp_path / "evidence_fixture"
    shutil.copytree(f.directory, folder)
    if damage == "summary":
        value = read_json(folder / "summary.json")
        value["modes"][MODES[0]]["response_observation"]["dn_spikes"] = True
        write_json(folder / "summary.json", value)
    elif damage == "differences":
        with np.load(folder / "differences.npz", allow_pickle=False) as data:
            values = {name: data[name] for name in data.files}
        values[next(iter(values))][0, 0] += 1
        np.savez(folder / "differences.npz", **values)
    else:
        (folder / "report.md").write_text("Unsupported claimed success")
    with pytest.raises(ValueError):
        validate_summary_files(folder, f.protocol)
