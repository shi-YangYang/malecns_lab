"""Persistence/parent failure semantics only; these fixtures are not full brains."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from malecns_lab.brain.metadata import canonical_hash, file_hashes  # noqa: E402
from malecns_lab.brain.trainable import TrainableLIF  # noqa: E402
from malecns_lab.learning.checkpoint import load_checkpoint, read_checkpoint, restore_checkpoint, save_checkpoint  # noqa: E402
from malecns_lab.learning.probe import run_registered  # noqa: E402
from malecns_lab.learning.resource_guard import GIB, resource_budget, supervise  # noqa: E402


def fixture_model(dtype=torch.float32):
    return TrainableLIF.for_numerical_test(sparse.csc_matrix(np.array([[0, 0], [3, 0]], dtype=np.int64)),
                                         [0], types=np.array(["input", "readout"]), dtype=dtype)


def rewrite_metadata(directory, mutate):
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text())
    mutate(metadata)
    metadata.pop("checkpoint_id")
    metadata["checkpoint_id"] = canonical_hash(metadata)
    path.write_text(json.dumps(metadata))


def test_checkpoint_exact_roundtrip_and_explicit_test_role(tmp_path):
    model = fixture_model()
    with torch.no_grad():
        model.theta.copy_(torch.tensor([0.01, -0.23, 0.4]))
    expected = model.theta.detach().clone()
    metadata = save_checkpoint(model, tmp_path / "checkpoint")
    assert metadata["model"]["graph"]["role"] == "numerical_test_only"
    with torch.no_grad():
        model.theta.zero_()
    restore_checkpoint(model, tmp_path / "checkpoint")
    torch.testing.assert_close(model.theta, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="full_reference"):
        load_checkpoint(tmp_path / "checkpoint", reference=model.graph)
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path / "checkpoint")


@pytest.mark.parametrize("change", ["bytes", "metadata", "dtype", "input", "grouping", "shape", "nonfinite"])
def test_invalid_checkpoint_never_mutates_parameters(tmp_path, change):
    model = fixture_model()
    directory = tmp_path / "checkpoint"
    save_checkpoint(model, directory)
    if change == "bytes":
        with (directory / "parameters.npz").open("ab") as stream:
            stream.write(b"changed")
    elif change == "metadata":
        path = directory / "checkpoint.json"
        value = json.loads(path.read_text())
        value["schema"] = "other"
        path.write_text(json.dumps(value))
    elif change == "dtype":
        model = fixture_model(torch.float64)
    elif change == "input":
        rewrite_metadata(directory, lambda item: item["model"].update(input_body_ids=[]))
    elif change == "grouping":
        rewrite_metadata(directory, lambda item: item["model"]["graph"]["grouping"].update(mapping_sha256="wrong"))
    else:
        theta = np.zeros(4 if change == "shape" else 3, dtype=np.float32)
        if change == "nonfinite":
            theta[0] = np.nan
        np.savez(directory / "parameters.npz", theta=theta)
        rewrite_metadata(directory, lambda item: item.update(arrays_sha256=file_hashes(directory / "parameters.npz")["sha256"]))
    before = model.theta.detach().clone()
    with pytest.raises(ValueError):
        restore_checkpoint(model, directory)
    torch.testing.assert_close(model.theta, before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["learnable", "fixed"])
def test_tonic_checkpoint_save_and_restore_rejected_before_side_effects(tmp_path, mode):
    from malecns_lab.brain.tonic import LaminaTonicLIF
    model = LaminaTonicLIF.for_numerical_test(sparse.csc_matrix((1, 1), dtype=np.int64),
                                             types=["L1"], tonic_types=("L1",), mode=mode)
    directory = tmp_path / "new_parent" / "rejected"
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="dynamics schema"):
        save_checkpoint(model, directory)
    assert not directory.parent.exists()
    # Even a missing file must not be read before rejecting the incompatible model.
    with pytest.raises(ValueError, match="dynamics schema"):
        restore_checkpoint(model, directory)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("alias", [False, True])
def test_checkpoint_rejects_extra_registered_parameters_before_writing_or_restore(tmp_path, alias):
    model = fixture_model()
    directory = tmp_path / "original"
    save_checkpoint(model, directory)
    model.extra = model.theta if alias else torch.nn.Parameter(torch.ones(1))
    before = model.theta.detach().clone()
    rejected = tmp_path / "new_parent" / "rejected"
    with pytest.raises(ValueError, match="exact parameter set"):
        save_checkpoint(model, rejected)
    assert not rejected.parent.exists()
    with pytest.raises(ValueError, match="exact parameter set"):
        restore_checkpoint(model, directory)
    assert torch.equal(model.theta, before)


@pytest.mark.parametrize("change", ["extra", "missing", "new_schema"])
def test_checkpoint_rejects_unrepresented_metadata_before_reading_graph_or_restore(tmp_path, change):
    model = fixture_model()
    directory = tmp_path / "checkpoint"
    save_checkpoint(model, directory)
    def mutate(metadata):
        if change == "extra":
            metadata["parameters"]["rho"] = {"shape": [5], "dtype": "<f4"}
        elif change == "missing":
            metadata["parameters"] = {}
        else:
            metadata["model"]["schema"] = "cnsbridge.lamina_tonic_lif.v1"
    rewrite_metadata(directory, mutate)
    before = model.theta.detach().clone()
    with pytest.raises(ValueError, match="theta|dynamics schema"):
        read_checkpoint(directory)
    with pytest.raises(ValueError, match="theta|dynamics schema"):
        restore_checkpoint(model, directory)
    with pytest.raises(ValueError, match="theta|dynamics schema"):
        load_checkpoint(directory, reference=tmp_path / "graph_must_not_be_read.npz")
    assert torch.equal(model.theta, before)


def test_resource_budget_reserves_system_memory():
    assert resource_budget(12 * GIB)["effective_rss_limit_bytes"] == 6 * GIB
    assert resource_budget(4 * GIB)["effective_rss_limit_bytes"] == 3 * GIB
    assert resource_budget(GIB + 1)["effective_rss_limit_bytes"] == 1
    assert not resource_budget(GIB)["can_launch"]
    assert not resource_budget(GIB - 1)["can_launch"]


@pytest.mark.parametrize("available", [GIB - 1, GIB])
def test_preflight_failure_is_registered_and_does_not_launch(tmp_path, monkeypatch, available):
    import malecns_lab.learning.probe as probe
    monkeypatch.setattr(probe.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    monkeypatch.setattr(probe, "supervise", lambda *args, **kwargs: pytest.fail("Child must not launch"))
    directory = tmp_path / "failed"
    result = run_registered(Path("not-even-loaded.npz"), directory)
    assert result["reason"] == "resource_preflight_failed"
    assert "1 GiB system reserve" in result["detail"]
    assert json.loads((directory / "registration.json").read_text())["protocol"]["steps"] == 200
    assert json.loads((directory / "result.json").read_text())["worker_status"] == "not_launched"
    with pytest.raises(FileExistsError):
        run_registered(Path("not-even-loaded.npz"), directory)


@pytest.mark.parametrize("available,expected_limit", [(GIB + 1, 1), (4 * GIB, 3 * GIB), (8 * GIB, 6 * GIB)])
def test_probe_supervision_uses_registered_budget(tmp_path, monkeypatch, available, expected_limit):
    import malecns_lab.learning.probe as probe
    from malecns_lab.learning.probe_evidence import validate_registration
    monkeypatch.setattr(probe.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    calls = []

    def stop_without_loading(command, directory, **budget):
        registration = json.loads((directory / "registration.json").read_text())
        validate_registration(registration, probe.PROTOCOL)
        assert budget == {"rss_limit_bytes": expected_limit, "wall_seconds": 600,
                          "reserve_bytes": GIB, "poll_seconds": 0.1}
        assert budget["reserve_bytes"] == registration["resources"]["reserve_bytes"]
        calls.append(command)
        return {"status": "failed", "reason": "rss_limit", "returncode": -15}

    monkeypatch.setattr(probe, "supervise", stop_without_loading)
    result = run_registered(tmp_path / "not_loaded.npz", tmp_path / "registered")
    assert len(calls) == 1
    assert result == {"status": "failed", "reason": "rss_limit", "returncode": -15}


@pytest.mark.parametrize("mode,expected", [("status_only", "completed"), ("checkpoint_only", "completed"),
                                          ("nonzero_exit", "failed"), ("malformed", "completed")])
def test_supervisor_completion_never_claims_numeric_success(tmp_path, mode, expected):
    directory = tmp_path / mode
    directory.mkdir()
    report = directory / "worker_result.json"
    code = "from pathlib import Path; import sys; "
    if mode in ("status_only", "nonzero_exit"):
        code += f"Path({str(report)!r}).write_text('{{\"status\": \"passed\"}}'); "
    elif mode == "malformed":
        code += f"Path({str(report)!r}).write_text('broken JSON'); "
    else:
        code += f"Path({str(directory / 'checkpoint')!r}).mkdir(); "
    code += "sys.exit(7)" if mode == "nonzero_exit" else "sys.exit(0)"
    result = supervise([sys.executable, "-c", code], directory, rss_limit_bytes=GIB,
                       reserve_bytes=0, wall_seconds=10, poll_seconds=0.02)
    assert result["status"] == expected
    assert result["status"] != "passed"
    assert result["returncode"] == (7 if mode == "nonzero_exit" else 0)
    assert json.loads((directory / "result.json").read_text()) == result


def test_timeout_terminates_child_and_retains_failure(tmp_path):
    result = supervise([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                       rss_limit_bytes=GIB, reserve_bytes=0, wall_seconds=0.1, poll_seconds=0.02)
    assert result["status"] == "failed" and result["reason"] == "wall_limit"
    assert result["returncode"] != 0
    assert result["wall_seconds"] < 5


@pytest.mark.parametrize("available,expected", [(GIB, "completed"), (GIB - 1, "failed")])
def test_supervisor_default_reserve_enforces_running_boundary(tmp_path, monkeypatch, available, expected):
    import malecns_lab.learning.resource_guard as guard
    monkeypatch.setattr(guard.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))
    result = supervise([sys.executable, "-c", "pass"], tmp_path, rss_limit_bytes=GIB,
                       wall_seconds=10, poll_seconds=0.02)
    assert result["status"] == expected
    assert result["reason"] == ("worker_exited" if expected == "completed" else "system_reserve")
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_rss_stop_is_distinct_from_success(tmp_path):
    result = supervise([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                       rss_limit_bytes=1, reserve_bytes=0, wall_seconds=10, poll_seconds=0.02)
    assert result["status"] == "failed" and result["reason"] == "rss_limit"
    assert result["peak_observed_rss_bytes"] > 1
