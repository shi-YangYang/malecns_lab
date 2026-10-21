"""Local API/process boundary tests; fake subprocesses never claim brain evidence."""

import http.client
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
import pytest

from malecns_lab.diagnostics.sources_v2 import GeneratedSourceV2, STIMULUS_KINDS
from malecns_lab.visualization.jobs import GIB, JobError, JobManager
from malecns_lab.visualization.server import AtlasHTTPServer, Handler
from malecns_lab.visualization.stimulus import AtlasStimulus, DEFAULT_PARAMETERS, validate_parameters
from malecns_lab.visualization.storage import DISPLAY_ASSETS, EVIDENCE_FILES, read_json, write_json


@pytest.mark.parametrize("stimulus", STIMULUS_KINDS)
def test_default_stimulus_pixels_equal_existing_generator(stimulus):
    old = GeneratedSourceV2(stimulus)
    new = AtlasStimulus({**DEFAULT_PARAMETERS, "stimulus": stimulus})
    assert all(np.array_equal(old.frame(i), new.frame(i)) for i in range(50))
    assert new.metadata["texture_seed"] == 7


def test_stimulus_controls_change_pixels_not_warmup_motion():
    normal = AtlasStimulus(DEFAULT_PARAMETERS)
    faster = AtlasStimulus({**DEFAULT_PARAMETERS, "speed": 2})
    lower = AtlasStimulus({**DEFAULT_PARAMETERS, "contrast": 0.5})
    assert np.array_equal(normal.frame(39), faster.frame(39))
    assert not np.array_equal(normal.frame(49), faster.frame(49))
    assert lower.frame(49).std() < normal.frame(49).std()
    assert normal.metadata["source_id"] != faster.metadata["source_id"]


@pytest.mark.parametrize("patch", [{"speed": True}, {"speed": float("nan")}, {"contrast": float("inf")},
                                    {"speed": 0.2}, {"contrast": 1.1}, {"duration_s": 3.05},
                                    {"duration_s": 0.12}, {"duration_s": "0.5"}, {"stimulus": "unknown"},
                                    {"graph": "other.npz"}, {"stimulus": []}])
def test_invalid_parameters_rejected(patch):
    with pytest.raises(ValueError):
        validate_parameters({**DEFAULT_PARAMETERS, **patch})


# Only used by this isolated process-ownership test suite. Production cannot
# select this command and actual full-reference verification is separate.
FAKE_PROCESS = '''
import json, pathlib, sys, time
d, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
request = json.loads((d / "request.json").read_text())
n = 40 + round(request["parameters"]["duration_s"] * 20)
time.sleep(0.25)
(d / "progress.json").write_text(json.dumps({"status":"running", "frames_completed":1, "simulation_s":0.05}))
if mode == "wait":
    time.sleep(30)
elif mode == "fail":
    (d / "failure.json").write_text(json.dumps({"error":"视觉输入未通过检查。"}))
    sys.exit(2)
elif mode in ("success", "incomplete"):
    names = ["catalog.json", "atlas.json", "run.json", "stimulus.png", "manifest.json", "records.npz", "inputs.npz", "neuron_totals.npz"]
    for name in names:
        (d / name).write_bytes(b"test-only-placeholder")
    (d / "result.json").write_text(json.dumps({"status":"completed", "frames_completed":n,
        "total_frames":n,"neural_steps":n*100,"simulation_s":n/20,
        "coverage":{"valid_input_frames":n if mode=="success" else n-1},
        "artifacts":{name:"test-only-placeholder" for name in names}}))
'''


@pytest.fixture
def manager_factory(tmp_path):
    managers = []
    graph, annotations = tmp_path / "graph.npz", tmp_path / "annotations.feather"
    graph.write_bytes(b"test-only-placeholder")
    annotations.write_bytes(b"test-only-placeholder")

    def make(mode="wait", **kwargs):
        root = kwargs.pop("root", tmp_path / str(uuid.uuid4()))
        manager = JobManager(root, graph=graph, annotations=annotations,
                             available_memory=kwargs.pop("available_memory", lambda: 8 * GIB),
                             command_factory=lambda path: [sys.executable, "-c", FAKE_PROCESS, str(path), mode], **kwargs)
        managers.append(manager)
        return manager
    yield make
    for manager in managers:
        manager.close()


def wait_for(manager, job_id, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if predicate(job):
            return job
        time.sleep(0.02)
    raise AssertionError(manager.get(job_id))


def test_retired_experiments_stay_unmodified_and_outside_current_api(manager_factory, tmp_path):
    root = tmp_path / "mixed-history"
    directory = root / str(uuid.uuid4())
    directory.mkdir(parents=True)
    parameters = {"experiment": "yaw_stabilization", "direction": "left",
                  "disturbance_rad_s": 0.25, "duration_s": 2.0}
    write_json(directory / "job.json", {"id": directory.name, "status": "running", "parameters": parameters})
    original = (directory / "job.json").read_bytes()

    manager = manager_factory(root=root)
    assert manager.list() == []
    with pytest.raises(JobError) as missing:
        manager.get(directory.name)
    assert missing.value.status == 404
    with pytest.raises(JobError) as rejected:
        manager.create(parameters)
    assert rejected.value.status == 400
    assert (directory / "job.json").read_bytes() == original


def test_progress_concurrency_cancel_and_owned_process_cleanup(manager_factory):
    manager = manager_factory()
    job = manager.create(DEFAULT_PARAMETERS)
    with pytest.raises(JobError) as error:
        manager.create(DEFAULT_PARAMETERS)
    assert error.value.status == 409
    observed = wait_for(manager, job["id"], lambda item: item["frames_completed"] == 1)
    assert observed["simulation_s"] == 0.05 and observed["elapsed_s"] > 0
    assert manager.cancel(job["id"])["status"] == "cancelled"
    assert not psutil.pid_exists(job["worker_pid"])
    assert manager.health()["active_job_id"] is None
    assert read_json(manager.root / job["id"] / "job.json")["catalog_url"] is None
    assert manager.cancel(job["id"])["status"] == "cancelled"


@pytest.mark.parametrize("mode,expected", [("success", "completed"), ("incomplete", "failed"), ("fail", "failed")])
def test_worker_exit_requires_complete_evidence(manager_factory, mode, expected):
    manager = manager_factory(mode)
    created = manager.create(DEFAULT_PARAMETERS)
    job = wait_for(manager, created["id"], lambda item: item["status"] in ("completed", "failed"))
    assert job["status"] == expected
    assert (job["catalog_url"] is not None) == (expected == "completed")
    if expected == "completed":
        assert manager.asset(job["id"], "catalog.json").is_file()
        with pytest.raises(JobError):
            manager.asset(job["id"], "records.npz")
        (manager.root / job["id"] / "run.json").unlink()
        (manager.root / job["id"] / "run.json").symlink_to(manager.graph)
        with pytest.raises(JobError):
            manager.asset(job["id"], "run.json")
    else:
        with pytest.raises(JobError):
            manager.asset(job["id"], "catalog.json")


def test_memory_preflight_and_monitor_cover_load(manager_factory):
    memory = [GIB]
    manager = manager_factory(available_memory=lambda: memory[0])
    with pytest.raises(JobError) as error:
        manager.create(DEFAULT_PARAMETERS)
    assert error.value.status == 503
    assert manager.list() == []
    memory[0] = 8 * GIB
    job = manager.create(DEFAULT_PARAMETERS)
    memory[0] = GIB - 1
    result = wait_for(manager, job["id"], lambda item: item["status"] == "failed")
    assert result["frames_completed"] == 0
    assert "内存" in result["error"]
    assert not psutil.pid_exists(job["worker_pid"])


def test_budget_and_service_shutdown_stop_worker(manager_factory):
    manager = manager_factory(budget_s=0.3)
    job = manager.create(DEFAULT_PARAMETERS)
    failed = wait_for(manager, job["id"], lambda item: item["status"] == "failed")
    assert "预算" in failed["error"]
    assert not psutil.pid_exists(job["worker_pid"])
    other = manager_factory()
    job = other.create(DEFAULT_PARAMETERS)
    other.close()
    assert other.get(job["id"])["status"] == "cancelled"
    assert not psutil.pid_exists(job["worker_pid"])


def test_restart_marks_interrupted_and_refuses_live_owner(manager_factory, tmp_path):
    manager = manager_factory()
    job = manager.create(DEFAULT_PARAMETERS)
    with pytest.raises(JobError):
        manager_factory(root=manager.root)
    assert psutil.pid_exists(job["worker_pid"])
    manager.close()
    directory = manager.root / job["id"]
    saved = read_json(directory / "job.json")
    saved.update(status="running", owner_pid=99999999, worker_pid=99999999)
    write_json(directory / "job.json", saved)
    restarted = manager_factory(root=manager.root)
    assert restarted.get(job["id"])["status"] == "failed"
    assert "中断" in restarted.get(job["id"])["error"]


def test_api_local_origin_header_json_and_asset_boundaries(manager_factory):
    manager = manager_factory()
    server = AtlasHTTPServer(("127.0.0.1", 0), Handler)
    server.manager = manager
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.server_address[1]

    def request(path="/api/experiments", method="POST", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.request(method, path, body=json.dumps(DEFAULT_PARAMETERS) if body is None else body,
                           headers={"Content-Type": "application/json", "X-MaleCNS-Request": "1", **(headers or {})})
        response = connection.getresponse()
        status, value = response.status, json.loads(response.read())
        connection.close()
        return status, value
    try:
        assert request(headers={"Origin": "https://other.example"})[0] == 403
        assert request(headers={"Origin": "null"})[0] == 403
        assert request(headers={"Host": "evil.example:8766"})[0] == 403
        assert request(headers={"X-MaleCNS-Request": "0"})[0] == 403
        assert request(body='{"stimulus":"left","stimulus":"right"}')[0] == 400
        assert request(body="[1, 2]")[0] == 400
        assert request(body=" " * 4097)[0] == 413
        assert request(path="/api/experiments/%2e%2e/assets/request.json", method="GET")[0] == 404
        assert request(path="/api/health", method="GET")[1]["pid"] > 0
        status, created = request(headers={"Origin": "http://127.0.0.1:8765"})
        assert status == 202
        assert request(path=f"/api/experiments/{created['id']}/assets/catalog.json", method="GET")[0] == 404
        assert request(path=f"/api/experiments/{created['id']}/cancel")[1]["status"] == "cancelled"
        assert request(path="/api/experiments", method="GET")[1]["experiments"][0]["id"] == created["id"]
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()
