"""Desktop lifecycle failure injection only: no model, Docker, VM or ROS IO."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from malecns_lab.experiments import x2_desktop as desktop


def make_runtime(tmp_path):
    root = tmp_path
    (root / "data").mkdir(exist_ok=True)
    output = root / "data/x2_sim/agibot/data/var/desktop-20261023-120000-1234abcd"
    events = []
    runtime = desktop.Runtime(output, root, desktop.Commands(io.StringIO()), events.append)
    runtime.log_directory = root / "logs"
    runtime.log_directory.mkdir()
    return runtime, events


def stop_evidence(runtime, *, stopped=True, zero=True, run_id=None):
    runtime.output.mkdir(parents=True, exist_ok=True)
    runtime.run_id = "same-run"
    (runtime.output / "bridge_summary.json").write_text(json.dumps({
        "run_id": run_id or runtime.run_id, "final_stopped": stopped, "active_forward": False}))
    (runtime.output / "bridge_events.jsonl").write_text(json.dumps({
        "kind": "tick", "state": "STOPPED" if stopped else "STOPPING",
        "attempts": [{"command": [0., 0., 0.], "ok": zero}]}) + "\n")


def test_import_preview_and_help_do_not_start_resources(tmp_path, monkeypatch, capsys):
    probe = subprocess.run([sys.executable, "-c", "import sys; import malecns_lab.experiments.x2_desktop; "
                            "assert not {'torch','numpy','scipy','rclpy'}.intersection(sys.modules)"],
                           capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected process"))
    assert desktop.main(["--output", str(tmp_path / "untouched")]) == 0
    assert json.loads(capsys.readouterr().out)["simulator_started"] is False
    assert not (tmp_path / "untouched").exists()


def test_fixed_shared_output_and_exclusive_lock(tmp_path):
    (tmp_path / "data").mkdir()
    candidate = tmp_path / "data/x2_sim/agibot/data/var/desktop-20261023-120000-1234abcd"
    assert desktop.output_directory(candidate, tmp_path) == candidate
    for invalid in (tmp_path / "outside", candidate.parent / "../desktop-20261023-120000-1234abcd",
                    candidate.parent / "desktop-20261023-120000-$(bad)"):
        with pytest.raises(ValueError):
            desktop.output_directory(invalid, tmp_path)
    candidate.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        desktop.output_directory(candidate, tmp_path)
    with desktop.desktop_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with desktop.desktop_lock(tmp_path):
                pytest.fail("Concurrent desktop run")


@pytest.mark.parametrize("text,reason", [
    ("", "desktop_disconnected"), ('{"command":"stop"}\n', "user_stop"),
    ('{"command":"shell"}\n', "invalid_desktop_command"),
    ('{"command":"continue","path":"/tmp"}\n', "invalid_desktop_command"),
    ("x" * 1025, "invalid_desktop_command"),
])
def test_input_disconnect_stop_and_invalid_commands(text, reason):
    commands = desktop.Commands(io.StringIO(text))
    commands.read()
    assert commands.stop.is_set() and commands.reason == reason


def test_low_reserve_and_parent_loss_abort_before_launch(tmp_path, monkeypatch):
    import psutil
    runtime, _ = make_runtime(tmp_path)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=desktop.GIB - 1))
    with pytest.raises(RuntimeError, match="1 GiB"):
        runtime.guard()
    monkeypatch.setattr(desktop.os, "getppid", lambda: runtime.parent_pid + 1)
    with pytest.raises(desktop.StopRequested, match="parent_lost"):
        runtime.guard()


def test_running_vm_is_never_adopted_or_stopped(tmp_path, monkeypatch):
    runtime, _ = make_runtime(tmp_path)
    calls = []
    monkeypatch.setattr(runtime, "guard", lambda: None)
    def command(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "lighter is running", "")
    monkeypatch.setattr(runtime, "command", command)
    with pytest.raises(RuntimeError, match="already be stopped"):
        runtime.preflight()
    runtime.cleanup()
    assert len(calls) == 1 and calls[0][-1] == "status"
    assert not runtime.resources_stopped and not runtime.owns_vm


@pytest.mark.parametrize("stopped,zero,run_id,expected", [
    (True, True, "same-run", True), (False, True, "same-run", False),
    (True, False, "same-run", False), (True, True, "another-run", False),
])
def test_stop_requires_this_run_actual_zero_and_stopped(tmp_path, stopped, zero, run_id, expected):
    runtime, _ = make_runtime(tmp_path)
    stop_evidence(runtime, stopped=stopped, zero=zero, run_id=run_id)
    assert runtime.stopped_evidence() is expected


def test_abort_precedes_cleanup_and_only_owned_resources_are_stopped(tmp_path, monkeypatch):
    runtime, events = make_runtime(tmp_path)
    stop_evidence(runtime)
    runtime.bridge_started = runtime.owns_vm = runtime.owns_container = True
    runtime.vm_started = desktop.time.monotonic()
    runtime.bridge = SimpleNamespace(poll=lambda: 1)
    calls = []
    def command(args, **kwargs):
        assert (runtime.output / "host_abort.json").exists()
        assert runtime.stop_confirmed is True
        calls.append(args)
        stdout = "lighter is not running." if args[-1] == "status" else "exited" if "inspect" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout, "")
    monkeypatch.setattr(runtime, "command", command)
    runtime.stop_motion("user_stop")
    runtime.cleanup()
    assert runtime.resources_stopped
    assert calls[0][:3] == runtime.docker and calls[0][3:6] == ["stop", "--time", "2"]
    assert calls[2] == [str(runtime.runtime / "bin/lighter"), "stop"]
    assert all("context" not in call and "restart" not in call for call in calls)
    assert events[-1]["state"] == "stopping"  # No premature terminal success.


def test_shutdown_failure_retains_unconfirmed_resource_state(tmp_path, monkeypatch):
    runtime, _ = make_runtime(tmp_path)
    runtime.owns_vm = runtime.owns_container = True
    def command(args, **kwargs):
        raise TimeoutError("injected shutdown failure")
    monkeypatch.setattr(runtime, "command", command)
    runtime.cleanup()
    assert not runtime.resources_stopped
    assert runtime.owns_vm and runtime.owns_container
    assert len(runtime.errors) == 2


def test_no_bridge_cancellation_does_not_claim_measured_stop(tmp_path):
    runtime, events = make_runtime(tmp_path)
    runtime.output.mkdir(parents=True)
    runtime.run_id = "same-run"
    runtime.stop_motion("user_stop")
    runtime.cleanup()
    assert runtime.stop_confirmed is None and runtime.resources_stopped
    value = json.loads((runtime.output / "host_abort.json").read_text())
    assert value["reason"] == "user_stop" and value["run_id"] == "same-run"
    assert all(event["stop_confirmed"] is None for event in events)


def test_terminal_follows_cleanup_and_session_snapshot_matches(tmp_path, monkeypatch):
    runtime, events = make_runtime(tmp_path)
    runtime.output.mkdir(parents=True)
    runtime.run_id = "same-run"
    def cancel():
        raise desktop.StopRequested("user_stop")
    monkeypatch.setattr(runtime, "preflight", cancel)
    result = runtime.execute(runtime.log_directory)
    assert result["state"] == "cancelled" and result["resources_stopped"]
    assert events[-1]["state"] == "cancelled" and events[-1]["resources_stopped"]
    assert json.loads((runtime.output / "desktop_session.json").read_text()) == events[-1]
    assert events[-2]["state"] == "stopping"


def test_missing_stop_evidence_is_failed_even_if_resources_closed(tmp_path, monkeypatch):
    runtime, events = make_runtime(tmp_path)
    runtime.output.mkdir(parents=True)
    runtime.run_id = "same-run"
    runtime.bridge_started = True
    runtime.bridge = SimpleNamespace(poll=lambda: 1)
    def cancel():
        raise desktop.StopRequested("user_stop")
    monkeypatch.setattr(runtime, "preflight", cancel)
    result = runtime.execute(runtime.log_directory)
    assert result["state"] == "failed" and result["resources_stopped"]
    assert result["stop_confirmed"] is False and events[-1]["state"] == "failed"


def test_motion_flag_requires_actual_publication_not_neural_intent(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    runtime.output.mkdir(parents=True)
    runtime.run_id = "same-run"
    (runtime.output / "decision-001.json").write_text(json.dumps({"run_id": "same-run", "intent": "FORWARD"}))
    runtime.poll_phase()
    assert not runtime.motion_started
    path = runtime.output / "bridge_events.jsonl"
    path.write_text(json.dumps({"kind": "tick", "attempts": [{"command": [.2, 0., 0.], "ok": True}]}) + "\n")
    runtime.poll_phase()
    assert runtime.motion_started


def test_stdout_failure_requests_stop_without_interrupting_cleanup(tmp_path):
    class BrokenPipe:
        def write(self, _):
            raise BrokenPipeError()
    commands = desktop.Commands(io.StringIO())
    events = desktop.Events(BrokenPipe(), io.StringIO(), commands.request_stop)
    events.emit({"state": "preparing"})
    events.drain()
    assert commands.stop.is_set() and commands.reason == "desktop_output_disconnected"


@pytest.mark.parametrize("scenario", ["ready_after_retry", "never_ready", "stop", "timeout", "no_budget"])
def test_official_sd_readiness_keeps_evidence_and_budgets(tmp_path, monkeypatch, scenario):
    runtime, _ = make_runtime(tmp_path)
    clock, children = [0.], []
    runtime.started = runtime.vm_started = 0.
    runtime.host = SimpleNamespace(poll=lambda: None)
    if scenario == "no_budget":
        clock[0] = desktop.VM_SECONDS - desktop.CLEANUP_SECONDS - 1.
    monkeypatch.setattr(desktop.time, "monotonic", lambda: clock[0])

    def guard():
        if runtime.commands.stop.is_set():
            raise desktop.StopRequested(runtime.commands.reason)
        if clock[0] >= runtime.deadline():
            raise TimeoutError("Reserved stopping/cleanup window reached")

    def sleep(duration):
        clock[0] += duration
        if scenario == "stop":
            runtime.commands.request_stop("user_stop")

    class Child:
        def __init__(self, args, **kwargs):
            self.returncode = None
            self.number = len(children) + 1
            children.append(self)

        def communicate(self, timeout):
            if scenario == "timeout" and self.returncode is None:
                clock[0] += timeout
                raise subprocess.TimeoutExpired("fake official SD", timeout)
            clock[0] += .1
            if self.returncode is None:
                self.returncode = 0
            message = ("Robot mode set successfully" if scenario == "ready_after_retry" and self.number == 3
                       else "service available, action not accepted yet")
            return "official stdout", message

    monkeypatch.setattr(runtime, "guard", guard)
    monkeypatch.setattr(desktop.time, "sleep", sleep)
    monkeypatch.setattr(desktop.subprocess, "Popen", Child)
    monkeypatch.setattr(runtime, "_kill_command", lambda child: setattr(child, "returncode", -9))
    if scenario == "ready_after_retry":
        runtime.wait_for_official_sd()
        assert len(children) == 3 and clock[0] >= 3.
    else:
        error = {"never_ready": RuntimeError, "stop": desktop.StopRequested,
                 "timeout": TimeoutError, "no_budget": TimeoutError}[scenario]
        with pytest.raises(error):
            runtime.wait_for_official_sd()
        assert len(children) == {"never_ready": 5, "stop": 1, "timeout": 1, "no_budget": 0}[scenario]
    records = [json.loads(path.read_text()) for path in sorted(runtime.log_directory.glob("sd-attempt-*.json"))]
    assert len(records) == len(children)
    assert all(row["stdout"] == "official stdout" and row["stderr"] for row in records)
    assert all(row["returncode"] == (-9 if scenario == "timeout" else 0) for row in records)
    if scenario == "timeout":
        assert records[0]["error"].startswith("TimeoutError:")
    assert clock[0] < runtime.deadline() and not runtime.simulator_ready
    assert (desktop.WALL_SECONDS, desktop.VM_SECONDS, desktop.CLEANUP_SECONDS) == (240., 120., 30.)
