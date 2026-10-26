"""Bounded recovery for observed cold startup and delayed VM shutdown."""
import io
import json
import subprocess

import pytest

from malecns_lab.runtime.x2 import skills
from malecns_lab.runtime.x2.desktop import Commands, StopRequested


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 100.
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            self.now += seconds
    value = Clock()
    monkeypatch.setattr(skills.time, "monotonic", value.monotonic)
    monkeypatch.setattr(skills.time, "sleep", value.sleep)
    return value


@pytest.mark.parametrize("remaining,expected", [(60., 25.), (10., 10.)])
def test_sd_uses_one_request_within_remaining_budget(tmp_path, clock, remaining, expected):
    runtime = skills.SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda event: None)
    runtime.deadline = lambda cleanup=False: clock.now + remaining
    calls = []
    def command(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Robot mode set successfully", "")
    runtime.command = command
    runtime.wait_for_sd()
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == expected
    assert f"{expected-2.:.3f}s" in calls[0][0][-1]


def test_sd_timeout_is_not_resubmitted_and_cancel_prevents_request(tmp_path, clock):
    runtime = skills.SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda event: None)
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        raise TimeoutError("sd-01.json")
    runtime.command = command
    with pytest.raises(TimeoutError, match="sd-01.json"):
        runtime.wait_for_sd()
    assert len(calls) == 1
    runtime.commands.request_stop("user_stop")
    with pytest.raises(StopRequested):
        runtime.wait_for_sd()
    assert len(calls) == 1


def test_sd_explicit_failed_response_can_retry_within_original_budget(tmp_path, clock):
    runtime = skills.SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda event: None)
    runtime.deadline = lambda cleanup=False: 160.
    calls = []
    def command(args, **kwargs):
        calls.append(kwargs)
        clock.sleep(2.)
        output = "Failed (code=0): " if len(calls) == 1 else "Robot mode set successfully"
        return subprocess.CompletedProcess(args, 0, "", output)
    runtime.command = command
    runtime.wait_for_sd()
    assert [call["record"] for call in calls] == ["sd-01.json", "sd-02.json"]
    assert [call["timeout"] for call in calls] == [25., 21.5]
    assert clock.now == 105.5


@pytest.mark.parametrize("stops,socket_stays", [(True, False), (False, False), (True, True)])
def test_vm_shutdown_observation_is_bounded_and_does_not_resend_stop(tmp_path, clock, stops, socket_stays):
    calls = []
    class Runtime(skills.SkillRuntime):
        def command(self, args, **kwargs):
            calls.append(args[-1])
            if args[-1] == "stop":
                clock.sleep(6.)
                raise TimeoutError("stop client timed out")
            assert args[-1] == "status"
            state = "lighter is not running." if stops and clock.now >= 108. else "running"
            return subprocess.CompletedProcess(args, 0, state, "")
        def stop_host(self):
            calls.append("host_cleanup")
    runtime = Runtime(tmp_path, tmp_path, Commands(io.StringIO()), lambda event: None)
    runtime.deadline = lambda cleanup=False: 123.
    runtime.owns_vm = True
    if socket_stays:
        runtime.runtime.mkdir(parents=True)
        (runtime.runtime/"docker.sock").touch()
    runtime.cleanup()
    assert calls.count("stop") == 1 and calls[-1] == "host_cleanup"
    assert clock.now <= 115.  # Keep eight seconds for child cleanup.
    confirmed = stops and not socket_stays
    assert runtime.resources_stopped is confirmed
    assert runtime.owns_vm is not confirmed
    receipt = json.loads((tmp_path/"vm-shutdown.json").read_text())
    assert receipt["confirmed"] is confirmed
    assert "stop client timed out" in receipt["stop_command_error"]
    assert bool(runtime.errors) is not confirmed
