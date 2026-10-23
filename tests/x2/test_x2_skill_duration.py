"""CLI-to-executor duration propagation with inert ROS/host transport doubles."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from malecns_lab.control import x2_skills as control
from malecns_lab.runtime.x2 import skills as host
from malecns_lab.experiments import x2_skills as probe


def test_bounded_duration_reaches_executor_and_all_run_records(tmp_path, monkeypatch, capsys):
    cores, published = [], []
    original_executor = control.SkillExecutor

    def executor(*args, **kwargs):
        core = original_executor(*args, **kwargs)
        cores.append(core)
        return core

    # Reject input-source registration immediately after configuration. This
    # exercises the real runner construction/recording without any movement,
    # timer waits, services, subprocesses, or installed ROS dependency.
    class Request:
        def __init__(self):
            self.request = NS(header=NS(stamp=None))
            self.action = NS(value=None)
            self.input_source = NS(name=None, priority=None, timeout=None)

    class Node:
        def __init__(self, name):
            pass

        def create_publisher(self, *args):
            return NS(publish=published.append)

        def create_client(self, *args):
            return NS(service_is_ready=lambda: True,
                      call_async=lambda request: NS(done=lambda: True,
                          result=lambda: NS(response=NS(header=NS(code=1)))))

        def create_subscription(self, *args):
            pass

        def get_clock(self):
            return NS(now=lambda: NS(to_msg=lambda: None))

        def destroy_node(self):
            pass

    inert = {
        "rclpy": NS(init=lambda **kw: None, ok=lambda: True, shutdown=lambda: None),
        "rclpy.node": NS(Node=Node),
        "rclpy.qos": NS(QoSProfile=lambda **kw: None,
                        ReliabilityPolicy=NS(BEST_EFFORT=1), DurabilityPolicy=NS(VOLATILE=1)),
        "rclpy.signals": NS(SignalHandlerOptions=NS(NO=0)),
        "rosidl_runtime_py.convert": NS(message_to_ordereddict=lambda message: {}),
        "aimdk_msgs.msg": NS(McCommonState=object, JointStateArray=object,
            McLocomotionVelocity=lambda: NS(header=NS(stamp=None)),
            McInputAction=NS(INPUTACTION_ADD=1, INPUTACTION_ENABLE=2)),
        "aimdk_msgs.srv": NS(**{name: NS(Request=Request) for name in
            ("SetMcInputSource", "SetMcPresetMotion", "GetMcPresetMotionState")}),
        "nav_msgs.msg": NS(Odometry=object),
    }
    for name, module in inert.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(control, "SkillExecutor", executor)
    monkeypatch.setenv("ROS_DOMAIN_ID", "47")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")

    class InertRuntime(host.SkillRuntime):
        def preflight(self):
            self.default_context = "desktop-linux"
            self.remote = str(self.output)

        def prepare_transport(self):
            pass

        def start_simulator(self):
            pass

        def spawn(self, command, log_name):
            tokens = shlex.split(command[-1])
            argv = tokens[tokens.index("malecns_lab.experiments.x2_skills")+1:]
            code = probe.main(argv)
            return NS(poll=lambda: code, returncode=code)

        def cleanup(self):
            self.resources_stopped = True

        def command(self, *args, **kwargs):
            return subprocess.CompletedProcess([], 0, "desktop-linux\n", "")

    monkeypatch.setattr(host, "SkillRuntime", InertRuntime)
    monkeypatch.setattr(host, "output_directory", lambda value: Path(value))
    monkeypatch.setattr(host, "numerical_exclusion", lambda root: nullcontext())
    monkeypatch.setattr(host.desktop, "desktop_lock", lambda root: nullcontext())

    for number, (arguments, expected) in enumerate((([], 1.), (["--command-seconds", "2"], 2.))):
        for main in (host.main, probe.main):
            assert main(arguments) == 0
            preview = json.loads(capsys.readouterr().out)
            assert preview["command_seconds"] == expected
            assert not preview.get("simulator_started", preview.get("ros_connected"))
        output = tmp_path / f"run-{number}"
        assert host.main(["--execute", "--sim-only", "--skill", "TURN_LEFT",
                          "--output", str(output), *arguments]) == 1
        capsys.readouterr()
        assert cores[-1].command_seconds == expected
        summary = json.loads((output / "host_summary.json").read_text())
        assert summary["command_seconds"] == expected
        assert summary["resources_stopped"]
        assert len(summary["probes"]) == 1
        probe_summary = summary["probes"][0]["summary"]
        assert probe_summary["command_seconds"] == expected
        assert "source_registration_failed" in probe_summary["failure"]
        events = [json.loads(line) for line in (output / "probe-00-turn_left/events.jsonl").read_text().splitlines()]
        configuration = next(event for event in events if event["kind"] == "configuration")
        assert configuration["command_seconds"] == expected
        assert (configuration["max_target"], configuration["wave_seconds"], configuration["wall_seconds"]) == (.2, 12., 40.)

    for value in ("0", "-1", "1.5", "2.001", "nan", "inf", "-inf"):
        for main in (host.main, probe.main):
            with pytest.raises(SystemExit) as error:
                main(["--command-seconds", value])
            assert error.value.code == 2
        with pytest.raises(ValueError):
            InertRuntime(tmp_path, tmp_path, host.desktop.Commands(io.StringIO()), lambda event: None,
                         command_seconds=float(value))

    # The inherited four-argument M4 constructor still chooses 1 second.
    assert InertRuntime(tmp_path, tmp_path, host.desktop.Commands(io.StringIO()),
                        lambda event: None).command_seconds == 1.
    assert len(cores) == 2
    assert all((message.forward_velocity, message.lateral_velocity, message.angular_velocity) == (0., 0., 0.)
               for message in published)
