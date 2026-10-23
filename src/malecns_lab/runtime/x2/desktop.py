"""Bounded, opt-in desktop coordinator for the existing GPU simulation loop.

Only this fixed local runtime is managed. Import/help/preview are inert; neither
the desktop nor this wrapper changes the trained model or official SDK.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[4]
CONTAINER = "malecns-x2-gpu-trial"
SCHEMA = "malecns_lab.desktop_event.v1"
GIB = 1024**3
WALL_SECONDS = 240.
VM_SECONDS = 120.
STOP_SECONDS = 12.
CLEANUP_SECONDS = 30.
TRANSPORT = ("__init__.py", *("control/" + name + ".py" for name in
             ("__init__", "adapters", "body", "command", "legacy", "x2", "x2_protocol")),
             *("experiments/" + name + ".py" for name in ("__init__", "x2", "x2_bridge")))


class StopRequested(Exception):
    pass


def output_directory(value, root=ROOT):
    """A single new basename under the existing shared volume, never a shell path."""
    directory = Path(value).resolve()
    shared = (Path(root) / "data/x2_sim/agibot/data/var").resolve()
    if directory.parent != shared or not re.fullmatch(r"desktop-\d{8}-\d{6}-[a-f0-9]{8,32}", directory.name):
        raise ValueError("Output must be a new desktop-YYYYMMDD-HHMMSS-hex run in the shared X2 data directory")
    if directory.exists():
        raise FileExistsError(directory)
    return directory


@contextmanager
def desktop_lock(root):
    with (Path(root) / "data/.x2-desktop.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Commands:
    """The only incoming commands; malformed input and disconnection fail closed."""
    def __init__(self, stream):
        self.pending = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.reason = None
        self.stream = stream

    def request_stop(self, reason):
        self.reason = self.reason or reason
        self.stop.set()

    def read(self):
        try:
            while True:
                line = self.stream.readline(1025)
                if not line:
                    self.request_stop("desktop_disconnected")
                    return
                if len(line) > 1024:
                    raise ValueError("Oversized desktop command")
                value = json.loads(line)
                if value == {"command": "stop"}:
                    self.request_stop("user_stop")
                    return
                if value != {"command": "continue"}:
                    raise ValueError("Unknown desktop command")
                self.pending.put_nowait("continue")
        except (OSError, ValueError, queue.Full):
            self.request_stop("invalid_desktop_command")

    def start(self):
        threading.Thread(target=self.read, daemon=True).start()

    def consume_continue(self):
        try:
            return self.pending.get_nowait() == "continue"
        except queue.Empty:
            return False


class Events:
    """A slow/closed renderer cannot block the safety coordinator's main thread."""
    def __init__(self, stream, log, on_failure=lambda reason: None):
        self.stream, self.log = stream, log
        self.on_failure = on_failure
        self.pending = queue.Queue(maxsize=64)
        self.dropped = 0
        self.broken = False
        threading.Thread(target=self._send, daemon=True).start()

    def _send(self):
        while True:
            line = self.pending.get()
            try:
                self.stream.write(line)
                self.stream.flush()
            except (BrokenPipeError, OSError):
                self.broken = True
                self.on_failure("desktop_output_disconnected")
            finally:
                self.pending.task_done()

    def emit(self, value):
        value = dict(value, delivery_dropped=self.dropped)
        line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            self.log.write(line)
            self.log.flush()
        except OSError:
            self.on_failure("desktop_event_log_failed")
        try:
            self.pending.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def drain(self):
        deadline = time.monotonic() + .2
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.01)


class Runtime:
    def __init__(self, output, root, commands, emit):
        from ...control import x2_protocol
        self.protocol = x2_protocol
        self.root, self.output = Path(root).resolve(), Path(output)
        self.runtime = self.root / "data/x2_sim/gpu_probe/runtime-rebuild"
        self.remote = "/agibot/data/var/" + self.output.name
        self.environment = dict(os.environ, LIGHTER_HOME=str(self.runtime))
        # Explicit --host takes precedence; inherited contexts/TLS are not used.
        for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
            self.environment.pop(name, None)
        self.docker = ["docker", "--host", "unix://" + str(self.runtime / "docker.sock")]
        self.commands, self.emit = commands, emit
        self.started, self.parent_pid = time.monotonic(), os.getppid()
        self.vm_started = None
        self.vm_stopped = None
        self.owns_vm = self.owns_container = False
        self.foreign_resources = False
        self.host = self.bridge = None
        self.run_id = None
        self.simulator_ready = self.motion_started = self.resources_stopped = False
        self.stop_confirmed = None
        self.bridge_started = False
        self.state = "preparing"
        self.errors, self.logs = [], []
        self.last_phase = None
        self.default_context = None
        self.stop_request_ns = None
        self.bridge_event_offset = 0

    def status(self, state, message, **fields):
        self.state = state
        value = {"schema": SCHEMA, "type": "status", "state": state, "message": message,
                   "output": str(self.output), "run_id": self.run_id,
                   "simulator_ready": self.simulator_ready, "motion_started": self.motion_started,
                   "stop_confirmed": self.stop_confirmed, "resources_stopped": self.resources_stopped,
                   "host_monotonic_ns": time.monotonic_ns(), **fields}
        if self.output.exists():
            temporary = self.output / ("desktop_session." + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                temporary.replace(self.output / "desktop_session.json")
            except OSError as error:
                self.errors.append("desktop_session_write:" + str(error))
                self.commands.request_stop("desktop_session_write_failed")
                if state in ("completed", "cancelled"):
                    self.state = value["state"] = "failed"
                    value["message"] = "实验记录未能封存"
        self.emit(value)

    def deadline(self, cleanup=False):
        deadline = self.started + WALL_SECONDS
        if self.vm_started is not None:
            deadline = min(deadline, self.vm_started + VM_SECONDS)
        return deadline if cleanup else deadline - CLEANUP_SECONDS

    def guard(self):
        import psutil
        if os.getppid() != self.parent_pid:
            self.commands.request_stop("desktop_parent_lost")
        if self.commands.stop.is_set():
            raise StopRequested(self.commands.reason)
        if time.monotonic() >= self.deadline():
            raise TimeoutError("Reserved stopping/cleanup window reached")
        if psutil.virtual_memory().available < GIB:
            raise RuntimeError("Less than 1 GiB system reserve")

    def command(self, args, timeout=8., *, cleanup=False, check=True, record=None):
        """Poll commands so stop/EOF stays responsive even while lighter starts."""
        end = min(self.deadline(cleanup), time.monotonic() + timeout)
        if end <= time.monotonic():
            raise TimeoutError("No command budget remains")
        child = None
        stdout = stderr = ""
        failure = None
        try:
            child = subprocess.Popen(args, cwd=self.root, env=self.environment, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, start_new_session=True)
            while True:
                if not cleanup:
                    self.guard()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Command timed out: " + str(args[:3]))
                try:
                    stdout, stderr = child.communicate(timeout=min(.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            failure = type(error).__name__ + ":" + str(error)
            if child is not None:
                self._kill_command(child)
                if record:
                    stdout, stderr = child.communicate(timeout=1.)
            raise
        finally:
            if record:
                (self.log_directory / record).write_text(json.dumps({
                    "args": args, "returncode": None if child is None else child.returncode,
                    "stdout": stdout, "stderr": stderr, "error": failure,
                    "host_monotonic_ns": time.monotonic_ns(),
                }, ensure_ascii=False) + "\n")
        result = subprocess.CompletedProcess(args, child.returncode, stdout, stderr)
        if check and result.returncode:
            raise RuntimeError("Command failed: " + str(args[:4]) + ": " + stderr[-1000:])
        return result

    @staticmethod
    def _kill_command(child):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=1)

    def preflight(self):
        self.guard()
        status = self.command([str(self.runtime / "bin/lighter"), "status"], check=False)
        if "lighter is not running." not in status.stdout + status.stderr or (self.runtime / "docker.sock").exists():
            self.foreign_resources = True
            raise RuntimeError("GPU VM must already be stopped; refusing to take ownership")
        self.default_context = self.command(["docker", "context", "show"]).stdout.strip()

    def spawn(self, args, name, environment=None):
        stream = (self.log_directory / name).open("x")
        self.logs.append(stream)
        return subprocess.Popen(args, cwd=self.root, env=environment or self.environment,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self):
        end = min(self.deadline(), time.monotonic() + 80.)
        while not (self.output / "ready.json").exists():
            self.guard()
            if self.host.poll() is not None:
                raise RuntimeError("Complete-brain process exited before ready")
            if time.monotonic() >= end:
                raise TimeoutError("Complete-brain readiness timed out")
            time.sleep(.1)
        registration = self.protocol.read(self.output / "registration.json")
        self.run_id = registration["run_id"]
        ready = self.protocol.verify(self.protocol.read(self.output / "ready.json"), "ready", self.run_id, 0)
        if ready["registration_id"] != registration["registration_id"]:
            raise ValueError("Complete-brain registration changed")
        return registration

    def transport(self, registration):
        manifest = {}
        for name in TRANSPORT:
            source = self.root / "src/malecns_lab" / name
            content = source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            expected = registration["wrapper_sha256"].get(name)
            if expected is not None and expected != digest:
                raise ValueError("Registered wrapper changed: " + name)
            target = self.output / "transport_src/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[name] = digest
        self.protocol.atomic(self.output / "transport_manifest.json", manifest)

    def start_simulator(self):
        self.preflight()  # The complete brain may have taken time; do not adopt a newly started VM.
        # Ownership is acquired only after an observed stopped preflight, before
        # start so a partially completed start is also cleaned up.
        self.vm_started = time.monotonic()
        self.owns_vm = True
        self.status("preparing", "正在启动 GPU 仿真", stage="vm_start")
        self.command([str(self.runtime / "bin/lighter"), "start", "--timeout", "60"], timeout=60.)
        running = self.command(self.docker + ["ps", "--format", "{{.Names}}"])
        if running.stdout.strip():
            # Never power off another owner's auto-started workloads.
            self.owns_vm = False
            self.foreign_resources = True
            raise RuntimeError("Private VM contains pre-existing running containers; resources left untouched")
        state = self.command(self.docker + ["inspect", "--format", "{{.State.Status}}", CONTAINER]).stdout.strip()
        if state not in ("exited", "created"):
            self.owns_vm = False
            self.foreign_resources = True
            raise RuntimeError("GPU container is not stopped; refusing ownership")
        self.owns_container = True
        self.command(self.docker + ["start", CONTAINER])
        verification = ("import pathlib,json,hashlib; p=pathlib.Path(" + repr(self.remote) + "); "
                        "m=json.loads((p/'transport_manifest.json').read_text()); "
                        "a={k:hashlib.sha256((p/'transport_src/malecns_lab'/k).read_bytes()).hexdigest() for k in m}; "
                        "assert a==m")
        self.command(self.docker + ["exec", CONTAINER, "python3", "-c", verification])
        self.command(self.docker + ["exec", "-d", "-e", "DRI_NO_MSAA=1", "-e", "VGL_SAMPLES=0",
                     "-e", "LIBGL_KOPPER_DISABLE=true", CONTAINER, "bash", "-lc",
                     "printf '0\\n' | /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 /usr/local/bin/x2-component sim > "
                     + shlex.quote(self.remote + "/sim.log") + " 2>&1"])
        self.command(self.docker + ["exec", "-d", CONTAINER, "bash", "-lc",
                     "printf '0\\n' | /usr/local/bin/x2-component mc > " + shlex.quote(self.remote + "/mc.log") + " 2>&1"])
        self.wait_for_official_sd()
        # This means display/services are available, not that Reset/Run happened.
        self.simulator_ready = True
        while self.commands.consume_continue():
            pass  # A command before the prompt cannot certify the official UI.
        self.status("awaiting_simulation", "请在左侧仿真中点击 Reset 和 Run，再点击开始实验", stage="await_ui_reset_run")
        end = min(self.deadline(), self.vm_started + 65.)
        while not self.commands.consume_continue():
            self.guard()
            if self.host.poll() is not None:
                raise RuntimeError("Complete-brain process exited during simulator setup")
            if time.monotonic() >= end:
                raise TimeoutError("Official UI preparation timed out")
            time.sleep(.1)
        self.guard()

    def wait_for_official_sd(self):
        # SD already waits for the official service. A discovered service can
        # still reject the action during MC startup, so space bounded retries.
        end = min(self.deadline(), time.monotonic() + 25.)
        for attempt in range(1, 6):
            self.guard()
            if self.host.poll() is not None:
                raise RuntimeError("Complete-brain process exited during MC setup")
            remaining = end - time.monotonic()
            if remaining <= 2.:
                raise TimeoutError("Official SD readiness budget exhausted; see sd-attempt logs")
            self.status("preparing", "正在等待官方控制服务就绪", stage="mc_ready", sd_attempt=attempt)
            duration = min(15., remaining - 2.)
            result = self.command(self.docker + ["exec", CONTAINER, "bash", "-lc",
                                  f"timeout --kill-after=1s {duration:.3f}s ros2 run py_examples set_mc_action SD"],
                                  timeout=min(17., remaining), check=False, record=f"sd-attempt-{attempt:02d}.json")
            if result.returncode == 0 and "Robot mode set successfully" in result.stdout + result.stderr:
                return
            if attempt < 5:
                retry_at = min(end, time.monotonic() + 1.5)
                while time.monotonic() < retry_at:
                    self.guard()
                    if self.host.poll() is not None:
                        raise RuntimeError("Complete-brain process exited during MC setup")
                    time.sleep(max(0., min(.1, retry_at - time.monotonic())))
        raise RuntimeError("Official SD did not confirm success; see sd-attempt logs")

    def start_bridge(self):
        script = ("source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && "
                  "PYTHONPATH=" + shlex.quote(self.remote + "/transport_src") + ":$PYTHONPATH "
                  "ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_bridge "
                  "--execute --sim-only --output " + shlex.quote(self.remote))
        self.bridge = self.spawn(self.docker + ["exec", CONTAINER, "bash", "-lc", script], "bridge-cli.log")
        self.bridge_started = True
        self.stop_confirmed = False
        self.status("running", "实验已启动，正在等待机器人反馈", stage="bridge_start")

    def poll_phase(self):
        # A brain decision is not a published command. Inspect only the actual
        # bridge's publication evidence; displacement remains separate evidence.
        path = self.output / "bridge_events.jsonl"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(self.bridge_event_offset)
                for _ in range(512):
                    line = stream.readline(65537)
                    if not line or not line.endswith(b"\n"):
                        break
                    self.bridge_event_offset = stream.tell()
                    value = json.loads(line)
                    if value.get("kind") == "tick" and any(
                            item.get("ok") is True and item.get("command") == [.2, 0., 0.]
                            for item in value.get("attempts", [])):
                        self.motion_started = True
        path = self.output / "phase.json"
        if path.exists():
            value = self.protocol.read(path)
            phase = (value.get("phase"), value.get("sequence"))
            if phase != self.last_phase:
                self.last_phase = phase
                label = {"forward": "脑模型正在计算", "wait_observation": "正在等待新的机器人观测",
                         "wait_stopped_ack": "正在等待机器人停稳", "save_state": "正在记录脑状态"}.get(phase[0], "实验运行中")
                self.status("running", label, stage=phase[0], sequence=phase[1])

    def stopped_evidence(self):
        summary_path = self.output / "bridge_summary.json"
        event_path = self.output / "bridge_events.jsonl"
        if not summary_path.exists() or not event_path.exists():
            return False
        summary = self.protocol.read(summary_path)
        if (summary.get("run_id") != self.run_id or summary.get("final_stopped") is not True
                or summary.get("active_forward") is not False):
            return False
        with event_path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get("kind") == "tick":
                return event.get("state") == "STOPPED" and any(
                    attempt.get("ok") is True and attempt.get("command") == [0., 0., 0.]
                    for attempt in event.get("attempts", []))
        return False

    def request_abort(self, reason):
        if not self.output.exists():
            return
        if self.run_id is None and (self.output / "registration.json").exists():
            self.run_id = self.protocol.read(self.output / "registration.json")["run_id"]
        path = self.output / "host_abort.json"
        if self.run_id is not None and not path.exists():
            try:
                self.protocol.atomic(path, self.protocol.seal({"kind": "abort", "run_id": self.run_id, "sequence": 0, "reason": reason}))
            except (FileExistsError, ValueError):
                if not path.exists():
                    raise
        self.stop_request_ns = time.monotonic_ns()

    def stop_motion(self, reason):
        self.status("stopping", "正在请求停车并等待反馈", stage="request_stop")
        self.request_abort(reason)
        if not self.bridge_started:
            return  # No claimed controller: not evidence of a measured stop.
        end = min(self.deadline(cleanup=True) - 16., time.monotonic() + STOP_SECONDS)
        while time.monotonic() < end:
            if self.stopped_evidence():
                self.stop_confirmed = True
                break
            if self.bridge.poll() is not None:
                break
            time.sleep(.05)
        if self.stop_confirmed is not True:
            self.stop_confirmed = self.stopped_evidence()
        self.status("stopping", "已收到零速与停稳反馈" if self.stop_confirmed else "未能确认停稳，正在关闭仿真资源",
                    stage="stop_feedback")

    def stop_host(self):
        if self.host is None or self.host.poll() is not None:
            return
        import psutil
        descendants = []
        try:
            descendants = psutil.Process(self.host.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        self.host.terminate()  # Host supervisor gives its numerical child a bounded exit.
        try:
            self.host.wait(timeout=4.)
        except subprocess.TimeoutExpired:
            self.host.kill()
            self.host.wait(timeout=1.)
        for process in descendants:
            try:
                if process.is_running():
                    process.kill()  # psutil verifies PID identity before sending signals.
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(descendants, timeout=1.)
        if alive:
            raise RuntimeError("Owned complete-brain child did not exit")

    def cleanup(self):
        self.status("stopping", "正在关闭本次实验的资源", stage="cleanup")
        clean = True
        if self.owns_container:
            try:
                self.command(self.docker + ["stop", "--time", "2", CONTAINER], timeout=5., cleanup=True)
                status = self.command(self.docker + ["inspect", "--format", "{{.State.Status}}", CONTAINER],
                                      timeout=2., cleanup=True).stdout.strip()
                if status != "exited":
                    raise RuntimeError("GPU container shutdown not confirmed")
                self.owns_container = False
            except Exception as error:
                clean = False
                self.errors.append("container_cleanup:" + str(error))
        if self.owns_vm:
            try:
                self.command([str(self.runtime / "bin/lighter"), "stop"], timeout=6., cleanup=True)
                result = self.command([str(self.runtime / "bin/lighter"), "status"], timeout=2., cleanup=True, check=False)
                if "lighter is not running." not in result.stdout + result.stderr or (self.runtime / "docker.sock").exists():
                    raise RuntimeError("GPU VM shutdown not confirmed")
                self.owns_vm = False
                self.vm_stopped = time.monotonic()
            except Exception as error:
                clean = False
                self.errors.append("vm_cleanup:" + str(error))
        try:
            self.stop_host()
        except Exception as error:
            clean = False
            self.errors.append("host_cleanup:" + str(error))
        if self.bridge is not None and self.bridge.poll() is None:
            self._kill_command(self.bridge)
        self.resources_stopped = clean and not self.owns_vm and not self.owns_container and not self.foreign_resources
        if self.resources_stopped:
            self.simulator_ready = False
        for stream in self.logs:
            stream.close()

    def execute(self, log_directory):
        self.log_directory = Path(log_directory)
        terminal, reason = "failed", "startup_incomplete"
        self.status("preparing", "正在检查环境并准备完整脑模型", stage="preflight")
        try:
            self.preflight()
            self.host = self.spawn([str(self.root / ".venv-train/bin/python"), "-u", "-m",
                                   "malecns_lab.experiments.x2_neural", "--execute", "--sim-only", "--telemetry",
                                   "--output", str(self.output)], "host-cli.log",
                                  dict(os.environ, PYTHONPATH=str(self.root / "src")))
            registration = self.wait_ready()
            self.transport(registration)
            self.start_simulator()
            self.start_bridge()
            while self.bridge.poll() is None:
                self.guard()
                self.poll_phase()
                if self.host.poll() is not None and not (self.output / "summary.json").exists():
                    raise RuntimeError("Complete-brain process disappeared")
                time.sleep(.1)
            end = min(self.deadline(), time.monotonic() + 5.)
            while self.host.poll() is None and time.monotonic() < end:
                self.guard()
                time.sleep(.05)
            summary = self.protocol.read(self.output / "summary.json")
            self.stop_confirmed = self.stopped_evidence()
            if (self.bridge.returncode != 0 or self.host.poll() != 0 or summary.get("status") != "completed"
                    or summary.get("run_id") != self.run_id or not self.stop_confirmed):
                raise RuntimeError("Experiment did not seal completed brain and actual stop evidence")
            terminal, reason = "completed", "completed"
        except StopRequested as error:
            terminal, reason = "cancelled", str(error)
        except BaseException as error:
            terminal, reason = "failed", type(error).__name__ + ":" + str(error)
        finally:
            if terminal != "completed":
                try:
                    self.stop_motion(reason)
                except Exception as error:
                    self.errors.append("stop_request:" + str(error))
            try:
                self.cleanup()
            except Exception as error:
                self.errors.append("cleanup_exception:" + str(error))
                self.resources_stopped = False
            if self.errors or not self.resources_stopped or (self.bridge_started and self.stop_confirmed is not True):
                terminal = "failed"
            result = {"state": terminal, "reason": reason, "errors": self.errors,
                      "run_id": self.run_id, "output": str(self.output), "stop_confirmed": self.stop_confirmed,
                      "resources_stopped": self.resources_stopped, "motion_started": self.motion_started,
                      "default_docker_context_before": self.default_context,
                      "wall_seconds": time.monotonic() - self.started,
                      "vm_seconds": None if self.vm_started is None else (self.vm_stopped or time.monotonic()) - self.vm_started}
            for path in [self.log_directory / "summary.json"] + ([self.output / "desktop_summary.json"] if self.output.exists() else []):
                try:
                    self.protocol.atomic(path, result)
                except (OSError, ValueError) as error:
                    self.errors.append("summary_write:" + str(error))
                    terminal = result["state"] = "failed"
            self.status(terminal, {"completed": "实验完成，已停稳并关闭资源", "cancelled": "实验已取消，资源已关闭",
                                   "failed": "实验未完成，请查看失败记录"}[terminal], reason=reason, errors=self.errors)
            result["state"] = self.state
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "preview", "brain_loaded": False, "simulator_started": False,
                          "wall_seconds": WALL_SECONDS, "vm_seconds": VM_SECONDS,
                          "system_reserve_bytes": GIB, "rounds": 2, "requires_official_reset_run": True}))
        return 0
    if not args.sim_only or args.output is None:
        parser.error("--execute requires --sim-only and a new --output")
    old_signals = {}
    try:
        directory = output_directory(args.output)
        commands = Commands(sys.stdin)
        old_signals = {number: signal.signal(number, lambda number, _frame: commands.request_stop("signal:" + str(number)))
                       for number in (signal.SIGINT, signal.SIGTERM)}
        with desktop_lock(ROOT):
            logs = ROOT / "logs/desktop_runs" / directory.name
            logs.mkdir(parents=True, exist_ok=False)
            with (logs / "lifecycle.jsonl").open("x") as log:
                events = Events(sys.stdout, log, commands.request_stop)
                commands.start()
                result = Runtime(directory, ROOT, commands, events.emit).execute(logs)
                events.drain()
        return 0 if result["state"] in ("completed", "cancelled") else 1
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "type": "status", "state": "failed", "message": "无法启动实验",
                          "reason": type(error).__name__ + ":" + str(error), "output": str(args.output),
                          "run_id": None, "simulator_ready": False, "motion_started": False,
                          "stop_confirmed": None, "resources_stopped": False,
                          "host_monotonic_ns": time.monotonic_ns()}, ensure_ascii=False), flush=True)
        return 1
    finally:
        for number, handler in old_signals.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
