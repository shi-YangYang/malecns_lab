"""Own one private GPU simulation for bounded official skill probes, then close it.

No brain or simulation starts on import/preview. The operator performs the
visible official Reset/Run, then writes {"command":"continue"} to continue.json
in this fresh run directory. This runner never operates real hardware.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sys
import time

from . import desktop
from ...experiments.x2_skills import validate_command_seconds, validate_intensity

ROOT = desktop.ROOT
SKILLS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")
ASSETS = (
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc_sim.yaml",
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/action_setting.yaml",
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation_player.yaml",
    "mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/animation/SM_wave_right_hand.csv",
    "aimdk/src/aimdk_msgs/interface/robot/mc/motion/srv/SetMcPresetMotion.srv",
    "aimdk/src/aimdk_msgs/interface/robot/mc/motion/srv/GetMcPresetMotionState.srv",
    "aimdk/src/aimdk_msgs/interface/robot/mc/motion/msg/McPlayerState.msg",
    "aimdk/docs/cn/_build/html/zh-cn/dev/Interface/control_mod/preset_motion.html",
)
TRANSPORT = desktop.TRANSPORT + ("control/x2_skills.py", "experiments/x2_skills.py",
    "experiments/x2_preset_identity.py", "experiments/x2_preset_compat.py", "experiments/x2_preset_helper.py")
TIMEOUT_PRESETS = ((80., 240.), (180., 360.), (300., 480.))
CLEANUP_RESERVE = 30.
AFTER_UI_RESERVE = 60.


def validate_timeouts(ui_timeout, wall_timeout):
    if (ui_timeout, wall_timeout) not in TIMEOUT_PRESETS:
        raise ValueError("Use ui/wall timeouts 80/240, 180/360, or 300/480 seconds")


def validate_skills(skills):
    if (not skills or len(skills) > 5 or any(skill not in SKILLS for skill in skills)
            or any(skills.count(skill) > (2 if skill == "ADVANCE" else 1) for skill in skills)):
        raise ValueError("Request at most five probes; only ADVANCE may repeat, at most twice")


@contextmanager
def numerical_exclusion(root):
    with (root / "data/.continuous-training.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def output_directory(value):
    path = Path(value).resolve()
    if path.parent != (ROOT / "data/x2_sim/agibot/data/var").resolve() or not re.fullmatch(r"skills-\d{8}-\d{6}-[a-f0-9]{8,32}", path.name):
        raise ValueError("Use a new skills-YYYYMMDD-HHMMSS-hex directory in data/x2_sim/agibot/data/var")
    if path.exists():
        raise FileExistsError(path)
    return path


class SkillRuntime(desktop.Runtime):
    def __init__(self, output, root, commands, emit, *, ui_timeout=80., wall_timeout=240., command_seconds=1., intensity=1.):
        validate_timeouts(ui_timeout, wall_timeout)
        validate_command_seconds(command_seconds)
        validate_intensity(intensity)
        self.ui_timeout, self.wall_timeout = float(ui_timeout), float(wall_timeout)
        self.command_seconds = float(command_seconds)
        self.intensity = float(intensity)
        super().__init__(output, root, commands, emit)

    def budget_record(self):
        # Derive the effective deadlines so subclasses with their own runtime
        # limits do not report the standalone skill runner's wall budget.
        end, work_end = self.deadline(True), self.deadline()
        return dict(ui_timeout_seconds=self.ui_timeout, wall_timeout_seconds=end-self.started,
                    cleanup_reserve_seconds=end-work_end, after_ui_reserve_seconds=AFTER_UI_RESERVE,
                    wall_deadline_monotonic=end, work_deadline_monotonic=work_end)

    def ui_deadline(self, now):
        return min(self.deadline()-AFTER_UI_RESERVE, now+self.ui_timeout)

    def status(self, state, message, **fields):
        # Evidence output is fallible. Request a controlled exit, but never let
        # a broken log/stdout prevent the independent physical cleanup stages.
        try:
            super().status(state, message, budget=self.budget_record(), **fields)
        except (OSError, ValueError) as error:
            self.errors.append("status_output:" + repr(error))
            self.commands.request_stop("skill_status_output_failed")

    def cleanup(self):
        try:
            self.status("stopping", "正在关闭本次实验的资源", stage="cleanup")
        except BaseException as error:
            self.errors.append("cleanup_status:" + repr(error))
        if self.owns_container:
            try:
                self.command(self.docker + ["stop", "--time", "2", desktop.CONTAINER], timeout=5., cleanup=True)
            except BaseException as error:
                self.errors.append("container_stop:" + repr(error))
            try:
                status = self.command(self.docker + ["inspect", "--format", "{{.State.Status}}", desktop.CONTAINER],
                                      timeout=2., cleanup=True).stdout.strip()
                if status != "exited":
                    raise RuntimeError("GPU container shutdown not confirmed")
                self.owns_container = False
            except BaseException as error:
                self.errors.append("container_cleanup:" + repr(error))
        if self.owns_vm:
            try:
                self.command([str(self.runtime / "bin/lighter"), "stop"], timeout=6., cleanup=True)
            except BaseException as error:
                self.errors.append("vm_stop:" + repr(error))
            try:
                result = self.command([str(self.runtime / "bin/lighter"), "status"], timeout=2., cleanup=True, check=False)
                if "lighter is not running." not in result.stdout + result.stderr or (self.runtime / "docker.sock").exists():
                    raise RuntimeError("GPU VM shutdown not confirmed")
                self.owns_vm = False
                self.vm_stopped = time.monotonic()
            except BaseException as error:
                self.errors.append("vm_cleanup:" + repr(error))
        children_stopped = True
        try:
            self.stop_host()
            if self.host is not None and self.host.poll() is None:
                raise RuntimeError("Owned host process shutdown not confirmed")
        except BaseException as error:
            children_stopped = False
            self.errors.append("host_cleanup:" + repr(error))
        try:
            if self.bridge is not None and self.bridge.poll() is None:
                self._kill_command(self.bridge)
            if self.bridge is not None and self.bridge.poll() is None:
                raise RuntimeError("Owned probe process shutdown not confirmed")
        except BaseException as error:
            children_stopped = False
            self.errors.append("probe_cleanup:" + repr(error))
        self.resources_stopped = (children_stopped and not self.owns_vm and not self.owns_container
                                  and not self.foreign_resources)
        if self.resources_stopped:
            self.simulator_ready = False
        for stream in self.logs:
            try:
                stream.close()
            except BaseException as error:
                self.errors.append("log_close:" + repr(error))

    def deadline(self, cleanup=False):
        # Includes VM startup, UI wait, all probes, and final resource teardown.
        return self.started + self.wall_timeout - (0. if cleanup else CLEANUP_RESERVE)

    def prepare_transport(self):
        manifest = {}
        for name in TRANSPORT:
            content = (self.root / "src/malecns_lab" / name).read_bytes()
            target = self.output / "transport_src/malecns_lab" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        (self.output / "host_source.py").write_bytes(Path(__file__).read_bytes())
        self.protocol.atomic(self.output / "transport_manifest.json", manifest)
        assets = {name: hashlib.sha256((self.root / "data/x2_sim/packages" / name).read_bytes()).hexdigest() for name in ASSETS}
        self.protocol.atomic(self.output / "official_assets.json", dict(
            configured_versions=dict(aimdk="v1.1.0-gfaf4c94", mc="v1.0.0-20260522", sim="v1.0.0-20260522", model="lx2501_3_t2d5"),
            sha256=assets, note="Configured version labels plus exact local files; no on-site version claim"))

    def prepare_preset_compat(self):
        from ...experiments import x2_preset_identity as identity
        root = self.root / "data/x2_sim/packages" / identity.DIRECTORY
        binding = identity.verify_build(root)
        prepared = identity.read(root / "prepared.json")
        for name in ("x2_preset_identity.py", "x2_preset_compat.py", "x2_preset_helper.py"):
            if identity.sha(self.root / "src/malecns_lab/experiments" / name) != prepared["tool_files"]["malecns_lab/experiments/" + name]:
                raise ValueError("Current helper source differs from inspected build")
        archive = self.output / "preset_compat"
        archive.mkdir(exist_ok=False)
        for name in ("prepared.json", "build.json", "inspection.json"):
            (archive / name).write_bytes((root / name).read_bytes())
        for name in identity.EVIDENCE:
            (archive / name).write_bytes((root / "evidence" / name).read_bytes())
        self.protocol.atomic(archive / "binding.json", binding)

    def wait_for_sd(self):
        end = min(self.deadline(), time.monotonic()+25.)
        for attempt in range(1,6):
            self.guard()
            remaining = end-time.monotonic()
            if remaining <= 2.:
                raise TimeoutError("Official SD readiness budget exhausted")
            duration = min(15., remaining-2.)
            r = self.command(self.docker+["exec", desktop.CONTAINER, "bash", "-lc",
                f"timeout --kill-after=1s {duration:.3f}s ros2 run py_examples set_mc_action SD"],
                timeout=min(17.,remaining),check=False,record=f"sd-{attempt:02d}.json")
            if r.returncode == 0 and "Robot mode set successfully" in r.stdout+r.stderr:
                return
            if attempt < 5:
                time.sleep(min(1.5,max(0.,end-time.monotonic())))
        raise RuntimeError("Official SD not confirmed")

    def verify_simulator_configuration(self):
        """Official startup has no derived simulation configuration."""

    def simulator_launch_shell(self):
        return "printf '0\\n' | /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 /usr/local/bin/x2-component sim > "+shlex.quote(self.remote+"/sim.log")+" 2>&1"

    def prepare_simulator_ui(self):
        """CLI skill probes retain their explicit operator preparation step."""
        return False

    def start_simulator(self):
        self.preflight()
        self.vm_started, self.owns_vm = time.monotonic(), True
        self.command([str(self.runtime / "bin/lighter"),"start","--timeout","60"],timeout=60.)
        running = self.command(self.docker+["ps","--format","{{.Names}}"])
        if running.stdout.strip():
            self.owns_vm, self.foreign_resources = False, True
            raise RuntimeError("Pre-existing private workloads; refusing ownership")
        state = self.command(self.docker+["inspect","--format","{{.State.Status}}",desktop.CONTAINER]).stdout.strip()
        if state not in ("exited","created"):
            self.owns_vm, self.foreign_resources = False, True
            raise RuntimeError("GPU container is not stopped")
        self.owns_container = True
        self.command(self.docker+["start",desktop.CONTAINER])
        check = ("import pathlib,json,hashlib;p=pathlib.Path("+repr(self.remote)+");"
            "m=json.loads((p/'transport_manifest.json').read_text());"
            "assert {k:hashlib.sha256((p/'transport_src/malecns_lab'/k).read_bytes()).hexdigest() for k in m}==m;"
            "a=json.loads((p/'official_assets.json').read_text())['sha256'];"
            "assert {k:hashlib.sha256((pathlib.Path('/workspace')/k).read_bytes()).hexdigest() for k in a}==a")
        self.command(self.docker+["exec",desktop.CONTAINER,"python3","-c",check])
        self.verify_simulator_configuration()
        self.command(self.docker+["exec","-d","-e","DRI_NO_MSAA=1","-e","VGL_SAMPLES=0","-e","LIBGL_KOPPER_DISABLE=true",
            desktop.CONTAINER,"bash","-lc",self.simulator_launch_shell()])
        self.command(self.docker+["exec","-d",desktop.CONTAINER,"bash","-lc","printf '0\\n' | /usr/local/bin/x2-component mc > "+shlex.quote(self.remote+"/mc.log")+" 2>&1"])
        self.wait_for_sd()
        self.simulator_ready=True
        automatic = self.prepare_simulator_ui()
        end = self.ui_deadline(time.monotonic())
        self.protocol.atomic(self.output / "ready.json",dict(state="awaiting_desktop_connection" if automatic else "awaiting_official_reset_run", url="http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale", continue_file=str(self.output/"continue.json"), budget=self.budget_record(), ui_deadline_monotonic=end))
        self.status("awaiting_simulation", "正在连接仿真画面" if automatic else "请先 Pause → Reset，确认直立且暂停后 Run，再写入 continue.json",
                    stage="await_desktop_connection" if automatic else "await_ui_reset_run", ui_deadline_monotonic=end)
        while time.monotonic() < end:
            self.guard()
            path=self.output/"continue.json"
            if path.exists():
                if self.protocol.read(path) != {"command":"continue"}:
                    raise ValueError("Invalid continue command")
                return
            time.sleep(.1)
        raise TimeoutError("Official UI preparation timed out")

    def execute_skills(self, skills, wave_probe, cancel_after, wave_cancel_probe=False):
        validate_skills(skills)
        self.log_directory=self.output
        outcome={"state":"failed","reason":"startup_incomplete","probes":[],
                 "command_seconds":self.command_seconds,"intensity":self.intensity}
        try:
            self.preflight()
            self.prepare_transport()
            if "WAVE_RIGHT" in skills:
                self.prepare_preset_compat()  # Fail before VM/MC/Sim or any action.
            self.start_simulator()
            for number,skill in enumerate(skills):
                self.guard()
                if self.deadline()-time.monotonic() < 41.:
                    raise TimeoutError("Insufficient remaining budget for another bounded skill")
                remote=self.remote+f"/probe-{number:02d}-{skill.lower()}"
                command=("source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && "
                    "PYTHONPATH="+shlex.quote(self.remote+"/transport_src")+":$PYTHONPATH "
                    "ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2_skills "
                    "--execute --sim-only --skill "+skill+" --output "+shlex.quote(remote)+
                    " --command-seconds "+str(self.command_seconds)+" --intensity "+str(self.intensity))
                if wave_probe:
                    command += " --wave-probe"
                if wave_cancel_probe:
                    command += " --wave-cancel-probe"
                if cancel_after is not None:
                    command += " --cancel-after "+str(cancel_after)
                self.status("running","正在核验官方技能",skill=skill,command_seconds=self.command_seconds,intensity=self.intensity)
                self.bridge=self.spawn(self.docker+["exec",desktop.CONTAINER,"bash","-lc",command],f"probe-{number:02d}-cli.log")
                while self.bridge.poll() is None:
                    self.guard()
                    time.sleep(.1)
                path=self.output/f"probe-{number:02d}-{skill.lower()}"/"summary.json"
                summary=self.protocol.read(path)
                outcome["probes"].append(dict(skill=skill, returncode=self.bridge.returncode, summary=summary))
                if self.bridge.returncode != 0 or not summary.get("success"):
                    raise RuntimeError("Skill probe did not complete: "+skill)
                self.bridge=None
            outcome.update(state="completed",reason="requested_skill_probes_completed")
        except BaseException as error:
            outcome.update(state="failed",reason=type(error).__name__+":"+str(error))
        finally:
            # SIGTERM targets the fixed probe entrypoint in this exclusively
            # owned simulation container. The runner
            # reserves bounded finally cleanup; container teardown is last.
            if self.bridge is not None and self.bridge.poll() is None and self.owns_container:
                try:
                    self.command(self.docker+["exec",desktop.CONTAINER,"pkill","-TERM","-f","^python3 -m malecns_lab.experiments.x2_skills "],timeout=2.,cleanup=True,check=False)
                    end=min(self.deadline(True)-15.,time.monotonic()+12.)
                    while self.bridge.poll() is None and time.monotonic()<end:
                        time.sleep(.1)
                except Exception as error:
                    self.errors.append("probe_stop:"+str(error))
            try:
                self.cleanup()
            except BaseException as error:
                self.errors.append("cleanup_exception:"+repr(error))
                self.resources_stopped=False
            context=None
            try:
                context=self.command(["docker","context","show"],timeout=2.,cleanup=True,check=False).stdout.strip()
            except BaseException as error:
                self.errors.append("context_check:"+repr(error))
            if context != self.default_context:
                self.errors.append("default_context_changed")
            if self.errors or not self.resources_stopped:
                outcome["state"]="failed"
            outcome.update(resources_stopped=self.resources_stopped,errors=self.errors,
                budget=self.budget_record(),
                default_context_before=self.default_context,default_context_after=context,
                wall_seconds=time.monotonic()-self.started,
                vm_seconds=None if self.vm_started is None else (self.vm_stopped or time.monotonic())-self.vm_started,
                wave_cancellation_verified=False)
            self.status(outcome["state"],"技能核验结束，已执行资源清理",reason=outcome["reason"])
            if self.errors:
                outcome["state"] = "failed"
            try:
                self.protocol.atomic(self.output/"host_summary.json",outcome)
            except (OSError, ValueError) as error:
                self.errors.append("summary_write:" + repr(error))
                outcome["state"] = "failed"
        return outcome


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--sim-only",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--skill",choices=SKILLS,nargs="+",default=["HOLD"])
    p.add_argument("--wave-probe",action="store_true")
    p.add_argument("--wave-cancel-probe",action="store_true")
    p.add_argument("--cancel-after",type=float)
    p.add_argument("--command-seconds",type=float,default=1.,help="continuous motion pulse: 1 or 2 seconds; does not alter preset wave playback")
    p.add_argument("--intensity",type=float,default=1.,help="skill intensity in (0, 1]; continuous velocity uses the existing 0.2 scaling")
    p.add_argument("--ui-timeout",type=float,default=80.)
    p.add_argument("--wall-timeout",type=float,default=240.)
    args=p.parse_args(argv)
    try:
        validate_timeouts(args.ui_timeout, args.wall_timeout)
        validate_command_seconds(args.command_seconds)
        validate_intensity(args.intensity)
        validate_skills(args.skill)
    except ValueError as error:
        p.error(str(error))
    if not args.execute:
        print(json.dumps(dict(mode="preview",brain_loaded=False,simulator_started=False,skills=args.skill,
            command_seconds=args.command_seconds,intensity=args.intensity,
            wall_seconds=args.wall_timeout,ui_timeout_seconds=args.ui_timeout,cleanup_reserve_seconds=CLEANUP_RESERVE,
            system_reserve_bytes=desktop.GIB)))
        return 0
    if not args.sim_only or args.output is None or ("WAVE_RIGHT" in args.skill and not args.wave_probe):
        p.error("execute requires --sim-only and a new --output; wave also requires --wave-probe")
    if args.cancel_after is not None and not .1 <= args.cancel_after <= 4.:
        p.error("cancel-after must be within .1..4 seconds")
    if "WAVE_RIGHT" in args.skill and (args.skill != ["WAVE_RIGHT"] or args.cancel_after is not None
            or (args.ui_timeout, args.wall_timeout) not in ((180., 360.), (300., 480.))):
        p.error("Compatibility trial requires only WAVE_RIGHT, no cancel-after, and budgets 180/360 or 300/480")
    if args.wave_cancel_probe and (args.skill != ["WAVE_RIGHT"] or not args.wave_probe):
        p.error("wave-cancel-probe requires only WAVE_RIGHT and wave-probe")
    directory=output_directory(args.output)
    commands=desktop.Commands(sys.stdin)  # No stdin reader; only signals stop this CLI.
    handlers={sig:signal.signal(sig,lambda sig,frame:commands.request_stop("signal:"+str(sig))) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with desktop.desktop_lock(ROOT), numerical_exclusion(ROOT):
            directory.mkdir(parents=True,exist_ok=False)
            with (directory/"lifecycle.jsonl").open("x") as stream:
                def emit(value):
                    line=json.dumps(value,ensure_ascii=False)+"\n"
                    stream.write(line)
                    stream.flush()
                    print(line,end="",flush=True)
                result=SkillRuntime(directory,ROOT,commands,emit,ui_timeout=args.ui_timeout,
                    wall_timeout=args.wall_timeout,command_seconds=args.command_seconds,intensity=args.intensity).execute_skills(
                        args.skill,args.wave_probe,args.cancel_after,args.wave_cancel_probe)
        return 0 if result["state"]=="completed" else 1
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == "__main__":
    raise SystemExit(main())
