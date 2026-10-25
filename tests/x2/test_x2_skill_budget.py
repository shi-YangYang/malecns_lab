"""Bounded host preparation budgets only; no subprocess, ROS, or simulation."""
import io
import json

import pytest

from malecns_lab.runtime.x2.desktop import Commands
from malecns_lab.runtime.x2.skills import SkillRuntime, main


def test_behavior_preparation_preserves_vm_total_and_followup_cleanup_reserves(tmp_path):
    from malecns_lab.runtime.x2.behavior import BehaviorRuntime
    runtime = BehaviorRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda value: None,
                              tmp_path/'checkpoint', {'policy': {}})
    runtime.started = 100.
    assert runtime.ui_timeout == 180. and runtime.wall_timeout == 360.
    assert runtime.deadline(True) == 700.  # Overall 600-second budget is unchanged.
    runtime.vm_started = 140.
    assert runtime.deadline(True) == 500.  # VM lifetime is at most 360 seconds.
    assert runtime.deadline() == 465.
    assert runtime.ui_deadline(160.) == 340.
    assert runtime.ui_deadline(450.) == 405.  # Still reserves 60 + 35 seconds.
    budget = runtime.budget_record()
    assert budget['cleanup_reserve_seconds'] == 35.
    assert budget['after_ui_reserve_seconds'] == 60.
    runtime.vm_started = 500.
    assert runtime.deadline(True) == 700.  # A late VM cannot extend total wall time.
    assert runtime.ui_deadline(550.) == 605.


def test_registered_ui_budgets_preserve_cleanup_and_subclass_deadlines(tmp_path, capsys):
    for arguments, ui, wall in (([], 80., 240.),
                               (["--ui-timeout", "180", "--wall-timeout", "360"], 180., 360.),
                               (["--ui-timeout", "300", "--wall-timeout", "480"], 300., 480.)):
        assert main(arguments) == 0
        preview = json.loads(capsys.readouterr().out)
        assert not preview["simulator_started"] and not preview["brain_loaded"]
        assert preview["ui_timeout_seconds"] == ui
        assert preview["wall_seconds"] == wall
        assert preview["cleanup_reserve_seconds"] == 30.
        events = []
        runtime = SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), events.append,
                               ui_timeout=ui, wall_timeout=wall)
        runtime.started = 100.
        assert runtime.deadline(True) == 100.+wall
        assert runtime.deadline() == 100.+wall-30.
        assert runtime.ui_deadline(110.) == min(110.+ui, runtime.deadline()-60.)
        assert runtime.ui_deadline(runtime.deadline()-65.) == runtime.deadline()-60.
        runtime.status("preparing", "准备中")
        assert events[-1]["budget"]["wall_timeout_seconds"] == wall
        assert events[-1]["budget"]["cleanup_reserve_seconds"] == 30.

    for ui, wall in ((180., 240.), (80., 360.), (300., 360.), (180., 480.), (0., 240.), (180., 361.),
                     (float("inf"), 360.), (80., float("nan"))):
        with pytest.raises(ValueError):
            SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda value: None,
                         ui_timeout=ui, wall_timeout=wall)
        with pytest.raises(SystemExit) as error:
            main(["--ui-timeout", str(ui), "--wall-timeout", str(wall)])
        assert error.value.code == 2

    class DesktopRuntime(SkillRuntime):
        def deadline(self, cleanup=False):
            return self.started+600.-(0. if cleanup else 30.)

    # M4 retains the four-argument constructor and its own deadline override.
    runtime = DesktopRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda value: None)
    runtime.started = 100.
    assert runtime.ui_timeout == 80.
    assert runtime.deadline(True) == 700.
    assert runtime.budget_record()["wall_timeout_seconds"] == 600.


def test_wave_probe_accepts_both_finite_ui_budgets_before_runtime(tmp_path, monkeypatch):
    from malecns_lab.runtime.x2 import skills
    class ReachedOutputValidation(Exception):
        pass
    def intercept_output(_):
        raise ReachedOutputValidation()
    monkeypatch.setattr(skills, "output_directory", intercept_output)
    base = ["--execute", "--sim-only", "--wave-probe", "--skill", "WAVE_RIGHT", "--output", str(tmp_path)]
    for ui, wall in ((180, 360), (300, 480)):
        with pytest.raises(ReachedOutputValidation):
            main(base + ["--ui-timeout", str(ui), "--wall-timeout", str(wall)])
    with pytest.raises(SystemExit) as error:
        main(base + ["--ui-timeout", "80", "--wall-timeout", "240"])
    assert error.value.code == 2


def test_low_speed_probe_arguments_reject_invalid_values_before_runtime(tmp_path, monkeypatch, capsys):
    from malecns_lab.experiments import x2_skills as probe
    from malecns_lab.runtime.x2 import skills

    class ReachedOutputValidation(Exception):
        pass

    def intercept_output(_):
        raise ReachedOutputValidation()

    monkeypatch.setattr(skills, "output_directory", intercept_output)
    for entry in (skills.main, probe.main):
        assert entry([]) == 0
        assert json.loads(capsys.readouterr().out)["intensity"] == 1.
        assert entry(["--skill", "ADVANCE", "--command-seconds", "2", "--intensity", ".5"]) == 0
        preview = json.loads(capsys.readouterr().out)
        assert preview["intensity"] == .5 and preview["command_seconds"] == 2.
        for value in ("nan", "inf", "0", "-0.5", "1.01"):
            with pytest.raises(SystemExit) as error:
                entry(["--execute", "--sim-only", "--output", str(tmp_path), "--intensity", value])
            assert error.value.code == 2
            assert "intensity must be finite and within (0, 1]" in capsys.readouterr().err

    base = ["--execute", "--sim-only", "--output", str(tmp_path),
            "--command-seconds", "2", "--intensity", ".5", "--skill"]
    with pytest.raises(ReachedOutputValidation):
        skills.main(base + ["ADVANCE", "ADVANCE"])
    for requested in (["ADVANCE"] * 3, ["HOLD"] * 2, ["TURN_LEFT"] * 2,
                      ["TURN_RIGHT"] * 2, ["WAVE_RIGHT"] * 2,
                      ["ADVANCE", "ADVANCE", "HOLD", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT"]):
        with pytest.raises(SystemExit) as error:
            skills.main(base + requested)
        assert error.value.code == 2
    runtime = SkillRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda value: None)
    monkeypatch.setattr(runtime, "preflight", lambda: pytest.fail("invalid probes reached preflight"))
    with pytest.raises(ValueError, match="at most twice"):
        runtime.execute_skills(["ADVANCE"] * 3, False, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("first_result", ["success", "failed_summary", "failed_exit"])
def test_low_speed_probe_forwards_intensity_and_stops_repeats_on_failure(tmp_path, first_result):
    import shlex
    import subprocess
    from types import SimpleNamespace
    from malecns_lab.experiments import x2_skills as probe

    starts, launches = [], []

    class FakeRuntime(SkillRuntime):
        def preflight(self):
            self.default_context = "desktop-linux"
        def prepare_transport(self):
            pass
        def start_simulator(self):
            starts.append("single_simulation")
            self.owns_vm = self.owns_container = True
        def guard(self, **kwargs):
            pass
        def spawn(self, command, name):
            words = shlex.split(command[-1])
            args = probe.parser().parse_args(words[words.index("-m") + 2:])
            launches.append(args)
            assert args.skill == "ADVANCE" and args.command_seconds == 2. and args.intensity == .5
            directory = self.output / args.output.name
            directory.mkdir()
            success = first_result != "failed_summary"
            self.protocol.atomic(directory / "summary.json", dict(success=success, requested_intensity=args.intensity))
            code = 1 if first_result == "failed_exit" else 0
            return SimpleNamespace(returncode=code, poll=lambda: code)
        def cleanup(self):
            self.owns_vm = self.owns_container = False
            self.resources_stopped = True
        def command(self, args, **kwargs):
            assert args == ["docker", "context", "show"]
            return subprocess.CompletedProcess(args, 0, "desktop-linux\n", "")

    runtime = FakeRuntime(tmp_path, tmp_path, Commands(io.StringIO()), lambda value: None,
                          command_seconds=2., intensity=.5)
    result = runtime.execute_skills(["ADVANCE", "ADVANCE"], False, None)
    assert len(starts) == 1 and len(launches) == (2 if first_result == "success" else 1)
    assert [args.output.name for args in launches] == [f"probe-{i:02d}-advance" for i in range(len(launches))]
    assert result["state"] == ("completed" if first_result == "success" else "failed")
    assert result["intensity"] == .5 and result["command_seconds"] == 2.
    assert result["resources_stopped"]
