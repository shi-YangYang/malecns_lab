"""Bounded host preparation budgets only; no subprocess, ROS, or simulation."""
import io
import json

import pytest

from malecns_lab.runtime.x2.desktop import Commands
from malecns_lab.runtime.x2.skills import SkillRuntime, main


def test_registered_ui_budgets_preserve_cleanup_and_subclass_deadlines(tmp_path, capsys):
    for arguments, ui, wall in (([], 80., 240.),
                               (["--ui-timeout", "180", "--wall-timeout", "360"], 180., 360.)):
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

    for ui, wall in ((180., 240.), (80., 360.), (0., 240.), (180., 361.),
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
