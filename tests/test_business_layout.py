"""Contracts for package entry points, source assets and offline imports."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY_MODULES = (
    "malecns_lab.experiments.tello.check",
    "malecns_lab.experiments.tello.hover",
    "malecns_lab.experiments.tello.shadow",
    "malecns_lab.experiments.tello.trial",
    "malecns_lab.experiments.tello.record",
    "malecns_lab.diagnostics.replay_v2",
    "malecns_lab.diagnostics.report_v2",
    "malecns_lab.learning.train_v2",
    "malecns_lab.learning.experiment_v2",
    "malecns_lab.learning.train_snn",
    "malecns_lab.learning.train_mps",
    "malecns_lab.learning.train_gpu",
    "malecns_lab.learning.evaluate_snn",
    "malecns_lab.learning.gradient_probe",
    "malecns_lab.learning.generate_rewiring",
    "malecns_lab.diagnostics.observability",
)


@pytest.fixture
def offline_environment(tmp_path):
    # Fail even on an SDK/video import; no real device can be constructed by a
    # mistakenly eager entry point. subprocesses also start outside the project.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class RejectHardwareImports:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('djitellopy', 'cv2', 'torch'):\n"
        "            raise AssertionError('Unexpected optional dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, RejectHardwareImports())\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    return tmp_path, env


def run_offline(args, offline_environment):
    cwd, env = offline_environment
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, text=True,
        capture_output=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_module_help_from_another_cwd(module, offline_environment):
    output = run_offline(["-m", module, "--help"], offline_environment)
    assert "usage:" in output


@pytest.mark.parametrize("module", (
    "malecns_lab.experiments.tello.hover", "malecns_lab.experiments.tello.trial",
))
def test_flight_entry_requires_takeoff_before_sdk_import(module, offline_environment):
    output = run_offline(["-m", module], offline_environment)
    assert "No connection made. --takeoff is required to fly." in output


def test_cli_and_low_level_imports_stay_offline(offline_environment):
    assert "build-brain" in run_offline(["-m", "malecns_lab", "--help"], offline_environment)
    run_offline(["-c", (
        "import malecns_lab.control, malecns_lab.learning; "
        "import malecns_lab.control.safety, malecns_lab.control.calibration"
    )], offline_environment)


def test_source_assets_use_real_workspace_from_another_cwd(offline_environment):
    output = run_offline(["-c", """
import json
from unittest.mock import patch
from malecns_lab.experiments.tello import record, shadow, trial
from malecns_lab.paths import source_project_root
with patch.object(trial.Path, 'is_file', return_value=True):
    cfg = trial.trial_config('full')
    cruise_cfg = trial.trial_config('cruise')
print(json.dumps({
    'roots': [str(module.PROJECT) for module in (record, shadow, trial)],
    'located': str(source_project_root()),
    'brain': cfg['brain']['source'], 'readout': cfg['decoder']['readout_file'],
    'cruise': cruise_cfg['decoder']['cruise'],
}))
"""], offline_environment)
    result = json.loads(output)
    assert result["roots"] == [str(ROOT)] * 3
    assert result["located"] == str(ROOT)
    assert result["brain"] == str(ROOT / "data" / "malecns_brain.npz")
    assert result["readout"] == str(ROOT / "readout_malecns.json")
    assert result["cruise"] == 0.25
