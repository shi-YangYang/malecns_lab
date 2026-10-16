"""Migration contracts for callers, source assets and hardware-free entry points."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cnsbridge.config import default_config
from cnsbridge.learning.prepare import RecordingBrain


ROOT = Path(__file__).resolve().parents[1]
TOOLS = (
    "tello_check", "tello_hover", "tello_shadow", "tello_yaw_trial",
    "tello_record_dataset", "prepare_motion_dataset", "train_motion_decoder",
)


@pytest.mark.parametrize(("old", "new", "names"), [
    ("motor", "control", "BodyVelocity DifferentialDriveAdapter FlightCommand FlightCommandAdapter MotorDecoder"),
    ("motor", "learning", "MotionDecoder NeuralObservation NeuralSchema RidgeMotionDecoder"),
    ("motor.body", "control.body", "BODY_AXES BODY_FRAME BODY_UNITS BodyVelocity"),
    ("motor.command", "control.command", "AXES FlightCommand"),
    ("motor.body_adapters", "control.adapters", "Adaptation WheelSpeeds FlightCommandAdapter DifferentialDriveAdapter"),
    ("motor.decoder", "control.legacy", "MotorDecoder"),
    ("motor.neural", "learning.decoder", "MotionDecoder NeuralObservation NeuralSchema RidgeMotionDecoder"),
    ("motor.dataset", "learning.dataset", "MotionDataset concatenate_datasets evaluate_split"),
    ("safety", "control.safety", "Telemetry SafetyGovernor"),
    ("calibrate", "control.calibration", "STIMULI record_responses fit_readout calibrate"),
])
def test_old_imports_are_the_same_implementation(old, new, names):
    legacy = importlib.import_module(f"cnsbridge.{old}")
    canonical = importlib.import_module(f"cnsbridge.{new}")
    for name in names.split():
        assert getattr(legacy, name) is getattr(canonical, name)


@pytest.fixture
def offline_environment(tmp_path):
    # Fail even on an SDK/video import; no real device can be constructed by a
    # mistakenly eager entry point. subprocesses also start outside the project.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class RejectHardwareImports:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('djitellopy', 'cv2'):\n"
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


@pytest.mark.parametrize("tool", TOOLS)
def test_old_tool_help_from_another_cwd(tool, offline_environment):
    output = run_offline([str(ROOT / "tools" / f"{tool}.py"), "--help"], offline_environment)
    assert "usage:" in output


@pytest.mark.parametrize("tool", ("tello_hover", "tello_yaw_trial"))
def test_old_flight_entry_requires_takeoff_before_sdk_import(tool, offline_environment):
    output = run_offline([str(ROOT / "tools" / f"{tool}.py")], offline_environment)
    assert "No connection made. --takeoff is required to fly." in output


def test_cli_and_low_level_imports_stay_offline(offline_environment):
    assert "build-brain" in run_offline(["-m", "cnsbridge", "--help"], offline_environment)
    run_offline(["-c", (
        "import cnsbridge.control, cnsbridge.learning, cnsbridge.motor; "
        "import cnsbridge.safety, cnsbridge.calibrate"
    )], offline_environment)


def test_source_assets_use_real_workspace_from_another_cwd(offline_environment):
    output = run_offline(["-c", """
import json
from unittest.mock import patch
from cnsbridge.experiments.tello import record, shadow, trial
from cnsbridge.learning import prepare
from cnsbridge.paths import source_project_root
with patch.object(trial.Path, 'is_file', return_value=True):
    cfg = trial.trial_config('full')
print(json.dumps({
    'roots': [str(module.PROJECT) for module in (record, shadow, trial, prepare)],
    'located': str(source_project_root()),
    'brain': cfg['brain']['source'], 'readout': cfg['decoder']['readout_file'],
    'cruise': trial.trial_config('cruise')['decoder']['cruise'],
}))
"""], offline_environment)
    result = json.loads(output)
    assert result["roots"] == [str(ROOT)] * 4
    assert result["located"] == str(ROOT)
    assert result["brain"] == str(ROOT / "data" / "malecns_brain.npz")
    assert result["readout"] == str(ROOT / "readout_malecns.json")
    assert result["cruise"] == 0.25


def test_recording_feature_identity_matches_pre_migration_baseline():
    # Captured from the original tools/prepare_motion_dataset.py at 8546f6c,
    # before moving implementation files. This is not two aliases of new code.
    schema = RecordingBrain(default_config()).schema
    assert schema.source_id == "ground-replay-sha256:17277d1fcd8e9f68ac683ff913a1a7f6f51534060fcf3da6e507d486bf513ef5"
    assert schema.feature_names == tuple(
        f"neuron_index_{index}" for index in (*range(407, 425), *range(832, 850))
    )
    assert schema.rate_window_ms == 50.0
