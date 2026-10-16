"""Contracts for package entry points, source assets and offline imports."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

import pytest

from cnsbridge.learning import prepare


ROOT = Path(__file__).resolve().parents[1]
ENTRY_MODULES = (
    "cnsbridge.experiments.tello.check",
    "cnsbridge.experiments.tello.hover",
    "cnsbridge.experiments.tello.shadow",
    "cnsbridge.experiments.tello.trial",
    "cnsbridge.experiments.tello.record",
    "cnsbridge.learning.prepare",
    "cnsbridge.learning.train",
)


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


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_module_help_from_another_cwd(module, offline_environment):
    output = run_offline(["-m", module, "--help"], offline_environment)
    assert "usage:" in output


@pytest.mark.parametrize("module", (
    "cnsbridge.experiments.tello.hover", "cnsbridge.experiments.tello.trial",
))
def test_flight_entry_requires_takeoff_before_sdk_import(module, offline_environment):
    output = run_offline(["-m", module], offline_environment)
    assert "No connection made. --takeoff is required to fly." in output


def test_cli_and_low_level_imports_stay_offline(offline_environment):
    assert "build-brain" in run_offline(["-m", "cnsbridge", "--help"], offline_environment)
    run_offline(["-c", (
        "import cnsbridge.control, cnsbridge.learning; "
        "import cnsbridge.control.safety, cnsbridge.control.calibration"
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
    cruise_cfg = trial.trial_config('cruise')
print(json.dumps({
    'roots': [str(module.PROJECT) for module in (record, shadow, trial, prepare)],
    'located': str(source_project_root()),
    'brain': cfg['brain']['source'], 'readout': cfg['decoder']['readout_file'],
    'cruise': cruise_cfg['decoder']['cruise'],
}))
"""], offline_environment)
    result = json.loads(output)
    assert result["roots"] == [str(ROOT)] * 4
    assert result["located"] == str(ROOT)
    assert result["brain"] == str(ROOT / "data" / "malecns_brain.npz")
    assert result["readout"] == str(ROOT / "readout_malecns.json")
    assert result["cruise"] == 0.25


def test_recording_feature_identity_tracks_real_graph_and_configuration(malecns_graph, malecns_config):
    with patch.object(prepare, "load_connectome", return_value=malecns_graph):
        engine = prepare.RecordingBrain(malecns_config)
        schema = engine.schema
        expected = np.unique(np.concatenate([malecns_graph.group(name) for name in malecns_config["outputs"]]))
        assert schema.feature_names == tuple(f"neuron_index_{index}" for index in expected)
        assert schema.source_id.startswith("ground-replay-sha256:")
        assert len(schema.source_id.split(":")[1]) == 64
        assert schema.rate_window_ms == 50.0
        assert engine.provenance["connectome_name"] == "malecns-v1.0"
        assert engine.provenance["neurons"] == 166700
        malecns_config["brain"]["bias"]["DNg02_L"] += 1
        assert prepare.RecordingBrain(malecns_config).schema.source_id != schema.source_id
