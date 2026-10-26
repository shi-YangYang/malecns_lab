"""Configuration/file checks only: never start a process, VM, ROS or brain."""
import hashlib
import json
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import pytest

from malecns_lab.experiments import x2_continuity
from malecns_lab.runtime.x2 import behavior, sim_configuration as config
from malecns_lab.runtime.x2.skills import SkillRuntime


MODEL = '<mujoco><compiler meshdir="../meshes"/><option timestep="0.001" /><worldbody/></mujoco>\n'
SCENE = '<mujoco><include file="x2.xml"/><worldbody><geom type="plane" size="1 1 1"/></worldbody></mujoco>\n'
YAML = '''aimrt:
  configurator:
    temp_cfg_path: ./cfg/tmp
  plugin:
    path: libaimrt_sm_plugin.so
  log:
    path: ./log
MujocoSimModule:
  simulation_model: ${SIM_ROBOT_PATH}/model_info/scene.xml
  robot_model: ${SIM_ROBOT_PATH}/model_info/default.yaml
  frequency: 1000
'''


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'root'
    for name, content in zip(config.SOURCES, (MODEL, SCENE, 'odom: pelvis\n', YAML, 'official launcher', 'binary')):
        path = root/config.PACKAGE/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in config.LIBRARIES:
        path = root/config.PACKAGE/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'installed library: '+name.encode())
    output = tmp_path/'desktop-fixture'
    output.mkdir()
    return root, output, '/agibot/data/var/'+output.name


def sha(value):
    return hashlib.sha256(value).hexdigest()


def test_default_is_a_noop_and_unknown_profile_fails(tmp_path):
    output = tmp_path/'absent'
    assert config.prepare_configuration(tmp_path, output, '/irrelevant', 'official') is None
    assert not output.exists()
    with pytest.raises(ValueError, match='Unsupported'):
        config.prepare_configuration(tmp_path, output, '/irrelevant', 'noslip2')


def test_single_physics_change_and_sealed_bytes_without_asset_copy(source):
    root, output, remote = source
    before = {name: (root/config.PACKAGE/name).read_bytes() for name in (*config.SOURCES, *config.LIBRARIES)}
    manifest = config.prepare_configuration(root, output, remote, 'noslip1')
    directory = output/'simulation_config'
    assert sorted(p.name for p in directory.iterdir()) == ['identity.json', 'scene.xml', 'simulator.yaml', 'x2.xml']
    model = ET.fromstring((directory/'x2.xml').read_text())
    assert model.find('option').attrib == {'timestep': '0.001', 'noslip_iterations': '1'}
    assert model.find('compiler').get('meshdir') == config.REMOTE_PACKAGE+'/'+config.ROBOT+'/meshes'
    del model.find('option').attrib['noslip_iterations']
    model.find('compiler').set('meshdir', '../meshes')
    assert ET.tostring(model) == ET.tostring(ET.fromstring(MODEL))
    assert (directory/'scene.xml').read_text() == SCENE
    expected_yaml = YAML.replace('./cfg/tmp', remote+'/simulation_config/cfg/tmp').replace(
        './log', remote+'/simulation_config/log').replace(
        '${SIM_ROBOT_PATH}/model_info/scene.xml', remote+'/simulation_config/scene.xml').replace(
        '${SIM_ROBOT_PATH}/model_info/default.yaml', config.REMOTE_PACKAGE+'/'+config.ROBOT+'/model_info/default.yaml')
    assert (directory/'simulator.yaml').read_text() == expected_yaml
    assert {name: (root/config.PACKAGE/name).read_bytes() for name in before} == before
    assert manifest['original_sha256'] == {config.REMOTE_PACKAGE+'/'+name: sha(before[name]) for name in config.SOURCES}
    assert manifest['installed_library_hashes'] == {config.REMOTE_PACKAGE+'/'+name: sha(before[name]) for name in config.LIBRARIES}
    assert manifest['derived_sha256'] == {remote+'/simulation_config/'+name: sha((directory/name).read_bytes())
                                         for name in ('x2.xml', 'scene.xml', 'simulator.yaml')}
    assert json.loads((directory/'identity.json').read_text()) == manifest
    assert manifest['status'] == 'prepared_not_executed'
    assert 'not loaded runtime version' in manifest['evidence_scope']
    identity = {key: value for key, value in manifest.items() if key not in ('configuration_id', 'run_id', 'status', 'evidence_scope')}
    assert manifest['configuration_id'] == sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode())
    with pytest.raises(FileExistsError):
        config.prepare_configuration(root, output, remote, 'noslip1')


@pytest.mark.parametrize('change', ['option', 'include', 'yaml', 'library_symlink', 'run_identity'])
def test_changed_input_or_run_identity_is_rejected_before_output(source, change):
    root, output, remote = source
    if change in ('option', 'include', 'yaml'):
        index, old, new = {'option': (0, 'timestep="0.001"', 'timestep="0.002"'),
                           'include': (1, 'file="x2.xml"', 'file="other.xml"'),
                           'yaml': (3, 'temp_cfg_path: ./cfg/tmp', 'temp_cfg_path: ./elsewhere')}[change]
        path = root/config.PACKAGE/config.SOURCES[index]
        path.write_text(path.read_text().replace(old, new))
    elif change == 'library_symlink':
        path = root/config.PACKAGE/config.LIBRARIES[0]
        path.unlink()
        path.symlink_to(path.with_name('libmujoco.so.3.3.7'))
    else:
        remote += '-another-run'
    with pytest.raises(ValueError):
        config.prepare_configuration(root, output, remote, 'noslip1')
    assert not (output/'simulation_config').exists()


def test_hash_receipt_is_not_a_runtime_readback_and_tampering_fails(tmp_path):
    # Execute only the generated stdlib file verification against temporary files.
    asset = tmp_path/'asset'
    asset.write_bytes(b'unchanged')
    manifest = dict(working_directory=str(tmp_path), run_id='fixture', configuration_id='fixture-id',
                    original_sha256={str(asset): sha(asset.read_bytes())}, installed_library_hashes={}, derived_sha256={})
    (tmp_path/'identity.json').write_text(json.dumps(manifest))
    exec(config.verification_code(manifest), {})
    receipt_path = tmp_path/'file_verification.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['files_match'] and not receipt['mjoption_readback'] and not receipt['simulator_started']
    receipt_path.unlink()
    asset.write_bytes(b'changed')
    with pytest.raises(AssertionError, match='Simulation file changed'):
        exec(config.verification_code(manifest), {})
    assert not receipt_path.exists()


def test_launch_is_explicit_verified_and_does_not_use_package_cache(source):
    root, output, remote = source
    runtime = object.__new__(behavior.BehaviorRuntime)
    runtime.remote, runtime.sim_profile = remote, 'official'
    old = "printf '0\\n' | /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 /usr/local/bin/x2-component sim > "+shlex.quote(remote+'/sim.log')+' 2>&1'
    assert runtime.simulator_launch_shell() == SkillRuntime.simulator_launch_shell(runtime) == old
    runtime.sim_profile = 'noslip1'
    runtime.simulation_configuration = config.prepare_configuration(root, output, remote, 'noslip1')
    runtime.simulation_files_verified = False
    with pytest.raises(ValueError, match='not verified'):
        runtime.simulator_launch_shell()
    runtime.simulation_files_verified = True
    shell = runtime.simulator_launch_shell()
    assert 'source /etc/profile.d/x2.sh\nunset AGIBOT_SOFTWARE_HOME' in shell
    assert 'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp' in shell
    assert 'export LD_LIBRARY_PATH=/workspace/sim_mujoco/bin:/workspace/sim_mujoco/bin:/workspace/sim_mujoco/lib:' in shell
    assert 'export SIM_RESOURCE_MODEL_PATH=/workspace/sim_mujoco/resource/model' in shell
    assert '\ncd '+remote+'/simulation_config\n' in shell
    assert 'exec /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 /workspace/sim_mujoco/bin/aima-sim-app --cfg_file_path='+remote+'/simulation_config/simulator.yaml' in shell
    assert 'start_sim.sh' not in shell and 'cache' not in shell and 'rm ' not in shell


def test_cli_preview_and_forwarding_never_start_anything(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Configuration preview attempted a process')
    monkeypatch.setattr(x2_continuity.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(behavior, 'candidate', forbidden)
    output = tmp_path/'absent'
    for entry in (behavior.main, x2_continuity.main):
        assert entry(['--output', str(output), '--sim-profile', 'noslip1']) == 0
        preview = json.loads(capsys.readouterr().out)
        assert preview['sim_profile'] == 'noslip1'
        assert not preview['brain_loaded'] and not preview['simulator_started']
        assert entry([]) == 0
        assert json.loads(capsys.readouterr().out)['sim_profile'] == 'official'
    assert not output.exists()
    old = x2_continuity.runtime_command('/python', output, 'qualification')
    assert '--sim-profile' not in old
    assert x2_continuity.runtime_command('/python', output, 'qualification', sim_profile='noslip1') == old+['--sim-profile', 'noslip1']


def test_installed_source_layout_builds_without_modifying_package(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root/config.PACKAGE/config.SOURCES[0]).exists():
        pytest.skip('Optional official assets are not installed')
    output = tmp_path/'installed-config-fixture'
    output.mkdir()
    manifest = config.prepare_configuration(root, output, '/agibot/data/var/'+output.name, 'noslip1')
    assert len(manifest['installed_library_hashes']) == 3
    assert set(p.suffix for p in (output/'simulation_config').iterdir()) == {'.xml', '.yaml', '.json'}
