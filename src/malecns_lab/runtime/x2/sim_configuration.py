"""Pure construction of one opt-in simulation configuration; never launch it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import xml.etree.ElementTree as ET

PROFILES = ('official', 'noslip1')
SCHEMA = 'malecns.x2.simulation_configuration.v1'
PACKAGE = Path('data/x2_sim/packages/sim_mujoco')
ROBOT = 'configuration/robot/lx2501_3_t2d5'
REMOTE_PACKAGE = '/workspace/sim_mujoco'
SOURCES = (f'{ROBOT}/model_info/x2.xml', f'{ROBOT}/model_info/scene.xml',
           f'{ROBOT}/model_info/default.yaml', f'{ROBOT}/simulator/default.yaml',
           'bin/start_sim.sh', 'bin/aima-sim-app')
LIBRARIES = ('lib/libmujoco.so', 'lib/libmujoco.so.3.3.7', 'lib/libmujoco.so.3.3.8')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _replace_once(text, old, new):
    _require(text.count(old) == 1, 'Official configuration layout changed: '+old)
    return text.replace(old, new, 1)


def prepare_configuration(root, output, remote, profile):
    """Write only three small derived files and their manifest in a new run."""
    _require(profile in PROFILES, 'Unsupported simulation profile')
    if profile == 'official':
        return None
    root, output = Path(root), Path(output)
    _require(PurePosixPath(remote) == PurePosixPath('/agibot/data/var')/output.name,
             'Simulation configuration must belong to the current run')
    original = {}
    for name in SOURCES:
        path = root/PACKAGE/name
        _require(path.is_file() and not path.is_symlink(), 'Missing or linked simulation source: '+name)
        original[name] = path.read_bytes()
    installed_library_hashes = {}
    for name in LIBRARIES:
        path = root/PACKAGE/name
        _require(path.is_file() and not path.is_symlink(), 'Missing or linked simulation library: '+name)
        installed_library_hashes[f'{REMOTE_PACKAGE}/{name}'] = _sha(path.read_bytes())
    model = original[SOURCES[0]].decode()
    parsed = ET.fromstring(model)
    options, compilers = parsed.findall('option'), parsed.findall('compiler')
    _require(len(options) == 1 and options[0].attrib == {'timestep': '0.001'}
             and len(compilers) == 1 and compilers[0].get('meshdir') == '../meshes',
             'Official physics/compiler configuration changed')
    model = _replace_once(model, '<option timestep="0.001" />',
                          '<option timestep="0.001" noslip_iterations="1" />')
    meshdir = f'{REMOTE_PACKAGE}/{ROBOT}/meshes'
    model = _replace_once(model, 'meshdir="../meshes"', f'meshdir="{meshdir}"')
    scene = original[SOURCES[1]].decode()
    _require([node.attrib for node in ET.fromstring(scene).findall('include')] == [{'file': 'x2.xml'}]
             and not ET.fromstring(scene).findall('option'), 'Official scene includes or physics changed')
    # Keep the scene include relative to the three run-local files.
    cfg_dir = remote+'/simulation_config'
    config = original[SOURCES[3]].decode()
    config = _replace_once(config, 'temp_cfg_path: ./cfg/tmp', f'temp_cfg_path: {cfg_dir}/cfg/tmp')
    config = _replace_once(config, 'path: ./log', f'path: {cfg_dir}/log')
    config = _replace_once(config, 'simulation_model: ${SIM_ROBOT_PATH}/model_info/scene.xml',
                           f'simulation_model: {cfg_dir}/scene.xml')
    config = _replace_once(config, 'robot_model: ${SIM_ROBOT_PATH}/model_info/default.yaml',
                           f'robot_model: {REMOTE_PACKAGE}/{ROBOT}/model_info/default.yaml')
    derived = {'x2.xml': model.encode(), 'scene.xml': scene.encode(), 'simulator.yaml': config.encode()}
    identity = dict(schema=SCHEMA, profile=profile,
        original_sha256={f'{REMOTE_PACKAGE}/{name}': _sha(content) for name, content in original.items()},
        installed_library_hashes=installed_library_hashes,
        derived_sha256={f'{cfg_dir}/{name}': _sha(content) for name, content in derived.items()},
        physics_change=dict(attribute='option.noslip_iterations', original='absent', candidate=1),
        path_rebindings=dict(meshdir=meshdir, simulation_model=cfg_dir+'/scene.xml',
                             robot_model=f'{REMOTE_PACKAGE}/{ROBOT}/model_info/default.yaml',
                             temp_cfg_path=cfg_dir+'/cfg/tmp', log=cfg_dir+'/log'),
        executable=f'{REMOTE_PACKAGE}/bin/aima-sim-app', working_directory=cfg_dir,
        environment_source=f'{REMOTE_PACKAGE}/bin/start_sim.sh:set_environment_variables/start_robot')
    configuration_id = _sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode())
    manifest = dict(**identity, configuration_id=configuration_id, run_id=output.name,
        status='prepared_not_executed', evidence_scope='configuration bytes and installed library hashes only; not loaded runtime version, compiled mjOption readback or successful simulation')
    directory = output/'simulation_config'
    directory.mkdir(exist_ok=False)
    for name, content in derived.items():
        (directory/name).write_bytes(content)
    (directory/'identity.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    return manifest


def verification_code(manifest):
    """Container-side file hash check, distinct from loading/physics readback."""
    directory = manifest['working_directory']
    return '\n'.join((
        'import hashlib,json,pathlib,time',
        'expected='+repr(manifest),
        'directory=pathlib.Path('+repr(directory)+')',
        "assert json.loads((directory/'identity.json').read_text()) == expected, 'Simulation manifest changed'",
        "for name,sha in {**expected['original_sha256'],**expected['installed_library_hashes'],**expected['derived_sha256']}.items():",
        ' p=pathlib.Path(name)',
        " assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'Simulation file changed: '+name",
        "receipt=dict(schema='malecns.x2.simulation_file_verification.v1',run_id=expected['run_id'],configuration_id=expected['configuration_id'],files_match=True,mjoption_readback=False,simulator_started=False,remote_monotonic=time.monotonic())",
        "with (directory/'file_verification.json').open('x') as stream: json.dump(receipt,stream,sort_keys=True)",
    ))


def launch_shell(manifest, remote):
    """Official binary/environment without the official launcher's cache writes."""
    _require(manifest['profile'] == 'noslip1' and manifest['working_directory'] == remote+'/simulation_config',
             'Simulation launch identity differs')
    directory = manifest['working_directory']
    # Match the official launcher. Relative library names resolve through the
    # official bin/lib paths; assets stay at their original absolute paths.
    return '\n'.join((
        'set -eo pipefail',
        'source /etc/profile.d/x2.sh',
        'unset AGIBOT_SOFTWARE_HOME',
        'export LD_LIBRARY_PATH=/workspace/sim_mujoco/bin:/workspace/sim_mujoco/bin:/workspace/sim_mujoco/lib:${LD_LIBRARY_PATH:-}',
        'export EM_APP_NAME=${EM_APP_NAME:-sim}',
        'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp',
        'export AGIBOT_ENABLE_HDS_COMPONENT=false AGIBOT_ENABLE_EVENT_COMPONENT=false',
        'export AGIBOT_ENABLE_MONITOR_COMPONENT=false AGIBOT_ENABLE_MONITOR=true',
        'export AGIBOT_ENABLE_AUDIT_COMPONENT=false AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US=300',
        'export SIM_ROBOT_PATH='+shlex.quote(f'{REMOTE_PACKAGE}/{ROBOT}'),
        'export SIM_RESOURCE_MODEL_PATH='+shlex.quote(f'{REMOTE_PACKAGE}/resource/model'),
        'cd '+shlex.quote(directory),
        'exec /opt/VirtualGL/bin/vglrun -d egl0 -ms 0 '+shlex.quote(manifest['executable'])+
        ' --cfg_file_path='+shlex.quote(directory+'/simulator.yaml')+' > '+shlex.quote(remote+'/sim.log')+' 2>&1',
    ))
