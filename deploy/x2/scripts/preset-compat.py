#!/usr/bin/env python3
"""Explicit prepare/build/inspect stages; never starts a container or simulation.

Host: PYTHONPATH=src python deploy/x2/scripts/preset-compat.py prepare
Linux (owned, mounted /workspace; clean shell is launched here):
  python3 /workspace/aimdk-preset-v1_0_0/preset-compat.py build
  python3 /workspace/aimdk-preset-v1_0_0/preset-compat.py inspect
Only the coordinator may run these after engineering acceptance.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import subprocess
import sys
import time
import zipfile

# Installed tool snapshot takes precedence only inside the fixed build directory.
HERE = Path(__file__).resolve()
if HERE.parent == Path('/workspace/aimdk-preset-v1_0_0'):
    sys.path.insert(0, str(HERE.parent / 'tool_src'))
from malecns_lab.experiments import x2_preset_identity as identity
from malecns_lab.experiments.x2_preset_compat import isolated_environment

FILES = ('x2_preset_identity.py', 'x2_preset_compat.py', 'x2_preset_helper.py')


def prepare():
    repo = HERE.parents[3]
    source = repo / 'data/x2_sim/compatibility-v1_0_0-20261023'
    root = repo / 'data/x2_sim/packages' / identity.DIRECTORY
    archive = source / identity.ARCHIVE
    if archive.stat().st_size != 156945205 or identity.sha(archive) != identity.ZIP_SHA:
        raise ValueError('Official archive identity mismatch')
    for name, digest in identity.EVIDENCE.items():
        if identity.sha(source / name) != digest:
            raise ValueError('Official compatibility evidence mismatch')
    root.mkdir(exist_ok=False)
    (root / 'evidence').mkdir()
    for name in (*identity.EVIDENCE, 'source.json'):
        (root / 'evidence' / name).write_bytes((source / name).read_bytes())
    destination = root / 'source/aimdk_msgs'
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            if not item.filename.startswith(identity.ZIP_ROOT) or item.is_dir():
                continue
            relative = item.filename[len(identity.ZIP_ROOT):]
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive member')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(zipped.read(item))
    if (destination / 'prebuilt_x86_64').exists():
        raise ValueError('Unexpected x86 prebuilt; source build required')
    official = identity.read(root / 'evidence/official-source-sha256.json')
    actual = {str(p.relative_to(destination)): identity.sha(p) for p in (destination / 'interface').rglob('*') if p.is_file()}
    if actual != {k.removeprefix('src/aimdk_msgs/'): v for k, v in official.items()}:
        raise ValueError('Full official interface inventory mismatch')
    tool_dir = root / 'tool_src/malecns_lab/experiments'
    tool_dir.mkdir(parents=True)
    for name in FILES:
        (tool_dir / name).write_bytes((repo / 'src/malecns_lab/experiments' / name).read_bytes())
    (root / 'preset-compat.py').write_bytes(HERE.read_bytes())
    prepared = dict(schema=identity.SCHEMA, zip_sha256=identity.ZIP_SHA,
                    source_files=identity.inventory(destination),
                    tool_files=identity.inventory(root / 'tool_src'),
                    build_script_sha256=identity.sha(root / 'preset-compat.py'),
                    new_install_files=identity.inventory(root.parent / 'aimdk/install/aimdk_msgs'))
    identity.write_new(root / 'prepared.json', prepared)
    print(root)


def clean_command(stage):
    root = identity.LINUX_ROOT
    source = 'source /opt/ros/humble/setup.bash; '
    if stage == '_inspect':
        source += 'source ' + str(root / 'install/local_setup.bash') + '; '
    return ['/bin/bash', '--noprofile', '--norc', '-ec', source +
            'exec /usr/bin/python3 ' + str(root / 'preset-compat.py') + ' ' + stage]


def linux_stage(stage):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or HERE.parent != identity.LINUX_ROOT:
        raise ValueError('Build/inspect only in the fixed mounted Linux x86_64 directory')
    root = identity.LINUX_ROOT
    prepared = identity.read(root / 'prepared.json')
    if identity.sha(HERE) != prepared['build_script_sha256']:
        raise ValueError('Build script changed')
    identity.verify_inventory(root / 'tool_src', prepared['tool_files'])
    env = isolated_environment()
    env['PYTHONPATH'] = str(root / 'tool_src')
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
    env['MAKEFLAGS'] = '-j2'
    # One process group bounds colcon and its descendants; no build retries.
    with (root / (stage + '.log')).open('xb') as log:
        process = subprocess.Popen(clean_command('_' + stage), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=900 if stage == 'build' else 30)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
            raise
    if code:
        raise RuntimeError(stage + ' failed; retained log, no automatic retry')


def build():
    root = identity.LINUX_ROOT
    prepared = identity.read(root / 'prepared.json')
    if any((root / name).exists() for name in ('build', 'install', 'build.json')):
        raise ValueError('One fresh build only')
    identity.verify_inventory(root / 'source/aimdk_msgs', prepared['source_files'])
    identity.selected_sources(root)
    if identity.sha(root.parent / identity.MC_LIBRARY) != identity.MC_SHA:
        raise ValueError('MC identity mismatch')
    tools = {name: subprocess.check_output([name, '--version'], text=True) for name in ('cmake', 'gcc', 'g++')}
    began = time.monotonic()
    result = subprocess.run(['colcon', '--log-base', str(root / 'log'), 'build',
        '--base-paths', str(root / 'source/aimdk_msgs'), '--build-base', str(root / 'build'),
        '--install-base', str(root / 'install'), '--merge-install', '--executor', 'sequential',
        '--packages-select', 'aimdk_msgs', '--cmake-args', '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release'],
        cwd=root, check=False)
    record = dict(schema=identity.SCHEMA, platform='Linux/x86_64', returncode=result.returncode,
                  prepared_sha256=identity.sha(root / 'prepared.json'), tools=tools,
                  ros_setup_sha256=identity.sha('/opt/ros/humble/setup.bash'), environment=dict(os.environ),
                  wall_seconds=time.monotonic()-began,
                  install_files=identity.inventory(root / 'install') if result.returncode == 0 else {})
    identity.write_new(root / 'build.json', record)
    if result.returncode:
        raise RuntimeError('colcon failed')


def inspect():
    root = identity.LINUX_ROOT
    binding = identity.verify_build(root, accepted=False)
    actual = identity.runtime_identity(root, 'v1.0.0')
    identity.write_new(root / 'inspection.json', dict(schema=identity.SCHEMA,
        build_sha256=binding['build_sha256'], prepared_sha256=binding['prepared_sha256'],
        mc_sha256=identity.MC_SHA, recursive_layout_verified=True, identity=actual,
        scope='Actual old Python/type-support import and recursive fields; no DDS/action proof'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'build', 'inspect', '_build', '_inspect'))
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
    elif stage.startswith('_'):
        if HERE.parent != identity.LINUX_ROOT or platform.machine() != 'x86_64':
            raise ValueError('Internal Linux stage only')
        (build if stage == '_build' else inspect)()
    else:
        linux_stage(stage)


if __name__ == '__main__':
    main()
