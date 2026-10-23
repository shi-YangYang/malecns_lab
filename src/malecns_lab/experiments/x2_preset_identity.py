"""Pinned official provenance and runtime identity for the isolated preset client.

No ROS imports, builds or process starts on import. Paths are fixed by the owned
simulation deployment, never received from the request IPC channel.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

DIRECTORY = 'aimdk-preset-v1_0_0'
LINUX_ROOT = Path('/workspace') / DIRECTORY
ARCHIVE = 'aimdk-sdk-v1_0_0-aimdk-aarch64-a424add7-artifacts.zip'
ZIP_SHA = '5bbcf724d54fb28f153db0d272f9acb7906bb1d2cac7dd7ccdc699a5c7eeab35'
ZIP_ROOT = 'aimdk-aarch64-a424add7-artifacts/src/aimdk_msgs/'
MC_LIBRARY = 'mc/lib/libaimdk_msgs__rosidl_typesupport_introspection_cpp.so'
MC_SHA = '935d4226013617a7aee2cdb525e31b6c46e44d608bf88f99eb61467b746cb931'
EVIDENCE = {
    'official-source-sha256.json': '1b018b76641c8f28f1dc33a96d5b883916d87592fd1d301a1c514f5914175608',
    'selected-sources.json': '3113d897dc55dd928e606d27f1b327c564592e07ea13f60a4bb706c44abd3b74',
    'compatibility-inspection.json': '8af884b74726ee9f2fbaa699566ab9ef65cdec1837fb9cc19d78642a93f340a8',
}
SCHEMA = 'malecns.x2.preset-compat.v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def inventory(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Artifact symlink escapes its prefix')
            result[str(path.relative_to(root))] = sha(path)
    if not result:
        raise ValueError('Empty artifact inventory')
    return result


def verify_inventory(root, expected, *, exact=True):
    if not isinstance(expected, dict) or not expected:
        raise ValueError('Missing inventory')
    for relative, digest in expected.items():
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()) or sha(target) != digest:
            raise ValueError('Identity mismatch: ' + relative)
    if exact and inventory(root) != expected:
        raise ValueError('Artifact inventory changed')


def selected_sources(root):
    selected = read(root / 'evidence/selected-sources.json')
    result = {}
    for key, value in selected.items():
        old = key.startswith('v1.0.0|')
        marker = 'src/aimdk_msgs/' if old else 'data/x2_sim/packages/'
        relative = value['path'].split(marker, 1)[1]
        path = root / 'source/aimdk_msgs' / relative if old else root.parent / relative
        if sha(path) != value['sha256']:
            raise ValueError('Official selected source changed: ' + key)
        result[key] = path
    return result


def verify_build(root, *, accepted=True):
    root = Path(root)
    for name, digest in EVIDENCE.items():
        if sha(root / 'evidence' / name) != digest:
            raise ValueError('Compatibility evidence changed: ' + name)
    prepared = read(root / 'prepared.json')
    if prepared['schema'] != SCHEMA or prepared['zip_sha256'] != ZIP_SHA:
        raise ValueError('Unrecognized official source')
    # CMake may add prebuilt_x86_64 to the disposable source copy at install.
    verify_inventory(root / 'source/aimdk_msgs', prepared['source_files'], exact=False)
    actual_source = inventory(root / 'source/aimdk_msgs')
    if any(k not in prepared['source_files'] and not k.startswith('prebuilt_x86_64/') for k in actual_source):
        raise ValueError('Unexpected source addition')
    selected_sources(root)
    if sha(root.parent / MC_LIBRARY) != MC_SHA:
        raise ValueError('MC library changed')
    build = read(root / 'build.json')
    if (build['schema'] != SCHEMA or build['prepared_sha256'] != sha(root / 'prepared.json')
            or build['platform'] != 'Linux/x86_64' or build['returncode'] != 0):
        raise ValueError('Missing successful isolated build')
    verify_inventory(root / 'install', build['install_files'])
    verify_inventory(root.parent / 'aimdk/install/aimdk_msgs', prepared['new_install_files'])
    if accepted:
        inspection = read(root / 'inspection.json')
        if (inspection['schema'] != SCHEMA or inspection['build_sha256'] != sha(root / 'build.json')
                or inspection['prepared_sha256'] != sha(root / 'prepared.json')
                or inspection['mc_sha256'] != MC_SHA or not inspection['recursive_layout_verified']):
            raise ValueError('Missing matching runtime inspection')
    return dict(schema=SCHEMA, zip_sha256=ZIP_SHA, mc_sha256=MC_SHA,
                prepared_sha256=sha(root / 'prepared.json'), build_sha256=sha(root / 'build.json'),
                inspection_sha256=sha(root / 'inspection.json') if accepted else None)


def _type(value):
    for old, new in (('boolean', 'bool'), ('double', 'float64'), ('float', 'float32')):
        value = re.sub(r'\b' + old + r'\b', new, value)
    for package in ('aimdk_msgs', 'builtin_interfaces'):
        value = re.sub(r'\b' + package + r'/(?!msg/)', package + '/msg/', value)
    return value


def source_fields(path, name):
    source = path.read_text()
    if name.endswith('_Request'):
        source = source.split('---')[0]
    elif name.endswith('_Response'):
        source = source.split('---')[1]
    result = {}
    for raw in source.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or '=' in line:
            continue
        field_type, field_name, *_ = line.split()
        if '/' not in field_type and field_type[:1].isupper():
            field_type = 'aimdk_msgs/' + field_type
        # This selected closure contains only scalar and unbounded sequences.
        if field_type.endswith('[]'):
            field_type = 'sequence<' + field_type[:-2] + '>'
        result[field_name] = _type(field_type)
    return result


def runtime_identity(root, version):
    """Verify generated recursive fields, loaded Python modules and actual .so maps."""
    from rosidl_generator_py import import_type_support
    prefix = root / 'install' if version == 'v1.0.0' else root.parent / 'aimdk/install/aimdk_msgs'
    prefix = prefix.resolve()
    selected = selected_sources(root)
    fields = {}
    for key, source in selected.items():
        v, kind_name = key.split('|')
        if v != version:
            continue
        kind, name = kind_name.split('/')
        module = importlib.import_module('aimdk_msgs.' + kind)
        if kind == 'srv':
            service, part = name.rsplit('_', 1)
            cls = getattr(getattr(module, service), part)
        else:
            cls = getattr(module, name)
        actual = {k: _type(v) for k, v in cls.get_fields_and_field_types().items()}
        expected = source_fields(source, name)
        if list(actual.items()) != list(expected.items()):
            raise ValueError('Runtime recursive layout mismatch: ' + key)
        fields[key] = actual
    time_type = importlib.import_module('builtin_interfaces.msg').Time
    if time_type.get_fields_and_field_types() != {'sec': 'int32', 'nanosec': 'uint32'}:
        raise ValueError('Unexpected ROS Time layout')
    import_type_support('aimdk_msgs')
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'aimdk_msgs' or name.startswith('aimdk_msgs.'):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(prefix):
                raise ValueError('Mixed aimdk_msgs Python prefixes: ' + str(path))
            modules[str(path.relative_to(prefix))] = sha(path)
    libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if 'aimdk_msgs' in path and '.so' in path:
            actual = Path(path).resolve()
            if not actual.is_relative_to(prefix):
                raise ValueError('Mixed aimdk_msgs library prefixes: ' + str(actual))
            libraries[str(actual.relative_to(prefix))] = sha(actual)
    if not modules or not libraries:
        raise ValueError('No actual message type support loaded')
    return dict(version=version, prefix=str(prefix), fields=fields, modules=modules, libraries=libraries)
