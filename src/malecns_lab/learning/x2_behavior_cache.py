"""Strict read-only admission of the one preregistered frozen-feature archive.

NumPy is imported only by admission, never by preview. No graph/state is created.
Cached bytes are checked before parsing, and features are retained from those same
bytes. State assets are audited one at a time, never copied or loaded into a brain.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import zipfile

from . import x2_inference as frozen

SCHEMA = 'malecns_lab.x2_readout_cache.v1'
CACHE_RUN = 'data/x2_behavior_readout_20261023_r1'
CACHE_REGISTRATION = 'd051209beb7573fd64dfe4da9795866da2f15083cb0e0fec1a3debf9bbc7cccd'
INITIAL = 'data/x2_behavior_probe_20261023_r2/initial-checkpoint'
INITIAL_ID = '394e39ec8eed0b622712fe2ffeca20d247e1ed437a9e6945479716eb007f3959'
POLICY_ID = 'cf621475ebe520cb23b05c57b8cc4f9d24c04952e57b1009b59e26d745ba1647'
INPUT_ID = '6250f324b20a9a28498153347ca90b93b1d6c870d72c0ae53530bac4c446a2b8'
ALLOWED_CHANGED = ('experiments/x2_behavior_training.py', 'learning/x2_behavior_training.py')
ALLOWED_ADDED = ('learning/x2_behavior_cache.py',)
UNCHANGED_FUNCTIONS = ('contract', 'select_episodes', 'save_checkpoint', 'restore_checkpoint', 'loss_for',
                       '_prediction', '_gradient', '_audited_gradient_update', '_save_state', '_save_output',
                       'cached_outputs', 'hashlib_counts')
NODE_COUNT, INPUT_COUNT, FEATURE_COUNT, THETA_COUNT, PENDING_COUNT = 166700, 3377, 2628, 11752, 18
FRAME_COUNTS = {'train': 416, 'validation': 402}
PARAMETER_SHAPES = {'brain.theta': (THETA_COUNT,), 'behavior_head.weight': (5, FEATURE_COUNT),
    'behavior_head.bias': (5,), 'phase_head.weight': (3, FEATURE_COUNT), 'phase_head.bias': (3,),
    'parameter_head.weight': (1, FEATURE_COUNT), 'parameter_head.bias': (1,)}


def validate_request(configuration, checkpoint_path, cache_path, root):
    frozen.require(configuration['mode'] == 'readout', 'Feature cache is readout-only')
    frozen.require(all(configuration[key] == value for key, value in dict(seed=2401, max_updates=200,
        train_episodes=50, validation_episodes=50, wall_seconds=600, initialization='migrated',
        spike_backward_gain=.1).items()), 'Cache recovery requires fixed 50/50 episodes, 200 updates and 600 seconds')
    root = Path(root).resolve()
    for value, relative in ((cache_path, CACHE_RUN), (checkpoint_path, INITIAL)):
        frozen.require(value is not None, 'Cache recovery requires the original initial checkpoint')
        given = Path(value).absolute()
        frozen.require(given == root / relative and given.resolve() == given and given.is_dir(), 'Only the registered unlinked cache/initial checkpoint is admitted')


def _read(base, relative, *, limit=32*1024*1024):
    parts = Path(relative).parts
    frozen.require(parts and not Path(relative).is_absolute() and all(p not in ('..', '.') for p in parts)
                   and not any('locked' in p for p in parts), 'Cache path or locked-test access rejected')
    base = Path(base)
    frozen.require(base.absolute() == base.resolve(), 'Linked cache root rejected')
    target = base
    for part in parts:
        target /= part
        frozen.require(not target.is_symlink(), 'Linked cache file/directory rejected')
    with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        info = os.fstat(stream.fileno())
        frozen.require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'Cache file exceeds bound')
        content = stream.read(limit+1)
        frozen.require(len(content) <= limit, 'Growing cache file exceeds bound')
    return content


def _json(content):
    def unique(items):
        value = {}
        for key, item in items:
            frozen.require(key not in value, 'Duplicate cache JSON key')
            value[key] = item
        return value
    return json.loads(content, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _array_identity(value):
    return {'shape': list(value.shape), 'dtype': value.dtype.str, 'sha256': _sha(value.tobytes(order='C'))}


def _arrays(content, specs, declarations=None):
    import numpy as np
    # ZIP bounds are checked before decompression/np.load. Duplicate keys and
    # object arrays cannot be hidden behind matching outer file hashes.
    with zipfile.ZipFile(io.BytesIO(content)) as zipped:
        names = zipped.namelist()
        frozen.require(len(names) == len(set(names)) and set(names) == {key+'.npy' for key in specs}, 'Cache array inventory differs')
        for item in zipped.infolist():
            shape, dtype = specs[item.filename[:-4]]
            frozen.require(item.file_size <= int(np.prod(shape))*np.dtype(dtype).itemsize+1024, 'Cache array decompressed bound exceeded')
    result = {}
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        for key, (shape, dtype) in specs.items():
            value = archive[key]
            frozen.require(value.shape == tuple(shape) and value.dtype.str == np.dtype(dtype).str and np.isfinite(value).all(), 'Cache array shape/dtype/finiteness differs')
            if declarations is not None:
                frozen.require(_array_identity(value) == declarations[key], 'Cache array declaration differs')
            result[key] = value
    return result


def _function_bodies(content):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(content))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in UNCHANGED_FUNCTIONS}


def readout_update_ast(content):
    matches = []
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, ast.For) and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                and child.func.id == 'phase' and child.args and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == 'readout_update' for child in ast.walk(node)):
            matches.append(node)
    frozen.require(len(matches) == 1, 'Unique fixed readout update loop required')
    return matches[0]


def _source_changes(root, old, current, read):
    frozen.require(set(old).issubset(current) and set(current)-set(old) <= set(ALLOWED_ADDED), 'Unregistered added or removed cache source')
    differences = {}
    for name, expected in old.items():
        original = read('sources/malecns_lab/'+name)
        frozen.require(_sha(original) == expected, 'Original cache source snapshot differs')
        actual = _read(root / 'src/malecns_lab', name)
        frozen.require(_sha(actual) == current[name], 'Recovery source changed before admission')
        if expected != current[name]:
            frozen.require(name in ALLOWED_CHANGED, 'Numerical/encoder/data source differs from cache')
            differences[name] = {'cache_sha256': expected, 'recovery_sha256': current[name]}
            if name == 'learning/x2_behavior_training.py':
                before, after = _function_bodies(original), _function_bodies(actual)
                frozen.require(set(before) == set(UNCHANGED_FUNCTIONS) and before == after, 'Training loss/selection/numeric helper changed')
                frozen.require(ast.dump(readout_update_ast(original), include_attributes=False) == ast.dump(readout_update_ast(actual), include_attributes=False), 'Readout update/validation/earliest-selection loop changed')
    for name in set(current)-set(old):
        frozen.require(_sha(_read(root / 'src/malecns_lab', name)) == current[name], 'New cache source changed')
        differences[name] = {'cache_sha256': None, 'recovery_sha256': current[name]}
    return differences


def audit_cache(root, cache_path, *, configuration, checkpoint, sources, runtime, expected=None,
                keep_features=False, progress=lambda: None):
    """Parent seals full inventory; worker repeats audit and retains checked features."""
    import numpy as np
    from .x2_behavior_training import select_episodes
    from ..control.x2_behavior import Observation, Decision
    from ..senses.x2_behavior import channel_values
    root, cache_path = Path(root).resolve(), Path(cache_path).absolute()
    validate_request(configuration, checkpoint['path'], cache_path, root)
    files, dataset_bytes = {}, {}
    def read(name):
        progress()
        content = _read(cache_path, name)
        digest = _sha(content)
        if expected is not None:
            frozen.require(expected['files'].get(name) == digest, 'Sealed cache file changed after registration')
        files[name] = digest
        return content
    old = _json(read('registration.json'))
    frozen.bound(old, 'registration_id')
    frozen.require(old['registration_id'] == CACHE_REGISTRATION and old['output'] == str(cache_path)
                   and old['root'] == str(root), 'Cache registration is not the fixed failed run')
    frozen.require(old['checkpoint'] == checkpoint and checkpoint['checkpoint_id'] == INITIAL_ID, 'Cache initial checkpoint differs')
    frozen.require({k:v for k,v in old['configuration'].items() if k != 'wall_seconds'} ==
                   {k:v for k,v in configuration.items() if k != 'wall_seconds'} and old['configuration']['wall_seconds'] == 5400,
                   'Frozen-feature experiment configuration differs')
    frozen.require(old['runtime'] == runtime, 'Cache numeric runtime differs')
    outcome = _json(read('summary.json'))
    frozen.require(outcome['status'] == 'failed' and outcome['supervisor']['reason'] == 'wall_limit'
                   and not (cache_path/'worker_result.json').exists(), 'Cache must retain its failed wall-limit result')
    changes = _source_changes(root, old['source_sha256'], sources, read)
    policy = _json(read('policy.json')); frozen.bound(policy, 'policy_id')
    mapping = _json(read('input_map.json')); frozen.bound(mapping, 'input_map_id')
    frozen.require(policy['policy_id'] == POLICY_ID and mapping['input_map_id'] == INPUT_ID
                   and policy['input_map'] == mapping and policy['features']['width'] == FEATURE_COUNT
                   and policy['frame_steps'] == 500 and policy['precondition_steps'] == 600, 'Cache policy/input identity differs')
    initial_root = root / INITIAL
    initial_bytes = {name: _read(initial_root, name) for name in ('checkpoint.json','arrays.npz')}
    initial = _json(initial_bytes['checkpoint.json']); frozen.bound(initial, 'checkpoint_id')
    frozen.require(initial['checkpoint_id'] == INITIAL_ID and initial['policy'] == policy
                   and initial['context']['optimizer_steps'] == 0 and initial['context']['role'] == 'initialization'
                   and _sha(initial_bytes['checkpoint.json']) == checkpoint['metadata_sha256']
                   and _sha(initial_bytes['arrays.npz']) == checkpoint['arrays_sha256'] == initial['arrays_sha256'], 'Original initial checkpoint changed')
    initial_specs = {'parameter.'+key: (shape, '<f4') for key, shape in PARAMETER_SHAPES.items()}
    frozen.require(set(initial['arrays']) == set(initial_specs), 'Initial must contain all seven parameter groups and no optimizer')
    _arrays(initial_bytes['arrays.npz'], initial_specs, initial['arrays'])
    dataset_bytes['manifest.json'] = read('dataset/manifest.json')
    dataset = _json(dataset_bytes['manifest.json'])
    frozen.require(dataset == old['dataset'], 'Cache dataset manifest differs')
    frozen.bound(dataset['identity'], 'dataset_id')
    selected_record = _json(read('selected_episodes.json'))
    frozen.require(set(selected_record) == {'train','validation','locked_test_opened'} and selected_record['locked_test_opened'] is False, 'Cache selection opened locked test')
    episodes, features, counts, state_count, last_time, initial_arrays = {}, {}, {}, 0, -1., None
    known_episodes, families = set(), set()
    shapes = {'v': ((NODE_COUNT,),'<f4'), 'g': ((NODE_COUNT,),'<f4'), 'refractory_until': ((NODE_COUNT,),'<i8'),
              'phase': ((INPUT_COUNT,),'<f4'), 'event_counts': ((INPUT_COUNT,),'<i8'),
              **{f'pending.{i:02d}': ((NODE_COUNT,),'<f4') for i in range(PENDING_COUNT)}}
    frame_specs = {'encoded_input': ((INPUT_COUNT,),'<f4'), 'features_mv': ((FEATURE_COUNT,),'<f4'),
                   'behavior_logits': ((5,),'<f4'), 'phase_logits': ((3,),'<f4'), 'parameter_raw': ((1,),'<f4')}
    for split in ('train','validation'):
        file = split+'.json'
        frozen.require(dataset['files'][split]['path'] == file, 'Unexpected split path')
        dataset_bytes[file] = read('dataset/'+file)
        all_episodes = _json(dataset_bytes[file])
        frozen.require(_sha(dataset_bytes[file]) == dataset['files'][split]['sha256']
                       and frozen.canonical_hash(all_episodes) == dataset['identity']['split_sha256'][split], 'Cache dataset split differs')
        episodes[split] = select_episodes(all_episodes, 50)
        frozen.require([e['episode_id'] for e in episodes[split]] == selected_record[split], 'Cache episode selection/order differs')
        folder = cache_path / (split+'-features')
        frozen.require(not folder.is_symlink() and {p.name for p in folder.iterdir()} == set(selected_record[split]), 'Missing/extra cached episode')
        split_families = {e['family_id'] for e in episodes[split]}
        frozen.require(not families.intersection(split_families), 'Cache split family leakage'); families.update(split_families)
        features[split], counts[split] = {}, 0
        for episode in episodes[split]:
            eid, rows = episode['episode_id'], episode['frames']
            frozen.require(eid not in known_episodes and len(eid) == 64 and all(c in '0123456789abcdef' for c in eid)
                           and episode['split'] == split and len(rows) >= 2 and episode['sequence_sha256'] == frozen.canonical_hash(rows), 'Cache episode identity/sequence differs')
            known_episodes.add(eid)
            relative = split+'-features/'+eid
            names = {f'frame-{i:04d}.{suffix}' for i in range(len(rows)) for suffix in ('json','npz')}
            names |= {f'state-{i:04d}.{suffix}' for i in range(len(rows)+1) for suffix in ('json','npz')}
            frozen.require(not (cache_path/relative).is_symlink() and {p.name for p in (cache_path/relative).iterdir()} == names, 'Missing/extra cache frame or state')
            def state(index):
                nonlocal state_count, initial_arrays
                metadata = _json(read(relative+f'/state-{index:04d}.json'))
                payload = {k:v for k,v in metadata.items() if k not in ('state_id','file_sha256')}
                frozen.require(metadata['state_id'] == frozen.canonical_hash(payload)
                    and metadata['schema'] == 'malecns_lab.x2_behavior_state.v1' and metadata['policy_id'] == POLICY_ID
                    and metadata['session_id'] == eid and metadata['frame_index'] == index
                    and metadata['observation_seq'] == index-1 and metadata['neural_step'] == 600+index*500
                    and set(metadata['arrays']) == set(shapes), 'Cache state identity/sequence differs')
                content = read(relative+f'/state-{index:04d}.npz')
                frozen.require(_sha(content) == metadata['file_sha256'], 'Cache state NPZ hash differs')
                _arrays(content, shapes, metadata['arrays'])
                if index == 0:
                    if initial_arrays is None: initial_arrays = metadata['arrays']
                    frozen.require(metadata['arrays'] == initial_arrays, 'Episode prewarm state differs')
                state_count += 1
                return metadata['state_id']
            before = state(0); features[split][eid] = []
            for index, row in enumerate(rows):
                observation = Observation.from_dict(row['observation'])
                frozen.require(row['frame_index'] == index and observation.run_id == eid and observation.session_id == eid
                    and observation.observation_seq == index and str(observation.task) == episode['task'], 'Dataset observation sequence differs')
                metadata = _json(read(relative+f'/frame-{index:04d}.json'))
                frozen.require(metadata['observation'] == row['observation'], 'Cached actual observation differs')
                at = metadata['host_monotonic']
                frozen.require(type(at) in (int,float) and np.isfinite(at) and at > last_time, 'Cached host order differs'); last_time = at
                content = read(relative+f'/frame-{index:04d}.npz')
                frozen.require(_sha(content) == metadata['arrays_sha256'], 'Cache frame NPZ hash differs')
                arrays = _arrays(content, frame_specs)
                values = tuple(channel_values(observation).values())
                encoded_input = np.asarray([values[i] for i in mapping['channel_indices']], dtype=np.float32)
                frozen.require(np.array_equal(arrays['encoded_input'], encoded_input), 'Encoded cached observation differs')
                decision = Decision.from_dict(metadata['decision']); after = state(index+1)
                frozen.require(decision.run_id == eid and decision.session_id == eid and decision.request_id == observation.request_id
                    and decision.observation_seq == index and decision.decision_seq == index and decision.checkpoint_id == INITIAL_ID
                    and decision.policy_id == POLICY_ID and decision.input_map_id == INPUT_ID
                    and decision.state_before_id == before and decision.state_after_id == after
                    and decision.neural_start_step == 600+index*500 and decision.neural_end_step == 1100+index*500,
                    'Cache decision/state chain differs')
                for key in ('behavior_logits','phase_logits','parameter_raw'):
                    frozen.require(np.array_equal(arrays[key], np.asarray(getattr(decision,key), dtype=np.float32)), 'Cached head evidence differs')
                if keep_features: features[split][eid].append(arrays['features_mv'].copy())
                before = after; counts[split] += 1
        frozen.require(counts[split] == FRAME_COUNTS[split], 'Incomplete cached split frame count')
    frozen.require(state_count == sum(FRAME_COUNTS.values())+100, 'Incomplete cached state count')
    manifest = {'schema': SCHEMA, 'cache_root': str(cache_path), 'source_registration_id': CACHE_REGISTRATION,
        'source_registration_sha256': files['registration.json'], 'initial_checkpoint': checkpoint,
        'policy_id': POLICY_ID, 'input_map_id': INPUT_ID, 'dataset_id': dataset['identity']['dataset_id'],
        'selected_episodes': selected_record, 'frame_counts': counts, 'state_count': state_count,
        'initial_state_arrays': initial_arrays, 'files': dict(sorted(files.items())),
        'source_changes': changes, 'allowed_changed_sources': list(ALLOWED_CHANGED), 'allowed_added_sources': list(ALLOWED_ADDED),
        'cache_runtime': old['runtime'], 'recovery_runtime': runtime, 'locked_test_opened': False,
        'state_use': 'audit only; never copied or installed in a brain', 'feature_use': 'features_mv only; no old logits/labels as head inputs'}
    manifest['cache_id'] = frozen.canonical_hash(manifest)
    if expected is not None: frozen.require(manifest == expected, 'Cache manifest changed during worker revalidation')
    return manifest, features, dataset_bytes


def forbid_neural_calls(policy_class, brain_class, result):
    """Install before construction, so any reset/forward/warmup is counted and fails."""
    counts, original = {}, []
    for cls, names in ((policy_class, ('forward','initial_state','_step','warmup')), (brain_class, ('step','reset','forward','warmup'))):
        for name in names:
            if not hasattr(cls, name): continue
            key = cls.__name__+'.'+name; counts[key] = 0
            saved = getattr(cls, name); own = name in cls.__dict__; original.append((cls,name,saved,own))
            def reject(*args, _key=key, **kwargs):
                counts[_key] += 1
                raise ValueError('Cached readout forbids neural call: '+_key)
            setattr(cls,name,reject)
    result['neural_calls'] = counts
    def restore():
        for cls,name,saved,own in reversed(original):
            if own: setattr(cls,name,saved)
            else: delattr(cls,name)
    return restore
