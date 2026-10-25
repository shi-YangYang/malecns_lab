"""CPU-only execution-contract derivation; no training, brain, or device startup."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from ..control.x2_behavior import digest, require
from ..control.x2_interaction import interaction_profile, fixed_advance_interaction_profile
from ..control.x2_skill_profile import fixed_advance_skill_profile

SCHEMA = "malecns_lab.x2_sim_execution_policy.v1"
SOURCE_SCHEMA = "malecns_lab.x2_interaction_paired_policy.v1"
SOURCE = {
    "path": "data/x2_behavior_short_trial_interaction_paired_20261024_r1/checkpoint",
    "checkpoint_id": "e224de705e2406f8407e7831bba0693785a886fd073c20ac31d672c6b93f11e0",
    "policy_id": "8c88faa96bdc5af04a98d4a1ba3c56acc4dac5472b80b5790dd4b23bfacb5c45",
    "metadata_sha256": "b9982448433cf21e6bd223a9de5fe216b9540fb7fc881f7d4b46b15a06ff5ab8",
    "arrays_sha256": "c2e7f735a72ee38c26eba8189be4ee32495972cac41e35eb849f1bff9a500ab4",
}
PARAMETERS = ("brain.theta", "behavior_head.weight", "behavior_head.bias",
              "phase_head.weight", "phase_head.bias", "parameter_head.weight", "parameter_head.bias")
SOURCES = ("control/x2_skill_profile.py", "control/x2_interaction.py", "control/x2_skills.py",
           "learning/x2_sim_execution_policy.py", "learning/x2_behavior_session.py",
           "runtime/x2/behavior.py", "experiments/x2_behavior_bridge.py")


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_checkpoint(root):
    from .x2_behavior_training import inspect_checkpoint
    root = Path(root).resolve()
    path = root / SOURCE['path']
    require(path.resolve(strict=True) == path.absolute()
            and all(not (path/name).is_symlink() for name in ('checkpoint.json', 'arrays.npz')),
            'Execution source must not use symlinks')
    require(sha256(path/'checkpoint.json') == SOURCE['metadata_sha256']
            and sha256(path/'arrays.npz') == SOURCE['arrays_sha256'], 'Execution source assets changed')
    metadata = inspect_checkpoint(path)
    require(metadata['checkpoint_id'] == SOURCE['checkpoint_id']
            and metadata['policy']['policy_id'] == SOURCE['policy_id']
            and metadata['policy']['schema'] == SOURCE_SCHEMA
            and metadata['policy']['interaction_profile'] == interaction_profile(),
            'Execution source identity differs')
    return path, metadata


def execution_policy_identity(source_identity):
    require(source_identity['schema'] == SOURCE_SCHEMA and source_identity['policy_id'] == SOURCE['policy_id']
            and source_identity['interaction_profile'] == interaction_profile(), 'Fixed paired source policy required')
    require(digest({k:v for k,v in source_identity.items() if k != 'policy_id'}) == SOURCE['policy_id'],
            'Execution source policy digest differs')
    payload = deepcopy(source_identity)
    payload.pop('policy_id')
    payload.update(schema=SCHEMA, execution_source=deepcopy(SOURCE),
        interaction_profile=fixed_advance_interaction_profile(), execution_profile=fixed_advance_skill_profile(),
        execution_adaptation={"operation": "execution_contract_derivation", "training_updates": 0,
            "theta_updates": 0, "head_updates": 0, "parameters": "byte-identical source arrays.npz",
            "meaning": "brain selects behavior/phase; fixed SIM ADVANCE velocity is not learned speed",
            "source_limitations": "source multi-behavior qualification failed; no neural qualification inherited"},
        execution_implementation_sha256=sha256(__file__))
    return {**payload, 'policy_id': digest(payload)}


def bind_execution_policy(policy):
    """Call only after the unchanged loader has strictly restored the source."""
    identity = execution_policy_identity(policy.identity)
    policy._identity, policy.policy_id = identity, identity['policy_id']
    return policy


def validate_execution_checkpoint(root, path, metadata=None):
    """Check the entire disk identity and unchanged parameter archive before loading."""
    from .x2_behavior_training import inspect_checkpoint
    path = Path(path)
    require(path.resolve(strict=True) == path.absolute()
            and all(not (path/name).is_symlink() for name in ('checkpoint.json', 'arrays.npz')),
            'Execution candidate must not use symlinks')
    full = inspect_checkpoint(path)
    source_path, original = source_checkpoint(root)
    require(full['policy'] == execution_policy_identity(original['policy']), 'Execution policy identity differs')
    require(full['arrays_sha256'] == original['arrays_sha256'] == SOURCE['arrays_sha256']
            and full['arrays'] == original['arrays'], 'Execution parameters differ from source')
    require(full['context'] == derivation_context(), 'Execution derivation context differs')
    if metadata is not None:
        from .x2_behavior_session import checkpoint_registration_metadata
        require(checkpoint_registration_metadata(full) == checkpoint_registration_metadata(metadata),
                'Registered execution candidate differs')
    return full, source_path, original


def derivation_context():
    return dict(operation='execution_contract_derivation', execution_source=deepcopy(SOURCE),
        training_updates=0, optimizer_steps=0, theta_updates=0, head_updates=0,
        qualification_only=True, simulation_only=True, neural_qualification='not_evaluated')


def derive_candidate(root, output):
    """Seal new metadata and copy the original bytes, without constructing a model."""
    import numpy as np
    from .x2_behavior_training import CHECKPOINT_SCHEMA
    root = Path(root).resolve()
    source_path, original = source_checkpoint(root)
    # Authenticate array declarations as well as the pinned complete archive.
    with np.load(source_path/'arrays.npz', allow_pickle=False) as archive:
        require(set(archive.files) == {'parameter.'+name for name in PARAMETERS}
                and len(archive.files) == len(PARAMETERS), 'Source parameter inventory differs')
        for name in archive.files:
            value = archive[name]
            identity = dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
            require(identity == original['arrays'][name] and value.dtype == np.float32
                    and np.isfinite(value).all(), 'Source parameter declaration differs')
    profile = fixed_advance_skill_profile()
    evidence = profile['advance_execution']['evidence']
    run = root/'data/x2_sim/agibot/data/var'/evidence['run_id']
    require(all(sha256(run/name) == expected for name,expected in evidence['sha256'].items()),
            'Fixed ADVANCE qualification evidence changed')
    output = Path(output)
    output = output if output.is_absolute() else root/output
    require(output.absolute().parent == root/'data' and output.name.startswith('x2_behavior_sim_execution_')
            and not output.exists() and not output.is_symlink()
            and output.resolve() == output.absolute(), 'New top-level execution candidate directory required')
    metadata = dict(schema=CHECKPOINT_SCHEMA, policy=execution_policy_identity(original['policy']),
                    context=derivation_context(), arrays=deepcopy(original['arrays']), arrays_sha256=original['arrays_sha256'])
    metadata['checkpoint_id'] = digest(metadata)
    output.mkdir()
    with (source_path/'arrays.npz').open('rb') as src, (output/'arrays.npz').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    (output/'checkpoint.json').write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+'\n')
    # Keep implementation provenance beside the derived research asset.
    sources = {}
    for name in SOURCES:
        source = root/'src/malecns_lab'/name
        target = output/'sources/malecns_lab'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[name] = sha256(target)
    (output/'sources.json').write_text(json.dumps(sources, indent=2, sort_keys=True)+'\n')
    validate_execution_checkpoint(root, output, metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--derive', type=Path, help='new data/x2_behavior_sim_execution_* directory')
    args = parser.parse_args(argv)
    if args.derive is None:
        print(json.dumps(dict(mode='preview', source=SOURCE, profile=fixed_advance_skill_profile(),
                              brain_loaded=False, training_updates=0)))
        return
    metadata = derive_candidate(args.root, args.derive)
    print(json.dumps(dict(checkpoint_id=metadata['checkpoint_id'], policy_id=metadata['policy']['policy_id'],
        arrays_sha256=metadata['arrays_sha256'], operation='execution_contract_derivation', training_updates=0)))


if __name__ == '__main__':
    main()
