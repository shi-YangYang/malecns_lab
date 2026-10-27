#!/usr/bin/env python3
"""Restore cells omitted by display decimation, using original coordinates only.

No brain execution, training, network calls, or changes to the base export.
Requires the selected official SWC files when local synapse/soma data is absent.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'data/brain_visualization_v1'
RAW = ROOT / 'data/malecns_v1'
OUT = ROOT / 'data/brain_neuron_coverage_v1'
SYN = RAW / 'syn-points-male-cns-v1.0-minconf-0.5.feather'
ANN = RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
SKELS = RAW / 'skeletons-malecns-display-coverage'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_position(value):
    return value is not None and len(value) == 3 and all(v is not None and np.isfinite(v) for v in value)


def main():
    assert not (OUT / 'coverage.json').exists(), 'Preserve existing export; use a new output version'
    m = json.loads((BASE / 'manifest.json').read_text())
    ids = np.asarray(m['body_ids'], dtype=np.int64)
    assert len(ids) == 166700 and np.all(ids[1:] > ids[:-1])
    assert sha(BASE / 'owners.u32') == m['files']['owners.u32']['sha256']
    existing = np.fromfile(BASE / 'owners.u32', dtype='<u4')
    present = np.zeros(len(ids), dtype=bool)
    present[existing] = True
    missing = np.flatnonzero(~present)
    wanted = ids[missing]
    found = np.zeros(len(missing), dtype=bool)
    xyz = np.zeros((len(missing), 3), dtype=np.float64)
    sources = [None] * len(missing)
    # First original row per missing cell guarantees coverage. Spatially thinned
    # background stays unchanged; no generated/jittered coordinates are added.
    with pa.memory_map(str(SYN), 'r') as source:
        schema = ipc.open_file(source).schema
        reader = ipc.open_file(source, options=ipc.IpcReadOptions(
            included_fields=[schema.get_field_index(n) for n in ['body', 'x', 'y', 'z']]))
        for bi in range(reader.num_record_batches):
            batch = reader.get_batch(bi)
            body = batch.column(batch.schema.get_field_index('body')).to_numpy()
            index = np.searchsorted(wanted, body)
            safe = np.minimum(index, len(wanted) - 1)
            candidates = np.flatnonzero((index < len(wanted)) & (wanted[safe] == body) & ~found[safe])
            if len(candidates):
                unique, first = np.unique(index[candidates], return_index=True)
                rows = candidates[first]
                xyz[unique] = np.column_stack([batch.column(batch.schema.get_field_index(n)).to_numpy()[rows] for n in ['x', 'y', 'z']])
                for owner, row in zip(unique, rows):
                    sources[int(owner)] = dict(kind='synapse', batch=bi, row=int(row))
                found[unique] = True
            if bi % 1000 == 0:
                print(f'Original synapse rows: {bi}/{reader.num_record_batches}; restored {found.sum()}', flush=True)
    annotations = feather.read_table(ANN, columns=['bodyId', 'somaLocation'])
    somas = dict(zip(annotations['bodyId'].to_pylist(), annotations['somaLocation'].to_pylist()))
    downloads = json.loads((SKELS / 'download-manifest.json').read_text())
    skeletons = {v['body_id']: v for v in downloads['files'] if 'error' not in v}
    for i in np.flatnonzero(~found):
        body = str(wanted[i])
        if valid_position(somas.get(int(body))):
            xyz[i] = somas[int(body)]
            sources[i] = dict(kind='soma', field='somaLocation')
        else:
            entry = skeletons.get(body)
            assert entry, f'No original position available for {body}'
            path = SKELS / f'{body}.swc'
            assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
            assert entry['url'] == f'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body}.swc'
            swc = np.loadtxt(path, comments='#', ndmin=2)
            assert swc.shape[1] == 7 and len(swc) > 0 and np.isfinite(swc).all()
            # Select an existing skeleton node nearest the median, not an
            # invented centroid. Official non-mirrored SWC uses 8nm units,
            # matching the synapse table and the original display transform.
            coords = swc[:, 2:5]
            selected = int(np.argmin(np.sum((coords - np.median(coords, axis=0)) ** 2, axis=1)))
            xyz[i] = coords[selected]
            sources[i] = dict(kind='skeleton', node_id=int(swc[selected, 0]), row=selected, **entry)
        found[i] = True
    assert found.all() and all(source is not None for source in sources)
    transform = m['coordinate_transform']
    positions = ((xyz - np.asarray(transform['center'])) * transform['scale']).astype('<f4')
    assert np.isfinite(positions).all()
    assert len(np.unique(np.r_[existing, missing])) == len(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    positions.tofile(OUT / 'coverage-points.f32')
    missing.astype('<u4').tofile(OUT / 'coverage-owners.u32')
    evidence = [dict(owner=int(owner), body_id=str(ids[owner]), source=source, source_xyz=point.tolist())
                for owner, source, point in zip(missing, sources, xyz)]
    (OUT / 'source-points.json').write_text(json.dumps(evidence, separators=(',', ':')) + '\n')
    files = {name: dict(bytes=(OUT / name).stat().st_size, sha256=sha(OUT / name))
             for name in ['coverage-points.f32', 'coverage-owners.u32']}
    release = json.loads((RAW / 'download_manifest.json').read_text())
    synapse_release = next(v for v in release['files'] if Path(v['name']).name == SYN.name)
    assert SYN.stat().st_size == synapse_release['size']
    coverage = dict(schema='malecns_lab.neuron_coverage.v1', node_count=len(ids),
        point_count=len(missing), positioned_count=len(ids), base_positioned_count=int(present.sum()),
        graph_sha256=m['provenance']['graph_sha256'], base_manifest_sha256=sha(BASE / 'manifest.json'),
        body_ids_sha256=hashlib.sha256(json.dumps(m['body_ids'], separators=(',', ':')).encode()).hexdigest(),
        files=files, source_counts={kind: sum(s['kind'] == kind for s in sources) for kind in ['synapse', 'soma', 'skeleton']},
        provenance=dict(source_page=downloads['source_page'], license='CC-BY-4.0',
            coordinate_transform=transform, source_units='original MaleCNS 8nm voxel coordinates',
            synapse_release=synapse_release, synapse_verification='release manifest size and per-point source rows; no new full-file hash',
            annotation_sha256=sha(ANN), skeleton_download_manifest_sha256=sha(SKELS / 'download-manifest.json'),
            source_points_sha256=sha(OUT / 'source-points.json'), exporter_sha256=sha(Path(__file__)),
            selection='One original synapse, soma, or skeleton point per previously omitted cell; no interpolation or generated position'))
    (OUT / 'coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({k: coverage[k] for k in ['point_count', 'positioned_count', 'source_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
