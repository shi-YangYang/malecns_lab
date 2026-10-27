#!/usr/bin/env python3
"""Read back every supplemental point from the independent original source."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/brain_neuron_coverage_v1'
BASE = ROOT / 'data/brain_visualization_v1'
RAW = ROOT / 'data/malecns_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m = json.loads((OUT / 'coverage.json').read_text())
    base = json.loads((BASE / 'manifest.json').read_text())
    assert sha(BASE / 'manifest.json') == m['base_manifest_sha256']
    assert sha(OUT / 'source-points.json') == m['provenance']['source_points_sha256']
    for name, identity in m['files'].items():
        assert sha(OUT / name) == identity['sha256'] and (OUT / name).stat().st_size == identity['bytes']
    points = np.fromfile(OUT / 'coverage-points.f32', dtype='<f4').reshape(-1, 3)
    owners = np.fromfile(OUT / 'coverage-owners.u32', dtype='<u4')
    original = np.fromfile(BASE / 'owners.u32', dtype='<u4')
    assert len(np.unique(owners)) == len(owners) == m['point_count']
    assert not np.intersect1d(original, owners).size
    assert np.array_equal(np.unique(np.r_[original, owners]), np.arange(m['node_count']))
    evidence = json.loads((OUT / 'source-points.json').read_text())
    batches, expected = {}, np.zeros_like(points, dtype=np.float64)
    somas = feather.read_table(RAW / 'body-annotations-male-cns-v1.0-minconf-0.5.feather', columns=['bodyId', 'somaLocation'])
    somas = dict(zip(somas['bodyId'].to_pylist(), somas['somaLocation'].to_pylist()))
    for i, item in enumerate(evidence):
        assert item['owner'] == int(owners[i]) and item['body_id'] == base['body_ids'][owners[i]]
        source = item['source']
        if source['kind'] == 'synapse':
            batches.setdefault(source['batch'], []).append(i)
        elif source['kind'] == 'soma':
            expected[i] = somas[int(item['body_id'])]
        else:
            assert source['kind'] == 'skeleton'
            path = RAW / 'skeletons-malecns-display-coverage' / f"{item['body_id']}.swc"
            assert sha(path) == source['sha256']
            row = np.loadtxt(path, comments='#', ndmin=2)[source['row']]
            assert int(row[0]) == source['node_id']
            expected[i] = row[2:5]
    with pa.memory_map(str(RAW / 'syn-points-male-cns-v1.0-minconf-0.5.feather'), 'r') as source:
        schema = ipc.open_file(source).schema
        reader = ipc.open_file(source, options=ipc.IpcReadOptions(included_fields=[schema.get_field_index(n) for n in ['body', 'x', 'y', 'z']]))
        for bi, selected in batches.items():
            batch = reader.get_batch(bi)
            rows = [evidence[i]['source']['row'] for i in selected]
            bodies = batch.column(batch.schema.get_field_index('body')).to_numpy()[rows]
            assert list(map(str, bodies)) == [evidence[i]['body_id'] for i in selected]
            expected[selected] = np.column_stack([batch.column(batch.schema.get_field_index(n)).to_numpy()[rows] for n in ['x', 'y', 'z']])
    transform = base['coordinate_transform']
    assert np.array_equal(expected, [item['source_xyz'] for item in evidence])
    assert np.array_equal(((expected - np.asarray(transform['center'])) * transform['scale']).astype('<f4'), points)
    print(json.dumps(dict(result='PASS', original_cells=len(np.unique(original)), restored_cells=len(owners),
        displayed_cells=m['node_count'], source_counts=m['source_counts'], verified_original_positions=len(points))))


if __name__ == '__main__':
    main()
