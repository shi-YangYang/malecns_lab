#!/usr/bin/env python3
"""Read-only source audit of display exports; no model imports or execution."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/brain_visualization_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m = json.loads((OUT/'manifest.json').read_text())
    p = m['provenance']
    for name, identity in m['files'].items():
        assert (OUT/name).stat().st_size == identity['bytes']
        assert sha(OUT/name) == identity['sha256']
    assert sha(ROOT/p['events_file']) == p['events_sha256']
    assert sha(ROOT/p['graph_file']) == p['graph_file_sha256']
    assert sha(Path(__file__).with_name('export-brain-structure.py')) == p['exporter_sha256']
    run = (ROOT/p['events_file']).parent
    registration = json.loads((run/'behavior_registration.json').read_text())
    policy = registration['checkpoint_metadata']['policy']
    source = 'data/x2_model_redesign_20261024_r1/neural_features_v2.py'
    assert sha(ROOT/source) == p['readout_source_sha256'] == policy['source_sha256'][source]
    protocol_path = 'data/x2_model_redesign_20261024_r1/request_prefix_cache_v1/protocol.json'
    assert sha(ROOT/protocol_path) == policy['artifacts'][protocol_path]
    protocol = json.loads((ROOT/protocol_path).read_text())
    assert protocol['engine_id'] == policy['numeric_engine_id']
    with np.load(ROOT/p['graph_file'], allow_pickle=False) as graph:
        assert json.loads(str(graph['meta']))['graph_sha256'] == p['graph_sha256']
        expected_readout = np.flatnonzero(np.isin(graph['types'], ['L1', 'L2']))
        assert expected_readout.tolist() == m['readout_indices']
        assert [str(v) for v in graph['body_ids']] == m['body_ids']
    # Verify all exported activity values, identities and interval lengths
    # against the original (not regenerated) runtime decision records.
    original = {}
    for line in (ROOT/p['events_file']).read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'raw_decision':
            original[event['decision']['decision_seq']] = event
    for clip in m['clips']:
        event = original[clip['decision']['decision_seq']]
        d = event['decision']
        assert clip['decision'] == d
        assert d['policy_id'] == policy['policy_id']
        assert d['checkpoint_id'] == registration['checkpoint_metadata']['checkpoint_id']
        assert d['neural_end_step']-d['neural_start_step'] == 500
        assert np.array_equal(np.asarray(clip['currents'], dtype=np.float32).ravel(), np.asarray(event['features_mv'], dtype=np.float32))
    comparison = np.asarray([c['currents'] for c in m['clips'] if c['decision']['decision_seq'] in m['comparison_decisions']])
    assert np.array_equal(comparison.mean(axis=0), m['reference_currents'])
    assert float(np.quantile(np.abs(comparison-comparison.mean(axis=0)), .99)) == m['display_scale_mv']
    # Check 192 spatially spread exported rows directly against Arrow source.
    # Coordinates must be exact float32 transforms; body IDs must match.
    positions = np.fromfile(OUT/'synapses.f32', dtype='<f4').reshape(-1, 3)
    owners = np.fromfile(OUT/'owners.u32', dtype='<u4')
    rows = np.fromfile(OUT/'source-rows.u32', dtype='<u4').reshape(-1, 2)
    picks = np.linspace(0, len(positions)-1, 192, dtype=int)
    center = np.asarray(m['coordinate_transform']['center'])
    scale = m['coordinate_transform']['scale']
    with pa.memory_map(str(ROOT/p['synapse_file']), 'r') as file:
        schema = ipc.open_file(file).schema
        reader = ipc.open_file(file, options=ipc.IpcReadOptions(included_fields=[schema.get_field_index(n) for n in ['x', 'y', 'z', 'body']]))
        for b in np.unique(rows[picks, 0]):
            selected = picks[rows[picks, 0] == b]
            batch = reader.get_batch(int(b))
            offsets = rows[selected, 1]
            xyz = np.column_stack([batch.column(batch.schema.get_field_index(n)).to_numpy()[offsets] for n in ['x', 'y', 'z']])
            body = batch.column(batch.schema.get_field_index('body')).to_numpy()[offsets]
            assert np.array_equal(((xyz-center)*scale).astype('<f4'), positions[selected])
            assert [str(v) for v in body] == [m['body_ids'][i] for i in owners[selected]]
    print(json.dumps(dict(result='PASS', source_positions_checked=len(picks), activity_values_checked=35550,
        readout_neurons=len(expected_readout), original_decisions=[c['decision']['decision_seq'] for c in m['clips']],
        point_count=m['point_count'], edges_checked_by_exporter=len(m['edges']), graph_sha256=p['graph_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
