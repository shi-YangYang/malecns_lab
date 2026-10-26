#!/usr/bin/env python3
"""Bounded, deterministic display-asset extraction. No model execution/imports.

Positions are original synapse coordinates, never generated neurites. Lines are
actual graph edges between representative synapses, NOT axon trajectories.
Run from the repository with its existing numpy/pyarrow environment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/brain_visualization_v1'
RUN = ROOT / 'data/x2_sim/agibot/data/var/desktop-20261026-093738-5a66765f'
GRAPH = ROOT / 'data/malecns_v2/full_reference.npz'
SYN = ROOT / 'data/malecns_v1/syn-points-male-cns-v1.0-minconf-0.5.feather'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    atlas = json.loads((ROOT / 'apps/atlas/public/data/atlas.json').read_text())
    with np.load(GRAPH, allow_pickle=False) as graph:
        ids, types = graph['body_ids'], graph['types']
        meta = json.loads(str(graph['meta']))
        indptr, indices, weights = graph['indptr'], graph['indices'], graph['data']
    assert len(ids) == 166700 and len(indices) == 25582938
    assert np.all(ids[1:] > ids[:-1])
    readout = np.flatnonzero(np.isin(types, ['L1', 'L2']))
    assert len(readout) == 3555
    transform = atlas['coordinate_transform']
    center, scale = np.asarray(transform['center']), transform['scale']
    batches, owners, coordinates, source_rows, regions = [], [], [], [], []
    # Cover every batch: the source is spatially ordered, so skipping batches
    # would create false anatomical bands. Project columns before decompression.
    with pa.memory_map(str(SYN), 'r') as source:
        schema = ipc.open_file(source).schema
        columns = ['x', 'y', 'z', 'body', 'primary']
        options = ipc.IpcReadOptions(included_fields=[schema.get_field_index(n) for n in columns])
        reader = ipc.open_file(source, options=options)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            body = batch.column(batch.schema.get_field_index('body')).to_numpy()
            owner = np.searchsorted(ids, body)
            valid = (owner < len(ids)) & (ids[np.minimum(owner, len(ids)-1)] == body)
            # Background 1/128 of rows, measured L1/L2 1/8. Every z range is
            # covered; the spatial cap below only selects original points.
            readout_mask = np.isin(owner, readout)
            sequence = np.arange(len(body))
            row = np.flatnonzero(valid & ((sequence % 128 == 0) | (readout_mask & (sequence % 8 == 0))))
            xyz = np.column_stack([batch.column(batch.schema.get_field_index(n)).to_numpy()[row] for n in ['x', 'y', 'z']])
            owners.append(owner[row]); coordinates.append((xyz-center)*scale)
            source_rows.append(np.column_stack([np.full(len(row), b), row]))
            regions.extend(batch.column(batch.schema.get_field_index('primary')).take(pa.array(row)).to_pylist())
            batches.append(b)
            if b % 1000 == 0:
                print(f'Spatial asset extraction: {b}/{reader.num_record_batches} batches', flush=True)
    xyz = np.concatenate(coordinates).astype('<f4')
    owner = np.concatenate(owners).astype('<u4')
    rows = np.concatenate(source_rows).astype('<u4')
    regions = np.asarray([v or 'unassigned' for v in regions])
    # Uniform density via existing points only: at most two points per cell,
    # separately for measured and unmeasured neurons to retain signal coverage.
    cells = np.floor(xyz / .009).astype(np.int32)
    keys = np.column_stack([cells, np.isin(owner, readout)])
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    # Source rows are spatially ordered. Taking the first point of each voxel
    # would bias z towards voxel boundaries. Hash only chooses among ORIGINAL
    # source rows; it never creates/jitters positions or connections.
    priority = (rows[:, 0].astype(np.uint64) * 2654435761) ^ (rows[:, 1].astype(np.uint64) * 2246822519)
    priority &= 0xffffffff
    order = np.lexsort((priority, inverse))
    sorted_keys = inverse[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    second = starts[starts + 1 < len(order)] + 1
    second = second[sorted_keys[second] == sorted_keys[second - 1]]
    keep = np.sort(np.r_[order[starts], order[second]])
    xyz, owner, rows, regions = xyz[keep], owner[keep], rows[keep], regions[keep]
    counts = np.bincount(owner, minlength=len(ids))
    centers = np.zeros((len(ids), 3), dtype='<f4')
    for axis in range(3):
        centers[:, axis] = np.bincount(owner, weights=xyz[:, axis], minlength=len(ids)) / np.maximum(counts, 1)
    # A long neuron's centroid can lie in empty space between brain and cord.
    # Anchor each graph node at its closest ORIGINAL synapse instead; this
    # avoids drawing a fictitious dense tract through unsupported centroids.
    distance_to_center = np.sum((xyz-centers[owner])**2, axis=1)
    anchor_order = np.lexsort((distance_to_center, owner))
    ordered_owner = owner[anchor_order]
    anchor_first = np.r_[0, np.flatnonzero(np.diff(ordered_owner)) + 1]
    anchor_points = anchor_order[anchor_first]
    centers[owner[anchor_points]] = xyz[anchor_points]
    # Up to two strongest spatially local, well-supported outgoing edges per
    # neuron. A stable cap keeps rendering bounded. No arbitrary curve fitting.
    edges = []
    for pre in np.flatnonzero(counts >= 6):
        a, b = indptr[pre:pre+2]
        post, w = indices[a:b], weights[a:b]
        distance = np.linalg.norm(centers[post]-centers[pre], axis=1)
        eligible = np.flatnonzero((counts[post] >= 6) & (post != pre) & (distance > .018) & (distance < .16) & (w != 0))
        if not len(eligible):
            continue
        best = eligible[np.argsort(-np.abs(w[eligible]), kind='stable')[:2]]
        edges.extend((int(pre), int(post[j]), int(w[j])) for j in best)
    # Spread throughout the eligible source ordering, not only the densest hub.
    if len(edges) > 18000:
        edges = [edges[i] for i in np.linspace(0, len(edges)-1, 18000, dtype=int)]
    # Independently assert every exported line corresponds to a nonzero edge.
    for pre, post, weight in edges:
        a, b = indptr[pre:pre+2]
        assert np.any((indices[a:b] == post) & (weights[a:b] == weight))
    clips = []
    selections = {1: ('forward', '前进', 'ADVANCE'), 9: ('left', '左转', 'TURN_LEFT'), 17: ('right', '右转', 'TURN_RIGHT'), 25: ('wave', '挥手', 'WAVE_RIGHT'), 29: ('idle', '待机', 'HOLD')}
    events_path = RUN / 'behavior_events.jsonl'
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') != 'raw_decision':
            continue
        decision = event['decision']
        if decision['decision_seq'] not in selections:
            continue
        key, label, behavior = selections[decision['decision_seq']]
        assert decision['behavior'] == behavior and decision['joint_readout']['model_advanced_this_frame']
        features = np.asarray(event['features_mv'], dtype='<f4')
        assert features.shape == (7110,) and np.isfinite(features).all()
        clips.append(dict(id=key, label=label, decision=decision, currents=features.reshape(2, -1).tolist()))
    assert len(clips) == 5
    # Compare identical neural windows for the four request-start excerpts.
    # HOLD is a later FINISH excerpt, so it must not define their baseline.
    comparison = np.asarray([c['currents'] for c in clips if c['id'] != 'idle'])
    baseline = comparison.mean(axis=0)
    deltas = np.abs(comparison-baseline)
    display_scale = float(np.quantile(deltas, .99))
    assert display_scale > 0 and all(np.max(np.abs(np.asarray(c['currents'])-baseline)) > 0 for c in clips)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {'synapses.f32': xyz, 'owners.u32': owner, 'centers.f32': centers, 'source-rows.u32': rows}
    for name, array in arrays.items():
        array.tofile(OUT / name)
    region_names, region_indices = np.unique(regions, return_inverse=True)
    region_indices.astype('<u2').tofile(OUT / 'regions.u16')
    files = {name: dict(bytes=(OUT/name).stat().st_size, sha256=sha(OUT/name)) for name in [*arrays, 'regions.u16']}
    manifest = dict(schema='malecns_lab.brain_display.v1', node_count=len(ids), point_count=len(xyz),
        body_ids=[str(v) for v in ids], readout_indices=readout.tolist(), edges=edges,
        regions=region_names.tolist(), clips=clips, files=files, display_scale_mv=display_scale,
        display_signal='abs(recorded signed current g - four request-start clips mean in matching 250-step window); common p99 scale of those four clips; HOLD is a later FINISH excerpt shown on the same scale',
        comparison_decisions=[1, 9, 17, 25], reference_currents=baseline.tolist(),
        signal_scope='L1/L2 only; per-neuron scalar painted on sampled synapses; not per-synapse measurements or spikes',
        window_steps=250, dt_ms=.1, coordinate_transform=transform,
        provenance=dict(graph_file=str(GRAPH.relative_to(ROOT)), graph_file_sha256=sha(GRAPH), graph_sha256=meta['graph_sha256'],
            synapse_file=str(SYN.relative_to(ROOT)), synapse_file_bytes=SYN.stat().st_size,
            source_rows='pairs of Arrow record-batch number and row within batch; original coordinates and body identity can be recovered exactly',
            batches=batches, selection='every batch; every 128th row plus every 8th L1/L2 row; two original points per .009 cell per measured/unmeasured class, source-row hash priority to avoid z-order bias; no coordinate generation',
            edge_geometry='straight schematic connection between representative ORIGINAL synapses nearest each neuron sampled-synapse centroid; actual directed CSC edge (post row, pre column); NOT an axon skeleton',
            run_id=RUN.name, events_file=str(events_path.relative_to(ROOT)), events_sha256=sha(events_path),
            readout_source='data/x2_model_redesign_20261024_r1/neural_features_v2.py: graph-order L1/L2, g, two 250-step means',
            readout_source_sha256=sha(ROOT / 'data/x2_model_redesign_20261024_r1/neural_features_v2.py'),
            exporter_sha256=sha(Path(__file__)), license='CC-BY-4.0',
            interpretation='Five historical excerpts, not an action-specific biological atlas, live inference or proof of robot motion.'))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps(dict(points=len(xyz), positioned_neurons=int(np.count_nonzero(counts)), links=len(edges), readout_covered=int(np.count_nonzero(counts[readout])), display_scale_mv=display_scale, bytes=sum(v['bytes'] for v in files.values()), output=str(OUT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
