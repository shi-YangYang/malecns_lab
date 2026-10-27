#!/usr/bin/env python3
"""Export bounded live-display edges from the existing full graph and positions.

No model computation, raw Feather scan, new coordinates, or activity-dependent
selection. These are straight graph links between sampled original synapses,
not axon trajectories or measured signal propagation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STRUCTURE = ROOT / 'data/brain_visualization_v1'
GRAPH = ROOT / 'data/malecns_v2/full_reference.npz'
# Only the fixed group membership is read; no activity batches or Hz values.
GROUPS = ROOT / 'data/x2_sim/agibot/data/var/desktop-20261026-185601-618e42af/telemetry/manifest.json'
OUT = ROOT / 'data/brain_activity_edges_v1/activity-edges.json'
PER_NEURON = 4
PER_GROUP = 240
MIN_DISTANCE, MAX_DISTANCE = .018, .16


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    m = json.loads((STRUCTURE / 'manifest.json').read_text())
    groups = json.loads(GROUPS.read_text())['groups']
    with np.load(GRAPH, allow_pickle=False) as graph:
        ids, ptr, posts, weights = (graph[name] for name in ('body_ids', 'indptr', 'indices', 'data'))
        meta = json.loads(str(graph['meta']))
    assert meta['graph_sha256'] == m['provenance']['graph_sha256']
    assert sha(GRAPH) == m['provenance']['graph_file_sha256']
    assert [str(v) for v in ids] == m['body_ids']
    for name in ('centers.f32', 'owners.u32', 'synapses.f32'):
        assert sha(STRUCTURE / name) == m['files'][name]['sha256']
    centers = np.fromfile(STRUCTURE / 'centers.f32', dtype='<f4').reshape(-1, 3)
    owners = np.fromfile(STRUCTURE / 'owners.u32', dtype='<u4')
    positions = np.fromfile(STRUCTURE / 'synapses.f32', dtype='<f4').reshape(-1, 3)
    counts = np.bincount(owners, minlength=len(ids))
    lookup = {str(body): i for i, body in enumerate(ids)}
    existing = {(edge[0], edge[1]) for edge in m['edges']}
    selected, coverage = {}, []
    assert len(groups) == 22 and len({g['id'] for g in groups}) == 22
    for group in groups:
        members = sorted({lookup[str(body)] for body in group['body_ids']})
        by_neuron = []
        eligible_count = old_support_count = 0
        for pre in members:
            if counts[pre] == 0:
                continue
            a, b = ptr[pre:pre + 2]
            post, w = posts[a:b], weights[a:b]
            distance = np.linalg.norm(centers[post] - centers[pre], axis=1)
            eligible = np.flatnonzero((counts[post] > 0) & (post != pre) & (w != 0) &
                                      (distance > MIN_DISTANCE) & (distance < MAX_DISTANCE))
            eligible_count += len(eligible)
            old_support_count += int(np.count_nonzero((counts[pre] >= 6) & (counts[post[eligible]] >= 6)))
            # Strongest actual signed edges, then post index breaks ties.
            order = eligible[np.lexsort((post[eligible], -np.abs(w[eligible].astype(np.int64))))]
            links = [(int(pre), int(post[j]), int(w[j])) for j in order if (pre, int(post[j])) not in existing][:PER_NEURON]
            if links:
                by_neuron.append(links)
        # First cover distinct presynaptic cells, then their second/third/fourth
        # links. Stratify a large group by graph order rather than dense hubs.
        chosen = []
        for rank in range(PER_NEURON):
            candidates = [links[rank] for links in by_neuron if len(links) > rank]
            budget = PER_GROUP - len(chosen)
            if len(candidates) > budget:
                candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, budget, dtype=int)]
            chosen.extend(candidates)
            if len(chosen) == PER_GROUP:
                break
        for edge in chosen:
            selected[edge[:2]] = edge
        coverage.append(dict(id=group['id'], members=len(members), positioned_members=int(np.count_nonzero(counts[members])),
                             local_candidate_edges=eligible_count, previous_support_candidate_edges=old_support_count,
                             selected_edges=len(chosen), selected_presynaptic_neurons=len({e[0] for e in chosen})))
    edges = sorted(selected.values())
    assert 0 < len(edges) <= 6000
    # Independently verify every exported edge and every used original anchor.
    endpoints = {node for pre, post, _ in edges for node in (pre, post)}
    actual_anchors = set(owners[np.all(positions == centers[owners], axis=1)].tolist())
    assert endpoints <= actual_anchors
    for pre, post, weight in edges:
        a, b = ptr[pre:pre + 2]
        assert np.any((posts[a:b] == post) & (weights[a:b] == weight))
        assert MIN_DISTANCE < np.linalg.norm(centers[post] - centers[pre]) < MAX_DISTANCE
        assert (pre, post) not in existing
    body_ids_hash = hashlib.sha256(json.dumps(m['body_ids'], separators=(',', ':')).encode()).hexdigest()
    value = dict(schema='malecns_lab.activity_edges.v1', node_count=len(ids), edges=edges, coverage=coverage,
                 graph_sha256=meta['graph_sha256'], body_ids_sha256=body_ids_hash,
                 centers_sha256=m['files']['centers.f32']['sha256'],
                 provenance=dict(graph_file=str(GRAPH.relative_to(ROOT)), graph_file_sha256=sha(GRAPH),
                                 groups_file=str(GROUPS.relative_to(ROOT)), groups_file_sha256=sha(GROUPS),
                                 owners_sha256=m['files']['owners.u32']['sha256'],
                                 exporter_sha256=sha(Path(__file__)),
                                 selection=dict(group_ids=[g['id'] for g in groups], per_neuron=PER_NEURON, per_group=PER_GROUP,
                                                min_anchor_distance=MIN_DISTANCE, max_anchor_distance=MAX_DISTANCE,
                                                rule='Fixed recorded-group membership; outgoing nonzero CSC edges; both endpoints have original sampled synapses; strongest absolute weights per neuron; stratified graph-order group cap; exclude base display edges; no activity values read'),
                                 geometry='Straight schematic graph links between representative ORIGINAL synapses. Not axon skeletons or measured propagation.'))
    serialized = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != serialized:
        raise FileExistsError(f'Preserve the existing exported asset; use a new version directory: {OUT}')
    OUT.write_text(serialized)
    print(json.dumps(dict(output=str(OUT), edges=len(edges), all_edges_verified=True, all_anchors_original=True, coverage=coverage), ensure_ascii=False))


if __name__ == '__main__':
    main()
