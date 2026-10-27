import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/brain-data.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { mergeNeuronCoverage } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const data = {
  manifest: { node_count: 4, provenance: { graph_sha256: 'graph' } },
  positions: new Float32Array([1, 2, 3, 4, 5, 6]), owners: new Uint32Array([0, 2]),
  centers: new Float32Array([1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0]), activityEdges: [[0, 2, 7]],
};
const coverage = { schema: 'malecns_lab.neuron_coverage.v1', node_count: 4, point_count: 2,
  positioned_count: 4, base_positioned_count: 2, graph_sha256: 'graph', base_manifest_sha256: 'base', body_ids_sha256: 'bodies' };
const points = new Float32Array([7, 8, 9, 10, 11, 12]), owners = new Uint32Array([1, 3]);
const merge = (c = coverage, p = points, o = owners) => mergeNeuronCoverage(data, c, 'base', 'bodies', p, o);

test('adds every missing cell once, preserves existing geometry and edge anchors', () => {
  const result = merge();
  assert.deepEqual([...result.owners], [0, 2, 1, 3]);
  assert.equal(new Set(result.owners).size, 4);
  assert.deepEqual([...result.positions], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.deepEqual([...result.centers], [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]);
  assert.deepEqual([...data.centers], [1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0]);
  assert.equal(result.activityEdges, data.activityEdges);
});

test('rejects foreign geometry, ordering, counts, repeated and out-of-range cells', () => {
  for (const changed of [{ graph_sha256: 'other' }, { base_manifest_sha256: 'other' }, { body_ids_sha256: 'other' },
    { node_count: 3 }, { positioned_count: 3 }, { base_positioned_count: 1 }, { point_count: 5 }]) {
    assert.throws(() => merge({ ...coverage, ...changed }));
  }
  for (const invalid of [[0, 3], [1, 1], [1, 4]]) assert.throws(() => merge(coverage, points, new Uint32Array(invalid)));
});

test('rejects incomplete coverage, wrong lengths and fabricated nonfinite positions', () => {
  assert.throws(() => merge({ ...coverage, point_count: 1 }, points.slice(0, 3), owners.slice(0, 1)), /缺少/);
  assert.throws(() => merge(coverage, points.slice(0, 3)), /长度/);
  const invalid = new Float32Array(points); invalid[0] = NaN;
  assert.throws(() => merge(coverage, invalid), /无效/);
});
