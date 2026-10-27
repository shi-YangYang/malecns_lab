import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/activity-timeline.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { ActivityTimeline } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const baseline = { kind: 'baseline', rates: null };

test('idle display advances without inventing neural telemetry and completion returns to baseline', () => {
  const timeline = new ActivityTimeline(1000);
  timeline.record(baseline, 1000);
  assert.equal(timeline.record(baseline, 1250), 250);
  const rates = [53.2, 50.4, 0];
  timeline.record({ kind: 'measured', rates }, 1500);
  timeline.record(baseline, 1750);
  assert.deepEqual(timeline.points.map(point => point.reading.kind), ['baseline', 'baseline', 'measured', 'baseline']);
  assert.equal(timeline.points.at(-1).reading.rates, null);
  assert.equal(timeline.points[2].reading.rates, rates);
  assert.deepEqual(rates, [53.2, 50.4, 0]);
});

test('unknown/stale data and skipped batches retain explicit curve gaps', () => {
  const timeline = new ActivityTimeline(0);
  timeline.record({ kind: 'measured', rates: [5] }, 0);
  timeline.record({ kind: 'gap', rates: null }, 250);
  timeline.record({ kind: 'measured', rates: [6] }, 500, true);
  assert.deepEqual(timeline.points.map(point => point.reading.kind), ['measured', 'gap', 'gap', 'measured']);
  assert.equal(timeline.points.filter(point => point.reading.kind === 'baseline').length, 0);
});

test('hidden pages pause display time and resume with a break instead of fabricated history', () => {
  const timeline = new ActivityTimeline(1000);
  timeline.record(baseline, 1000);
  timeline.pause(2000);
  assert.equal(timeline.record(baseline, 9000), 1000);
  assert.equal(timeline.points.at(-1).reading.kind, 'gap');
  timeline.resume(12000);
  assert.equal(timeline.record(baseline, 12250), 1250);
  assert.equal(timeline.points.at(-2).reading.kind, 'gap');
  assert.equal(timeline.points.at(-1).atMs, 1250);
});

test('rolling history stays bounded and drops activity after the visible window', () => {
  const timeline = new ActivityTimeline(0);
  timeline.record({ kind: 'measured', rates: [5] }, 0);
  for (let now = 250; now <= 60000; now += 250) timeline.record(baseline, now);
  assert.ok(timeline.points.length <= 82);
  assert.ok(timeline.points[1].atMs >= 40000);
  assert.ok(timeline.points.every(point => point.reading.kind === 'baseline'));
  for (let now = 60001; now < 61000; now++) timeline.record({ kind: 'measured', rates: [now] }, now);
  assert.ok(timeline.points.length <= 256);
});
