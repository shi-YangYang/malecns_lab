import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { NeuralStateReader } from './neural-state.mjs';
import { ExperimentController } from './controller.mjs';

const count = 166700;
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'malecns-state-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, 'telemetry/neural'), { recursive: true });
  const body_ids = Array.from({ length: count }, (_, i) => String(i + 1));
  const body_ids_sha256 = sha(JSON.stringify(body_ids));
  const manifest = { run_id: 'fixture-run', execution_id: 'fixture-execution', session_id: 'fixture-session',
    checkpoint_id: 'fixture-checkpoint', input_map_id: 'fixture-input', dt_ms: .1,
    graph: { reference_graph_sha256: 'a'.repeat(64) },
    neural_state: { schema: 'malecns_lab.neural_state_manifest.v1', node_count: count, body_ids, body_ids_sha256,
      graph_sha256: 'a'.repeat(64), dtype: 'float32-le', channels: ['voltage_mean_mv', 'voltage_delta_mean_mv', 'spike_rate_hz'],
      v_rest_mv: -52, v_threshold_mv: -45, window_steps: 50, baseline: 'request_start_pre_step', sample: 'post_step_window_mean' } };
  const buffer = Buffer.alloc(count * 3 * 4);
  buffer.writeFloatLE(-51.75, 0); buffer.writeFloatLE(.25, count * 4);
  const batch = { run_id: manifest.run_id, execution_id: manifest.execution_id, session_id: manifest.session_id,
    checkpoint_id: manifest.checkpoint_id, input_map_id: manifest.input_map_id, batch_sequence: 1,
    step_start: 600, step_end: 650, dt_ms: .1, request_id: 'request-fixture', phase: 'forward', decision_sequence: 1,
    neural_state: { schema: 'malecns_lab.neural_state_ref.v1', file: 'neural/000001.f32', bytes: buffer.length,
      sha256: sha(buffer), body_ids_sha256, node_count: count, archived: true } };
  const file = path.join(root, 'telemetry', batch.neural_state.file);
  await writeFile(file, buffer);
  return { root, manifest, batch, file, buffer };
}

test('full-neuron packet preserves voltage change with zero spikes and cell order', async t => {
  const f = await fixture(t), frame = await new NeuralStateReader().read(f.root, f.manifest, f.batch);
  assert.equal(frame.values.length, count * 3);
  assert.equal(frame.values[0], -51.75); assert.equal(frame.values[count], .25);
  assert.equal(frame.values[count * 2], 0);
  assert.equal(frame.request_id, f.batch.request_id);
  assert.equal(frame.body_ids_sha256, f.manifest.neural_state.body_ids_sha256);
});
test('foreign identity, neuron ordering and reference sequence are rejected', async t => {
  const f = await fixture(t), reader = new NeuralStateReader();
  await assert.rejects(reader.read(f.root, f.manifest, { ...f.batch, execution_id: 'foreign' }), /身份/);
  await assert.rejects(reader.read(f.root, f.manifest, { ...f.batch, batch_sequence: 2 }), /文件名/);
  f.manifest.neural_state.body_ids.reverse();
  await assert.rejects(reader.read(f.root, f.manifest, f.batch), /顺序/);
});
test('unsafe paths, symlinks, truncated and corrupt files are rejected', async t => {
  const f = await fixture(t), reader = new NeuralStateReader();
  await assert.rejects(reader.read(f.root, f.manifest, { ...f.batch, neural_state: { ...f.batch.neural_state, file: '../outside' } }), /文件名/);
  await writeFile(f.file, f.buffer.subarray(4));
  await assert.rejects(reader.read(f.root, f.manifest, f.batch), /长度/);
  const corrupt = Buffer.from(f.buffer); corrupt.writeFloatLE(2, 0); await writeFile(f.file, corrupt);
  await assert.rejects(reader.read(f.root, f.manifest, f.batch), /校验/);
  await writeFile(path.join(f.root, 'target.f32'), f.buffer); await rm(f.file); await symlink(path.join(f.root, 'target.f32'), f.file);
  await assert.rejects(reader.read(f.root, f.manifest, f.batch), /符号链接/);
});
test('nonfinite voltage and impossible rates are rejected even with valid hash', async t => {
  const f = await fixture(t);
  f.buffer.writeFloatLE(NaN, 0); await writeFile(f.file, f.buffer); f.batch.neural_state.sha256 = sha(f.buffer);
  await assert.rejects(new NeuralStateReader().read(f.root, f.manifest, f.batch), /非有限/);
  f.buffer.writeFloatLE(-52, 0); f.buffer.writeFloatLE(-1, count * 8); await writeFile(f.file, f.buffer); f.batch.neural_state.sha256 = sha(f.buffer);
  await assert.rejects(new NeuralStateReader().read(f.root, f.manifest, f.batch), /放电率/);
});
test('bounded live slots require matching hash; legacy batch has no fabricated voltage', async t => {
  const f = await fixture(t);
  assert.equal(await new NeuralStateReader().read(f.root, {}, {}), null);
  f.batch.neural_state = { ...f.batch.neural_state, file: 'neural/live-0.f32', archived: false };
  await writeFile(path.join(f.root, 'telemetry/neural/live-0.f32'), f.buffer);
  assert.ok(await new NeuralStateReader().read(f.root, f.manifest, f.batch));
  f.batch.neural_state.sha256 = 'b'.repeat(64);
  await assert.rejects(new NeuralStateReader().read(f.root, f.manifest, f.batch), /实时槽/);
});
test('snapshot reports neural-state failure independently of existing rate batch', async t => {
  const f = await fixture(t);
  await writeFile(path.join(f.root, 'telemetry/manifest.json'), JSON.stringify(f.manifest));
  await writeFile(path.join(f.root, 'telemetry/latest.json'), JSON.stringify(f.batch));
  await rm(f.file);
  const controller = new ExperimentController(f.root);
  controller.output = f.root;
  const snapshot = await controller.snapshot();
  assert.equal(snapshot.read_error, null); assert.ok(snapshot.neural_state_error);
  assert.equal(snapshot.neural_state, null); assert.equal(snapshot.batch.batch_sequence, 1);
});
