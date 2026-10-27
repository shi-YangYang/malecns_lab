import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

if (!globalThis.crypto) globalThis.crypto = webcrypto;
const source = await readFile(new URL('../src/neural-state.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { createNeuralMapping, mapNeuralSignals } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const ids = ['31', '12', '99'], idsHash = createHash('sha256').update(JSON.stringify(ids)).digest('hex');
const graphHash = 'b'.repeat(64);
const state = { schema: 'malecns_lab.neural_state_manifest.v1', node_count: 3, body_ids: ids, body_ids_sha256: idsHash, graph_sha256: graphHash,
  channels: ['voltage_mean_mv', 'voltage_delta_mean_mv', 'spike_rate_hz'], dtype: 'float32-le', v_rest_mv: -60, v_threshold_mv: -50,
  window_steps: 50, baseline: 'request_start_pre_step', sample: 'post_step_window_mean' };
const display = { node_count: 3, body_ids: ['99', '31', '12'], provenance: { graph_sha256: graphHash } };
const manifest = { run_id: 'run1', execution_id: 'exec1', neural_state: state };
const ref = { schema: 'malecns_lab.neural_state_ref.v1', file: 'neural/0001.f32', bytes: 36, sha256: 'c'.repeat(64), body_ids_sha256: idsHash, node_count: 3, archived: true };
const batch = { run_id: 'run1', execution_id: 'exec1', request_id: 'req1', batch_sequence: 1, neural_state: ref, group_rates_hz: [1000] };
const frame = { ...manifest, request_id: 'req1', batch_sequence: 1, body_ids_sha256: idsHash,
  values: new Float32Array([-60, -58, -62, 0, 2, -2, 50, 0, 0]) };

test('maps graph-order cells by identity and shows subthreshold voltage at zero spikes', async () => {
  const mapping = await createNeuralMapping(state, display);
  assert.deepEqual([...mapping.graphToDisplay], [1, 2, 0]);
  const signals = mapNeuralSignals(frame, batch, manifest, mapping);
  assert.deepEqual([...signals.voltage], [.5, 0, .5]);
  assert.deepEqual([...signals.spikes], [0, .5, 0]);
  assert.equal(signals.voltage[1], 0, 'spiking does not invent a voltage delta');
  assert.equal(signals.spikes[2], 0, 'subthreshold voltage does not light outgoing connections');
  assert.deepEqual([...frame.values], [-60, -58, -62, 0, 2, -2, 50, 0, 0]);
});

test('uses one fixed magnitude scale across frames, independent of group rates', async () => {
  const mapping = await createNeuralMapping(state, display);
  const stronger = new Float32Array(frame.values); stronger[4] = 6;
  const first = mapNeuralSignals(frame, batch, manifest, mapping);
  const second = mapNeuralSignals({ ...frame, values: stronger }, { ...batch, group_rates_hz: [0] }, manifest, mapping);
  assert.equal(second.voltage[2], .75);
  assert.equal(first.voltage[0], second.voltage[0]);
  assert.deepEqual([...first.spikes], [...second.spikes]);
  const zero = mapNeuralSignals({ ...frame, values: new Float32Array(9) }, batch, manifest, mapping);
  assert.ok(zero.voltage.every(value => value === 0) && zero.spikes.every(value => value === 0));
});

test('rejects wrong graph, body ordering hash and missing display identities', async () => {
  await assert.rejects(createNeuralMapping({ ...state, graph_sha256: 'd'.repeat(64) }, display), /身份/);
  await assert.rejects(createNeuralMapping({ ...state, body_ids: [...ids].reverse() }, display), /顺序校验/);
  await assert.rejects(createNeuralMapping(state, { ...display, body_ids: ['99', '31', '100'] }), /不一致/);
  await assert.rejects(createNeuralMapping({ ...state, channels: ['spike_rate_hz'] }, display), /通道/);
});

test('rejects mismatched batch or request and invalid channel values without group fallback', async () => {
  const mapping = await createNeuralMapping(state, display);
  for (const changed of [{ batch_sequence: 2 }, { request_id: 'req2' }, { run_id: 'run2' }, { execution_id: 'exec2' }, { body_ids_sha256: 'd'.repeat(64) }, { values: new Float32Array(8) }]) {
    assert.throws(() => mapNeuralSignals({ ...frame, ...changed }, batch, manifest, mapping), /不匹配/);
  }
  for (const [index, value] of [[0, NaN], [3, Infinity], [7, -1]]) {
    const values = new Float32Array(frame.values); values[index] = value;
    assert.throws(() => mapNeuralSignals({ ...frame, values }, batch, manifest, mapping), /无效数值/);
  }
  assert.throws(() => mapNeuralSignals(frame, { ...batch, neural_state: null }, manifest, mapping), /不匹配/);
});
