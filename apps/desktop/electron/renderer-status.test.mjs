import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/live-status.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { ActivityDisplay, LiveStatus, canEmergencyStop, canSubmitTask, startAction } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const status = state => ({ state, resources_stopped: state === 'idle' });
const dataSource = await readFile(new URL('../src/data.ts', import.meta.url), 'utf8');
const dataModule = ts.transpileModule(dataSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { groupActivityRate, validBatch } = await import(`data:text/javascript;base64,${Buffer.from(dataModule.outputText).toString('base64')}`);
const brainSource = await readFile(new URL('../src/brain-data.ts', import.meta.url), 'utf8');
const brainModule = ts.transpileModule(brainSource, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { validateBrainGraphInfo } = await import(`data:text/javascript;base64,${Buffer.from(brainModule.outputText).toString('base64')}`);

const active = (request = 'request-a', fields = {}) => ({ state: 'running', mode: 'brain_control', brain_controlled: true, run_id: 'run-a', active_request_id: request, request_id: request, phase: 'computing', ...fields });
const batchFor = (request = 'request-a', fields = {}) => ({ run_id: 'run-a', request_id: request, phase: 'forward', group_rates_hz: [7, 11], ...fields });

test('activity display freezes real values when action completion begins feedback processing', () => {
  const display = new ActivityDisplay();
  assert.equal(display.update(active()), true);
  const history = [];
  const receive = batch => { if (display.accepts(batch)) history.push(batch); };
  receive(batchFor());
  display.update(active('request-a', { phase: 'executing' }));
  receive(batchFor('request-a', { group_rates_hz: [9, 13] }));
  assert.equal(history.length, 2);
  display.update(active('request-a', { closing: true }));
  assert.equal(display.mode, 'closing');
  receive(batchFor('request-a', { group_rates_hz: [12, 16] }));
  // A later notification without closing must not reopen this request.
  display.update(active());
  assert.equal(display.mode, 'closing');
  display.update(active(null, { phase: 'waiting_task', request_id: 'request-a' }));
  assert.equal(display.mode, 'waiting');
  receive(batchFor('request-a', { group_rates_hz: [0, 0] }));
  assert.equal(history.length, 2);
  assert.deepEqual(history.at(-1).group_rates_hz, [9, 13]);
});

test('activity display isolates the next request from latest completion feedback', () => {
  const display = new ActivityDisplay();
  display.update(active());
  display.update(active('request-a', { closing: true }));
  display.update(active(null, { phase: 'waiting_task' }));
  display.beginRequest();
  assert.equal(display.accepts(batchFor()), false);
  assert.equal(display.update(active('request-b', { phase: 'queued' })), true);
  assert.equal(display.accepts(batchFor()), false);
  assert.equal(display.accepts(batchFor('request-b')), true);
});

test('activity display blocks interrupted telemetry while END awaits its new request id', () => {
  const display = new ActivityDisplay();
  display.update(active('request-a', { phase: 'executing' }));
  display.beginRequest();
  display.update(active('request-a', { phase: 'queued', request_pending: true, task: 'END_INTERACTION' }));
  assert.equal(display.mode, 'waiting');
  assert.equal(display.accepts(batchFor()), false);
  display.update(active('request-end', { phase: 'queued', task: 'END_INTERACTION' }));
  assert.equal(display.mode, 'active');
  assert.equal(display.accepts(batchFor()), false);
  assert.equal(display.accepts(batchFor('request-end')), true);
});

test('activity display admits initialization only in its phase and rejects stale run and execution identities', () => {
  const display = new ActivityDisplay();
  display.update(active(null, { phase: 'initializing' }));
  assert.equal(display.accepts(batchFor(null, { phase: 'precondition' })), true);
  display.update(active(null, { phase: 'waiting_task' }));
  assert.equal(display.accepts(batchFor(null, { phase: 'precondition' })), false);
  display.update(active());
  assert.equal(display.accepts(batchFor('request-a', { run_id: 'old-run' })), false);
  const manifest = { schema: 'malecns_lab.x2_behavior_telemetry_manifest.v1', run_id: 'run-a', execution_id: 'execution-a', session_id: 'session-a', checkpoint_id: 'checkpoint-a', input_map_id: 'input-a', selected_neurons: [], groups: [{ id: 'input' }], dt_ms: .1 };
  const batch = batchFor('request-a', { schema: 'malecns_lab.x2_behavior_telemetry_batch.v1', execution_id: 'old-execution', session_id: 'session-a', checkpoint_id: 'checkpoint-a', input_map_id: 'input-a', decision_sequence: 1, batch_sequence: 1, step_start: 0, step_end: 50, dt_ms: .1, group_rates_hz: [5], spike_events: [] });
  assert.equal(display.accepts(batch), true);
  assert.equal(validBatch(batch, manifest), false);
  display.reset();
  assert.equal(display.accepts(batchFor()), false);
  assert.equal(display.requestId, null);
  display.update(active('request-b', { run_id: 'run-b' }));
  assert.equal(display.accepts(batchFor()), false);
});

test('activity display resumes the authoritative request after submission rejection and freezes on stop', () => {
  const display = new ActivityDisplay();
  display.update(active());
  display.beginRequest();
  display.update(active('request-a', { notice: 'request_rejected' }));
  assert.equal(display.accepts(batchFor()), true);
  display.update(active('request-a', { state: 'stopping' }));
  assert.equal(display.accepts(batchFor()), false);
});

test('brain info uses full graph counts with matching atlas and display identity', () => {
  const graph = { node_count: 166700, edge_count: 25582938, graph_sha256: 'a'.repeat(64) };
  const source = { schema: 'malecns_lab.atlas.v1', graph };
  const display = { node_count: 166700, edges: [[1, 2, 3]], provenance: { graph_sha256: graph.graph_sha256 } };
  assert.deepEqual(validateBrainGraphInfo(source, 166700, display), graph);
  assert.throws(() => validateBrainGraphInfo(source, 166699, display), /身份不匹配/);
  assert.throws(() => validateBrainGraphInfo(source, 166700, { ...display, provenance: { graph_sha256: 'b'.repeat(64) } }), /身份不匹配/);
  for (const edge_count of [0, -1, 1.5, '25582938']) assert.throws(() => validateBrainGraphInfo({ ...source, graph: { ...graph, edge_count } }, 166700, display), /身份不匹配/);
});

test('group overview shows recorded sensory activity even while the old default DN is silent', () => {
  // First forward batch 13 from desktop-20261026-091042-38a5a08e:
  // R16 L/R fired before ADVANCE STARTED, while all other recorded groups were silent.
  const rates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53.41726618705036, 50.949227373068425, 0, 0, 0, 0, 0, 0, 0, 0];
  assert.equal(groupActivityRate(rates, 0), 0);
  assert.ok(Math.abs(groupActivityRate(rates, null) - 4.743931525459944) < 1e-12);
  assert.equal(groupActivityRate(rates, 12), rates[12]);
  assert.equal(groupActivityRate(Array(22).fill(0), null), 0);
});

test('group overview weights recorded groups equally and keeps missing data distinct from silence', () => {
  // Even if groups overlap or differ in size, this is an arithmetic mean of group rates.
  assert.equal(groupActivityRate([2, 8], null), 5);
  assert.equal(groupActivityRate(null, null), undefined);
  assert.equal(groupActivityRate([], null), undefined);
  assert.equal(groupActivityRate([2, 8], -1), undefined);
});

test('one start prepares clean sessions without exposing an intermediate start button', () => {
  for (const state of ['idle', 'completed', 'cancelled', 'failed']) {
    assert.equal(startAction({ ...status(state), resources_stopped: true }, false), 'prepare');
    assert.equal(startAction({ ...status(state), resources_stopped: false }, false), null);
  }
  assert.equal(startAction(status('awaiting_simulation'), false), null);
  for (const state of ['preparing', 'running', 'stopping']) assert.equal(startAction(status(state), false), null);
  for (const state of ['idle', 'awaiting_simulation']) {
    assert.equal(startAction(status(state), true), null);
    assert.equal(startAction({ ...status(state), request_pending: true }, false), null);
  }
});

test('prepare acknowledgement persists and an older idle poll cannot roll it back', () => {
  const live = new LiveStatus();
  live.accept(status('idle'), live.revision);
  const oldPoll = live.revision;
  const prepare = live.beginChange();
  assert.equal(live.accept(status('preparing'), prepare), true);
  assert.equal(live.accept(status('idle'), oldPoll), false);
  assert.equal(live.current.state, 'preparing');
});

test('a fresh poll wins over a delayed prepare acknowledgement', () => {
  const live = new LiveStatus();
  const prepare = live.beginChange();
  assert.equal(live.accept(status('awaiting_simulation'), live.revision), true);
  assert.equal(live.accept(status('preparing'), prepare), false);
  assert.equal(live.current.state, 'awaiting_simulation');
});

test('emergency stop invalidates both an ordinary action and its in-flight poll', () => {
  const live = new LiveStatus();
  const prepare = live.beginChange();
  const preparingPoll = live.revision;
  const stop = live.beginChange();
  assert.equal(live.accept(status('preparing'), prepare), false);
  assert.equal(live.accept(status('awaiting_simulation'), preparingPoll), false);
  assert.equal(live.accept(status('stopping'), stop), true);
  assert.equal(live.accept(status('cancelled'), live.revision), true);
});

test('emergency stop remains available during ordinary operations, including prepare before acknowledgement', () => {
  assert.equal(canEmergencyStop(status('idle'), true, false), true);
  for (const state of ['preparing', 'awaiting_simulation', 'running']) {
    assert.equal(canEmergencyStop(status(state), false, false), true);
    assert.equal(canEmergencyStop(status(state), true, false), true);
    assert.equal(canEmergencyStop(status(state), true, true), false);
  }
  assert.equal(canEmergencyStop(status('stopping'), true, false), false);
  assert.equal(canEmergencyStop(status('idle'), false, false), false);
});

test('ordinary task buttons wait through computation, execution and stopping; END stays available', () => {
  const ready = { state: 'running', mode: 'brain_control', brain_controlled: true, phase: 'waiting_task', action_busy: false };
  assert.equal(canSubmitTask(ready, 'SHOW_FORWARD'), true);
  for (const phase of ['queued', 'computing', 'executing', 'settling', 'request_rejected']) {
    const busy = { ...ready, phase, action_busy: true, task: 'SHOW_FORWARD' };
    for (const task of ['GREET', 'FACE_LEFT', 'FACE_RIGHT', 'SHOW_FORWARD']) assert.equal(canSubmitTask(busy, task), false);
    assert.equal(canSubmitTask(busy, 'END_INTERACTION'), true);
  }
  assert.equal(canSubmitTask({ ...ready, request_pending: true }, 'FACE_LEFT'), false);
  assert.equal(canSubmitTask({ ...ready, action_busy: true, task: 'END_INTERACTION' }, 'END_INTERACTION'), false);
  assert.equal(canSubmitTask({ ...ready, state: 'stopping' }, 'END_INTERACTION'), false);
});
