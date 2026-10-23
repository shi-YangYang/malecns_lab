import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ExperimentController } from './controller.mjs';

function fakeChild() { const child = new EventEmitter(); child.stdin = new PassThrough(); child.stdout = new PassThrough(); child.stderr = new PassThrough(); return child; }
function status(controller, state, extra = {}) { return { schema: 'malecns_lab.desktop_event.v1', type: 'status', state, message: state, output: controller.output, run_id: null, simulator_ready: false, motion_started: false, stop_confirmed: null, resources_stopped: false, ...extra }; }
function send(child, value) { child.stdout.write(`${JSON.stringify(value)}\n`); }
test('launch uses only fixed executable and arguments; repeated prepare and start are rejected', () => {
  const child = fakeChild(), calls = [];
  const controller = new ExperimentController('/test/repo', () => {}, (...args) => { calls.push(args); return child; });
  controller.prepare();
  assert.equal(calls[0][0], '/test/repo/.venv-train/bin/python');
  assert.deepEqual(calls[0][1].slice(0, 7), ['-u', '-m', 'malecns_lab.experiments.x2_desktop', '--execute', '--sim-only', '--output', controller.output]);
  assert.equal(calls[0][2].shell, false); assert.throws(() => controller.prepare()); assert.throws(() => controller.continue());
  send(child, status(controller, 'awaiting_simulation', { simulator_ready: true }));
  controller.continue(); assert.throws(() => controller.continue());
  assert.equal(child.stdin.read().toString(), '{"command":"continue"}\n');
  controller.stop(); assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
  assert.equal(controller.safeToClose, false);
});
test('exit zero without runtime terminal confirmation is a failure, not cleanup success', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare(); send(child, status(controller, 'running', { motion_started: true })); child.emit('exit', 0, null);
  assert.equal(controller.state.state, 'failed'); assert.equal(controller.state.resources_stopped, false); assert.equal(controller.state.stop_confirmed, false); assert.equal(controller.safeToClose, false);
});
test('shutdown waits for runtime resource confirmation and actual child exit', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare(); controller.stop(); send(child, status(controller, 'cancelled', { resources_stopped: true }));
  assert.equal(controller.safeToClose, false); child.emit('exit', 0, null); assert.equal(controller.safeToClose, true);
});
test('malformed status requests safe stop, rather than killing child or accepting unsafe success', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare(); send(child, { schema: 'wrong', state: 'completed' });
  assert.equal(controller.state.state, 'stopping'); assert.equal(controller.state.resources_stopped, false); assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
});
test('record replay only accepts sealed sessions and returns bounded sequence files', async t => {
  const repo = await mkdtemp(path.join(tmpdir(), 'malecns-controller-')); t.after(() => rm(repo, { recursive: true, force: true }));
  const controller = new ExperimentController(repo), id = 'desktop-20261023-115959-abcdef12';
  const root = path.join(controller.recordsRoot, id); await mkdir(path.join(root, 'telemetry/batches'), { recursive: true });
  const session = { schema: 'malecns_lab.desktop_event.v1', state: 'running', resources_stopped: false };
  await writeFile(path.join(root, 'desktop_session.json'), JSON.stringify(session));
  assert.deepEqual(await controller.listRecords(), []); await assert.rejects(controller.openRecord(id), /封存/);
  session.state = 'completed'; session.resources_stopped = true;
  await writeFile(path.join(root, 'desktop_session.json'), JSON.stringify(session));
  await writeFile(path.join(root, 'telemetry/manifest.json'), '{}'); await writeFile(path.join(root, 'telemetry/summary.json'), '{"complete":true}');
  await writeFile(path.join(root, 'telemetry/batches/000001.json'), '{"batch_sequence":1}');
  assert.equal((await controller.listRecords()).length, 1); assert.deepEqual((await controller.openRecord(id)).sequences, [1]);
  assert.deepEqual(await controller.readBatch(id, 1), { batch_sequence: 1 }); await assert.rejects(controller.readBatch(id, '../'), /序号/);
});
