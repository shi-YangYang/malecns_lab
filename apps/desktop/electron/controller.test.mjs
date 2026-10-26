import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ExperimentController, TASK_ACK_TIMEOUT_MS } from './controller.mjs';
import { CLOCK_SCHEMA } from './recording.mjs';

function fakeChild() { const child = new EventEmitter(); child.stdin = new PassThrough(); child.stdout = new PassThrough(); child.stderr = new PassThrough(); return child; }
function status(controller, state, extra = {}) { return { schema: 'malecns_lab.desktop_event.v1', type: 'status', state, message: state, output: controller.output, run_id: null, simulator_ready: false, motion_started: false, stop_confirmed: null, resources_stopped: false, ...extra }; }
function send(child, value) { child.stdout.write(`${JSON.stringify(value)}\n`); }
test('launch uses only fixed executable and arguments; repeated prepare and start are rejected', () => {
  const child = fakeChild(), calls = [];
  const controller = new ExperimentController('/test/repo', () => {}, (...args) => { calls.push(args); return child; });
  controller.prepare();
  assert.equal(calls[0][0], '/test/repo/.venv-train/bin/python');
  assert.deepEqual(calls[0][1], ['-u', '-m', 'malecns_lab.runtime.x2.behavior', '--execute', '--sim-only', '--qualification', '--session-profile', 'continuous', '--output', controller.output]);
  assert.equal(controller.state.mode, 'brain_control'); assert.equal(controller.state.brain_controlled, false);
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

test('task IPC admits only five enums during a ready running session; END remains a task', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  assert.throws(() => controller.task('GREET'));
  controller.prepare();
  assert.throws(() => controller.task('GREET'));
  send(child, status(controller, 'running', { mode: 'brain_control', brain_controlled: true, phase: 'waiting_task', qualification_only: true }));
  for (const task of ['GREET', 'FACE_LEFT', 'FACE_RIGHT', 'SHOW_FORWARD', 'END_INTERACTION']) {
    controller.task(task);
    assert.equal(child.stdin.read().toString(), JSON.stringify({ command: 'task', task }) + '\n');
    send(child, status(controller, 'running', { mode: 'brain_control', brain_controlled: true, phase: 'queued', task, action_busy: true, active_request_id: task }));
    send(child, status(controller, 'running', { mode: 'brain_control', brain_controlled: true, phase: 'waiting_task', task, action_busy: false, active_request_id: task, stop_confirmed: true }));
  }
  for (const bad of ['WAVE_RIGHT', 'rm -rf', '../checkpoint', { task: 'GREET' }, null]) assert.throws(() => controller.task(bad));
  controller.stop();
  assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
  assert.throws(() => controller.task('END_INTERACTION'));
});

test('brain input waits for backend readiness; END enters the model and emergency stop stays independent', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare();
  for (const extra of [
    { mode: 'brain_control', phase: 'waiting_task' },
    { mode: 'brain_control', brain_controlled: false, phase: 'waiting_task' },
    { mode: 'manual_demo', brain_controlled: true, phase: 'waiting_task' },
    { mode: 'brain_control', brain_controlled: true, phase: 'initializing' },
  ]) {
    send(child, status(controller, 'running', extra));
    assert.throws(() => controller.task('SHOW_FORWARD'), /脑模型尚未准备完成/);
    assert.equal(child.stdin.read(), null);
  }
  send(child, status(controller, 'running', { mode: 'brain_control', brain_controlled: true, phase: 'executing' }));
  controller.task('END_INTERACTION');
  assert.equal(child.stdin.read().toString(), '{"command":"task","task":"END_INTERACTION"}\n');
  assert.equal(controller.state.phase, 'queued');
  controller.stop();
  assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
  child.emit('exit', 0, null);
  assert.equal(controller.state.brain_controlled, false);
});

function clockReply(request, extra = {}) {
  return { schema: CLOCK_SCHEMA, type: 'clock', run_id: request.run_id, nonce: request.nonce,
    clock_id: 'c'.repeat(32), python_monotonic_ns: '9000000000',
    clock_source: { api: 'time.monotonic_ns', implementation: 'fixture', monotonic: true, adjustable: false }, ...extra };
}
function clockFixture() {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child, () => 70010000000000n);
  controller.prepare(); send(child, status(controller, 'running'));
  return { child, controller, readRequest: () => JSON.parse(child.stdin.read().toString()) };
}
test('clock replies bind nonce/run and preserve distinct clocks without changing experiment state', async () => {
  const { child, controller, readRequest } = clockFixture(), original = controller.state;
  const pending = controller.sampleHostClock(), request = readRequest();
  send(child, clockReply(request, { nonce: 'f'.repeat(32) }));
  assert.equal(controller.hostClockId, null); assert.equal(controller.clockRequests.size, 1);
  send(child, clockReply(request));
  const sample = await pending;
  assert.equal(sample.python_monotonic_ns, '9000000000'); assert.equal(sample.node_send_ns, '70010000000000');
  assert.equal(controller.state, original); assert.equal(controller.clockRequests.size, 0);
  for (const extra of [{ run_id: 'another-run' }, { nonce: undefined }, { clock_id: 'd'.repeat(32) }]) {
    const result = controller.sampleHostClock(), rejected = assert.rejects(result, /时钟/), query = readRequest();
    send(child, clockReply(query, extra)); await rejected;
    assert.equal(controller.state, original);
  }
});
test('clock timeout yields unknown age; exit and emergency stop cancel pending requests immediately', async () => {
  const { child, controller, readRequest } = clockFixture();
  controller.optionalJson = async (_root, name) => name.endsWith('latest.json') ? { host_monotonic_ns: 9000000000 } : null;
  const snapshot = await controller.snapshot();
  assert.equal(snapshot.transport_delay_ms, null); assert.equal(snapshot.transport_clock, null);
  assert.equal(controller.state.state, 'running'); assert.equal(controller.clockRequests.size, 0); readRequest();
  const pending = controller.sampleHostClock(), rejected = assert.rejects(pending, /停止/); readRequest();
  controller.stop();
  assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n'); await rejected;
  assert.equal(controller.clockRequests.size, 0);
  const second = clockFixture(), onExit = second.controller.sampleHostClock(), exited = assert.rejects(onExit, /退出/);
  second.readRequest(); second.child.emit('exit', 0, null); await exited;
  assert.equal(second.controller.clockRequests.size, 0);
});
test('batch age uses only the Python sample and never subtracts the Node epoch', async () => {
  const { child, controller, readRequest } = clockFixture();
  controller.optionalJson = async (_root, name) => name.endsWith('latest.json') ? { host_monotonic_ns: 8000000000 } : null;
  const pending = controller.snapshot();
  while (!controller.clockRequests.size) await new Promise(resolve => setImmediate(resolve));
  send(child, clockReply(readRequest()));
  assert.equal((await pending).transport_delay_ms, 1000);
});

test('busy task lock survives stale waiting events and END remains available until its own acknowledgement', () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare();
  const running = { mode: 'brain_control', brain_controlled: true };
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false, active_request_id: 'previous' }));
  controller.task('SHOW_FORWARD');
  assert.equal(controller.state.request_pending, true);
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false, active_request_id: 'previous' }));
  assert.throws(() => controller.task('FACE_LEFT'), /尚未结束/);
  controller.task('END_INTERACTION');
  assert.throws(() => controller.task('END_INTERACTION'), /正在结束/);
  send(child, status(controller, 'running', { ...running, phase: 'queued', task: 'SHOW_FORWARD', action_busy: true, active_request_id: 'forward' }));
  assert.equal(controller.state.task, 'END_INTERACTION');
  assert.equal(controller.state.request_pending, true);
  send(child, status(controller, 'running', { ...running, phase: 'queued', task: 'END_INTERACTION', action_busy: true, active_request_id: 'end' }));
  send(child, status(controller, 'running', { ...running, phase: 'executing', task: 'END_INTERACTION', action_busy: true, active_request_id: 'end' }));
  assert.throws(() => controller.task('GREET'), /尚未结束/);
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', task: 'END_INTERACTION', action_busy: false, active_request_id: 'end', stop_confirmed: true }));
  controller.task('GREET');
  assert.equal(controller.state.request_pending, true);
  assert.deepEqual(child.stdin.read().toString().trim().split('\n').map(JSON.parse).map(item => item.task), ['SHOW_FORWARD', 'END_INTERACTION', 'GREET']);
});

for (const legacy of [false, true]) test(`request rejection preserves idle/active authority and clears matching ACK (${legacy ? 'legacy' : 'notice'})`, () => {
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare();
  const running = { mode: 'brain_control', brain_controlled: true };
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false, active_request_id: null }));
  controller.task('SHOW_FORWARD'); child.stdin.read();
  send(child, status(controller, 'running', { ...running, phase: legacy ? 'request_rejected' : 'waiting_task',
    notice: legacy ? undefined : 'request_rejected', rejected_task: 'GREET', action_busy: false, active_request_id: null }));
  assert.equal(controller.state.request_pending, true); // Unrelated rejection is not this request's ACK.
  send(child, status(controller, 'running', { ...running, phase: legacy ? 'request_rejected' : 'waiting_task',
    notice: legacy ? undefined : 'request_rejected', rejected_task: 'SHOW_FORWARD', action_busy: false,
    request_pending: false, active_request_id: null, message: '本次任务次数已达上限' }));
  assert.equal(controller.taskAwaitingAck, null); assert.equal(controller.state.phase, 'waiting_task');
  assert.equal(controller.state.message, '本次任务次数已达上限');
  controller.task('GREET'); child.stdin.read();
  send(child, status(controller, 'running', { ...running, phase: 'queued', task: 'GREET', action_busy: true, active_request_id: 'greet' }));
  send(child, status(controller, 'running', { ...running, phase: 'executing', task: 'GREET', action_busy: true, active_request_id: 'greet' }));
  send(child, status(controller, 'running', { ...running, phase: legacy ? 'request_rejected' : 'executing',
    notice: legacy ? undefined : 'request_rejected', rejected_task: 'FACE_LEFT', active_task: 'GREET',
    task: 'GREET', action_busy: true, active_request_id: 'greet', message: '当前动作尚未结束' }));
  assert.equal(controller.state.phase, 'executing'); assert.equal(controller.state.task, 'GREET');
  assert.throws(() => controller.task('FACE_RIGHT'), /尚未结束/);
  controller.stop();
});

test('missing task ACK independently stops; stale and late statuses never unlock or erase the error', t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare();
  const running = { mode: 'brain_control', brain_controlled: true };
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false, active_request_id: null }));
  controller.task('SHOW_FORWARD'); child.stdin.read();
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false, active_request_id: null }));
  t.mock.timers.tick(TASK_ACK_TIMEOUT_MS);
  assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
  assert.equal(controller.state.state, 'stopping'); assert.equal(controller.state.reason, 'task_ack_timeout');
  assert.equal(controller.safeToClose, false); assert.match(controller.state.message, /接收确认超时/);
  send(child, status(controller, 'running', { ...running, phase: 'queued', task: 'SHOW_FORWARD', action_busy: true, active_request_id: 'late' }));
  assert.equal(controller.state.state, 'stopping'); assert.throws(() => controller.task('END_INTERACTION'));
  send(child, status(controller, 'cancelled', { stop_confirmed: true, resources_stopped: true }));
  child.emit('exit', 0, null);
  assert.equal(controller.state.state, 'failed'); assert.equal(controller.state.stop_confirmed, true);
  assert.equal(controller.safeToClose, true); assert.match(controller.state.message, /接收确认超时/);
});

test('a matching task ACK cancels its deadline and END owns a fresh bounded ACK', t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const child = fakeChild(), controller = new ExperimentController('/test/repo', () => {}, () => child);
  controller.prepare();
  const running = { mode: 'brain_control', brain_controlled: true };
  send(child, status(controller, 'running', { ...running, phase: 'waiting_task', action_busy: false }));
  controller.task('SHOW_FORWARD'); child.stdin.read();
  t.mock.timers.tick(TASK_ACK_TIMEOUT_MS - 1);
  send(child, status(controller, 'running', { ...running, phase: 'queued', task: 'SHOW_FORWARD', action_busy: true, active_request_id: 'forward' }));
  t.mock.timers.tick(TASK_ACK_TIMEOUT_MS);
  assert.equal(controller.state.state, 'running'); assert.equal(child.stdin.read(), null);
  controller.task('END_INTERACTION'); child.stdin.read();
  t.mock.timers.tick(TASK_ACK_TIMEOUT_MS);
  assert.equal(child.stdin.read().toString(), '{"command":"stop"}\n');
  assert.equal(controller.state.state, 'stopping');
});
