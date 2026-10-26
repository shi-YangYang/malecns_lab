import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import vm from 'node:vm';
import { DISPLAY_SURFACE_UNAVAILABLE } from './startup.mjs';

// Exercise actual main-process orchestration with no Electron/VM or renderer.
async function fixture({ sealComplete = true, captureError = null, emptyFrame = false } = {}) {
  const file = new URL('./main.mjs', import.meta.url);
  const source = (await readFile(file, 'utf8')).replace(/^import .*;$/gm, '').replaceAll('import.meta.url', JSON.stringify(file.href));
  const calls = [], intervals = [], sent = [], windowCalls = [];
  const controller = { output: '/fixture/run', state: { state: 'running', run_id: 'run', session_profile: 'continuous', simulator_ready: true, resources_stopped: false }, stop() { calls.push('control_stop'); this.state.state = 'stopping'; return this.state; } };
  const capture = { complete: false, freeze: () => calls.push('freeze'), async seal() { calls.push('capture_seal'); this.complete = sealComplete; }, status() { return { complete: this.complete }; }, viewChanged() {} };
  const video = { complete: false, requestStop: () => calls.push('video_stop'), async seal() { calls.push('video_seal'); this.complete = sealComplete; }, status() { return { complete: this.complete }; }, viewChanged() {} };
  const context = { URL, path, fileURLToPath, randomBytes, Promise, Date, setImmediate, DISPLAY_SURFACE_UNAVAILABLE,
    setTimeout: (fn, ms) => ({ fn, ms }), clearTimeout() {},
    setInterval: (fn, ms) => { const value = { fn, ms }; intervals.push(value); return value; }, clearInterval(value) { if (value) value.cleared = true; },
    protocol: { registerSchemesAsPrivileged() {} },
    app: { setName() {}, requestSingleInstanceLock: () => true, on() {}, whenReady: () => ({ then: () => ({ catch() {} }) }) },
    fixtures: { controller, capture, video, sent, windowCalls,
      capturePage: async () => { windowCalls.push('capture'); if (captureError) throw captureError; return { isEmpty: () => emptyFrame }; } },
  };
  vm.runInNewContext(source + `
    ({ controller, capture, video } = fixtures);
    simulationLoaded = true; simulationVisible = true;
    rectangle = { x: 0, y: 0, width: 640, height: 400 };
    window = { isDestroyed: () => false, isVisible: () => true, isMinimized: () => false,
      show: () => fixtures.windowCalls.push('show'), focus: () => fixtures.windowCalls.push('focus') };
    simulation = { setVisible() {}, setBounds() {}, webContents: { setBackgroundThrottling() {}, capturePage: fixtures.capturePage,
      send: (channel, message) => fixtures.sent.push({channel, message}) } };
    globalThis.api = { limitMediaWindow, watchLiveSource, enforceLiveSource, confirmCaptureSource, recoverStartupSurface, startupSnapshot,
      closure: () => captureClosing,
      pending: () => sourceCheck,
      reply: value => sourceCheck.resolve(value),
      failStartup: (output, message) => { startupFailure = { output, message }; },
      close: () => { closing = true; },
      change: field => { if (field === 'hidden') simulationVisible = false; if (field === 'reload') { simulationLoading = true; documentGeneration++; } },
    };`, context);
  return { ...context.api, calls, intervals, sent, controller, windowCalls };
}

test('normal media limit freezes both writers before deferred seal and retains a single live source check', async () => {
  const f = await fixture(); f.limitMediaWindow('duration_limit'); f.limitMediaWindow('second_limit');
  assert.deepEqual(f.calls, ['freeze', 'video_stop']);
  assert.equal(f.intervals.length, 1); assert.equal(f.intervals[0].ms, 5000);
  await f.closure();
  assert.deepEqual(f.calls, ['freeze', 'video_stop', 'capture_seal', 'video_seal']);
  assert.equal(f.controller.state.state, 'running');
  f.watchLiveSource(); f.watchLiveSource();
  assert.equal(f.sent.length, 1); assert.equal(f.sent[0].channel, 'simulation:source-check');
  assert.ok(f.pending());
  f.reply({ ready: true }); await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.pending(), null); assert.equal(f.controller.state.state, 'running');
  f.watchLiveSource(); assert.equal(f.sent.length, 2);
  f.reply({ ready: false, reason: 'RFB disconnected' }); await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.controller.state.state, 'stopping');
  assert.equal(f.calls.filter(c => c === 'control_stop').length, 1);
  f.watchLiveSource(); assert.equal(f.intervals[0].cleared, true);
});

test('hidden or reloaded live source still stops after both archives have sealed', async () => {
  for (const field of ['hidden', 'reload']) {
    const f = await fixture(); f.limitMediaWindow('duration_limit'); await f.closure();
    f.change(field); f.enforceLiveSource();
    assert.equal(f.controller.state.state, 'stopping', field);
    assert.equal(f.calls.filter(c => c === 'control_stop').length, 1);
  }
});

test('initial window seal errors stop control instead of being treated as capacity success', async () => {
  const f = await fixture({ sealComplete: false });
  f.limitMediaWindow('duration_limit'); await f.closure();
  assert.equal(f.controller.state.state, 'stopping');
  assert.equal(f.calls.filter(c => c === 'control_stop').length, 1);
});

test('only the exact native capture failure gets the recoverable surface code', async () => {
  for (const [message, code] of [
    ['Current display surface not available for capture', DISPLAY_SURFACE_UNAVAILABLE],
    ['Other capture failure', undefined],
  ]) {
    const f = await fixture({ captureError: new Error(message) });
    const check = f.confirmCaptureSource();
    f.reply({ ready: true });
    await assert.rejects(check, error => error.code === code && error.message.includes(message));
    assert.deepEqual(f.windowCalls, ['capture']); assert.equal(f.pending(), null);
  }
});

test('an empty real frame still fails instead of being treated as recovered', async () => {
  const f = await fixture({ emptyFrame: true });
  const check = f.confirmCaptureSource(); f.reply({ ready: true });
  await assert.rejects(check, error => error.code === undefined && /仍为空/.test(error.message));
});

test('surface failure during a running session stops without raising the window', async () => {
  const f = await fixture({ captureError: new Error('Current display surface not available for capture') });
  f.limitMediaWindow('duration_limit'); await f.closure();
  f.watchLiveSource(); f.reply({ ready: true }); await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.controller.state.state, 'stopping');
  assert.deepEqual(f.windowCalls, ['capture']);
});

test('window recovery is limited to the current active startup', async () => {
  for (const change of ['none', 'stop', 'close', 'session', 'running', 'stopped']) {
    const f = await fixture();
    f.controller.state.state = 'awaiting_simulation';
    if (change === 'stop') f.controller.stopRequested = true;
    if (change === 'close') f.close();
    if (change === 'session') f.controller.output = '/fixture/another';
    if (change === 'running') f.controller.state.state = 'running';
    if (change === 'stopped') f.controller.state.resources_stopped = true;
    assert.equal(f.recoverStartupSurface('/fixture/run'), change === 'none', change);
    assert.deepEqual(f.windowCalls, change === 'none' ? ['show', 'focus'] : [], change);
  }
});

test('startup failure display retains the cause without changing backend cleanup evidence', async () => {
  const f = await fixture();
  const value = { status: { state: 'cancelled', output: '/fixture/run', message: '已结束', resources_stopped: true, stop_confirmed: null } };
  f.failStartup('/fixture/run', 'Current display surface not available for capture');
  const projected = f.startupSnapshot(value);
  assert.equal(projected.status.state, 'failed');
  assert.match(projected.status.message, /Current display surface/);
  assert.equal(projected.status.resources_stopped, true); assert.equal(projected.status.stop_confirmed, null);
  assert.equal(value.status.state, 'cancelled'); assert.equal(value.status.message, '已结束');
  const other = { status: { ...value.status, output: '/fixture/another' } };
  assert.equal(f.startupSnapshot(other), other);
  f.controller.output = '/fixture/another';
  assert.equal(f.startupSnapshot(value), value);
});
