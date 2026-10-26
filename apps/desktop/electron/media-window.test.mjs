import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import vm from 'node:vm';

// Exercise actual main-process orchestration with no Electron/VM or renderer.
async function fixture({ sealComplete = true } = {}) {
  const file = new URL('./main.mjs', import.meta.url);
  const source = (await readFile(file, 'utf8')).replace(/^import .*;$/gm, '').replaceAll('import.meta.url', JSON.stringify(file.href));
  const calls = [], intervals = [], sent = [];
  const controller = { output: '/fixture/run', state: { state: 'running', run_id: 'run', session_profile: 'continuous', simulator_ready: true, resources_stopped: false }, stop() { calls.push('control_stop'); this.state.state = 'stopping'; return this.state; } };
  const capture = { complete: false, freeze: () => calls.push('freeze'), async seal() { calls.push('capture_seal'); this.complete = sealComplete; }, status() { return { complete: this.complete }; }, viewChanged() {} };
  const video = { complete: false, requestStop: () => calls.push('video_stop'), async seal() { calls.push('video_seal'); this.complete = sealComplete; }, status() { return { complete: this.complete }; }, viewChanged() {} };
  const context = { URL, path, fileURLToPath, randomBytes, Promise, Date, setImmediate,
    setTimeout: (fn, ms) => ({ fn, ms }), clearTimeout() {},
    setInterval: (fn, ms) => { const value = { fn, ms }; intervals.push(value); return value; }, clearInterval(value) { if (value) value.cleared = true; },
    protocol: { registerSchemesAsPrivileged() {} },
    app: { setName() {}, requestSingleInstanceLock: () => true, on() {}, whenReady: () => ({ then: () => ({ catch() {} }) }) },
    fixtures: { controller, capture, video, sent },
  };
  vm.runInNewContext(source + `
    ({ controller, capture, video } = fixtures);
    simulationLoaded = true; simulationVisible = true;
    rectangle = { x: 0, y: 0, width: 640, height: 400 };
    window = { isDestroyed: () => false, isVisible: () => true, isMinimized: () => false };
    simulation = { setVisible() {}, webContents: { send: (channel, message) => fixtures.sent.push({channel, message}) } };
    globalThis.api = { limitMediaWindow, watchLiveSource, enforceLiveSource,
      closure: () => captureClosing,
      pending: () => sourceCheck,
      reply: value => sourceCheck.resolve(value),
      change: field => { if (field === 'hidden') simulationVisible = false; if (field === 'reload') { simulationLoading = true; documentGeneration++; } },
    };`, context);
  return { ...context.api, calls, intervals, sent, controller };
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
