import test from 'node:test';
import assert from 'node:assert/strict';
import { DISPLAY_SURFACE_UNAVAILABLE, finishStartup } from './startup.mjs';

const unavailableSurface = () => Object.assign(new Error('Current display surface not available for capture'), { code: DISPLAY_SURFACE_UNAVAILABLE });

function fixture() {
  let time = 0, continued = 0, stopped = 0;
  const controller = { output: '/run/one', state: { state: 'preparing', resources_stopped: false },
    continue() { continued++; this.state = { state: 'running' }; return this.state; } };
  return { controller, now: () => time, wait: async ms => { time += ms; },
    confirmSource: async () => {}, stop: () => { stopped++; controller.stopRequested = true; },
    counts: () => ({ continued, stopped }) };
}

test('one start waits for backend and connected canvas, then continues exactly once', async () => {
  const f = fixture();
  let probes = 0;
  const wait = f.wait;
  f.wait = async ms => { await wait(ms); f.controller.state.state = 'awaiting_simulation'; };
  f.confirmSource = async () => { if (++probes < 3) throw new Error('connecting'); };
  assert.equal((await finishStartup(f)).state, 'running');
  assert.equal(probes, 3);
  assert.deepEqual(f.counts(), { continued: 1, stopped: 0 });
});

for (const change of ['stop', 'close', 'session', 'failure']) {
  test(`${change} during a pending source check cannot continue the session`, async () => {
    const f = fixture();
    let closed = false;
    f.controller.state.state = 'awaiting_simulation';
    f.closing = () => closed;
    f.confirmSource = async () => {
      if (change === 'stop') f.controller.stopRequested = true; // Even if an old status still says awaiting.
      if (change === 'close') closed = true;
      if (change === 'session') f.controller.output = '/run/two';
      if (change === 'failure') f.controller.state.state = 'failed';
    };
    await finishStartup(f);
    assert.deepEqual(f.counts(), { continued: 0, stopped: 0 });
  });
}

test('source timeout stops the owned startup and preserves its error', async () => {
  const f = fixture();
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { throw new Error('canvas disconnected'); };
  await assert.rejects(finishStartup({ ...f, sourceTimeoutMs: 500 }), /canvas disconnected/);
  assert.deepEqual(f.counts(), { continued: 0, stopped: 1 });
});

test('backend preparation timeout stops without probing the canvas', async () => {
  const f = fixture();
  f.confirmSource = async () => assert.fail('backend not ready');
  await assert.rejects(finishStartup({ ...f, timeoutMs: 500 }), /启动超时/);
  assert.deepEqual(f.counts(), { continued: 0, stopped: 1 });
});

test('native surface recovery runs once and must pass a fresh capture before continue', async () => {
  const f = fixture(), events = [];
  let probes = 0, recoveries = 0;
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { if (++probes === 1) throw unavailableSurface(); };
  f.recoverSource = () => { recoveries++; assert.equal(probes, 1); assert.equal(f.counts().continued, 0); };
  f.onEvent = event => events.push(event.type);
  await finishStartup(f);
  assert.equal(probes, 2); assert.equal(recoveries, 1);
  assert.deepEqual(f.counts(), { continued: 1, stopped: 0 });
  assert.deepEqual(events, ['started', 'source_wait_started', 'source_not_ready', 'surface_recovery_requested',
    'surface_recovery_completed', 'source_confirmed', 'continue_requested']);
});

test('an unrecovered surface keeps the original deadline and never repeats recovery', async () => {
  const f = fixture(); let recoveries = 0;
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { throw unavailableSurface(); };
  f.recoverSource = () => { recoveries++; };
  await assert.rejects(finishStartup({ ...f, sourceTimeoutMs: 500 }), { code: DISPLAY_SURFACE_UNAVAILABLE });
  assert.equal(recoveries, 1); assert.equal(f.now(), 600);
  assert.deepEqual(f.counts(), { continued: 0, stopped: 1 });
});

test('ordinary source failures do not raise the window', async () => {
  const f = fixture();
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { throw new Error('RFB disconnected'); };
  f.recoverSource = () => assert.fail('ordinary disconnection must not recover the native surface');
  await assert.rejects(finishStartup({ ...f, sourceTimeoutMs: 500 }), /RFB disconnected/);
});

for (const change of ['stop', 'close', 'session', 'failure']) {
  test(`${change} during a failed capture cannot trigger window recovery`, async () => {
    const f = fixture(); let closed = false;
    f.controller.state.state = 'awaiting_simulation';
    f.closing = () => closed;
    f.confirmSource = async () => {
      if (change === 'stop') f.controller.stopRequested = true;
      if (change === 'close') closed = true;
      if (change === 'session') f.controller.output = '/run/two';
      if (change === 'failure') f.controller.state.state = 'failed';
      throw unavailableSurface();
    };
    f.recoverSource = () => assert.fail('inactive startup cannot recover');
    await finishStartup(f);
    assert.deepEqual(f.counts(), { continued: 0, stopped: 0 });
  });
}

test('a capture rejection after the deadline cannot raise the window', async () => {
  const f = fixture();
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { await f.wait(600); throw unavailableSurface(); };
  f.recoverSource = () => assert.fail('expired startup cannot recover');
  await assert.rejects(finishStartup({ ...f, sourceTimeoutMs: 500 }), { code: DISPLAY_SURFACE_UNAVAILABLE });
  assert.deepEqual(f.counts(), { continued: 0, stopped: 1 });
});

test('stop during window recovery prevents another capture and continue', async () => {
  const f = fixture(); let probes = 0;
  f.controller.state.state = 'awaiting_simulation';
  f.confirmSource = async () => { probes++; throw unavailableSurface(); };
  f.recoverSource = async () => { f.controller.stopRequested = true; };
  await finishStartup(f);
  assert.equal(probes, 1);
  assert.deepEqual(f.counts(), { continued: 0, stopped: 0 });
});
