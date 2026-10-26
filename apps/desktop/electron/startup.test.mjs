import test from 'node:test';
import assert from 'node:assert/strict';
import { finishStartup } from './startup.mjs';

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
