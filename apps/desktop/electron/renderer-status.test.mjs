import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

const source = await readFile(new URL('../src/live-status.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
const { LiveStatus, canEmergencyStop } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const status = state => ({ state, resources_stopped: state === 'idle' });

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
