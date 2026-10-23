import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, symlink, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { JsonLineReader, boundedRectangle, readBoundedJson, recordIdAllowed, simulationRequestAllowed, trustedFrame } from './boundaries.mjs';

async function temporary(t) { const root = await mkdtemp(path.join(tmpdir(), 'malecns-desktop-')); t.after(() => rm(root, { recursive: true, force: true })); return root; }
test('noVNC only accepts exact loopback HTTP and WebSocket port', () => {
  for (const url of ['http://127.0.0.1:6081/vnc.html', 'ws://127.0.0.1:6081/websockify']) assert.equal(simulationRequestAllowed(url), true);
  for (const url of ['https://127.0.0.1:6081/', 'http://localhost:6081/', 'http://127.0.0.1:6082/', 'http://127.0.0.1.evil:6081/', 'file:///tmp/a', 'javascript:alert(1)', 'http://user@127.0.0.1:6081/']) assert.equal(simulationRequestAllowed(url), false, url);
});
test('IPC requires exact main contents, frame, and application URL', () => {
  const mainFrame = { url: 'workbench://app/index.html' }, contents = { mainFrame };
  assert.equal(trustedFrame({ sender: contents, senderFrame: mainFrame }, contents), true);
  assert.equal(trustedFrame({ sender: {}, senderFrame: mainFrame }, contents), false);
  assert.equal(trustedFrame({ sender: contents, senderFrame: { ...mainFrame } }, contents), false);
  mainFrame.url = 'http://127.0.0.1:6081/vnc.html';
  assert.equal(trustedFrame({ sender: contents, senderFrame: mainFrame }, contents), false);
});
test('view rectangle cannot cover title bar or extend beyond content', () => {
  assert.deepEqual(boundedRectangle({ x: -1, y: -99, width: 9999, height: 9999 }, 1100, 720), { x: 0, y: 62, width: 1100, height: 658 });
  assert.throws(() => boundedRectangle({ x: NaN, y: 1, width: 10, height: 10 }, 1100, 720));
});
test('record identifiers do not accept arbitrary relative or absolute paths', () => {
  assert.equal(recordIdAllowed('desktop-20261023-115959-abcdef12'), true);
  for (const id of ['../desktop-20261023-115959-abcdef12', '/tmp/a', 'desktop-user-chosen', 'desktop-20261023-115959-abcdef12/telemetry']) assert.equal(recordIdAllowed(id), false);
});
test('archive reading bounds size, traversal, and symlink targets', async t => {
  const root = await temporary(t), allowed = path.join(root, 'records'); await mkdir(allowed);
  await writeFile(path.join(allowed, 'ok.json'), '{"real":true}');
  await writeFile(path.join(root, 'outside.json'), '{"private":true}');
  await symlink(path.join(root, 'outside.json'), path.join(allowed, 'escape.json'));
  assert.deepEqual(await readBoundedJson(allowed, 'ok.json'), { real: true });
  await assert.rejects(readBoundedJson(allowed, '../outside.json'), /路径/);
  await assert.rejects(readBoundedJson(allowed, 'escape.json'), /越出/);
  await assert.rejects(readBoundedJson(allowed, 'ok.json', 5), /上限/);
});
test('NDJSON accepts split chunks and reports malformed or oversized lines without unbounded buffering', () => {
  const events = [], errors = [], reader = new JsonLineReader(value => events.push(value), error => errors.push(error), 32);
  reader.push('{"state":'); reader.push('"running"}\n'); reader.push('bad\n'); reader.push('x'.repeat(40));
  reader.push('{"ok":true}\n');
  assert.deepEqual(events, [{ state: 'running' }, { ok: true }]); assert.equal(errors.length, 2); assert.equal(reader.buffer, '');
});
